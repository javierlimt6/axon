# Axon: KV-Cache–Aware LLM Inference Router

> A small, production-inspired LLM inference stack that demonstrates cache-aware routing, continuous batching, and realistic benchmarking on top of vLLM or (mini-)SGLang, with an optional Gemma 4 extension.

---

## 1. Project Goals

### 1.1 Primary goals

- Build a **prefix/KV‑cache–aware router** for LLM inference that:
  - Knows which worker replica holds which prompt prefix in its KV cache.
  - Routes incoming requests to maximise cache hits.
  - Exposes clear metrics: TTFT, TPS, cache‑hit rate, queue depth.
- Provide a **traffic generator / benchmarking harness** that can replay realistic workloads (shared prefixes, bursty traffic, varied ISL/OSL).
- Demonstrate the effect of cache‑aware routing vs naïve routing with concrete numbers and plots.
- Keep the codebase small and readable so a single engineer can grok it end‑to‑end.

### 1.2 Secondary goals

- Integrate **speculative decoding** on one or more workers and measure its effect.
- Experiment with **quantised models** (4‑bit weights, possibly KV‑cache quantisation) and analyse speed–quality trade‑offs.
- Capture basic **GPU profiling** (PyTorch Profiler + Nsight Systems/Compute) for prefill vs decode.
- Optionally support **disaggregated prefill/decode** on a single machine (prefill process + decode process).
- Optionally **add Gemma 4 text support to mini‑SGLang** and/or demo Axon on Gemma 4 workers.

### 1.3 Non‑goals

- Not a full Kubernetes‑native system (no CRDs, controllers, or Gateway API integration).
- Not a replacement for vLLM/SGLang; they remain the underlying serving engines.
- No multi‑tenant billing, auth, or UI.

---

## 2. Architecture Overview

```text
+------------------------+
|   Clients / Bench     |
+-----------+------------+
            |
            v
+------------------------+
|  Axon Router (FastAPI/Go)|
|  - prefix index          |
|  - routing policy        |
|  - metrics               |
+----+---------------------+
     |            |
     | HTTP       | HTTP
     v            v
+----------+   +----------+
| Worker 0 |   | Worker 1 |
| (vLLM /  |   | (vLLM /  |
|  SGLang) |   |  SGLang) |
+----------+   +----------+
```

### 2.1 Components

- **Workers** (Python)
  - vLLM or (mini-)SGLang instances exposing an OpenAI‑compatible HTTP API.
  - Handle model loading, KV cache, batching, attention kernels, etc.

- **Axon Router** (Go or Python)
  - Stateless HTTP service in front of workers.
  - Maintains an in‑memory index mapping `prefix_hash → worker_id`.
  - Implements routing policies (naïve, cache‑aware, A/B).
  - Exposes Prometheus metrics for TTFT, TPS, cache‑hit rate, and per‑worker load.

- **Benchmark Harness** (Python)
  - Generates synthetic but realistic traffic (tenants, prefixes, ISL/OSL distributions, burst patterns).
  - Measures TTFT, TPS, cache‑hit rate, and per‑worker utilisation.
  - Produces CSVs/plots for analysis.

### 2.2 Hardware layout

- **Yoga Pro 9i (RTX 40‑series, 6–8 GB VRAM)**
  - Runs all GPU workers (vLLM/SGLang), Nsight profiling, and heavy benchmarking under WSL2.

- **MacBook Air M‑series**
  - Runs the router and benchmarking harness for everyday development.

- Optional: run router on Mac and workers on Yoga over LAN to simulate multi‑node deployment.

---

## 3. Router Design

### 3.1 Data structures

```python
class PrefixEntry(BaseModel):
    prefix_hash: str
    worker_id: str
    last_used_at: float  # unix timestamp
    prompt_len: int      # tokens in prefix
    hits: int            # number of times reused

PrefixIndex = dict[str, PrefixEntry]
```

- **Key**: `prefix_hash` — hash over a canonicalised prefix representation.
- **Value**: metadata used for scoring (LRU, popularity, length).

### 3.2 Prefix hashing

Canonical prefix representation:

```text
canonical_prefix = system_prompt + "\n" + first_N_tokens_of_user_prompt
N = 128 (configurable)
```

- Tokenise using the worker’s tokenizer (e.g. HuggingFace tokenizer or model’s native tokenizer).
- Compute hash: SHA‑256 or xxHash64 over the token ID sequence.
- Store both hash and length (number of tokens) in the index.

### 3.3 Routing policies

#### 3.3.1 Naïve baseline

- Algorithm: **round‑robin** or **least‑loaded** worker.
- Ignores prefix cache entirely.
- Used as baseline for benchmarking and for sanity checks.

#### 3.3.2 Cache‑aware routing

High‑level pseudo‑code:

```python
async def route(request: InferenceRequest) -> Worker:
    prefix_hash, prefix_len = compute_prefix_hash(request)

    entry = prefix_index.get(prefix_hash)

    if entry is not None and worker_is_healthy(entry.worker_id):
        # Cache hit path
        entry.last_used_at = now()
        entry.hits += 1
        return workers[entry.worker_id]

    # Cache miss path: pick a target with least load
    target = pick_least_loaded_worker()

    # Register / update index asynchronously
    prefix_index[prefix_hash] = PrefixEntry(
        prefix_hash=prefix_hash,
        worker_id=target.id,
        last_used_at=now(),
        prompt_len=prefix_len,
        hits=0,
    )

    return target
```

- **Load metric**: moving average of in‑flight tokens per second or queue length per worker.
- **Health checking**: periodic ping or workers’ `/health` endpoint.

#### 3.3.3 Eviction

- Bound index size with `MAX_PREFIXES` (e.g. 10k).
- On insert, if `len(index) > MAX_PREFIXES`:
  - Evict least‑recently used by `last_used_at`.
  - Optionally prefer evicting entries with low `hits`.

### 3.4 API contract

Router exposes OpenAI‑compatible endpoints:

- `POST /v1/chat/completions`
- Optional: `POST /v1/completions`

Behaviour:

1. Client sends request to router.
2. Router computes prefix hash and chooses worker.
3. Router forwards request to worker as‑is.
4. Router streams back response to client.
5. Router attaches tracing headers:
   - `X-Axon-Worker-Id`
   - `X-Axon-Prefix-Hash`
   - `X-Axon-Cache-Hit: true|false`

### 3.5 Metrics

Prometheus metrics exposed at `/metrics`:

- `axon_requests_total{route="chat"}`
- `axon_ttft_seconds_bucket{cache_hit="true|false"}`
- `axon_tps_tokens_per_second{worker="w0"}`
- `axon_cache_hits_total`
- `axon_cache_misses_total`
- `axon_worker_queue_depth{worker="w0"}`

---

## 4. Worker Setup

### 4.1 Using mini‑SGLang (for understanding & early phases)

- Run N worker processes (start with 2) on distinct ports.
- Each worker:
  - Loads the same model checkpoint (e.g. Llama‑3.2‑1B, Qwen‑3‑0.6B).
  - Exposes OpenAI‑compatible API (`/v1/chat/completions`).
  - Enables continuous batching / chunked prefill.

Example launch scripts (pseudo‑CLI):

```bash
python -m minisgl.serve \
  --model path/to/model \
  --port 8001 \
  --tensor-parallel-size 1 \
  --max-batch-size 32

python -m minisgl.serve \
  --model path/to/model \
  --port 8002 \
  --tensor-parallel-size 1 \
  --max-batch-size 32
```

### 4.2 Using full SGLang (for Gemma 4 demo & speculation)

- Swap mini‑SGLang for full SGLang workers with identical API.
- Use Gemma‑4 E4B (Q4‑bit) checkpoints so they fit in 6–8 GB VRAM.
- Enable speculative decoding on one worker for Phase 5 experiments.

### 4.3 vLLM alternative

- Use `python -m vllm.entrypoints.openai.api_server` with different `--port`.
- Configure prefix caching options where relevant.

---

## 5. Benchmark Harness

### 5.1 Workload model

Simulate two primary tenant types:

- **Tenant A (high prefix reuse)**
  - All requests share a long system prompt (1–2k tokens) and optionally a shared document.
  - User prompts are short (10–50 tokens) appended to that prefix.

- **Tenant B (low prefix reuse)**
  - Each request has a near‑random prompt, minimal shared prefix.

Distributions:

- ISL (input sequence length):
  - Tenant A: log‑normal around ~1k tokens.
  - Tenant B: log‑normal around ~200 tokens.

- OSL (output sequence length):
  - Geometric distribution with mean ~128 tokens.

- Arrival process:
  - Poisson arrivals with bursts (e.g. spike of 50 rps for 10 seconds).

### 5.2 Implementation

- Python (`asyncio` + `httpx` or `aiohttp`).
- Parameters:
  - Concurrent clients.
  - Tenant mix (e.g. 70% A, 30% B).
  - Total duration per run.
  - Routing policy (`naive`, `cache_aware`).

Per request:

1. Construct prompt based on tenant + ISL/OSL.
2. Record start time.
3. On first streamed token, record TTFT.
4. On completion, record total tokens and compute TPS.
5. Log metrics to CSV or SQLite.

### 5.3 Reporting

Post‑process metrics with Pandas/Plotly:

- Compute p50/p95/p99 TTFT and TPS for:
  - Naïve vs cache‑aware routing.
  - Tenant A vs Tenant B.

- Plot over time:
  - Cache hit rate.
  - Per‑worker queue depth.
  - Throughput per worker.

---

## 6. Phased Timeline

### Phase 1 — Foundation (1–2 weeks)

**Goal:** basic end‑to‑end stack.

- Launch 2 mini‑SGLang workers with a small dense model (Llama‑3.2‑1B or Qwen‑3‑0.6B).
- Implement Axon Router with naïve round‑robin routing.
- Implement a minimal traffic generator logging TTFT/TPS.

Deliverables:

- Two worker processes.
- Router proxying requests.
- CSV of per‑request TTFT/TPS for sanity check.

---

### Phase 2 — Cache‑Aware Routing (1–2 weeks)

**Goal:** implement prefix‑aware routing and show latency gains.

- Implement prefix hashing and in‑memory index.
- Add cache‑aware routing alongside naïve routing, controllable via config or query param.
- Add Prometheus metrics for TTFT, TPS, hit/miss.
- Run experiments on mixed Tenant A/B workloads.

Deliverables:

- Working cache‑aware router.
- Benchmark plots showing TTFT improvement for Tenant A vs naïve routing.

---

### Phase 3 — Benchmarking Harness (1 week)

**Goal:** robust, reproducible benchmarks.

- Fully parameterise traffic generator (ISL/OSL distributions, arrival processes, tenant mix).
- Implement multiple runs and seed control.
- Generate clean plots and summary tables.

Deliverables:

- `bench/runner.py` with CLI flags.
- Plots and a short write‑up in README.

---

### Phase 4 — GPU Profiling (3–4 days)

**Goal:** connect system metrics to GPU behaviour.

- Use PyTorch Profiler around model forward passes to separate prefill vs decode cost.
- Run Nsight Systems on Yoga Pro 9i via WSL2 to inspect kernel timeline:
  - Naïve routing vs cache‑aware routing.
- Optionally profile single attention kernel with Nsight Compute for arithmetic intensity and memory bandwidth.

Deliverables:

- Profiling traces (screenshots / saved sessions).
- Explanation in docs showing how cache hits reduce prefill compute.

---

### Phase 5 — Speculative Decoding & Quantisation (1–2 weeks)

**Goal:** extend Axon with two orthogonal optimisation techniques.

**Speculative decoding:**

- Enable speculation on one SGLang worker (e.g. EAGLE‑style draft tokens).
- Route a subset of traffic to speculative worker.
- Measure end‑to‑end latency speedup and acceptance rate.

**Quantisation:**

- Run workers with 4‑bit quantised model vs BF16 baseline.
- Measure VRAM usage, TTFT/TPS changes.
- Note any quality difference on a small eval set.

Deliverables:

- Comparative results tables/plots for speculation and quantisation.

---

### Phase 6 — Gemma 4 Demo & mini‑SGLang Contribution (2–3 weeks)

**Goal:** tie everything to a modern model and make an OSS contribution.

**Gemma 4 demo:**

- Swap workers to full SGLang running Gemma 4 E4B (Q4‑bit) on the RTX GPU.
- Re‑run Phase 2–3 benchmarks and update README with Gemma 4‑specific results.
- Optionally send image‑heavy prompts to show VLM behaviour and increased KV cache usage.

**mini‑SGLang Gemma 4 text support:**

- Implement text‑only Gemma 4 E4B in mini‑SGLang:
  - Dual‑config attention (interleaved sliding window + global layers with differing KV geometry).
  - Proportional RoPE (p‑RoPE) on global layers.
  - Shared K=V for global layers.
- Skip the vision encoder initially; focus on text.
- Submit as PR to `sgl-project/mini-sglang`.

Deliverables:

- Gemma 4 benchmark section in README.
- Open PR adding Gemma 4 support to mini‑SGLang.

---

## 7. Knowledge Checklist by Phase

### For Phase 1

- Python async (`asyncio`, `await`, `httpx` or `aiohttp`).
- Basic HTTP and JSON.
- Running Python projects from source, virtualenvs.
- Basic PyTorch (forward pass, tensor shapes).

### For Phase 2–3

- Tokenisation and why prefixes matter for KV cache.
- Hash tables and LRU cache patterns.
- Prometheus metrics (counters, histograms, gauges).
- Simple statistics for latency (p50/p95/p99) and throughput.

### For Phase 4–5

- GPU architecture basics: VRAM vs compute, SMs, kernels, roofline intuition.
- PyTorch Profiler usage.
- Nsight Systems basics (understanding GPU timelines).
- Quantisation concepts (Q4/Q8, speed‑quality tradeoffs).
- Speculative decoding concepts (draft tokens, acceptance rate).

### For Phase 6

- Transformer internals (attention, MLPs, KV cache) at implementation level.
- Rotary embeddings (RoPE) and proportional RoPE (p‑RoPE).
- Handling attention layers with different `head_dim` and `num_kv_heads`.
- Reading model configs (e.g. Gemma 4 `config.json`) and mapping them to code.

---

## 8. Suggested Repo Layout

```text
axon/
  router/
    main.go            # or main.py
    router.go          # routing + HTTP handlers
    prefix_index.go    # index + eviction
    metrics.go         # Prometheus instrumentation

  workers/
    minisgl_config.yaml
    launch_minisgl.sh
    sglang_gemma4.yaml
    launch_sglang.sh

  bench/
    traffic_model.py   # tenant + ISL/OSL definitions
    runner.py          # async load generator
    analysis.ipynb     # plots + summary

  docs/
    DESIGN.md          # this document (or trimmed version)
    ARCHITECTURE.md

  CLAUDE.md            # instructions for Claude Code / other AI tools
  pyproject.toml / go.mod
  README.md
```

---

## 9. README Elevator Pitch

> Axon is a small LLM inference stack that sits in front of vLLM/SGLang and routes each request to the GPU that already remembers its prompt, turning KV‑cache hits into real TTFT and throughput wins — demonstrated on modern models like Gemma 4.
