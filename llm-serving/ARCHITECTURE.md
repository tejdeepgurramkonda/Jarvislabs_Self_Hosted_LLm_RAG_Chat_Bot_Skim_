# LLM‑Serving — Full Architecture & Data‑Flow Guide

> A ground‑up walkthrough of this project: every file, every step, and every
> piece of the runtime — from downloading the model off Hugging Face, through the
> quantization experiments, to serving the winning model on a public endpoint.
> Read this and you should understand the system well enough to rebuild it by hand.

**Model:** Qwen2.5‑7B‑Instruct • **GPU:** NVIDIA L4 (Ada, 24 GB, compute 8.9) •
**Serving engine:** vLLM 0.24 • **Host:** JarvisLabs container (region IN2).

---

## 0. How to read this document

The project does **two jobs**, and they share the same core code:

1. **Experiment** — benchmark several precisions/quantizations (bf16, fp16, fp8,
   INT4‑GPTQ, INT4‑AWQ) on the same prompts and pick the best for RAG.
2. **Serve** — take the winner (GPTQ INT4) and expose it as an OpenAI‑compatible
   HTTP API on a public URL.

Two workflows, described in **§6** (experiment) and **§7** (serving). The vLLM
internals you asked about — **KV cache, PagedAttention, continuous batching,
prefix caching, fp8 KV cache** — are in **§8**. If you only want those, jump there.

---

## 1. System overview

```
                          ┌─────────────────────────────────────────────┐
                          │   Hugging Face Hub (model weights + config)   │
                          └───────────────────────┬─────────────────────┘
                                                  │  download (once)
                                                  ▼
              ┌──────────────────────────  /home/model_cache  ──────────────────────────┐
              │  models--Qwen--Qwen2.5-7B-Instruct           (15 GB, bf16/fp16/fp8)       │
              │  models--Qwen--Qwen2.5-7B-Instruct-GPTQ-Int4 (5.3 GB, the winner)         │
              │  models--Qwen--Qwen2.5-7B-Instruct-AWQ       (5.2 GB)                     │
              └───────────────┬──────────────────────────────────────┬───────────────────┘
                              │                                       │
             WORKFLOW A       │                          WORKFLOW B   │
        (experiments/*, offline)                     (vllm serve, online)
                              ▼                                       ▼
                  build_engine → VLLMEngine.generate()      vllm serve  (OpenAI API server)
                              │                                       │  binds 0.0.0.0:8000
                       metrics + RAG samples                          ▼
                              │                          JarvisLabs reverse proxy (HTTPS)
                              ▼                                       ▼
                  benchmarks/results/*.json          https://f2e3b14426882.notebooksn.jarvislabs.net/v1
                  compare.py → table + charts                (your public, API‑key‑protected endpoint)
```

**Serving is done by vLLM's own OpenAI‑compatible server** (`vllm serve` via
`start_gptq.sh`) — PagedAttention + continuous batching + OpenAI routes, deployed as
your live endpoint. There is no custom HTTP app: an earlier thin FastAPI wrapper
(`app/main.py` + `app/api/*`) was **removed** to keep the project focused on the one
serving path you actually use. What remains under `app/` is the reusable core
(config, loaders, the engine interface, metrics) that the experiments import.

vLLM itself is used in **two modes**:

- **Offline** (`class LLM`) — inside `VLLMEngine` (`app/services/inference.py`), used by the **experiments** to batch‑generate and time.
- **Online** (`vllm serve`) — the **public endpoint**, and also the short‑lived servers used to measure streaming TTFT/ITL.

---

## 2. Repository map

```
llm-serving/
├── app/                         # reusable core (imported by the experiments)
│   ├── configs/config.py        # ALL settings (env-overridable) — the single source of truth
│   ├── schemas/request.py       # GenerationParams (the sampling knobs)
│   ├── services/
│   │   ├── tokenizer.py         # load_tokenizer + render_chat (applies the chat template)
│   │   ├── model_loader.py      # resolve_model_name, load_transformers_model, load_vllm_llm
│   │   └── inference.py         # TransformersEngine, VLLMEngine, build_engine  ← the engine interface
│   └── utils/
│       └── benchmark.py         # timing primitives + metric math + table formatting
│
├── experiments/                 # one folder per variant; each run.py is a 3-line wrapper
│   ├── _common.py               # THE experiment harness: run_and_save() does the real work
│   └── bf16/ fp16/ fp8/ gptq/ awq/   run.py wrappers (each calls run_and_save with its settings)
│
├── benchmarks/
│   ├── results/                 # experiment outputs land here (*.json, *_samples.txt, *.log, *.png)
│   ├── compare.py               # merge all results → comparison table + throughput.png + weights.png
│   ├── _inject_weights.py       # parse vLLM's "Model loading took X GiB" log → weights_gb into JSON
│   ├── stream_measure.py        # streaming TTFT/ITL client (hits a running vLLM server)
│   └── load_test.py             # concurrency stress test vs a running vLLM server
│
├── .env                         # overrides for config.py (model name, cache dir, port, etc.)
├── requirements.txt             # base env deps (transformers stack + benchmarking)
└── ARCHITECTURE.md              # this file
```

Helper scripts that live **on the instance** (not in the repo): `start_gptq.sh`
(launch the public server), `serve_and_measure.sh` (drove the TTFT/ITL pass),
`/home/.vllm_key` (the API key).

---

## 3. The two Python environments (and why)

vLLM pins its *own* build of PyTorch, which clashes with the version the
transformers stack wants. So the instance has **two isolated virtualenvs**:

| Env | Path | Contains | Used for |
|---|---|---|---|
| **base** | `/home/llm-serving/.venv` | transformers, accelerate, pandas, **matplotlib**, tabulate | `compare.py` charts, transformers‑engine experiments |
| **vLLM** | `/home/vllm-env` | vllm 0.24, torch 2.11, openai, httpx, tabulate, pydantic‑settings | all the experiments (they run on vLLM) and the public server |

Both are created with `uv` and both **persist under `/home`** across pause/resume,
so they never need reinstalling. Rule of thumb: *anything touching vLLM →
`/home/vllm-env/bin/python`; charts → `/home/llm-serving/.venv/bin/python`.*

---

## 4. Configuration — one source of truth

Everything configurable lives in [`app/configs/config.py`](app/configs/config.py).
It uses **pydantic‑settings**: each field has a default, and any field can be
overridden by an environment variable or a line in `.env` (same name, upper‑case).

Key fields and what they control:

| Setting | Default | Meaning |
|---|---|---|
| `model_name` | `Qwen/Qwen2.5-7B-Instruct` | the **base** checkpoint (HF repo id) |
| `model_cache_dir` | `/home/model_cache` | where weights are downloaded/loaded (persistent) |
| `engine` | `transformers` | `transformers` or `vllm` — which backend `build_engine` builds |
| `quantization` | `none` | `none/int8/nf4/awq/gptq/fp8` |
| `dtype` | `float16` | compute dtype for full‑precision paths |
| `max_model_len` | `8192` | max context (prompt + output) tokens |
| `gpu_memory_utilization` | `0.90` | fraction of the 24 GB vLLM may use (weights + KV cache) — see §8 |
| `max_new_tokens`, `temperature`, `top_p`, `top_k`, `repetition_penalty` | — | default generation knobs (§10) |
| `benchmark_warmup`, `benchmark_runs` | `1`, `5` | untimed warmups + timed runs per prompt in the harness |

`settings = Settings()` at import time gives every module the same object. The
`.env` on the instance sets `PORT=8000`, `MODEL_CACHE_DIR=/home/model_cache`, etc.

---

## 5. Core abstractions (shared by both workflows)

### 5.1 The generation knobs — `GenerationParams`
[`app/schemas/request.py`](app/schemas/request.py) defines a validated Pydantic
model: `max_new_tokens`, `temperature`, `top_p`, `top_k`, `repetition_penalty`,
each with bounds. This is the single object that carries "how to sample" through
the whole system. (Full explanation of each knob in **§10**.)

### 5.2 Tokenizer + chat template
[`app/services/tokenizer.py`](app/services/tokenizer.py):
- `load_tokenizer(model_name, cache_dir)` → `AutoTokenizer.from_pretrained(...)`.
- `render_chat(tokenizer, messages)` → `tokenizer.apply_chat_template(...)`. This
  wraps your `[{role, content}, ...]` list into Qwen's exact chat format
  (`<|im_start|>system … <|im_end|><|im_start|>user … <|im_end|><|im_start|>assistant`).
  Getting this right is what makes the model actually behave like an assistant and
  stop cleanly on `<|im_end|>` (its EOS).

### 5.3 Which checkpoint for which quantization — `resolve_model_name`
[`app/services/model_loader.py`](app/services/model_loader.py):
```python
QUANT_CHECKPOINTS = {
    "awq":  "Qwen/Qwen2.5-7B-Instruct-AWQ",
    "gptq": "Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4",
}
def resolve_model_name(base_model, quantization):
    return QUANT_CHECKPOINTS.get(quantization, base_model)
```
- **AWQ / GPTQ** are *pre‑quantized* checkpoints — separate HF repos with 4‑bit
  weights baked in. So they map to their own repo.
- **bf16 / fp16 / fp8** all use the **base** repo. fp8 is special: there is no
  "fp8 repo" — vLLM quantizes the base weights to fp8 *on the fly at load time*
  ("online dynamic quantization"). That's why fp8 reuses the 15 GB base download.

### 5.4 The engine interface — `build_engine`
[`app/services/inference.py`](app/services/inference.py) hides the backend behind
one interface. Every engine exposes the same methods:

```
.generate(messages, params) -> {text, prompt_tokens, completion_tokens,
                                 ttft_s, total_time_s, tokens_per_second}
.stream(messages, params)    -> yields text pieces
.gpu_memory()                -> live VRAM report
.load_time                   -> seconds to load
```

`build_engine(engine=..., quantization=..., dtype=..., ...)`:
1. `model_name = resolve_model_name(base_model, quantization)`
2. if `engine == "vllm"` → `VLLMEngine(...)`, else → `TransformersEngine(...)`.

- **`TransformersEngine`** loads a HF model and does *true token streaming* using a
  background thread + `TextIteratorStreamer` (so it can measure real TTFT/ITL). Best
  for bf16/int8 experiments where you want per‑token timing.
- **`VLLMEngine`** wraps vLLM's offline `class LLM`. Its `generate()` returns the
  whole completion at once with a correct token count; its `stream()` yields that
  text in one piece (the offline API has no per‑token stream — that's why streaming
  TTFT/ITL is measured against the *server* instead; see §6.4).

Because the API and the experiments both talk to this interface, they don't care
which backend is running.

---

## 6. WORKFLOW A — the quantization experiment pipeline

Goal: for each variant, load it, generate on a fixed prompt set, record
memory + latency + throughput, save the actual answers for quality review.

### 6.1 The entry point (a 3‑line wrapper)
Example — [`experiments/gptq/run.py`](experiments/gptq/run.py):
```python
from experiments._common import run_and_save
run_and_save(quantization="gptq", engine="vllm", label="gptq")
```
`bf16` passes `dtype="bfloat16"`, `fp16` passes `dtype="float16"`, `fp8` passes
`quantization="fp8"`. That's the *only* thing that differs between variants — a
deliberate design so the comparison is apples‑to‑apples.

### 6.2 The harness — `run_and_save` (the real work)
[`experiments/_common.py`](experiments/_common.py), step by step:

```
run_and_save(quantization, engine, label, dtype)
  │
  1. params = QWEN_PARAMS            # temp 0.7, top_p 0.8, top_k 20, rep 1.05, 256 tokens
  │
  2. eng = build_engine(engine="vllm", quantization=..., dtype=..., cache_dir=/home/model_cache, ...)
  │        └─► VLLMEngine.__init__ ─► load_vllm_llm(...) ─► vllm.LLM(model=<repo>, quantization=<kernel>, dtype=...)
  │                                                          └─► Hugging Face download (first time) → /home/model_cache
  │                                                          └─► weights loaded onto GPU; KV-cache pool reserved
  │
  3. warmup: eng.generate(PROMPTS[0]) once  (untimed — lets CUDA/kernels settle)
  │
  4. TIMED: for each of 4 PROMPTS × benchmark_runs (5) → eng.generate(...)   # 20 generations
  │        each returns total_time_s, completion_tokens, tokens_per_second
  │
  5. result = summarize_generations(label, gens, extra={engine, quantization, dtype,
  │                                  load_time_s, vram_*})   # means over the 20 runs
  │
  6. _save_samples(label, eng, params): generate answers to 3 RAG prompts →
  │        benchmarks/results/<label>_samples.txt   (for human quality review)
  │
  7. save_result(result) → benchmarks/results/<label>.json ;  print a one-row table
```

Why `generate()` and not `stream()` for vLLM timing? Because vLLM's offline
`stream()` yields the full text as a single piece — feeding that to a per‑token
timer would count "1 token" and report garbage throughput. `generate()` returns the
real `completion_tokens`, so `summarize_generations`
([`app/utils/benchmark.py`](app/utils/benchmark.py)) computes correct
tokens/sec and E2E. (TTFT/ITL are left `null` here on purpose — see §6.4.)

### 6.3 Capturing memory — `_inject_weights.py`
`nvidia-smi` is useless for comparing variants, because vLLM pre‑reserves
`gpu_memory_utilization × 24 GB` for the KV cache on *every* run — so total usage
looks identical regardless of quantization. The honest number is vLLM's own
startup log line: `Model loading took 14.29 GiB memory …`.
[`benchmarks/_inject_weights.py`](benchmarks/_inject_weights.py) greps that out of
`<label>.log` and writes `weights_gb` into `<label>.json`. That's what shows the
real memory story: **bf16 14.3 GB → fp8 8.2 GB → INT4 ~5.3 GB.**

### 6.4 Measuring TTFT/ITL — the streaming pass
The offline API can't produce per‑token timing, and for RAG the metric that
matters (TTFT) is dominated by prefill over a *long* context. So TTFT/ITL are
measured against a **running server** with a streaming client:
- `serve_and_measure.sh` (on the instance) loops each variant: `vllm serve …` →
  wait for `/health` → run `stream_measure.py` → tear the server down → wait for
  the GPU to free → next variant.
- [`benchmarks/stream_measure.py`](benchmarks/stream_measure.py) opens streaming
  chat completions on RAG‑length prompts, timestamps the **first content token**
  (TTFT) and derives **ITL** = `(e2e − ttft) / (tokens − 1)`, then merges
  `ttft_ms_mean / ttft_ms_p90 / itl_ms_mean` into `<label>.json`.

### 6.5 Aggregating — `compare.py`
[`benchmarks/compare.py`](benchmarks/compare.py) reads every
`benchmarks/results/*.json`, orders them (`bf16, fp16, fp8, gptq, awq`), prints a
GitHub‑markdown table, and saves two bar charts: `throughput.png` and `weights.png`.
Run it from the **base** env (it needs matplotlib).

### 6.6 What the study concluded
Quality was a tie on the RAG prompts (even INT4 correctly refuses an
unanswerable question). INT4 wins on memory (~5.3 vs 14.3 GB), throughput
(52 vs 17.5 tok/s), TTFT (68 vs 107 ms) and ITL (19 vs 57 ms). **Winner: GPTQ INT4.**

---

## 7. WORKFLOW B — serving the winner on a public endpoint

### 7.1 Launch — `start_gptq.sh`
```bash
/home/vllm-env/bin/vllm serve Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4 \
  --host 0.0.0.0 --port 8000 --served-model-name qwen \
  --quantization gptq_marlin \
  --max-model-len 8192 --gpu-memory-utilization 0.90 \
  --download-dir /home/model_cache \
  --enable-prefix-caching \
  --api-key "$(cat /home/.vllm_key)"
```
What each flag does:
- `serve <repo>` — load this checkpoint (from `/home/model_cache`, no re‑download).
- `--served-model-name qwen` — the name clients pass as `"model": "qwen"` (decoupled from the repo id).
- `--quantization gptq_marlin` — use the **Marlin** 4‑bit GEMM kernels (fast INT4 matmul on Ada).
- `--max-model-len 8192` — max prompt+output length; also bounds KV‑cache block math.
- `--gpu-memory-utilization 0.90` — vLLM may use 90% of 24 GB; after weights (~5.3 GB) the rest becomes KV cache (§8).
- `--download-dir /home/model_cache` — persistent cache location.
- `--enable-prefix-caching` — reuse KV for shared prompt prefixes (huge for RAG; §8.5).
- `--api-key …` — require `Authorization: Bearer <key>` on every request.

It's started with `setsid … &` so it **survives** the SSH session closing —
a persistent background server, logging to `/home/serve_gptq.log`.

### 7.2 From a private port to a public URL (the mapping you asked about)
The server binds `0.0.0.0:8000` **inside the container**. JarvisLabs was told to
expose port 8000 (`http_ports: "8000"`), so it runs a **reverse proxy** that maps a
public HTTPS subdomain to that container port and terminates TLS:

```
client (your laptop)
   │  HTTPS  (https://f2e3b14426882.notebooksn.jarvislabs.net/v1/...)
   ▼
JarvisLabs edge proxy   ── terminates TLS, routes by subdomain ──►  container:8000
                                                                        │
                                                                        ▼
                                                        vLLM API server (uvicorn/FastAPI)
```
The instance advertised **two** endpoint URLs (`…881`, `…882`); only the one wired
to 8000 answers — we found empirically that `…882` returns the model list and
`…881` gives 502. So the live base URL is
`https://f2e3b14426882.notebooksn.jarvislabs.net/v1`.

### 7.3 The lifecycle of one request
```
POST /v1/chat/completions            {model:"qwen", messages:[...], stream:true, temperature, top_p, top_k}
   │  Authorization: Bearer <key>
   ▼
[edge proxy] → [container:8000] → vLLM OpenAI server
   1. auth check (API key)                          → 401 if missing/wrong
   2. render messages with Qwen chat template       → token ids (prefill input)
   3. Scheduler admits the request into the running batch (continuous batching, §8.4)
   4. PREFILL: compute K,V for every prompt token in parallel; write them into KV-cache blocks (§8.2–8.3)
        └─ if the prefix is already cached (§8.5), skip recomputing it
   5. DECODE loop: one token per step
        └─ compute the new token's Q,K,V; attend over all cached K,V; sample next token (temp/top_p/top_k)
        └─ append the new token's K,V to the cache; stream the token back as an SSE `data:` chunk
   6. stop when EOS (<|im_end|>) or max_tokens reached; emit `data: [DONE]`
```
Your `load_test.py` exercised exactly this over the public URL at concurrency
1→50; aggregate throughput scaled 51→1252 tok/s because of steps 3–5 (§8.4).

### 7.4 Why there's no custom app
Earlier the repo also had a thin **FastAPI** wrapper (`app/main.py` + `app/api/*`)
exposing our own `/health`, `/chat`, `/generate`, `/metrics`. It was **removed**:
`vllm serve` already provides an OpenAI‑compatible API with far better throughput
(continuous batching), so the wrapper was redundant and only added confusion. If you
ever need custom endpoint names, a custom response shape, or business logic (extra
auth, logging, request shaping), you'd reintroduce a small app that calls
`build_engine(...)` (or proxies vLLM) — but for straight serving, vLLM's server is
the whole story. Note vLLM's built‑ins overlap in *name* only: its `/health` returns
a bare 200, and its `/metrics` is Prometheus text (not JSON); chat/generate live at
`/v1/chat/completions` and `/v1/completions`.

---

## 8. Inside vLLM — the memory & batching internals

This section answers the KV‑cache / PagedAttention / continuous batching /
prefix‑caching / fp8 questions directly.

### 8.1 GPU memory layout
When the server starts, vLLM claims `gpu_memory_utilization × total` = `0.90 × 24 ≈
21.6 GB` and splits it:

```
┌──────────────────────── ~21.6 GB vLLM budget (of 24 GB) ────────────────────────┐
│  model weights           │           KV-cache pool (the rest)                    │
│  GPTQ INT4 ≈ 5.3 GB      │  ≈ 15–16 GB  →  divided into fixed-size BLOCKS         │
└──────────────────────────┴───────────────────────────────────────────────────────┘
```
**This is why quantization matters for serving:** smaller weights → a *bigger* KV
pool → more concurrent requests and/or longer contexts. Going bf16 (14.3 GB) →
INT4 (5.3 GB) frees ~9 GB, which becomes KV capacity.

### 8.2 What the KV cache **is**, and why it must exist
A decoder‑only transformer generates one token at a time. To produce token *t*, the
attention layers need the **Key (K)** and **Value (V)** vectors of *every previous
token*. Those depend only on the tokens already seen — so recomputing them every
step would be O(N²) wasted work.

The **KV cache** stores each token's K and V the first time they're computed, so
each new step only computes K,V for the *one* new token and attends over the cached
rest. That turns per‑step cost from O(sequence) into O(1) new work + a lookup —
the single biggest reason autoregressive LLM serving is tractable.

**It is always on.** You don't "enable" the KV cache; you can only change *how it's
stored/managed*. Concretely for Qwen2.5‑7B (28 layers, 4 KV heads, head_dim 128,
GQA):
```
KV per token ≈ 2 (K and V) × 28 layers × 4 kv_heads × 128 dims × 2 bytes (fp16)
             ≈ 56 KB / token   →  an 8192-token context ≈ ~0.45 GB for ONE sequence
```
Multiply by many concurrent sequences and you see why the KV pool, not the weights,
usually decides how many users you can serve.

### 8.3 PagedAttention — how vLLM manages that cache
Naively you'd store each sequence's KV in one contiguous block sized to
`max_model_len`. That wastes huge amounts of memory (most sequences are short) and
fragments the pool. **PagedAttention** borrows the OS virtual‑memory idea:

- The KV pool is chopped into fixed‑size **blocks** (e.g., 16 tokens of KV each).
- Each sequence gets a **block table** mapping its logical positions → physical
  blocks, which need **not** be contiguous.
- Blocks are allocated on demand as a sequence grows, and freed when it finishes.

Result: almost no wasted/fragmented memory, so far more sequences fit — and blocks
can be **shared** between sequences (the basis of prefix caching, §8.5). This is
vLLM's core innovation and it's automatic.

### 8.4 Continuous batching — why throughput scaled 24×
Traditional "static" batching waits for a whole batch to finish before starting the
next — a long request stalls short ones. vLLM does **iteration‑level (continuous)
scheduling**: at *every* decode step the scheduler can admit newly arrived requests
and retire finished ones, repacking the GPU batch each iteration. The GPU stays
saturated. That's why in your load test aggregate throughput went 51 → 1252 tok/s
(1→50 concurrent) while per‑request E2E barely moved (5.0 → 8.4 s). Automatic; no
flag needed.

### 8.5 Prefix caching — **enabled** (`--enable-prefix-caching`), and why it's big for RAG
If two requests share a leading prefix — e.g., the *same system prompt*, or the
*same retrieved document* pasted as context — their prefix KV blocks are identical.
Prefix caching **reuses** those already‑computed blocks instead of recomputing the
prefill. In RAG you constantly resend a big fixed instruction and often the same
retrieved passages, so this cuts prefill work and TTFT substantially. Enabled via
the flag; it rides on PagedAttention's block sharing.

### 8.6 fp8 KV cache — **not** enabled (`--kv-cache-dtype fp8`), what it'd do
By default the KV cache is stored in the model's compute dtype (fp16/bf16, ~2
bytes/element). `--kv-cache-dtype fp8` stores it in 8‑bit (~1 byte), roughly
**doubling** KV capacity — i.e., ~2× the concurrent sequences or ~2× the context
length for the *same* VRAM. It's separate from **weight** quantization (GPTQ shrinks
the *weights*; this shrinks the *cache*).

Why we didn't turn it on yet:
- It's an extra, opt‑in knob that trades a *small* amount of numerical precision in
  attention for capacity. We deliberately shipped the honest, conservative config
  first and left this for the tuning round.
- It's most worthwhile when you're KV‑bound — very long RAG contexts or high
  concurrency. On the L4 with INT4 weights you already have a large KV pool, so it's
  a "need more headroom" upgrade rather than a default.

**How to enable it:** add `--kv-cache-dtype fp8` to `start_gptq.sh`'s `vllm serve`
line and restart the server. (Optionally pair with a longer `--max-model-len` to
spend the freed space on context.) Verify quality afterward on your RAG prompts —
the risk is a slight, usually negligible, quality change on hard cases.

### 8.7 How it all composes for your deployment
```
GPTQ INT4 weights (5.3 GB)  → leaves ~16 GB for KV
      + PagedAttention      → that 16 GB packs many sequences with no fragmentation
      + continuous batching → the GPU stays busy as users come and go  (throughput ↑)
      + prefix caching      → shared RAG context/system prompt isn't recomputed  (TTFT ↓)
      [ + fp8 KV cache ]     → OPTIONAL: ~2× KV capacity for longer contexts / more users
```

---

## 9. Quantization methods, in one place

| Variant | Bits (weights) | Where quantized | vLLM flag | Weights on L4 |
|---|---|---|---|---|
| **bf16** | 16 | none (native) | `--dtype bfloat16` | 14.3 GB |
| **fp16** | 16 | none (native) | `--dtype float16` | 14.3 GB |
| **fp8** | 8 | **at load** (dynamic, base repo) | `--quantization fp8` | 8.2 GB |
| **GPTQ** | ~4 (INT4) | **offline**, pre‑quantized repo | `--quantization gptq_marlin` | 5.3 GB |
| **AWQ** | ~4 (INT4) | **offline**, pre‑quantized repo | `--quantization awq_marlin` | 5.2 GB |

- **GPTQ** minimizes layer‑wise reconstruction error when rounding to 4 bits.
- **AWQ** ("activation‑aware") protects the most salient weight channels using
  activation statistics.
- **Marlin** kernels are optimized mixed‑precision (INT4×FP16) GEMM kernels — they're
  what make INT4 *fast* (not just small) on Ampere/Ada. `gptq_marlin` / `awq_marlin`
  force them.
- Quantization shrinks **weights**; it does not by itself touch the **KV cache**
  (that's §8.6).

---

## 10. The generation knobs, end to end

Defined in `GenerationParams` ([`app/schemas/request.py`](app/schemas/request.py)),
carried through `generate()`/`stream()`, and translated to each backend
(vLLM `SamplingParams` in `VLLMEngine._sampling`; HF `generate` kwargs in
`TransformersEngine._prepare`). For the server, clients pass them in the JSON body.

| Knob | What it does | Our RAG value |
|---|---|---|
| `temperature` | randomness of sampling; 0 = greedy/deterministic | 0.7 |
| `top_p` | nucleus sampling: consider the smallest set of tokens whose prob ≥ p | 0.8 |
| `top_k` | consider only the k most likely tokens | 20 |
| `repetition_penalty` | >1 discourages repeating tokens | 1.05 |
| `max_new_tokens` / `max_tokens` | hard cap on output length | 256 |
| **EOS / stop** | generation halts on the model's end token | Qwen's `<|im_end|>` (from the chat template) |

EOS is not a numeric knob — it's implicit in the chat template. Because
`render_chat` applies Qwen's template, the model emits `<|im_end|>` when done and
the engine stops there (that's why our samples were 7–72 tokens, not padded to 256).

---

## 11. Persistence & lifecycle

- **Everything we created lives under `/home`**, which is the *only* part of a
  JarvisLabs container that survives pause/resume: `/home/model_cache` (all 3
  models), both venvs, the code, `benchmarks/results`, `start_gptq.sh`, `.vllm_key`.
- Anything *outside* `/home` (system `apt` installs, `/tmp`, `/root/.cache`) is wiped
  on pause — we intentionally put nothing there.
- **`jl pause <id>`** stops compute billing (storage persists); **`jl resume`**
  brings it back — but may return a **new `machine_id`** and a **new public IP**
  (it did: 443202→443328, IP …82→…39). After a resume: `jl list` for the new id,
  update `~/.ssh/config` with the new IP, then `bash /home/start_gptq.sh` to relaunch.
- **The endpoint is only alive while the instance is Running** and the server
  process is up.

---

## 12. Security

- The public server requires `Authorization: Bearer <key>`; without it you get
  **401** (verified). The key is in `/home/.vllm_key` (mode 600) and a local copy in
  the scratchpad. Rotate by editing that file and restarting the server.
- The URL is public, so the key is the only thing standing between the world and
  your GPU. Treat it like a password; don't commit it.

---

## 13. End‑to‑end runbook (rebuild from scratch)

```bash
# 0. instance
jl create --gpu L4 --storage 100 --yes --json          # note the machine_id
#    add your SSH key IP to ~/.ssh/config so `jl exec` works

# 1. code + envs (on the instance)
#    upload the repo to /home/llm-serving
uv venv --system-site-packages /home/llm-serving/.venv
uv pip install --python /home/llm-serving/.venv/bin/python -r /home/llm-serving/requirements.txt
uv venv /home/vllm-env
uv pip install --python /home/vllm-env/bin/python vllm openai httpx tabulate pydantic-settings

# 2. experiments (vLLM env), one variant at a time
cd /home/llm-serving
for v in bf16 fp16 fp8 gptq awq; do
  /home/vllm-env/bin/python -m experiments.$v.run > benchmarks/results/$v.log 2>&1
  /home/vllm-env/bin/python benchmarks/_inject_weights.py $v benchmarks/results/$v.log
done
#    (optional) streaming TTFT/ITL: serve_and_measure.sh bf16 fp16 fp8 gptq awq

# 3. compare (base env, for charts)
/home/llm-serving/.venv/bin/python benchmarks/compare.py

# 4. serve the winner (vLLM env), persistent
echo "your-strong-key" > /home/.vllm_key && chmod 600 /home/.vllm_key
bash /home/start_gptq.sh
curl -s localhost:8000/health          # wait for 200

# 5. test from anywhere
curl -s https://<public-url>/v1/chat/completions \
  -H "Authorization: Bearer $(cat /home/.vllm_key)" -H "Content-Type: application/json" \
  -d '{"model":"qwen","messages":[{"role":"user","content":"hi"}],"max_tokens":64}'
python benchmarks/load_test.py --url https://<public-url>/v1 --api-key <key> --model qwen --levels 1,5,10,20,50

# 6. when done
jl pause <id>                          # stops compute billing; /home persists
```

---

## 14. Glossary

- **Prefill** — the first forward pass that processes the whole prompt and fills the
  KV cache; its cost scales with prompt length (why long RAG contexts raise TTFT).
- **Decode** — the per‑token generation loop after prefill.
- **TTFT** — Time To First Token (responsiveness); prefill‑bound.
- **ITL** — Inter‑Token Latency (streaming smoothness); decode‑bound.
- **E2E** — total time for a full response.
- **Throughput** — tokens/sec, especially aggregate under concurrency.
- **KV cache** — stored Keys/Values of past tokens so attention isn't recomputed (§8.2).
- **PagedAttention** — vLLM's block‑based KV memory manager (§8.3).
- **Continuous batching** — per‑iteration (re)scheduling of the running batch (§8.4).
- **Prefix caching** — reuse of KV blocks for shared prompt prefixes (§8.5).
- **Marlin** — fast INT4×FP16 GEMM kernels used for GPTQ/AWQ.
- **GQA** — Grouped‑Query Attention: fewer KV heads than query heads (Qwen has 4 KV
  heads), which shrinks the KV cache.

---

## 15. Where our numbers came from (this study)

| variant | weights (GB) | TTFT (ms)* | ITL (ms)* | throughput (tok/s)* | E2E (s)* |
|---|---|---|---|---|---|
| bf16 | 14.29 | 109.4 | 56.68 | 17.5 | 9.91 |
| fp16 | 14.29 | 106.5 | 56.78 | 17.5 | 9.90 |
| fp8 | 8.17 | 79.6 | 34.41 | 28.8 | 5.97 |
| **gptq** | **5.27** | **68.3** | **18.92** | **52.5** | **3.23** |
| awq | 5.29 | 68.6 | 19.03 | 52.0 | 3.25 |

\* single‑user, on‑instance (no internet). Public‑endpoint load test (from a laptop,
so TTFT includes the internet round‑trip to IN2): 51.4 tok/s @1 concurrent → 1252
tok/s @50 concurrent. Raw artifacts: `benchmarks/results/*.json`, `*_samples.txt`,
`throughput.png`, `weights.png`.
