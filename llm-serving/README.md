# LLM Serving — quantization benchmarks + fast vLLM serving

A single project that does two jobs:

1. **Experiment** — benchmark several quantization methods (fp16, int8, awq, gptq)
   on the same prompts and compare them on VRAM, load time, latency, and throughput.
2. **Serve** — expose the chosen model over a FastAPI API (with `/health`, `/chat`,
   `/generate`, `/metrics`), backed by either the transformers or vLLM engine.

Target model: **Qwen2.5-7B-Instruct** on an NVIDIA **L4** (24 GB).

## The metrics we measure

| Metric | Meaning | Why it matters |
|--------|---------|----------------|
| **TTFT** (time to first token) | delay before the first token appears | how *responsive* it feels |
| **ITL** (inter-token latency) | average gap between generated tokens | how *smooth* the stream is |
| **E2E latency** | total time for the whole response | overall wait |
| **Throughput** (tokens/sec) | tokens produced per second, esp. under load | how many users you can serve |
| **VRAM** | GPU memory used | whether it fits / how much KV-cache headroom |

TTFT and ITL come from *streaming* generation (we timestamp each token). Throughput
under concurrency comes from the load test hitting a running vLLM server.

## Project structure

```
llm-serving/
├── app/                      # the FastAPI service (production code)
│   ├── api/                  # one router per endpoint
│   │   ├── health.py         #   GET  /health
│   │   ├── chat.py           #   POST /chat, POST /generate (with ?stream)
│   │   └── metrics.py        #   GET  /metrics (live GPU + rolling stats)
│   ├── services/
│   │   ├── model_loader.py   # low-level transformers/vLLM loaders
│   │   ├── inference.py      # TransformersEngine + VLLMEngine (one interface)
│   │   └── tokenizer.py      # tokenizer + chat-template helpers
│   ├── configs/config.py     # all settings (env-overridable)
│   ├── schemas/              # Pydantic request/response models
│   ├── utils/
│   │   ├── logger.py
│   │   └── benchmark.py      # timing primitives + metric math + tables
│   └── main.py               # assembles the app; loads the model ONCE at startup
├── experiments/              # one folder per method; each run.py is a 3-line wrapper
│   ├── fp16/  int8/  awq/  gptq/  vllm/
│   └── _common.py            # shared harness: load -> benchmark -> save JSON
├── benchmarks/
│   ├── results/              # experiment output JSONs land here
│   ├── compare.py            # merge all results into one comparison table + chart
│   └── load_test.py          # concurrency stress test vs a vLLM server
├── notebooks/                # exploration
├── requirements.txt
└── README.md
```

## The two engines

Both live behind one interface (`app/services/inference.py`), so the API and the
benchmark harness are engine-agnostic.

- **transformers** — reliable for `fp16` / `bf16` and `int8` / `nf4` (bitsandbytes).
  Supports true token streaming, so it gives real TTFT/ITL numbers.
- **vLLM** — the fast serving path and the most reliable way to run the official
  **AWQ / GPTQ** checkpoints (auto-selects the Marlin kernel on the L4).

> **Environment note:** vLLM pins its own torch build, which can clash with the
> transformers stack. Install vLLM in a **separate venv** and run the `awq`/`gptq`/
> `vllm` experiments from there; run `fp16`/`int8` from your main environment.

## Setup

```bash
pip install -r requirements.txt          # torch already present on the instance

# Separate env for vLLM (recommended):
python -m venv ~/vllm-env && ~/vllm-env/bin/pip install vllm openai httpx tabulate
```

## Run the experiments

From the repo root:

```bash
python -m experiments.fp16.run     # transformers, full precision
python -m experiments.int8.run     # transformers, 8-bit
~/vllm-env/bin/python -m experiments.awq.run    # vLLM, 4-bit AWQ
~/vllm-env/bin/python -m experiments.gptq.run   # vLLM, 4-bit GPTQ
~/vllm-env/bin/python -m experiments.vllm.run   # vLLM, fp16 (engine baseline)
```

Each writes `benchmarks/results/<label>.json`. Then compare:

```bash
python benchmarks/compare.py       # prints a table + saves throughput.png
```

## Results — the comparison

All five variants, same 20 prompts, Qwen2.5-7B-Instruct on one L4 (vLLM engine):

| Variant | Weights (GB) | TTFT (ms) | ITL (ms) | Throughput (tok/s) | E2E (s, ~170 tok) |
|---------|:------------:|:---------:|:--------:|:------------------:|:-----------------:|
| bf16 (baseline)     | 14.29 | 109.4 | 56.7 | 17.5 | 9.91 |
| fp16 (baseline)     | 14.29 | 106.5 | 56.8 | 17.5 | 9.90 |
| fp8                 |  8.17 |  79.6 | 34.4 | 28.8 | 5.97 |
| **gptq (Int4)** ✅  | **5.27** | **68.3** | **18.9** | **52.5** | **3.23** |
| awq (Int4)          |  5.29 |  68.6 | 19.0 | 52.0 | 3.25 |

![Throughput comparison](benchmarks/results/throughput.png)
![Weight size comparison](benchmarks/results/weights.png)

**Winner: GPTQ-Int4.** vs the fp16 baseline it is **~2.7× smaller** (5.3 vs 14.3 GB) and
**~3× faster** (52.5 vs 17.5 tok/s, 3.2 s vs 9.9 s end-to-end), with the lowest
inter-token latency. AWQ is effectively tied; GPTQ was chosen for its slightly better
throughput and stable Marlin kernel on the L4. The smaller footprint also frees VRAM
for a larger KV-cache (more context / concurrency). Raw per-variant logs and sample
generations are in `benchmarks/results/` (`*.log`, `*_samples.txt`) — spot-check those
to confirm quantization didn't degrade answer quality.

## Serve the winner (production) — `deploy/`

The RAG backend talks to vLLM's **OpenAI-compatible** server directly (fastest path).
The chosen GPTQ model is served by [`deploy/start_gptq.sh`](deploy/start_gptq.sh):

```bash
vllm serve Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4 \
  --host 0.0.0.0 --port 8000 --served-model-name qwen \
  --quantization gptq_marlin --max-model-len 8192 \
  --api-key "$(cat /home/.vllm_key)"          # key kept OUT of the repo
```

It exposes an OpenAI `/v1` API (`GET /v1/models`, `POST /v1/chat/completions`, …) as the
served model name **`qwen`**. On JarvisLabs the instance's public URL proxies port 8000.
Because the instance ID + public URL **change on every pause/resume**, the full
bring-up procedure lives in [`deploy/RESUME_RUNBOOK.md`](deploy/RESUME_RUNBOOK.md).
`deploy/` also contains the setup + benchmark-orchestration scripts and their run logs.

> The `app/` FastAPI wrapper (port 6006, `python -m app.main` with `ENGINE=vllm`,
> `QUANTIZATION=gptq`) is an alternative custom serving layer; the production RAG
> backend uses the raw vLLM server above.

Load-test a running server:

```bash
python benchmarks/load_test.py --url https://<url>/v1 --api-key <key> --model qwen
```

## Suggested workflow

1. Run all experiments → `compare.py` → pick the best VRAM/latency/quality trade-off.
2. Serve that config via `app/` (or vLLM's server) on port 6006.
3. Run `load_test.py` at rising concurrency to confirm it holds up under load.
4. Point your local backend/frontend at the public URL.
