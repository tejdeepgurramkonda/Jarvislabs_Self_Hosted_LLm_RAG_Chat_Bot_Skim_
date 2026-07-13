# Deploy / serving artifacts (from the JarvisLabs instance)

These are the exact orchestration scripts and their run logs used to set up the
environment, run the quantization experiments, and serve the model on the JarvisLabs
L4 instance. They live at `/home/` on that instance (outside the app package) and are
copied here so the repo reflects how the server was actually built and run.

The vLLM server itself runs **only on JarvisLabs** — these scripts drive it there.

## Scripts
| File | Purpose |
|------|---------|
| `setup_envs.sh` | Builds two venvs with `uv`: a base env (`requirements.txt`) and a separate `vllm-env` (vllm + openai + httpx). |
| `run_rest.sh` | Runs the remaining quantization experiments sequentially (fp16, fp8, gptq, awq), one model on the GPU at a time; failures are recorded, not fatal. |
| `serve_and_measure.sh` | For each variant: start its vLLM OpenAI server, wait for `/health`, measure streaming TTFT/ITL via `benchmarks/stream_measure.py`, then tear down and wait for VRAM to free. |
| `start_gptq.sh` | Starts the **persistent** GPTQ-Int4 server on port 8000 (`--served-model-name qwen`, `--max-model-len 8192`). Reads its API key from `/home/.vllm_key` — **the key is not stored in this repo**. This is the server the `tests/vllm/` suite targets. |

> Note: after every JarvisLabs pause/resume the server does **not** auto-start and the
> public host changes — re-run `start_gptq.sh` on the instance and refresh `BASE_URL`.

## logs/
Run/output logs captured from the instance (setup, per-variant serve/measure runs,
orchestration). `serve_gptq.log` is intentionally **excluded** because that log
contains the live API key.
