# vLLM Server Functional Tests

Isolated, **read-only** functional tests for the OpenAI-compatible **vLLM** server
(Qwen2.5-7B-Instruct, served as `qwen`) running on JarvisLabs. The suite only sends
HTTP requests to the server's `/v1` endpoint — it does **not** touch the FastAPI
backend, the frontend, or the `llm-serving/` reference code, and it never
restarts/reconfigures the server.

## What it covers
| Area | Tests |
|------|-------|
| Reachability & health | H1 (`/v1/models`), H2 (`/health`) |
| Basic correctness | C1 (coherent reply), C2 (fact), C3 (arithmetic) |
| Parameters | P1 (`max_tokens` cap), P2a (temp 0 determinism), P2b (high-temp variety), P3 (stop / finish_reason) |
| Streaming | S1 (incremental chunks), S2 (TTFT + tokens/sec) |
| Chat behavior | B1 (multi-turn memory), B2 (system-prompt style) |
| Robustness | E1 (empty), E2 (near context limit), E3 (huge max_tokens), E4 (bad model), E5 (bad key), E6 (malformed body) |
| Concurrency | X1/X2/X3 (1, 5, 10 concurrent streams) |

See [TEST_PLAN.md](TEST_PLAN.md) for each test's exact request and ideal result, and
[TEST_RESULTS.md](TEST_RESULTS.md) for the latest run (actual vs. ideal, PASS/FAIL).

## Configure
Requires Python 3.10+, and `openai`, `httpx`, `python-dotenv` (plus `pytest` for the
pytest entrypoint) — all already present in this environment.

Copy the template and fill it in:

```bash
cp .env.example .env
```

`.env` keys:
- `BASE_URL` — OpenAI-compatible base **including `/v1`**, e.g. `https://<host>.jarvislabs.net/v1`
- `API_KEY` — the key the server was launched with; leave blank if the server has no auth
- `MODEL` — served model name (default `qwen`)

> The JarvisLabs instance's public URL changes on every pause/resume, so update
> `BASE_URL` after each resume. Find it with `jl get <id> --json` (the exposed
> port-8000 entry in `endpoints`).

## Run
Primary entrypoint — runs everything and writes `TEST_RESULTS.md`:

```bash
python run_suite.py          # from tests/vllm/
# or from the repo root:
python tests/vllm/run_suite.py
```

Re-runnable pytest form (asserts each check reaches an acceptable status):

```bash
pytest tests/vllm/ -v
```

Both hit the **live** server, so the instance must be running and `BASE_URL` current.
The runner catches per-test errors, so one failure never stops the run; failures are
recorded in full in `TEST_RESULTS.md`.

## Cost note
The suite is deliberately modest (~48 small requests). When you're done testing,
**pause the JarvisLabs instance** (`jl pause <id> --yes`) so it stops billing GPU time.
