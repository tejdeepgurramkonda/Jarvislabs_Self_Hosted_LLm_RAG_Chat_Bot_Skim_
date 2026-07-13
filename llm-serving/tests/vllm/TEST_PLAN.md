# vLLM Functional Test Plan

Target: an OpenAI-compatible **vLLM** server on JarvisLabs serving **Qwen2.5-7B-Instruct**
under the model name `qwen`. This suite is **isolated and read-only** — it only sends
HTTP requests to the `/v1` endpoint. It does not touch the FastAPI backend, the
frontend, or the `llm-serving/` reference code, and it never restarts/reconfigures the
server.

For each test: an **ID**, what it **checks**, the **request**, and the **IDEAL** result
used to judge actual-vs-expected in `TEST_RESULTS.md`.

> The long-prompt test (E2) reads the real `max_model_len` from `GET /v1/models` (H1)
> and adapts, rather than hardcoding a context size.

## 1. Reachability & health
| ID | Checks | Request | Ideal result |
|----|--------|---------|--------------|
| H1 | `/v1/models` responds & lists the model | `GET {BASE_URL}/models` | 200; JSON `data[]` contains an id equal to `MODEL` / containing "qwen". Capture reported `max_model_len`. |
| H2 | `/health` liveness | `GET {root}/health` (vLLM serves this at host root, not under `/v1`) | 200. If 404, note "not exposed" — non-fatal PARTIAL. |

## 2. Basic correctness
| ID | Checks | Request | Ideal result |
|----|--------|---------|--------------|
| C1 | Coherent on-topic answer | chat: user "Say hello in one short sentence." (max_tokens 32, temp 0) | Non-empty greeting; `role=assistant`; `finish_reason=stop`. |
| C2 | Known fact | chat: user "What is the capital of France? One word." (temp 0) | Answer contains "Paris". |
| C3 | Simple reasoning | chat: user "What is 17 + 25? Reply with just the number." (temp 0) | Answer contains "42". |

## 3. Parameters behave
| ID | Checks | Request | Ideal result |
|----|--------|---------|--------------|
| P1 | `max_tokens` caps output | chat: "Write a long paragraph about the ocean." `max_tokens=16` | `completion_tokens ≤ 16`; `finish_reason=length`. |
| P2a | temp 0 (near-)deterministic | same prompt ×3, `temperature=0`, `max_tokens=48` | 3 outputs identical (or ≥2/3 identical). |
| P2b | high temp varies | same prompt ×3, `temperature=1.3`, `top_p=0.95` | ≥2 distinct outputs. |
| P3 | stop sequence + finish_reason | "Count from 1 to 9 separated by commas." `stop=["5"]` | Output stops before/at "5"; `finish_reason=stop`. |

## 4. Streaming
| ID | Checks | Request | Ideal result |
|----|--------|---------|--------------|
| S1 | Incremental tokens | chat `stream=True`: "List three fruits, one per line." | ≥2 chunks arrive over time (not a single blob); reassembled text is coherent. |
| S2 | TTFT & tokens/sec | chat `stream=True`, ~120-token answer; time first chunk & total | Records TTFT (ideal < ~2s) and tokens/sec; PASS if it streams and numbers are sane. |

## 5. Chat behavior
| ID | Checks | Request | Ideal result |
|----|--------|---------|--------------|
| B1 | Multi-turn memory | messages: system + user "My name is Ada." + assistant "Nice to meet you, Ada." + user "What's my name?" | Answer contains "Ada". |
| B2 | System prompt changes style | same user "Describe the sea." with system A "terse pirate" vs system B "formal oceanographer" | Two outputs clearly differ in style/tone. |

## 6. Robustness / edge cases (expect graceful handling, not crashes)
| ID | Checks | Request | Ideal result |
|----|--------|---------|--------------|
| E1 | Empty prompt | chat: user content `""` | Valid completion **or** clean 4xx — no hang, no 500. |
| E2 | Long prompt near context limit | prompt ≈ (reported `max_model_len` − margin) tokens | Either completes **or** returns a clear context-length 4xx — no hang/500. |
| E3 | Huge `max_tokens` | chat: `max_tokens=999999` | Server clamps or returns 400 context error; no crash. |
| E4 | Wrong model name | chat with `model="not-a-real-model"` | 404/400 error naming the bad model. |
| E5 | Invalid API key | request with dummy `Authorization: Bearer bad-key` | 401 (if auth enabled). If server has no auth → **N/A**, noted. |
| E6 | Malformed body | raw POST with `messages` as a string and `temperature="hot"` | 400/422 validation error; no hang. |

## 7. Concurrency (light, read-only)
| ID | Checks | Request | Ideal result |
|----|--------|---------|--------------|
| X1 | 1 stream (baseline) | 1 streaming request, short answer | Succeeds; record TTFT + tokens/sec. |
| X2 | 5 concurrent streams | 5 simultaneous streaming requests | All succeed; record avg TTFT, aggregate tokens/sec, failures=0. |
| X3 | 10 concurrent streams | 10 simultaneous streaming requests | All succeed; TTFT degrades gracefully; failures=0. |

**Approx request budget:** ~48 requests, mostly ≤64-token generations — deliberately
modest to limit GPU time.
