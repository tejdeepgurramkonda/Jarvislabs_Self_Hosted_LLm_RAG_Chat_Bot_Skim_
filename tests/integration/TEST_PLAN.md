# RAG System — Integration Test Plan

Tests the **connections between** the three (individually-working) services, with
**real requests, real network, real streaming — no mocks**:
- **AI**: vLLM `qwen` on JarvisLabs (OpenAI-compatible)
- **Backend**: FastAPI + FAISS (`:8090`)
- **Frontend**: React/Vite (`:5173`) — exercised at the **API level** (browser layer deferred, see README)

Positives hit the **live** backend; SEAM-1 negatives use an **isolated in-process**
backend (throwaway index, bad LLM target) so a bad URL/key never disturbs the running
server. Each test uses a unique `X-Session-Id` and deletes its documents afterward.

## SEAM 1 — Backend ↔ AI (vLLM) — pairwise integration
| ID | Seam | Input | Ideal |
|----|------|-------|-------|
| S1.1 | BE reaches vLLM + auth | `GET /health` (live) | 200; `llm.reachable=true`, model `qwen` |
| S1.2 | tokens stream BE←vLLM | upload zephyr; `POST /chat` (live) | multiple token frames (incremental); `status=answered` |
| S1.3 | bad key → clean error | in-process BE → real vLLM, **wrong key** | vLLM 401 caught → `llm_error` fallback; no crash |
| S1.4 | unreachable → fallback | in-process BE → `http://127.0.0.1:9` | connection error caught → `llm_error` fallback; stream still ends |
| S1.5 | context reaches model | upload zephyr; ask its invented fact (live) | answer contains **512** — grounding, not model priors |

## SEAM 2 — Frontend ↔ Backend (API level) — pairwise integration
| ID | Seam | Input | Ideal |
|----|------|-------|-------|
| S2.1 | upload path UI uses | `POST /documents/upload/stream` + Origin (live) | SSE `stage`(extract/chunk/index)→`done{doc_id,chunk_count}`; `GET /documents` shows it |
| S2.2 | chat SSE consumable | `POST /chat` (live) | `text/event-stream`; well-formed `token`/`metadata`/`done` frames |
| S2.3 | CORS for FE origin | preflight `OPTIONS /chat` from `:5173`; disallowed origin | allow-origin echoes `:5173`, allow-credentials=true; disallowed not echoed |
| S2.4 | error shape | 400 (no header), 422 (empty query), 415 (non-PDF) | each JSON has a `detail` field (what `api.js` reads) |

## FULL CHAIN — Frontend → Backend → AI → back — end-to-end (API level)
| ID | Seam | Input | Ideal |
|----|------|-------|-------|
| F1 | grounded answer e2e | upload zephyr; ask answerable Q (live) | streams token-by-token; contains 512; `status=answered`; sources cite the doc |
| F2 | no-context fallback e2e | ask an absent Q (live) | client gets `NO_CONTEXT_FALLBACK`; `status=no_context`; sources=[] |
| F3 | multi-turn context | Q1 then follow-up Q2, same session (live) | both grounded from the doc. NB backend is stateless (no chat memory) — checks doc-context persistence |

## CROSS-CUTTING — end-to-end streaming / schema / UX
| ID | Seam | Input | Ideal |
|----|------|-------|-------|
| X1 | streaming incremental | time the `/chat` frames (live) | ≥2 token frames over time (spread>0); record TTFT/latency/tok-s |
| X2 | final event + schema | inspect order + validate (live) | all tokens precede the single `metadata`; validates `ChatMetadata` |
| X3 | realistic UX sequence | upload→ask→`GET /documents`→select→ask again scoped (live) | every step ok; recent lists the doc; 2nd doc-scoped answer grounded |

**Perf:** TTFT / end-to-end latency / approx tokens-per-sec captured for F1 and X1.
Real-LLM call volume kept modest.
