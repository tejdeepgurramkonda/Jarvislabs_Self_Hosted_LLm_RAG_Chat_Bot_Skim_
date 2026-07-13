# RAG Backend — Functional Test Plan

Isolated functional tests for the `rag-backend/` FastAPI service (PDF → chunk → embed
→ FAISS → retrieve → prompt → stream). Each endpoint and pipeline stage is tested on
its own. The **remote LLM is mocked** (deterministic, offline, free); one clearly
marked, skippable smoke test hits the real vLLM. Chunking/embedding/retrieval use the
**real** code. Storage is redirected to a throwaway temp dir; the real `data/index/`
is never touched.

For each test: **ID**, what it **checks**, the **input**, and the **IDEAL** result.

## 1. Health
| ID | Checks | Input | Ideal |
|----|--------|-------|-------|
| H1 | `/health` ok + reports LLM config | GET /health, `check_llm` mocked reachable | 200; `status=ok`; `llm.reachable=true`; `base_url` & `model` present |
| H2 | health degrades gracefully when LLM down | GET /health, `check_llm` mocked failing | 200 (never crashes); `llm.reachable=false`; `detail` set |

## 2. Document ingestion
| ID | Checks | Input | Ideal |
|----|--------|-------|-------|
| I1 | valid PDF → chunks embedded+indexed+persisted | POST /documents/upload (paris.pdf) + header | 200; `chunk_count>0`, `page_count>=1`, `doc_id`; vectors added; metadata has `doc_id/session_id/filename/page/chunk_idx/text`; index files written |
| I2 | non-PDF rejected cleanly | upload a .txt | 415, no crash |
| I3 | corrupt PDF handled | .pdf name, garbage bytes | clean 4xx ideally; must not hang/crash (flag if 500) |
| I4a | empty (0-byte) file | upload empty.pdf | 400 |
| I4b | PDF with no extractable text | blank page PDF | 422 (`IngestionError`) |
| I5 | listing, session-scoped | GET /documents after I1, as owner and other session | owner sees 1 (correct totals); other sees 0 |
| I6 | delete removes from index+store+file | DELETE /documents/{id} | cross-session → 404; owner → 200 `deleted_chunks>0`, vectors drop, PDF unlinked; unknown id → 404 |
| I7 | missing session header | upload without `X-Session-Id` | 400 |

## 3. Chunking (unit)
| ID | Checks | Input | Ideal |
|----|--------|-------|-------|
| CK1 | size + overlap behave | `split_text(long, 800, 120)` | ≥2 chunks; each ≤ chunk_size; consecutive chunks overlap; no empty chunks |
| CK2 | short text | `split_text('one short line...', 800, 120)` | exactly 1 chunk, preserved |

## 4. Embeddings (unit)
| ID | Checks | Input | Ideal |
|----|--------|-------|-------|
| E1 | deterministic | `embed_texts(['hello world'])` ×2 | identical vectors |
| E2 | L2-normalized | norm of a vector | ≈ 1.0 (±1e-5) |
| E3 | dimension + dtype | `embed_texts([...])` | shape `(n, 384)`, float32 |
| E4 | empty input | `embed_texts([])` | shape `(0, 384)`, no error |

## 5. Retrieval
| ID | Checks | Input | Ideal |
|----|--------|-------|-------|
| R1 | obvious answer retrieved | ingest paris+biology; `retrieve('Where is the Eiffel Tower?')` | `found=true`; top chunk mentions Paris/Eiffel; `top_score` ≥ threshold; ≤ top_k |
| R2 | no-match → threshold path | `retrieve('boiling point of mercury?')` | `found=false`; best score < threshold; empty kept list |
| R3 | cosine correctness | query == exact chunk text, threshold 0 | top ≈ 1.0; scores in [-1,1]; sorted desc |
| R4 | session scoping | ingest paris→A, biology→B; retrieve scoped | A returns paris; B never sees paris |

## 6. Prompt building
| ID | Checks | Input | Ideal |
|----|--------|-------|-------|
| PB1 | context injected + grounding rule | `build_messages(q, chunks)` | system forbids outside knowledge; user has `CONTEXT:` (chunk text) + `QUESTION:` |
| PB2 | source labels / empty | `format_context(chunks)` and `[]` | labels show filename+page; `[]` → "(no context available)" |

## 7. Chat generation (`/chat`, LLM mocked)
| ID | Checks | Input | Ideal |
|----|--------|-------|-------|
| C1 | happy path streams + validated metadata | POST /chat (matches paris), mock yields tokens | ≥2 `token` events; `metadata` status=answered, sources non-empty, fallback=false, valid `ChatMetadata`; `done` |
| C2 | SSE order | same | tokens… → metadata → done |
| C3 | grounded prompt passed to LLM | capture mock args | user message has `CONTEXT:` with fixture text |

## 8. Validation & fallback
| ID | Checks | Input | Ideal |
|----|--------|-------|-------|
| V1 | schema rejects bad body | `/chat` empty/missing query, top_k=0, threshold=1.5 | every case → 422 |
| V2 | missing session header | `/chat` without header | 400 |
| V3 | no relevant context → fallback | `/chat` with empty index | token=NO_CONTEXT_FALLBACK; status=no_context; sources=[]; fallback=true; LLM not called |
| V4 | LLM failure → fallback | mock raises immediately | token=LLM_ERROR_FALLBACK; status=llm_error; fallback=true; done present |
| V5 | mid-stream LLM failure | mock yields 2 tokens then raises | partial tokens; status=llm_error; done present; no duplicate fallback text |
| V6 | bad final payload → safe fallback | patch `run_chat_stream` to emit invalid status | route catches ValidationError → fallback token + safe metadata (llm_error) + done |

## 9. Real-LLM smoke (marked, skippable)
| ID | Checks | Input | Ideal |
|----|--------|-------|-------|
| SM1 | end-to-end wiring vs live vLLM | real `/chat` (RUN_REAL_LLM=1) | streams a real grounded answer; `metadata.status=answered`. Auto-skips unless enabled and reachable |
