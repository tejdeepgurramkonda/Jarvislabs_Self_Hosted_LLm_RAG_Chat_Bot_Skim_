# RAG System — Integration Test Results

- **Run at:** 2026-07-13 17:13:40
- **Type:** connections BETWEEN services, real requests/network/streaming (no mocks).
- **Labels:** SEAM1/SEAM2 = pairwise integration; FULL/CROSS = end-to-end (API-level; real-browser layer deferred).

## Summary

| Status | Count |
|--------|-------|
| ✅ PASS | 15 |
| ❌ FAIL | 0 |
| 🟡 PARTIAL | 0 |
| ⚪ SKIP | 0 |
| **Total** | **15** |

| ID | Seam | Test | Status |
|----|------|------|--------|
| S1.1 | SEAM1 | backend LLM client reaches live vLLM (auth OK) | ✅ PASS |
| S1.2 | SEAM1 | tokens stream incrementally BE<-vLLM | ✅ PASS |
| S1.3 | SEAM1 | bad API key -> clean fallback (no crash) | ✅ PASS |
| S1.4 | SEAM1 | vLLM unreachable/timeout -> fallback (no crash) | ✅ PASS |
| S1.5 | SEAM1 | retrieved context actually reaches the model | ✅ PASS |
| S2.1 | SEAM2 | upload via /documents/upload/stream (frontend's path) | ✅ PASS |
| S2.2 | SEAM2 | chat returns an SSE stream the UI can consume | ✅ PASS |
| S2.3 | SEAM2 | CORS allows the frontend origin | ✅ PASS |
| S2.4 | SEAM2 | error responses carry a `detail` field (UI shape) | ✅ PASS |
| F1 | FULL | grounded answer streams end-to-end with sources | ✅ PASS |
| F2 | FULL | no-context fallback travels back to the client | ✅ PASS |
| F3 | FULL | multi-turn: same session stays grounded on the doc | ✅ PASS |
| X1 | CROSS | streaming is incremental end-to-end (not one blob) | ✅ PASS |
| X2 | CROSS | final structured event after text + matches schema | ✅ PASS |
| X3 | CROSS | realistic sequence: upload -> ask -> recent -> ask again | ✅ PASS |

## Performance (streamed full-chain calls)

| call | approx_tok_per_s | token_frames | total_s | ttft_s |
|------|------|------|------|------|
| X1 (stream) | 56.0 | 21 | 0.721 | 0.346 |
| F1 (grounded) | 52.9 | 21 | 0.596 | 0.199 |

## Details

### S1.1 — backend LLM client reaches live vLLM (auth OK)  ✅ **PASS**

- **Seam:** SEAM1
- **Input:** GET {BACKEND_URL}/health
- **Ideal:** 200; llm.reachable=true; model 'qwen'
- **Actual:** reachable=True; model=qwen; base=https://f2e3b14426882.notebooksn.jarvislabs.net/v1

### S1.2 — tokens stream incrementally BE<-vLLM  ✅ **PASS**

- **Seam:** SEAM1
- **Input:** upload zephyr; POST /chat 'What core temperature does the Zephyr-7 reactor operate at?'
- **Ideal:** multiple token frames (incremental); metadata.status=answered
- **Actual:** status=200; token_frames=24; ttft=0.09936990001006052; meta_status=answered

### S1.3 — bad API key -> clean fallback (no crash)  ✅ **PASS**

- **Seam:** SEAM1
- **Input:** in-process BE -> real vLLM base, WRONG key; /chat with a doc
- **Ideal:** vLLM 401 caught -> llm_error fallback; no crash/stacktrace to client
- **Actual:** status=200; meta_status=llm_error; fallback=True; text="Sorry, I'm having trouble answering right now. Please try ag"

### S1.4 — vLLM unreachable/timeout -> fallback (no crash)  ✅ **PASS**

- **Seam:** SEAM1
- **Input:** in-process BE -> bad LLM_BASE_URL http://127.0.0.1:9; /chat with a doc
- **Ideal:** connection error caught -> llm_error fallback; stream still completes with done
- **Actual:** status=200; frames=['token', 'metadata', 'done']; meta_status=llm_error; text="Sorry, I'm having trouble answering right now. Please try ag"

### S1.5 — retrieved context actually reaches the model  ✅ **PASS**

- **Seam:** SEAM1
- **Input:** upload zephyr (invented fact); ask 'What core temperature does the Zephyr-7 reactor operate at?'
- **Ideal:** answer reflects the doc (contains '512') — grounding, not model priors
- **Actual:** answer='The Zephyr-7 reactor operates at a core temperature of exactly 512 kelvin during nominal operation.'

### S2.1 — upload via /documents/upload/stream (frontend's path)  ✅ **PASS**

- **Seam:** SEAM2
- **Input:** POST /documents/upload/stream (multipart) w/ X-Session-Id + Origin
- **Ideal:** SSE stage frames (extract/chunk/index) then done{doc_id,chunk_count}; GET /documents shows it
- **Actual:** content_type='text/event-stream; charset=utf-8'; stages=['extract', 'chunk', 'index']; done={'stage': 'done', 'step': 3, 'doc_id': 'c6279cf344e0457cbcd46e87a46a318c', 'filename': 'zephyr.pdf', 'chunk_count': 2, 'page_count': 2}; listed=1 docs / 2 vectors

### S2.2 — chat returns an SSE stream the UI can consume  ✅ **PASS**

- **Seam:** SEAM2
- **Input:** POST /chat (SSE)
- **Ideal:** content-type text/event-stream; well-formed token/metadata/done frames
- **Actual:** content_type='text/event-stream; charset=utf-8'; event_kinds=['done', 'metadata', 'token']; token_frames=24

### S2.3 — CORS allows the frontend origin  ✅ **PASS**

- **Seam:** SEAM2
- **Input:** OPTIONS /chat preflight from Origin http://localhost:5173; and a disallowed origin
- **Ideal:** preflight echoes allow-origin=frontend origin, allow-credentials=true; disallowed origin not echoed
- **Actual:** preflight status=200; allow_origin='http://localhost:5173'; allow_credentials='true'; disallowed_echoed=None

### S2.4 — error responses carry a `detail` field (UI shape)  ✅ **PASS**

- **Seam:** SEAM2
- **Input:** missing header (400), empty query (422), non-PDF (415)
- **Ideal:** each is a JSON body with a `detail` field, as api.js reads
- **Actual:** missing=400/detail=True; bad_body=422/detail=True; bad_type=415/detail=True

### F1 — grounded answer streams end-to-end with sources  ✅ **PASS**

- **Seam:** FULL
- **Input:** upload zephyr; ask 'What core temperature does the Zephyr-7 reactor operate at?'
- **Ideal:** streams token-by-token; answer contains '512'; status=answered; sources cite the uploaded doc
- **Actual:** tokens=21; status=answered; contains_expected=True; src_has_doc=True; ttft=0.19880219997139648; answer='The Zephyr-7 reactor operates at a core temperature of exactly 512 kelvin.'

### F2 — no-context fallback travels back to the client  ✅ **PASS**

- **Seam:** FULL
- **Input:** upload zephyr; ask absent question 'What is the annual GDP of France in US dollars?'
- **Ideal:** client gets NO_CONTEXT_FALLBACK text; status=no_context; sources=[]
- **Actual:** status=no_context; sources=0; text="I couldn't find that in your documents."

### F3 — multi-turn: same session stays grounded on the doc  ✅ **PASS**

- **Seam:** FULL
- **Input:** upload zephyr; Q1 temperature; Q2 follow-up about the designer (same session)
- **Ideal:** both answered from the same doc (Q1~512, Q2~Vantathe). NB backend is stateless (no chat memory by design) — this checks doc-context persistence
- **Actual:** Q1 status=answered has512=True; Q2 status=answered hasVantathe=True; Q2='The lead designer of the Zephyr-7 fusion reactor was Dr. Marisol Vantathe.'
- **Note:** Backend does not store conversation history; each turn is retrieved independently from the session's doc.

### X1 — streaming is incremental end-to-end (not one blob)  ✅ **PASS**

- **Seam:** CROSS
- **Input:** stream /chat, time the frames
- **Ideal:** >=2 token frames arriving over time; record TTFT + latency + tok/s
- **Actual:** token_frames=21; ttft=0.346s; total=0.721s; spread=0.375s; tok/s~56.0

### X2 — final structured event after text + matches schema  ✅ **PASS**

- **Seam:** CROSS
- **Input:** inspect stream order + validate metadata
- **Ideal:** all token frames precede the single metadata; metadata validates against ChatMetadata
- **Actual:** order_ok=True; sequence_tail=['token', 'token', 'metadata', 'done']; schema_valid=True

### X3 — realistic sequence: upload -> ask -> recent -> ask again  ✅ **PASS**

- **Seam:** CROSS
- **Input:** upload docA; ask; GET /documents; select docA; ask again with doc_id=docA
- **Ideal:** every step ok; recent lists docA; second (doc-scoped) answer is grounded
- **Actual:** ask1 status=answered; recent_has_doc=True; ask2 status=answered; ask2_grounded=True; ask2="The Zephyr-7 reactor's coolant loop uses a proprietary fluid code-named BlueSalt"
