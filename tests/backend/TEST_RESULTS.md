# RAG Backend — Functional Test Results

- **Run at:** 2026-07-13 08:29:38
- **Scope:** endpoints + pipeline stages in isolation; LLM **mocked** (except the skippable smoke test).
- **Index/uploads:** throwaway temp dir (real `data/index/` untouched).

## Summary

| Status | Count |
|--------|-------|
| ✅ PASS | 30 |
| ❌ FAIL | 0 |
| 🟡 PARTIAL | 1 |
| ⚪ SKIP | 0 |
| **Total** | **31** |

| ID | Area | Test | Status |
|----|------|------|--------|
| H1 | Health | /health ok + reports LLM config | ✅ PASS |
| H2 | Health | health degrades gracefully when LLM down | ✅ PASS |
| I1 | Ingestion | valid PDF -> chunks embedded + indexed + persisted | ✅ PASS |
| I2 | Ingestion | non-PDF rejected cleanly | ✅ PASS |
| I3 | Ingestion | corrupt PDF handled gracefully | 🟡 PARTIAL |
| I4a | Ingestion | empty (0-byte) file rejected | ✅ PASS |
| I4b | Ingestion | PDF with no extractable text handled | ✅ PASS |
| I5 | Ingestion | GET /documents lists doc, session-scoped | ✅ PASS |
| I6 | Ingestion | DELETE removes chunks from index + store + file | ✅ PASS |
| I7 | Ingestion | missing X-Session-Id -> 400 | ✅ PASS |
| CK1 | Chunking | chunk size + overlap behave as configured | ✅ PASS |
| CK2 | Chunking | short text -> single chunk | ✅ PASS |
| E1 | Embeddings | same input -> same vector | ✅ PASS |
| E2 | Embeddings | vectors are L2-normalized | ✅ PASS |
| E3 | Embeddings | correct dimension + dtype | ✅ PASS |
| E4 | Embeddings | empty input handled | ✅ PASS |
| R1 | Retrieval | obvious answer retrieved in top_k | ✅ PASS |
| R2 | Retrieval | no-match query -> threshold path (no_context) | ✅ PASS |
| R3 | Retrieval | cosine similarity correct (normalized IP) | ✅ PASS |
| R4 | Retrieval | retrieval is session-scoped | ✅ PASS |
| PB1 | Prompt | context injected + 'answer only from context' | ✅ PASS |
| PB2 | Prompt | source labels rendered; empty context handled | ✅ PASS |
| C1 | Chat | happy path streams tokens + validated metadata | ✅ PASS |
| C2 | Chat | SSE event order: tokens -> metadata -> done | ✅ PASS |
| C3 | Chat | retrieved context is injected into the LLM prompt | ✅ PASS |
| V1 | Validation | malformed request body -> 422 | ✅ PASS |
| V2 | Validation | missing X-Session-Id -> 400 | ✅ PASS |
| V3 | Validation | no relevant context -> graceful fallback, no LLM call | ✅ PASS |
| V4 | Validation | LLM failure (immediate) -> graceful fallback | ✅ PASS |
| V5 | Validation | mid-stream LLM failure -> partial answer, no duplicate fallback | ✅ PASS |
| V6 | Validation | invalid final payload -> safe fallback, not a broken response | ✅ PASS |

## Details

### H1 — /health ok + reports LLM config  ✅ **PASS**

- **Input:** GET /health (check_llm mocked reachable)
- **Ideal:** 200; status=ok; llm.reachable=true; base_url & model present
- **Actual:** 200; {'status': 'ok', 'service': 'rag-backend', 'llm': {'reachable': True, 'base_url': 'http://localhost:8000/v1', 'model': 'qwen', 'detail': None}}

### H2 — health degrades gracefully when LLM down  ✅ **PASS**

- **Input:** GET /health (check_llm mocked failing)
- **Ideal:** 200 (never crashes); llm.reachable=false; detail set
- **Actual:** 200; reachable=False; detail='APIConnectionError: simulated unreachable'

### I1 — valid PDF -> chunks embedded + indexed + persisted  ✅ **PASS**

- **Input:** POST /documents/upload (paris.pdf) + X-Session-Id
- **Ideal:** 200; chunk_count>0, page_count>=1, doc_id; vectors added; metadata has doc_id/session_id/filename/page/chunk_idx/text; index files written
- **Actual:** 200; body={'doc_id': 'be27449482aa4c458db4ee14e8e9f001', 'filename': 'paris.pdf', 'chunk_count': 1, 'page_count': 1}; vectors=1; meta_keys=['chunk_idx', 'doc_id', 'filename', 'page', 'session_id', 'text']; files_written=True

### I2 — non-PDF rejected cleanly  ✅ **PASS**

- **Input:** upload notes.txt (text/plain)
- **Ideal:** 415, no crash
- **Actual:** 415; {'detail': 'Only PDF files are supported.'}

### I3 — corrupt PDF handled gracefully  🟡 **PARTIAL**

- **Input:** upload bad.pdf (garbage bytes, application/pdf)
- **Ideal:** clean 4xx (415/422) ideally; must not hang/crash
- **Actual:** 500; {'detail': 'Failed to process the PDF.'}
- **Note:** Handled cleanly (JSON error, no crash) but returns 500, not a 4xx. Suggest mapping PdfReadError/PdfStreamError to a 422 in upload_document.

### I4a — empty (0-byte) file rejected  ✅ **PASS**

- **Input:** upload empty.pdf (0 bytes)
- **Ideal:** 400 (empty file), no crash
- **Actual:** 400; {'detail': 'The uploaded file is empty.'}

### I4b — PDF with no extractable text handled  ✅ **PASS**

- **Input:** upload blank.pdf (page, no text)
- **Ideal:** 422 IngestionError ('no extractable text'), no crash
- **Actual:** 422; {'detail': 'No extractable text found in the PDF (is it a scanned image?).'}

### I5 — GET /documents lists doc, session-scoped  ✅ **PASS**

- **Input:** upload as session A, then GET /documents as A and as B
- **Ideal:** A sees 1 doc with correct totals; B sees 0
- **Actual:** A: total_documents=1 total_vectors=1; B: total_documents=0

### I6 — DELETE removes chunks from index + store + file  ✅ **PASS**

- **Input:** upload, cross-session delete, owner delete, unknown-id delete
- **Ideal:** cross-session -> 404; owner -> 200 deleted_chunks>0, vectors drop, file unlinked; unknown -> 404
- **Actual:** cross=404; owner=200 {'doc_id': 'c9b2dcceef954b25a9fe4e98c14fcdf7', 'deleted_chunks': 1, 'status': 'deleted'}; vectors 1->0; pdf_gone=True; unknown=404

### I7 — missing X-Session-Id -> 400  ✅ **PASS**

- **Input:** POST /documents/upload without X-Session-Id
- **Ideal:** 400 (Missing X-Session-Id header)
- **Actual:** 400; {'detail': 'Missing X-Session-Id header.'}

### CK1 — chunk size + overlap behave as configured  ✅ **PASS**

- **Input:** split_text(len=5860, size=800, overlap=120)
- **Ideal:** >=2 chunks; each <= chunk_size; consecutive chunks share overlap; no empty chunks
- **Actual:** n_chunks=9; max_len=798; best_consecutive_overlap=97

### CK2 — short text -> single chunk  ✅ **PASS**

- **Input:** split_text('one short line of text', 800, 120)
- **Ideal:** exactly 1 chunk, content preserved
- **Actual:** n_chunks=1; chunk='one short line of text'

### E1 — same input -> same vector  ✅ **PASS**

- **Input:** embed_texts(['hello world']) x2
- **Ideal:** identical vectors
- **Actual:** array_equal=True; max_abs_diff=0.0

### E2 — vectors are L2-normalized  ✅ **PASS**

- **Input:** norm(embed_texts([...])[0])
- **Ideal:** L2 norm ~= 1.0 (+/- 1e-5)
- **Actual:** norm=1.000000

### E3 — correct dimension + dtype  ✅ **PASS**

- **Input:** embed_texts(['one','two','three'])
- **Ideal:** shape (3, 384), float32
- **Actual:** shape=(3, 384); dtype=float32

### E4 — empty input handled  ✅ **PASS**

- **Input:** embed_texts([])
- **Ideal:** shape (0, 384), no error
- **Actual:** shape=(0, 384)

### R1 — obvious answer retrieved in top_k  ✅ **PASS**

- **Input:** ingest paris+biology; retrieve('Where is the Eiffel Tower located?')
- **Ideal:** found=true; top chunk mentions Paris/Eiffel; top_score high; <= top_k results
- **Actual:** found=True; n=1; top_score=0.821; top_text='The Eiffel Tower is located in Paris, France. It was complet'

### R2 — no-match query -> threshold path (no_context)  ✅ **PASS**

- **Input:** ingest paris; retrieve('boiling point of mercury ...')
- **Ideal:** found=false; best score below threshold; empty kept list
- **Actual:** found=False; kept=0; top_score=0.031442880630493164

### R3 — cosine similarity correct (normalized IP)  ✅ **PASS**

- **Input:** retrieve(exact chunk text, threshold=0)
- **Ideal:** top score ~1.0; all scores in [-1,1]; sorted descending
- **Actual:** top=0.9553; in_range=True; descending=True; scores=[0.955]

### R4 — retrieval is session-scoped  ✅ **PASS**

- **Input:** ingest paris->A, biology->B; retrieve('Eiffel Tower') scoped to each
- **Ideal:** A returns paris chunks; B never returns paris content
- **Actual:** A_found=True A_docs=1; B_found=False B_has_paris=False

### PB1 — context injected + 'answer only from context'  ✅ **PASS**

- **Input:** build_messages('Where is the Eiffel Tower?', [paris chunk])
- **Ideal:** system forbids outside knowledge; user has CONTEXT with chunk text + QUESTION
- **Actual:** roles=['system', 'user']; system_has_only_context=True; user_has_context=True; user_has_question=True

### PB2 — source labels rendered; empty context handled  ✅ **PASS**

- **Input:** format_context([paris chunk]) and format_context([])
- **Ideal:** labels show filename + page; [] -> '(no context available)'
- **Actual:** labeled_has_src=True; empty='(no context available)'

### C1 — happy path streams tokens + validated metadata  ✅ **PASS**

- **Input:** POST /chat 'Where is the Eiffel Tower?' (LLM mocked, 3 tokens)
- **Ideal:** multiple token events (not one blob); metadata status=answered, sources non-empty, fallback=false, valid ChatMetadata; done event
- **Actual:** token_events=3; text='The Eiffel Tower is in Paris.'; status=answered; sources=1; fallback=False; schema_valid=True; done=1

### C2 — SSE event order: tokens -> metadata -> done  ✅ **PASS**

- **Input:** POST /chat (mocked)
- **Ideal:** all token events precede metadata, which precedes done
- **Actual:** sequence=['token', 'metadata', 'done']

### C3 — retrieved context is injected into the LLM prompt  ✅ **PASS**

- **Input:** capture messages passed to mocked stream_chat
- **Ideal:** messages built from retrieved chunks: user content has CONTEXT with fixture text
- **Actual:** called=True; roles=['system', 'user']; context_has_eiffel=True

### V1 — malformed request body -> 422  ✅ **PASS**

- **Input:** POST /chat with invalid bodies: empty query, missing query, top_k=0, threshold=1.5
- **Ideal:** every case -> 422 (schema rejects)
- **Actual:** {'empty query': 422, 'missing query': 422, 'top_k=0': 422, 'threshold=1.5': 422}

### V2 — missing X-Session-Id -> 400  ✅ **PASS**

- **Input:** POST /chat without X-Session-Id
- **Ideal:** 400 (Missing X-Session-Id header)
- **Actual:** 400; {'detail': 'Missing X-Session-Id header.'}

### V3 — no relevant context -> graceful fallback, no LLM call  ✅ **PASS**

- **Input:** POST /chat with empty index
- **Ideal:** token=NO_CONTEXT_FALLBACK; status=no_context; sources=[]; fallback=true; LLM not called
- **Actual:** text="I couldn't find that in your documents."; status=no_context; sources=0; fallback=True; llm_called=False

### V4 — LLM failure (immediate) -> graceful fallback  ✅ **PASS**

- **Input:** POST /chat; mocked stream_chat raises before any token
- **Ideal:** token=LLM_ERROR_FALLBACK; status=llm_error; fallback=true; no crash; done present
- **Actual:** text="Sorry, I'm having trouble answering right now. Please try again."; status=llm_error; fallback=True; done=1

### V5 — mid-stream LLM failure -> partial answer, no duplicate fallback  ✅ **PASS**

- **Input:** POST /chat; mock yields 2 tokens then raises
- **Ideal:** partial tokens shown; status=llm_error; stream ends with done; NO fallback text prepended (already emitted)
- **Actual:** text='The Eiffel Tower'; status=llm_error; fallback=True; done=1

### V6 — invalid final payload -> safe fallback, not a broken response  ✅ **PASS**

- **Input:** patch run_chat_stream to yield final status='bogus_status'
- **Ideal:** route catches ValidationError -> fallback token + safe metadata (llm_error) + done; response well-formed
- **Actual:** text="partial answer Sorry, I'm having trouble answering right now. Please try again."; metadata_status=llm_error; valid_schema=True; done=1
