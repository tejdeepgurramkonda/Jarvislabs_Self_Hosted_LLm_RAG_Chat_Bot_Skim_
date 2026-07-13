# Skim — RAG Chat Frontend (React + Vite)

The web UI for the RAG chatbot: upload a PDF, watch it get indexed, then chat with it and
get answers that **stream token-by-token** and **cite the pages** they came from. No dummy
data — real streamed upload progress, real session-isolated retrieval, real answers.

> _Screenshots / demo (drop assets in `../docs/assets/`):_
> ![Upload screen](../docs/assets/placeholder-upload.png)
> ![Chat with streamed answer + sources](../docs/assets/placeholder-chat.png)
> 🎥 Demo video: `../docs/assets/placeholder-demo.mp4`

## Stack
- **React 18** + **Vite 5** (JavaScript, no UI framework — hand-rolled styles, minimal deps)
- Native `fetch` + streamed `ReadableStream` SSE parsing (no extra deps)

## Features
- **Upload → Processing → Chat** flow with a live per-stage checklist (extract → chunk →
  index) driven by the backend's `/documents/upload/stream` SSE.
- **Streaming chat** — tokens render live with a typing caret; the final event carries the
  **source pages** the answer used.
- **Session isolation** — a per-tab id in `sessionStorage`, sent as `X-Session-Id` on every
  request, so each browser session only sees and queries its own PDF.
- **Persistent per-document chats** — each document keeps its own conversation
  (a `conversations` map persisted to `sessionStorage`); switching between docs in the
  "Recent" list restores that chat, and a refresh keeps them.
- **Markdown rendering** — assistant messages render **bold**, numbered/bulleted lists, and
  `code` via a small dependency-free renderer (`src/components/Markdown.jsx`).

## How it maps to the backend
- Upload → `POST /documents/upload/stream` (SSE stages drive the 4-step checklist)
- Chat → `POST /chat` (SSE `token` → bubble, `metadata` → source pages, `done`)
- Recent list → `GET /documents`, Delete → `DELETE /documents/{id}`
- Every request carries `X-Session-Id` (from `sessionStorage`)

## Structure
```
src/
  api.js         backend client (upload/list/delete/chat) + SSE reader
  session.js     per-tab session id
  config.js      API_BASE (VITE_API_BASE or http://localhost:8090)
  App.jsx        state + upload→processing→chat orchestration + per-doc persistence
  screens/       UploadScreen, ProcessingScreen, ChatScreen
  components/     Markdown.jsx, icons, LogoMark
```

## Run
```bash
cp .env.example .env          # set VITE_API_BASE to your backend URL
npm install
npm run dev                    # http://localhost:5173
npm run build                  # production build -> dist/
```
Backend CORS already allows `http://localhost:5173`. Requires the
[RAG backend](../rag-backend) running; for real Qwen answers the
[vLLM server](../llm-serving) must be up (otherwise chat shows the graceful fallback —
no frontend change needed once the backend `.env` points at a live endpoint).

## Notes
- No auth. Chat history is **client-side only** (per-tab `sessionStorage`) — there's no
  server-side conversation memory, so each question is answered from the document via
  retrieval, not from prior turns.
