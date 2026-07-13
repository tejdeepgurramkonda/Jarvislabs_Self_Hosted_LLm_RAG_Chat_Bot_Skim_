"""FastAPI application entrypoint for the RAG backend.

Run from the rag-backend/ directory with:
    uvicorn app.main:app --reload --port 8080

Routers are added phase by phase. Phase A wires only /health.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .api import chat, documents, health, retrieval
from .configs.config import settings
from .services import reranker
from .utils.logger import get_logger

log = get_logger(__name__)

app = FastAPI(title="RAG Chat Backend", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(documents.router)
app.include_router(retrieval.router)  # TEMPORARY: retrieval inspection (Phase C)
app.include_router(chat.router)


@app.on_event("startup")
def _startup() -> None:
    log.info("RAG backend starting. vLLM -> %s | model=%s", settings.llm_v1_url, settings.llm_model)
    log.info(
        "Embedding=%s | faiss_top_k=%s (cosine gate>=%s) -> rerank(%s) -> final_top_k=%s",
        settings.embedding_model, settings.faiss_top_k, settings.similarity_threshold,
        settings.rerank_model, settings.final_top_k,
    )
    # Load the cross-encoder reranker ONCE at startup so no request pays the cost.
    reranker.load_model()


@app.get("/")
def root() -> dict:
    return {"service": "rag-backend", "docs": "/docs", "health": "/health"}
