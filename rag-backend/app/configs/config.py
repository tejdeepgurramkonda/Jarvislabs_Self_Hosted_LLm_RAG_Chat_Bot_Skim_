"""Central configuration for the RAG backend.

Every value can be overridden via a .env file or environment variables.
Secrets (the vLLM URL and API key) live ONLY in .env — see .env.example.
"""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# Project root = rag-backend/  (this file is app/configs/config.py)
BASE_DIR = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        protected_namespaces=(),  # allow field names beginning with "model_"
        extra="ignore",
    )

    # --- LLM endpoint (my vLLM server on JarvisLabs; OpenAI-compatible) ---
    # NOTE: the JarvisLabs subdomain changes every time the instance resumes, so
    # this is ALWAYS read from .env / env vars — never hardcoded anywhere else.
    # LLM_BASE_URL may be given with or without a trailing /v1; we normalize it.
    llm_base_url: str = "http://localhost:8000"
    llm_api_key: str = "EMPTY"           # vLLM often ignores this, but the SDK needs a value
    llm_model: str = "qwen"              # the served model name

    # --- Embeddings (CPU) ---
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_dim: int = 384             # all-MiniLM-L6-v2 output dimension

    # --- Chunking ---
    chunk_size: int = 400                # characters per chunk
    chunk_overlap: int = 80            # characters of overlap between chunks

    # --- Retrieval (two-stage: FAISS recall gate -> cross-encoder rerank for order) ---
    top_k: int = 8                       # default k for the offline evaluator (evaluation/eval_retrieval.py)
    faiss_top_k: int = 20                # 1st stage: candidate chunks pulled from FAISS
    final_top_k: int = 4                 # 2nd stage: chunks kept after reranking
    # 1st-stage COSINE floor = the "is there relevant context?" gate (drives the
    # no-context fallback). Cosine separates real questions (~0.3-0.6) from noise
    # (~0.15) cleanly; the reranker score does NOT, so it must not be the gate.
    similarity_threshold: float = 0.25

    # --- Reranking (cross-encoder; REORDERS the recalled candidates for precision) ---
    rerank_model: str = "BAAI/bge-reranker-base"
    # Optional 2nd-stage floor on the reranker score. Keep at 0.0 (ordering only):
    # bge-reranker scores are small in absolute terms, so a nonzero gate here would
    # wrongly drop valid chunks (e.g. broad/summary questions score near 0).
    rerank_threshold: float = 0.0

    # --- Generation ---
    max_tokens: int = 512                # max tokens the LLM may generate per answer
    temperature: float = 0.2

    # --- Networking / infra ---
    llm_timeout_seconds: float = 60.0
    cors_origins: list[str] = ["http://localhost:5173"]

    # --- Storage paths (absolute, derived from BASE_DIR) ---
    uploads_dir: Path = BASE_DIR / "data" / "uploads"
    index_dir: Path = BASE_DIR / "data" / "index"

    @property
    def faiss_index_path(self) -> Path:
        return self.index_dir / "faiss.index"

    @property
    def chunk_store_path(self) -> Path:
        return self.index_dir / "chunks.json"

    @property
    def llm_v1_url(self) -> str:
        """OpenAI-compatible endpoint, ending in exactly one /v1.

        Tolerates LLM_BASE_URL given with or without a trailing /v1 (the
        JarvisLabs URL is copy-pasted by hand, so both forms show up).
        """
        base = self.llm_base_url.rstrip("/")
        if base.endswith("/v1"):
            return base
        return base + "/v1"

    def ensure_dirs(self) -> None:
        self.uploads_dir.mkdir(parents=True, exist_ok=True)
        self.index_dir.mkdir(parents=True, exist_ok=True)


settings = Settings()
settings.ensure_dirs()
