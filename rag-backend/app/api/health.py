"""GET /health — liveness of this service plus reachability of the vLLM backend."""

from fastapi import APIRouter

from ..configs.config import settings
from ..services.llm_client import check_llm

router = APIRouter(tags=["health"])


@router.get("/health")
def health() -> dict:
    llm = check_llm()
    return {
        "status": "ok",
        "service": "rag-backend",
        "llm": {
            "reachable": llm["reachable"],
            "base_url": settings.llm_v1_url,
            "model": llm["model"],
            "detail": llm["detail"],
        },
    }
