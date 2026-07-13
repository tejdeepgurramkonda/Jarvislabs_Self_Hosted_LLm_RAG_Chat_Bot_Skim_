"""GET /health — liveness + which model/engine is loaded."""

from fastapi import APIRouter, Request

from ..schemas.response import HealthResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    engine = getattr(request.app.state, "engine", None)
    return HealthResponse(
        status="ok" if engine else "loading",
        model_name=getattr(engine, "model_name", "unknown"),
        engine=getattr(engine, "name", "unknown"),
        quantization=getattr(engine, "quantization", "unknown"),
        model_loaded=engine is not None,
    )
