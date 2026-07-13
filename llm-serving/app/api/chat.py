"""POST /generate and POST /chat — with optional token streaming."""

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from ..schemas.request import ChatRequest, GenerateRequest
from ..schemas.response import GenerationResponse

router = APIRouter()


def _engine_or_503(request: Request):
    engine = getattr(request.app.state, "engine", None)
    if engine is None:
        raise HTTPException(status_code=503, detail="Model still loading, try again shortly.")
    return engine


def _record(request: Request, result: dict) -> None:
    """Update the rolling metrics counters used by /metrics."""
    m = request.app.state.metrics
    m["requests_served"] += 1
    if result.get("ttft_s") is not None:
        m["ttft_sum"] += result["ttft_s"]
        m["ttft_count"] += 1
    m["tps_sum"] += result.get("tokens_per_second", 0.0)


@router.post("/generate", response_model=GenerationResponse)
def generate(req: GenerateRequest, request: Request):
    engine = _engine_or_503(request)
    messages = [{"role": "user", "content": req.prompt}]
    if req.stream:
        return StreamingResponse(
            (piece for piece in engine.stream(messages, req.params)),
            media_type="text/plain",
        )
    try:
        result = engine.generate(messages, req.params)
        _record(request, result)
        return GenerationResponse(**result)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/chat", response_model=GenerationResponse)
def chat(req: ChatRequest, request: Request):
    engine = _engine_or_503(request)
    messages = [m.model_dump() for m in req.messages]
    if req.stream:
        return StreamingResponse(
            (piece for piece in engine.stream(messages, req.params)),
            media_type="text/plain",
        )
    try:
        result = engine.generate(messages, req.params)
        _record(request, result)
        return GenerationResponse(**result)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc))
