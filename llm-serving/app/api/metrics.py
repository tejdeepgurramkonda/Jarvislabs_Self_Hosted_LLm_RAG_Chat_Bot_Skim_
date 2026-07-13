"""GET /metrics — live GPU usage + rolling performance counters."""

from fastapi import APIRouter, Request

from ..schemas.response import GPUMemory, MetricsResponse

router = APIRouter()


@router.get("/metrics", response_model=MetricsResponse)
def metrics(request: Request) -> MetricsResponse:
    engine = getattr(request.app.state, "engine", None)
    m = request.app.state.metrics

    avg_ttft = (m["ttft_sum"] / m["ttft_count"]) if m["ttft_count"] else None
    avg_tps = (m["tps_sum"] / m["requests_served"]) if m["requests_served"] else None

    return MetricsResponse(
        model_name=getattr(engine, "model_name", "unknown"),
        engine=getattr(engine, "name", "unknown"),
        quantization=getattr(engine, "quantization", "unknown"),
        load_time_s=getattr(engine, "load_time", None),
        requests_served=m["requests_served"],
        avg_ttft_s=round(avg_ttft, 3) if avg_ttft is not None else None,
        avg_tokens_per_second=round(avg_tps, 2) if avg_tps is not None else None,
        gpu=GPUMemory(**engine.gpu_memory()) if engine else GPUMemory(cuda_available=False),
    )
