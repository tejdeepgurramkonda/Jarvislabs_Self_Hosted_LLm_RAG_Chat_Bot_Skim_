"""Outgoing response shapes, including per-request performance telemetry."""

from typing import Optional

from pydantic import BaseModel


class GPUMemory(BaseModel):
    cuda_available: bool
    device: Optional[str] = None
    allocated_gb: Optional[float] = None
    reserved_gb: Optional[float] = None
    total_gb: Optional[float] = None


class GenerationResponse(BaseModel):
    text: str
    prompt_tokens: int
    completion_tokens: int
    ttft_s: Optional[float] = None        # time to first token
    total_time_s: float
    tokens_per_second: float


class HealthResponse(BaseModel):
    status: str
    model_name: str
    engine: str
    quantization: str
    model_loaded: bool


class MetricsResponse(BaseModel):
    model_name: str
    engine: str
    quantization: str
    load_time_s: Optional[float] = None
    requests_served: int
    avg_ttft_s: Optional[float] = None
    avg_tokens_per_second: Optional[float] = None
    gpu: GPUMemory
