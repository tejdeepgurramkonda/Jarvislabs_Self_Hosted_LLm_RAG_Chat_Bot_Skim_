"""
FastAPI application entry point.

Run it:
    python -m app.main
or:
    uvicorn app.main:app --host 0.0.0.0 --port 6006

The engine (chosen by your settings/.env) is loaded ONCE during startup and
shared by every request via app.state.
"""

from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI

from .api import chat, health, metrics
from .configs.config import settings
from .services.inference import build_engine
from .utils.logger import configure_logging, get_logger

configure_logging()
logger = get_logger("app.main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(
        "Loading engine=%s quantization=%s ...", settings.engine, settings.quantization
    )
    app.state.engine = build_engine(
        engine=settings.engine,
        base_model=settings.model_name,
        quantization=settings.quantization,
        dtype=settings.dtype,
        cache_dir=settings.model_cache_dir,
        device_map=settings.device_map,
        max_model_len=settings.max_model_len,
        gpu_memory_utilization=settings.gpu_memory_utilization,
    )
    app.state.metrics = {
        "requests_served": 0,
        "ttft_sum": 0.0,
        "ttft_count": 0,
        "tps_sum": 0.0,
    }
    logger.info("Engine ready (load_time=%.1fs).", app.state.engine.load_time)
    yield
    logger.info("Shutting down.")


app = FastAPI(title="LLM Serving", version="0.1.0", lifespan=lifespan)
app.include_router(health.router)
app.include_router(chat.router)
app.include_router(metrics.router)


if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level.lower(),
        workers=1,
    )
