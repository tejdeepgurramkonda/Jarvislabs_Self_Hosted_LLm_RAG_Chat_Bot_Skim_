"""Central configuration. Override any value with a .env file or env vars."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        protected_namespaces=(),
        extra="ignore",
    )

    # --- Model + how to load it ---
    model_name: str = "Qwen/Qwen2.5-7B-Instruct"   # base (unquantized) checkpoint
    model_cache_dir: str = "/home/model_cache"      # persistent on JarvisLabs
    engine: str = "transformers"                    # "transformers" | "vllm"
    quantization: str = "none"                      # none | int8 | nf4 | awq | gptq
    dtype: str = "float16"                          # float16 | bfloat16 | float32
    device_map: str = "auto"

    # --- vLLM-specific ---
    max_model_len: int = 8192
    gpu_memory_utilization: float = 0.90

    # --- Generation defaults ---
    max_new_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 20
    repetition_penalty: float = 1.05

    # --- Benchmark defaults ---
    benchmark_warmup: int = 1        # untimed warmup runs (JIT/caches settle)
    benchmark_runs: int = 5          # timed runs per prompt

    # --- Server ---
    host: str = "0.0.0.0"
    port: int = 6006                 # JarvisLabs' auto-exposed public port
    log_level: str = "INFO"


settings = Settings()
