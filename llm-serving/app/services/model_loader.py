"""
Low-level model loaders.

Two engines are supported:
  - transformers: reliable for fp16/bf16 and int8/nf4 (bitsandbytes).
  - vllm:         the fast serving path; also the most reliable way to run the
                  official AWQ / GPTQ checkpoints (auto-selects the Marlin kernel).

`resolve_model_name` maps a quantization choice to the right HuggingFace
checkpoint, so you don't have to remember the exact repo names.
"""

import torch
from transformers import AutoModelForCausalLM

_DTYPES = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}

# Official pre-quantized Qwen checkpoints.
QUANT_CHECKPOINTS = {
    "awq": "Qwen/Qwen2.5-7B-Instruct-AWQ",
    "gptq": "Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4",
}


def resolve_model_name(base_model: str, quantization: str) -> str:
    """awq/gptq have dedicated checkpoints; everything else uses the base model."""
    return QUANT_CHECKPOINTS.get(quantization, base_model)


def load_transformers_model(
    model_name: str,
    quantization: str,
    dtype: str,
    cache_dir: str,
    device_map: str,
):
    kwargs = {
        "cache_dir": cache_dir,
        "device_map": device_map,
        "dtype": _DTYPES[dtype],
    }

    # bitsandbytes quantization (loaded on the full-precision base checkpoint).
    if quantization in ("int8", "nf4"):
        from transformers import BitsAndBytesConfig

        if quantization == "int8":
            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        else:  # nf4 = 4-bit
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=_DTYPES[dtype],
            )
        kwargs.pop("dtype", None)  # bitsandbytes controls storage dtype

    # awq / gptq: the pre-quantized checkpoint carries its own quant config;
    # transformers loads it automatically IF the kernels are installed
    # (autoawq / gptqmodel). For these, the vLLM engine is usually smoother.
    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    model.eval()
    return model


def load_vllm_llm(
    model_name: str,
    quantization: str,
    max_model_len: int,
    gpu_memory_utilization: float,
    download_dir: str,
    dtype: str = "auto",
):
    from vllm import LLM  # imported lazily so the transformers path doesn't need vLLM

    kwargs = {
        "model": model_name,
        "max_model_len": max_model_len,
        "gpu_memory_utilization": gpu_memory_utilization,
        "download_dir": download_dir,
        # "auto" lets vLLM pick the checkpoint's native dtype; pass "bfloat16"/"float16"
        # explicitly to distinguish the bf16 vs fp16 baselines on the same weights.
        "dtype": dtype,
    }
    # Force the Marlin kernels for awq/gptq (fastest on the L4's Ada cores);
    # fp8 uses vLLM's online dynamic quant on the BASE checkpoint (no special repo).
    if quantization == "awq":
        kwargs["quantization"] = "awq_marlin"
    elif quantization == "gptq":
        kwargs["quantization"] = "gptq_marlin"
    elif quantization == "fp8":
        kwargs["quantization"] = "fp8"

    return LLM(**kwargs)
