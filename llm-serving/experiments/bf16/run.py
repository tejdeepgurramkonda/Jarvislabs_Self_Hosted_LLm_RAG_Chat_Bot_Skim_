"""BF16 baseline via vLLM (full-precision, bfloat16). Run: python -m experiments.bf16.run"""
from experiments._common import run_and_save

if __name__ == "__main__":
    run_and_save(quantization="none", engine="vllm", label="bf16", dtype="bfloat16")
