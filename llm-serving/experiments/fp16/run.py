"""FP16 baseline via vLLM (full-precision, float16). Run: python -m experiments.fp16.run"""
from experiments._common import run_and_save

if __name__ == "__main__":
    run_and_save(quantization="none", engine="vllm", label="fp16", dtype="float16")
