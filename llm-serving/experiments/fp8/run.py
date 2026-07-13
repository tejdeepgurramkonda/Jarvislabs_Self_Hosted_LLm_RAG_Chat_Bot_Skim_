"""FP8 via vLLM online dynamic quantization on the BASE checkpoint.
Run: python -m experiments.fp8.run"""
from experiments._common import run_and_save

if __name__ == "__main__":
    run_and_save(quantization="fp8", engine="vllm", label="fp8")
