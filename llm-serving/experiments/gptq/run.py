"""GPTQ (4-bit Int4) via vLLM. Run: python -m experiments.gptq.run"""
from experiments._common import run_and_save

if __name__ == "__main__":
    run_and_save(quantization="gptq", engine="vllm", label="gptq")
