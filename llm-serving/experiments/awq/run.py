"""AWQ (4-bit) via vLLM. Run: python -m experiments.awq.run"""
from experiments._common import run_and_save

if __name__ == "__main__":
    # AWQ runs most reliably (and fast, via Marlin) through vLLM.
    run_and_save(quantization="awq", engine="vllm", label="awq")
