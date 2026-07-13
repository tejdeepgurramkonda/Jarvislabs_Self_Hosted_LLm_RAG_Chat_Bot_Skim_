"""
vLLM baseline (unquantized fp16) via vLLM, to isolate the engine's effect
vs the transformers fp16 baseline. Run: python -m experiments.vllm.run
"""
from experiments._common import run_and_save

if __name__ == "__main__":
    run_and_save(quantization="none", engine="vllm", label="vllm_fp16")
