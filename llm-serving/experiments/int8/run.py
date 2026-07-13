"""INT8 (bitsandbytes) via transformers. Run: python -m experiments.int8.run"""
from experiments._common import run_and_save

if __name__ == "__main__":
    run_and_save(quantization="int8", engine="transformers", label="int8")
