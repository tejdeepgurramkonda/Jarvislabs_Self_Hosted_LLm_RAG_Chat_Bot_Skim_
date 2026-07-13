"""Parse a variant's run log for vLLM's reported model-weights VRAM and inject
`weights_gb` into benchmarks/results/<label>.json.

During engine init vLLM prints the model's GPU memory, e.g.
`Model loading took 14.29 GiB memory ...` (older builds: `Model weights take 14.99GiB`).
That is the honest per-variant memory number (nvidia-smi can't distinguish variants
because vLLM pre-reserves the KV cache to gpu_memory_utilization on every run).

Usage: python benchmarks/_inject_weights.py <label> <logfile>
"""
import json
import re
import sys
from pathlib import Path

label, logfile = sys.argv[1], sys.argv[2]
text = Path(logfile).read_text(errors="ignore")
# Match both "Model loading took 14.29 GiB memory" and "Model weights take 14.99GiB".
matches = re.findall(
    r"model (?:loading took|weights take)\s*([\d.]+)\s*GiB", text, flags=re.IGNORECASE
)

result_path = Path("benchmarks/results") / f"{label}.json"
data = json.loads(result_path.read_text())
if matches:
    data["weights_gb"] = round(float(matches[-1]), 2)
else:
    data["weights_gb"] = None
    print(f"WARNING: no 'Model weights take' line found in {logfile}")
result_path.write_text(json.dumps(data, indent=2))
print(f"weights_gb={data.get('weights_gb')} -> {result_path}")
