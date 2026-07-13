#!/usr/bin/env bash
# Sets up BOTH environments in parallel using uv, with per-env logs and done-markers.
set -u
cd /home/llm-serving

install_base() {
  {
    echo "[base] creating venv (inherits system torch)"
    uv venv --system-site-packages /home/llm-serving/.venv || exit 11
    echo "[base] installing requirements.txt"
    uv pip install --python /home/llm-serving/.venv/bin/python -r /home/llm-serving/requirements.txt || exit 12
    echo "[base] verifying imports"
    /home/llm-serving/.venv/bin/python - <<'PY' || exit 13
import torch, transformers, fastapi, bitsandbytes
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
print("transformers", transformers.__version__)
print("BASE_OK")
PY
  } >/home/setup_base.log 2>&1
  echo $? >/home/setup_base.done
}

install_vllm() {
  {
    echo "[vllm] creating fresh venv"
    uv venv /home/vllm-env || exit 21
    echo "[vllm] installing vllm openai httpx tabulate"
    uv pip install --python /home/vllm-env/bin/python vllm openai httpx tabulate || exit 22
    echo "[vllm] verifying imports"
    /home/vllm-env/bin/python - <<'PY' || exit 23
import vllm, torch
print("vllm", vllm.__version__)
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
print("VLLM_OK")
PY
  } >/home/setup_vllm.log 2>&1
  echo $? >/home/setup_vllm.done
}

rm -f /home/setup_base.done /home/setup_vllm.done /home/setup_all.done
install_base &
BPID=$!
install_vllm &
VPID=$!
wait $BPID
wait $VPID
echo "done" >/home/setup_all.done
