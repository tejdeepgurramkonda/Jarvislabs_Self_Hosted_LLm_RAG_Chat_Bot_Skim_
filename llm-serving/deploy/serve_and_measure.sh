#!/usr/bin/env bash
# For each variant: start its vLLM server, wait for health, measure streaming TTFT/ITL,
# then tear the server down and wait for the GPU to free before the next variant.
cd /home/llm-serving
PY=/home/vllm-env/bin/python
VLLM=/home/vllm-env/bin/vllm
COMMON="--host 0.0.0.0 --port 8000 --served-model-name qwen --max-model-len 8192 --gpu-memory-utilization 0.90 --download-dir /home/model_cache"

serve_args() {
  case "$1" in
    bf16) echo "Qwen/Qwen2.5-7B-Instruct --dtype bfloat16" ;;
    fp16) echo "Qwen/Qwen2.5-7B-Instruct --dtype float16" ;;
    fp8)  echo "Qwen/Qwen2.5-7B-Instruct --quantization fp8" ;;
    gptq) echo "Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4 --quantization gptq_marlin" ;;
    awq)  echo "Qwen/Qwen2.5-7B-Instruct-AWQ --quantization awq_marlin" ;;
  esac
}

wait_gpu_free() {
  for i in $(seq 1 60); do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    [ "$used" -lt 2000 ] && return 0
    sleep 3
  done
}

for label in "$@"; do
  echo ">>> [$label] starting server at $(date -u +%H:%M:%S)"
  args=$(serve_args "$label")
  setsid $VLLM serve $args $COMMON > /home/serve_${label}.log 2>&1 < /dev/null &
  SPID=$!

  # Wait for health (first run may download the checkpoint).
  ready=0
  for i in $(seq 1 120); do
    code=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/health 2>/dev/null)
    if [ "$code" = "200" ]; then ready=1; break; fi
    # bail early if the server process died
    kill -0 $SPID 2>/dev/null || { echo ">>> [$label] server process exited early"; break; }
    sleep 5
  done

  if [ "$ready" = "1" ]; then
    echo ">>> [$label] healthy; measuring TTFT/ITL at $(date -u +%H:%M:%S)"
    $PY benchmarks/stream_measure.py --label "$label" --repeats 5 >> /home/serve_${label}.log 2>&1
    echo "$?" > /home/ttft_${label}.done
  else
    echo "ERR: server never became healthy" >> /home/serve_${label}.log
    echo "99" > /home/ttft_${label}.done
  fi

  # Tear down the server (whole process group) and wait for VRAM to free.
  kill -TERM -$SPID 2>/dev/null
  sleep 3
  pkill -9 -f "vllm serve" 2>/dev/null
  wait_gpu_free
  echo ">>> [$label] done at $(date -u +%H:%M:%S)"
done

echo "all_done" > /home/ttft_all.done
