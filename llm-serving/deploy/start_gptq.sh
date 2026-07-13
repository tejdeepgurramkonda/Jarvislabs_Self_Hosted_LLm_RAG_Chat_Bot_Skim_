#!/usr/bin/env bash
# Start the GPTQ INT4 vLLM OpenAI server persistently on port 8000.
# Reads the API key from /home/.vllm_key. Logs to /home/serve_gptq.log.
KEY=$(cat /home/.vllm_key)
cd /home/llm-serving
setsid /home/vllm-env/bin/vllm serve Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4 \
  --host 0.0.0.0 --port 8000 --served-model-name qwen \
  --quantization gptq_marlin \
  --max-model-len 8192 --gpu-memory-utilization 0.90 \
  --download-dir /home/model_cache \
  --enable-prefix-caching \
  --api-key "$KEY" \
  > /home/serve_gptq.log 2>&1 < /dev/null &
echo "server launched pid $!"
