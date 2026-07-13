#!/usr/bin/env bash
# Run the remaining variants sequentially (one model on the GPU at a time).
# Resilient: a failure in one variant is recorded but does not abort the others.
cd /home/llm-serving
PY=/home/vllm-env/bin/python
RES=benchmarks/results

run_variant() {
  label="$1"; module="$2"
  echo ">>> starting $label at $(date -u +%H:%M:%S)"
  $PY -m "$module" > "$RES/$label.log" 2>&1
  ec=$?
  $PY benchmarks/_inject_weights.py "$label" "$RES/$label.log" >> "$RES/$label.log" 2>&1 || true
  echo "$ec" > "/home/run_${label}.done"
  echo ">>> finished $label exit=$ec at $(date -u +%H:%M:%S)"
}

run_variant fp16 experiments.fp16.run
run_variant fp8  experiments.fp8.run
run_variant gptq experiments.gptq.run
run_variant awq  experiments.awq.run

echo "all_done" > /home/run_rest.done
