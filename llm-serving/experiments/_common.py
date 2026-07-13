"""
Shared experiment harness.

Each experiments/<method>/run.py is a 3-line wrapper that calls run_and_save
with its settings. This module does the real work:

  1. build the engine for that quantization
  2. warm up (untimed), then run N timed generations per prompt
  3. record E2E latency / throughput + VRAM + load time (metrics)
  4. generate answers to a fixed RAG prompt set and save them for quality review
  5. save a JSON to benchmarks/results/<label>.json and print a summary

All variants in this round run on the vLLM engine, so timed metrics come from
eng.generate() (correct completion-token counts). Per-token TTFT/ITL are not
available from the offline vLLM API — those come from the vLLM *server* in the
later serving phase.

Run from the repo root, e.g.:
    python -m experiments.awq.run
"""

from pathlib import Path

from app.configs.config import settings
from app.schemas.request import GenerationParams
from app.services.inference import build_engine
from app.utils.benchmark import (
    format_table,
    gpu_memory_report,
    save_result,
    summarize_generations,
)

# A small, fixed prompt set so every method is measured on the same work (timing).
PROMPTS = [
    "Explain what a GPU is in two sentences.",
    "Write a short paragraph about the benefits of quantization.",
    "List five practical uses for a 7B language model.",
    "Summarize how HTTP works for a beginner.",
]

# RAG-style prompts for QUALITY comparison (not timing). Each is a context passage
# plus a question. The system prompt pins the model to the provided context so we can
# see faithfulness vs hallucination across quantizations. Prompt 3 is deliberately
# UN-answerable from its context — a good model should say so rather than invent an answer.
_RAG_SYSTEM = (
    "You are a helpful assistant. Answer the question using ONLY the information in the "
    "provided context. If the answer is not in the context, say you don't have enough "
    "information. Be concise."
)
RAG_PROMPTS = [
    (
        "grounded_single_fact",
        [
            {"role": "system", "content": _RAG_SYSTEM},
            {
                "role": "user",
                "content": (
                    "Context:\n"
                    "The Apollo 11 mission launched on July 16, 1969. Neil Armstrong and "
                    "Buzz Aldrin landed on the Moon on July 20, 1969, while Michael Collins "
                    "remained in lunar orbit aboard the command module Columbia.\n\n"
                    "Question: Who stayed in orbit while the others landed on the Moon?"
                ),
            },
        ],
    ),
    (
        "grounded_synthesis",
        [
            {"role": "system", "content": _RAG_SYSTEM},
            {
                "role": "user",
                "content": (
                    "Context:\n"
                    "Redis is an in-memory key-value store; reads and writes typically complete "
                    "in well under a millisecond. PostgreSQL is a disk-based relational database "
                    "that offers strong durability and rich SQL queries but higher read latency. "
                    "A common pattern is to use PostgreSQL as the source of truth and Redis as a "
                    "cache in front of it.\n\n"
                    "Question: Based on the context, why would a team put Redis in front of "
                    "PostgreSQL, and what is the trade-off?"
                ),
            },
        ],
    ),
    (
        "unanswerable_refusal",
        [
            {"role": "system", "content": _RAG_SYSTEM},
            {
                "role": "user",
                "content": (
                    "Context:\n"
                    "The company's Paris office opened in 2015 and focuses on European sales. "
                    "It has grown to 40 employees as of 2023.\n\n"
                    "Question: What was the company's total revenue in 2023?"
                ),
            },
        ],
    ),
]

# Qwen2.5 recommended sampling for "best response generation", applied uniformly so
# quality differences reflect the quantization, not the sampling settings.
QWEN_PARAMS = GenerationParams(
    max_new_tokens=settings.max_new_tokens,
    temperature=0.7,
    top_p=0.8,
    top_k=20,
    repetition_penalty=1.05,
)

RESULTS_DIR = Path("benchmarks/results")


def _save_samples(label: str, eng, params) -> str:
    """Generate answers to the RAG prompts and save them for human quality review."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"{label}_samples.txt"
    lines = [f"# RAG quality samples — variant: {label}", ""]
    for name, messages in RAG_PROMPTS:
        result = eng.generate(messages, params)
        user_msg = messages[-1]["content"]
        lines.append("=" * 78)
        lines.append(f"[{name}]")
        lines.append(user_msg)
        lines.append("-" * 78)
        lines.append(result["text"])
        lines.append(
            f"(completion_tokens={result.get('completion_tokens')}, "
            f"total_time_s={result.get('total_time_s')})"
        )
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    return str(path)


def run_and_save(
    quantization: str,
    engine: str,
    label: str | None = None,
    dtype: str | None = None,
) -> dict:
    label = label or quantization
    params = QWEN_PARAMS

    eng = build_engine(
        engine=engine,
        base_model=settings.model_name,
        quantization=quantization,
        dtype=dtype or settings.dtype,
        cache_dir=settings.model_cache_dir,
        device_map=settings.device_map,
        max_model_len=settings.max_model_len,
        gpu_memory_utilization=settings.gpu_memory_utilization,
    )

    # Warmup (not timed): lets CUDA/JIT caches settle so numbers are stable.
    for _ in range(settings.benchmark_warmup):
        eng.generate([{"role": "user", "content": PROMPTS[0]}], params)

    # Timed runs — use generate() so vLLM reports correct completion-token counts.
    gens = []
    for prompt in PROMPTS:
        for _ in range(settings.benchmark_runs):
            messages = [{"role": "user", "content": prompt}]
            gens.append(eng.generate(messages, params))

    vram = gpu_memory_report()
    result = summarize_generations(
        label,
        gens,
        extra={
            "engine": engine,
            "quantization": quantization,
            "dtype": dtype or settings.dtype,
            "load_time_s": eng.load_time,
            "vram_allocated_gb": vram.get("allocated_gb"),
            "vram_reserved_gb": vram.get("reserved_gb"),
        },
    )

    # Quality: save actual RAG answers for side-by-side review.
    samples_path = _save_samples(label, eng, params)

    path = save_result(result)
    print("\n" + format_table([result]))
    print(f"\nSaved metrics -> {path}")
    print(f"Saved RAG samples -> {samples_path}")
    return result
