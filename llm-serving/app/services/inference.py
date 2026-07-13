"""
Inference engines behind ONE common interface, so the API and the benchmark
harness don't care which backend is running.

Every engine exposes:
    .stream(messages, params)   -> yields text pieces (one per token-ish)
    .generate(messages, params) -> full result dict with timing
    .gpu_memory()               -> live VRAM report
    .load_time                  -> seconds it took to load

`build_engine(...)` picks the right one from your settings.
"""

import threading
import time

import torch
from transformers import TextIteratorStreamer

from ..utils.benchmark import gpu_memory_report
from .model_loader import (
    load_transformers_model,
    load_vllm_llm,
    resolve_model_name,
)
from .tokenizer import load_tokenizer, render_chat


class TransformersEngine:
    name = "transformers"

    def __init__(self, model_name, quantization, dtype, cache_dir, device_map):
        t0 = time.perf_counter()
        self.model_name = model_name
        self.quantization = quantization
        self.tokenizer = load_tokenizer(model_name, cache_dir)
        self.model = load_transformers_model(
            model_name, quantization, dtype, cache_dir, device_map
        )
        self.load_time = round(time.perf_counter() - t0, 2)

    def _prepare(self, messages, params):
        prompt = render_chat(self.tokenizer, messages)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        gen_kwargs = {
            "max_new_tokens": params.max_new_tokens,
            "repetition_penalty": params.repetition_penalty,
            "pad_token_id": self.tokenizer.eos_token_id,
        }
        if params.temperature and params.temperature > 0:
            gen_kwargs.update(
                do_sample=True,
                temperature=params.temperature,
                top_p=params.top_p,
                top_k=params.top_k,
            )
        else:
            gen_kwargs["do_sample"] = False
        return inputs, gen_kwargs

    @torch.inference_mode()
    def stream(self, messages, params):
        inputs, gen_kwargs = self._prepare(messages, params)
        streamer = TextIteratorStreamer(
            self.tokenizer, skip_prompt=True, skip_special_tokens=True
        )
        # model.generate blocks, so run it in a thread and read tokens as they land.
        thread = threading.Thread(
            target=self.model.generate, kwargs={**inputs, **gen_kwargs, "streamer": streamer}
        )
        thread.start()
        for text in streamer:
            if text:
                yield text
        thread.join()

    def generate(self, messages, params) -> dict:
        prompt = render_chat(self.tokenizer, messages)
        prompt_tokens = self.tokenizer(prompt, return_tensors="pt")["input_ids"].shape[1]
        start = time.perf_counter()
        first = None
        pieces = []
        n = 0
        for piece in self.stream(messages, params):
            if first is None:
                first = time.perf_counter()
            pieces.append(piece)
            n += 1
        end = time.perf_counter()
        elapsed = end - start
        return {
            "text": "".join(pieces).strip(),
            "prompt_tokens": int(prompt_tokens),
            "completion_tokens": n,
            "ttft_s": round(first - start, 3) if first else None,
            "total_time_s": round(elapsed, 3),
            "tokens_per_second": round(n / elapsed, 2) if elapsed > 0 else 0.0,
        }

    def gpu_memory(self):
        return gpu_memory_report()


class VLLMEngine:
    name = "vllm"

    def __init__(self, model_name, quantization, cache_dir, max_model_len, gpu_memory_utilization, dtype="auto"):
        t0 = time.perf_counter()
        self.model_name = model_name
        self.quantization = quantization
        self.tokenizer = load_tokenizer(model_name, cache_dir)
        self.llm = load_vllm_llm(
            model_name, quantization, max_model_len, gpu_memory_utilization, cache_dir, dtype
        )
        self.load_time = round(time.perf_counter() - t0, 2)

    def _sampling(self, params):
        from vllm import SamplingParams

        return SamplingParams(
            max_tokens=params.max_new_tokens,
            temperature=params.temperature,
            top_p=params.top_p,
            top_k=params.top_k if params.top_k > 0 else -1,
            repetition_penalty=params.repetition_penalty,
        )

    def generate(self, messages, params) -> dict:
        prompt = render_chat(self.tokenizer, messages)
        prompt_tokens = len(self.tokenizer(prompt)["input_ids"])
        start = time.perf_counter()
        out = self.llm.generate([prompt], self._sampling(params))
        elapsed = time.perf_counter() - start
        text = out[0].outputs[0].text
        completion_tokens = len(out[0].outputs[0].token_ids)
        return {
            "text": text.strip(),
            "prompt_tokens": int(prompt_tokens),
            "completion_tokens": int(completion_tokens),
            "ttft_s": None,  # offline batch API has no per-token stream
            "total_time_s": round(elapsed, 3),
            "tokens_per_second": round(completion_tokens / elapsed, 2) if elapsed > 0 else 0.0,
        }

    def stream(self, messages, params):
        # The offline vLLM API returns the full text at once. For TRUE streaming
        # TTFT/ITL under vLLM, benchmark the vLLM *server* with benchmarks/load_test.py.
        yield self.generate(messages, params)["text"]

    def gpu_memory(self):
        return gpu_memory_report()


def build_engine(
    engine: str,
    base_model: str,
    quantization: str,
    dtype: str = "float16",
    cache_dir: str = "/home/model_cache",
    device_map: str = "auto",
    max_model_len: int = 8192,
    gpu_memory_utilization: float = 0.90,
):
    model_name = resolve_model_name(base_model, quantization)
    if engine == "vllm":
        return VLLMEngine(
            model_name, quantization, cache_dir, max_model_len, gpu_memory_utilization, dtype
        )
    return TransformersEngine(model_name, quantization, dtype, cache_dir, device_map)
