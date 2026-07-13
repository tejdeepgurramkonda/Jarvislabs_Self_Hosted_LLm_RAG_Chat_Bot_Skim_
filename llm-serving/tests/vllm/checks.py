"""
The actual test logic — one `check_*` function per test ID in TEST_PLAN.md.

Each check performs its request(s), compares the actual response to the IDEAL, and
returns a `Result`. Every check catches its own exceptions and records them, so a
single failing test never stops the run (the runner and the pytest suite both rely
on this). Generations are kept small to limit GPU cost.
"""

from __future__ import annotations

import concurrent.futures as cf
import json
import time

import httpx
from openai import APIStatusError, OpenAI

from helpers import (
    ERROR,
    FAIL,
    NA,
    PARTIAL,
    PASS,
    Context,
    Result,
    make_client,
    raw_client,
    timed_stream,
    trim,
)


def _content(resp) -> str:
    return (resp.choices[0].message.content or "").strip()


# --------------------------------------------------------------------------- #
# 1. Reachability & health
# --------------------------------------------------------------------------- #
def check_H1(ctx: Context) -> Result:
    r = Result(
        "H1", "Reachability", "GET /v1/models responds and lists the served model",
        request=f"GET {ctx.cfg.base_url}/models",
        ideal=f"HTTP 200; data[] contains id '{ctx.cfg.model}' (or 'qwen'). Capture max_model_len.",
    )
    try:
        with raw_client(ctx.cfg) as c:
            resp = c.get("/models")
        body = resp.json()
        ids = [m.get("id") for m in body.get("data", [])]
        # capture max_model_len for later adaptive tests
        for m in body.get("data", []):
            if m.get("max_model_len"):
                ctx.max_model_len = int(m["max_model_len"])
        r.actual = f"HTTP {resp.status_code}; ids={ids}; max_model_len={ctx.max_model_len}"
        ok = resp.status_code == 200 and any(
            ctx.cfg.model == i or "qwen" in (i or "").lower() for i in ids
        )
        r.status = PASS if ok else FAIL
        if not ok:
            r.note = "Model id not found or non-200."
    except Exception as e:
        r.actual, r.status, r.note = f"{type(e).__name__}: {e}", ERROR, "request failed"
    return r


def check_H2(ctx: Context) -> Result:
    r = Result(
        "H2", "Reachability", "GET /health liveness endpoint",
        request=f"GET {ctx.cfg.root_url}/health",
        ideal="HTTP 200 (empty body). 404 => not exposed (PARTIAL, non-fatal).",
    )
    try:
        with httpx.Client(timeout=20) as c:
            resp = c.get(ctx.cfg.root_url + "/health")
        r.actual = f"HTTP {resp.status_code}"
        if resp.status_code == 200:
            r.status = PASS
        elif resp.status_code == 404:
            r.status, r.note = PARTIAL, "/health not exposed by this deployment."
        else:
            r.status, r.note = FAIL, "unexpected status"
    except Exception as e:
        r.actual, r.status, r.note = f"{type(e).__name__}: {e}", ERROR, "request failed"
    return r


# --------------------------------------------------------------------------- #
# 2. Basic correctness
# --------------------------------------------------------------------------- #
def _simple_qa(ctx, rid, desc, prompt, must_contain, ideal, max_tokens=48):
    r = Result(rid, "Correctness", desc,
               request=f"chat: user={prompt!r} (temperature=0, max_tokens={max_tokens})",
               ideal=ideal)
    try:
        resp = ctx.client.chat.completions.create(
            model=ctx.cfg.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0, max_tokens=max_tokens,
        )
        text = _content(resp)
        fr = resp.choices[0].finish_reason
        role = resp.choices[0].message.role
        r.actual = f"role={role} finish={fr} content={text!r}"
        hit = all(m.lower() in text.lower() for m in must_contain)
        r.status = PASS if (text and hit) else FAIL
        if not hit:
            r.note = f"expected to contain {must_contain}"
    except Exception as e:
        r.actual, r.status, r.note = f"{type(e).__name__}: {e}", ERROR, "request failed"
    return r


def check_C1(ctx):
    return _simple_qa(ctx, "C1", "Simple chat completion is coherent/on-topic",
                      "Say hello in one short sentence.", [],
                      "Non-empty greeting; role=assistant; finish_reason=stop.", max_tokens=32)


def check_C2(ctx):
    return _simple_qa(ctx, "C2", "Known factual question",
                      "What is the capital of France? Answer in one word.", ["Paris"],
                      "Answer contains 'Paris'.")


def check_C3(ctx):
    return _simple_qa(ctx, "C3", "Simple arithmetic reasoning",
                      "What is 17 + 25? Reply with just the number.", ["42"],
                      "Answer contains '42'.")


# --------------------------------------------------------------------------- #
# 3. Parameters
# --------------------------------------------------------------------------- #
def check_P1(ctx: Context) -> Result:
    r = Result("P1", "Parameters", "max_tokens caps output length",
               request="chat: 'Write a long paragraph about the ocean.' max_tokens=16",
               ideal="completion_tokens <= 16 AND finish_reason == 'length'.")
    try:
        resp = ctx.client.chat.completions.create(
            model=ctx.cfg.model,
            messages=[{"role": "user", "content": "Write a long paragraph about the ocean."}],
            max_tokens=16, temperature=0,
        )
        ct = resp.usage.completion_tokens
        fr = resp.choices[0].finish_reason
        r.actual = f"completion_tokens={ct} finish_reason={fr} text={_content(resp)!r}"
        ok = ct <= 16 and fr == "length"
        r.status = PASS if ok else (PARTIAL if ct <= 16 else FAIL)
        if not ok:
            r.note = "expected <=16 tokens and finish_reason=length"
    except Exception as e:
        r.actual, r.status, r.note = f"{type(e).__name__}: {e}", ERROR, "request failed"
    return r


def check_P2a(ctx: Context) -> Result:
    r = Result("P2a", "Parameters", "temperature=0 is (near-)deterministic across repeats",
               request="same prompt x3, temperature=0, max_tokens=48",
               ideal="3 outputs identical (or >= 2/3 identical).")
    prompt = "In two sentences, explain what a rainbow is."
    try:
        outs = []
        for _ in range(3):
            resp = ctx.client.chat.completions.create(
                model=ctx.cfg.model, messages=[{"role": "user", "content": prompt}],
                temperature=0, max_tokens=48, seed=1234,
            )
            outs.append(_content(resp))
        uniq = set(outs)
        r.actual = f"{len(uniq)} distinct of 3. sample={outs[0]!r}"
        if len(uniq) == 1:
            r.status = PASS
        elif len(outs) - len(uniq) >= 1:  # at least two identical
            r.status, r.note = PARTIAL, "not fully identical (sampler jitter possible)"
        else:
            r.status, r.note = FAIL, "all three differ under temperature=0"
    except Exception as e:
        r.actual, r.status, r.note = f"{type(e).__name__}: {e}", ERROR, "request failed"
    return r


def check_P2b(ctx: Context) -> Result:
    r = Result("P2b", "Parameters", "high temperature produces varied output",
               request="same prompt x3, temperature=1.3, top_p=0.95, max_tokens=48",
               ideal=">= 2 distinct outputs.")
    prompt = "Write one imaginative sentence about a city."
    try:
        outs = []
        for _ in range(3):
            resp = ctx.client.chat.completions.create(
                model=ctx.cfg.model, messages=[{"role": "user", "content": prompt}],
                temperature=1.3, top_p=0.95, max_tokens=48,
            )
            outs.append(_content(resp))
        uniq = set(outs)
        r.actual = f"{len(uniq)} distinct of 3. samples={[trim(o,80) for o in outs]}"
        r.status = PASS if len(uniq) >= 2 else FAIL
        if len(uniq) < 2:
            r.note = "no variation at high temperature"
    except Exception as e:
        r.actual, r.status, r.note = f"{type(e).__name__}: {e}", ERROR, "request failed"
    return r


def check_P3(ctx: Context) -> Result:
    r = Result("P3", "Parameters", "stop sequence honored + finish_reason correct",
               request="'Count from 1 to 9 separated by commas.' stop=['5']",
               ideal="Output stops before/at '5'; finish_reason == 'stop'.")
    try:
        resp = ctx.client.chat.completions.create(
            model=ctx.cfg.model,
            messages=[{"role": "user", "content": "Count from 1 to 9 separated by commas."}],
            stop=["5"], max_tokens=64, temperature=0,
        )
        text = _content(resp)
        fr = resp.choices[0].finish_reason
        r.actual = f"finish_reason={fr} text={text!r}"
        stopped = "5" not in text and fr == "stop"
        r.status = PASS if stopped else (PARTIAL if fr == "stop" else FAIL)
        if not stopped:
            r.note = "expected text truncated before '5' with finish_reason=stop"
    except Exception as e:
        r.actual, r.status, r.note = f"{type(e).__name__}: {e}", ERROR, "request failed"
    return r


# --------------------------------------------------------------------------- #
# 4. Streaming
# --------------------------------------------------------------------------- #
def check_S1(ctx: Context) -> Result:
    r = Result("S1", "Streaming", "stream=True yields tokens incrementally",
               request="chat stream=True: 'List three fruits, one per line.'",
               ideal=">= 2 content chunks (not one final blob); coherent reassembled text.")
    try:
        stats = timed_stream(
            ctx.client, model=ctx.cfg.model,
            messages=[{"role": "user", "content": "List three fruits, one per line."}],
            max_tokens=48, temperature=0,
        )
        r.actual = f"n_chunks={stats.n_chunks} ttft={stats.ttft:.3f}s text={stats.text!r}"
        r.status = PASS if stats.n_chunks >= 2 and stats.text.strip() else FAIL
        if stats.n_chunks < 2:
            r.note = "arrived as a single blob"
    except Exception as e:
        r.actual, r.status, r.note = f"{type(e).__name__}: {e}", ERROR, "request failed"
    return r


def check_S2(ctx: Context) -> Result:
    r = Result("S2", "Streaming", "measure TTFT and tokens/sec",
               request="chat stream=True, ~120-token answer; time first chunk & total",
               ideal="TTFT recorded (ideal < ~2s) and tokens/sec recorded; streams incrementally.")
    try:
        stats = timed_stream(
            ctx.client, model=ctx.cfg.model,
            messages=[{"role": "user",
                       "content": "Write a short paragraph (about 100 words) about the sea."}],
            max_tokens=140, temperature=0.7,
        )
        ctx.perf["S2"] = {
            "ttft_s": round(stats.ttft, 3),
            "total_s": round(stats.total, 3),
            "completion_tokens": stats.completion_tokens,
            "tokens_per_sec": round(stats.tokens_per_sec, 1) if stats.tokens_per_sec else None,
            "n_chunks": stats.n_chunks,
        }
        tps = round(stats.tokens_per_sec, 1) if stats.tokens_per_sec else "n/a"
        r.actual = (f"TTFT={stats.ttft:.3f}s total={stats.total:.3f}s "
                    f"tokens={stats.completion_tokens} tok/s={tps} chunks={stats.n_chunks}")
        r.status = PASS if stats.n_chunks >= 2 and stats.tokens_per_sec else PARTIAL
        if stats.ttft > 2.0:
            r.note = f"TTFT {stats.ttft:.2f}s above ~2s ideal (still streaming fine)"
    except Exception as e:
        r.actual, r.status, r.note = f"{type(e).__name__}: {e}", ERROR, "request failed"
    return r


# --------------------------------------------------------------------------- #
# 5. Chat behavior
# --------------------------------------------------------------------------- #
def check_B1(ctx: Context) -> Result:
    r = Result("B1", "Chat", "multi-turn conversation is respected",
               request="system + user 'My name is Ada.' + assistant + user 'What is my name?'",
               ideal="Answer contains 'Ada'.")
    try:
        resp = ctx.client.chat.completions.create(
            model=ctx.cfg.model, temperature=0, max_tokens=32,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "My name is Ada."},
                {"role": "assistant", "content": "Nice to meet you, Ada."},
                {"role": "user", "content": "What is my name?"},
            ],
        )
        text = _content(resp)
        r.actual = f"content={text!r}"
        r.status = PASS if "ada" in text.lower() else FAIL
        if "ada" not in text.lower():
            r.note = "did not recall the name from earlier turns"
    except Exception as e:
        r.actual, r.status, r.note = f"{type(e).__name__}: {e}", ERROR, "request failed"
    return r


def check_B2(ctx: Context) -> Result:
    r = Result("B2", "Chat", "system prompt changes the style of the answer",
               request="same user 'Describe the sea.' with system A (terse pirate) vs B (formal oceanographer)",
               ideal="Two outputs clearly differ in style/tone.")
    try:
        def ask(system):
            resp = ctx.client.chat.completions.create(
                model=ctx.cfg.model, temperature=0.5, max_tokens=60,
                messages=[{"role": "system", "content": system},
                          {"role": "user", "content": "Describe the sea."}],
            )
            return _content(resp)
        a = ask("You are a terse pirate. Answer in pirate slang, very briefly.")
        b = ask("You are a formal oceanographer. Answer in precise scientific prose.")
        r.actual = f"A(pirate)={trim(a,180)!r}\nB(scientist)={trim(b,180)!r}"
        r.status = PASS if a.strip() and b.strip() and a.strip() != b.strip() else FAIL
        if a.strip() == b.strip():
            r.note = "system prompt had no effect"
    except Exception as e:
        r.actual, r.status, r.note = f"{type(e).__name__}: {e}", ERROR, "request failed"
    return r


# --------------------------------------------------------------------------- #
# 6. Robustness / edge cases
# --------------------------------------------------------------------------- #
def check_E1(ctx: Context) -> Result:
    r = Result("E1", "Robustness", "empty prompt handled gracefully",
               request="chat: user content='' (max_tokens=16)",
               ideal="Valid completion OR clean 4xx — no hang, no 500.")
    try:
        resp = ctx.client.chat.completions.create(
            model=ctx.cfg.model, messages=[{"role": "user", "content": ""}],
            max_tokens=16, temperature=0,
        )
        r.actual = f"HTTP 200; content={_content(resp)!r}"
        r.status = PASS
        r.note = "server accepted empty prompt and returned a completion"
    except APIStatusError as e:
        r.actual = f"HTTP {e.status_code}: {trim(str(e), 200)}"
        r.status = PASS if 400 <= e.status_code < 500 else FAIL
        r.note = "clean 4xx" if r.status == PASS else "unexpected 5xx"
    except Exception as e:
        r.actual, r.status, r.note = f"{type(e).__name__}: {e}", ERROR, "request failed"
    return r


def check_E2(ctx: Context) -> Result:
    limit = ctx.max_model_len or 8192
    # aim near the context limit: ~ (limit - margin) tokens. ~4 chars/token, use a
    # repeated token-ish word so the count is roughly predictable.
    approx_tokens = max(limit - 300, 512)
    filler = ("data " * approx_tokens).strip()
    r = Result("E2", "Robustness", "very long prompt near context limit handled gracefully",
               request=f"chat with ~{approx_tokens} tokens of filler (max_model_len={limit}), max_tokens=16",
               ideal="Either completes OR clear context-length 4xx — no hang/500.")
    try:
        resp = ctx.client.chat.completions.create(
            model=ctx.cfg.model,
            messages=[{"role": "user", "content": "Summarize in one word: " + filler}],
            max_tokens=16, temperature=0,
        )
        r.actual = f"HTTP 200; prompt_tokens={resp.usage.prompt_tokens}; content={_content(resp)!r}"
        r.status = PASS
        r.note = "completed near context limit"
    except APIStatusError as e:
        r.actual = f"HTTP {e.status_code}: {trim(str(e), 240)}"
        r.status = PASS if 400 <= e.status_code < 500 else FAIL
        r.note = "graceful context-length 4xx" if r.status == PASS else "unexpected 5xx"
    except Exception as e:
        r.actual, r.status, r.note = f"{type(e).__name__}: {e}", ERROR, "request failed"
    return r


def check_E3(ctx: Context) -> Result:
    r = Result("E3", "Robustness", "huge max_tokens handled gracefully",
               request="chat: max_tokens=999999",
               ideal="Server clamps OR returns 400 context error; no crash.")
    try:
        resp = ctx.client.chat.completions.create(
            model=ctx.cfg.model, messages=[{"role": "user", "content": "Hi"}],
            max_tokens=999999, temperature=0,
        )
        r.actual = f"HTTP 200; completion_tokens={resp.usage.completion_tokens} (clamped)"
        r.status = PASS
        r.note = "server clamped to context window"
    except APIStatusError as e:
        r.actual = f"HTTP {e.status_code}: {trim(str(e), 240)}"
        r.status = PASS if 400 <= e.status_code < 500 else FAIL
        r.note = "graceful 4xx" if r.status == PASS else "unexpected 5xx"
    except Exception as e:
        r.actual, r.status, r.note = f"{type(e).__name__}: {e}", ERROR, "request failed"
    return r


def check_E4(ctx: Context) -> Result:
    r = Result("E4", "Robustness", "wrong model name returns a proper error",
               request="chat with model='not-a-real-model'",
               ideal="404/400 error naming the unknown model.")
    try:
        ctx.client.chat.completions.create(
            model="not-a-real-model", messages=[{"role": "user", "content": "Hi"}],
            max_tokens=8,
        )
        r.actual = "HTTP 200 (unexpected success)"
        r.status = FAIL
        r.note = "server accepted a nonexistent model"
    except APIStatusError as e:
        r.actual = f"HTTP {e.status_code}: {trim(str(e), 240)}"
        r.status = PASS if e.status_code in (400, 404) else PARTIAL
        if r.status != PASS:
            r.note = "errored but not with 400/404"
    except Exception as e:
        r.actual, r.status, r.note = f"{type(e).__name__}: {e}", ERROR, "request failed"
    return r


def check_E5(ctx: Context) -> Result:
    r = Result("E5", "Robustness", "invalid API key returns 401-style error",
               request="chat with Authorization: Bearer bad-key-123",
               ideal="401 if auth enabled; N/A if the server runs without auth.")
    try:
        bad = make_client(ctx.cfg, api_key="bad-key-123", timeout=30)
        bad.chat.completions.create(
            model=ctx.cfg.model, messages=[{"role": "user", "content": "Hi"}], max_tokens=8,
        )
        r.actual = "HTTP 200 with a bad key"
        r.status = NA
        r.note = "server has no API-key auth enabled; test not applicable."
    except APIStatusError as e:
        r.actual = f"HTTP {e.status_code}: {trim(str(e), 200)}"
        r.status = PASS if e.status_code == 401 else PARTIAL
        if r.status != PASS:
            r.note = "rejected but not with 401"
    except Exception as e:
        r.actual, r.status, r.note = f"{type(e).__name__}: {e}", ERROR, "request failed"
    return r


def check_E6(ctx: Context) -> Result:
    r = Result("E6", "Robustness", "malformed request body returns 4xx (not a hang)",
               request="raw POST /chat/completions with messages='hello' and temperature='hot'",
               ideal="400/422 validation error; no hang.")
    try:
        headers = {"Content-Type": "application/json"}
        if ctx.cfg.api_key and ctx.cfg.api_key != "EMPTY":
            headers["Authorization"] = f"Bearer {ctx.cfg.api_key}"
        with raw_client(ctx.cfg, timeout=30) as c:
            resp = c.post("/chat/completions", headers=headers, content=json.dumps({
                "model": ctx.cfg.model, "messages": "hello", "temperature": "hot",
            }))
        r.actual = f"HTTP {resp.status_code}: {trim(resp.text, 240)}"
        r.status = PASS if resp.status_code in (400, 422) else FAIL
        if r.status != PASS:
            r.note = "expected 400/422 for malformed body"
    except Exception as e:
        r.actual, r.status, r.note = f"{type(e).__name__}: {e}", ERROR, "request failed"
    return r


# --------------------------------------------------------------------------- #
# 7. Concurrency
# --------------------------------------------------------------------------- #
def _one_stream(cfg, model):
    """A single short streaming request; returns (ok, ttft, tokens, err)."""
    client = make_client(cfg, timeout=60)
    try:
        stats = timed_stream(
            client, model=model,
            messages=[{"role": "user", "content": "Name three colors, comma separated."}],
            max_tokens=32, temperature=0.7,
        )
        return True, stats.ttft, stats.completion_tokens or 0, None
    except Exception as e:
        return False, None, 0, f"{type(e).__name__}: {e}"


def _concurrency(ctx: Context, rid: str, n: int) -> Result:
    r = Result(rid, "Concurrency", f"{n} simultaneous streaming request(s)",
               request=f"{n} concurrent streaming chat requests (max_tokens=32)",
               ideal="All succeed; record avg TTFT + aggregate tokens/sec; failures=0.")
    try:
        wall_start = time.perf_counter()
        with cf.ThreadPoolExecutor(max_workers=n) as ex:
            outs = list(ex.map(lambda _: _one_stream(ctx.cfg, ctx.cfg.model), range(n)))
        wall = time.perf_counter() - wall_start
        oks = [o for o in outs if o[0]]
        fails = [o for o in outs if not o[0]]
        ttfts = [o[1] for o in oks if o[1] is not None]
        total_tokens = sum(o[2] for o in oks)
        avg_ttft = sum(ttfts) / len(ttfts) if ttfts else None
        agg_tps = total_tokens / wall if wall > 0 else None
        ctx.perf[rid] = {
            "n": n, "ok": len(oks), "fail": len(fails),
            "avg_ttft_s": round(avg_ttft, 3) if avg_ttft else None,
            "wall_s": round(wall, 3),
            "agg_tokens_per_sec": round(agg_tps, 1) if agg_tps else None,
        }
        r.actual = (f"ok={len(oks)}/{n} fail={len(fails)} avg_ttft="
                    f"{f'{avg_ttft:.3f}s' if avg_ttft else 'n/a'} "
                    f"agg_tok/s={f'{agg_tps:.1f}' if agg_tps else 'n/a'} wall={wall:.2f}s")
        if fails:
            r.note = "errors: " + "; ".join(trim(f[3], 80) for f in fails[:3])
        r.status = PASS if not fails else (PARTIAL if oks else FAIL)
    except Exception as e:
        r.actual, r.status, r.note = f"{type(e).__name__}: {e}", ERROR, "request failed"
    return r


def check_X1(ctx):
    return _concurrency(ctx, "X1", 1)


def check_X2(ctx):
    return _concurrency(ctx, "X2", 5)


def check_X3(ctx):
    return _concurrency(ctx, "X3", 10)


# Ordered registry used by both the runner and the pytest suite.
ALL_CHECKS = [
    check_H1, check_H2,
    check_C1, check_C2, check_C3,
    check_P1, check_P2a, check_P2b, check_P3,
    check_S1, check_S2,
    check_B1, check_B2,
    check_E1, check_E2, check_E3, check_E4, check_E5, check_E6,
    check_X1, check_X2, check_X3,
]
