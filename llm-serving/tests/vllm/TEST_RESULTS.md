# vLLM Functional Test — Results

- **Run at:** 2026-07-13 07:23:48
- **Target:** `https://f2e3b14426882.notebooksn.jarvislabs.net/v1`
- **Model:** `qwen`
- **max_model_len (reported):** 8192

## Summary

| Status | Count |
|--------|-------|
| ✅ PASS | 22 |
| ❌ FAIL | 0 |
| 🟡 PARTIAL | 0 |
| ⚪ N/A | 0 |
| 💥 ERROR | 0 |
| **Total** | **22** |

| ID | Area | Test | Status |
|----|------|------|--------|
| H1 | Reachability | GET /v1/models responds and lists the served model | ✅ PASS |
| H2 | Reachability | GET /health liveness endpoint | ✅ PASS |
| C1 | Correctness | Simple chat completion is coherent/on-topic | ✅ PASS |
| C2 | Correctness | Known factual question | ✅ PASS |
| C3 | Correctness | Simple arithmetic reasoning | ✅ PASS |
| P1 | Parameters | max_tokens caps output length | ✅ PASS |
| P2a | Parameters | temperature=0 is (near-)deterministic across repeats | ✅ PASS |
| P2b | Parameters | high temperature produces varied output | ✅ PASS |
| P3 | Parameters | stop sequence honored + finish_reason correct | ✅ PASS |
| S1 | Streaming | stream=True yields tokens incrementally | ✅ PASS |
| S2 | Streaming | measure TTFT and tokens/sec | ✅ PASS |
| B1 | Chat | multi-turn conversation is respected | ✅ PASS |
| B2 | Chat | system prompt changes the style of the answer | ✅ PASS |
| E1 | Robustness | empty prompt handled gracefully | ✅ PASS |
| E2 | Robustness | very long prompt near context limit handled gracefully | ✅ PASS |
| E3 | Robustness | huge max_tokens handled gracefully | ✅ PASS |
| E4 | Robustness | wrong model name returns a proper error | ✅ PASS |
| E5 | Robustness | invalid API key returns 401-style error | ✅ PASS |
| E6 | Robustness | malformed request body returns 4xx (not a hang) | ✅ PASS |
| X1 | Concurrency | 1 simultaneous streaming request(s) | ✅ PASS |
| X2 | Concurrency | 5 simultaneous streaming request(s) | ✅ PASS |
| X3 | Concurrency | 10 simultaneous streaming request(s) | ✅ PASS |

## Streaming performance (S2)

| Metric | Value |
|--------|-------|
| Time to first token | 0.151 s |
| Total time | 2.675 s |
| Completion tokens | 136 |
| Tokens/sec | 53.9 |
| Content chunks | 135 |

## Concurrency (X1/X2/X3)

| Concurrency | OK | Fail | Avg TTFT (s) | Wall (s) | Aggregate tok/s |
|-------------|----|------|--------------|----------|-----------------|
| 1 | 1 | 0 | 0.148 | 1.11 | 5.4 |
| 5 | 5 | 0 | 0.906 | 4.795 | 6.3 |
| 10 | 10 | 0 | 1.479 | 4.625 | 13.0 |

## Detailed results

### H1 — GET /v1/models responds and lists the served model  ✅ **PASS**

- **Area:** Reachability
- **Request:** GET https://f2e3b14426882.notebooksn.jarvislabs.net/v1/models
- **Ideal:** HTTP 200; data[] contains id 'qwen' (or 'qwen'). Capture max_model_len.
- **Actual:** HTTP 200; ids=['qwen']; max_model_len=8192

### H2 — GET /health liveness endpoint  ✅ **PASS**

- **Area:** Reachability
- **Request:** GET https://f2e3b14426882.notebooksn.jarvislabs.net/health
- **Ideal:** HTTP 200 (empty body). 404 => not exposed (PARTIAL, non-fatal).
- **Actual:** HTTP 200

### C1 — Simple chat completion is coherent/on-topic  ✅ **PASS**

- **Area:** Correctness
- **Request:** chat: user='Say hello in one short sentence.' (temperature=0, max_tokens=32)
- **Ideal:** Non-empty greeting; role=assistant; finish_reason=stop.
- **Actual:** role=assistant finish=stop content='Hello!'

### C2 — Known factual question  ✅ **PASS**

- **Area:** Correctness
- **Request:** chat: user='What is the capital of France? Answer in one word.' (temperature=0, max_tokens=48)
- **Ideal:** Answer contains 'Paris'.
- **Actual:** role=assistant finish=stop content='Paris'

### C3 — Simple arithmetic reasoning  ✅ **PASS**

- **Area:** Correctness
- **Request:** chat: user='What is 17 + 25? Reply with just the number.' (temperature=0, max_tokens=48)
- **Ideal:** Answer contains '42'.
- **Actual:** role=assistant finish=stop content='42'

### P1 — max_tokens caps output length  ✅ **PASS**

- **Area:** Parameters
- **Request:** chat: 'Write a long paragraph about the ocean.' max_tokens=16
- **Ideal:** completion_tokens <= 16 AND finish_reason == 'length'.
- **Actual:** completion_tokens=16 finish_reason=length text='The ocean, vast and mysterious, covers approximately 71% of our planet'

### P2a — temperature=0 is (near-)deterministic across repeats  ✅ **PASS**

- **Area:** Parameters
- **Request:** same prompt x3, temperature=0, max_tokens=48
- **Ideal:** 3 outputs identical (or >= 2/3 identical).
- **Actual:** 1 distinct of 3. sample='A rainbow is a spectrum of colors that appear when light is refracted, reflected, and dispersed in water droplets in the atmosphere, creating a colorful arc or circle. It typically forms in the direction opposite the sun and can be seen when'

### P2b — high temperature produces varied output  ✅ **PASS**

- **Area:** Parameters
- **Request:** same prompt x3, temperature=1.3, top_p=0.95, max_tokens=48
- **Ideal:** >= 2 distinct outputs.
- **Actual:** 3 distinct of 3. samples=['The city bloomed like a vibrant flower under the soft glow of万家灯火(wan家家火), each  …[+55 chars]', 'Above the bustling streets of Luminary City gleams an immense, iridescent skybri …[+134 chars]', 'Above the bustling streets of Luminescent City, skyscrapers sparkled like coloss …[+64 chars]']

### P3 — stop sequence honored + finish_reason correct  ✅ **PASS**

- **Area:** Parameters
- **Request:** 'Count from 1 to 9 separated by commas.' stop=['5']
- **Ideal:** Output stops before/at '5'; finish_reason == 'stop'.
- **Actual:** finish_reason=stop text='1, 2, 3, 4,'

### S1 — stream=True yields tokens incrementally  ✅ **PASS**

- **Area:** Streaming
- **Request:** chat stream=True: 'List three fruits, one per line.'
- **Ideal:** >= 2 content chunks (not one final blob); coherent reassembled text.
- **Actual:** n_chunks=7 ttft=0.114s text='Apple\nBanana\nCherry'

### S2 — measure TTFT and tokens/sec  ✅ **PASS**

- **Area:** Streaming
- **Request:** chat stream=True, ~120-token answer; time first chunk & total
- **Ideal:** TTFT recorded (ideal < ~2s) and tokens/sec recorded; streams incrementally.
- **Actual:** TTFT=0.151s total=2.675s tokens=136 tok/s=53.9 chunks=135

### B1 — multi-turn conversation is respected  ✅ **PASS**

- **Area:** Chat
- **Request:** system + user 'My name is Ada.' + assistant + user 'What is my name?'
- **Ideal:** Answer contains 'Ada'.
- **Actual:** content='Your name is Ada.'

### B2 — system prompt changes the style of the answer  ✅ **PASS**

- **Area:** Chat
- **Request:** same user 'Describe the sea.' with system A (terse pirate) vs B (formal oceanographer)
- **Ideal:** Two outputs clearly differ in style/tone.
- **Actual:** A(pirate)="Salty, vast, and full o' treasure awaitin'."
B(scientist)="The sea, a vast and dynamic body of saltwater that covers approximately 71% of Earth's surface, is a complex and multifaceted system. It extends from the shoreline to the deep ocea …[+101 chars]"

### E1 — empty prompt handled gracefully  ✅ **PASS**

- **Area:** Robustness
- **Request:** chat: user content='' (max_tokens=16)
- **Ideal:** Valid completion OR clean 4xx — no hang, no 500.
- **Actual:** HTTP 200; content='您好！您需要帮助吗？请告诉我您的问题或需要了解的内容。'
- **Note:** server accepted empty prompt and returned a completion

### E2 — very long prompt near context limit handled gracefully  ✅ **PASS**

- **Area:** Robustness
- **Request:** chat with ~7892 tokens of filler (max_model_len=8192), max_tokens=16
- **Ideal:** Either completes OR clear context-length 4xx — no hang/500.
- **Actual:** HTTP 200; prompt_tokens=7928; content="It looks like you're trying to provide some data or information, but the text"
- **Note:** completed near context limit

### E3 — huge max_tokens handled gracefully  ✅ **PASS**

- **Area:** Robustness
- **Request:** chat: max_tokens=999999
- **Ideal:** Server clamps OR returns 400 context error; no crash.
- **Actual:** HTTP 400: Error code: 400 - {'error': {'message': 'max_tokens=999999 cannot be greater than max_model_len=max_total_tokens=8192. Please request fewer output tokens. (parameter=max_tokens, value=999999)', 'type': 'BadRequestError', 'param': 'max_token …[+17 chars]
- **Note:** graceful 4xx

### E4 — wrong model name returns a proper error  ✅ **PASS**

- **Area:** Robustness
- **Request:** chat with model='not-a-real-model'
- **Ideal:** 404/400 error naming the unknown model.
- **Actual:** HTTP 404: Error code: 404 - {'error': {'message': 'The model `not-a-real-model` does not exist.', 'type': 'NotFoundError', 'param': 'model', 'code': 404}}

### E5 — invalid API key returns 401-style error  ✅ **PASS**

- **Area:** Robustness
- **Request:** chat with Authorization: Bearer bad-key-123
- **Ideal:** 401 if auth enabled; N/A if the server runs without auth.
- **Actual:** HTTP 401: Error code: 401 - {'error': 'Unauthorized'}

### E6 — malformed request body returns 4xx (not a hang)  ✅ **PASS**

- **Area:** Robustness
- **Request:** raw POST /chat/completions with messages='hello' and temperature='hot'
- **Ideal:** 400/422 validation error; no hang.
- **Actual:** HTTP 400: {"error":{"message":"2 validation errors:\n  {'type': 'list_type', 'loc': ('body', 'messages'), 'msg': 'Input should be a valid list', 'input': 'hello'}\n  {'type': 'float_parsing', 'loc': ('body', 'temperature'), 'msg': 'Input should be a  …[+554 chars]

### X1 — 1 simultaneous streaming request(s)  ✅ **PASS**

- **Area:** Concurrency
- **Request:** 1 concurrent streaming chat requests (max_tokens=32)
- **Ideal:** All succeed; record avg TTFT + aggregate tokens/sec; failures=0.
- **Actual:** ok=1/1 fail=0 avg_ttft=0.148s agg_tok/s=5.4 wall=1.11s

### X2 — 5 simultaneous streaming request(s)  ✅ **PASS**

- **Area:** Concurrency
- **Request:** 5 concurrent streaming chat requests (max_tokens=32)
- **Ideal:** All succeed; record avg TTFT + aggregate tokens/sec; failures=0.
- **Actual:** ok=5/5 fail=0 avg_ttft=0.906s agg_tok/s=6.3 wall=4.80s

### X3 — 10 simultaneous streaming request(s)  ✅ **PASS**

- **Area:** Concurrency
- **Request:** 10 concurrent streaming chat requests (max_tokens=32)
- **Ideal:** All succeed; record avg TTFT + aggregate tokens/sec; failures=0.
- **Actual:** ok=10/10 fail=0 avg_ttft=1.479s agg_tok/s=13.0 wall=4.62s
