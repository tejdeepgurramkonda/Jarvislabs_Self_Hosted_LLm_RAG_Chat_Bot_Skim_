# vLLM server — pause / resume runbook

How to pause the JarvisLabs instance and bring the `qwen` vLLM server back later.
The server does **NOT** auto-start on resume — you must launch it yourself, and the
instance ID + public URL change every time.

> Values change each resume. As of the last session: instance `443703`, public IP
> `217.18.55.93`, region IN2, 1× L4. The API key lives in `/home/.vllm_key` on the
> instance and in `tests/vllm/.env` locally — never hard-code it in a committed file.
>
> ⚠️ **Placeholders below (`<ID>`, `<PUBLIC_IP>`, `<HOST>`, `<API_KEY>`) are NOT
> literal.** Substitute the real values from the `jl` JSON / your `.env` before
> running. Pasting a command with `<HOST>` still in it gives
> `curl: (3) URL rejected: Bad hostname`; leaving `<API_KEY>` gives a 401.
> Commands here are shown for Windows PowerShell (your shell) — use `curl.exe`, not
> the `curl` alias.

---

## Pause (stop GPU billing, keep data)
```bash
jl pause <ID> --yes
```
Data under `/home` (code, model cache, scripts) persists. Storage still bills a little.

---

## Resume and start the server

### 1. Resume the instance
```bash
jl resume <ID> --yes --json
```
Note the **new** `machine_id` in the output — use it for every command below.

### 2. Find the current public IP + endpoint
```bash
jl get <NEW_ID> --json
```
From the JSON, grab:
- `public_ip`  → for SSH
- `endpoints`  → two `...notebooksn.jarvislabs.net` URLs; one maps to port 8000.
  You'll confirm which in step 5.

### 3. SSH in and start the vLLM server
There's a stale `~/.ssh/config` entry that rewrites SSH to a dead IP, so bypass it
with `-F /dev/null` and the identity key. **The login is `root@<PUBLIC_IP>` — a
single `root@`.** Writing `root@root@<IP>` makes SSH treat `root@<IP>` as the
*username* and fails with `Permission denied (publickey)`.
```powershell
ssh -F /dev/null -i ~/.ssh/jarvislabs -o StrictHostKeyChecking=no root@<PUBLIC_IP> "bash /home/start_gptq.sh"
```
This starts the GPTQ-Int4 server on port 8000 (`--served-model-name qwen`,
`--max-model-len 8192`), reading the API key from `/home/.vllm_key`, logging to
`/home/serve_gptq.log`. Model weights are cached, so it's ready in ~1–2 min.

### 4. (optional) Watch it come up
```powershell
ssh -F /dev/null -i ~/.ssh/jarvislabs root@<PUBLIC_IP> "tail -f /home/serve_gptq.log"
# ready when you see: "Application startup complete."   (Ctrl-C to stop tailing)
```

### 5. Health-check the public endpoint (find the right URL)
`jl get` returns **two** `notebooksn.jarvislabs.net` endpoints; only one proxies
port 8000, and which one changes each resume — so probe both. This snippet reads the
real key from your `.env` and substitutes it for you (no `<API_KEY>` to forget).
Paste your two **real** endpoint URLs from the JSON into the list:
```powershell
$key = ((Get-Content .\tests\vllm\.env) -match '^API_KEY=') -replace '^API_KEY=',''
foreach ($e in 'https://<HOST1>.notebooksn.jarvislabs.net',
               'https://<HOST2>.notebooksn.jarvislabs.net') {
  Write-Host "== $e =="
  curl.exe -s -H "Authorization: Bearer $key" "$e/v1/models"; Write-Host ""
}
```
The endpoint that lists `"id":"qwen"` is your `BASE_URL` (append `/v1`).
A single-URL check, if you already know the host — note `curl.exe` and no angle
brackets anywhere:
```powershell
curl.exe -H "Authorization: Bearer $key" https://f2e3b14426882.notebooksn.jarvislabs.net/v1/models
```

### 6. Point the tests / your app at it
Edit `tests/vllm/.env`:
```
BASE_URL=https://<HOST>.notebooksn.jarvislabs.net/v1
API_KEY=<API_KEY>
MODEL=qwen
```

### 7. Use it
```bash
python tests/vllm/run_suite.py          # re-run the functional suite
# or point your backend / any OpenAI client at BASE_URL + API_KEY
```

---

## When you're done
```bash
jl pause <NEW_ID> --yes
```

## Quick reference
| Thing | Where |
|-------|-------|
| Start script | `/home/start_gptq.sh` (on the instance) |
| API key | `/home/.vllm_key` (instance) / `tests/vllm/.env` (local) |
| Server log | `/home/serve_gptq.log` (on the instance) |
| Served model name | `qwen` |
| Port | 8000 |
| Context length | 8192 |
| SSH gotcha | `ssh -F /dev/null -i ~/.ssh/jarvislabs root@<ip>` (single `root@`) |
| curl gotcha | use `curl.exe`, substitute real host/key — never leave `<HOST>`/`<API_KEY>` |
