# Building GemmaikuChat — your own OpenWebUI

By now (if you came from [Module 03](../03-huggingface-to-ollama)) you have a model running locally in Ollama and you've talked to it with `ollama run` or a raw `curl` call. That's a terminal. What you actually want is a chat window — a text box, a streaming reply, a model picker, something you'd hand to a friend without them needing to open a shell.

You could `docker run` [OpenWebUI](https://openwebui.com) and be done in five minutes. That's genuinely the right call if you just want a daily driver. But it's a black box: you type, tokens stream in, and you have no idea what's actually happening in between. This module rips that black box open. You're going to build the same shape of thing — **GemmaikuChat** — from two files worth of Python and three files of plain HTML/CSS/JS, so the next time you use OpenWebUI (or any AI chat product, really) you know exactly what it's doing on your behalf.

By the end you'll have a real, running local ChatGPT-style interface, pointed at whatever models you've pulled into Ollama.

## What you'll build

```
04-gemmaikuchat-fastapi/
├── backend/
│   ├── main.py            # FastAPI app — 3 routes, ~140 lines
│   └── requirements.txt
└── frontend/
    ├── index.html
    └── static/
        ├── css/style.css
        └── js/app.js       # no framework, no build step
```

One backend process. Three routes. No database, no auth, no bundler. `uvicorn main:app` and a browser tab is the whole stack.

## Prerequisites

- **Ollama running locally** with at least one model pulled (`ollama pull gemma3:1b`, or the `gemmaiku` model from [Module 01](../01-fine-tuning-gemma-mlx)/[02](../02-fine-tuning-gemma-unsloth) via [Module 03](../03-huggingface-to-ollama)). You'll know this works when `ollama list` shows at least one model and `curl http://localhost:11434/api/tags` returns JSON.
- **Python 3.10+** and a way to make a virtual environment (`venv`, `uv`, `conda` — whatever you use elsewhere in this repo).
- Basic HTML/CSS/JS reading comprehension. No React, no Vue, no build tooling — this module is deliberately vanilla so there's nothing between you and the browser APIs doing the work.

## Concepts, before you read the code

**Why FastAPI, and why does the browser need a backend at all?**
Ollama already exposes an HTTP API on `localhost:11434`. In theory the browser could call it directly with `fetch()`. In practice you don't want your frontend hardcoding `localhost:11434` — it breaks the moment you deploy this somewhere Ollama isn't co-located, it can't do server-side things later (auth, rate limiting, swapping which backend serves which model), and browsers can be picky about cross-origin requests to a bare port like that. So the pattern every real chat product uses — OpenWebUI included — is: **browser talks to your backend, your backend talks to the model server.** Your backend is a thin, boring proxy. That's the whole job of `main.py`.

**Why streaming, and what is Server-Sent Events (SSE)?**
If you wait for the model to finish generating the entire reply before showing anything, a 200-token haiku-turned-essay feels like the app froze. Every real chat UI shows tokens as they're generated. Ollama's `/api/chat` endpoint, when called with `"stream": true`, returns a sequence of newline-delimited JSON objects — one per token — over a single open HTTP connection, closing only when generation is done. Our backend reads that stream and re-emits it to the browser using **Server-Sent Events**: a plain-text protocol where the server keeps a connection open and pushes `data: ...\n\n` frames whenever it has something new. SSE is one-directional (server → client only) and needs zero libraries on either end — it's just a `fetch()` with a readable stream on the frontend, and a generator function on the backend. That's exactly why it's the right tool here: we don't need bidirectional messaging like WebSockets would give us, just "keep pushing tokens until you're done."

**Why no frontend framework?**
Because the entire point of this module is seeing the mechanism. A React app would hide the streaming logic behind a hook and a state library. Fifty lines of vanilla `app.js` and you can trace every token from Ollama to pixel with your own eyes.

## Step-by-step walkthrough

### 1. Set up the backend environment

```bash
cd classrooms/04-gemmaikuchat-fastapi/backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` is intentionally short:

```text
fastapi>=0.115
uvicorn[standard]>=0.32
httpx>=0.27
pydantic>=2.9
```

`httpx` is the only interesting addition — it's what lets our backend make its own outbound streaming HTTP request to Ollama while simultaneously streaming a response back to the browser.

### 2. Read `backend/main.py` — there are exactly three routes

**`GET /api/models`** — calls Ollama's `GET /api/tags`, reshapes the result into `{name, size}` pairs, and returns it as JSON. This is what fills the model dropdown in the sidebar. If Ollama isn't running, this route catches the `httpx.ConnectError` and returns a proper `503` with a message telling you to start it — check the code for this pattern, it's the difference between a helpful error and a silent hang.

**`POST /api/chat`** — the interesting one. It takes a model name and a list of `{role, content}` messages (the whole conversation so far — LLMs are stateless, so every request has to resend the full history), forwards it to `POST http://localhost:11434/api/chat` with `stream: true`, and returns a `StreamingResponse`. The generator function `_stream_ollama_chat` is the core of the whole app:

```python
async for line in response.aiter_lines():
    chunk = json.loads(line)
    if chunk.get("done"):
        yield b"event: done\ndata: {}\n\n"
        return
    token = chunk.get("message", {}).get("content", "")
    if token:
        yield f"data: {json.dumps({'token': token})}\n\n".encode()
```

Ollama hands us one token at a time; we immediately re-wrap each one as an SSE frame and `yield` it. FastAPI keeps the HTTP connection to the browser open and flushes each `yield` as it happens — nothing is buffered until the end.

**Static file mount** — the last line, `app.mount("/", StaticFiles(...))`, serves `frontend/` (and everything under `frontend/static/`) directly. This has to be registered *last* — FastAPI matches routes in order, and a `/` mount registered first would swallow `/api/*` requests before they ever reached your API routes. This is a real, easy-to-hit bug — if you add new API routes later, keep them above the static mount.

### 3. Read `frontend/static/js/app.js` — no framework, just `fetch` and a `ReadableStream`

The interesting part is `streamAssistantReply`:

```js
const reader = res.body.getReader();
const decoder = new TextDecoder();
let buffer = "";

while (true) {
  const { done, value } = await reader.read();
  if (done) break;
  buffer += decoder.decode(value, { stream: true });
  const frames = buffer.split("\n\n");
  buffer = frames.pop(); // last chunk may be incomplete — hold it for next read
  for (const frame of frames) {
    // parse "event: ...\ndata: ..." and append the token to the bubble
  }
}
```

`fetch()`'s response body is a `ReadableStream` of raw bytes — network chunks don't respect your message boundaries, so a single SSE frame can arrive split across two `read()` calls, or two frames can arrive in one `read()`. The buffer-and-split-on-double-newline dance is how you turn "arbitrary byte chunks" back into "complete SSE frames" reliably. This exact pattern — chunk, decode, buffer, split — is worth remembering; you'll hit it anywhere you consume a streaming HTTP response by hand.

Everything else in `app.js` is plumbing you already know how to read: `localStorage` for conversation history (`loadConversations`/`saveConversations`), a bit of DOM manipulation to append chat bubbles, and an auto-growing `<textarea>`.

### 4. Run it

```bash
cd classrooms/04-gemmaikuchat-fastapi/backend
uvicorn main:app --reload
```

Open `http://localhost:8000`. Pick a model from the sidebar dropdown, type a message, hit enter.

## Checkpoint

You know this worked when:
1. The sidebar model dropdown lists the models `ollama list` shows you.
2. Sending a message shows tokens appearing progressively (a blinking cursor while streaming, not a message that pops in all at once after a delay).
3. If you picked `gemmaiku` (from Modules 01–03) as the model, the reply is a real 5-7-5 haiku.
4. Refreshing the page still shows your chat history in the sidebar (that's `localStorage` doing its job).
5. Stopping `ollama serve` and reloading shows a clear "could not reach Ollama" message in the sidebar and a graceful in-chat error — not a silent freeze or a raw stack trace.

## Common pitfalls

- **Blank page / 404 on refresh of a sub-route** — this app is a single page (`index.html`), it doesn't have client-side routing, so there's nothing to refresh-break — but if you extend it with routes later, remember the static mount is a catch-all.
- **CORS errors if you split frontend/backend onto different ports during development** — not an issue here since FastAPI serves both from the same origin. If you ever do split them, you'll need `fastapi.middleware.cors.CORSMiddleware`.
- **Streaming looks buffered/stuck** — some reverse proxies (nginx, certain cloud load balancers) buffer streaming responses by default. That's why `main.py` sets the `X-Accel-Buffering: no` header — it tells nginx specifically not to do that. If you deploy this behind a different proxy, check its docs for the equivalent setting.
- **`httpx.ConnectError` even though Ollama is running** — check it's actually bound to `localhost:11434` and not a different port or a Docker container's internal network.

## Things to try next

- **Add a stop button** — abort the `fetch()` mid-stream with an `AbortController` so the user can cut off a runaway generation.
- **Persist conversations server-side** instead of `localStorage`, so history survives across devices — this is the natural next step toward "real app," and a good excuse to add a database module to this repo.
- **Add a system-prompt field** in the UI, so you can give any pulled model a persona on the fly (tie this back to [Module 03](../03-huggingface-to-ollama)'s point about prompting vs. fine-tuning as two different levers on the same problem).
- **Render markdown** in assistant replies (code blocks, bold, lists) instead of plain text — a good excuse to read a small markdown-to-HTML library's source rather than pull one in blind.
- **Swap in a different model provider** (an OpenAI-compatible endpoint, a different local runner) behind the same `/api/chat` route, and notice how little of the frontend has to change — that's the whole point of the thin-backend-as-adapter pattern.
