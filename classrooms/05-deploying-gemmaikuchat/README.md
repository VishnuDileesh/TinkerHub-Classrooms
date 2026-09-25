# Deploying GemmaikuChat — Render + Vercel

Everything in [Module 04](../04-gemmaikuchat-fastapi) ran on your machine: `ollama serve` in one terminal, `uvicorn main:app` in another, both talking over `localhost`. That's the right way to build something — fast iteration, no network latency, no bill. But "runs on my laptop" and "runs on the internet" are different problems, and the gap between them is where most people bounce off deployment. This module closes that gap, on purpose, by making you change the two files from Module 04 yourself rather than handing you pre-configured code — the changes are small, and making them is the lesson.

## The two problems deployment actually is

"Deploy GemmaikuChat" sounds like one task. It's two, and conflating them is where people get stuck:

1. **Where does the model live?** Something has to keep `ollama serve` running, with the `gemmaiku` GGUF loaded, 24/7, reachable over a network — not a request-scoped function, an actual resident process with weights sitting in RAM.
2. **Where does the app (FastAPI backend + HTML/CSS/JS frontend) live, and how does it reach the model?** This is a much lighter, mostly-stateless job — it just needs to be reachable and able to make outbound requests to wherever problem #1 lives.

These have different hosting requirements, which is exactly why this module uses two different platforms instead of one.

## Why the model can't go on Vercel

Vercel is built around **serverless functions**: your code runs only in response to a request, for at most a handful of seconds (longer on paid plans, but still bounded), on a fresh, ephemeral filesystem each time. That's a fantastic model for stateless API handlers and static sites. It is fundamentally the wrong shape for Ollama, because Ollama needs to:

- **Stay running in the background**, listening on a port, independent of any single request.
- **Keep a multi-hundred-megabyte-to-multi-gigabyte model resident in memory** between requests, instead of re-loading it from disk every time (which a serverless cold start would force).
- **Persist those model weights on disk** across invocations — Vercel's filesystem is wiped between them.
- Ideally get real, sustained CPU/RAM (or GPU) for the duration of generation, not a function timeout ticking down.

None of that exists in a serverless function. So: **the model never goes on Vercel.** Vercel's job in this module is only the static frontend — three files, no build step, exactly what it's good at.

## Why Render fits both jobs

[Render](https://render.com) runs **Web Services** as long-lived Docker containers, not request-scoped functions — which is the property Ollama actually needs. It also gives you:

- **Persistent Disks** you can attach to a service, so downloaded model weights survive restarts and redeploys instead of re-downloading every time.
- **Private networking between services in the same account** — your FastAPI backend can reach your Ollama service by an internal hostname, without that traffic ever touching the public internet.

So the plan: **Ollama runs on Render. FastAPI runs on Render too** (as a second, small service). **The frontend runs on Vercel**, talking to the Render backend over the public internet. You'll see at the end this also has a simpler one-service variant if you'd rather skip Vercel entirely — read that far before you start clicking around dashboards.

## Where can `gemmaiku` actually be deployed? (the model-hosting question)

Before touching Render's dashboard, it's worth knowing the full menu, because Render-with-a-CPU is the *accessible* answer, not the *only* one:

| Where | Good for | Trade-off |
|---|---|---|
| **Render Web Service, `ollama/ollama` image, CPU** | Free/cheap demos, this module | No GPU on free/entry tiers — the 270M `gemmaiku` model is snappy, the 1B model is noticeably slower, anything bigger is impractical. |
| **Render, paid plan with more RAM/CPU** | A steadier demo if 270M/1B on the free tier feels too slow | Costs money; still no GPU. |
| **Hugging Face Inference Endpoints** | Real GPU-backed inference, minutes to set up since `vi-c0de`'s models are already published there | Pay-per-hour while the endpoint is up; different API shape than Ollama (you'd adjust the backend's request/response handling). |
| **Fly.io GPU machines / Modal / RunPod** | Production-grade throughput, scale-to-zero GPU options | More infrastructure to learn; overkill for a Classrooms demo. |
| **Vercel, anywhere** | — | Not viable, per the section above. Don't. |

This module walks the first row — Render, CPU, free tier — because it's the one you can actually follow start to finish without a credit card doing anything scary. The others are listed so you know where to go when you outgrow it; **Things to try next** at the bottom points back at them.

## Step 1 — Deploy Ollama + `gemmaiku` on Render

1. On [render.com](https://render.com), **New → Web Service → Deploy an existing image**, and give it the public image `ollama/ollama`.
2. Add a **Disk** to the service — mount path `/root/.ollama`, a few GB (this is where model weights land; without it, every restart re-downloads the model from scratch).
3. Set the environment variable `OLLAMA_HOST=0.0.0.0` on this service. Ollama binds to `127.0.0.1` by default, which is unreachable from outside its own container — `0.0.0.0` tells it to listen on all interfaces so Render's networking can reach it. (This is a setting *on the Ollama container itself* — don't confuse it with the same-named variable you're about to add to the FastAPI service in Step 2; they're two different processes.)
4. Under the service's **Shell** tab (or a one-off Job), pull the model, using exactly the pattern from [Module 03](../03-huggingface-to-ollama):
   ```bash
   ollama pull hf.co/vi-c0de/gemmaiku-3-1b-it-GGUF-experimental
   # or, for a snappier free-tier CPU demo:
   ollama pull hf.co/vi-c0de/gemmaiku-3-270m-it-experimental
   ```
   This only needs to happen once — the persistent disk keeps the weights around across restarts.
5. Confirm it's alive from the Shell tab: `ollama list` should show the model; `curl localhost:11434/api/tags` should return JSON.
6. Note the service's **internal hostname** shown in the Render dashboard (something like `ollama-gemmaiku:11434`) — you'll point the backend at it next. You do not need to make this service public; keep it private and let only your own backend reach it over Render's internal network.

## Step 2 — Make the backend deployable (the one real code change)

Open `../04-gemmaikuchat-fastapi/backend/main.py`. It currently has this hardcoded near the top:

```python
OLLAMA_HOST = "http://localhost:11434"
```

That works on your laptop, where Ollama really is on `localhost`. On Render, Ollama is a *different service* reachable only by its internal hostname. Change it to read from the environment, falling back to `localhost` so local development still works unmodified:

```python
import os

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
```

That's the entire required change to `main.py`. This is the general shape every "make it deployable" pass takes: find the config that was quietly hardcoded for your machine, and make it an environment variable with a sane local default.

Now deploy this service on Render too:

1. **New → Web Service**, point it at this repo, with the **root directory** set to `classrooms/04-gemmaikuchat-fastapi/backend`.
2. Build command: `pip install -r requirements.txt`. Start command: `uvicorn main:app --host 0.0.0.0 --port $PORT` (Render assigns the port via `$PORT` — binding to `0.0.0.0` and reading that variable, rather than hardcoding `8000`, is required for Render's router to reach your container).
3. Set the environment variable `OLLAMA_HOST=http://ollama-gemmaiku:11434` (use the actual internal hostname from Step 1).
4. Deploy. Visit the service's public Render URL — you should get the GemmaikuChat UI, model dropdown populated, and a working chat, all on one URL, with no CORS to worry about (frontend and backend are being served from the same origin by this one service, exactly like Module 04 did locally).

**If you don't specifically need Vercel** — custom domain aside, this is already a complete deployment. Stop here; the app is live on a single Render URL. Read on only if you want the frontend served from Vercel specifically.

## Step 3 — Split the frontend onto Vercel

Serving the frontend from the same backend (Step 2) means one origin, one URL, zero CORS configuration — the simplest possible deploy. Putting the frontend on Vercel instead means two origins talking to each other, which costs you two small changes, both for the same underlying reason: **browsers block a page on one origin from calling an API on another origin unless that API explicitly allows it.**

**3a. Point the frontend at the Render backend.** Open `../04-gemmaikuchat-fastapi/frontend/static/js/app.js`. It currently calls the API with relative paths:

```js
const res = await fetch("/api/models");
...
const res = await fetch("/api/chat", { ... });
```

Relative paths mean "same origin as this page" — fine when the backend serves the frontend itself, broken the moment the frontend lives on `your-app.vercel.app` and the backend lives on `your-app.onrender.com`. Add one constant near the top of the file and use it in both places:

```js
const API_BASE = "https://your-backend.onrender.com"; // your Step 2 Render URL
...
const res = await fetch(`${API_BASE}/api/models`);
...
const res = await fetch(`${API_BASE}/api/chat`, { ... });
```

**3b. Allow the Vercel origin on the backend.** Without this, the browser console will show a CORS error and every request will fail before it reaches your code — the browser blocks it client-side. Add FastAPI's CORS middleware to `main.py`:

```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://your-app.vercel.app"],
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)
```

Redeploy the Render backend with this change.

**3c. Deploy the frontend to Vercel.** On [vercel.com](https://vercel.com), **New Project**, point it at this repo, and set the project's **root directory** to `classrooms/04-gemmaikuchat-fastapi/frontend`. There's no framework and no build step — Vercel will detect it as a static site and serve `index.html` and `static/` as-is. Deploy, then open the Vercel URL.

## Checkpoint

You know this worked when:
1. Visiting your public URL (Render-only, or the Vercel URL if you did Step 3) from your **phone on cellular data** — not your dev laptop, not the same Wi-Fi — shows the GemmaikuChat UI with a populated model dropdown.
2. Sending a message streams a real reply from your Render-hosted `gemmaiku` model, tokens appearing progressively, same as it did locally in Module 04.
3. If you did the Vercel split: opening your browser's dev console shows **no CORS errors** in red.

## Cost and performance realism

Render's free tier **spins a Web Service down after ~15 minutes of no traffic** and takes a beat to spin back up on the next request — expect the first message after a lull to feel slow (your Ollama service reloading the model into memory), then fast after that. There's no GPU on free or entry-level paid tiers, so this whole setup is CPU inference: the 270M `gemmaiku` model stays comfortably responsive, the 1B model is usable but visibly slower, and anything meaningfully bigger than 1B stops being pleasant to chat with on CPU. If that ceiling matters to you, that's exactly when to go back to the Hugging Face Inference Endpoints / GPU-host row in the table above — this module gets you a real, live URL; it doesn't claim to be production infrastructure.

## Troubleshooting

- **CORS error in the browser console** — the Vercel origin isn't in the backend's `allow_origins` list, or you redeployed the frontend to a new URL and forgot to update it on the backend.
- **502/504 from the Render backend** — usually means it can't reach the Ollama service: double-check the internal hostname in `OLLAMA_HOST`, and confirm `OLLAMA_HOST=0.0.0.0` is set *on the Ollama service itself* (Step 1.3), not just the backend.
- **First request after idle times out or hangs** — that's the free-tier cold start described above; wait it out once, it warms up.
- **`ollama list` shows nothing after a redeploy** — the persistent Disk isn't attached, or isn't mounted at `/root/.ollama`; without it, weights don't survive a restart and you'll need to `ollama pull` again.

## Things to try next

- **Skip the cold start**: a scheduled ping (a free cron service hitting your Render URL every 10 minutes) keeps the free tier warm — a cheap, slightly hacky first taste of the uptime tricks real production services rely on.
- **Swap Render's CPU Ollama service for a Hugging Face Inference Endpoint** (GPU-backed, and `vi-c0de`'s models are already published there) and adjust the backend's request/response shape to match its API — a good forcing function for understanding *why* Module 04's backend exists as a separate layer instead of the frontend calling Ollama directly: swapping the model host only touches one file.
- **Put a custom domain on the Vercel frontend** and see the whole thing under your own name.
- **Wire up auto-deploy on push** on both Render and Vercel, so the next Classrooms session's changes ship without you touching a dashboard.
