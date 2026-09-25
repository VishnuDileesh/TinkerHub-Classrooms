# The inference engine — deploying Gemmaiku itself, on Fly.io and Hugging Face

[Module 05](../05-deploying-gemmaikuchat) got GemmaikuChat onto a real URL, with the model served by Ollama on a free Render CPU box. That's a genuinely working deployment — and it has a real ceiling. CPU inference means every concurrent user queues behind the last, the 1B model is sluggish, and anything bigger is out of the question. This module is about the layer underneath all of that: the **inference engine** — the actual server process that loads model weights and turns a prompt into tokens — and what changes when you put a real GPU behind it.

You'll deploy `gemmaiku` two more times, on two platforms with opposite philosophies: **Fly.io**, where you bring your own container and rent a GPU by the second, and **Hugging Face Inference Endpoints**, where you point at a model repo and Hugging Face runs the whole server for you. Doing both is the point — you'll walk away knowing which one to reach for, and why.

## What "the inference engine" actually is

Up to now, "Ollama" has been a black box you `pull` and `run` things through. It's worth naming what it's actually doing, because that's what you're about to swap out:

- **Loading weights into memory** in a format the hardware can execute efficiently (GGUF on CPU/Metal via `llama.cpp`, under Ollama's hood).
- **Managing the KV cache** — the running memory of everything generated so far in a conversation, so the model doesn't recompute attention over the whole prompt on every new token.
- **Batching requests** — deciding whether to serve two users' prompts one after another or interleaved, which is the single biggest lever on throughput under real concurrent load.
- **Exposing an HTTP API** (`/api/chat`, `/api/generate`) so something like Module 04's FastAPI backend doesn't need to know any of the above.

Ollama is a fine, general-purpose engine for this — it's why Modules 01–05 use it throughout. But it's tuned for "one person, one machine, low concurrency." Production inference engines (`vLLM`, Hugging Face's `text-generation-inference`) exist specifically to make batching and GPU memory management aggressive enough to serve many concurrent users off one GPU efficiently — that's the actual thing you're paying a GPU provider for.

## Two philosophies, on purpose

| | **Fly.io** | **Hugging Face Inference Endpoints** |
|---|---|---|
| What you provide | A Docker container (you choose the engine — Ollama, vLLM, TGI, anything that speaks HTTP) | Just a model repo — `vi-c0de/gemmaiku-3-1b-it-experimental` is already published there |
| What they provide | A GPU VM, networking, scale-to-zero autostop, a `fly.toml` | A fully managed `text-generation-inference` server, autoscaling, a dashboard |
| Control | Full — pick the exact engine, quantization, batching config | Minimal — pick a hardware tier and a few knobs, they run the server |
| Billing | Per-second, machine can `autostop`/`autostart` to zero when idle | Per-hour the endpoint is provisioned; can also scale-to-zero on idle |
| Best for | Learning what an inference server actually does; needing an engine HF doesn't offer (vLLM, custom batching) | Fastest path from "published model" to "GPU endpoint," zero container work |

Do Fly.io first — it forces you to understand the moving parts. Hugging Face second will then feel like it's doing something legible, not magic.

## Path A — Fly.io: bring your own container, rent a GPU by the second

### 1. Install and authenticate

```bash
curl -L https://fly.io/install.sh | sh
fly auth login
```

### 2. Write a `fly.toml` for a GPU Ollama machine

Create a new directory for this deployment (keep it separate from Module 04's app — this machine's only job is serving the model):

```bash
mkdir gemmaiku-inference && cd gemmaiku-inference
fly launch --image ollama/ollama --no-deploy
```

`fly launch` scaffolds a `fly.toml`. Edit it to request a GPU and attach persistent storage for model weights:

```toml
app = "gemmaiku-inference"
primary_region = "sin"  # pick one near your users

[build]
  image = "ollama/ollama"

[env]
  OLLAMA_HOST = "0.0.0.0"

[[mounts]]
  source = "ollama_data"
  destination = "/root/.ollama"

[[vm]]
  size = "a10"      # Fly's shared-GPU tier — enough for a 1B model
  gpu_kind = "a10"

[http_service]
  internal_port = 11434
  force_https = true
  auto_stop_machines = true   # scale to zero when idle
  auto_start_machines = true  # wake on the next request
  min_machines_running = 0
```

`OLLAMA_HOST = "0.0.0.0"` matters for the same reason it did in Module 05 — Ollama binds to `127.0.0.1` by default, which is unreachable from outside its own container. `auto_stop_machines`/`auto_start_machines` is Fly's scale-to-zero: the GPU machine shuts down after idle traffic and wakes on the next request, so you're not paying for a GPU sitting idle overnight — the trade-off is a cold-start delay (weights reloading into GPU memory) on that first request.

### 3. Create the volume and deploy

```bash
fly volumes create ollama_data --size 10 --region sin
fly deploy
```

### 4. Pull the model onto the running machine

```bash
fly ssh console -C "ollama pull hf.co/vi-c0de/gemmaiku-3-1b-it-experimental"
```

This is the exact same `hf.co/...` pull syntax from [Module 03](../03-huggingface-to-ollama) — the mechanism doesn't change just because the machine has a GPU now. Because it's on a mounted volume, this only has to happen once.

### 5. Point a backend at it

Fly gives your app a public hostname (`gemmaiku-inference.fly.dev`) and, for service-to-service calls, private `.internal` networking over WireGuard if your backend is also on Fly. Either way, this is the same one-line change from Module 05:

```python
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
# set OLLAMA_HOST=https://gemmaiku-inference.fly.dev in your backend's environment
```

### Checkpoint

```bash
fly ssh console -C "ollama run gemmaiku 'the ocean at dawn'"
```

should return a real 5-7-5 haiku, and noticeably faster than the CPU Render box from Module 05 — that gap *is* the GPU doing its job.

## Path B — Hugging Face Inference Endpoints: point at the model, skip the container

No Dockerfile, no `fly.toml` — this is the managed end of the spectrum.

### 1. Create the endpoint

Go to [huggingface.co/vi-c0de/gemmaiku-3-1b-it-experimental](https://huggingface.co/vi-c0de/gemmaiku-3-1b-it-experimental) → **Deploy → Inference Endpoints**. Pick:
- **Hardware**: a GPU tier (an Nvidia T4 or L4 is enough for the 1B model).
- **Task**: Text Generation.
- **Scale to zero**: enable it, with an idle timeout — same trade-off as Fly's autostop, expressed as a dashboard toggle instead of a `fly.toml` field.

Hugging Face provisions a dedicated `text-generation-inference` (TGI) server for this exact model and hands you an HTTPS URL and an API token. That's the entire deployment — no image to build, no volume to mount.

### 2. Call it

The API shape is different from Ollama's — worth noticing precisely because it isn't, on the surface, the same contract:

```python
import requests

response = requests.post(
    "https://<your-endpoint>.endpoints.huggingface.cloud",
    headers={"Authorization": "Bearer hf_..."},
    json={
        "inputs": "<start_of_turn>user\nthe ocean at dawn<end_of_turn>\n<start_of_turn>model\n",
        "parameters": {"max_new_tokens": 60, "temperature": 0.6, "stop": ["<end_of_turn>"]},
    },
)
print(response.json())
```

Notice you're constructing Gemma's chat template (`<start_of_turn>user...<end_of_turn>`) by hand here — TGI's raw `/` endpoint doesn't know your model's chat format the way Ollama's `Modelfile` `TEMPLATE` directive did in [Module 03](../03-huggingface-to-ollama). (Newer TGI deployments also expose an OpenAI-compatible `/v1/chat/completions` route that handles templating for you — check your endpoint's **API** tab in the HF dashboard for the exact path it exposes.)

### 3. Bridge it into GemmaikuChat

Module 04's backend talks to Ollama's `/api/chat` shape. Pointing it at a Hugging Face endpoint instead means adapting `_stream_ollama_chat` in `main.py` to call this different URL/payload/response shape rather than just changing `OLLAMA_HOST` — a good forcing function for seeing exactly where "the backend" and "the specific inference engine" are coupled, and how you'd generalize that boundary if you wanted to support either engine interchangeably.

### Checkpoint

The `curl`/`requests` call above returns generated text through a dedicated GPU endpoint, with no server for you to SSH into, patch, or restart.

## Fly.io vs. Hugging Face — what actually differed

Run the same handful of prompts through both and compare, honestly:

- **Setup effort**: Fly required a `fly.toml`, a volume, and an SSH pull step. Hugging Face required picking a hardware tier in a dropdown.
- **Control**: on Fly, you could swap `ollama/ollama` for a `vllm/vllm-openai` image tomorrow and get continuous batching, or run any engine that speaks HTTP. On Hugging Face, you get what TGI offers.
- **API shape**: Ollama's `/api/chat` (Fly) vs. TGI's `/`-or-`/v1/chat/completions` (Hugging Face) — different enough that a real multi-provider backend needs an adapter layer, not just a URL swap.
- **Cost shape**: both can scale to zero; Fly bills the machine by the second it's running, Hugging Face bills the endpoint by the hour it's provisioned (check current pricing for both before leaving anything running unattended).

Neither is "correct" — Fly is the right answer when you need a specific engine or fine-grained control; Hugging Face is the right answer when the model's already published there and you want a GPU URL in five minutes.

## Things to try next

- **Swap Fly's image from `ollama/ollama` to `vllm/vllm-openai`**, pointed at the merged (non-GGUF) `gemmaiku` weights, and compare throughput under a handful of concurrent requests — this is where you'll actually feel what "continuous batching" means instead of just reading the term.
- **Benchmark**: hit both deployments with something like `hey` or a small concurrent `asyncio` script sending N simultaneous prompts, and compare tokens/sec — turn "GPU is faster" into a number you measured yourself.
- **Add a fallback** in the backend: try the GPU endpoint, fall back to the Module 05 Render/Ollama box if it's cold-starting — a small taste of real production resilience.
- **Autoscale past one GPU** on either platform (Fly's `min_machines_running`/concurrency settings, or Hugging Face's endpoint replica autoscaling) and see what it takes to serve more than one person at a time without everyone queuing behind each other.
