# 03 — From Hugging Face to your terminal, via Ollama

Modules 01 and 02 ended with a model on disk — either a merged fine-tune or a stack of LoRA adapters sitting in a `models/` folder, only reachable through a Python script and whatever framework trained it. That's fine for a notebook. It's useless for anything else. You can't point a chat app at a `.safetensors` directory and a training script. You need a server that loads the weights once, keeps them warm, and answers requests over a socket like any other piece of infrastructure.

That's what this module is about: taking *any* model — someone else's, or your own — and getting it running behind [Ollama](https://ollama.com), a local model server, so it's just sitting there on `localhost:11434` waiting to be talked to. Module 04 builds a chat UI on top of exactly that socket, so getting this right is the bridge between "I have a model" and "I have an app."

There are two roads in here, and you should walk both:

- **Path A** — the model you want already exists as GGUF on Hugging Face. Three commands, no conversion, done in the time it takes to download.
- **Path B** — the model you want is your own fine-tune (or someone else's non-GGUF checkpoint), and you have to convert it yourself. This is what you'll actually do with your Module 01/02 output.

The running example throughout is [`vi-c0de`](https://huggingface.co/vi-c0de) on Hugging Face — the Gemmaiku account — because it happens to have both cases published: a ready GGUF repo for Path A, and non-GGUF fine-tunes that mirror exactly what you'll have sitting in your own `models/` folder for Path B. Everything here generalizes past Gemmaiku the moment you swap the repo name.

**Hardware/account prerequisites:** any machine that can run Ollama (macOS, Linux, Windows — CPU is fine for small models, GPU speeds things up). No GPU required for this module specifically — GGUF inference through Ollama runs fine on a laptop CPU for 1B–3B models. You'll want a free Hugging Face account, and a `huggingface-cli login` token if you plan to push your own model or pull from a gated repo. Budget a few GB of disk per model you pull — Ollama keeps every layer it downloads.

**A note on structure:** this module is README-only. There's no code to write — it's entirely CLI commands and one config file (a `Modelfile`, which is closer to a Dockerfile than a script). If you want a worked example to reference while you build your own, module 01's [`Modelfile`](../01-fine-tuning-gemma-mlx/models/Modelfile) is real and working — we'll walk through it directive by directive further down.

---

## What Ollama actually is

Ollama is two things bolted together: a local inference server, and a CLI/packaging system on top of [`llama.cpp`](https://github.com/ggerganov/llama.cpp) that makes running GGUF models feel like running Docker containers.

The Docker comparison isn't just marketing — it's structurally how Ollama works. A model in Ollama is stored as a set of content-addressed layers: the weights blob, the template, the system prompt, the parameters. `ollama pull` fetches layers the same way `docker pull` fetches image layers — deduped, cached, resumable. `ollama create -f Modelfile` builds a new "image" from a base layer plus your own config, the same way a `Dockerfile` builds on a `FROM` image. That's why the syntax in a `Modelfile` will look familiar if you've ever written a `Dockerfile` — `FROM`, layered config, a build step that produces a named, runnable artifact.

Why this abstraction is worth having: without it, "running a model" means writing Python, picking an inference library (`transformers`, `llama-cpp-python`, `mlx`, whatever), wiring up tokenization and chat templates by hand, and keeping a process alive yourself. With Ollama, "running a model" means `ollama run <name>` — it starts a server if one isn't already up, loads the model into memory (and unloads it after idle timeout, so you're not burning RAM on models you're not using), and gives you a REPL or an HTTP API. Every model you've pulled or built behaves identically from the outside, regardless of whether it started life as a Meta release, a Hugging Face community fine-tune, or your own weekend project.

**Install it:**

```bash
# macOS
brew install ollama
# or download the .dmg from https://ollama.com/download

# Linux
curl -fsSL https://ollama.com/install.sh | sh
```

Start the server (on macOS the app does this for you; on Linux you may need to run it explicitly):

```bash
ollama serve
```

Leave that running in a terminal (or let the background service handle it), and confirm it's alive:

```bash
curl http://localhost:11434
# Ollama is running
```

Everything below assumes that's up.

---

## Path A — pulling a ready-made GGUF straight from Hugging Face

If someone has already published a GGUF version of the model you want, you don't need to touch `llama.cpp`, you don't need a `Modelfile`, and you don't need to think about quantization math. Ollama can pull GGUF repos directly off the Hugging Face Hub using a special `hf.co/` reference:

```bash
ollama run hf.co/vi-c0de/gemmaiku-3-1b-it-GGUF-experimental
```

That one line does three things: resolves the repo on Hugging Face, downloads the GGUF weight file (plus whatever tokenizer/config metadata it needs), and drops you into a chat REPL against it. No `ollama create`, no `Modelfile` — Ollama infers a reasonable chat template from the GGUF's embedded metadata when the repo doesn't ship its own `Modelfile`-equivalent config.

If you just want to fetch it without immediately chatting:

```bash
ollama pull hf.co/vi-c0de/gemmaiku-3-1b-it-GGUF-experimental
```

### Picking a quantization

A single GGUF repo often contains *multiple* quantized versions of the same model — `q4_K_M`, `q5_K_M`, `q8_0`, `f16`, and so on. These are the same weights compressed to different bit-widths: lower bit-width means a smaller file and faster inference, at some cost to output quality. Roughly:

| Quant | Size vs f16 | Quality | When to use it |
|---|---|---|---|
| `f16` / `bf16` | 100% | reference quality | you have RAM/VRAM to spare and want the ground truth |
| `q8_0` | ~50% | near-lossless | good default if size isn't a constraint |
| `q5_K_M` | ~35% | very close to q8 | solid middle ground |
| `q4_K_M` | ~25% | noticeably compressed but usually fine for chat | most common default — best size/quality tradeoff for laptops |
| `q3_K_M` / lower | <25% | visibly degraded | only when you're desperate for size/speed |

If the repo has more than one file, pick a specific one with a `:tag` suffix instead of letting Ollama grab whatever it defaults to:

```bash
ollama pull hf.co/vi-c0de/gemmaiku-3-1b-it-GGUF-experimental:Q4_K_M
ollama run hf.co/vi-c0de/gemmaiku-3-1b-it-GGUF-experimental:Q4_K_M
```

Check what's actually in a repo before you pull — the model card on the Hugging Face page lists the files, or you can look at the repo's "Files and versions" tab in the browser. For a small model like Gemmaiku's 1B, `q4_K_M` or `q5_K_M` is plenty — you won't notice the difference in haiku quality, and it downloads in seconds instead of minutes.

### Gated or private repos

Some repos require you to accept a license or be an authorized member before you can download anything — pulling one anonymously gets you a `403`. Fix it by authenticating with Hugging Face *before* you pull:

```bash
pip install -U huggingface_hub
huggingface-cli login
# paste a token from https://huggingface.co/settings/tokens
```

Or, non-interactively, export the token as an environment variable in the same shell you run `ollama pull` from:

```bash
export HF_TOKEN=hf_your_token_here
ollama pull hf.co/some-org/some-gated-repo
```

Ollama picks up Hugging Face credentials from the same places the `huggingface_hub` library looks (the cached token from `huggingface-cli login`, or `HF_TOKEN`), so once you're logged in once, gated pulls just work.

That's the entire fast path. If the model you want already has a GGUF repo, you're done — skip to [the REST API section](#what-module-04-needs-from-you) and go build something.

---

## Path B — converting your own fine-tune to GGUF

This is the path you'll actually use coming out of Module 01 or 02, where you have a fine-tuned Gemma sitting in a local directory in Hugging Face `transformers` format (config.json, tokenizer files, `.safetensors` weights) — not GGUF. Ollama can't load that directly. You have to convert it.

### 1. Merge LoRA adapters, if you haven't already

If your fine-tune produced separate LoRA adapter weights rather than a fully merged model, merge them into the base model first — GGUF conversion works on a single dense set of weights, not a base model plus an adapter delta layered on top at runtime. Both fine-tuning modules cover this as part of their own workflow (MLX's fuse step in [`../01-fine-tuning-gemma-mlx`](../01-fine-tuning-gemma-mlx), Unsloth's `save_pretrained_merged` in [`../02-fine-tuning-gemma-unsloth`](../02-fine-tuning-gemma-unsloth)) — if you followed either module through to the end, you likely already have a merged directory sitting on disk. This module picks up from there, so go merge first if you haven't.

### 2. Convert Hugging Face format to GGUF with llama.cpp

`llama.cpp` ships a Python script that reads a standard Hugging Face model directory and writes out a `.gguf` file. Clone it and install its conversion dependencies:

```bash
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
pip install -r requirements.txt
```

Then convert your merged model directory:

```bash
python convert_hf_to_gguf.py /path/to/your/merged-gemmaiku-model \
  --outtype f16 \
  --outfile gemmaiku-1b.gguf
```

`--outtype f16` keeps full precision in the conversion — do this first, even if you plan to quantize afterward, because quantizing from a clean f16 GGUF gives better results than trying to convert straight to a low-bit format in one step. If you want to skip a separate quantize pass entirely, you can pass a quantized `--outtype` directly (e.g. `q8_0`), but for anything below `q8_0` it's worth doing the two-step conversion-then-quantize below instead.

This step is architecture-aware — `convert_hf_to_gguf.py` needs to recognize the model's architecture (Gemma, Llama, Qwen, Mistral, etc.) from its `config.json`. Gemma is well supported. If you're converting something exotic, check `llama.cpp`'s supported-architectures list before spending time on it.

### 3. Quantize further (optional, but usually worth it)

Once you have an f16 GGUF, shrink it with `llama-quantize` (built as part of `llama.cpp` — build it with `cmake` per the repo's instructions, or grab a release binary):

```bash
./llama-quantize gemmaiku-1b.gguf gemmaiku-1b-q4_k_m.gguf Q4_K_M
```

Same tradeoff table as Path A applies here — you're choosing it yourself instead of picking it off someone else's repo. For a 1B model on a laptop, `Q4_K_M` or `Q5_K_M` is the sane default: a 1B f16 GGUF is roughly 2GB, `Q4_K_M` shrinks that to roughly 700MB–800MB with minimal perceptible quality loss for a task as constrained as haiku generation. The smaller the model already is, the more a low quant hurts proportionally — a 1B model quantized to `Q3` or below starts to feel noticeably dumber; an 8B+ model tolerates aggressive quantization much better.

### 4. Write the Modelfile

This is the part that actually determines whether your model behaves correctly. A `Modelfile` tells Ollama four things: what weights to load, what system prompt to bake in, what chat template to wrap every message in, and what generation parameters to default to. Here's the real one from Module 01, used to package the 1B Gemmaiku fine-tune:

```dockerfile
FROM ./gemmaiku-1b.gguf

SYSTEM "You are a specialized Haiku bot. You only speak in 3 lines of 5-7-5 syllables. No chatter."

TEMPLATE """<start_of_turn>user
{{ .Prompt }}<end_of_turn>
<start_of_turn>model
"""

PARAMETER stop "<end_of_turn>"
PARAMETER temperature 0.6
PARAMETER repeat_penalty 1.1
PARAMETER top_p 0.9
```

Directive by directive:

- **`FROM ./gemmaiku-1b.gguf`** — the base layer. Points at your local GGUF file (relative to the `Modelfile`'s location). This is the only required directive; everything else is optional config layered on top of it.
- **`SYSTEM "..."`** — the system prompt baked into the model permanently. Every conversation starts with this instruction already in context, without the caller having to send it. This is where you encode the model's *persona/constraint* — for Gemmaiku, "only speak in haiku" lives here, not in application code.
- **`TEMPLATE "..."`** — the chat template. This has to match the exact special-token format the base model was trained on, or you get garbage. Gemma models use `<start_of_turn>{role}` / `<end_of_turn>` to mark turns — the template above wraps whatever the user sends inside a `user` turn and opens a `model` turn for the response, exactly mirroring how Gemma saw conversations during training. `{{ .Prompt }}` is Ollama's template variable for "the user's message goes here." Get this wrong — wrong tokens, wrong role names, missing turn markers — and the model doesn't error out, it just produces incoherent or repetitive output, because you're feeding it a token sequence it never learned to continue sensibly. See the troubleshooting section below; this is the single most common way a from-scratch conversion goes wrong.
- **`PARAMETER stop "<end_of_turn>"`** — tells Ollama where a generation should stop. Without this, the model keeps generating past its own turn and starts hallucinating a fake continuation of the conversation (writing both sides of the dialogue). This must match whatever end-of-turn token the template uses.
- **`PARAMETER temperature 0.6`** — sampling temperature. Higher = more random/creative, lower = more deterministic/repetitive-but-safe. 0.6 is a moderate value — enough variation that the haiku don't feel copy-pasted, not so much that the model wanders off the constraint.
- **`PARAMETER repeat_penalty 1.1`** — penalizes the model for repeating tokens it's already used in the current generation, at this strength just enough to discourage the loops small models are prone to without visibly distorting the output.
- **`PARAMETER top_p 0.9`** — nucleus sampling cutoff: only sample from the smallest set of tokens whose cumulative probability reaches 90%. Works alongside temperature to keep sampling from picking absurdly unlikely tokens even when temperature is nonzero.

Write your own `Modelfile` the same way, pointing `FROM` at your own `.gguf`, matching the `TEMPLATE` to whatever base architecture you converted (Gemma's format above if you fine-tuned Gemma; check the base model's own chat template — usually in its `tokenizer_config.json`'s `chat_template` field — if you converted something else), and adjusting `SYSTEM`/parameters to whatever behavior you're going for.

### 5. Build and run it

```bash
ollama create gemmaiku -f Modelfile
ollama run gemmaiku
```

`ollama create` reads the `Modelfile`, hashes and stores each referenced layer, and registers a named model you can now run like anything pulled from the library. `ollama list` shows it alongside anything you pulled in Path A; `ollama show gemmaiku` prints back the resolved template/parameters/system prompt so you can sanity-check what actually got baked in.

### 6. Optional: publish your own GGUF back to Hugging Face

Once you've converted and validated a model locally, push it to the Hub so anyone else — including future-you on a different machine — can grab it with Path A instead of repeating the conversion:

```bash
pip install -U huggingface_hub
huggingface-cli login   # if you haven't already

hf upload your-username/your-model-GGUF ./gemmaiku-1b-q4_k_m.gguf
# or, for full control / multiple files, from Python:
```

```python
from huggingface_hub import upload_folder

upload_folder(
    folder_path="./gguf-release",   # a directory containing your .gguf file(s)
    repo_id="your-username/your-model-GGUF",
    repo_type="model",
)
```

Write a minimal `README.md` model card alongside the GGUF file before you upload it — at minimum: what base model it's derived from, what it was fine-tuned on (or note it's an unmodified quantization), the license it inherits, and the exact `ollama run hf.co/...` command that pulls it. That last line is the whole point — it's what turns your one-off conversion into something the next person can use in Path A without reading this entire module.

---

## What Module 04 needs from you

By the end of this module, you should have at least one model — pulled or built — sitting behind `ollama run <name>` and answering coherently. Module 04 builds a real chat application on top of this, and it won't shell out to the `ollama` CLI to do it — it talks to Ollama's REST API directly, the same server your CLI commands have been talking to all along on `localhost:11434`.

Two endpoints matter:

**`/api/generate`** — single-turn completion, no chat history management:

```bash
curl http://localhost:11434/api/generate -d '{
  "model": "gemmaiku",
  "prompt": "the ocean at dawn",
  "stream": false
}'
```

**`/api/chat`** — multi-turn, takes a `messages` array (role/content pairs), and this is the one Module 04 actually uses, because a chat UI needs to send the whole conversation, not just the latest line:

```bash
curl http://localhost:11434/api/chat -d '{
  "model": "gemmaiku",
  "messages": [
    { "role": "user", "content": "the ocean at dawn" }
  ]
}'
```

By default that streams — Ollama sends back a sequence of newline-delimited JSON objects as tokens are generated, each with a `message.content` fragment and a `done: false`, until a final object with `done: true`. That's exactly the shape Module 04's FastAPI backend needs to relay over Server-Sent Events to a browser: read the stream from Ollama, forward each chunk to the client as it arrives, and the page renders tokens as they're generated instead of waiting for the whole response. If you'd rather see one complete JSON blob instead of a stream (useful for quick testing), pass `"stream": false`.

Try it against whatever you built above right now, so you know the socket actually answers before you build an app around it:

```bash
curl http://localhost:11434/api/chat -d '{
  "model": "gemmaiku",
  "messages": [{ "role": "user", "content": "write me one about monsoon season in Kochi" }]
}'
```

That's the whole handoff. Module 04 ([`../04-gemmaikuchat-fastapi`](../04-gemmaikuchat-fastapi)) picks up exactly here.

---

## Troubleshooting

**`403` pulling from Hugging Face.** The repo is gated or private. Run `huggingface-cli login` (or set `HF_TOKEN`) and make sure you've accepted the repo's license/access request on the Hugging Face website first — logging in doesn't automatically grant access to gated repos, it just lets Ollama present your credentials once you already have it.

**Output is garbage, repetitive, or the model seems to ignore the prompt entirely.** Nine times out of ten this is a `TEMPLATE` mismatch — you're feeding the model a token sequence that doesn't match the chat format it was trained on. Go check the base model's actual chat template (its `tokenizer_config.json` on Hugging Face has a `chat_template` field, or the model card usually documents the turn-marker format directly) and make sure your `Modelfile`'s `TEMPLATE` and `stop` parameter reproduce it exactly — role names, special tokens, and all. A model that suddenly starts writing both sides of the conversation, or trails off into unrelated text, is almost always continuing past a turn boundary it doesn't recognize because your `stop` token doesn't match what the template actually emits.

**First pull is slow, or you're running low on disk.** GGUF files for anything above a few billion parameters get large fast (an 8B model at `q4_K_M` is still several gigabytes), and Ollama keeps every layer you've pulled or built on disk under `~/.ollama/models` until you remove it. Run `ollama list` to see what's actually stored, and `ollama rm <name>` to clear out anything you're not using. If you're testing multiple quantizations of the same repo, pull one at a time rather than grabbing every tag — you don't need `q4`, `q5`, and `q8` all sitting on disk simultaneously.

**`ollama create` succeeds but `ollama run` gives an error about the model architecture.** Usually means the GGUF conversion step failed silently on an unsupported or misidentified architecture. Re-check that `convert_hf_to_gguf.py` actually recognized your model's `config.json` `architectures` field — the conversion log names the architecture it detected; if it's wrong, the resulting GGUF is malformed even though the script exits cleanly.

---

## Checkpoint

You should be able to run:

```bash
ollama run gemmaiku
>>> the ocean at dawn
```

and get back three lines that scan 5-7-5, or close enough to it that you can see the model straining toward the constraint. If you pulled a different model in Path A, the equivalent checkpoint is simpler: you asked it something and it gave you a coherent, on-topic answer, streaming back through your terminal — proof the weights loaded, the template lines up, and the server's actually alive on `localhost:11434`.

If instead you get repetition, silence, or nonsense, you have a template mismatch or a stop-token problem — see troubleshooting above before you assume the model itself is bad.

## Things to try next

- **Pull a completely different model family** — a Llama or Qwen GGUF from Hugging Face, same `ollama run hf.co/...` syntax — and notice that nothing about the workflow changed. That's the actual point of this module: the mechanism is identical whether the weights came from Meta, Alibaba, or your own laptop last weekend.
- **Write a `SYSTEM` prompt persona on top of a general-purpose model, with zero fine-tuning.** Pull a plain instruction-tuned model, write a `Modelfile` that only changes the `SYSTEM` line — make it answer only in limericks, or only as a grumpy dock-worker from Fort Kochi, whatever you want — and compare how far prompting alone gets you versus what Module 01/02's actual fine-tune achieved. They're two different knobs for shaping model behavior — one changes the weights, one changes the instructions sitting in front of unchanged weights — and now you've felt the difference between them instead of just reading about it.
- **Push your own conversion back to Hugging Face** and have someone else at the next Classrooms session pull it with a one-liner. Nothing makes the Path A / Path B relationship click faster than watching your own Path B output become someone else's Path A.
