# Module 02 — The same fine-tune, with Unsloth

[Module 01](../01-fine-tuning-gemma-mlx) fine-tuned Gemma 3 to speak only in 5-7-5 haiku, entirely on a Mac, using Apple's MLX. This module does the *exact same thing* — same dataset, same model family, same "does the syllable counter agree with it" checkpoint — but swaps the engine room. Instead of MLX we're using [Unsloth](https://unsloth.ai), running on an NVIDIA GPU (yours, a rented one, or a free one Google hands you).

If you've done Module 01, you already know what LoRA is and why the dataset looks the way it does. This module is deliberately light on re-explaining that and heavy on what's different: the hardware target, the quantization strategy, the API shape, and — honestly — the vibe. If you're arriving here cold, we've included enough of a recap that you won't be lost, but read [Module 01](../01-fine-tuning-gemma-mlx) first if you want the full story of *why* Gemmaiku exists and how the dataset was built.

Same haiku bot. Different toolchain. Let's go.

---

## Why bother with a second framework for the same task

Short answer: because the hardware you own determines the framework you use, and most of the world doesn't own a Mac with an M-series chip.

MLX is Apple's array framework, built to run natively and efficiently on Apple Silicon's unified memory. It's genuinely great if you have a MacBook — no drivers, no CUDA install, the same chip runs your fine-tune and your Sunday's coding. But it doesn't run on NVIDIA GPUs at all, which rules it out for the vast majority of machines doing fine-tuning in the wild: gaming PCs with a 3060 or 4090 sitting idle, university lab machines, cloud GPU boxes, and — the great equalizer — free Google Colab.

[Unsloth](https://github.com/unslothai/unsloth) targets that world. It's a library purpose-built to make LoRA/QLoRA fine-tuning on CUDA GPUs faster and lighter, using hand-written Triton kernels and clever memory tricks to cut training time and VRAM usage dramatically compared to plain Hugging Face `transformers` + `peft`. It's also, practically speaking, the more common path — if you go looking for fine-tuning tutorials, Kaggle notebooks, or Discord threads about training small open models, Unsloth is what most of them are built on. Learning it isn't just "the CUDA version of Module 01" — it's learning the tool the broader open-source fine-tuning community has converged on.

So this module does double duty: it's the same Gemmaiku fine-tune from a different angle, and it's your on-ramp to the toolchain you'll actually see used outside this repo.

---

## Prerequisites

Check these off before writing any code:

- **An NVIDIA GPU with CUDA**, OR **a free Google Colab account with a T4 GPU runtime**. This is the no-hardware-required path — if you don't own a GPU, open [colab.research.google.com](https://colab.research.google.com), make a new notebook, and set `Runtime → Change runtime type → T4 GPU`. Everything below runs fine on a free T4.
- **Python 3.10+** (3.11 is the sweet spot for current Unsloth wheels; Colab ships something compatible already).
- **A Hugging Face account**, with the [Gemma license accepted](https://huggingface.co/google/gemma-3-1b-it) on the model page (Google gates the Gemma weights — you have to click "agree" once, per account, before you can download them), and a **Hugging Face access token** with at least read access. Set it as `HF_TOKEN` in your environment or Colab secrets.
- Comfort with a terminal / notebook cells. Nothing else.

You do **not** need Module 01's dataset repo cloned separately — we point straight at the file it produced. If you have this whole repo checked out, the path `../01-fine-tuning-gemma-mlx/data/processed/haikus_2000.json` just works.

---

## Concepts recap — LoRA in 90 seconds

Full fine-tuning updates every weight in a model. For a 1B-parameter model that's 1B numbers to store gradients and optimizer state for — expensive, and mostly wasteful, since most of what the base model already knows (grammar, facts, how to hold a conversation) doesn't need touching. You're only trying to *bias* its behavior toward one constraint: always answer in 5-7-5 haiku.

**LoRA (Low-Rank Adaptation)** freezes the entire base model and instead injects small trainable "adapter" matrices alongside a chosen set of layers (usually the attention projections). Each adapter is a pair of skinny matrices — rank `r` — whose product approximates the *change* you want to make to that layer's weights. You train only those tiny matrices. For Gemma 3's 1B variant, that's a few million trainable parameters instead of a billion — small enough to fit and train on modest hardware, fast enough to iterate on in an afternoon.

**QLoRA** takes it one step further: it loads the frozen base model in 4-bit quantized precision (instead of 16-bit) *before* attaching the LoRA adapters. The frozen weights barely need precision — they're not being updated — so quantizing them saves enormous amounts of VRAM with negligible quality loss, and the LoRA adapters themselves still train in higher precision on top.

That's the whole trick, in both Module 01 and this one. What differs is who's doing the heavy lifting underneath.

## What Unsloth actually adds

Module 01 used `mlx-lm` / `mlx-tune`, which lean on MLX's native support for Apple Silicon's unified memory architecture. Unsloth's version of the same idea, for CUDA:

- **Custom Triton kernels.** Unsloth hand-writes fused GPU kernels for the operations that dominate fine-tuning time (attention, RoPE, RMSNorm, the LoRA matmuls themselves), replacing the generic ones `transformers` uses by default. In practice this means noticeably faster training steps and lower memory overhead for the same LoRA config, on the same GPU.
- **Native 4-bit quantization (QLoRA) baked in.** Loading a model 4-bit-quantized is a first-class, one-argument option, not something you bolt on separately — which is what makes fitting a 1B (or bigger) model comfortably inside a free T4's 16GB of VRAM realistic.
- **One unified API: `FastLanguageModel`.** Instead of juggling separate quantization configs, model loading, and PEFT wiring, Unsloth wraps it behind `FastLanguageModel.from_pretrained(...)` for loading and `FastLanguageModel.get_peft_model(...)` for attaching LoRA — both drop-in compatible with the rest of the Hugging Face ecosystem (`transformers`, `trl`, `datasets`) once loaded.
- **Training via `trl.SFTTrainer`.** Unsloth doesn't reinvent the training loop — it patches the model so that Hugging Face's own `SFTTrainer` (from the `trl` library) runs faster on top of it. If you've used `trl` before, the training step here will look completely familiar.
- **Built-in GGUF export.** Unsloth ships `model.save_pretrained_gguf(...)`, which quantizes and converts your merged fine-tune straight to GGUF in one call — no separate `llama.cpp` conversion dance. That's the file format [Module 03](../03-huggingface-to-ollama) loads straight into Ollama.

Everything else — the dataset, the 5-7-5 constraint, the idea of "does the syllable counter agree" — is identical to Module 01. Only the machinery changed.

---

## Step-by-step walkthrough

A companion script `finetune_unsloth.py` mirroring the code below would live alongside this README if you want to run the whole thing non-interactively — the steps are laid out here as a notebook-shaped walkthrough since that's how most people will actually run this (Colab or a local Jupyter kernel).

### 1. Install

```bash
pip install --upgrade pip
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install --no-deps trl peft accelerate bitsandbytes
```

On a fresh Colab T4 runtime, the equivalent single cell is:

```python
!pip install --upgrade -qqq pip
!pip install -qqq "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
!pip install -qqq --no-deps trl peft accelerate bitsandbytes
```

Log in to Hugging Face so the gated Gemma weights will actually download:

```python
from huggingface_hub import login
login(token="hf_your_token_here")  # or set HF_TOKEN and skip this
```

### 2. Load the base model, 4-bit

This is the Unsloth equivalent of Module 01's `mlx_lm.load` — one call gets you a quantized model plus its tokenizer, ready for LoRA.

```python
from unsloth import FastLanguageModel
import torch

max_seq_length = 1024   # haikus are short; we don't need much more than the prompt + a few lines
dtype = None             # let Unsloth auto-detect (bfloat16 on Ampere+, float16 otherwise)
load_in_4bit = True      # QLoRA — this is what makes a T4 comfortable

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/gemma-3-1b-it",   # Unsloth mirrors Gemma checkpoints, pre-quantized and ready
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)
```

Swap `unsloth/gemma-3-1b-it` for `unsloth/gemma-3-270m-it` if you want the smaller variant Module 01 also targeted — same trade-off as before: the 270M trains faster and fits anywhere, the 1B holds instructions and syllable structure more reliably.

### 3. Attach LoRA adapters

```python
model = FastLanguageModel.get_peft_model(
    model,
    r=16,                       # LoRA rank — higher = more capacity, more VRAM, slower
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_alpha=16,
    lora_dropout=0,              # Unsloth's fused kernels are optimized for dropout=0
    bias="none",
    use_gradient_checkpointing="unsloth",  # Unsloth's own checkpointing impl — cuts VRAM further
    random_state=3407,
)
```

`r=16` / `alpha=16` is a reasonable starting point — enough capacity to firmly override the model's default behavior without blowing up training time on a free GPU. More on tuning this in "things to try next."

### 4. Load and reshape the Gemmaiku dataset

The dataset is the same 2,000-example ShareGPT-format haiku set from Module 01 — see [its dataset card](../01-fine-tuning-gemma-mlx/data/README.md) for how it was built and validated. We load the same file and reformat it into whatever chat template Gemma expects, using Unsloth's chat-template helper so the special tokens match what the base model was pretrained with.

```python
from datasets import load_dataset
from unsloth.chat_templates import get_chat_template

dataset = load_dataset(
    "json",
    data_files="../01-fine-tuning-gemma-mlx/data/processed/haikus_2000.json",
    split="train",
)

tokenizer = get_chat_template(
    tokenizer,
    chat_template="gemma-3",   # matches Gemma's <start_of_turn>/<end_of_turn> format
)

def to_chat_format(example):
    # dataset ships as {"conversations": [{"from": "human"/"gpt", "value": ...}, ...]}
    # trl/Unsloth's chat helpers expect {"role": "user"/"assistant", "content": ...}
    role_map = {"human": "user", "gpt": "assistant"}
    messages = [
        {"role": role_map[turn["from"]], "content": turn["value"]}
        for turn in example["conversations"]
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    return {"text": text}

dataset = dataset.map(to_chat_format, remove_columns=dataset.column_names)
```

If you'd rather point at the published version instead of a local checkout, the identical dataset is also on Hugging Face as [`vi-c0de/gemmaiku-dataset`](https://huggingface.co/datasets/vi-c0de/gemmaiku-dataset) — swap the `load_dataset` call for `load_dataset("vi-c0de/gemmaiku-dataset", split="train")` and everything else is unchanged.

### 5. Train with `trl.SFTTrainer`

This is the part that will feel most different from Module 01 if you've read it — MLX's tuning CLI hides the training loop behind a config file, while here you're constructing it directly, `trl`-style.

```python
from trl import SFTTrainer, SFTConfig

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    args=SFTConfig(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,   # effective batch size of 8
        warmup_steps=10,
        num_train_epochs=2,               # 2000 examples, small model — 2 passes is plenty
        learning_rate=2e-4,
        logging_steps=10,
        optim="adamw_8bit",               # 8-bit optimizer state, another VRAM saver
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir="outputs",
        report_to="none",
    ),
)

trainer_stats = trainer.train()
```

On a T4, 2,000 examples at this batch size takes somewhere in the neighborhood of 15-30 minutes depending on which Gemma size you picked — go make tea.

### 6. Sanity-check with inference

Flip the model into inference mode and throw a few prompts at it before doing anything else:

```python
FastLanguageModel.for_inference(model)  # enables Unsloth's faster inference path

def ask(prompt: str) -> str:
    messages = [{"role": "user", "content": prompt}]
    inputs = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
    ).to("cuda")
    outputs = model.generate(
        input_ids=inputs, max_new_tokens=64, temperature=0.6, top_p=0.9, do_sample=True
    )
    return tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)

print(ask("Tell me about the ocean."))
print(ask("What's the best way to debug a memory leak?"))
```

If it's working, you should get three lines back, not a paragraph — and definitely not "As an AI language model..."

### 7. Export to GGUF for Ollama

This is the step Unsloth genuinely makes trivial. One call quantizes and converts the merged model straight to a `.gguf` file:

```python
model.save_pretrained_gguf(
    "gemmaiku-unsloth-gguf",
    tokenizer,
    quantization_method="q4_k_m",   # a solid default: small file, minimal quality loss
)
```

That drops a `.gguf` file you can hand straight to Ollama with the same `Modelfile` shape Module 01 used — see [Module 03](../03-huggingface-to-ollama) for wiring it up end to end. The short version, once you have the file:

```
FROM ./gemmaiku-unsloth.gguf

SYSTEM "You are a specialized Haiku bot. You only speak in 3 lines of 5-7-5 syllables. No chatter."

TEMPLATE """<start_of_turn>user
{{ .Prompt }}<end_of_turn>
<start_of_turn>model
"""

PARAMETER stop "<end_of_turn>"
PARAMETER temperature 0.6
PARAMETER top_p 0.9
```

```bash
ollama create gemmaiku-unsloth -f Modelfile
ollama run gemmaiku-unsloth "What's the weather like on Mars?"
```

---

## Checkpoint — how you know it worked

Same bar as Module 01: run the base (non-fine-tuned) model and your fine-tuned model against a handful of prompts, and run every response through the same syllable counter from Module 01 ([`syllables.py`](../01-fine-tuning-gemma-mlx/src/gemmaiku/syllables.py)):

```python
import sys
sys.path.insert(0, "../01-fine-tuning-gemma-mlx/src")
from gemmaiku.syllables import get_syllable_count_for_line

def check_haiku(text: str) -> bool:
    lines = [l for l in text.strip().split("\n") if l.strip()]
    if len(lines) != 3:
        return False
    counts = [get_syllable_count_for_line(l) for l in lines]
    return counts == [5, 7, 5]

prompts = [
    "Tell me about the ocean.",
    "What's the best way to debug a memory leak?",
    "Describe your favorite food.",
    "How does gravity work?",
    "What happened in 1969?",
]

for p in prompts:
    response = ask(p)
    print(f"{'PASS' if check_haiku(response) else 'FAIL'} — {p}")
    print(response, "\n")
```

You're not aiming for 100% on the nose — the base Gemma checkpoint should fail almost every prompt (it'll happily write a paragraph), and your fine-tuned model should pass the clear majority, landing 5-7-5 without being told the rule in the prompt. That gap — base model ignoring the constraint, fine-tuned model obeying it unprompted — is the actual proof the LoRA adapter learned something, on either framework.

---

## MLX vs Unsloth — what actually differed

Doing both modules back to back, here's the honest comparison, not a "they're basically the same" hand-wave:

| | MLX (Module 01) | Unsloth (this module) |
|---|---|---|
| **Hardware** | Apple Silicon only. Zero setup on a Mac, zero portability off one. | NVIDIA CUDA GPUs, or free on Colab. Runs on far more machines, including ones you don't own. |
| **Setup friction** | Almost none if you're on a Mac — `pip install mlx-lm mlx-tune` and you're training. No driver wrangling. | A bit more moving parts on your own box (CUDA drivers, `bitsandbytes` version matching your CUDA version), but Colab sidesteps essentially all of it — zero local setup if you use the free tier. |
| **Speed / memory** | Solid on unified memory, but you're bound by whatever RAM the Mac has, shared with the OS. | Custom Triton kernels + native 4-bit quantization make it noticeably faster and lighter per training step on comparable hardware — this is Unsloth's whole reason for existing. |
| **API shape** | A CLI-driven workflow (`mlx_lm.lora`, config files) — less code, less visibility into what's happening under the hood. | Fully code-driven (`FastLanguageModel` + `trl.SFTTrainer`) — more lines, but every knob is visible and it's the same shape as most Hugging Face fine-tuning code you'll see elsewhere. |
| **Ecosystem maturity** | Smaller community, Apple-specific; great docs for the basics, thinner for edge cases. | Much larger surrounding ecosystem — because it plugs directly into `transformers`/`trl`/`peft`, nearly every Hugging Face tutorial, notebook, or GitHub issue you find searching for LoRA fine-tuning help transfers here almost directly. |
| **Export path** | Needs a separate conversion step to get to GGUF for Ollama. | `save_pretrained_gguf` does quantize + convert in one call. |

Neither one is "better" in the abstract — Unsloth is objectively the more portable, more battle-tested, faster-training option if you have or can rent an NVIDIA GPU. MLX's real advantage is that it asks nothing of you beyond owning a Mac. Pick based on what's actually plugged into the wall in front of you.

---

## Things to try next

- **Turn the LoRA rank up or down.** Try `r=8` (faster, less capacity — does it still learn 5-7-5?) and `r=32` or `r=64` (slower, more capacity — does accuracy actually improve, or does it just overfit the training haikus?).
- **Change the quantization method on export.** Swap `q4_k_m` for `q8_0` (bigger file, closer to full precision) or `q4_0` (smaller, more lossy) and see if you can hear the difference in output quality.
- **Go bigger.** Point `from_pretrained` at a larger Gemma checkpoint and see how far a free T4 can actually stretch before you run out of VRAM — that ceiling is itself worth knowing.
- **Break the constraint on purpose.** Swap the haiku dataset for something else entirely — limericks, one-line jokes, a fixed word count — and watch how much of this pipeline you can reuse unchanged. If it's most of it, you've actually learned the mental model, not just this one script.
- **Ship it.** Once you've got a `.gguf` you're happy with, head to [Module 03](../03-huggingface-to-ollama) to get it running locally through Ollama, and from there [Module 04](../04-gemmaikuchat-fastapi) to put a real chat interface in front of it.
