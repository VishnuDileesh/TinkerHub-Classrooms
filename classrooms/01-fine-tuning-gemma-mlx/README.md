# 01 — Fine-Tuning Gemma with MLX (Apple Silicon)

You're going to take a small Google Gemma model and beat it into a shape it wasn't born with: a model that speaks *only* in haiku, and gets the syllable count right almost every time. No system prompt tricks, no few-shot examples stuffed into every request — the constraint gets baked into the weights.

The point of this module isn't the haiku. It's everything underneath it: building a dataset that actually enforces a constraint, running a LoRA fine-tune on a laptop with no cloud GPU in sight, and proving — with numbers, not vibes — that the fine-tune changed the model's behavior. Haiku is just the training wheels. Swap it for anything else you care about once you've got the mechanics down.

This is the Apple Silicon / MLX version of the workflow. If you're on an NVIDIA GPU or want to run for free on Colab, [Module 02](../02-fine-tuning-gemma-unsloth) does the identical fine-tune with Unsloth instead.

---

## 1. What you're building, and why haiku

A haiku is 5-7-5 syllables across three lines. That's it. That's the whole rule.

That rule is a genuinely good teaching device, for reasons that have nothing to do with poetry:

- **It's small.** A haiku is seventeen syllables. You don't need billions of parameters or a data center to teach a model this — a 270-million-parameter model can learn it on an M-series MacBook in well under an hour.
- **It's mechanically checkable.** You don't need a human judge, an LLM-as-judge, or a vibe check to know if the model got it right. You write a syllable counter, run it over three lines, and compare the output to `[5, 7, 5]`. True or false. No ambiguity, no debate. That single property is what makes this whole module runnable without hand-wavy evaluation — you'll see the exact counter this project used at `src/gemmaiku/syllables.py`.
- **It touches the whole workflow.** Getting a model to reliably do this isn't "just prompt it nicely" — the base model, prompted well, still blows the syllable count more often than not. Fixing that requires a real dataset, a real fine-tune, and a real evaluation. Every stage of an applied-AI project shows up here in miniature: data → train → evaluate → ship.

By the end you'll have two fine-tuned models sitting in `models/` (not committed to git — model weights never are, see the "don't commit large binaries" convention in the repo root), a chart showing they actually learned the constraint, and a mental model for LoRA fine-tuning you can point at literally any other small, checkable behavior you want to teach a model next.

This project's published results live on Hugging Face under [huggingface.co/vi-c0de](https://huggingface.co/vi-c0de) — the dataset ([`vi-c0de/gemmaiku-dataset`](https://huggingface.co/datasets/vi-c0de/gemmaiku-dataset)) and both fine-tuned models ([`gemmaiku-3-270m-it-experimental`](https://huggingface.co/vi-c0de/gemmaiku-3-270m-it-experimental), [`gemmaiku-3-1b-it-experimental`](https://huggingface.co/vi-c0de/gemmaiku-3-1b-it-experimental), and a [GGUF build](https://huggingface.co/vi-c0de/gemmaiku-3-1b-it-GGUF-experimental) for running it via Ollama). You don't need to hit those numbers exactly — you need to see the same *shape* of result: fine-tuned model wins, decisively, over the base model.

---

## 2. Prerequisites

This module will not run on an Intel Mac, Linux, or Windows. Read this section before you install anything.

- **A Mac with Apple Silicon** (M1/M2/M3/M4, any variant). MLX is Apple's own array framework, built directly on top of the Metal API — it does not run on Intel Macs and has nothing to do with CUDA. If `uname -m` in your terminal doesn't print `arm64`, stop here and go to [Module 02](../02-fine-tuning-gemma-unsloth) instead.
- **At least 16GB of unified memory.** The 270M model is comfortable on 8GB; the 1B model and its training buffers want more headroom. If you're tight on RAM, do the 270M run first and treat the 1B run as optional.
- **[uv](https://docs.astral.sh/uv/)** for Python and dependency management. This project pins `requires-python = ">=3.14"` in `pyproject.toml` — uv will fetch that interpreter for you automatically, you don't need to have it installed system-wide.
  You'll know uv is working when `uv --version` prints a version number instead of "command not found."
- **A Hugging Face account**, with the [Gemma license accepted](https://huggingface.co/google/gemma-3-1b-it) on the model pages you plan to use (`google/gemma-3-270m-it` and/or `google/gemma-3-1b-it`), and an [access token](https://huggingface.co/settings/tokens) with at least read permission.
  You'll know this step worked when `huggingface-cli whoami` (or `hf auth whoami`, see below) prints your username instead of an authentication error, *and* when you can load the gated model without a 403.
- **[Ollama](https://ollama.com)**, installed and running, for the evaluation and dataset-generation scripts later in this module — they compare a fine-tuned model against a base model by talking to both over Ollama's local HTTP API. (This is the same tool Module 03 covers in depth — you don't need to have done that module first, just have Ollama installed and `ollama serve` reachable at `localhost:11434`.)

---

## 3. The concepts, from first principles

Skip this section if you've fine-tuned a model before. If you haven't, read it before you touch a notebook — the code will make a lot more sense once you know what it's for.

### Fine-tuning vs. prompting

Prompting changes what you *ask* a model to do, at inference time, every single time. Fine-tuning changes the model's *weights* — the numbers baked into the network — so the new behavior is there by default, without you having to re-explain it in every request.

You could get a base Gemma model to attempt haiku with a good system prompt and a few examples stuffed into the context window. It'll work sometimes. But you're paying for those instructions in tokens on every call, the model still drifts off the constraint under longer conversations, and you have zero guarantee it'll hold up on a topic your examples didn't cover. Fine-tuning trades a slow, upfront cost (build a dataset, spend GPU/NPU time training) for a model that just *is* the thing you wanted, every time, for free at inference time.

The rule of thumb: reach for fine-tuning when the behavior needs to be **consistent**, **structural**, and **cheap to run at scale** — not when you're still exploring what you even want the model to do. This module exists because "always respond in exactly 5-7-5 syllables" is exactly that kind of structural constraint that prompting alone struggles to hold onto reliably.

### LoRA: fine-tuning without touching (most of) the model

Full fine-tuning means updating every single weight in the model. For a 1-billion-parameter model, that's a lot of numbers to keep in memory at once — the weights themselves, their gradients, and the optimizer's running statistics for each one. That adds up to multiples of the model's raw size, which is exactly the kind of thing that doesn't fit on a laptop.

**LoRA** (Low-Rank Adaptation) sidesteps this. Instead of updating the original weight matrices directly, LoRA freezes them completely and injects small trainable "adapter" matrices alongside the layers that matter most (in this project: the attention projections `q_proj`/`k_proj`/`v_proj`/`o_proj` and the MLP projections `gate_proj`/`up_proj`/`down_proj` — you can see this listed explicitly in the training notebooks). Only those adapter matrices get trained. Everything else stays exactly as Google shipped it.

The math is more subtle than "3% of the weights," but the practical effect is what matters: **a fraction of the parameters to train, and a fraction of the memory needed to train them.** That's the difference between "needs an A100 cluster" and "runs on the laptop you're reading this on." Two numbers you'll see in the notebooks control how much capacity the adapters have:

- `r` (rank) — the size of the LoRA matrices. Higher rank means more capacity to learn, but more memory and a slower train. This project uses `r=32` for the 270M model and `r=16` for the 1B model.
- `lora_alpha` — a scaling factor on the adapter's contribution. This project uses `alpha = 2 × r` in both runs (64 and 32, respectively), which is a common starting ratio.

When training finishes, you're left with a small adapter file, separate from the base model. You then either keep them separate and load both at inference time, or **fuse** them — bake the adapter's changes permanently into a copy of the base model's weights, so you're left with one ordinary model file again. This module does the latter (see step 4 below).

### MLX: why this runs on a MacBook at all

[MLX](https://github.com/ml-explore/mlx) is Apple's array/ML framework, purpose-built for Apple Silicon. Two things about it matter here:

1. **Unified memory.** On a normal GPU setup, your data lives in system RAM and has to be copied over the bus into the GPU's separate VRAM before the GPU can touch it. Apple Silicon doesn't have that split — the CPU and GPU (and Neural Engine) all share the same pool of memory. MLX is designed around that: no copying data back and forth, no worrying about whether your model fits into a smaller, separate VRAM budget. If it fits in your Mac's RAM, MLX can generally get to it.
2. **No CUDA required.** MLX talks to the GPU through Apple's Metal API. You don't install CUDA toolkits, you don't fight driver versions, you don't need an NVIDIA card at all. This is the entire reason this module exists as a separate, parallel track to Module 02 — the *ideas* are identical, but the plumbing underneath is completely different.

This project layers **[mlx-tune](https://pypi.org/project/mlx-tune/)** on top of raw MLX — it gives you an Unsloth-style, high-level API (`FastLanguageModel`, `SFTTrainer`, `SFTConfig`) for loading models, attaching LoRA adapters, and running supervised fine-tuning, without you having to hand-write a training loop in raw MLX.

### "Instruction-tuned" (-it) vs. base, and why it matters here

Google ships each Gemma size in two flavors:

- **Base / pretrained (`-pt`)** — trained to predict the next token from a huge pile of internet text. It's a raw next-token predictor with no concept of "conversation." Ask it a question and it might continue your question instead of answering it.
- **Instruction-tuned (`-it`)** — the base model, further trained to follow instructions and hold a chat-style conversation (`<start_of_turn>user ... <end_of_turn>` / `<start_of_turn>model ...`). This is what you actually want to build on top of if your goal is "respond to a prompt with a haiku" — it already understands the shape of "user asks, model answers."

Every notebook in this module loads an `-it` checkpoint (`google/gemma-3-270m-it`, `google/gemma-3-1b-it`) for exactly this reason — fine-tuning starts from a model that already knows how to hold a conversation, and just needs to learn the extra constraint on top.

---

## 4. Project structure

```
01-fine-tuning-gemma-mlx/
├── README.md                      # this file
├── pyproject.toml / uv.lock       # dependencies: mlx-lm, mlx-tune, datasets, pronouncing, etc.
├── src/gemmaiku/
│   └── syllables.py                # the syllable counter — CMUDict lookup + heuristic fallback
├── data/
│   ├── README.md                   # full dataset card: how the 2,000-example dataset was built
│   ├── raw/haikus.json             # earliest hand/LLM-seeded haiku examples
│   └── processed/                  # haikus_500.json → haikus_2000.json(l), the final training set
├── notebooks/
│   ├── data_preparation/
│   │   ├── initial_dataset.ipynb   # first look at the raw dataset
│   │   ├── syllable_counter.ipynb  # building and testing the syllable counter
│   │   └── dataset_fixer.ipynb     # re-validating/repairing dataset entries
│   ├── gemma_3_270m/
│   │   ├── gemma_3_270M.ipynb           # base model, no instructions
│   │   ├── gemma_3_270M-it.ipynb        # base instruction-tuned model, unmodified
│   │   ├── gemma_3_270M-it-haiku.ipynb  # -it model, prompted (not fine-tuned) to try haiku
│   │   ├── gemmaiku-270m.ipynb          # the actual LoRA fine-tune — start here
│   │   └── gemmaiku_inference.ipynb     # load the fused fine-tuned model, chat with it
│   └── gemma_3_1b/                      # same structure, 1B-parameter model
├── scratch/
│   ├── generate_haikus.py          # rejection-sampling dataset generation pipeline
│   ├── fix_seed_syllables.py       # re-validates/repairs the seed dataset
│   ├── fix_dataset_syllables.py    # re-validates/repairs the full 2,000-example dataset
│   └── run_evaluation.py           # base vs. fine-tuned comparison, produces the chart
├── models/
│   └── Modelfile                   # Ollama config for serving the fused GGUF model
└── assets/                         # screenshots and charts referenced below
```

Model weights, `mlx_outputs/` (LoRA checkpoints), and generated dataset JSON aren't committed — see `.gitignore`. You'll generate all of it locally as you work through the notebooks.

---

## 5. Step-by-step walkthrough

### Step 0 — environment and auth

From inside `classrooms/01-fine-tuning-gemma-mlx/`:

```bash
uv sync
```

This reads `pyproject.toml` / `uv.lock`, fetches the pinned Python version if you don't have it, creates a `.venv`, and installs everything: `mlx-lm`, `mlx-tune`, `datasets`, `transformers`, `pronouncing` (syllable lookup), `matplotlib`/`seaborn`/`pandas` (evaluation charts).

Authenticate with Hugging Face so you can pull the gated Gemma weights:

```bash
uv run hf auth login
```

Paste in your access token when prompted. Before this will actually work, go to the model pages ([`google/gemma-3-270m-it`](https://huggingface.co/google/gemma-3-270m-it) and [`google/gemma-3-1b-it`](https://huggingface.co/google/gemma-3-1b-it)) and click through the license agreement — the token alone isn't enough if you haven't accepted the gate on each model card.

**Checkpoint:** `uv run hf auth whoami` prints your Hugging Face username, not an error.

Register the environment as a Jupyter kernel so the notebooks can find it, and launch:

```bash
uv run --with jupyter jupyter lab
```

### Step 1 — explore and understand the dataset

Before you train anything, understand what you're training on. Work through, in order:

1. **`notebooks/data_preparation/initial_dataset.ipynb`** — loads `data/raw/haikus.json` and looks at the raw shape of a training example. Every entry is a `conversations` list: a `human` turn (the prompt/topic) and a `gpt` turn (the haiku), which is the ShareGPT-style conversational format `mlx-tune` expects.
2. **`notebooks/data_preparation/syllable_counter.ipynb`** — builds and sanity-checks the syllable counter that everything downstream depends on. Read `src/gemmaiku/syllables.py` alongside it — it's short:
   - For each word, it first tries a **CMUDict lookup** via the `pronouncing` library — a real pronunciation dictionary, so it counts syllables the way an actual dictionary would (`pronouncing.syllable_count`).
   - If the word isn't in the dictionary (a typo, a made-up word, a proper noun), it falls back to a **vowel-cluster heuristic**: count transitions into a vowel, subtract for silent trailing `e`, add back for `-le` endings after a consonant. It's not perfect — no cheap syllable heuristic is — but it's consistent, which is what a training filter needs.

   A good habit worth stealing here: don't just trust your own counter. `assets/syllable_counter_ui.png` shows the kind of external, independent syllable checker this project cross-checked against while building `syllables.py` — feeding it the same lines and comparing counts line-by-line is a cheap way to catch a counter that's confidently wrong.
3. **`notebooks/data_preparation/dataset_fixer.ipynb`** — re-runs the counter over the dataset and flags/repairs anything that doesn't come out to `[5, 7, 5]`.

Read `data/README.md` in full here — it's the dataset card, and it documents something worth understanding before you copy this pattern elsewhere: this dataset was **not** hand-written. It was built in two stages:

- **500 seed examples**, generated with a locally-run Gemma 3 1B instruct model (with a bit of help from larger models via OpenRouter early on), then manually validated to strictly satisfy 5-7-5.
- **1,500 more examples**, generated by a *fine-tuned* Gemmaiku model itself, run through a **rejection-sampling loop**: generate a candidate haiku for a topic, run it through the syllable counter, and if it isn't exactly `[5, 7, 5]`, throw it away and try again (up to 15 attempts per topic before giving up on that topic). This is the clever part worth sitting with — you're using the model to bootstrap its own training data, with a cheap, mechanical, *programmatic* judge deciding what makes the cut. No human reviewed 1,500 haiku one at a time; the syllable counter did. That's the same trick behind a lot of real synthetic-data pipelines: an expensive-to-generate, cheap-to-verify problem is exactly where rejection sampling shines.

`scratch/generate_haikus.py` is the actual script behind that second stage. Skim it — the interesting parts are the prompt pool (a matrix of templates × topic lists covering science, history, geography, food, people, and more, producing thousands of unique candidate prompts) and the `while attempts < max_attempts and not success` loop that is rejection sampling in its entirety. `scratch/fix_seed_syllables.py` and `scratch/fix_dataset_syllables.py` do a second validation pass after the fact, using the finished fine-tuned model to regenerate any entry that still fails the counter — because the counter itself was refined partway through the project, and old entries needed re-checking against the new rules.

You don't have to regenerate the dataset — `data/processed/haikus_2000.json` is already there, ready to train on. But now you know exactly how it was built, and that recipe (bootstrap → filter → expand) is reusable for basically any constraint you want to teach a model next.

### Step 2 — fine-tune the 270M model first

Open **`notebooks/gemma_3_270m/gemmaiku-270m.ipynb`**. This is the actual training run, and it's the one to do first — 270M parameters trains fast, so it's where you want to be when you're still debugging whether your dataset or your config is wrong. Walking through what it does, in order:

1. **Load the dataset and standardize it**: `load_dataset("json", data_files="../../data/processed/haikus_dataset.json")`, then `standardize_sharegpt(dataset)` to normalize the `human`/`gpt` conversation format into the shape `mlx-tune` expects.
2. **Load the base model**: `FastLanguageModel.from_pretrained(model_name="google/gemma-3-270m-it", ...)`.
3. **Apply the chat template**: `get_chat_template(tokenizer, chat_template="gemma-3")` — this teaches the tokenizer how to wrap messages in Gemma's `<start_of_turn>`/`<end_of_turn>` format.
4. **Format every example** through `tokenizer.apply_chat_template(...)` into plain training text, then split 90/10 into train/eval sets.
5. **Attach LoRA adapters**:
   ```python
   model = FastLanguageModel.get_peft_model(
       model,
       r=32,
       target_modules=["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
                        "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"],
       lora_alpha=64,
       lora_dropout=0,
       bias="none",
   )
   ```
6. **Configure and run training**:
   ```python
   training_config = SFTConfig(
       output_dir="mlx_outputs",
       per_device_train_batch_size=2,
       gradient_accumulation_steps=4,
       max_steps=1500,
       max_length=2048,
       learning_rate=1e-4,
       logging_steps=15,
       lr_scheduler_type="constant",
       dataset_text_field="text",
   )
   trainer = SFTTrainer(model=model, processing_class=tokenizer, train_dataset=train_dataset, args=training_config)
   trainer.train()
   ```
   On an M-series Mac, 1,500 steps on the 270M model takes somewhere in the ballpark of tens of minutes — go get a coffee, don't stare at the loss curve.
7. **Fuse the adapter into a standalone model** so you're not stuck loading a base model plus a separate adapter every time:
   ```bash
   python -m mlx_lm fuse --model google/gemma-3-270m-it \
       --adapter-path mlx_outputs/adapters \
       --save-path ../../models/gemmaiku-3-270m-it
   ```
8. The notebook ends with a quick interactive loop using the fused model — type a topic, get a haiku back — so you can eyeball it before running the formal evaluation.

### Step 3 — repeat for the 1B model

**`notebooks/gemma_3_1b/gemmaiku-1b.ipynb`** is the same notebook, structurally, against `google/gemma-3-1b-it`. The differences worth noticing, because they tell you something about how LoRA hyperparameters get chosen:

- `r = 16` instead of 32 (with `lora_alpha = 32`, keeping the same `alpha = 2r` ratio) — a bigger base model needs less adapter capacity to move its behavior meaningfully, so the rank comes down.
- `max_steps = 2000` and `learning_rate = 2e-4` — more steps, and a higher learning rate, to make sure the larger model actually converges on the constraint rather than drifting only partway there.

Everything else — dataset loading, chat template, fuse command, interactive check — is identical, just pointed at the 1B checkpoint and a separate `models/gemmaiku-3-1b-it` output directory. This run will take noticeably longer and use more memory than the 270M run; if your machine struggled with 270M, this is the one to skip.

### Step 4 — sanity-check with the inference notebooks

`notebooks/gemma_3_270m/gemmaiku_inference.ipynb` and `notebooks/gemma_3_1b/gemmaiku_inference.ipynb` do one thing: load your fused, fine-tuned model with `mlx_lm.load(...)` and run a single prompt through `generate(...)`, printing whatever comes back before the `<end_of_turn>` token. This is your fast feedback loop — before you run a formal evaluation, just talk to the thing and see if it's obviously behaving.

Compare this against the un-fine-tuned notebooks in the same folders (`gemma_3_270M.ipynb`, `gemma_3_270M-it.ipynb`, `gemma_3_270M-it-haiku.ipynb`) — those load the *base* model and, at best, prompt it toward haiku without any fine-tuning. Running both side by side is the fastest way to feel the difference fine-tuning makes before you get to the formal numbers in the next step.

### Step 5 — serve it and run the formal evaluation

The evaluation script talks to models over Ollama's HTTP API rather than loading them directly in Python, so first get your fine-tuned model into Ollama. `models/Modelfile` is the config for this:

```
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

You'll need a GGUF export of your fused model first — [Module 03](../03-huggingface-to-ollama) covers exactly that conversion, and registering a local Modelfile with `ollama create`, in detail. Once you have `gemmaiku:latest` running locally (`ollama list` should show it), pull the base model too for comparison:

```bash
ollama pull gemma3:1b
```

Now run the evaluation:

```bash
uv run python scratch/run_evaluation.py
```

Read `scratch/run_evaluation.py` before you run it — it's short and it's the whole point of this step. For ten fixed topics ("artificial intelligence," "the French Revolution," "black holes in space," and so on), it:

1. Queries **`gemma3:1b`** (base model) with a one-shot example baked into the prompt to nudge it toward haiku, and counts how many attempts (up to 30) it needs before the syllable counter says `[5, 7, 5]`.
2. Queries **`gemmaiku:latest`** (your fine-tune) with *just the topic* — no coaching prompt, because it shouldn't need one — and counts attempts the same way.
3. Plots attempts-per-topic for both models side by side and saves it to `assets/evaluation_chart.png`.

This is a genuinely fair test, not a rigged one: the base model gets a helpful example in its prompt and the fine-tuned model doesn't, and the fine-tuned model still needs dramatically fewer attempts, because the constraint lives in its weights instead of in your prompt engineering.

---

## 6. Checkpoint — how you know it worked

Open `assets/evaluation_chart.png` after running the evaluation script. You're looking for the fine-tuned model's bars to sit consistently lower than the base model's across most or all of the ten topics — "attempts needed to hit 5-7-5" should drop from the base model regularly burning through a dozen-plus attempts (or maxing out at 30, meaning it never landed a valid haiku on that topic) to the fine-tuned model landing one, cleanly, on the first or second try most of the time.

You don't need to match the exact numbers from the published run — different hardware, different random seeds, and dataset regeneration will all shift the specifics. What you're checking is the *shape*: a clear, consistent gap in favor of the fine-tuned model. If your chart shows the fine-tune performing about the same as the base model, something upstream is wrong — recheck that the LoRA adapter actually got fused into the model you pointed the evaluation at, not the raw base checkpoint.

A secondary checkpoint, cheaper to run: inside `gemmaiku-270m.ipynb` (or the 1B equivalent), the `run_internal_eval_detailed` + `visualize_results` cells run the syllable counter over your held-out eval split (the 10% the model never trained on) and plot a syllable-count histogram plus a perfect/fail bar chart. You want that histogram to cluster tightly around 17 total syllables (5+7+5), and the accuracy bar to show a high percentage of "Perfect" — not necessarily 100%, but a large majority.

---

## 7. Common pitfalls

- **Gated model access.** If loading `google/gemma-3-270m-it` or `-1b-it` throws a 403 or "you don't have access," you've either not run `uv run hf auth login`, or you've authenticated but never clicked through the license on the model's Hugging Face page. Both steps are required, separately.
- **Repetition loops.** `assets/repetition_loop.png` is in this module's assets for a reason — small instruction-tuned models, especially at low temperature or with no `max_tokens` cap, can get stuck repeating a phrase or a line instead of stopping cleanly at `<end_of_turn>`. If your inference notebook hangs or the model spits out the same clause over and over, check that you're passing a `sampler` with a reasonable temperature (the project settles on `temp=0.3` for inference) and a `max_tokens` bound, and that `repeat_penalty` is set when serving via Ollama (see the Modelfile above — `1.1` was the value that worked here).
- **The syllable counter isn't infallible.** It leans on a real pronunciation dictionary (CMUDict via `pronouncing`) but falls back to a vowel-counting heuristic for anything not in that dictionary — proper nouns, invented words, unusual topics. That's exactly why `fix_seed_syllables.py` and `fix_dataset_syllables.py` exist: as the counter's rules got refined mid-project, old dataset entries needed a second pass to make sure they still validated. If your evaluation numbers look suspiciously worse than expected, check whether the topic contains words the counter might be mis-scoring before you blame the model.
- **Training on the wrong file.** `data/processed/` has several JSON files at different stages (`haikus_500.json`, `haikus_dataset.json`, `haikus_2000.json`/`.jsonl`). Double-check which one a notebook is loading — training on the 500-example seed set instead of the full 2,000-example set will still technically run, it'll just converge to a weaker, less generalized model.
- **Forgetting to fuse.** LoRA training leaves you with an *adapter*, not a standalone model. If you try to load `mlx_outputs/adapters` directly with `mlx_lm.load(...)` expecting fine-tuned behavior, you'll get the base model's weights with no adapter applied. Run the `mlx_lm fuse` step before you evaluate or serve anything.
- **Memory pressure on the 1B run.** If your Mac starts swapping hard or the kernel dies partway through `trainer.train()` on the 1B model, drop `per_device_train_batch_size` or `max_seq_length`, or just stick with the 270M model — it demonstrates the exact same mechanics for a fraction of the resource cost.

---

## 8. Things to try next

- **Swap the constraint.** Haiku was picked because it's mechanically checkable. What else is? Limericks (checkable via rhyme + meter, harder), exactly-N-word responses (trivial to check, good warm-up), valid JSON matching a schema (checkable via a parser), a fixed response format for a specific domain you care about. Write the checker first — if you can't automate "did it work," you can't automate building the dataset either.
- **Turn the LoRA knobs.** Try `r=8` on the 270M model and see how much worse (or how similar) the result is — that tells you how much capacity this particular constraint actually needed. Try a much higher rank and watch training slow down for little extra gain. Try changing `lora_alpha` independently of `r` instead of keeping the `2r` ratio.
- **Change `max_steps` and watch what happens both ways.** Cut `gemmaiku-270m.ipynb`'s `max_steps` from 1500 down to 300 and see the eval accuracy drop — that's underfitting, made visible. Push it up and see if it plateaus or starts to overfit toward memorizing training topics.
- **Try a bigger jump than 270M → 1B.** If you have the RAM, see whether `google/gemma-3-4b-it` needs fewer training steps to hit the same accuracy, or whether the returns diminish faster than you'd expect for a task this narrow.
- **Take this model further.** [Module 02](../02-fine-tuning-gemma-unsloth) runs the identical fine-tune on CUDA/Colab with Unsloth — worth doing once you understand this version, just to see how much of the workflow is universal versus framework-specific. [Module 03](../03-huggingface-to-ollama) picks up exactly where this module's evaluation step left off: converting your fused model to GGUF and serving it locally through Ollama for anything beyond a notebook.

---

## Credit

This module is built directly from **Gemmaiku**, a project that fine-tuned Gemma 3 to speak only in haiku on Apple Silicon. Published dataset and models: [huggingface.co/vi-c0de](https://huggingface.co/vi-c0de). The dataset is released under Apache-2.0 — see `data/README.md` for the full card.
