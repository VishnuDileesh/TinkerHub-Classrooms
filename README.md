# TinkerHub Classrooms

**You don't learn to build. You build to learn.**

Most of us are conditioned from a young age to accept the world as it is given to us. Follow the syllabus, pass the test, don't break the rules. Somewhere along the way, learning became about compliance instead of creation. We started treating tools like Python and AI as subjects to memorize for a grade — not raw materials to bend to our will. That model was built for a factory era that no longer exists.

Steve Jobs once pointed out that the moment you realize everything around you was built by people no smarter than you, the world opens up. You can poke life, push on it, and watch something pop out the other side. You can mold it. As Richard Feynman put it, the best way to master anything is to pursue what fascinates you in the most irreverent, original, and undisciplined way possible.

**Classrooms** is a weekly laboratory run by Vishnu Dileesh with [TinkerHub](https://tinkerhub.org) at TinkerSpace, Kochi, built on that realization: agency, rapid execution, and raw proof of work over lectures and grades.

This repo is the lab notebook. Every session we run gets turned into a self-contained module here — real code, real models, real running software — so anyone, whether they were in the room or not, can pick it up, run it, break it, and rebuild it their own way.

---

## The project running through these Classrooms: Gemmaiku

Most modules in this repo orbit one running experiment: **Gemmaiku** — teaching a small [Gemma](https://ai.google.dev/gemma) language model to speak *only* in haiku (5-7-5 syllables, every time, no chatter). It's a deliberately small, deliberately weird target — which is exactly what makes it a good teaching project. It's small enough to fine-tune on a laptop, constrained enough that "did it work?" has a precise, checkable answer (count the syllables), and complete enough to touch every stage of a real applied-AI workflow: build a dataset, fine-tune a model, evaluate it, ship it, and put a chat interface in front of it.

Published artifacts from this project live on Hugging Face under [huggingface.co/vi-c0de](https://huggingface.co/vi-c0de):
- **Dataset**: [`vi-c0de/gemmaiku-dataset`](https://huggingface.co/datasets/vi-c0de/gemmaiku-dataset) — 2,000 validated haiku conversation pairs
- **Gemmaiku-3-270m-it** — the 270M parameter fine-tune
- **Gemmaiku-3-1b-it** — the 1B parameter fine-tune
- **Gemmaiku-3-1b-it-GGUF** — the 1B fine-tune packaged to run locally via Ollama

You don't need to reproduce Gemmaiku exactly. Swap the dataset, swap the constraint (haiku → jokes → SQL → your regional language → whatever you're curious about), and the same modules still teach you the same underlying mechanics.

## How this repo is organized

Each numbered folder under [`classrooms/`](classrooms/) is one **module** — a standalone lesson with its own README, its own code, and its own "you'll know it worked when..." checkpoint. Modules are ordered the way the ideas build on each other, but each one is written to also stand alone — if you already have a fine-tuned model and just want the chat UI, skip straight to Module 4.

| # | Module | What you'll learn |
|---|--------|--------------------|
| [01](classrooms/01-fine-tuning-gemma-mlx) | [Fine-tuning Gemma with MLX (Apple Silicon)](classrooms/01-fine-tuning-gemma-mlx) | Building a constrained dataset, LoRA fine-tuning a small LLM entirely on a Mac using Apple's MLX framework, and evaluating whether it actually learned the constraint. |
| [02](classrooms/02-fine-tuning-gemma-unsloth) | [The same fine-tune, with Unsloth](classrooms/02-fine-tuning-gemma-unsloth) | The identical fine-tuning workflow from Module 1, redone with [Unsloth](https://unsloth.ai) on a CUDA GPU (locally or free on Google Colab) — so you can compare frameworks and pick whichever fits the hardware you actually have. |
| [03](classrooms/03-huggingface-to-ollama) | [From Hugging Face to your terminal, via Ollama](classrooms/03-huggingface-to-ollama) | Downloading any published model — including the [vi-c0de](https://huggingface.co/vi-c0de) Gemmaiku models — from Hugging Face and running it locally with [Ollama](https://ollama.com), plus converting your own fine-tunes to GGUF so they can run there too. |
| [04](classrooms/04-gemmaikuchat-fastapi) | [Building GemmaikuChat — your own OpenWebUI](classrooms/04-gemmaikuchat-fastapi) | Building a real, running chat interface from scratch — Python [FastAPI](https://fastapi.tiangolo.com) backend streaming tokens over Server-Sent Events, plain HTML/CSS/JS frontend, no framework magic — so you understand exactly how tools like OpenWebUI work under the hood, because you just built one. |

More modules get added after every Classrooms session. The structure is meant to keep growing sideways — new module, same pattern: a README that teaches the concept, code you can actually run, and a way to know you got it right.

## How to use this repo

1. **Pick a module.** Start at 01 if you want the full arc from dataset to chat UI. Jump to whichever number matches what you're curious about if you don't.
2. **Read the module's README first**, top to bottom, before running anything. Each one explains *why*, not just *how* — the goal is to walk away with a mental model you can reapply to a completely different project, not a script you copy-pasted.
3. **Run it. Break it. Change the constraint, the dataset, the model size, the prompt.** The fastest way to actually understand a fine-tuning run is to watch what happens when you make it worse on purpose.
4. **Bring what you build to the next Classrooms session** at TinkerSpace, Kochi. Proof of work beats a slide deck every time.

## Prerequisites

You should be comfortable in a terminal and know basic Python. Nothing else is assumed — each module explains the tools it introduces (MLX, LoRA, Ollama, FastAPI, etc.) from first principles. Hardware requirements differ per module and are called out at the top of each one (e.g., Module 1 needs Apple Silicon; Module 2 needs an NVIDIA GPU or a free Colab account).

## License

Code in this repository is released under the [MIT License](LICENSE). Where a module bundles a dataset or model card with its own license (e.g. the Gemmaiku dataset is Apache-2.0), that's noted in the module's own README.

---

*Run by [Vishnu Dileesh](https://www.vishnudileesh.com/) at [TinkerHub](https://tinkerhub.org), TinkerSpace Kochi. Questions, corrections, and pull requests welcome — this repo is itself a build-to-learn project.*

[Website](https://www.vishnudileesh.com/) &middot; [X / Twitter](https://x.com/vi_c0de) &middot; [LinkedIn](https://www.linkedin.com/in/vishnu-dileesh/)
