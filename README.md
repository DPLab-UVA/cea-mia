# CEA-MIA

Contrastive Evidence Accumulating Membership Inference Attack

## Status

This repository contains the research code for CEA-MIA against a memory-augmented
`nanobot` agent. The attack code assumes access to:

- a running chat-completions backend
- a local `nanobot` checkout
- a populated nanobot memory database

Those external dependencies are not vendored in this repository, so you will need
to point the code at your own environment.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## vLLM Backend

This project uses vLLM to serve a local LLM as the chat-completions backend.

### Start vLLM Server (Single GPU)

```bash
# Using Qwen2.5-7B-Instruct (recommended for single GPU)
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-7B-Instruct \
    --tensor-parallel-size 1 \
    --host 0.0.0.0 \
    --port 8000
```

### Configure Environment

Create a `.env` file or export these variables:

```bash
export HF_HOME=/path/to/hf_cache              # HuggingFace cache directory
export CEA_MI_API_BASE=http://localhost:8000/v1
export CEA_MI_MODEL=Qwen/Qwen2.5-7B-Instruct
```

Then source it before running:

```bash
source .env
```

### Multi-GPU Setup (Optional)

For larger models (e.g., 72B), use tensor parallelism across multiple GPUs:

```bash
# Example: 3x A6000 GPUs for a 72B model
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-72B-Instruct \
    --tensor-parallel-size 3 \
    --host 0.0.0.0 \
    --port 8000
```

Note: The number of attention heads must be divisible by `tensor-parallel-size`.

## Portable defaults

By default, generated artifacts now stay inside the repository instead of writing
to an author-specific `/bigtemp/...` path:

- datasets: `./data`
- evaluation outputs: `./results`
- nanobot project root: this repository directory

You can override any of the runtime paths with environment variables:

- `CEA_MI_DATA_DIR`
- `CEA_MI_OUTPUT_DIR`
- `CEA_MI_NANOBOT_PROJECT`
- `CEA_MI_NANOBOT_DB_PATH`
- `CEA_MI_DATASET`
- `CEA_MI_API_BASE`
- `CEA_MI_API_KEY`
- `CEA_MI_MODEL`

Example:

```bash
export CEA_MI_NANOBOT_PROJECT=/path/to/nanobot
export CEA_MI_NANOBOT_DB_PATH=~/.nanobot/memory/pmc.db
export CEA_MI_API_BASE=http://localhost:8000/v1
```

## Quick checks

Run the portability regression tests:

```bash
python3 -m unittest discover -s tests -v
```

Run the synthetic-memory experiment:

```bash
python3 main.py --access blackbox --members 10 --nonmembers 10
```

Run the natural-memory attack with an explicit dataset path:

```bash
python3 natural_attack.py \
  --access blackbox \
  --num-facts 30 \
  --dataset /path/to/benchmark_v2_dataset.json
```

Or export `CEA_MI_DATASET=/path/to/benchmark_v2_dataset.json` before using
`run_all_attacks_v3.sh`.

## Current limitation

`nanobot` is still an external dependency. If its Python package is not importable,
the attack entry points will fail until `CEA_MI_NANOBOT_PROJECT` is pointed at a
checkout that contains the `nanobot` module.
