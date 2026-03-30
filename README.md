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
