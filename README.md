# Multi-Recall Memory MIA (MRMMIA)

Multi-Recall Memory MIA (MRMMIA) is a membership inference attack for memory-augmented agents. Given a candidate memory unit, MRMMIA generates multiple direct recall probes, queries the target agent, and aggregates response evidence, gray-box logprob evidence when available, and white-box memory-retrieval evidence when available.

This repository contains the current MRMMIA MemoryDataset attack pipeline, baseline comparisons, and older synthetic or target-specific experiments. The main maintained entry points are:

- `natural_attack.py`: main MRMMIA implementation. It injects each user's member memories, probes member and non-member units with multiple recall probes, runs whitebox once, and writes derived blackbox, graybox, and whitebox outputs.
- `baselines/baseline_attacks.py`: comparison baselines such as naive single-query, loss, Min-K%, reference model, multi-contrastive, direct multi-probe, recall-no-reason, and multi-judge.
- `run_all_natural_attacks.sh` and `run_baseline_attacks.sh`: convenience wrappers for common multi-dataset and multi-baseline runs.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The attack code calls an OpenAI-compatible chat completions endpoint. You can serve one with vLLM, or point the code at an existing compatible backend.

```bash
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-7B-Instruct \
  --tensor-parallel-size 1 \
  --host 0.0.0.0 \
  --port 8000
```

## Required Configuration

The repository intentionally does not embed a lab host, server name, or local model checkpoint path. Set the LLM endpoint and model explicitly before running attacks:

```bash
export CEA_MI_API_BASE=http://127.0.0.1:8000/v1
export CEA_MI_MODEL=Qwen/Qwen2.5-7B-Instruct
export CEA_MI_API_KEY=token-vllm
```

Useful optional environment variables:

- `CEA_MI_DATA_DIR`: dataset directory, default `./data`
- `CEA_MI_OUTPUT_DIR`: Python entry-point output directory, default `./results`
- `CEA_MI_RESULTS_DIR`: shell wrapper output directory, default `results`
- `CEA_MI_LOG_DIR`: shell wrapper log directory, default `logs`
- `CEA_MI_NANOBOT_PROJECT`: path to a checkout containing the `nanobot` package
- `CEA_MI_NANOBOT_DB_PATH`: nanobot SQLite memory database path
- `CEA_MI_SEED`, `CEA_MI_NUM_FACTS`, `CEA_MI_CONCURRENCY`, `CEA_MI_DIRECT_PROBE_K`

## Datasets

Prebuilt MemoryDataset files are included under `data/`:

- `data/perltqa_seed42.json`
- `data/locomo_seed42.json`
- `data/msc_seed42.json`

CLI dataset arguments accept either an alias such as `perltqa`, `locomo`, or `msc`, or an explicit JSON path. A MemoryDataset stores per-user member and non-member memory units. During an attack, only the selected user's member units are injected into the target memory. We also provide the code of preprocessing in `memory_extractor.py`.


## Targets

`natural_attack.py` and the baseline runner support three target adapters:

- `mem0`: lightweight Mem0-style embedding memory backed by local JSON plus sentence-transformers.
- `memgpt`: lightweight MemGPT/Letta-style embedding memory backed by local JSON plus sentence-transformers.
- `nanobot`: external nanobot memory implementation. The nanobot package is not vendored here; set `CEA_MI_NANOBOT_PROJECT` if it is not importable from your environment.

For `mem0` and `memgpt`, the working memory file is created under the run output directory unless you pass `--memory-file`. For `nanobot`, the attack creates isolated SQLite databases so runs do not mutate the configured source database.

## MRMMIA Attack

Run a small Mem0 MRMMIA attack:

```bash
python3 natural_attack.py \
  --target mem0 \
  --dataset perltqa \
  --num-facts 20 \
  --max-users 2 \
  --concurrency 40
```

Run all supported datasets with the shell wrapper:

```bash
./run_all_natural_attacks.sh all mem0 auto
```

Pass an API host and port positionally if you do not want to export `CEA_MI_API_BASE`:

```bash
./run_all_natural_attacks.sh api-host.example.edu 8001 perltqa mem0 20
```

Outputs are written under:

```text
results/<dataset>_<target>_<scorer>_k<direct_probe_k>_seed<seed>/<access>/
```

Each access directory contains `report.json`, `predictions.json`, `comparison.json`, `meta.json`, and `per_user_metrics.json`.

## Baselines

Run all baselines for one target and dataset:

```bash
python3 baselines/baseline_attacks.py \
  --target mem0 \
  --dataset perltqa \
  --baseline all \
  --num-facts 20 \
  --concurrency 10
```

Run selected baselines through the wrapper:

```bash
./run_baseline_attacks.sh perltqa mem0 naive,mink,reference 20
```

Baseline access policy is handled internally: loss, Min-K%, and reference run with graybox information; the other baselines run whitebox once and write derived access-level outputs.

## Troubleshooting

If an attack exits with missing LLM configuration, set both `CEA_MI_API_BASE` and `CEA_MI_MODEL`. If nanobot is not importable, install it in the active environment or set `CEA_MI_NANOBOT_PROJECT` to a checkout containing the package. If you want to keep large generated files out of git, use the ignored `results/` and `logs/` directories.
