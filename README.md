# Multi-Recall Memory MIA (MRMMIA)

Multi-Recall Memory MIA (MRMMIA) is a membership inference attack for memory-augmented agents. Given a candidate memory unit, MRMMIA generates multiple direct recall probes, queries the target agent, and aggregates response evidence, gray-box logprob evidence when available, and white-box memory-retrieval evidence when available.

This repository contains the current MRMMIA MemoryDataset attack pipeline, baseline comparisons, and older synthetic or target-specific experiments. The main maintained entry points are:

- `natural_attack.py`: main MRMMIA implementation. It injects each user's member memories, probes member and non-member units with multiple recall probes, runs whitebox once, and writes derived blackbox, graybox, and whitebox outputs.
- `baselines/baseline_attacks.py`: comparison baselines such as naive single-query, loss, Min-K%, reference model, multi-contrastive, direct multi-probe, recall-no-reason, and multi-judge.
- `run_all_natural_attacks.sh` and `run_baseline_attacks.sh`: convenience wrappers for common multi-dataset and multi-baseline runs, including light and SDK memory backends. `baselines/run_baselines.sh` is a compact SDK-aware batch helper for baseline runs.

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

- `mem0`: Mem0-style memory. By default this uses the lightweight local JSON plus sentence-transformers backend; with `--memory-backend sdk`, it uses the Mem0 SDK with a local Qdrant path.
- `memgpt`: MemGPT/Letta-style memory. By default this uses the lightweight local JSON plus sentence-transformers backend; with `--memory-backend sdk`, it uses a local Letta server and Letta archival memory.
- `nanobot`: external nanobot memory implementation. The nanobot package is not vendored here; set `CEA_MI_NANOBOT_PROJECT` if it is not importable from your environment.

For light `mem0` and light `memgpt`, the working memory file is created under the run output directory unless you pass `--memory-file`. SDK backends use `--memory-store-path`/`CEA_MI_MEMORY_STORE_PATH` when you want an explicit local store location. For `nanobot`, the attack creates isolated SQLite databases so runs do not mutate the configured source database.

### Memory Backend Modes

`natural_attack.py` and `baselines/baseline_attacks.py` accept `--memory-backend light|sdk|full` for `mem0` and `memgpt`. `full` is normalized to `sdk`.

**Light backend**

The light backend is a controlled local retrieval baseline:

```text
member memory units
  -> sentence-transformers embeddings
  -> local JSON working memory
  -> cosine similarity recall, thresholded and top-k
  -> recalled memories are inserted into the vLLM prompt
```

The light `mem0` and light `memgpt` paths are intentionally simple and differ mostly in wrapper/prompt wording, not in a full product runtime.

**Mem0 SDK backend**

The Mem0 SDK path uses the Mem0 package for memory storage and retrieval:

```text
prepare_user_memory()
  -> reset Mem0 Memory
  -> add selected member units as raw statements with metadata
  -> store embeddings in local Qdrant path

query()
  -> memory.search(query, user_id=...)
  -> format top-k recalled memories
  -> call the configured vLLM/OpenAI-compatible chat endpoint
```

During attack queries this adapter is read-only: it searches memory and calls vLLM, but it does not add probe text or model responses back into Mem0 memory. By default insertion uses `infer=false`, so Mem0 stores the original sampled memory statement rather than an LLM-rewritten fact. Useful controls:

- `CEA_MI_MEMORY_BACKEND=sdk`: select SDK mode from shell wrappers.
- `CEA_MI_MEMORY_STORE_PATH`: optional explicit local SDK store path.
- `CEA_MI_MEM0_INFER=false`: default in code and shell wrappers; avoids LLM-based memory inference during insertion.
- `CEA_MI_MEM0_EMBEDDER_DEVICE=cpu`: default; keeps the Mem0 embedding model off the vLLM GPU.
- `MEM0_TELEMETRY=false`: disables Mem0 telemetry.

Example:

```bash
nohup env MEM0_TELEMETRY=false CEA_MI_MEMORY_BACKEND=sdk \
  ./run_all_natural_attacks.sh dplab04 8001 locomo mem0 auto \
  > logs/run_mem0_sdk_locomo_dplab04_port8001.log 2>&1 &
```

**MemGPT/Letta SDK backend**

The MemGPT SDK path uses Letta archival memory. Start the local embedding server and Letta server before running experiments:

```bash
./memgpt_target/start_embedding_server.sh dplab05 8290
./memgpt_target/start_letta_server.sh dplab05 8283
curl http://127.0.0.1:8290/health
curl http://127.0.0.1:8283/v1/health/
```

The default query mode is `readonly`:

```text
prepare_user_memory()
  -> create a temporary Letta agent for this user
  -> insert selected member units as raw passages into Letta archival memory

query(), readonly mode
  -> search Letta archival memory
  -> format top-k recalled passages
  -> call the configured vLLM/OpenAI-compatible chat endpoint
```

This treats Letta as the memory backend while keeping probe queries isolated from Letta's full agent runtime. The query stage does not call Letta `messages.create()` and does not let probe text write back into memory.

By default, readonly mode uses `CEA_MI_MEMGPT_RECALL_BACKEND=letta_native`. This calls Letta's own embedding-based archival retrieval path, equivalent to the internal `archival_memory_search` workflow:

```text
agent_manager.list_passages(
  query_text=probe,
  embed_query=True,
  embedding_config=agent_state.embedding_config,
)
```

This is different from the public `passages.list(search=...)` API, which is only a substring text filter. Native recall requires the experiment process to see the same `LETTA_DIR` as the Letta server. The start script writes a sourceable env file:

```bash
source logs/letta_server_dplab05_port8283.env
```

Other recall backends are available for ablations: `CEA_MI_MEMGPT_RECALL_BACKEND=letta_text` uses the public text-filter API, and `CEA_MI_MEMGPT_RECALL_BACKEND=shadow` uses a local semantic shadow index for debugging only.

Set `CEA_MI_MEMGPT_QUERY_MODE=agent` only when you want the full Letta agent runtime as an ablation:

```text
query(), agent mode
  -> Letta messages.create()/send_message()
  -> Letta decides tool use, recall, message buffer behavior, and writes
```

In `agent` mode the wrapper defaults `CEA_MI_TARGET_QUERY_CONCURRENCY=1` because local Letta/SQLite can lock under concurrent message writes. In `readonly` mode target-query concurrency is not forced by default.

Example:

```bash
nohup env CEA_MI_MEMORY_BACKEND=sdk CEA_MI_MEMGPT_QUERY_MODE=readonly \
  CEA_MI_MEMGPT_RECALL_BACKEND=letta_native \
  LETTA_BASE_URL=http://127.0.0.1:8283 \
  ./run_all_natural_attacks.sh dplab05 8001 locomo memgpt auto \
  > logs/run_memgpt_sdk_readonly_locomo_dplab05_port8001.log 2>&1 &
```

Whitebox scoring uses recalled memory snippets plus the configured response/logprob evidence. The LLM memory judge is intentionally strict: high memory-support scores require direct support from the recalled memory text itself, with the same subject and same attribute/value.

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
results/<dataset>_<target>_<scorer>[_sdk|_sdk-readonly|_sdk-agent]_k<direct_probe_k>_seed<seed>[_def-system_prompt][_rN]/<access>/
```

Both Python entry points and the shell wrappers support `--rounds` / trailing `rounds`. Round 1 uses `seed`, round 2 uses `seed+1`, and so on. When `rounds>1`, output directories include a suffix such as `_r3`. JSON outputs are round-wrapped at the top level:

```json
{
  "1": {"seed": 42},
  "2": {"seed": 43}
}
```

Each access directory contains round-wrapped `report.json`, `predictions.json`, `comparison.json`, `meta.json`, and `per_user_metrics.json`.

Both natural and baseline entry points also support `--defense none|system_prompt`. The default `none` preserves the original target prompt and output paths. `system_prompt` appends a privacy instruction to the target agent prompt and adds `_def-system_prompt` to output directories. In the shell wrappers, pass it after `rounds`, for example:

```bash
./run_all_natural_attacks.sh dplab04 8000 locomo mem0 auto sdk 1 system_prompt
./run_baseline_attacks.sh dplab04 8001 locomo mem0 multi_judge auto sdk 1 system_prompt
```

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

For SDK backends, pass `--memory-backend sdk` to the Python entry point, use the `memory_backend` positional argument in the root wrapper, or set `CEA_MI_MEMORY_BACKEND=sdk`:

```bash
python3 baselines/baseline_attacks.py \
  --target mem0 \
  --dataset locomo \
  --baseline naive \
  --memory-backend sdk

nohup env MEM0_TELEMETRY=false \
  ./run_baseline_attacks.sh dplab04 8001 locomo mem0 all auto sdk 3 \
  > logs/run_baseline_mem0_sdk_locomo_dplab04_port8001.log 2>&1 &

CEA_MI_MEMORY_BACKEND=sdk CEA_MI_DATASET=locomo CEA_MI_NUM_FACTS=none \
  ./baselines/run_baselines.sh 3 system_prompt
```

Baseline access policy is handled internally: loss, Min-K%, and reference run with graybox information; the other baselines run whitebox once and write derived access-level outputs.

## Troubleshooting

If an attack exits with missing LLM configuration, set both `CEA_MI_API_BASE` and `CEA_MI_MODEL`. If nanobot is not importable, install it in the active environment or set `CEA_MI_NANOBOT_PROJECT` to a checkout containing the package. If you want to keep large generated files out of git, use the ignored `results/` and `logs/` directories.
