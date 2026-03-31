#!/bin/bash
# CEA-MI attack against embedding-based memory agent (MemGPT comparison)
source activate nanobot
cd /bigtemp/trv3px/cea_mi

# Step 1: Set up embedding memory (one-time, ~5 min)
echo "=== Setting up embedding memory agent ==="
pip install sentence-transformers 2>&1 | tail -1
python3 memgpt_target/setup_memgpt.py standalone --dataset /bigtemp/trv3px/benchmark_v2_dataset.json
echo "=== Setup done ==="

# Step 2: Run attacks at all access levels
for ACCESS in blackbox graybox whitebox; do
    echo "=== Starting MemGPT ${ACCESS} ==="
    python3 memgpt_target/memgpt_attack.py --access ${ACCESS} --num-facts 30 --seed 42 \
        --memory-file memgpt_target/memgpt_memories.json \
        2>&1 | tee /bigtemp/trv3px/attack_memgpt_${ACCESS}.log
    echo "=== ${ACCESS} done ==="
done

echo "=== ALL MEMGPT ATTACKS DONE ==="
