#!/bin/bash
# CEA-MI attack against Mem0-style embedding memory agent
source activate /bigtemp/trv3px/conda_envs/memgpt2
cd /bigtemp/trv3px/cea_mi

# Step 1: Build embedding memory (one-time, ~1 min)
echo "=== Setting up Mem0 memory agent ==="
python3 mem0_target/setup_mem0.py standalone --dataset /bigtemp/trv3px/benchmark_v2_dataset.json
echo "=== Setup done ==="

# Step 2: Run attacks at all access levels
for ACCESS in blackbox graybox whitebox; do
    echo "=== Starting Mem0 ${ACCESS} ==="
    python3 mem0_target/mem0_attack.py --access ${ACCESS} --num-facts 30 --seed 42 \
        --memory-file mem0_memories.json \
        2>&1 | tee /bigtemp/trv3px/attack_mem0_${ACCESS}.log
    echo "=== ${ACCESS} done ==="
done

echo "=== ALL MEM0 ATTACKS DONE ==="
