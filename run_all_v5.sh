#!/bin/bash
# CEA-MI v5 — Fisher LDA learned weights + early stop fix
# Runs all 3 targets x 3 access levels = 9 experiments

set -e

echo "=============================================="
echo "CEA-MI v5: Learned Weights + Early Stop Fix"
echo "=============================================="

# --- Nanobot (keyword-based) ---
echo "=== Nanobot Blackbox ==="
source activate nanobot
cd /bigtemp/trv3px/cea_mi
python3 natural_attack.py --access blackbox --num-facts 30 --seed 42 --db ~/.nanobot/memory/pmc.db 2>&1 | tee /bigtemp/trv3px/attack_v5_nanobot_blackbox.log
echo "=== Nanobot Blackbox DONE ==="

echo "=== Nanobot Graybox ==="
python3 natural_attack.py --access graybox --num-facts 30 --seed 42 --db ~/.nanobot/memory/pmc.db 2>&1 | tee /bigtemp/trv3px/attack_v5_nanobot_graybox.log
echo "=== Nanobot Graybox DONE ==="

echo "=== Nanobot Whitebox ==="
python3 natural_attack.py --access whitebox --num-facts 30 --seed 42 --db ~/.nanobot/memory/pmc.db 2>&1 | tee /bigtemp/trv3px/attack_v5_nanobot_whitebox.log
echo "=== Nanobot Whitebox DONE ==="

# --- MemGPT (embedding-based) ---
echo "=== MemGPT Blackbox ==="
source activate /bigtemp/trv3px/conda_envs/memgpt2
cd /bigtemp/trv3px/cea_mi
python3 memgpt_target/memgpt_attack.py --access blackbox --num-facts 30 --seed 42 \
    --memory-file memgpt_target/memgpt_memories.json 2>&1 | tee /bigtemp/trv3px/attack_v5_memgpt_blackbox.log
echo "=== MemGPT Blackbox DONE ==="

echo "=== MemGPT Graybox ==="
python3 memgpt_target/memgpt_attack.py --access graybox --num-facts 30 --seed 42 \
    --memory-file memgpt_target/memgpt_memories.json 2>&1 | tee /bigtemp/trv3px/attack_v5_memgpt_graybox.log
echo "=== MemGPT Graybox DONE ==="

echo "=== MemGPT Whitebox ==="
python3 memgpt_target/memgpt_attack.py --access whitebox --num-facts 30 --seed 42 \
    --memory-file memgpt_target/memgpt_memories.json 2>&1 | tee /bigtemp/trv3px/attack_v5_memgpt_whitebox.log
echo "=== MemGPT Whitebox DONE ==="

# --- Mem0 (embedding-based) ---
echo "=== Mem0 Blackbox ==="
python3 mem0_target/mem0_attack.py --access blackbox --num-facts 30 --seed 42 \
    --memory-file mem0_memories.json 2>&1 | tee /bigtemp/trv3px/attack_v5_mem0_blackbox.log
echo "=== Mem0 Blackbox DONE ==="

echo "=== Mem0 Graybox ==="
python3 mem0_target/mem0_attack.py --access graybox --num-facts 30 --seed 42 \
    --memory-file mem0_memories.json 2>&1 | tee /bigtemp/trv3px/attack_v5_mem0_graybox.log
echo "=== Mem0 Graybox DONE ==="

echo "=== Mem0 Whitebox ==="
python3 mem0_target/mem0_attack.py --access whitebox --num-facts 30 --seed 42 \
    --memory-file mem0_memories.json 2>&1 | tee /bigtemp/trv3px/attack_v5_mem0_whitebox.log
echo "=== Mem0 Whitebox DONE ==="

echo "=============================================="
echo "ALL V5 EXPERIMENTS COMPLETE"
echo "=============================================="

# Print summary
echo ""
echo "=== RESULTS SUMMARY ==="
for f in /bigtemp/trv3px/attack_v5_*.log; do
    name=$(basename "$f" .log | sed 's/attack_v5_//')
    roc=$(grep "ROC-AUC" "$f" | tail -1 | grep -oP '[\d.]+' | head -1)
    acc=$(grep "Accuracy" "$f" | tail -1 | grep -oP '[\d.]+' | head -1)
    echo "$name: ROC-AUC=$roc Accuracy=$acc"
done
