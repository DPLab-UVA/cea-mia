#!/bin/bash
# Baseline attacks: Naive (blackbox), Min-K% (graybox, Shi'24), Reference Model (whitebox)
source activate /bigtemp/trv3px/conda_envs/memgpt2
cd /bigtemp/trv3px/cea_mi

for TARGET in memgpt mem0; do
    if [ "$TARGET" = "memgpt" ]; then
        MEM="memgpt_memories.json"
    else
        MEM="mem0_memories.json"
    fi

    echo "=== ${TARGET} Blackbox: Naive Single Query ==="
    python3 baselines/baseline_attacks.py --target ${TARGET} --access blackbox --num-facts 30 --seed 42 \
        --memory-file ${MEM} --baseline naive \
        2>&1 | tee /bigtemp/trv3px/baselines_${TARGET}_blackbox.log

    echo "=== ${TARGET} Graybox: Min-K% Prob (Shi'24) ==="
    python3 baselines/baseline_attacks.py --target ${TARGET} --access graybox --num-facts 30 --seed 42 \
        --memory-file ${MEM} --baseline mink \
        2>&1 | tee /bigtemp/trv3px/baselines_${TARGET}_graybox.log

    echo "=== ${TARGET} Whitebox: Reference Model (Carlini'22) ==="
    python3 baselines/baseline_attacks.py --target ${TARGET} --access graybox --num-facts 30 --seed 42 \
        --memory-file ${MEM} --baseline reference \
        2>&1 | tee /bigtemp/trv3px/baselines_${TARGET}_whitebox.log
done

# Nanobot
echo "=== Nanobot Blackbox: Naive Single Query ==="
python3 baselines/baseline_attacks.py --target nanobot --access blackbox --num-facts 30 --seed 42 \
    --db ~/.nanobot/memory/pmc.db --baseline naive \
    2>&1 | tee /bigtemp/trv3px/baselines_nanobot_blackbox.log

echo "=== Nanobot Graybox: Min-K% Prob (Shi'24) ==="
python3 baselines/baseline_attacks.py --target nanobot --access graybox --num-facts 30 --seed 42 \
    --db ~/.nanobot/memory/pmc.db --baseline mink \
    2>&1 | tee /bigtemp/trv3px/baselines_nanobot_graybox.log

echo "=== Nanobot Whitebox: Reference Model (Carlini'22) ==="
python3 baselines/baseline_attacks.py --target nanobot --access graybox --num-facts 30 --seed 42 \
    --db ~/.nanobot/memory/pmc.db --baseline reference \
    2>&1 | tee /bigtemp/trv3px/baselines_nanobot_whitebox.log

echo "=== ALL BASELINES DONE ==="
