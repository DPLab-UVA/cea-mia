#!/bin/bash
source activate nanobot
cd /bigtemp/trv3px/cea_mi

echo "=== Starting blackbox v3 ==="
python3 natural_attack.py --access blackbox --num-facts 30 --seed 42 --db ~/.nanobot/memory/pmc.db 2>&1 | tee /bigtemp/trv3px/attack_v3_blackbox.log
echo "=== Blackbox done ==="

echo "=== Starting graybox v3 ==="
python3 natural_attack.py --access graybox --num-facts 30 --seed 42 --db ~/.nanobot/memory/pmc.db 2>&1 | tee /bigtemp/trv3px/attack_v3_graybox.log
echo "=== Graybox done ==="

echo "=== Starting whitebox v3 ==="
python3 natural_attack.py --access whitebox --num-facts 30 --seed 42 --db ~/.nanobot/memory/pmc.db 2>&1 | tee /bigtemp/trv3px/attack_v3_whitebox.log
echo "=== ALL DONE ==="
