#!/bin/bash
# CEA-MI v4 attack — confirmation probes + calibration
source activate nanobot
cd /bigtemp/trv3px/cea_mi

echo "=== Starting blackbox v4 ==="
python3 natural_attack.py --access blackbox --num-facts 30 --seed 42 --db ~/.nanobot/memory/pmc.db 2>&1 | tee /bigtemp/trv3px/attack_v4_blackbox.log
echo "=== Blackbox done ==="

echo "=== Starting graybox v4 ==="
python3 natural_attack.py --access graybox --num-facts 30 --seed 42 --db ~/.nanobot/memory/pmc.db 2>&1 | tee /bigtemp/trv3px/attack_v4_graybox.log
echo "=== Graybox done ==="

echo "=== Starting whitebox v4 ==="
python3 natural_attack.py --access whitebox --num-facts 30 --seed 42 --db ~/.nanobot/memory/pmc.db 2>&1 | tee /bigtemp/trv3px/attack_v4_whitebox.log
echo "=== Whitebox done ==="

echo "=== ALL V4 ATTACKS DONE ==="
