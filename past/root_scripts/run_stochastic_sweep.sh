#!/bin/bash

# Stochastic LR Hyperparameter Sweep (10 Variations)
EPISODES=15000

echo "Starting Stochastic LR Hyperparameter Sweep..."

# Variation 1: Low prob, long tail
python spike.py --profile stochastic --episodes $EPISODES --stochastic_prob 0.02 --anneal_episodes 400 --stochastic_magnitude 0.5 > cartpole/runs/stoc_1.log 2>&1 &

# Variation 2: Low prob, full magnitude
python spike.py --profile stochastic --episodes $EPISODES --stochastic_prob 0.02 --anneal_episodes 400 --stochastic_magnitude 1.0 > cartpole/runs/stoc_2.log 2>&1 &

# Variation 3: Base prob, short duration
python spike.py --profile stochastic --episodes $EPISODES --stochastic_prob 0.05 --anneal_episodes 200 --stochastic_magnitude 0.8 > cartpole/runs/stoc_3.log 2>&1 &

# Variation 4: Base prob, base duration (Baseline)
python spike.py --profile stochastic --episodes $EPISODES --stochastic_prob 0.05 --anneal_episodes 400 --stochastic_magnitude 0.8 > cartpole/runs/stoc_4.log 2>&1 &

# Variation 5: Base prob, long duration
python spike.py --profile stochastic --episodes $EPISODES --stochastic_prob 0.05 --anneal_episodes 800 --stochastic_magnitude 0.8 > cartpole/runs/stoc_5.log 2>&1 &

# Variation 6: High prob, low magnitude
python spike.py --profile stochastic --episodes $EPISODES --stochastic_prob 0.10 --anneal_episodes 400 --stochastic_magnitude 0.5 > cartpole/runs/stoc_6.log 2>&1 &

# Variation 7: High prob, full magnitude
python spike.py --profile stochastic --episodes $EPISODES --stochastic_prob 0.10 --anneal_episodes 400 --stochastic_magnitude 1.0 > cartpole/runs/stoc_7.log 2>&1 &

# Variation 8: Very high prob
python spike.py --profile stochastic --episodes $EPISODES --stochastic_prob 0.20 --anneal_episodes 400 --stochastic_magnitude 0.8 > cartpole/runs/stoc_8.log 2>&1 &

# Variation 9: Base prob, very low magnitude spikes (weak resets)
python spike.py --profile stochastic --episodes $EPISODES --stochastic_prob 0.05 --anneal_episodes 400 --stochastic_magnitude 0.2 > cartpole/runs/stoc_9.log 2>&1 &

# Variation 10: High prob, short cycle, intense spikes
python spike.py --profile stochastic --episodes $EPISODES --stochastic_prob 0.10 --anneal_episodes 200 --stochastic_magnitude 1.0 > cartpole/runs/stoc_10.log 2>&1 &

echo "10 Stochastic experiments launched. Check cartpole/runs/stoc_*.log for output."
