#!/bin/bash

PROFILES=("sinusoidal" "sawtooth" "exponential" "stochastic" "plateau_spike")
EPISODES=15000

echo "Starting Multi-Profile LR Experiments..."

for profile in "${PROFILES[@]}"; do
    echo "Running profile: $profile in background..."
    python spike.py --profile "$profile" --episodes $EPISODES > "cartpole/runs/spike_${profile}_log.txt" 2>&1 &
done

echo "All experiments launched. Use 'jobs' to see status."
echo "Logs are available in cartpole/runs/spike_{profile}_log.txt"
