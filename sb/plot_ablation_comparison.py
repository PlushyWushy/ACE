#!/usr/bin/env python3
from pathlib import Path
import csv
import numpy as np
import matplotlib.pyplot as plt
import os

BASE_DIR = Path("/Users/chairstands/Desktop/Icarus/sb")
RUNS_DIR = BASE_DIR / "sb_normal/runs"
OUT_PNG = BASE_DIR / "sb_normal/ablation_comparison.png"
WINDOW = 100 # Using a larger window for smoother comparison

def read_is_optimal(csv_path: Path):
    vals = []
    with csv_path.open() as fh:
        reader = csv.DictReader(fh)
        if 'is_optimal' not in (reader.fieldnames or []):
            return None
        for row in reader:
            try:
                vals.append(float(row['is_optimal']))
            except (ValueError, KeyError):
                vals.append(0.0)
    return np.array(vals, dtype=float)

def windowed_reward_rate(arr: np.ndarray, window=WINDOW):
    n = len(arr)
    m = (n // window) * window
    if m == 0: return np.array([])
    arr = arr[:m].reshape(-1, window)
    return arr.mean(axis=1) * 100.0

def main():
    if not RUNS_DIR.exists():
        print(f"Runs directory not found: {RUNS_DIR}")
        return

    type_data = {}
    
    for d in sorted(RUNS_DIR.iterdir(), key=lambda p: p.name):
        if not d.is_dir(): continue
        
        parts = d.name.split("_")
        if len(parts) < 2: continue
        rtype = "_".join(parts[1:])
        
        csv_file = d / "data.csv"
        if not csv_file.exists(): continue
        
        iso = read_is_optimal(csv_file)
        if iso is None: continue
        
        rates = windowed_reward_rate(iso, WINDOW)
        if rates.size == 0: continue
        
        if rtype not in type_data:
            type_data[rtype] = []
        type_data[rtype].append(rates)

    plt.figure(figsize=(12, 8))
    
    colors = {
        'icarus_upgraded': 'black',
        'classic': 'gray',
        'only_ne': 'tab:green',
        'only_ach': 'tab:red'
    }
    
    labels = {
        'icarus_upgraded': 'Icarus Upgraded (Both)',
        'classic': 'Classic (Fixed LR, Fixed NE)',
        'only_ne': 'Only NE Modulation (Fixed LR)',
        'only_ach': 'Only ACh Modulation (Fixed NE=1.5)'
    }

    # Order of plotting (background to foreground)
    order = ['classic', 'only_ach', 'only_ne', 'icarus_upgraded']
    
    for rtype in order:
        if rtype not in type_data: continue
        
        data = type_data[rtype]
        min_len = min(len(r) for r in data)
        data = np.stack([r[:min_len] for r in data], axis=0)
        
        mean = data.mean(axis=0)
        std = data.std(axis=0)
        episodes = np.arange(1, min_len + 1) * WINDOW
        
        color = colors.get(rtype, None)
        label = labels.get(rtype, rtype)
        
        plt.plot(episodes, mean, label=label, color=color, lw=2.5 if rtype == 'icarus_upgraded' else 2)
        plt.fill_between(episodes, mean - std, mean + std, color=color, alpha=0.15)

    plt.axvline(x=10000, color='gray', linestyle='--', alpha=0.7, label='Task Switch')
    plt.xlabel('Episode')
    plt.ylabel('Optimal Selection Rate (%)')
    plt.title('Ablation Study: NE and ACh Modulation in Switching Bandit')
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 105)
    plt.legend(loc='lower right')
    
    plt.savefig(OUT_PNG, dpi=150, bbox_inches='tight')
    print(f"Saved figure to {OUT_PNG}")

if __name__ == "__main__":
    main()
