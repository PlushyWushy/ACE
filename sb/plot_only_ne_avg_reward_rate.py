#!/usr/bin/env python3
from pathlib import Path
import csv
import numpy as np
import matplotlib.pyplot as plt

# Adjusted for sb_normal/runs
BASE_DIR = Path("/Users/chairstands/Desktop/Icarus/sb")
RUNS_DIR = BASE_DIR / "sb_normal/runs"
OUT_PNG = BASE_DIR / "sb_normal/only_ne_avg_reward_rate.png"
WINDOW = 50

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
    runs = []
    for d in sorted(RUNS_DIR.iterdir()):
        if d.is_dir() and d.name.endswith('_only_ne'):
            csv_file = d / "data.csv"
            if not csv_file.exists(): continue
            iso = read_is_optimal(csv_file)
            if iso is None: continue
            rates = windowed_reward_rate(iso, WINDOW)
            if rates.size > 0:
                runs.append(rates)

    if not runs:
        print("No only_ne runs found.")
        return

    min_len = min(len(r) for r in runs)
    data = np.stack([r[:min_len] for r in runs], axis=0)
    mean = data.mean(axis=0)
    std = data.std(axis=0)
    episodes = np.arange(1, min_len + 1) * WINDOW

    plt.figure(figsize=(10, 6))
    plt.plot(episodes, mean, color='tab:green', lw=2, label='Mean reward rate')
    plt.fill_between(episodes, mean - std, mean + std, color='tab:green', alpha=0.25, label='±1 std')
    plt.axvline(x=10000, color='red', linestyle='--', alpha=0.5, label='Switch')
    plt.xlabel('Episode')
    plt.ylabel('Percent Optimal (%)')
    plt.title('Only NE Modulation - Average Reward Rate across seeds (±STD)')
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 105)
    plt.legend()
    plt.savefig(OUT_PNG, dpi=150, bbox_inches='tight')
    print(f"Saved plot to {OUT_PNG}")

if __name__ == "__main__":
    main()
