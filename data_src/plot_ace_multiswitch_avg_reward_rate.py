#!/usr/bin/env python3
"""
Plot per-episode average reward rate across ACE (icarus) multiswitch runs.
"""
from pathlib import Path
import csv
import numpy as np
import matplotlib.pyplot as plt

RUNS_DIR = Path("sb_multiswitch/runs")
OUT_PNG = Path("sb_multiswitch/ACE_avg_reward_rate.png")
OUT_STATS = Path("sb_multiswitch/ACE_reward_rate_stats.csv")
WINDOW = 50

def read_is_optimal(csv_path: Path):
    vals = []
    with csv_path.open() as fh:
        reader = csv.DictReader(fh)
        if 'is_optimal' not in (reader.fieldnames or []):
            raise ValueError(f"CSV {csv_path} missing 'is_optimal' column")
        for row in reader:
            try:
                vals.append(float(row['is_optimal']))
            except Exception:
                vals.append(0.0)
    return np.array(vals, dtype=float)

def windowed_reward_rate(arr: np.ndarray, window=WINDOW):
    n = len(arr)
    m = (n // window) * window
    if m == 0:
        return np.array([])
    arr = arr[:m]
    arr = arr.reshape(-1, window)
    rates = arr.mean(axis=1) * 100.0
    return rates

def main():
    runs = []
    for d in sorted(RUNS_DIR.iterdir(), key=lambda p: p.name):
        if not d.is_dir() or 'icarus' not in d.name:
            continue
        csvs = list(d.glob('*.csv'))
        if not csvs:
            continue
        chosen = csvs[0]
        for c in csvs:
            if d.name in c.name:
                chosen = c
                break
        try:
            iso = read_is_optimal(chosen)
        except Exception:
            continue
        rates = windowed_reward_rate(iso, WINDOW)
        if rates.size == 0:
            continue
        runs.append({'run': d.name, 'rates': rates})

    if not runs:
        print('No ACE (icarus) multiswitch runs found.')
        return

    lengths = [r['rates'].size for r in runs]
    min_len = min(lengths)
    data = np.stack([r['rates'][:min_len] for r in runs], axis=0)

    mean = data.mean(axis=0)
    std = data.std(axis=0)
    episodes = np.arange(1, min_len+1) * WINDOW

    plt.figure(figsize=(10,6))
    plt.plot(episodes, mean, color='tab:blue', lw=2, label='Mean reward rate')
    plt.fill_between(episodes, np.clip(mean - std, 0, 100), np.clip(mean + std, 0, 100), color='tab:blue', alpha=0.25, label='±1 std')
    
    # Switch points
    plt.axvline(x=15000, color='tab:red', linestyle='--', label='Switch 1')
    plt.axvline(x=30000, color='tab:orange', linestyle='--', label='Switch 2')
    plt.axvline(x=45000, color='tab:purple', linestyle='--', label='Switch 3')
    
    plt.xlabel('Episode')
    plt.ylabel('Percent Optimal (%)')
    plt.title('ACE - Multi-Switch Bandit Average Reward Rate (±STD)')
    plt.grid(True, alpha=0.3)
    plt.ylim(-5, 105)
    plt.legend()

    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT_PNG, dpi=150, bbox_inches='tight')
    print(f'Saved plot to {OUT_PNG}')

    with OUT_STATS.open('w', newline='') as fh:
        writer = csv.writer(fh)
        writer.writerow(['window_end_episode','mean_reward_rate','std_reward_rate'])
        for ep, m, s in zip(episodes, mean, std):
            writer.writerow([ep, f'{m:.6f}', f'{s:.6f}'])
    print(f'Saved stats to {OUT_STATS}')

if __name__ == '__main__':
    main()
