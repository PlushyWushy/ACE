#!/usr/bin/env python3
"""
Plot per-episode average reward (with std shading) across classic runs in icarussecondpaper.
"""
from pathlib import Path
import csv
import numpy as np
import matplotlib.pyplot as plt

RUNS_DIR = Path("icarussecondpaper/runs")
OUT_PNG = Path("icarussecondpaper/classic_per_episode_avg_reward.png")
OUT_STATS = Path("icarussecondpaper/classic_per_episode_reward_stats.csv")

def read_reward(csv_path: Path):
    vals = []
    with csv_path.open() as fh:
        reader = csv.DictReader(fh)
        if 'reward' not in (reader.fieldnames or []):
            raise ValueError(f"CSV {csv_path} missing 'reward' column")
        for row in reader:
            try:
                vals.append(float(row['reward']))
            except Exception:
                vals.append(0.0)
    return np.array(vals, dtype=float)

def main():
    runs = []
    for d in sorted(RUNS_DIR.iterdir(), key=lambda p: p.name):
        if not d.is_dir() or not d.name.endswith('_classic'):
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
            rewards = read_reward(chosen)
        except Exception:
            continue
        runs.append({'run': d.name, 'rewards': rewards})

    if not runs:
        print('No classic second paper runs found.')
        return

    lengths = [r['rewards'].size for r in runs]
    min_len = min(lengths)
    data = np.stack([r['rewards'][:min_len] for r in runs], axis=0)

    mean = data.mean(axis=0)
    std = data.std(axis=0)
    episodes = np.arange(min_len)

    plt.figure(figsize=(10,6))
    plt.plot(episodes, mean, color='tab:orange', lw=2, label='Mean reward')
    plt.fill_between(episodes, mean - std, mean + std, color='tab:orange', alpha=0.25, label='±1 std')
    
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Classic - Per-episode average reward across seeds (±STD) - Cartpole')
    plt.grid(True, alpha=0.3)
    plt.legend()

    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT_PNG, dpi=150, bbox_inches='tight')
    print(f'Saved plot to {OUT_PNG}')

    with OUT_STATS.open('w', newline='') as fh:
        writer = csv.writer(fh)
        writer.writerow(['episode','mean_reward','std_reward'])
        for ep, m, s in zip(episodes, mean, std):
            writer.writerow([int(ep), f'{m:.6f}', f'{s:.6f}'])
    print(f'Saved stats to {OUT_STATS}')

if __name__ == '__main__':
    main()
