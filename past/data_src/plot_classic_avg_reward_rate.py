#!/usr/bin/env python3
"""
Plot average reward rate (with std shading) across all `*_classic` runs.
- Reads `switch_bandit_experiment/runs/*_classic/*_classic.csv`
- Expects a column `is_optimal` per episode (0/1)
- Computes reward rates over non-overlapping 50-episode windows
- Plots mean ± std across runs and saves PNG and a stats CSV
"""
from pathlib import Path
import csv
import numpy as np
import matplotlib.pyplot as plt

RUNS_DIR = Path("sb/sb_normal/runs")
OUT_PNG = Path("sb/sb_normal/classic_avg_reward_rate.png")
OUT_STATS = Path("sb/sb_normal/classic_reward_rate_stats.csv")
WINDOW = 50


def read_is_optimal(csv_path: Path):
    vals = []
    with csv_path.open() as fh:
        reader = csv.DictReader(fh)
        # prefer `is_optimal`, otherwise try to infer
        flds = reader.fieldnames or []
        if 'is_optimal' not in flds:
            # try to find a column likely representing optimal (e.g., second column named 'is_optimal' in original)
            # fallback: try 'reward' and convert (>0 -> 1) — but for classic we expect is_optimal
            if 'reward' in flds:
                for row in reader:
                    try:
                        r = float(row.get('reward', 0.0))
                        vals.append(1.0 if r > 0 else 0.0)
                    except Exception:
                        vals.append(0.0)
                return np.array(vals, dtype=float)
            else:
                raise ValueError(f"CSV {csv_path} missing 'is_optimal' and no 'reward' fallback")
        # normal path
        for row in reader:
            try:
                vals.append(float(row['is_optimal']))
            except Exception:
                # fallback if value is not numeric
                v = row.get('is_optimal', '0')
                vals.append(1.0 if v in ('1', 'True', 'true', 'yes') else 0.0)
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
        if not d.is_dir() or not d.name.endswith('_classic'):
            continue
        csvs = list(d.glob('*.csv'))
        if not csvs:
            print(f'No CSV in {d}, skipping')
            continue
        chosen = None
        for c in csvs:
            if d.name in c.name:
                chosen = c
                break
        if chosen is None:
            chosen = csvs[0]
        try:
            iso = read_is_optimal(chosen)
        except Exception as e:
            print(f'Error reading {chosen}: {e}')
            continue
        rates = windowed_reward_rate(iso, WINDOW)
        if rates.size == 0:
            print(f'{chosen} too short for windowing, skipping')
            continue
        runs.append({'run': d.name, 'seed': d.name.split('_')[0], 'rates': rates})

    if not runs:
        print('No classic runs found.')
        return

    lengths = [r['rates'].size for r in runs]
    min_len = min(lengths)
    if any(l != min_len for l in lengths):
        print(f'Warning: runs have different number of windows; truncating to {min_len} windows')
    data = np.stack([r['rates'][:min_len] for r in runs], axis=0)

    mean = data.mean(axis=0)
    std = data.std(axis=0)
    episodes = np.arange(1, min_len+1) * WINDOW

    plt.figure(figsize=(10,6))
    plt.plot(episodes, mean, color='tab:orange', lw=2, label='Mean reward rate')
    plt.fill_between(episodes, np.clip(mean - std, 0, 100), np.clip(mean + std, 0, 100), color='tab:orange', alpha=0.25, label='±1 std')
    plt.axvline(x=10000, color='tab:red', linestyle='--', label='Switch')
    plt.xlabel('Episode')
    plt.ylabel('Percent Optimal (%)')
    plt.title('Classic - Average Reward Rate across seeds (±STD)')
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
