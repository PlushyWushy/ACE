#!/usr/bin/env python3
"""
Plot per-episode average reward (with std shading) across cartpole runs.
- Scans `cartpole/runs/*_classic/*` and `cartpole/runs/*_critic_ach_fastslow/*`
- Reads `reward` column per episode, aligns runs by truncating to minimum length,
- Computes mean and std across runs for each episode, plots mean ± std shading
- Saves PNG and a stats CSV for each group.
"""
from pathlib import Path
import csv
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path("cartpole")
RUNS_DIR = ROOT / "runs"

GROUPS = ["_classic", "_critic_ach_fastslow"]


def read_reward(csv_path: Path):
    vals = []
    with csv_path.open() as fh:
        reader = csv.DictReader(fh)
        flds = reader.fieldnames or []
        if 'reward' not in flds:
            raise ValueError(f"CSV {csv_path} missing 'reward' column")
        for row in reader:
            try:
                vals.append(float(row.get('reward', 0.0)))
            except Exception:
                vals.append(0.0)
    return np.array(vals, dtype=float)


def process_group(suffix: str):
    runs = []
    for d in sorted(RUNS_DIR.iterdir(), key=lambda p: p.name):
        if not d.is_dir() or not d.name.endswith(suffix):
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
            rewards = read_reward(chosen)
        except Exception as e:
            print(f'Error reading {chosen}: {e}')
            continue
        runs.append({'run': d.name, 'rewards': rewards})

    if not runs:
        print(f'No runs found for suffix {suffix}')
        return

    lengths = [r['rewards'].size for r in runs]
    min_len = min(lengths)
    if any(l != min_len for l in lengths):
        print(f'Warning: runs in {suffix} have different lengths; truncating to {min_len} episodes')

    data = np.stack([r['rewards'][:min_len] for r in runs], axis=0)
    mean = data.mean(axis=0)
    std = data.std(axis=0)

    episodes = np.arange(min_len)

    # output paths
    safe_suffix = suffix.lstrip('_')
    # rename critic_ach_fastslow to Icarus for output
    display_name = 'Icarus' if safe_suffix == 'critic_ach_fastslow' else safe_suffix
    out_png = ROOT / f"{display_name.lower()}_per_episode_avg_reward.png"
    out_csv = ROOT / f"{display_name.lower()}_per_episode_reward_stats.csv"

    color = 'tab:orange' if suffix == '_classic' else 'tab:blue'
    plt.figure(figsize=(10,6))
    plt.plot(episodes, mean, color=color, lw=2, label='Mean reward')
    plt.fill_between(episodes, mean - std, mean + std, color=color, alpha=0.25, label='±1 std')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    display_name = 'Icarus' if safe_suffix == 'critic_ach_fastslow' else safe_suffix
    plt.title(f'{display_name} - Per-episode average reward across seeds (±STD)')
    plt.grid(True, alpha=0.3)
    plt.legend()

    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved plot to {out_png}')

    with out_csv.open('w', newline='') as fh:
        writer = csv.writer(fh)
        writer.writerow(['episode','mean_reward','std_reward'])
        for ep, m, s in zip(episodes, mean, std):
            writer.writerow([int(ep), f'{m:.6f}', f'{s:.6f}'])
    print(f'Saved stats to {out_csv}')


def main():
    for g in GROUPS:
        process_group(g)


if __name__ == '__main__':
    main()
