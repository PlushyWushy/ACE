#!/usr/bin/env python3
"""
Plot per-episode average reward for NE and ACh successful CartPole ablations
separately on their own, producing exactly 2 standalone graphs:
- results/cartpole/only_ne_avg_reward_rate.png
- results/cartpole/only_ach_avg_reward_rate.png

And saves aggregation stats in:
- results/cartpole/ablation_comparison_stats.csv
"""

from pathlib import Path
import csv
import numpy as np
import matplotlib.pyplot as plt

# Paths
BASE_DIR = Path(__file__).resolve().parents[2] / "results"
NE_DIR = BASE_DIR / "cartpole/ne_only_ablation_runs"
ACH_DIR = BASE_DIR / "cartpole/ach_only_ablation_runs"

OUT_NE_PNG = BASE_DIR / "cartpole/only_ne_avg_reward_rate.png"
OUT_ACH_PNG = BASE_DIR / "cartpole/only_ach_avg_reward_rate.png"
OUT_CSV = BASE_DIR / "cartpole/ablation_comparison_stats.csv"

WINDOW = 100
SEEDS = list(range(1, 21))

def read_rewards(csv_path: Path):
    if not csv_path.exists():
        return None
    vals = []
    with csv_path.open() as fh:
        reader = csv.DictReader(fh)
        if 'reward' not in (reader.fieldnames or []):
            return None
        for row in reader:
            try:
                vals.append(float(row['reward']))
            except (ValueError, KeyError):
                vals.append(0.0)
    return np.array(vals, dtype=float)

def rolling_mean(arr: np.ndarray, window: int = WINDOW):
    if len(arr) < window:
        return arr
    return np.convolve(arr, np.ones(window) / window, mode='valid')

def plot_single_ablation(data, color, out_path, title):
    # Calculate statistics across runs per episode
    raw_mean = data.mean(axis=0)
    raw_std = data.std(axis=0)
    
    smoothed_mean = rolling_mean(raw_mean, WINDOW)
    smoothed_std = rolling_mean(raw_std, WINDOW)
    
    episodes = np.arange(WINDOW, len(smoothed_mean) + WINDOW)
    
    plt.figure(figsize=(10, 6))
    
    ax = plt.subplot(111)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Plot mean
    plt.plot(episodes, smoothed_mean, color=color, lw=2, label='Mean reward')
    
    # Plot standard deviation strictly clipped to [0, 500] for CartPole
    lower_bound = np.clip(smoothed_mean - smoothed_std, 0, 500)
    upper_bound = np.clip(smoothed_mean + smoothed_std, 0, 500)
    plt.fill_between(episodes, lower_bound, upper_bound, color=color, alpha=0.25, label='±1 std')
    
    plt.xlabel('Episode')
    plt.ylabel(f'Reward (smoothed over {WINDOW} episodes)')
    plt.title(title)
    
    plt.grid(True, linestyle='--', alpha=0.5, color='#e0e0e0')
    
    
    plt.xlim(WINDOW, 10000)
    plt.ylim(-10, 510)
    plt.axvline(x=5000, color='tab:red', linestyle='--', label='Switch')
    plt.legend(loc='lower right', frameon=True, facecolor='white', edgecolor='#e0e0e0', framealpha=0.9, fontsize=11)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved standalone plot to {out_path}")

def main():
    groups = {
        'ACE (Only NE)': {'dir': NE_DIR, 'suffix': '_flagship', 'file_suffix': '_flagship.csv', 'color': 'tab:green', 'title': 'Only NE Modulation - Average Reward across seeds (±STD) - Cartpole', 'out_png': OUT_NE_PNG},
        'ACE (Only ACh)': {'dir': ACH_DIR, 'suffix': '_flagship', 'file_suffix': '_flagship.csv', 'color': 'tab:red', 'title': 'Only ACh Modulation - Average Reward across seeds (±STD) - Cartpole', 'out_png': OUT_ACH_PNG}
    }

    csv_rows = []
    
    for gname, info in groups.items():
        all_runs = []
        for seed in SEEDS:
            run_dir = info['dir'] / f"{seed}{info['suffix']}"
            csv_path = run_dir / f"{seed}{info['file_suffix']}"
            rewards = read_rewards(csv_path)
            if rewards is not None:
                all_runs.append(rewards)
            else:
                print(f"Warning: Could not read rewards from {csv_path}")
        
        if not all_runs:
            print(f"Error: No data loaded for group {gname}")
            continue
            
        # Align lengths
        min_len = min(len(r) for r in all_runs)
        aligned_runs = np.stack([r[:min_len] for r in all_runs], axis=0)
        
        # Calculate raw statistics per episode across seeds
        raw_mean = aligned_runs.mean(axis=0)
        raw_std = aligned_runs.std(axis=0)
        
        # Record stats for saving
        for i, (m, s) in enumerate(zip(raw_mean, raw_std)):
            csv_rows.append({
                'group': gname,
                'episode': i,
                'mean_reward': m,
                'std_reward': s
            })
            
        total_mean = raw_mean.mean()
        late_mean = raw_mean[-1000:].mean()
        print(f"Group: {gname:<15} | Average Reward: {total_mean:.2f} | Late Episode Reward (last 1k): {late_mean:.2f}")

        # Plot standalone figure
        plot_single_ablation(
            data=aligned_runs,
            color=info['color'],
            out_path=info['out_png'],
            title=info['title']
        )

    # Save CSV Stats
    with OUT_CSV.open('w', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=['group', 'episode', 'mean_reward', 'std_reward'])
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"Saved aggregation statistics to {OUT_CSV}")

if __name__ == '__main__':
    main()
