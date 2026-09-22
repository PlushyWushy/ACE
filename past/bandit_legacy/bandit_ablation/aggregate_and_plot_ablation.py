#!/usr/bin/env python3
"""
Aggregate and plot NE and ACH ablation runs.
Produces:
 - bandit_ablation/ne_ablation_avg_reward_rate.png
 - bandit_ablation/ach_ablation_avg_reward_rate.png
 - bandit_ablation/ne_ablation_reward_rate_stats.csv
 - bandit_ablation/ach_ablation_reward_rate_stats.csv
 - bandit_ablation/total_rewards_comparison.csv
 - bandit_ablation/post_switch_rewards_comparison.csv

Assumptions:
 - Per-run CSVs are in `bandit_ablation/runs/<seed>_ne_ablation/<seed>_ne_ablation.csv`
   and `bandit_ablation/runs/<seed>_ach_ablation/<seed>_ach_ablation.csv`.
 - Per-run CSVs include columns `episode`, `is_optimal`, and `reward`.
 - Switch occurs at episode 2000 (post-switch episodes >= 2000).
"""
import math
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
RUNS_DIR = ROOT / "runs"
WINDOW = 50
SWITCH_EP = 2000


def find_runs(suffix):
    # returns dict seed -> csv_path
    runs = {}
    for d in RUNS_DIR.glob(f"*_{suffix}"):
        # expect csv inside named `<seed>_{suffix}.csv`
        seed = d.name.split("_")[0]
        # try expected file name first, otherwise pick any .csv in dir
        csv1 = d / f"{seed}_{suffix}.csv"
        if csv1.exists():
            runs[int(seed)] = csv1
            continue
        # fallback: pick first csv file inside directory
        csvs = list(d.glob('*.csv'))
        if csvs:
            runs[int(seed)] = csvs[0]
    return dict(sorted(runs.items()))


def load_run(csv_path):
    df = pd.read_csv(csv_path)
    # ensure episode present
    if 'episode' not in df.columns:
        # try first column
        df = df.reset_index()
        df = df.rename(columns={'index':'episode'})
    return df


def window_percent_optimal(df, window=WINDOW):
    if 'is_optimal' not in df.columns:
        raise ValueError(f"CSV {df} missing 'is_optimal' column")
    # Sort by episode
    df = df.sort_values('episode')
    vals = df['is_optimal'].to_numpy()
    n = len(vals)
    # truncate to multiple of window
    m = (n // window) * window
    if m == 0:
        return np.array([]), np.array([])
    vals = vals[:m]
    bins = vals.reshape(-1, window)
    pct = bins.mean(axis=1) * 100.0
    # return array of window mid-episode indices
    starts = np.arange(0, m, window)
    ends = starts + window - 1
    indices = list(zip(starts, ends))
    return pct, indices


def compute_stats_for_group(runs_dict, suffix_name):
    all_windows = []
    # we will also collect per-run totals
    per_run_totals = {}
    per_run_post_totals = {}
    for seed, csv in runs_dict.items():
        df = load_run(csv)
        # compute totals: prefer 'reward' column, otherwise use 'is_optimal' as a proxy
        if 'reward' in df.columns:
            total = float(df['reward'].sum())
            post = float(df.loc[df['episode'] >= SWITCH_EP, 'reward'].sum())
        elif 'is_optimal' in df.columns:
            # treat 1/0 optimal flags as "reward" proxy (counts of optimal choices)
            total = float(df['is_optimal'].sum())
            post = float(df.loc[df['episode'] >= SWITCH_EP, 'is_optimal'].sum())
        else:
            raise ValueError(f"CSV {csv} missing both 'reward' and 'is_optimal' columns")
        per_run_totals[seed] = total
        per_run_post_totals[seed] = post
        pct, indices = window_percent_optimal(df)
        all_windows.append(pct)
    if len(all_windows) == 0:
        raise ValueError(f"No runs found for {suffix_name}")
    # pad arrays to same length (truncate to shortest)
    minlen = min(a.shape[0] for a in all_windows)
    stacked = np.stack([a[:minlen] for a in all_windows], axis=0)
    mean = stacked.mean(axis=0)
    std = stacked.std(axis=0)
    # determine window episode ranges from first run's indices
    _, indices0 = window_percent_optimal(load_run(next(iter(runs_dict.values()))))
    indices0 = indices0[:minlen]
    return {
        'mean': mean,
        'std': std,
        'indices': indices0,
        'per_run_totals': per_run_totals,
        'per_run_post_totals': per_run_post_totals,
        'seeds': sorted(list(runs_dict.keys()))
    }


def save_plot(stats, out_png, out_csv, title):
    mean = stats['mean']
    std = stats['std']
    indices = stats['indices']
    # x axis as window center episode
    x = np.array([(s + e) / 2.0 for s, e in indices])
    plt.figure(figsize=(8,4))
    plt.plot(x, mean, color='tab:green')
    plt.fill_between(x, mean-std, mean+std, color='tab:green', alpha=0.3)
    plt.xlabel('Episode')
    plt.ylabel('Percent Optimal (%)')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()
    # save stats CSV
    rows = []
    for (s,e), m, sd in zip(indices, mean, std):
        rows.append({'window_start': int(s), 'window_end': int(e), 'mean_pct_opt': float(m), 'std_pct_opt': float(sd)})
    pd.DataFrame(rows).to_csv(out_csv, index=False)


def write_comparison_tables(ne_totals, ach_totals, out_total_csv, out_post_csv):
    # find seeds present in both
    seeds = sorted(set(ne_totals.keys()) & set(ach_totals.keys()))
    rows_total = []
    rows_post = []
    for s in seeds:
        rows_total.append({'seed': s, 'ne_total': ne_totals[s], 'ach_total': ach_totals[s]})
        rows_post.append({'seed': s, 'ne_post': ne_totals[s+'_post'], 'ach_post': ach_totals[s+'_post']})
    # The per-run post totals were passed as separate maps; but to simplify, we expect ach_totals and ne_totals to include postfix keys.
    # We'll instead accept the per_run_post dicts passed in separately (but to keep API, modify caller accordingly).
    # For safety, create from available data if above failed.


if __name__ == '__main__':
    ne_runs = find_runs('ne_ablation')
    ach_runs = find_runs('ach_ablation')
    print(f"Found NE runs: {len(ne_runs)}, ACH runs: {len(ach_runs)}")

    ne_stats = compute_stats_for_group(ne_runs, 'NE ablation')
    ach_stats = compute_stats_for_group(ach_runs, 'ACH ablation')

    # save plots and stats
    out_ne_png = ROOT / 'ne_ablation_avg_reward_rate.png'
    out_ne_csv = ROOT / 'ne_ablation_reward_rate_stats.csv'
    save_plot(ne_stats, out_ne_png, out_ne_csv, 'NE Ablation Average Reward Rate (mean ± std)')

    out_ach_png = ROOT / 'ach_ablation_avg_reward_rate.png'
    out_ach_csv = ROOT / 'ach_ablation_reward_rate_stats.csv'
    save_plot(ach_stats, out_ach_png, out_ach_csv, 'ACH Ablation Average Reward Rate (mean ± std)')

    # create the comparison tables: seeds present in both
    ne_totals = ne_stats['per_run_totals']
    ach_totals = ach_stats['per_run_totals']
    ne_post = ne_stats['per_run_post_totals']
    ach_post = ach_stats['per_run_post_totals']

    seeds = sorted(set(ne_totals.keys()) & set(ach_totals.keys()))
    rows_total = []
    rows_post = []
    for s in seeds:
        rows_total.append({'seed': s, 'ne_total': ne_totals[s], 'ach_total': ach_totals[s]})
        rows_post.append({'seed': s, 'ne_post': ne_post.get(s, 0.0), 'ach_post': ach_post.get(s, 0.0)})

    pd.DataFrame(rows_total).to_csv(ROOT / 'total_rewards_comparison.csv', index=False)
    pd.DataFrame(rows_post).to_csv(ROOT / 'post_switch_rewards_comparison.csv', index=False)

    print('Wrote: ', out_ne_png, out_ne_csv, out_ach_png, out_ach_csv)
    print('Wrote comparison tables under', ROOT)
