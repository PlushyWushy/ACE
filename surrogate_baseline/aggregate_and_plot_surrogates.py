#!/usr/bin/env python3
"""
Aggregate and plot surrogate-baseline runs for bandit and cartpole.

Produces:
- `surrogate_baseline/bandit_surrogate_mean_std.png`
- `surrogate_baseline/cartpole_surrogate_mean_std.png`
- `surrogate_baseline/total_bandit_surrogate.csv`
- `surrogate_baseline/post_switch_bandit_surrogate.csv`
- `surrogate_baseline/total_cartpole_surrogate.csv`
- `surrogate_baseline/post_switch_cartpole_surrogate.csv`

The script is robust to CSV column names: prefers `is_optimal` for bandit per-episode metric,
falls back to `reward` if `is_optimal` missing. For totals it sums `reward` if present else sums `is_optimal`.
"""

import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


ROOT = os.path.dirname(__file__)
RUNS_DIR = os.path.join(ROOT, "runs")
OUT_DIR = ROOT
SWITCH_EP = 2000
WINDOW = 50


def find_run_csv(run_dir: str):
    # find first csv in directory
    files = glob.glob(os.path.join(run_dir, "*.csv"))
    return files[0] if files else None


def load_runs(pattern: str):
    paths = sorted(glob.glob(os.path.join(RUNS_DIR, pattern)))
    runs = []
    for p in paths:
        csv = find_run_csv(p)
        if csv is None:
            continue
        try:
            df = pd.read_csv(csv)
        except Exception:
            continue
        seed_name = os.path.basename(p)
        runs.append((seed_name, df, csv))
    return runs


def align_and_stack(series_list):
    # truncate to shortest length and stack into 2D array (runs x episodes)
    lengths = [len(s) for s in series_list]
    if len(lengths) == 0:
        return np.array([])
    m = min(lengths)
    arr = np.vstack([np.asarray(s[:m]) for s in series_list])
    return arr


def aggregate_bandit():
    runs = load_runs("*_bandit_surrogate")
    print(f"Found {len(runs)} bandit runs")
    per_window_series = []
    totals = []
    post_totals = []
    names = []

    def window_percent_optimal(series_vals, window=WINDOW):
        vals = np.asarray(series_vals)
        n = len(vals)
        m = (n // window) * window
        if m == 0:
            return np.array([]), []
        vals = vals[:m]
        bins = vals.reshape(-1, window)
        pct = bins.mean(axis=1) * 100.0
        starts = np.arange(0, m, window)
        ends = starts + window - 1
        indices = list(zip(starts, ends))
        return pct, indices

    indices_list = []
    for name, df, csv in runs:
        names.append(name)
        # prefer is_optimal; if absent, fall back to reward if it's 0/1, otherwise attempt to binarize
        if 'is_optimal' in df.columns:
            series = df['is_optimal'].astype(float).to_numpy()
            total = float(df['is_optimal'].sum())
            post = float(df.loc[df.index >= SWITCH_EP, 'is_optimal'].sum()) if len(df) > SWITCH_EP else 0.0
        elif 'reward' in df.columns:
            # if reward is continuous, treat reward>0 as optimal proxy
            r = df['reward'].astype(float)
            if set(r.unique()) <= {0, 1}:
                series = r.to_numpy()
            else:
                series = (r > 0).astype(float).to_numpy()
            total = float(r.sum())
            post = float(r.iloc[SWITCH_EP:].sum()) if len(r) > SWITCH_EP else 0.0
        else:
            numeric_cols = df.select_dtypes(include=[float, int]).columns.tolist()
            if len(numeric_cols) == 0:
                continue
            s = df[numeric_cols[0]].astype(float)
            series = (s > 0).astype(float).to_numpy()
            total = float(s.sum())
            post = float(s.iloc[SWITCH_EP:].sum()) if len(s) > SWITCH_EP else 0.0

        pct, indices = window_percent_optimal(series, WINDOW)
        if pct.size == 0:
            continue
        per_window_series.append(pct)
        indices_list.append(indices)
        totals.append(float(total))
        post_totals.append(float(post))

    if len(per_window_series) == 0:
        print("No bandit runs found or no valid windows.")
        return

    minlen = min(a.shape[0] for a in per_window_series)
    stacked = np.stack([a[:minlen] for a in per_window_series], axis=0)
    mean = stacked.mean(axis=0)
    std = stacked.std(axis=0)

    # determine indices from first run's indices list, truncated to minlen
    indices0 = indices_list[0][:minlen]

    # plot percent-optimal
    plt.figure(figsize=(10,5))
    x = np.array([(s+e)/2.0 for s, e in indices0])
    plt.plot(x, mean, label='Percent Optimal (mean)')
    plt.fill_between(x, mean-std, mean+std, alpha=0.3)
    plt.xlabel('Episode')
    plt.ylabel('Percent Optimal (%)')
    plt.title('Bandit Surrogate - Percent Optimal (windowed) mean ± std')
    plt.grid(alpha=0.3)
    out_png = os.path.join(OUT_DIR, 'bandit_surrogate_mean_std.png')
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    print('Wrote', out_png)

    # totals tables
    df_tot = pd.DataFrame({'run': names, 'total': totals})
    df_post = pd.DataFrame({'run': names, 'post_switch_total': post_totals})
    df_tot.to_csv(os.path.join(OUT_DIR, 'total_bandit_surrogate.csv'), index=False)
    df_post.to_csv(os.path.join(OUT_DIR, 'post_switch_bandit_surrogate.csv'), index=False)
    print('Wrote total and post-switch bandit CSVs')


def aggregate_cartpole():
    runs = load_runs("*_cartpole_surrogate")
    print(f"Found {len(runs)} cartpole runs")
    per_episode_series = []
    totals = []
    post_totals = []
    names = []
    for name, df, csv in runs:
        names.append(name)
        if 'reward' in df.columns:
            series = df['reward'].astype(float)
            total = series.sum()
            post = series.iloc[SWITCH_EP:].sum() if len(series) > SWITCH_EP else 0.0
        else:
            numeric_cols = df.select_dtypes(include=[float, int]).columns.tolist()
            if len(numeric_cols) == 0:
                continue
            series = df[numeric_cols[0]].astype(float)
            total = series.sum()
            post = series.iloc[SWITCH_EP:].sum() if len(series) > SWITCH_EP else 0.0

        per_episode_series.append(series.values)
        totals.append(float(total))
        post_totals.append(float(post))

    if len(per_episode_series) == 0:
        print("No cartpole runs found.")
        return

    arr = align_and_stack(per_episode_series)
    mean = arr.mean(axis=0)
    std = arr.std(axis=0)

    # plot
    plt.figure(figsize=(10,5))
    x = np.arange(len(mean))
    plt.plot(x, mean, label='Cartpole reward (mean)', color='tab:orange')
    plt.fill_between(x, mean-std, mean+std, alpha=0.3, color='tab:orange')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('CartPole Surrogate - Per-episode mean ± std')
    plt.grid(alpha=0.3)
    out_png = os.path.join(OUT_DIR, 'cartpole_surrogate_mean_std.png')
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    print('Wrote', out_png)

    # totals tables
    df_tot = pd.DataFrame({'run': names, 'total': totals})
    df_post = pd.DataFrame({'run': names, 'post_switch_total': post_totals})

    df_tot.to_csv(os.path.join(OUT_DIR, 'total_cartpole_surrogate.csv'), index=False)
    df_post.to_csv(os.path.join(OUT_DIR, 'post_switch_cartpole_surrogate.csv'), index=False)
    print('Wrote total and post-switch cartpole CSVs')


def main():
    aggregate_bandit()
    aggregate_cartpole()


if __name__ == '__main__':
    main()
