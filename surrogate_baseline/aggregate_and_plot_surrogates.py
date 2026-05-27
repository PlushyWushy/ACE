#!/usr/bin/env python3
"""
Aggregate and plot surrogate-baseline runs for all 5 tasks:
  bandit, cartpole, acrobot, bandit_gradual, bandit_multiswitch

Produces one PNG per task in surrogate_baseline/:
  bandit_surrogate_mean_std.png
  cartpole_surrogate_mean_std.png
  acrobot_surrogate_mean_std.png
  bandit_gradual_surrogate_mean_std.png
  bandit_multiswitch_surrogate_mean_std.png
"""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

ROOT = os.path.dirname(os.path.abspath(__file__))
RUNS_DIR = os.path.join(ROOT, "runs")
OUT_DIR = ROOT
WINDOW = 50


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def find_csv(run_dir: str):
    files = glob.glob(os.path.join(run_dir, "*.csv"))
    return files[0] if files else None


def load_runs(pattern: str):
    paths = sorted(glob.glob(os.path.join(RUNS_DIR, pattern)))
    runs = []
    for p in paths:
        csv_path = find_csv(p)
        if csv_path is None:
            continue
        try:
            df = pd.read_csv(csv_path)
        except Exception:
            continue
        runs.append((os.path.basename(p), df))
    return runs


def windowed_mean(arr: np.ndarray, window: int):
    """Non-overlapping window mean. Returns (values, window_center_episodes)."""
    n = len(arr)
    m = (n // window) * window
    if m == 0:
        return np.array([]), np.array([])
    arr = arr[:m].reshape(-1, window)
    vals = arr.mean(axis=1)
    centers = (np.arange(arr.shape[0]) + 0.5) * window
    return vals, centers


def stack_runs(series_list, window):
    """Window each series and stack into (n_runs, n_windows), returning stacked array and centers."""
    windowed = []
    centers = None
    for s in series_list:
        w, c = windowed_mean(np.asarray(s, dtype=float), window)
        if w.size == 0:
            continue
        windowed.append(w)
        if centers is None:
            centers = c
    if not windowed:
        return None, None
    min_len = min(len(w) for w in windowed)
    stacked = np.stack([w[:min_len] for w in windowed], axis=0)
    return stacked, centers[:min_len]


def savefig(name: str):
    out = os.path.join(OUT_DIR, name)
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Wrote {out}")


# ---------------------------------------------------------------------------
# Task: switch bandit  (is_optimal, switch at ep 10000, 20000 total)
# ---------------------------------------------------------------------------

def plot_bandit():
    runs = load_runs("*_bandit_surrogate")
    print(f"bandit: {len(runs)} runs")
    series_list = [df["is_optimal"].astype(float).values for _, df in runs if "is_optimal" in df.columns]
    stacked, centers = stack_runs(series_list, WINDOW)
    if stacked is None:
        print("  no data"); return

    mean = stacked.mean(axis=0) * 100.0
    std  = stacked.std(axis=0)  * 100.0

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(centers, mean, color="tab:blue", lw=2, label="Mean % optimal")
    ax.fill_between(centers, np.clip(mean - std, 0, 100), np.clip(mean + std, 0, 100),
                    color="tab:blue", alpha=0.25, label="±1 std")
    ax.axvline(10000, color="tab:red", ls="--", lw=1.5, label="Switch (ep 10 000)")
    ax.set_xlabel("Episode"); ax.set_ylabel("Percent Optimal (%)")
    ax.set_title(f"Surrogate Baseline — Switch Bandit  (n={stacked.shape[0]} seeds)")
    ax.set_ylim(-5, 105); ax.grid(True, alpha=0.3); ax.legend()
    savefig("bandit_surrogate_mean_std.png")


# ---------------------------------------------------------------------------
# Task: switch cartpole  (reward, switch at ep 5000, 10000 total)
# ---------------------------------------------------------------------------

def plot_cartpole():
    runs = load_runs("*_cartpole_surrogate")
    print(f"cartpole: {len(runs)} runs")
    series_list = [df["reward"].astype(float).values for _, df in runs if "reward" in df.columns]
    stacked, centers = stack_runs(series_list, WINDOW)
    if stacked is None:
        print("  no data"); return

    mean = stacked.mean(axis=0)
    std  = stacked.std(axis=0)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(centers, mean, color="tab:orange", lw=2, label="Mean reward")
    ax.fill_between(centers, mean - std, mean + std,
                    color="tab:orange", alpha=0.25, label="±1 std")
    ax.axvline(5000, color="tab:red", ls="--", lw=1.5, label="Switch (ep 5 000)")
    ax.set_xlabel("Episode"); ax.set_ylabel("Episode Reward")
    ax.set_title(f"Surrogate Baseline — Switch CartPole  (n={stacked.shape[0]} seeds)")
    ax.grid(True, alpha=0.3); ax.legend()
    savefig("cartpole_surrogate_mean_std.png")


# ---------------------------------------------------------------------------
# Task: switch acrobot  (reward, switch at ep 7500, 15000 total)
# ---------------------------------------------------------------------------

def plot_acrobot():
    runs = load_runs("*_acrobot_surrogate")
    print(f"acrobot: {len(runs)} runs")
    series_list = [df["reward"].astype(float).values for _, df in runs if "reward" in df.columns]
    stacked, centers = stack_runs(series_list, WINDOW)
    if stacked is None:
        print("  no data"); return

    mean = stacked.mean(axis=0)
    std  = stacked.std(axis=0)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(centers, mean, color="tab:green", lw=2, label="Mean reward")
    ax.fill_between(centers, mean - std, mean + std,
                    color="tab:green", alpha=0.25, label="±1 std")
    ax.axvline(7500, color="tab:red", ls="--", lw=1.5, label="Switch (ep 7 500)")
    ax.set_xlabel("Episode"); ax.set_ylabel("Episode Reward")
    ax.set_title(f"Surrogate Baseline — Switch Acrobot  (n={stacked.shape[0]} seeds)")
    ax.grid(True, alpha=0.3); ax.legend()
    savefig("acrobot_surrogate_mean_std.png")


# ---------------------------------------------------------------------------
# Task: gradual bandit  (is_optimal, gradual switch ep 10000–10200, 20000 total)
# ---------------------------------------------------------------------------

def plot_bandit_gradual():
    runs = load_runs("*_bandit_gradual_surrogate")
    print(f"bandit_gradual: {len(runs)} runs")
    series_list = [df["is_optimal"].astype(float).values for _, df in runs if "is_optimal" in df.columns]
    stacked, centers = stack_runs(series_list, WINDOW)
    if stacked is None:
        print("  no data"); return

    mean = stacked.mean(axis=0) * 100.0
    std  = stacked.std(axis=0)  * 100.0

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(centers, mean, color="tab:purple", lw=2, label="Mean % optimal")
    ax.fill_between(centers, np.clip(mean - std, 0, 100), np.clip(mean + std, 0, 100),
                    color="tab:purple", alpha=0.25, label="±1 std")
    ax.axvspan(10000, 10200, color="tab:red", alpha=0.12, label="Gradual switch (ep 10 000–10 200)")
    ax.axvline(10000, color="tab:red", ls="--", lw=1.5)
    ax.set_xlabel("Episode"); ax.set_ylabel("Percent Optimal (%)")
    ax.set_title(f"Surrogate Baseline — Gradual Bandit  (n={stacked.shape[0]} seeds)")
    ax.set_ylim(-5, 105); ax.grid(True, alpha=0.3); ax.legend()
    savefig("bandit_gradual_surrogate_mean_std.png")


# ---------------------------------------------------------------------------
# Task: multiswitch bandit  (is_optimal, switches at 15000/30000/45000, 60000 total)
# ---------------------------------------------------------------------------

def plot_bandit_multiswitch():
    runs = load_runs("*_bandit_multiswitch_surrogate")
    print(f"bandit_multiswitch: {len(runs)} runs")
    series_list = [df["is_optimal"].astype(float).values for _, df in runs if "is_optimal" in df.columns]
    stacked, centers = stack_runs(series_list, WINDOW)
    if stacked is None:
        print("  no data"); return

    mean = stacked.mean(axis=0) * 100.0
    std  = stacked.std(axis=0)  * 100.0

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(centers, mean, color="tab:brown", lw=2, label="Mean % optimal")
    ax.fill_between(centers, np.clip(mean - std, 0, 100), np.clip(mean + std, 0, 100),
                    color="tab:brown", alpha=0.25, label="±1 std")
    for i, sw in enumerate([15000, 30000, 45000]):
        label = "Switch" if i == 0 else None
        ax.axvline(sw, color="tab:red", ls="--", lw=1.5, label=label)
    ax.set_xlabel("Episode"); ax.set_ylabel("Percent Optimal (%)")
    ax.set_title(f"Surrogate Baseline — Multi-Switch Bandit  (n={stacked.shape[0]} seeds)")
    ax.set_ylim(-5, 105); ax.grid(True, alpha=0.3); ax.legend()
    savefig("bandit_multiswitch_surrogate_mean_std.png")


# ---------------------------------------------------------------------------

def main():
    plot_bandit()
    plot_cartpole()
    plot_acrobot()
    plot_bandit_gradual()
    plot_bandit_multiswitch()


if __name__ == "__main__":
    main()
