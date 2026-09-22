#!/usr/bin/env python3
"""
Diagnostic plots for results/bandit/runs (50-episode windows, mean +/- 1 SD):
reward_rate.png, neuromodulators.png, and a stats CSV for each.
"""

import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "results", "bandit"))
RUNS = os.path.join(HERE, "runs")
WINDOW = 50
SWITCH_EP = 10000

STYLE = {                       # condition -> (label, colour)
    "ace":      ("ACE (both modulators adaptive)", "tab:blue"),
    "classic":  ("Classic RSTDP (no modulation)",  "tab:gray"),
    "only_ne":  ("NE-only (ACh frozen)",           "tab:green"),
    "only_ach": ("ACh-only (NE frozen)",           "tab:orange"),
}


def load(tag, column):
    """Stack `column` across every seed of `tag` -> (n_seeds, n_episodes)."""
    out = []
    for f in sorted(glob.glob(os.path.join(RUNS, f"*_{tag}", "data.csv")),
                    key=lambda p: int(os.path.basename(os.path.dirname(p)).split("_")[0])):
        out.append(pd.read_csv(f)[column].values)
    if not out:
        return None
    n = min(len(a) for a in out)
    return np.stack([a[:n] for a in out])


def windowed(arr, window=WINDOW, scale=100.0):
    """Non-overlapping window means -> (n_seeds, n_windows)."""
    m = (arr.shape[1] // window) * window
    return arr[:, :m].reshape(arr.shape[0], -1, window).mean(axis=2) * scale


def band(ax, x, data, label, colour, clip=None):
    mean, sd = data.mean(axis=0), data.std(axis=0)
    lo, hi = mean - sd, mean + sd
    if clip is not None:
        lo, hi = np.clip(lo, *clip), np.clip(hi, *clip)
    ax.plot(x, mean, color=colour, lw=2, label=label)
    ax.fill_between(x, lo, hi, color=colour, alpha=0.22, lw=0)
    return mean, sd


def save_stats(path, x, series):
    rows = {"window_end_episode": x}
    for name, (m, s) in series.items():
        rows[f"{name}_mean"] = m
        rows[f"{name}_std"] = s
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"  wrote {path}")


def fig_reward():
    curves = {t: windowed(load(t, "is_optimal")) for t in STYLE if load(t, "is_optimal") is not None}
    if "ace" not in curves:
        print("no ACE runs found — run run_batch.py first")
        return
    x = np.arange(1, curves["ace"].shape[1] + 1) * WINDOW

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    stats = {}
    for ax, tags, title in [
        (axes[0], ["classic", "ace"], "(a) ACE vs classic RSTDP"),
        (axes[1], ["ace", "only_ne", "only_ach"], "(b) Level-matched ablations"),
    ]:
        for t in tags:
            if t not in curves:
                continue
            lbl, col = STYLE[t]
            stats[t] = band(ax, x, curves[t], lbl, col, clip=(0, 100))
        ax.axvline(SWITCH_EP, color="tab:red", ls="--", lw=1.5, label="Switch")
        ax.set_xlabel("Episode")
        ax.set_title(title, fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-5, 105)
        ax.legend(loc="lower right", fontsize=8)
    axes[0].set_ylabel("Percent Optimal (%)")
    fig.suptitle("Default Switch Bandit — mean optimal-action rate across 20 seeds "
                 "(50-episode windows, shaded $\\pm$1 SD)", fontsize=12)
    fig.tight_layout()
    out = os.path.join(HERE, "reward_rate.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"  wrote {out}")
    save_stats(os.path.join(HERE, "reward_rate_stats.csv"), x, stats)


def fig_neuromod():
    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    stats = {}
    for ax, col, lbl, colour in [
        (axes[0], "ach", "ACh (actor learning rate $\\eta_t$)", "tab:purple"),
        (axes[1], "sigma_ne", "NE (exploration noise $\\sigma_{NE}$)", "tab:orange"),
    ]:
        raw = load("ace", col)
        if raw is None:
            print("no ACE runs found — run run_batch.py first")
            return
        w = windowed(raw, scale=1.0)
        x = np.arange(1, w.shape[1] + 1) * WINDOW
        stats[col] = band(ax, x, w, lbl, colour)
        ax.axvline(SWITCH_EP, color="tab:red", ls="--", lw=1.5, label="Switch")
        ax.set_ylabel(lbl, fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", fontsize=8)
        rng = (w.mean(axis=0).max() / max(w.mean(axis=0).min(), 1e-12) - 1) * 100
        ax.text(0.015, 0.9, f"modulation range: {rng:+.0f}%", transform=ax.transAxes,
                fontsize=8, va="top",
                bbox=dict(fc="white", ec="0.7", alpha=0.85, boxstyle="round,pad=0.3"))
    axes[1].set_xlabel("Episode")
    fig.suptitle("Default Switch Bandit — realised neuromodulator values, ACE, 20 seeds\n"
                 "(what the actor actually receives, not the raw surprise signal)",
                 fontsize=11)
    fig.tight_layout()
    out = os.path.join(HERE, "neuromodulators.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"  wrote {out}")
    save_stats(os.path.join(HERE, "neuromodulator_stats.csv"),
               np.arange(1, len(stats["ach"][0]) + 1) * WINDOW, stats)


if __name__ == "__main__":
    print("figures:")
    fig_reward()
    fig_neuromod()
