#!/usr/bin/env python3
"""
Hyperparameter sweep analysis for the CartPole switch experiment.

Produces:
  1. sweep_summary.csv        — per-config statistics across all 100 configs × 3 seeds
  2. top10_summary.csv        — per-config statistics for top-10 validation (5 seeds each)
  3. hyperparam_robustness.png — scatterplot: mean total reward vs mean post-switch reward
"""

import csv
import glob
import os
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT        = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "results", "cartpole"))
SWEEP_DIR   = os.path.join(ROOT, "hyperparam_search")
TOP10_DIR   = os.path.join(ROOT, "top10_runs")
CLASSIC_DIR = os.path.join(ROOT, "runs_saved")
SWITCH_EP   = 5000
VALID_SEEDS = set(range(1, 21))


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_rewards(path):
    with open(path) as f:
        return [float(row["reward"]) for row in csv.DictReader(f)]


def episode_stats(rewards, switch=SWITCH_EP):
    r = np.array(rewards, dtype=float)
    pre, post = r[:switch], r[switch:]
    return dict(
        mean_total = float(r.mean()),
        std_total  = float(r.std(ddof=1)) if len(r) > 1 else 0.0,
        mean_post  = float(post.mean()) if len(post) else float("nan"),
        n_episodes = len(r),
    )


# ── Load sweep results (100 configs × 3 seeds) ───────────────────────────────

print("Loading sweep results …")
sweep_per_run = {}
for path in glob.glob(os.path.join(SWEEP_DIR, "config_*_seed_*.csv")):
    fname = os.path.basename(path).replace(".csv", "").split("_")
    cid, seed = int(fname[1]), int(fname[3])
    rewards = load_rewards(path)
    sweep_per_run[(cid, seed)] = episode_stats(rewards)

config_ids = sorted({k[0] for k in sweep_per_run})
sweep_by_config = {}
for cid in config_ids:
    runs = [v for (c, s), v in sweep_per_run.items() if c == cid]
    if not runs:
        continue
    totals = [r["mean_total"] for r in runs]
    posts  = [r["mean_post"]  for r in runs]
    sweep_by_config[cid] = {
        "mean_total" : statistics.mean(totals),
        "std_total"  : statistics.stdev(totals) if len(totals) > 1 else 0.0,
        "mean_post"  : statistics.mean(posts),
        "std_post"   : statistics.stdev(posts)  if len(posts)  > 1 else 0.0,
        "n_seeds"    : len(runs),
    }

# ── Load top-10 validation results (5 seeds) ─────────────────────────────────

print("Loading top-10 validation results …")
top10_per_run = {}
for path in glob.glob(os.path.join(TOP10_DIR, "config_*_seed_*", "*.csv")):
    parts = os.path.basename(os.path.dirname(path)).split("_")
    cid, seed = int(parts[1]), int(parts[3])
    top10_per_run[(cid, seed)] = episode_stats(load_rewards(path))

top10_config_ids = sorted({k[0] for k in top10_per_run})
top10_by_config = {}
for cid in top10_config_ids:
    runs = [v for (c, s), v in top10_per_run.items() if c == cid]
    totals = [r["mean_total"] for r in runs]
    posts  = [r["mean_post"]  for r in runs]
    top10_by_config[cid] = {
        "mean_total" : statistics.mean(totals),
        "std_total"  : statistics.stdev(totals) if len(totals) > 1 else 0.0,
        "mean_post"  : statistics.mean(posts),
        "std_post"   : statistics.stdev(posts)  if len(posts)  > 1 else 0.0,
        "n_seeds"    : len(runs),
    }

# ── Load classic CartPole baseline (seeds 1–20 only) ─────────────────────────

print("Loading classic CartPole runs (seeds 1–20) …")
classic_rewards_all = []
for path in glob.glob(os.path.join(CLASSIC_DIR, "*_classic", "*_classic.csv")):
    seed_str = os.path.basename(os.path.dirname(path)).split("_")[0]
    try:
        seed = int(seed_str)
    except ValueError:
        continue
    if seed not in VALID_SEEDS:
        continue
    rewards = load_rewards(path)
    if len(rewards) >= SWITCH_EP:
        classic_rewards_all.append(rewards)

classic_per_seed_total = [np.mean(r)            for r in classic_rewards_all]
classic_per_seed_post  = [np.mean(r[SWITCH_EP:]) for r in classic_rewards_all]
classic_mean      = float(np.mean(classic_per_seed_total))
classic_std       = float(np.std(classic_per_seed_total, ddof=1))
classic_mean_post = float(np.mean(classic_per_seed_post))
classic_std_post  = float(np.std(classic_per_seed_post, ddof=1))
print(f"  Classic baseline: {len(classic_rewards_all)} seeds | "
      f"total={classic_mean:.1f}±{classic_std:.1f} | "
      f"post={classic_mean_post:.1f}±{classic_std_post:.1f}")

# ── Write sweep_summary.csv ───────────────────────────────────────────────────

sweep_csv = os.path.join(ROOT, "sweep_summary.csv")
with open(sweep_csv, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["config_id", "n_seeds", "mean_total", "std_total", "mean_post", "std_post"])
    w.writeheader()
    for cid in sorted(sweep_by_config):
        w.writerow({"config_id": cid, **sweep_by_config[cid]})
print(f"Saved {sweep_csv}")

# ── Write top10_summary.csv ───────────────────────────────────────────────────

top10_csv = os.path.join(ROOT, "top10_summary.csv")
with open(top10_csv, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["config_id", "n_seeds", "mean_total", "std_total", "mean_post", "std_post"])
    w.writeheader()
    for cid in sorted(top10_by_config, key=lambda c: -top10_by_config[c]["mean_total"]):
        w.writerow({"config_id": cid, **top10_by_config[cid]})
print(f"Saved {top10_csv}")

# ── Console summary ───────────────────────────────────────────────────────────

all_totals = [v["mean_total"] for v in sweep_by_config.values()]
all_posts  = [v["mean_post"]  for v in sweep_by_config.values()]

print(f"\n── Sweep (100 configs × 3 seeds) ──────────────────────────────────────")
print(f"  {'':12}  {'Mean':>8}  {'SD':>8}  {'Min':>8}  {'Max':>8}")
print(f"  {'Total':12}  {statistics.mean(all_totals):>8.1f}  {statistics.stdev(all_totals):>8.1f}  {min(all_totals):>8.1f}  {max(all_totals):>8.1f}")
print(f"  {'Post-switch':12}  {statistics.mean(all_posts):>8.1f}  {statistics.stdev(all_posts):>8.1f}  {min(all_posts):>8.1f}  {max(all_posts):>8.1f}")

print(f"\n── Classic baseline ({len(classic_rewards_all)} seeds, 1–20) ──────────────────────────────")
print(f"  {'':12}  {'Mean':>8}  {'SD':>8}")
print(f"  {'Total':12}  {classic_mean:>8.1f}  {classic_std:>8.1f}")
print(f"  {'Post-switch':12}  {classic_mean_post:>8.1f}  {classic_std_post:>8.1f}")

print(f"\n── Top-10 validation (5 seeds each) ───────────────────────────────────")
print(f"  {'Config':>8}  {'Total':>16}  {'Post-switch':>16}")
for cid in sorted(top10_by_config, key=lambda c: -top10_by_config[c]["mean_total"]):
    v = top10_by_config[cid]
    print(f"  {cid:>8}  {v['mean_total']:>7.1f}±{v['std_total']:>6.1f}  {v['mean_post']:>7.1f}±{v['std_post']:>6.1f}")

# ── Figure: scatterplot — mean total (x) vs mean post-switch (y) ──────────────

fig, ax = plt.subplots(figsize=(7, 5.5))

sweep_totals = [sweep_by_config[c]["mean_total"] for c in config_ids]
sweep_posts  = [sweep_by_config[c]["mean_post"]  for c in config_ids]
ax.scatter(sweep_totals, sweep_posts, s=30, alpha=0.6, color="#4C72B0",
           zorder=3, label="Sweep config (3 seeds)")

top10_totals = [top10_by_config[c]["mean_total"] for c in top10_config_ids]
top10_posts  = [top10_by_config[c]["mean_post"]  for c in top10_config_ids]
ax.scatter(top10_totals, top10_posts, s=80, marker="*", color="#C44E52",
           zorder=4, label="Top-10 validation (5 seeds)")

ax.scatter([classic_mean], [classic_mean_post], s=100, marker="D",
           color="#DD8452", zorder=5, label=f"Classic baseline (seeds 1–20)")

ax.set_xlabel("Mean total reward (all 10,000 episodes)")
ax.set_ylabel("Mean post-switch reward (ep 5001–10,000)")
ax.set_title("ACE hyperparameter robustness — CartPole switch experiment\n"
             "100 random configurations × 3 seeds")
ax.set_xlim(left=0)
ax.set_ylim(bottom=0)
ax.legend(fontsize=9)

out_fig = os.path.join(ROOT, "hyperparam_robustness.png")
plt.tight_layout()
plt.savefig(out_fig, dpi=150, bbox_inches="tight")
print(f"\nSaved figure → {out_fig}")
