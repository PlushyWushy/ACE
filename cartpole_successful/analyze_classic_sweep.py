#!/usr/bin/env python3
"""
Consolidate the classic CartPole grid-search results, rank configs by mean total
reward, write a summary CSV, then train the top-5 configs for 5 seeds each.
"""

import csv
import glob
import os
import statistics
import time
import subprocess
import concurrent.futures

import numpy as np

ROOT       = os.path.dirname(os.path.abspath(__file__))
SWEEP_DIR  = os.path.join(ROOT, "classic_hyperparam_search")
TOP5_DIR   = os.path.join(ROOT, "classic_top5_runs")
SCRIPT     = os.path.join(ROOT, "classic.py")
SWITCH_EP  = 5000
EPISODES   = 10000
SEEDS      = [1, 2, 3, 4, 5]
MAX_WORKERS = 5

# Load manifest at module level so run_one (called in worker processes) can see it
manifest = {}
with open(os.path.join(SWEEP_DIR, "configs.csv")) as f:
    for row in csv.DictReader(f):
        cid = int(row["config_id"])
        manifest[cid] = {k: float(v) for k, v in row.items()
                         if k not in ("config_id", "actor_lr_max")}


def load_rewards(path):
    with open(path) as f:
        return [float(row["reward"]) for row in csv.DictReader(f)]


def run_one(config_id, seed):
    params = manifest[config_id]
    cmd = [
        "python3", SCRIPT,
        "--seed",         str(seed),
        "--episodes",     str(EPISODES),
        "--config_id",    str(config_id),
        "--out_dir",      TOP5_DIR,
        "--actor_lr_min", str(params["actor_lr_min"]),
        "--actor_lr_max", str(params["actor_lr_min"]),   # pinned
        "--base_noise",   str(params["base_noise"]),
    ]
    start = time.time()
    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True)
        return config_id, seed, True, time.time() - start, ""
    except subprocess.CalledProcessError as e:
        return config_id, seed, False, time.time() - start, (e.stderr or e.stdout or str(e))[:400]


if __name__ == "__main__":

    # ── Load sweep CSVs ───────────────────────────────────────────────────────

    per_run = {}
    for path in glob.glob(os.path.join(SWEEP_DIR, "config_*_seed_*.csv")):
        fname = os.path.basename(path).replace(".csv", "").split("_")
        cid, seed = int(fname[1]), int(fname[3])
        rewards = load_rewards(path)
        r = np.array(rewards, dtype=float)
        per_run[(cid, seed)] = dict(
            mean_total = float(r.mean()),
            mean_post  = float(r[SWITCH_EP:].mean()) if len(r) > SWITCH_EP else float("nan"),
        )

    config_ids = sorted({k[0] for k in per_run})
    by_config = {}
    for cid in config_ids:
        runs   = [v for (c, s), v in per_run.items() if c == cid]
        totals = [v["mean_total"] for v in runs]
        posts  = [v["mean_post"]  for v in runs]
        by_config[cid] = dict(
            mean_total = statistics.mean(totals),
            std_total  = statistics.stdev(totals) if len(totals) > 1 else 0.0,
            mean_post  = statistics.mean(posts),
            std_post   = statistics.stdev(posts)  if len(posts)  > 1 else 0.0,
            n_seeds    = len(runs),
        )

    # ── Write summary CSV ─────────────────────────────────────────────────────

    summary_path = os.path.join(ROOT, "classic_sweep_summary.csv")
    with open(summary_path, "w", newline="") as f:
        fields = ["config_id", "actor_lr_min", "base_noise", "n_seeds",
                  "mean_total", "std_total", "mean_post", "std_post"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for cid in sorted(by_config, key=lambda c: -by_config[c]["mean_total"]):
            row = {"config_id": cid, **manifest[cid], **by_config[cid]}
            w.writerow(row)
    print(f"Summary → {summary_path}")

    # ── Print ranking ─────────────────────────────────────────────────────────

    ranked = sorted(by_config, key=lambda c: -by_config[c]["mean_total"])

    print(f"\n{'Rank':>4}  {'CID':>4}  {'LR_min':>8}  {'Noise':>6}  "
          f"{'Total':>14}  {'Post-switch':>14}")
    print("-" * 68)
    for rank, cid in enumerate(ranked, 1):
        v = by_config[cid]
        p = manifest[cid]
        mark = "  ← TOP 5" if rank <= 5 else ""
        print(f"{rank:>4}  {cid:>4}  {p['actor_lr_min']:>8.1e}  {p['base_noise']:>6.2f}  "
              f"{v['mean_total']:>7.1f}±{v['std_total']:>5.1f}  "
              f"{v['mean_post']:>7.1f}±{v['std_post']:>5.1f}{mark}")

    top5_ids = ranked[:5]
    print(f"\nTop-5 config IDs: {top5_ids}")

    # ── Run top-5 × 5 seeds ───────────────────────────────────────────────────

    os.makedirs(TOP5_DIR, exist_ok=True)
    jobs  = [(cid, seed) for cid in top5_ids for seed in SEEDS]
    total = len(jobs)

    print(f"\n{'='*60}")
    print(f"Top-5 validation  ({len(top5_ids)} configs × {len(SEEDS)} seeds = {total} runs)")
    print(f"Output: {TOP5_DIR}")
    print(f"{'='*60}")

    completed = failed = 0
    start_all = time.time()

    with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {ex.submit(run_one, cid, seed): (cid, seed) for cid, seed in jobs}
        for future in concurrent.futures.as_completed(futures):
            cid, seed = futures[future]
            config_id, seed_r, ok, elapsed, err = future.result()
            completed += 1
            p = manifest[config_id]
            tag = f"lr={p['actor_lr_min']:.0e} noise={p['base_noise']}"
            if ok:
                print(f"[{completed:3d}/{total}] OK   config {config_id:2d} ({tag}) seed {seed_r} — {elapsed:.0f}s")
            else:
                failed += 1
                print(f"[{completed:3d}/{total}] FAIL config {config_id:2d} ({tag}) seed {seed_r}")
                print(f"         {err}")

    elapsed_total = time.time() - start_all
    print(f"{'='*60}")
    print(f"Done in {elapsed_total/60:.1f} min  |  {total-failed}/{total} succeeded  |  {failed} failed")
    print(f"{'='*60}")
