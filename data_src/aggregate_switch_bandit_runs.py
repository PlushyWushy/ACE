#!/usr/bin/env python3
"""
Aggregate rewards for switch_bandit_experiment runs.
Produces:
- switch_bandit_experiment/total_rewards_comparison.csv
- switch_bandit_experiment/post_switch_rewards_comparison.csv

This script is robust to CSVs that use either a `reward` column or `is_optimal`.
It treats post-switch as episodes with index >= 2000 (0-based indexing).
"""
from pathlib import Path
import csv

RUNS_DIR = Path("switch_bandit_experiment/runs")
OUT_TOTAL = Path("switch_bandit_experiment/total_rewards_comparison.csv")
OUT_POST = Path("switch_bandit_experiment/post_switch_rewards_comparison.csv")
SWITCH_EP = 2000  # 0-based: episodes >= 2000 are post-switch (episodes 2001..)


def infer_and_sum(csv_path: Path):
    """Return (total_reward, post_switch_reward, n_rows)
    """
    total = 0.0
    post = 0.0
    n = 0
    with csv_path.open() as fh:
        # try DictReader first
        reader = csv.DictReader(fh)
        headers = reader.fieldnames
        if headers is None:
            return 0.0, 0.0, 0
        # determine episode column
        ep_col = None
        if "episode" in headers:
            ep_col = "episode"
        else:
            # fallback to first column name
            ep_col = headers[0]
        # determine reward column
        if "reward" in headers:
            reward_col = "reward"
        elif "is_optimal" in headers:
            reward_col = "is_optimal"
        else:
            # fallback: choose second header if exists
            if len(headers) >= 2:
                reward_col = headers[1]
            else:
                # give up
                return 0.0, 0.0, 0
        # iterate
        fh.seek(0)
        reader = csv.DictReader(fh)
        for row in reader:
            try:
                ep = int(row.get(ep_col, 0))
            except Exception:
                try:
                    ep = int(float(row.get(ep_col, 0)))
                except Exception:
                    ep = 0
            try:
                val = float(row.get(reward_col, 0.0))
            except Exception:
                try:
                    val = float(row.get(reward_col, 0) or 0)
                except Exception:
                    val = 0.0
            total += val
            if ep >= SWITCH_EP:
                post += val
            n += 1
    return total, post, n


def main():
    runs = []
    if not RUNS_DIR.exists():
        print(f"Runs directory not found: {RUNS_DIR}")
        return
    for d in sorted(RUNS_DIR.iterdir(), key=lambda p: p.name):
        if not d.is_dir():
            continue
        # find csv file in dir
        csvs = list(d.glob("*.csv"))
        if not csvs:
            print(f"No CSV in {d}")
            continue
        # prefer file containing dir name
        chosen = None
        for c in csvs:
            if d.name in c.name:
                chosen = c
                break
        if chosen is None:
            chosen = csvs[0]
        total, post, n = infer_and_sum(chosen)
        # parse type and seed from folder name e.g. '1_icarus' or '10_classic'
        parts = d.name.split("_")
        seed = parts[0]
        rtype = parts[1] if len(parts) > 1 else "unknown"
        runs.append({"run": d.name, "seed": seed, "type": rtype, "csv": str(chosen), "total": total, "post": post, "n_rows": n})

    # write total summary
    with OUT_TOTAL.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["run","seed","type","total_reward","n_rows"])
        for r in runs:
            writer.writerow([r["run"], r["seed"], r["type"], f"{r['total']:.6f}", r["n_rows"]])

    # write post-switch summary
    with OUT_POST.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["run","seed","type","post_switch_reward","n_rows"])
        for r in runs:
            writer.writerow([r["run"], r["seed"], r["type"], f"{r['post']:.6f}", r["n_rows"]])

    print(f"Wrote {OUT_TOTAL} and {OUT_POST}")

    # also print simple tables
    print("\nTotal rewards summary:")
    for r in runs:
        print(f"{r['run']:12s} | total={r['total']:.2f} | rows={r['n_rows']}")

    print("\nPost-switch rewards summary:")
    for r in runs:
        print(f"{r['run']:12s} | post={r['post']:.2f} | rows={r['n_rows']}")


if __name__ == '__main__':
    main()
