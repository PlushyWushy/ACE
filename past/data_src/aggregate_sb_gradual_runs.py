#!/usr/bin/env python3
"""
Aggregate rewards for sb_gradual runs.
Produces:
- sb_gradual/total_rewards_comparison.csv
- sb_gradual/post_switch_rewards_comparison.csv

This script treats post-switch as episodes with index >= 10000.
"""
from pathlib import Path
import csv

RUNS_DIR = Path("sb_gradual/runs")
OUT_TOTAL = Path("sb_gradual/total_rewards_comparison.csv")
OUT_POST = Path("sb_gradual/post_switch_rewards_comparison.csv")
SWITCH_EP = 10000  # 0-based: episodes >= 10000 are post-switch

def infer_and_sum(csv_path: Path):
    """Return (total_reward, post_switch_reward, n_rows)
    """
    total = 0.0
    post = 0.0
    n = 0
    with csv_path.open() as fh:
        reader = csv.DictReader(fh)
        headers = reader.fieldnames
        if headers is None:
            return 0.0, 0.0, 0
        ep_col = "episode" if "episode" in headers else headers[0]
        if "reward" in headers:
            reward_col = "reward"
        elif "is_optimal" in headers:
            reward_col = "is_optimal"
        else:
            reward_col = headers[1] if len(headers) >= 2 else None
        
        if not reward_col:
            return 0.0, 0.0, 0

        fh.seek(0)
        reader = csv.DictReader(fh)
        for row in reader:
            try:
                ep = int(float(row.get(ep_col, 0)))
                val = float(row.get(reward_col, 0.0))
            except Exception:
                continue
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
        csvs = list(d.glob("*.csv"))
        if not csvs:
            continue
        chosen = csvs[0]
        for c in csvs:
            if d.name in c.name:
                chosen = c
                break
        total, post, n = infer_and_sum(chosen)
        parts = d.name.split("_")
        seed = parts[0]
        rtype = "_".join(parts[1:]) if len(parts) > 1 else "unknown"
        if rtype == "icarus_upgraded":
            rtype = "ACE"
        elif rtype == "icarus":
            rtype = "ACE (old)"
        runs.append({"run": d.name, "seed": seed, "type": rtype, "csv": str(chosen), "total": total, "post": post, "n_rows": n})

    OUT_TOTAL.parent.mkdir(parents=True, exist_ok=True)
    with OUT_TOTAL.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["run","seed","type","total_reward","n_rows"])
        for r in runs:
            writer.writerow([r["run"], r["seed"], r["type"], f"{r['total']:.6f}", r["n_rows"]])

    with OUT_POST.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["run","seed","type","post_switch_reward","n_rows"])
        for r in runs:
            writer.writerow([r["run"], r["seed"], r["type"], f"{r['post']:.6f}", r["n_rows"]])

    print(f"Wrote {OUT_TOTAL} and {OUT_POST}")

if __name__ == '__main__':
    main()
