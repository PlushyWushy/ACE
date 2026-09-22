#!/usr/bin/env python3
"""
Aggregate rewards for sb_multiswitch runs.
Produces:
- sb_multiswitch/total_rewards_comparison.csv
- sb_multiswitch/late_rewards_comparison.csv
"""
from pathlib import Path
import csv

RUNS_DIR = Path("sb_multiswitch/runs")
OUT_TOTAL = Path("sb_multiswitch/total_rewards_comparison.csv")
OUT_LATE = Path("sb_multiswitch/late_rewards_comparison.csv")
LATE_EP = 20000  # 0-based: episodes >= 20000

def infer_and_sum(csv_path: Path):
    total = 0.0
    late = 0.0
    n = 0
    with csv_path.open() as fh:
        reader = csv.DictReader(fh)
        headers = reader.fieldnames
        if not headers:
            return 0.0, 0.0, 0
        
        ep_col = "episode" if "episode" in headers else headers[0]
        reward_col = "is_optimal" if "is_optimal" in headers else ("reward" if "reward" in headers else headers[1])

        fh.seek(0)
        reader = csv.DictReader(fh)
        for row in reader:
            try:
                ep = int(float(row.get(ep_col, n)))
                val = float(row.get(reward_col, 0.0))
            except Exception:
                continue
            total += val
            if ep >= LATE_EP:
                late += val
            n += 1
    return total, late, n

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
        total, late, n = infer_and_sum(chosen)
        parts = d.name.split("_")
        seed = parts[0]
        # type is everything after the seed up to 'multiswitch'
        rtype = "_".join(parts[1:]) if len(parts) > 1 else "unknown"
        if "icarus" in rtype:
            rtype = "ACE"
        elif "classic" in rtype:
            rtype = "classic"
            
        runs.append({"run": d.name, "seed": seed, "type": rtype, "csv": str(chosen), "total": total, "late": late, "n_rows": n})

    OUT_TOTAL.parent.mkdir(parents=True, exist_ok=True)
    with OUT_TOTAL.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["run","seed","type","total_reward","n_rows"])
        for r in runs:
            writer.writerow([r["run"], r["seed"], r["type"], f"{r['total']:.6f}", r["n_rows"]])

    with OUT_LATE.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["run","seed","type","late_reward","n_rows"])
        for r in runs:
            writer.writerow([r["run"], r["seed"], r["type"], f"{r['late']:.6f}", r["n_rows"]])

    print(f"Wrote {OUT_TOTAL} and {OUT_LATE}")

if __name__ == '__main__':
    main()
