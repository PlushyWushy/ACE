#!/usr/bin/env python3
from pathlib import Path
import csv
import os

# Adjust paths for sb_normal
BASE_DIR = Path("/Users/chairstands/Desktop/Icarus/sb")
RUNS_DIR = BASE_DIR / "sb_normal/runs"
OUT_TOTAL = BASE_DIR / "sb_normal/total_rewards_comparison.csv"
OUT_POST = BASE_DIR / "sb_normal/post_switch_rewards_comparison.csv"
SWITCH_EP = 10000  # Episodes 10,001..20,000 are post-switch

def infer_and_sum(csv_path: Path):
    total = 0.0
    post = 0.0
    n = 0
    with csv_path.open() as fh:
        reader = csv.DictReader(fh)
        headers = reader.fieldnames
        if not headers:
            return 0.0, 0.0, 0
        
        # Determine columns
        ep_col = "episode" if "episode" in headers else headers[0]
        reward_col = "is_optimal" if "is_optimal" in headers else ("reward" if "reward" in headers else headers[1])
        
        for row in reader:
            try:
                ep = int(row[ep_col])
                val = float(row[reward_col])
                total += val
                if ep >= SWITCH_EP:
                    post += val
                n += 1
            except (ValueError, KeyError):
                continue
    return total, post, n

def main():
    if not RUNS_DIR.exists():
        print(f"Runs directory not found: {RUNS_DIR}")
        return

    runs = []
    # Filter for only_ne and only_ach folders to avoid classic/icarus_upgraded if they exist
    target_types = ["only_ne", "only_ach"]
    
    for d in sorted(RUNS_DIR.iterdir(), key=lambda p: p.name):
        if not d.is_dir():
            continue
            
        parts = d.name.split("_")
        if len(parts) < 2:
            continue
            
        seed = parts[0]
        rtype = "_".join(parts[1:])
        
        # Only process the ablation types requested if they were specifically mentioned
        # Actually, let's process all types found in sb_normal/runs
        
        csv_file = d / "data.csv"
        if not csv_file.exists():
            continue
            
        total, post, n = infer_and_sum(csv_file)
        runs.append({
            "run": d.name,
            "seed": seed,
            "type": rtype,
            "total_reward": total,
            "post_switch_reward": post,
            "n_rows": n
        })

    if not runs:
        print("No valid runs found to consolidate.")
        return

    # Write total summary
    with OUT_TOTAL.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["run", "seed", "type", "total_reward", "n_rows"])
        for r in runs:
            writer.writerow([r["run"], r["seed"], r["type"], f"{r['total_reward']:.6f}", r["n_rows"]])

    # Write post-switch summary
    with OUT_POST.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["run", "seed", "type", "post_switch_reward", "n_rows"])
        for r in runs:
            writer.writerow([r["run"], r["seed"], r["type"], f"{r['post_switch_reward']:.6f}", r["n_rows"]])

    print(f"Wrote {OUT_TOTAL} and {OUT_POST}")

    # Print summary table by type
    stats = {}
    for r in runs:
        t = r["type"]
        if t not in stats:
            stats[t] = {"total": [], "post": []}
        stats[t]["total"].append(r["total_reward"])
        stats[t]["post"].append(r["post_switch_reward"])

    print("\nAverages by Type:")
    print(f"{'Type':20s} | {'Avg Total':12s} | {'Avg Post-Switch':15s} | {'Count':5s}")
    print("-" * 60)
    for t, data in sorted(stats.items()):
        avg_total = sum(data["total"]) / len(data["total"])
        avg_post = sum(data["post"]) / len(data["post"])
        print(f"{t:20s} | {avg_total:12.2f} | {avg_post:15.2f} | {len(data['total']):5d}")

if __name__ == "__main__":
    main()
