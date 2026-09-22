#!/usr/bin/env python3
"""
Train the top 10 sweep configurations (ranked by mean total reward across all episodes)
for 5 seeds each, using src/cartpole/flagship.py.

Top 10 configs (from random_search_configs.csv, ranked by mean reward over 10k episodes):
  Rank  Config  Mean reward
     1      71       186.34
     2      38       175.49
     3      77       154.30
     4       2       144.40
     5      49       140.85
     6      25       135.45
     7      28       133.90
     8      18       132.49
     9      48       128.67
    10      54       127.87
"""

import os
import time
import subprocess
import concurrent.futures

SCRIPT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "flagship.py")
OUT_DIR     = os.path.join(os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "results", "cartpole")), "top10_runs")
SEEDS       = [1, 2, 3, 4, 5]
EPISODES    = 10000
MAX_WORKERS = 6

TOP10_CONFIGS = {
    71: dict(actor_lr_min=5e-05,  base_noise=0.25, ne_max=1.0, ach_max=1.0, ne_k=0.2, ach_k=10.0, ne_center=15.0, ach_center=5.0, critic_base_lr=0.001),
    38: dict(actor_lr_min=1e-05,  base_noise=0.5,  ne_max=1.0, ach_max=1.0, ne_k=0.2, ach_k=4.0,  ne_center=15.0, ach_center=5.0, critic_base_lr=0.0003),
    77: dict(actor_lr_min=5e-05,  base_noise=0.25, ne_max=3.0, ach_max=0.5, ne_k=0.2, ach_k=10.0, ne_center=15.0, ach_center=5.0, critic_base_lr=0.001),
     2: dict(actor_lr_min=5e-05,  base_noise=0.25, ne_max=1.0, ach_max=1.0, ne_k=0.2, ach_k=4.0,  ne_center=10.0, ach_center=5.0, critic_base_lr=0.0003),
    49: dict(actor_lr_min=1e-05,  base_noise=0.25, ne_max=1.0, ach_max=1.0, ne_k=0.2, ach_k=10.0, ne_center=15.0, ach_center=5.0, critic_base_lr=0.0003),
    25: dict(actor_lr_min=5e-05,  base_noise=1.0,  ne_max=1.0, ach_max=0.5, ne_k=0.1, ach_k=4.0,  ne_center=15.0, ach_center=5.0, critic_base_lr=0.0003),
    28: dict(actor_lr_min=1e-05,  base_noise=0.5,  ne_max=1.0, ach_max=1.0, ne_k=0.2, ach_k=4.0,  ne_center=15.0, ach_center=5.0, critic_base_lr=0.0003),
    18: dict(actor_lr_min=1e-05,  base_noise=0.25, ne_max=1.0, ach_max=1.0, ne_k=0.2, ach_k=10.0, ne_center=15.0, ach_center=5.0, critic_base_lr=0.001),
    48: dict(actor_lr_min=1e-05,  base_noise=0.25, ne_max=1.0, ach_max=1.0, ne_k=0.1, ach_k=10.0, ne_center=15.0, ach_center=5.0, critic_base_lr=0.001),
    54: dict(actor_lr_min=1e-05,  base_noise=0.5,  ne_max=3.0, ach_max=0.5, ne_k=0.1, ach_k=4.0,  ne_center=10.0, ach_center=5.0, critic_base_lr=0.001),
}


def run_one(config_id, seed, params):
    run_out = os.path.join(OUT_DIR, f"config_{config_id}_seed_{seed}")
    cmd = [
        "python3", SCRIPT_PATH,
        "--seed",     str(seed),
        "--episodes", str(EPISODES),
        "--config_id", str(config_id),
        "--out_dir",  run_out,
    ]
    for k, v in params.items():
        cmd.extend([f"--{k}", str(v)])

    start = time.time()
    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True)
        return config_id, seed, True, time.time() - start, ""
    except subprocess.CalledProcessError as e:
        return config_id, seed, False, time.time() - start, (e.stderr or e.stdout or str(e))[:400]


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    jobs = [(cid, seed, params) for cid, params in TOP10_CONFIGS.items() for seed in SEEDS]
    total = len(jobs)

    print("=" * 60)
    print(f"Top-10 config validation run  ({len(TOP10_CONFIGS)} configs × {len(SEEDS)} seeds = {total} runs)")
    print(f"Episodes per run: {EPISODES}  |  Workers: {MAX_WORKERS}")
    print(f"Output: {OUT_DIR}")
    print("=" * 60)

    completed = failed = 0
    start_all = time.time()

    with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {ex.submit(run_one, cid, seed, params): (cid, seed)
                   for cid, seed, params in jobs}

        for future in concurrent.futures.as_completed(futures):
            cid, seed = futures[future]
            config_id, seed_r, ok, elapsed, err = future.result()
            completed += 1
            if ok:
                print(f"[{completed:3d}/{total}] OK   config {config_id:2d} seed {seed_r} — {elapsed:.0f}s")
            else:
                failed += 1
                print(f"[{completed:3d}/{total}] FAIL config {config_id:2d} seed {seed_r} — {elapsed:.0f}s")
                print(f"         {err}")

    elapsed_total = time.time() - start_all
    print("=" * 60)
    print(f"Done in {elapsed_total/60:.1f} min  |  {total - failed}/{total} succeeded  |  {failed} failed")
    print("=" * 60)


if __name__ == "__main__":
    main()
