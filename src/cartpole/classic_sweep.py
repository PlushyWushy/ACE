#!/usr/bin/env python3
"""
Grid search over actor learning rate x base noise for classic.py, 3 seeds per
config. Writes to results/cartpole/classic_hyperparam_search/.
"""

import os
import time
import subprocess
import concurrent.futures
import itertools
import csv

SCRIPT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "classic.py")
OUT_DIR     = os.path.join(os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "results", "cartpole")), "classic_hyperparam_search")
SEEDS       = [1, 2, 3]
EPISODES    = 10000
MAX_WORKERS = 6

ACTOR_LR_MIN_VALUES = [1e-4, 3e-4, 5e-4, 1e-3, 3e-3]
BASE_NOISE_VALUES   = [0.5, 1.0, 1.5, 2.0, 3.0]

# Build ordered config list so config_id is stable
CONFIGS = {}
for cid, (lr_min, noise) in enumerate(itertools.product(ACTOR_LR_MIN_VALUES, BASE_NOISE_VALUES)):
    # actor_lr_max = actor_lr_min keeps the LR fixed
    CONFIGS[cid] = dict(actor_lr_min=lr_min, actor_lr_max=lr_min, base_noise=noise)


def run_one(config_id, seed, params):
    run_out = OUT_DIR
    stem    = f"config_{config_id}_seed_{seed}"
    cmd = [
        "python3", SCRIPT_PATH,
        "--seed",      str(seed),
        "--episodes",  str(EPISODES),
        "--config_id", str(config_id),
        "--out_dir",   run_out,
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

    # Write config manifest
    manifest = os.path.join(OUT_DIR, "configs.csv")
    with open(manifest, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["config_id", "actor_lr_min", "actor_lr_max", "base_noise"])
        w.writeheader()
        for cid, params in sorted(CONFIGS.items()):
            w.writerow({"config_id": cid, **params})
    print(f"Config manifest → {manifest}")

    jobs  = [(cid, seed, params) for cid, params in CONFIGS.items() for seed in SEEDS]
    total = len(jobs)

    print("=" * 60)
    print(f"Classic CartPole grid search  ({len(CONFIGS)} configs × {len(SEEDS)} seeds = {total} runs)")
    print(f"ACTOR_LR_MIN values: {ACTOR_LR_MIN_VALUES}")
    print(f"BASE_NOISE values:   {BASE_NOISE_VALUES}")
    print(f"Episodes per run:  {EPISODES}  |  Workers: {MAX_WORKERS}")
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
            p = CONFIGS[config_id]
            tag = f"lr_min={p['actor_lr_min']:.0e} noise={p['base_noise']}"
            if ok:
                print(f"[{completed:3d}/{total}] OK   config {config_id:2d} ({tag}) seed {seed_r} — {elapsed:.0f}s")
            else:
                failed += 1
                print(f"[{completed:3d}/{total}] FAIL config {config_id:2d} ({tag}) seed {seed_r} — {elapsed:.0f}s")
                print(f"         {err}")

    elapsed_total = time.time() - start_all
    print("=" * 60)
    print(f"Done in {elapsed_total/60:.1f} min  |  {total - failed}/{total} succeeded  |  {failed} failed")
    print("=" * 60)


if __name__ == "__main__":
    main()
