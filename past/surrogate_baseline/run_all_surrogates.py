#!/usr/bin/env python3
"""
Batch runner to execute 20 seeds (1 to 20) for all 5 aligned surrogate baselines.
Organizes all runs neatly inside surrogate_baseline/runs/.
Runs concurrently using a ProcessPoolExecutor.
"""

import os
import time
import subprocess
import concurrent.futures

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

TASKS = [
    {
        "script": "switch_bandit_surrogate.py",
        "episodes": 20000,
        "name": "bandit"
    },
    {
        "script": "switch_cartpole_surrogate.py",
        "episodes": 10000,
        "name": "cartpole"
    },
    {
        "script": "acrobot_surrogate.py",
        "episodes": 15000,
        "name": "acrobot"
    },
    {
        "script": "switch_bandit_gradual_surrogate.py",
        "episodes": 20000,
        "name": "bandit_gradual"
    },
    {
        "script": "switch_bandit_multiswitch_surrogate.py",
        "episodes": 60000,
        "name": "bandit_multiswitch"
    }
]

def run_single(script_name: str, episodes: int, seed: int):
    script_path = os.path.join(ROOT_DIR, script_name)
    cmd = [
        "python",
        script_path,
        "--episodes", str(episodes),
        "--seed", str(seed)
    ]
    start = time.time()
    try:
        # Run subprocess silently to avoid terminal clutter, checking for errors
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        elapsed = time.time() - start
        return {
            "status": "SUCCESS",
            "script": script_name,
            "seed": seed,
            "elapsed": elapsed,
            "output": result.stdout
        }
    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start
        return {
            "status": "FAILED",
            "script": script_name,
            "seed": seed,
            "elapsed": elapsed,
            "error": e.stderr or e.stdout or str(e)
        }

def main():
    print("=" * 60)
    print("Starting Batch Runner for Aligned Surrogate Baselines")
    print(f"Targeting 5 tasks x 20 seeds = 100 total runs.")
    print("=" * 60)

    jobs = []
    for task in TASKS:
        for seed in range(1, 21):
            jobs.append((task["script"], task["episodes"], seed))

    total_jobs = len(jobs)
    completed = 0
    failed = 0

    start_time = time.time()

    # Use 4-8 parallel workers (depending on CPU cores) - 6 is a good sweet spot
    max_workers = 6
    print(f"Launching pool with {max_workers} workers...")

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(run_single, script, ep, seed): (script, seed)
            for script, ep, seed in jobs
        }

        for future in concurrent.futures.as_completed(futures):
            script, seed = futures[future]
            completed += 1
            res = future.result()
            if res["status"] == "SUCCESS":
                print(f"[{completed}/{total_jobs}] SUCCESS: {script} | Seed: {seed} | Time: {res['elapsed']:.1f}s")
            else:
                failed += 1
                print(f"[{completed}/{total_jobs}] FAILED: {script} | Seed: {seed} | Time: {res['elapsed']:.1f}s")
                print(f"Error output:\n{res['error']}")

    total_elapsed = time.time() - start_time
    print("=" * 60)
    print("Batch Run Completed!")
    print(f"Total Time: {total_elapsed / 60.0:.1f} minutes")
    print(f"Successful Runs: {total_jobs - failed} / {total_jobs}")
    print(f"Failed Runs: {failed} / {total_jobs}")
    print("=" * 60)

if __name__ == "__main__":
    main()
