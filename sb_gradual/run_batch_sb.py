import subprocess
import concurrent.futures
import os
import time

def run_experiment(script_path, seed, episodes):
    cmd = [
        "python",
        script_path,
        "--seed", str(seed),
        "--episodes", str(episodes)
    ]
    name = os.path.basename(script_path).replace(".py", "")
    print(f"Starting: {name} | Seed: {seed}")
    start_time = time.time()
    
    try:
        # Run and capture output to prevent terminal flooding
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        elapsed = time.time() - start_time
        return f"Finished: {name} | Seed: {seed} | Time: {elapsed:.1f}s"
    except subprocess.CalledProcessError as e:
        return f"FAILED: {name} | Seed: {seed} | Error: {e.stderr}"

def main():
    base_dir = "/Users/a../Desktop/Icarus/sb_gradual"
    scripts = [
        ("icarus_upgraded.py", 20000),
        ("classic.py", 20000)
    ]
    
    seeds = range(1, 21)
    tasks = []
    
    for script, episodes in scripts:
        full_path = os.path.join(base_dir, script)
        for seed in seeds:
            tasks.append((full_path, seed, episodes))
            
    print(f"Queueing {len(tasks)} runs...")
    
    # Using 4 workers for a balance of speed and stability
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        future_to_task = {executor.submit(run_experiment, *task): task for task in tasks}
        for future in concurrent.futures.as_completed(future_to_task):
            print(future.result())

if __name__ == "__main__":
    main()
