import subprocess
import concurrent.futures
import os
import time

def run_experiment(seed, ne_max, ach_max, base_noise, base_lr, tag, episodes=20000):
    cmd = [
        "python3",
        "icarus_upgraded.py",
        "--seed", str(seed),
        "--episodes", str(episodes),
        "--ne_max", str(ne_max),
        "--ach_max", str(ach_max),
        "--base_noise", str(base_noise),
        "--base_lr", str(base_lr),
        "--tag", tag
    ]
    print(f"Starting: {tag} | Seed: {seed}")
    start_time = time.time()
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        elapsed = time.time() - start_time
        return f"Finished: {tag} | Seed: {seed} | Time: {elapsed:.1f}s"
    except subprocess.CalledProcessError as e:
        return f"FAILED: {tag} | Seed: {seed} | Error: {e.stderr or e.stdout}"

def main():
    # Ensure we are in the correct directory
    #os.chdir("/Users/chairstands/Desktop/Icarus/sb")
    
    tasks = []
    seeds = range(1, 21)
    
    # Only NE modulation (LR constant at classic value 1e-2)
    for seed in seeds:
        tasks.append((seed, 2.0, 0.0, 0.0, 1e-2, "only_ne"))
        
    # Only ACh modulation (NE constant at 1.5)
    for seed in seeds:
        tasks.append((seed, 0.0, 1.0, 0.36, 1e-2, "only_ach"))
            
    print(f"Queueing {len(tasks)} runs...")
    
    # Using 8 workers for faster execution on Mac
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        future_to_task = {executor.submit(run_experiment, *task): task for task in tasks}
        for future in concurrent.futures.as_completed(future_to_task):
            print(future.result())

if __name__ == "__main__":
    main()
