import subprocess
import concurrent.futures
import os
import random
import csv
import itertools
from collections import defaultdict

# Define hyperparameter grid
HYPERPARAMETERS = {
    'actor_lr_min': [1e-5, 5e-5, 1e-4],
    'base_noise': [0.25, 0.5, 1.0],
    'ne_max': [1.0, 3.0],
    'ach_max': [0.5, 1.0],
    'ne_k': [0.1, 0.2],
    'ach_k': [4.0, 10.0],
    'ne_center': [10.0, 15.0],
    'ach_center': [5.0, 8.0],
    'critic_base_lr': [3e-4, 1e-3, 3e-3]
}

NUM_RANDOM_CONFIGS = 100
SEEDS = [1, 2, 3]
EPISODES = 10000 # Using a smaller number of episodes for the random search. Will adjust if necessary.
SCRIPT_PATH = "/Users/a../Desktop/Icarus/cartpole_successful/flagship.py"

def generate_random_configs(grid, num_configs):
    keys, values = zip(*grid.items())
    all_possible_configs = [dict(zip(keys, v)) for v in itertools.product(*values)]
    random.shuffle(all_possible_configs)
    return all_possible_configs[:num_configs]

def run_experiment(config, seed, config_id):
    out_dir = "cartpole_successful/hyperparam_search"
    cmd = [
        "python3", SCRIPT_PATH,
        "--seed", str(seed),
        "--episodes", str(EPISODES),
        "--config_id", str(config_id),
        "--out_dir", out_dir
    ]
    for k, v in config.items():
        cmd.extend([f"--{k}", str(v)])
        
    try:
        # We need to capture the reward to evaluate the config.
        # Assuming the script prints the final average reward or we parse it from saved CSVs.
        # Looking at flagship.py, it saves to cartpole_successful/runs_saved/
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        # return the config id and seed so we can look up the result
        return config_id, seed, True, ""
    except subprocess.CalledProcessError as e:
        return config_id, seed, False, e.stderr

def main():
    configs = generate_random_configs(HYPERPARAMETERS, NUM_RANDOM_CONFIGS)
    
    tasks = []
    for config_id, config in enumerate(configs):
        for seed in SEEDS:
            tasks.append((config, seed, config_id))
            
    print(f"Queueing {len(tasks)} runs ({NUM_RANDOM_CONFIGS} configs * {len(SEEDS)} seeds)...")
    
    results = defaultdict(list)
    
    # We use ThreadPoolExecutor to run tasks in parallel.
    # Because these are RL tasks, they might be CPU bound, but we'll try 8 workers.
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        future_to_task = {executor.submit(run_experiment, *task): task for task in tasks}
        
        for i, future in enumerate(concurrent.futures.as_completed(future_to_task)):
            config_id, seed, success, error = future.result()
            if success:
                print(f"[{i+1}/{len(tasks)}] Config {config_id} Seed {seed} Finished Successfully.")
            else:
                print(f"[{i+1}/{len(tasks)}] Config {config_id} Seed {seed} FAILED.")
                
    print("Sweep complete. Please aggregate results to find the best configs.")
    
    # Save the configs used so we know which ID maps to which parameters
    with open('random_search_configs.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['config_id'] + list(HYPERPARAMETERS.keys()))
        writer.writeheader()
        for i, config in enumerate(configs):
            row = {'config_id': i}
            row.update(config)
            writer.writerow(row)

if __name__ == "__main__":
    main()
