import os
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor

# Sweep parameters
decays = [0.1, 0.5, 0.9]
boosts = [0.001, 0.01, 0.1]
trials = [1, 2, 3]

# Directories
base_dir = os.path.dirname(os.path.abspath(__file__))
result_dir = os.path.join(base_dir, "result")
sweep_plot_dir = os.path.join(result_dir, "sweep_graphs")
os.makedirs(sweep_plot_dir, exist_ok=True)

# Path to python in the active virtual environment
python_bin = "python"
script_path = os.path.join(base_dir, "tdstdpll.py")

def run_experiment(params):
    decay, boost, trial = params
    
    # Create a sub-folder for this parameter combination
    combo_dir = os.path.join(sweep_plot_dir, f"decay_{decay}_boost_{boost}")
    os.makedirs(combo_dir, exist_ok=True)
    
    # Naming the plot and log files uniquely for this trial
    plot_name = f"trial_{trial}.png"
    log_name = f"log_trial_{trial}.txt"
    log_path = os.path.join(combo_dir, log_name)
    
    # Setup environment variables specifically for this run
    env = os.environ.copy()
    env["SWEEP_LR_DECAY"] = str(decay)
    env["SWEEP_LR_BOOST"] = str(boost)
    env["SWEEP_PLOT_DIR"] = combo_dir
    env["SWEEP_PLOT_NAME"] = plot_name
    env["SWEEP_LOG_PATH"] = log_path
    
    print(f"Starting run: Decay={decay}, Boost={boost}, Trial={trial} | Logs -> {log_name}")
    start_time = time.time()
    
    # Run the script and write output to a log file
    with open(log_path, "w") as log_file:
        process = subprocess.Popen(
            [python_bin, script_path],
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT
        )
        process.wait()
        
    duration = time.time() - start_time
    print(f"Finished run: Decay={decay}, Boost={boost}, Trial={trial} in {duration:.1f} seconds. Plot -> {combo_dir}/{plot_name}")

if __name__ == "__main__":
    # Create the list of all combinations
    tasks = [(d, b, t) for d in decays for b in boosts for t in trials]
    
    print(f"Starting grid search: {len(tasks)} total tasks.")
    
    # Using a process pool to run 3 tests in parallel
    max_workers = 3
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        executor.map(run_experiment, tasks)
        
    print("All sweeping tasks completed!")
