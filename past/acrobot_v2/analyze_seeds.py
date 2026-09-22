import csv
import glob
import os
import numpy as np

def analyze_seeds(pattern):
    files = glob.glob(pattern)
    results = []
    
    for f in sorted(files):
        seed_name = os.path.basename(os.path.dirname(f))
        rewards = []
        lrs = []
        
        with open(f, 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                rewards.append(float(row['reward']))
                lrs.append(float(row['actor_lr']))
        
        if not rewards:
            continue
            
        avg_last_500 = np.mean(rewards[-500:]) if len(rewards) >= 500 else np.mean(rewards)
        max_r = np.max(rewards)
        
        results.append({
            'seed': seed_name,
            'avg_last': avg_last_500,
            'max_r': max_r,
            'avg_lr': np.mean(lrs)
        })
    
    print(f"{'Seed':<15} | {'Last-500-Avg':<12} | {'Max-R':<8} | {'Avg-LR':<8}")
    print("-" * 60)
    for r in results:
        print(f"{r['seed']:<15} | {r['avg_last']:<12.1f} | {r['max_r']:<8.1f} | {r['avg_lr']:<8.6f}")

if __name__ == "__main__":
    print("ANALYZING FLAGSHIP RUNS:")
    analyze_seeds("acrobot/runs/*_flagship/data.csv")
    print("\nANALYZING CLASSIC RUNS:")
    analyze_seeds("acrobot/runs/*_classic/data.csv")
