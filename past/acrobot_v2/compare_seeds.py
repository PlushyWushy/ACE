import csv
import matplotlib.pyplot as plt
import glob
import os
import numpy as np

def plot_seeds(pattern, title, output):
    files = glob.glob(pattern)
    plt.figure(figsize=(12, 6))
    for f in sorted(files):
        seed_name = os.path.basename(os.path.dirname(f))
        episodes = []
        rewards = []
        with open(f, 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                episodes.append(int(row['episode']))
                rewards.append(float(row['reward']))
        
        if rewards:
            # Simple rolling mean using numpy
            window = 100
            if len(rewards) >= window:
                rolling_rewards = np.convolve(rewards, np.ones(window)/window, mode='valid')
                plt.plot(episodes[window-1:], rolling_rewards, label=seed_name, alpha=0.7)
            else:
                plt.plot(episodes, rewards, label=seed_name, alpha=0.3)
    
    plt.title(title)
    plt.xlabel('Episode')
    plt.ylabel('Reward (Rolling 100)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output)
    print(f"Saved plot to {output}")

if __name__ == "__main__":
    plot_seeds("acrobot/runs/*_flagship/data.csv", "Acrobot Flagship Seeds Comparison", "acrobot/seed_comparison.png")
