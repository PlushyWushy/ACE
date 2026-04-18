import sys
import re
import matplotlib.pyplot as plt

def parse_log(file_path):
    episodes = []
    rewards = []
    avg_rewards = []
    variances = []
    lrs = []
    
    pattern = re.compile(
        r"(\d+): Ret:\s*([-\d.]+);\s*"
        r"Last 100 Avg Ret:\s*([-\d.]+);\s*"
        r"Var:\s*([-\d.]+);\s*"
        r"NE:.*?"
        r"ActLR:\s*([-\d.]+);"
    )
    
    with open(file_path, 'r') as f:
        for line in f:
            match = pattern.search(line)
            if match:
                ep, ret, avg_ret, var, lr = match.groups()
                episodes.append(int(ep))
                rewards.append(float(ret))
                avg_rewards.append(float(avg_ret))
                variances.append(float(var))
                lrs.append(float(lr))
                
    return episodes, rewards, avg_rewards, variances, lrs

def plot_metrics(file_path, output_path):
    episodes, rewards, avg_rewards, variances, lrs = parse_log(file_path)
    
    if not episodes:
        print(f"No data found in {file_path}")
        return

    plt.figure(figsize=(12, 10))

    # 1. Reward Plot
    plt.subplot(3, 1, 1)
    plt.plot(episodes, rewards, label='Return', color='lightgray', alpha=0.6)
    plt.plot(episodes, avg_rewards, label='Last 100 Avg Return', color='blue', linewidth=2)
    plt.ylabel('Reward')
    plt.title('Training Reward and Average Reward vs Episode')
    plt.legend()
    plt.grid(True)

    # 2. Variance Plot
    plt.subplot(3, 1, 2)
    plt.plot(episodes, variances, label='Variance', color='orange')
    plt.ylabel('Variance')
    plt.yscale('log')  # Variance can have small details
    plt.title('Log Variance vs Episode')
    plt.grid(True)

    # 3. Learning Rate Plot
    plt.subplot(3, 1, 3)
    plt.plot(episodes, lrs, label='Actor LR', color='green')
    plt.xlabel('Episode')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate vs Episode')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Saved plot: {output_path}")

if __name__ == '__main__':
    log_file = sys.argv[1] if len(sys.argv) > 1 else 'result/log_chunk1.txt'
    output_file = sys.argv[2] if len(sys.argv) > 2 else 'result/all_metrics.png'
    plot_metrics(log_file, output_file)
