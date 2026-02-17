#!/usr/bin/env python3
"""
MLP Ablation for Lunar Lander:
- Standard Multi-Layer Perceptron (non-spiking).
- Trained with standard Policy Gradient (REINFORCE with Baseline).
- Includes the same control inversion switch at episode 5000.
"""

import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
import matplotlib.pyplot as plt

try:
    import gymnasium as gym
except ImportError:
    import gym

# --------------------------------------------------------------------------- 
# Constants & Hyperparameters
# --------------------------------------------------------------------------- 
GAMMA = 0.99
LR = 0.001
SEED = 1234
SWITCH_EPISODE = 5000

def set_global_seed(seed: int | None):
    if seed is None: return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

# --------------------------------------------------------------------------- 
# Networks
# --------------------------------------------------------------------------- 
class PolicyNet(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super(PolicyNet, self).__init__()
        self.fc1 = nn.Linear(obs_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, act_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return F.softmax(self.fc3(x), dim=-1)

class ValueNet(nn.Module):
    def __init__(self, obs_dim):
        super(ValueNet, self).__init__()
        self.fc1 = nn.Linear(obs_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# --------------------------------------------------------------------------- 
# Training Function
# --------------------------------------------------------------------------- 
def train(args):
    set_global_seed(args.seed)
    device = torch.device("cpu")

    env = gym.make("LunarLander-v3", render_mode="human" if args.render else None)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n

    policy = PolicyNet(obs_dim, act_dim).to(device)
    value = ValueNet(obs_dim).to(device)
    optimizer_p = optim.Adam(policy.parameters(), lr=args.lr)
    optimizer_v = optim.Adam(value.parameters(), lr=args.lr)

    print(f"Start MLP Ablation Lunar Lander. Episodes: {args.episodes}")

    reward_history = []

    for ep in range(args.episodes):
        obs, _ = env.reset(seed=args.seed + ep if args.seed else None)
        done = False
        total_reward = 0.0
        
        states, actions, rewards, log_probs = [], [], [], []
        
        inverted = (ep > SWITCH_EPISODE)

        while not done:
            st = torch.tensor(obs, dtype=torch.float32, device=device)
            states.append(st)
            
            probs = policy(st)
            m = torch.distributions.Categorical(probs)
            action = m.sample()
            log_probs.append(m.log_prob(action))
            
            action_code = action.item()
            # Control Inversion: Swap Left (1) and Right (3) thrusters
            if inverted:
                if action_code == 1: real_action = 3
                elif action_code == 3: real_action = 1
                else: real_action = action_code
            else:
                real_action = action_code

            next_obs, reward, terminated, truncated, _ = env.step(real_action)
            done = terminated or truncated
            
            rewards.append(reward)
            obs = next_obs
            total_reward += reward

        # Update policy and value nets (REINFORCE with Baseline)
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + GAMMA * G
            returns.insert(0, G)
        
        returns = torch.tensor(returns, dtype=torch.float32, device=device)
        states = torch.stack(states)
        log_probs = torch.stack(log_probs)
        
        # Value loss
        values = value(states).squeeze()
        value_loss = F.mse_loss(values, returns)
        
        optimizer_v.zero_grad()
        value_loss.backward()
        optimizer_v.step()
        
        # Policy loss
        advantages = (returns - values.detach())
        policy_loss = -(log_probs * advantages).mean()
        
        optimizer_p.zero_grad()
        policy_loss.backward()
        optimizer_p.step()

        reward_history.append(total_reward)
        avg_r = np.mean(reward_history[-20:]) if len(reward_history) > 0 else 0.0

        if ep % 100 == 0:
            status = "NORMAL" if not inverted else "INVERTED"
            print(f"Ep {ep:4d} | {status} | R: {total_reward:3.0f} | Avg: {avg_r:4.1f}")

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(range(len(reward_history)), reward_history, color="tab:blue", alpha=0.3)
    window_size = 20
    if len(reward_history) >= window_size:
        rolling_avg = np.convolve(reward_history, np.ones(window_size) / window_size, mode="valid")
        ax.plot(range(window_size - 1, len(reward_history)), rolling_avg, color="tab:blue", linewidth=2)
    ax.axvline(x=SWITCH_EPISODE, color="r", linestyle="--", label="Switch Point")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    ax.set_title("MLP Ablation Lunar Lander (REINFORCE + Baseline)")
    ax.grid(True, alpha=0.3)
    
    out_dir = f"ll/runs/{args.seed}_mlp_ablation" if args.seed is not None else "ll/runs/noseed_mlp_ablation"
    os.makedirs(out_dir, exist_ok=True)
    png_path = os.path.join(out_dir, f"{args.seed}_mlp_ablation.png")
    fig.savefig(png_path, dpi=150)
    
    csv_path = os.path.join(out_dir, f"{args.seed}_mlp_ablation.csv")
    with open(csv_path, "w") as fh:
        fh.write("episode,reward\n")
        for i, r in enumerate(reward_history):
            fh.write(f"{i},{r}\n")
    print(f"Plot saved to {png_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=10000)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()
    train(args)
