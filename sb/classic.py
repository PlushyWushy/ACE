#!/usr/bin/env python3
"""
Switching Bandit Experiment (Classic Hybrid R-STDP).

Task:
- 2-Armed Bandit.
- Episodes 1-200: Arm 0 (80%), Arm 1 (20%).
- Episodes 201-400: Arm 0 (20%), Arm 1 (80%).

System:
- Hybrid MLP Critic + SNN Actor (Universal Weights).
- Based on `sorta_plausible_rstdp.py`.
"""

import argparse
import math
import numpy as np
import random
import os
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Tuple
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DT = 0.02           # Simulation time step
TAU_M = 0.02        # Membrane time constant
ACTOR_LR = 1e-2     # Actor learning rate (Higher for bandit)
ACTOR_THETA = 2.0   # Actor threshold
GAMMA = 0.0         # Discount factor (Bandit is immediate reward)

# ---------------------------------------------------------------------------
# 1. MLP Critic (The "Brain")
# ---------------------------------------------------------------------------

class MLPCritic(nn.Module):
    """
    Standard Feedforward Network for Value Estimation V(s).
    Input: 1 (Bias)
    Output: 1 (Value)
    """
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)

# ---------------------------------------------------------------------------
# 2. SNN Actor (The "Implant" - Universal Weights)
# ---------------------------------------------------------------------------

class SNNActor(nn.Module):
    """
    Learns Policy using R-STDP.
    Input: 1 (Bias)
    Output: 2 (Arm 0, Arm 1)
    """
    def __init__(self, device: torch.device):
        super().__init__()
        self.device = device
        
        # Weights: [1, 2]
        # Universal weights (can be negative/positive)
        self.w = torch.empty(1, 2, device=device).normal_(mean=0.0, std=0.1)
        
        # Traces
        self.z_eps = torch.zeros(1, device=device)
        self.decay_eps = math.exp(-DT / TAU_M)
        
    def reset_state(self):
        self.z_eps.zero_()
        
    def forward(self, input_spikes: torch.Tensor) -> Tuple[int, torch.Tensor]:
        # Update trace
        self.z_eps = self.z_eps * self.decay_eps + input_spikes
        
        # Potential
        v_mem = torch.matmul(self.z_eps, self.w)
        
        # Spiking
        rho = 100.0 * torch.exp((v_mem - ACTOR_THETA) / 2.0)
        probs = 1.0 - torch.exp(-rho * DT)
        spikes = torch.bernoulli(torch.clamp(probs, 0.0, 1.0))
        
        # Action Selection
        s_cpu = spikes.cpu().numpy()
        if s_cpu[0] == 1 and s_cpu[1] == 0:
            action = 0
        elif s_cpu[0] == 0 and s_cpu[1] == 1:
            action = 1
        elif s_cpu[0] == 1 and s_cpu[1] == 1:
            # Tie-break with potential
            action = int(torch.argmax(v_mem).item())
        else:
            # Argmax fallback (Practical)
            action = int(torch.argmax(v_mem).item())
            
        return action, spikes

    def update(self, td_error: float, output_spikes: torch.Tensor):
        """
        R-STDP Update: dW = lr * delta * (Pre_trace * Post_spike)
        """
        eligibility = torch.outer(self.z_eps, output_spikes)
        dw = ACTOR_LR * td_error * eligibility
        self.w += dw
        self.w.clamp_(-10.0, 10.0)

# ---------------------------------------------------------------------------
# 3. Training Loop
# ---------------------------------------------------------------------------

def set_global_seed(seed: int | None):
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train(episodes=4000, seed: int | None = None):
    set_global_seed(seed)
    device = torch.device("cpu")
    
    # Components
    actor = SNNActor(device)
    critic = MLPCritic().to(device)
    critic_optim = optim.Adam(critic.parameters(), lr=1e-2)
    
    print(f"Start Switching Bandit Experiment: {episodes} episodes")
    print("Episodes 1-200: Arm 0 (80%), Arm 1 (20%)")
    print("Episodes 201-400: Arm 0 (20%), Arm 1 (80%)")
    
    # Constant input (Bias)
    input_state = torch.tensor([1.0], device=device)
    
    history = []
    reward_list = []  # Track all rewards for plotting
    
    for ep in range(1, episodes + 1):
        # 1. Determine Probabilities
        if ep <= 2000:
            probs = [1, 0]
            optimal = 0
        else:
            probs = [0, 1]
            optimal = 1
            
        # 2. Actor Step
        # Input spikes: Bias neuron always fires
        input_spikes = torch.tensor([1.0], device=device)
        
        # Reset actor state (it's a one-step task, but traces decay)
        # For bandit, we can treat each episode as a single step or a short duration.
        # Let's treat it as a single interaction.
        actor.reset_state()
        
        # We need to run for at least one step to generate spikes
        # Actually, let's run for a few steps to allow potential to build up?
        # Or just one step with pre-charged trace?
        # Let's do 1 step.
        action, act_spikes = actor(input_spikes)
        
        # 3. Environment Step
        reward = 1.0 if np.random.rand() < probs[action] else -1.0
        reward_list.append(reward)
        
        # 4. Critic Step
        v_curr = critic(input_state)
        
        # 5. TD Error
        # Target = Reward (Gamma=0)
        td_error = reward - v_curr
        
        # 6. Update Critic
        critic_loss = td_error.pow(2)
        critic_optim.zero_grad()
        critic_loss.backward()
        critic_optim.step()
        
        # 7. Update Actor
        delta = td_error.item()
        actor.update(delta, act_spikes)
        
        history.append(action == optimal)
        
        if ep % 20 == 0:
            avg_opt = np.mean(history[-20:])
            w = actor.w.detach().numpy().flatten()
            print(f"Ep {ep:3d} | Optimal%: {avg_opt*100:3.0f}% | "
                  f"W: [{w[0]:.2f}, {w[1]:.2f}] | "
                  f"V: {v_curr.item():.2f}")
    
    # Plot results
    plot_episodes = list(range(50, episodes + 1, 50))
    reward_rates = []
    for i in range(50, episodes + 1, 50):
        window = reward_list[i-50:i]
        reward_rate = (sum(1 for r in window if r > 0) / len(window)) * 100
        reward_rates.append(reward_rate)
    
    plt.figure(figsize=(10, 6))
    plt.plot(plot_episodes, reward_rates, linewidth=2)
    plt.axvline(x=2000, color='r', linestyle='--', label='Switch Point')
    plt.xlabel('Episode')
    plt.ylabel('Reward Rate (%)')
    plt.title('Switch Bandit - Classic R-STDP')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim([0, 105])
    # Ensure output directory exists and save both CSV and PNG in runs folder
    out_dir = f"switch_bandit_experiment/runs/{seed}_classic" if seed is not None else "switch_bandit_experiment/runs/noseed_classic"
    os.makedirs(out_dir, exist_ok=True)

    png_path = os.path.join(out_dir, f"{seed}_classic.png")
    plt.savefig(png_path, dpi=150, bbox_inches='tight')

    csv_path = os.path.join(out_dir, f"{seed}_classic.csv")
    with open(csv_path, "w") as fh:
        fh.write("episode,reward,is_optimal\n")
        for i, (r, opt) in enumerate(zip(reward_list, history)):
            fh.write(f"{i},{r},{int(opt)}\n")

    print(f"\nPlot saved as '{png_path}' and CSV saved as '{csv_path}'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=None, help="Random seed (optional)")
    args = parser.parse_args()

    train(args.episodes, seed=args.seed)
