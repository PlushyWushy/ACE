#!/usr/bin/env python3
"""
Switching Bandit with DAMPENED Meta-Learning (Reward: +1/-1).

Changes:
- Rewards are now +1 (Win) or -1 (Loss).
- This creates stronger LTD (Long Term Depression) signals when the agent fails,
  forcing it to "unlearn" bad habits faster than a 0.0 reward would.
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import Tuple
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DT = 0.02           
TAU_M = 0.02        
ACTOR_LR = 1e-2     
ACTOR_THETA = 2.0   

# NEUROMODULATION CONSTANTS
# With -1 rewards, errors are larger (max 2.0 instead of 1.0).
# We keep gain moderate to prevent panic from 1-2 bad luck results.
BASE_NOISE = 0.5    
NE_GAIN = 5.0       
SURPRISE_DECAY = 0.95 

# ---------------------------------------------------------------------------
# 1. Meta-MLP Critic
# ---------------------------------------------------------------------------
class MetaCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )
        self.value_head = nn.Linear(64, 1)
        self.var_head = nn.Linear(64, 1)
        
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.trunk(state)
        val = self.value_head(x)
        # Softplus + epsilon ensures variance is always positive
        var = F.softplus(self.var_head(x)) + 1e-4 
        return val, var

# ---------------------------------------------------------------------------
# 2. SNN Actor
# ---------------------------------------------------------------------------
class SNNActor(nn.Module):
    def __init__(self, device: torch.device):
        super().__init__()
        self.device = device
        self.w = torch.empty(1, 2, device=device).normal_(mean=0.0, std=0.1)
        self.z_eps = torch.zeros(1, device=device)
        self.decay_eps = math.exp(-DT / TAU_M)
        
    def reset_state(self):
        self.z_eps.zero_()
        
    def forward(self, input_spikes: torch.Tensor, noise_scale: float) -> Tuple[int, torch.Tensor]:
        self.z_eps = self.z_eps * self.decay_eps + input_spikes
        
        v_det = torch.matmul(self.z_eps, self.w)
        
        # Modulated Noise
        noise = torch.randn_like(v_det) * noise_scale
        v_mem = v_det + noise
        
        rho = 100.0 * torch.exp((v_mem - ACTOR_THETA) / 2.0)
        probs = 1.0 - torch.exp(-rho * DT)
        spikes = torch.bernoulli(torch.clamp(probs, 0.0, 1.0))
        
        s_cpu = spikes.cpu().numpy()
        if s_cpu[0] == 1 and s_cpu[1] == 0:
            action = 0
        elif s_cpu[0] == 0 and s_cpu[1] == 1:
            action = 1
        else:
            action = int(torch.argmax(v_mem).item())
            
        return action, spikes

    def update(self, td_error: float, output_spikes: torch.Tensor):
        eligibility = torch.outer(self.z_eps, output_spikes)
        dw = ACTOR_LR * td_error * eligibility
        self.w += dw
        self.w.clamp_(-10.0, 10.0)

# ---------------------------------------------------------------------------
# 3. Training Loop
# ---------------------------------------------------------------------------
def train(episodes=4000):
    device = torch.device("cpu")
    
    actor = SNNActor(device)
    critic = MetaCritic().to(device)
    critic_optim = optim.Adam(critic.parameters(), lr=1e-2)
    
    print(f"Start Experiment (Reward +1/-1): {episodes} episodes")

    
    input_state = torch.tensor([1.0], device=device)
    history = []
    reward_list = []  # Track all rewards for plotting
    
    last_var = torch.tensor([1.0], device=device)
    avg_surprise = 0.0  
    
    for ep in range(1, episodes + 1):
        # 1. Environment Logic
        if ep <= 2000:
            probs = [1, 0]
            optimal = 0
        else:
            probs = [0, 1]
            optimal = 1
            
        # 2. Calculate NE Level (Smoothed)
        if ep == 1:
            ne_level = BASE_NOISE
            td_error_val = 0.0
        else:
            # Surprise calculation using last error
            raw_surprise = abs(td_error_val) / last_var.item()
            avg_surprise = SURPRISE_DECAY * avg_surprise + (1.0 - SURPRISE_DECAY) * raw_surprise
            ne_level = BASE_NOISE * (1.0 + avg_surprise * NE_GAIN)
            
        # 3. Actor Step
        input_spikes = torch.tensor([1.0], device=device)
        actor.reset_state()
        action, act_spikes = actor(input_spikes, noise_scale=ne_level)
        
        # 4. Env Step (UPDATED REWARD)
        # Return +1 for win, -1 for loss
        reward = 1.0 if np.random.rand() < probs[action] else -1.0
        reward_list.append(reward)
        
        # 5. Critic Step
        v_curr, var_curr = critic(input_state)
        var_curr = torch.clamp(var_curr, min=0.01)
        last_var = var_curr.detach()
        
        # 6. Loss & Update Critic
        td_error = reward - v_curr
        td_error_val = td_error.item()
        
        val_loss = td_error.pow(2)
        target_var = td_error.detach().pow(2)
        var_loss = (var_curr - target_var).pow(2)
        
        total_loss = val_loss + var_loss
        critic_optim.zero_grad()
        total_loss.backward()
        critic_optim.step()
        
        # 7. Update Actor
        actor.update(td_error_val, act_spikes)
        
        history.append(action == optimal)
        
        if ep % 20 == 0:
            avg_opt = np.mean(history[-20:])
            w = actor.w.detach().numpy().flatten()
            print(f"Ep {ep:3d} | Optimal%: {avg_opt*100:3.0f}% | "
                  f"W: [{w[0]:.2f}, {w[1]:.2f}] | "
                  f"NE: {ne_level:.2f} | "
                  f"S_Avg: {avg_surprise:.2f}")
    
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
    plt.title('Switch Bandit - NE Modulation')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim([0, 105])
    plt.savefig('switch_bandit_rstdp_ne_results.png', dpi=150, bbox_inches='tight')
    print(f"\nPlot saved as 'switch_bandit_rstdp_ne_results.png'")

if __name__ == "__main__":
    train()

