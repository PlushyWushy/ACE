#!/usr/bin/env python3
"""
Switching Bandit with NE (Noise) AND ACh (Learning Rate).

Improvements:
- The "Surprise" signal now controls TWO variables:
  1. Noise (NE): To break out of the old habit.
  2. Learning Rate (ACh): To quickly learn the new habit.
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
ACTOR_THETA = 2.0   

# BASELINE PARAMETERS
BASE_LR = 1e-2      # Normal learning rate
BASE_NOISE = 0.5    # Normal exploration

# MODULATION GAINS
# When Surprise is high:
NE_GAIN = 5.0       # How much Noise boosts (Exploration)
ACH_GAIN = 10.0     # How much LR boosts (Plasticity) - CRITICAL for fast switching

SURPRISE_DECAY = 0.95 #lower since sparser reward

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
        var = F.softplus(self.var_head(x)) + 1e-4 
        return val, var

# ---------------------------------------------------------------------------
# 2. SNN Actor (Now accepts dynamic LR)
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
        
        # NE EFFECT: Increases Noise
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

    # UPDATED: Accepts dynamic learning rate
    def update(self, td_error: float, output_spikes: torch.Tensor, current_lr: float):
        eligibility = torch.outer(self.z_eps, output_spikes)
        
        # ACH EFFECT: Increases Step Size
        dw = current_lr * td_error * eligibility
        
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
    
    print(f"Start Experiment (NE + ACh Modulation): {episodes} episodes")
 
    
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
            
        # 2. Calculate Modulation Levels
        if ep == 1:
            current_ne = BASE_NOISE
            current_ach = BASE_LR
            td_error_val = 0.0
        else:
            raw_surprise = abs(td_error_val) / last_var.item()
            avg_surprise = SURPRISE_DECAY * avg_surprise + (1.0 - SURPRISE_DECAY) * raw_surprise
            
            # NE (Norepinephrine) -> Controls Noise
            current_ne = BASE_NOISE * (1.0 + avg_surprise * NE_GAIN)
            
            # ACh (Acetylcholine) -> Controls Learning Rate
            # If Surprise is high, we need to learn FAST.
            current_ach = BASE_LR * (1.0 + avg_surprise * ACH_GAIN)
            
        # 3. Actor Step
        input_spikes = torch.tensor([1.0], device=device)
        actor.reset_state()
        
        # Pass NE (Noise) here
        action, act_spikes = actor(input_spikes, noise_scale=current_ne)
        
        # 4. Env Step (+1/-1 Reward)
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
        
        critic_optim.zero_grad()
        (val_loss + var_loss).backward()
        critic_optim.step()
        
        # 7. Update Actor (Pass ACh/LR here)
        actor.update(td_error_val, act_spikes, current_lr=current_ach)
        
        history.append(action == optimal)
        
        if ep % 20 == 0:
            avg_opt = np.mean(history[-20:])
            w = actor.w.detach().numpy().flatten()
            print(f"Ep {ep:3d} | Opt%: {avg_opt*100:3.0f}% | "
                  f"NE: {current_ne:.2f} | ACh(LR): {current_ach:.4f} | "
                  f"W: [{w[0]:.2f}, {w[1]:.2f}]")
    
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
    plt.title('Switch Bandit - NE+ACh Modulation')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim([0, 105])
    plt.savefig('switch_bandit_rstdp_ne_ach_results.png', dpi=150, bbox_inches='tight')
    print(f"\nPlot saved as 'switch_bandit_rstdp_ne_ach_results.png'")

if __name__ == "__main__":
    train()