#!/usr/bin/env python3
"""
MLP Version of Classic CartPole (Switch Task).
Matches the architecture of classic.py but uses continuous rate-based neurons (MLP) instead of Spiking Neurons.

Architecture:
- Encoder: Place Cell (RBF) Encoder (Fixed, deterministic rates output).
- Actor: Linear Policy (Input -> Actions). Output is Logits -> Softmax.
- Critic: MLP (Input -> Hidden -> Value/Variance). 
  - Matches SNN Critic's structure: Input(2304) -> Hidden(256) -> [Val(1), Var(1)].
  - Uses ReLU activation instead of Spiking dynamics.

Mechanism:
- Episodes 0-2000: Normal CartPole.
- Episodes 2000+:  INVERTED CONTROLS (Left becomes Right).
"""

import argparse
import math
import numpy as np
import random
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import Tuple
import matplotlib.pyplot as plt

try:
    import gymnasium as gym
except ImportError:
    import gym

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DT = 0.02             
RHO_PC = 50.0         # Firing rate scale for Place Cells
GAMMA = 0.99          

# --- NEUROMODULATION PARAMETERS (Matches classic.py) ---
BASE_LR = 5e-4       
BASE_NOISE = 0.5      

# Gains (0 in classic.py, kept for compatibility)
NE_GAIN = 0         
ACH_GAIN = 0        

SURPRISE_DECAY = 0.90 

# Editable global seed
SEED = 123

def set_global_seed(seed: int | None):
    """Set seeds for python, numpy and torch for reproducibility."""
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass

# ---------------------------------------------------------------------------
# 1. Place Cell Encoder (Rate-Based)
# ---------------------------------------------------------------------------
class RatePlaceCellEncoder(nn.Module):
    def __init__(self, device: torch.device):
        super().__init__()
        self.device = device
        # Same grid as classic.py
        m = torch.linspace(-2.5, 2.5, 6, device=device)
        n = torch.linspace(-2.0, 2.0, 6, device=device) 
        p = torch.linspace(-0.25, 0.25, 8, device=device) 
        q = torch.linspace(-2.0, 2.0, 8, device=device) 
        
        mesh = torch.meshgrid(m, n, p, q, indexing="ij")
        centers = torch.stack([x.flatten() for x in mesh], dim=1)
        self.register_buffer("centers", centers)
        
        s1 = (m[1] - m[0]) / 1.5
        s2 = (n[1] - n[0]) / 1.5
        s3 = (p[1] - p[0]) / 1.5
        s4 = (q[1] - q[0]) / 1.5
        sigmas = torch.tensor([s1, s2, s3, s4], device=device).unsqueeze(0)
        self.register_buffer("sigmas", sigmas)
        self.n_neurons = centers.shape[0]

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        # Calculate RBF activation (rates)
        diff = (state.unsqueeze(0) - self.centers) ** 2
        exponent = -torch.sum(diff / (2 * self.sigmas ** 2), dim=1)
        rates = RHO_PC * torch.exp(exponent)
        
        # In classic.py: probs = 1.0 - torch.exp(-rates * DT)
        # We use these probabilities directly as continuous inputs to the MLP
        probs = 1.0 - torch.exp(-rates * DT)
        return probs

# ---------------------------------------------------------------------------
# 2. MLP Critic
# ---------------------------------------------------------------------------
class MLPCritic(nn.Module):
    def __init__(self, n_input: int, n_hidden: int = 256):
        super().__init__()
        self.fc1 = nn.Linear(n_input, n_hidden)
        self.fc_val = nn.Linear(n_hidden, 1)
        self.fc_var = nn.Linear(n_hidden, 1)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = F.relu(self.fc1(x))
        val = self.fc_val(h)
        # Variance must be positive
        var = F.softplus(self.fc_var(h)) + 1e-4
        return val, var

# ---------------------------------------------------------------------------
# 3. MLP Actor
# ---------------------------------------------------------------------------
class MLPActor(nn.Module):
    def __init__(self, n_input: int, device: torch.device):
        super().__init__()
        self.device = device
        # Linear policy (Input -> 2 Actions)
        # classic.py initializes weights with normal_(0.0, 0.05)
        self.fc = nn.Linear(n_input, 2, bias=False) # No bias to match simple weight matrix? Or keep bias? 
        # classic.py has `self.w` (n_input, 2). No bias mentions.
        nn.init.normal_(self.fc.weight, mean=0.0, std=0.05)
        
    def forward(self, x: torch.Tensor) -> Tuple[int, torch.distributions.Categorical]:
        logits = self.fc(x)
        # In classic.py, noise is added to membrane potential: v_mem = v_det + noise
        # In MLP, we use Softmax sampling which inherently has stochasticity.
        # However, to be "as similar as possible", we could add noise to logits before softmax?
        # But standard MLP RL uses categorical sampling.
        
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        return action.item(), dist

# ---------------------------------------------------------------------------
# 4. Training Loop
# ---------------------------------------------------------------------------
def train(episodes=4000, render=False, seed: int | None = None):
    device = torch.device("cpu") # Keep CPU for simple CartPole
    set_global_seed(seed)
    
    env = gym.make("CartPole-v1", render_mode="human" if render else None)
    
    # Seeding environment (Gym specific handling)
    if seed is not None:
        try:
            env.reset(seed=seed)
        except TypeError:
            pass
            
    encoder = RatePlaceCellEncoder(device)
    actor = MLPActor(encoder.n_neurons, device)
    critic = MLPCritic(encoder.n_neurons).to(device)
    
    # Optimizers
    # classic.py uses manual update for actor: w += lr * td * eligibility
    # This is equivalent to SGD on log_prob with td as weight.
    actor_optim = optim.SGD(actor.parameters(), lr=BASE_LR) 
    # classic.py uses Adam for Critic (lr=1e-3)
    critic_optim = optim.Adam(critic.parameters(), lr=1e-3)
    
    print(f"Start MLP Classic CartPole. Episodes: {episodes}")
    
    avg_surprise = 0.0
    reward_history = []
    td_history = []
    
    for ep in range(episodes):
        # Per-episode seeding logic
        if seed is not None:
             obs, _ = env.reset(seed=seed + ep)
        else:
             obs, _ = env.reset()
             
        obs_t = torch.tensor(obs, dtype=torch.float32, device=device)
        
        done = False
        total_reward = 0
        
        inverted = (ep > 500)
        
        # Neuromodulator Levels (Matches classic.py logic)
        current_ne = BASE_NOISE * (1.0 + avg_surprise * NE_GAIN)
        current_ach = BASE_LR * (1.0 + avg_surprise * ACH_GAIN)
        current_ne = min(current_ne, 5.0)
        
        # Update Actor learning rate dynamically
        for param_group in actor_optim.param_groups:
            param_group['lr'] = current_ach
            
        td_sum = 0.0
        td_count = 0
        
        while not done:
            # 1. Encode State
            input_feats = encoder(obs_t)
            
            # 2. Actor Action
            action, dist = actor(input_feats)
            log_prob = dist.log_prob(torch.tensor(action, device=device))
            
            real_action = 1 - action if inverted else action
            
            next_obs, reward, terminated, truncated, _ = env.step(real_action)
            done = terminated or truncated
            next_obs_t = torch.tensor(next_obs, dtype=torch.float32, device=device)
            
            # 3. Critic Step
            v_curr, var_curr = critic(input_feats)
            
            with torch.no_grad():
                if not done:
                    next_feats = encoder(next_obs_t)
                    v_next, _ = critic(next_feats)
                else:
                    v_next = torch.tensor([0.0], device=device)
            
            target = reward + GAMMA * v_next
            td_error = target - v_curr
            td_error_val = td_error.item()
            
            td_sum += abs(td_error_val)
            td_count += 1
            
            # 4. Critic Update
            val_loss = td_error.pow(2)
            target_var = td_error.detach().pow(2)
            var_loss = (var_curr - target_var).pow(2)
            
            critic_optim.zero_grad()
            (val_loss + var_loss).backward()
            critic_optim.step()
            
            # 5. Actor Update
            # Loss = -log_prob * td_error (Policy Gradient)
            # We use detach() on td_error because we don't want to backprop into Critic here
            actor_loss = -log_prob * td_error.detach()
            
            actor_optim.zero_grad()
            actor_loss.backward()
            actor_optim.step()
            
            # 6. Surprise Calculation
            current_sigma = torch.sqrt(var_curr.detach()).item()
            if current_sigma < 1e-3: current_sigma = 1e-3
            
            raw_surprise = abs(td_error_val) / current_sigma
            adjusted_surprise = max(0.0, raw_surprise - 1.0)
            avg_surprise = SURPRISE_DECAY * avg_surprise + (1.0 - SURPRISE_DECAY) * adjusted_surprise
            
            # Update Neuromodulation
            current_ne = BASE_NOISE * (1.0 + avg_surprise * NE_GAIN)
            current_ach = BASE_LR * (1.0 + avg_surprise * ACH_GAIN)
            current_ne = min(current_ne, 5.0)
            
            # Apply updated LR to Actor
            for param_group in actor_optim.param_groups:
                param_group['lr'] = current_ach
            
            obs_t = next_obs_t
            total_reward += reward
            
        reward_history.append(total_reward)
        mean_abs_td = (td_sum / td_count) if td_count > 0 else 0.0
        td_history.append(mean_abs_td)
        avg_r = np.mean(reward_history[-20:]) if len(reward_history) > 0 else 0
        
        if ep % 100 == 0:
            status = "NORMAL" if not inverted else "INVERTED"
            print(f"Ep {ep:4d} | {status} | R: {total_reward:3.0f} | Avg: {avg_r:4.1f} | "
                  f"Surprise: {avg_surprise:.2f} | TD: {mean_abs_td:.4f}")
                  
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(reward_history)), reward_history, linewidth=1, alpha=0.6)
    window_size = 20
    if len(reward_history) >= window_size:
        rolling_avg = np.convolve(reward_history, np.ones(window_size)/window_size, mode='valid')
        plt.plot(range(window_size-1, len(reward_history)), rolling_avg, linewidth=2, label=f'{window_size}-ep Avg')
    plt.axvline(x=2000, color='r', linestyle='--', label='Switch Point')
    plt.xlabel('Episode')
    plt.ylabel('Episode Reward')
    plt.title('Switch CartPole - MLP Classic')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    out_dir = f"cartpole/runs/{seed}_mlp_classic" if seed is not None else "cartpole/runs/noseed_mlp_classic"
    os.makedirs(out_dir, exist_ok=True)
    
    png_path = os.path.join(out_dir, f"{seed}_mlp_classic.png")
    plt.savefig(png_path, dpi=150, bbox_inches="tight")
    
    csv_path = os.path.join(out_dir, f"{seed}_mlp_classic.csv")
    with open(csv_path, "w") as fh:
        fh.write("episode,reward,avg_surprise,mean_abs_td\n")
        for i, (r, td) in enumerate(zip(reward_history, td_history)):
            fh.write(f"{i},{r},0.0,{td}\n")
            
    print(f"\nPlot saved as '{png_path}' and CSV saved as '{csv_path}'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=4000)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    
    train(args.episodes, args.render, seed=args.seed)
