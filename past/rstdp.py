#!/usr/bin/env python3
"""
Switch CartPole with NE (Noise) AND ACh (Learning Rate).
STABLE LOCAL LEARNING VERSION.

Fixes applied:
1. Clamped max ACh (Learning Rate) to prevent weight explosion.
2. Clamped Critic weights (Stability).
3. Added NaN protection in spike generation.
"""

import argparse
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
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
RHO_PC = 50.0         
TAU_M = 0.02          
ACTOR_THETA = 2.0     
GAMMA = 0.99          

# --- NEUROMODULATION PARAMETERS ---
BASE_LR = 5e-3        
BASE_NOISE = 0.5      

# Gains 
NE_GAIN = 2.0         # Boosts Noise (Exploration)
ACH_GAIN = 5.0        # Boosts Plasticity (Learning Speed)

# SAFETY LIMITS (New)
MAX_ACH = 0.1         # Cap learning rate to maintain stability
MAX_NE = 5.0          # Cap noise
SURPRISE_DECAY = 0.95 

# --- CRITIC PARAMETERS ---
CRITIC_TAU_TRACE = 0.05   
CRITIC_WEIGHT_DECAY = 1e-4 

# ---------------------------------------------------------------------------
# 1. Place Cell Encoder
# ---------------------------------------------------------------------------
class PlaceCellEncoder(nn.Module):
    def __init__(self, device: torch.device):
        super().__init__()
        self.device = device
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
        diff = (state.unsqueeze(0) - self.centers) ** 2
        exponent = -torch.sum(diff / (2 * self.sigmas ** 2), dim=1)
        rates = RHO_PC * torch.exp(exponent)
        probs = 1.0 - torch.exp(-rates * DT)
        spikes = torch.bernoulli(torch.clamp(probs, 0.0, 1.0))
        return spikes

# ---------------------------------------------------------------------------
# 2. Local Learning Critic (Stable)
# ---------------------------------------------------------------------------
class LocalCritic(nn.Module):
    def __init__(self, n_input: int, device: torch.device):
        super().__init__()
        self.device = device
        self.w_val = torch.zeros(n_input, device=device).normal_(0, 0.01)
        self.w_var = torch.zeros(n_input, device=device).normal_(0, 0.01)
        self.epsp_trace = torch.zeros(n_input, device=device)
        self.trace_decay = math.exp(-DT / CRITIC_TAU_TRACE)
        
    def reset_state(self):
        self.epsp_trace.zero_()
        
    def forward(self, input_spikes: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        self.epsp_trace = self.epsp_trace * self.trace_decay + input_spikes
        
        val = torch.dot(self.w_val, self.epsp_trace)
        
        # Variance calculation with stability check
        raw_var = torch.dot(self.w_var, self.epsp_trace)
        var = F.softplus(raw_var) + 1e-4
        
        return val, var

    def update(self, td_error: float, current_lr: float):
        # Update Value Weights
        dw_val = current_lr * td_error * self.epsp_trace
        self.w_val += dw_val
        self.w_val *= (1.0 - CRITIC_WEIGHT_DECAY)
        self.w_val.clamp_(-10.0, 10.0) # CLAMP ADDED

        # Update Variance Weights
        current_var = F.softplus(torch.dot(self.w_var, self.epsp_trace)).item()
        var_error = (td_error ** 2) - current_var
        
        # Clamp variance error to prevent explosions
        var_error = max(min(var_error, 50.0), -50.0)
        
        dw_var = current_lr * var_error * self.epsp_trace
        self.w_var += dw_var
        self.w_var *= (1.0 - CRITIC_WEIGHT_DECAY)
        self.w_var.clamp_(-10.0, 10.0) # CLAMP ADDED

# ---------------------------------------------------------------------------
# 3. Modulated Actor (Stable)
# ---------------------------------------------------------------------------
class ModulatedActor(nn.Module):
    def __init__(self, n_input: int, device: torch.device):
        super().__init__()
        self.device = device
        self.w = torch.empty(n_input, 2, device=device).normal_(0.0, 0.05)
        self.z_eps = torch.zeros(n_input, device=device)
        self.decay_eps = math.exp(-DT / TAU_M)
        
    def reset_state(self):
        self.z_eps.zero_()
        
    def forward(self, input_spikes: torch.Tensor, noise_scale: float) -> Tuple[int, torch.Tensor]:
        self.z_eps = self.z_eps * self.decay_eps + input_spikes
        v_det = torch.matmul(self.z_eps, self.w)
        
        noise = torch.randn_like(v_det) * noise_scale
        v_mem = v_det + noise
        
        rho = 100.0 * torch.exp((v_mem - ACTOR_THETA) / 2.0)
        
        # SAFETY CHECK: Handle Infinite/NaN rho
        rho = torch.nan_to_num(rho, posinf=1e6, neginf=0.0)
        
        probs = 1.0 - torch.exp(-rho * DT)
        
        # SAFETY CHECK: Ensure probs is clean before bernoulli
        probs = torch.nan_to_num(probs, nan=0.0, posinf=1.0, neginf=0.0)
        probs = torch.clamp(probs, 0.0, 1.0)
        
        spikes = torch.bernoulli(probs)
        
        s_cpu = spikes.cpu().numpy()
        if s_cpu[0] == 1 and s_cpu[1] == 0:
            action = 0
        elif s_cpu[0] == 0 and s_cpu[1] == 1:
            action = 1
        else:
            action = int(torch.argmax(v_mem).item())
            
        return action, spikes

    def update(self, td_error: float, output_spikes: torch.Tensor, current_lr: float):
        eligibility = torch.outer(self.z_eps, output_spikes)
        dw = current_lr * td_error * eligibility
        self.w += dw
        self.w.clamp_(-10.0, 10.0)

# ---------------------------------------------------------------------------
# 4. Training Loop
# ---------------------------------------------------------------------------
def train(episodes=4000, render=False):
    device = torch.device("cpu")
    env = gym.make("CartPole-v1", render_mode="human" if render else None)
    
    encoder = PlaceCellEncoder(device)
    actor = ModulatedActor(encoder.n_neurons, device)
    critic = LocalCritic(encoder.n_neurons, device)
    
    print(f"Start Switch CartPole (NE+ACh + LOCAL CRITIC + STABILITY). Episodes: {episodes}")
    print("Episodes 0-150: Normal Controls.")
    print("Episodes 151+ : INVERTED CONTROLS.")
    
    avg_surprise = 0.0
    reward_history = []
    
    for ep in range(episodes):
        obs, _ = env.reset()
        obs_t = torch.tensor(obs, dtype=torch.float32, device=device)
        
        actor.reset_state()
        critic.reset_state()
        
        done = False
        total_reward = 0
        
        inverted = (ep > 2000)
        
        # Calculate Neuromodulators with CLAMPING
        # 1. ACh (Learning Rate)
        raw_ach = BASE_LR * (1.0 + avg_surprise * ACH_GAIN)
        current_ach = min(raw_ach, MAX_ACH) # Safety Cap
        
        # 2. NE (Noise/Exploration)
        raw_ne = BASE_NOISE * (1.0 + avg_surprise * NE_GAIN)
        current_ne = min(raw_ne, MAX_NE)    # Safety Cap
        
        while not done:
            spikes = encoder(obs_t)
            action_code, act_spikes = actor(spikes, noise_scale=current_ne)
            
            real_action = 1 - action_code if inverted else action_code
            
            next_obs, reward, terminated, truncated, _ = env.step(real_action)
            done = terminated or truncated
            next_obs_t = torch.tensor(next_obs, dtype=torch.float32, device=device)
            
            v_curr, var_curr = critic(spikes)
            
            with torch.no_grad():
                if not done:
                    next_spikes = encoder(next_obs_t)
                    temp_trace = critic.epsp_trace * critic.trace_decay + next_spikes
                    v_next = torch.dot(critic.w_val, temp_trace)
                else:
                    v_next = torch.tensor(0.0, device=device)
            
            # Clip TD Error for stability (bio-plausible saturation)
            td_error = reward + GAMMA * v_next - v_curr
            td_error_val = np.clip(td_error.item(), -10.0, 10.0) 
            
            critic.update(td_error_val, current_lr=current_ach)
            actor.update(td_error_val, act_spikes, current_lr=current_ach)
            
            # Surprise Dynamics
            current_sigma = torch.sqrt(var_curr).item()
            if current_sigma < 1e-3: current_sigma = 1e-3
            
            raw_surprise = abs(td_error_val) / current_sigma
            adjusted_surprise = max(0.0, raw_surprise - 1.0)
            
            # Smooth Surprise
            avg_surprise = SURPRISE_DECAY * avg_surprise + (1.0 - SURPRISE_DECAY) * adjusted_surprise
            
            # Update modulators for next step
            raw_ach = BASE_LR * (1.0 + avg_surprise * ACH_GAIN)
            current_ach = min(raw_ach, MAX_ACH)
            
            raw_ne = BASE_NOISE * (1.0 + avg_surprise * NE_GAIN)
            current_ne = min(raw_ne, MAX_NE)
            
            obs_t = next_obs_t
            total_reward += reward
            
        reward_history.append(total_reward)
        avg_r = np.mean(reward_history[-20:]) if len(reward_history) > 0 else 0
        
        if ep % 100 == 0:
            status = "NORMAL" if not inverted else "INVERTED"
            print(f"Ep {ep:4d} | {status} | R: {total_reward:3.0f} | Avg: {avg_r:4.1f} | "
                  f"NE: {current_ne:.2f} | ACh: {current_ach:.4f} | "
                  f"Surprise: {avg_surprise:.2f}")
    
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(reward_history)), reward_history, linewidth=1, alpha=0.6)
    window_size = 20
    if len(reward_history) >= window_size:
        rolling_avg = np.convolve(reward_history, np.ones(window_size)/window_size, mode='valid')
        plt.plot(range(window_size-1, len(reward_history)), rolling_avg, linewidth=2, label=f'{window_size}-ep Avg')
    plt.axvline(x=150, color='r', linestyle='--', label='Switch Point')
    plt.xlabel('Episode')
    plt.ylabel('Episode Reward')
    plt.title('Local-Learning SNN Switch CartPole (Stable)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('local_switch_cartpole_stable.png', dpi=150, bbox_inches='tight')
    print(f"\nPlot saved as 'local_switch_cartpole_stable.png'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=4000)
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()
    
    train(args.episodes, args.render)