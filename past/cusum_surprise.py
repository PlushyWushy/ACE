#!/usr/bin/env python3
"""
Switch CartPole with NE (Noise) AND ACh (Learning Rate) + SNN CRITIC.

This variant uses per-dimension CUSUM change-point detectors on observations
to compute surprise, combined with TD_error / variance^n. Critic LR is modulated by ACh.
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
RHO_PC = 50.0         
TAU_M = 0.02          
ACTOR_THETA = 2.0     
GAMMA = 0.99          

# --- NEUROMODULATION PARAMETERS ---
BASE_LR = 5e-4       
BASE_NOISE = 1.0      

# NE logistic mapping params (bounded tonic level)
NE_MAX = 5.0
NE_K = 2.0
NE_CENTER = 9.0

# ACh logistic mapping params (learning rate will be in [0, ACH_MAX])
ACH_MAX = 1.0
ACH_K = 2.0
ACH_CENTER = 4.0

SURPRISE_DECAY = 0.98 

# If =1.0 -> divide by sigma (current z-score). If =0.0 -> ignore variance term.
SURPRISE_VARIANCE_WEIGHT = 0
# Small floor for sigma to avoid huge surprises when variance ~ 0
SURPRISE_EPS = 1e-3

# Critic dynamics params (same defaults as `flagship.py`)
CRITIC_TAU_M = 0.02
CRITIC_THRESH = 1.0

# ---------------------------------------------------------------------------
# Critic LR modulation params (new in this file)
# ---------------------------------------------------------------------------
# Base learning rate used to construct the optimizer. The effective update size
# will be determined by scaling the loss using `critic_scale` derived from ACh.
CRITIC_BASE_LR = 1e-3
# How ACh maps to a multiplier on critic loss: we compute a normalized ACh and
# map it to [CRITIC_ACH_MIN_SCALE, CRITIC_ACH_MAX_SCALE].
CRITIC_ACH_MIN_SCALE = 0.1
CRITIC_ACH_MAX_SCALE = 5.0

# ---------------------------------------------------------------------------
# CUSUM parameters for per-dimension surprise
# ---------------------------------------------------------------------------
N_OBS_DIM = 4  # CartPole observations: x, x_dot, theta, theta_dot
CUSUM_ALPHA_MEAN = 0.99  # EMA alpha for mean (tau ~ 1s at dt=0.02)
CUSUM_ALPHA_VAR = 0.99   # EMA alpha for variance
CUSUM_LAMBDA = 0.01      # Leak rate for CUSUM accumulator (tau ~ 2s)
CUSUM_K = 0.5            # Slack/reference value (z-units)
CUSUM_H = 10.0           # Detection threshold
CUSUM_REFRACTORY = 50    # Steps to suppress after trigger (1s at dt=0.02)

# Editable global seed (set to None for non-deterministic runs)
SEED = 6777 

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
# 2. SNN Critic (NEW)
# ---------------------------------------------------------------------------

class SurrogateHeaviside(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return (input > 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output / (1.0 + torch.abs(input) * 5.0).pow(2) 
        return grad_input

def surrogate_spike(input):
    return SurrogateHeaviside.apply(input)

class SpikingCritic(nn.Module):
    def __init__(self, n_input: int, n_hidden: int = 256):
        super().__init__()
        self.n_input = n_input
        self.n_hidden = n_hidden
        
        self.fc1 = nn.Linear(n_input, n_hidden)
        self.fc_val = nn.Linear(n_hidden, 1)
        self.fc_var = nn.Linear(n_hidden, 1)
        
        self.decay_mem = math.exp(-DT / CRITIC_TAU_M)
        self.decay_syn = math.exp(-DT / CRITIC_TAU_M) 
        
        self.register_buffer("mem_hidden", torch.zeros(n_hidden))
        self.register_buffer("syn_val", torch.zeros(n_hidden)) 
        
    def reset_state(self):
        self.mem_hidden.zero_()
        self.syn_val.zero_()
        
    def forward(self, input_spikes: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Detach state
        self.mem_hidden = self.mem_hidden.detach()
        self.syn_val = self.syn_val.detach()
        
        current_in = self.fc1(input_spikes)
        self.mem_hidden = self.mem_hidden * self.decay_mem + current_in
        
        spike_input = self.mem_hidden - CRITIC_THRESH
        hidden_spikes = surrogate_spike(spike_input)
        
        self.mem_hidden = self.mem_hidden * (1.0 - hidden_spikes.detach()) 
        
        self.syn_val = self.syn_val * self.decay_syn + hidden_spikes
        
        val = self.fc_val(self.syn_val)
        var = F.softplus(self.fc_var(self.syn_val)) + 1e-4
        
        return val, var

# ---------------------------------------------------------------------------
# 3. Universal Actor (Modulated)
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

    def update(self, td_error: float, output_spikes: torch.Tensor, current_lr: float):
        eligibility = torch.outer(self.z_eps, output_spikes)
        dw = current_lr * td_error * eligibility
        self.w += dw
        self.w.clamp_(-10.0, 10.0)

# ---------------------------------------------------------------------------
# 4. Training Loop
# ---------------------------------------------------------------------------
def train(episodes=4000, render=False, seed: int | None = None):
    device = torch.device("cpu")
    # Seed global RNGs before creating the environment
    set_global_seed(seed)
    env = gym.make("CartPole-v1", render_mode="human" if render else None)
    # Seed environment RNGs / spaces where supported
    if seed is not None:
        try:
            env.reset(seed=seed)
        except TypeError:
            try:
                env.seed(seed)
            except Exception:
                pass
        try:
            env.action_space.seed(seed)
        except Exception:
            pass
        try:
            env.observation_space.seed(seed)
        except Exception:
            pass
    
    encoder = PlaceCellEncoder(device)
    actor = ModulatedActor(encoder.n_neurons, device)
    critic = SpikingCritic(encoder.n_neurons).to(device)
    
    critic_optim = optim.Adam(critic.parameters(), lr=CRITIC_BASE_LR)
    
    print(f"Start Switch CartPole (NE+ACh -> Critic ACh Modulation + CUSUM Surprise). Episodes: {episodes}")

    
    avg_surprise = 0.0
    
    reward_history = []
    surprise_history = []
    td_history = []
    
    # Initialize CUSUM state per dimension
    cusum_mean = np.zeros(N_OBS_DIM)
    cusum_var = np.ones(N_OBS_DIM)
    cusum_C = np.zeros(N_OBS_DIM)
    cusum_last_trigger = np.full(N_OBS_DIM, -CUSUM_REFRACTORY)
    
    for ep in range(episodes):
        # Per-episode seeding for reproducibility (if SEED is set)
        if seed is not None:
            try:
                obs, _ = env.reset(seed=seed + ep)
            except TypeError:
                try:
                    env.seed(seed + ep)
                    obs, _ = env.reset()
                except Exception:
                    obs, _ = env.reset()
        else:
            obs, _ = env.reset()
        obs_t = torch.tensor(obs, dtype=torch.float32, device=device)
        actor.reset_state()
        critic.reset_state() 
        
        done = False
        total_reward = 0
        
        inverted = (ep > 2000) 
        
        # Determine Neuromodulator Levels 
        # Logistic dependence for NE (bounded tonic level)
        current_ne = NE_MAX / (1.0 + math.exp(-NE_K * (avg_surprise - NE_CENTER))) + BASE_NOISE
        # Logistic dependence for ACh (learning rate) bounded in [0, ACH_MAX]
        current_ach = ACH_MAX / (1.0 + math.exp(-ACH_K * (avg_surprise - ACH_CENTER))) + BASE_LR
        # safety clamps
        current_ne = min(current_ne, 5.0)
        current_ach = min(max(current_ach, 0.0), ACH_MAX)
        
        td_sum = 0.0
        td_count = 0
        while not done:
            spikes = encoder(obs_t)
            action_code, act_spikes = actor(spikes, noise_scale=current_ne)
            
            real_action = 1 - action_code if inverted else action_code
            
            next_obs, reward, terminated, truncated, _ = env.step(real_action)
            done = terminated or truncated
            next_obs_t = torch.tensor(next_obs, dtype=torch.float32, device=device)
            
            # Critic Update
            v_curr, var_curr = critic(spikes)

            with torch.no_grad():
                if not done:
                    next_spikes = encoder(next_obs_t)

                    # --- snapshot critic internal state ---
                    mem_backup = critic.mem_hidden.clone()
                    syn_backup = critic.syn_val.clone()

                    # compute v_next (this will mutate critic state internally)
                    v_next, _ = critic(next_spikes)

                    # --- restore critic internal state ---
                    critic.mem_hidden.copy_(mem_backup)
                    critic.syn_val.copy_(syn_backup)
                else:
                    v_next = torch.tensor([0.0], device=device)

            
            target = reward + GAMMA * v_next
            td_error = target - v_curr
            td_error_val = td_error.item()
            # accumulate absolute TD for per-episode average
            td_sum += abs(td_error_val)
            td_count += 1
            
            val_loss = td_error.pow(2)
            target_var = td_error.detach().pow(2) 
            var_loss = (var_curr - target_var).pow(2)

            # --- Critic loss scaling by ACh: derive a safe multiplier from current_ach
            # Normalize current_ach by ACH_MAX (if ACH_MAX > 0) and map to [min_scale, max_scale]
            if ACH_MAX > 0:
                ach_norm = current_ach / ACH_MAX
            else:
                ach_norm = 0.0
            desired_scale = 1.0 + ach_norm * (CRITIC_ACH_MAX_SCALE - 1.0)
            critic_scale = max(CRITIC_ACH_MIN_SCALE, min(CRITIC_ACH_MAX_SCALE, desired_scale))

            total_loss = (val_loss + var_loss)
            scaled_loss = total_loss * critic_scale

            critic_optim.zero_grad()
            scaled_loss.backward()
            critic_optim.step()
            
            # Actor Update
            actor.update(td_error_val, act_spikes, current_lr=current_ach)
            
            # --- SURPRISE CALCULATION: combination of CUSUM and TD/variance^n ---
            # Update CUSUM per dimension on raw observations
            obs_np = obs_t.cpu().numpy()
            cusum_surprise = 0.0
            for i in range(N_OBS_DIM):
                x = obs_np[i]
                cusum_mean[i] = CUSUM_ALPHA_MEAN * cusum_mean[i] + (1 - CUSUM_ALPHA_MEAN) * x
                cusum_var[i] = CUSUM_ALPHA_VAR * cusum_var[i] + (1 - CUSUM_ALPHA_VAR) * (x - cusum_mean[i]) ** 2
                z = (x - cusum_mean[i]) / (np.sqrt(cusum_var[i]) + 1e-6)
                cusum_C[i] = max(0, (1 - CUSUM_LAMBDA) * cusum_C[i] + (z - CUSUM_K))
                if cusum_C[i] > CUSUM_H and (ep * 1000 + td_count) - cusum_last_trigger[i] > CUSUM_REFRACTORY:
                    cusum_surprise += cusum_C[i]  # Accumulate trigger strength
                    cusum_last_trigger[i] = ep * 1000 + td_count
                    cusum_C[i] = 0  # Reset after trigger
            
            # TD-based surprise
            current_sigma = torch.sqrt(var_curr.detach()).item()
            current_sigma = max(current_sigma, SURPRISE_EPS)
            mean_abs_td_so_far = (td_sum / td_count) if td_count > 0 else abs(td_error_val)
            td_surprise = mean_abs_td_so_far / (current_sigma ** SURPRISE_VARIANCE_WEIGHT)
            
            # Combine: CUSUM contribution + TD contribution
            raw_surprise = cusum_surprise + td_surprise
            adjusted_surprise = max(0.0, raw_surprise - 1.0)

            avg_surprise = SURPRISE_DECAY * avg_surprise + (1.0 - SURPRISE_DECAY) * adjusted_surprise
            
            # Update dynamics for next step
            # Logistic dependence for NE (bounded tonic level)
            current_ne = NE_MAX / (1.0 + math.exp(-NE_K * (avg_surprise - NE_CENTER))) + BASE_NOISE
            # Logistic dependence for ACh (learning rate) bounded in [0, ACH_MAX]
            current_ach = ACH_MAX / (1.0 + math.exp(-ACH_K * (avg_surprise - ACH_CENTER))) + BASE_LR
            # safety clamps
            current_ne = min(current_ne, 5.0)
            current_ach = min(max(current_ach, 0.0), ACH_MAX)
            
            obs_t = next_obs_t
            total_reward += reward
            
        reward_history.append(total_reward)
        surprise_history.append(avg_surprise)
        mean_abs_td = (td_sum / td_count) if td_count > 0 else 0.0
        td_history.append(mean_abs_td)
        avg_r = np.mean(reward_history[-20:]) if len(reward_history) > 0 else 0
        
        if ep % 100 == 0:
            status = "NORMAL" if not inverted else "INVERTED"
            print(f"Ep {ep:4d} | {status} | R: {total_reward:3.0f} | Avg: {avg_r:4.1f} | "
                  f"NE: {current_ne:.2f} | ACh: {current_ach:.4f} | CriticScale: {critic_scale:.3f} | "
                  f"Surprise: {avg_surprise:.2f}")
    
    # Plot results
    # Plot episode rewards and avg_surprise on the same figure (two y-axes)
    fig, ax1 = plt.subplots(figsize=(10, 6))

    ax1.plot(range(len(reward_history)), reward_history, color='tab:blue', linewidth=1, alpha=0.6, label='Episode Reward')
    window_size = 20
    if len(reward_history) >= window_size:
        rolling_avg = np.convolve(reward_history, np.ones(window_size)/window_size, mode='valid')
        ax1.plot(range(window_size-1, len(reward_history)), rolling_avg, color='tab:blue', linewidth=2, label=f'{window_size}-ep Avg')
    ax1.axvline(x=2000, color='r', linestyle='--', label='Switch Point')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Episode Reward', color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    # Twin axis for surprise
    ax2 = ax1.twinx()
    ax2.plot(range(len(surprise_history)), surprise_history, color='tab:orange', linewidth=1.5, alpha=0.9, label='Avg Surprise')
    # Also plot mean absolute TD per episode on the same (right) axis
    try:
        ax2.plot(range(len(td_history)), td_history, color='tab:green', linewidth=1.2, alpha=0.9, label='Mean |TD|')
    except NameError:
        # td_history may not exist in older versions; ignore if missing
        pass
    ax2.set_ylabel('Avg Surprise', color='tab:orange')
    ax2.tick_params(axis='y', labelcolor='tab:orange')

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

    ax1.grid(True, alpha=0.3)
    fig.tight_layout()
    plt.title('Switch CartPole - NE+ACh (Critic ACh Modulation + CUSUM Surprise) + SNN Critic')
    # Ensure output directory exists and save both CSV and PNG
    out_dir = f"runs/{seed}_cusum_surprise" if seed is not None else "runs/noseed_cusum_surprise"
    os.makedirs(out_dir, exist_ok=True)

    png_path = os.path.join(out_dir, f"{seed}_cusum_surprise.png")
    fig.savefig(png_path, dpi=150, bbox_inches='tight')

    # Save CSV with per-episode data
    csv_path = os.path.join(out_dir, f"{seed}_cusum_surprise.csv")
    with open(csv_path, "w") as fh:
        fh.write("episode,reward,avg_surprise,mean_abs_td\n")
        for i, (r, s, td) in enumerate(zip(reward_history, surprise_history, td_history)):
            fh.write(f"{i},{r},{s},{td}\n")

    print(f"\nPlot saved as '{png_path}' and CSV saved as '{csv_path}'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=4000)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed (overrides top-level SEED)")
    args = parser.parse_args()
    
    train(args.episodes, args.render, seed=args.seed)