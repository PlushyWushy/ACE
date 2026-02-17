#!/usr/bin/env python3
"""
Switch Lunar Lander with decoupled uncertainty following Yu & Dayan's conjecture:
- NE (Noradrenaline) driven by *unexpected* uncertainty (fast-slow TD novelty)
- ACh (Acetylcholine) driven by *expected* uncertainty (critic's variance estimate)

Critic uses a local TD-LTP style update with loss as a multiplicative factor.
Actor LR decays each step and is boosted by ACh.
"""

import argparse
import math
import numpy as np
import random
import os
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
GAMMA = 0.9995 

# --- NEUROMODULATION PARAMETERS ---
BASE_LR = 6.25e-5 
BASE_NOISE = 1

# NE logistic mapping params
NE_MAX = 3
NE_K = 0.2
NE_CENTER = 15

# ACh logistic mapping params
ACH_MAX = 1
ACH_K = 10 
ACH_CENTER = 5

# Surprise EMA
EXP_SURPRISE_DECAY = 0.01  
UNEXP_SURPRISE_DECAY = 0.8 

# Surprise weighting
SURPRISE_VARIANCE_WEIGHT = 0
SURPRISE_EPS = 1e-3

# TD novelty params
TD_SIGNAL_CLIP = 20.0
TD_FAST_ALPHA = 0.097663     
TD_SLOW_ALPHA = 0.003642    
TD_NOVELTY_MARGIN = 0.0

# --------------------------------------------------------------------------- 
# Hyperparameters
# --------------------------------------------------------------------------- 
VAR_DECAY = 0
CRITIC_BASE_LR = 0.001
REWARD_SCALE = 0.05
ACTOR_LR_DECAY = 0.1  
ACTOR_LR_BOOST = 0.1
ACTOR_LR_MIN = 1e-6
ACTOR_LR_MAX = 0.1

SEED = 1234

def set_global_seed(seed: int | None):
    if seed is None: return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def logistic_drive(max_val: float, k: float, center: float, signal: float, base: float) -> float:
    z = k * (signal - center)
    if z >= 700: return max_val + base
    if z <= -700: return base
    return max_val / (1.0 + math.exp(-z)) + base

# --------------------------------------------------------------------------- 
# 1. Place Cell Encoder
# --------------------------------------------------------------------------- 
class PlaceCellEncoder(nn.Module):
    def __init__(self, device: torch.device):
        super().__init__()
        self.device = device
        m0 = torch.linspace(-1.0, 1.0, 4, device=device)
        m1 = torch.linspace(-0.3, 1.2, 4, device=device)
        m2 = torch.linspace(-2.0, 2.0, 4, device=device)
        m3 = torch.linspace(-2.0, 2.0, 4, device=device)
        m4 = torch.linspace(-1.0, 1.0, 4, device=device)
        m5 = torch.linspace(-1.0, 1.0, 4, device=device)
        m6 = torch.linspace(0.0, 1.0, 2, device=device)
        m7 = torch.linspace(0.0, 1.0, 2, device=device)

        mesh = torch.meshgrid(m0, m1, m2, m3, m4, m5, m6, m7, indexing="ij")
        centers = torch.stack([x.flatten() for x in mesh], dim=1)
        self.register_buffer("centers", centers)

        s0, s1, s2, s3, s4, s5 = [(m[1]-m[0])/1.5 for m in [m0, m1, m2, m3, m4, m5]]
        sigmas = torch.tensor([s0, s1, s2, s3, s4, s5, 0.5, 0.5], device=device).unsqueeze(0)
        self.register_buffer("sigmas", sigmas)
        self.n_neurons = centers.shape[0]

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        diff = (state.unsqueeze(0) - self.centers) ** 2
        exponent = -torch.sum(diff / (2 * self.sigmas ** 2), dim=1)
        rates = RHO_PC * torch.exp(exponent)
        probs = 1.0 - torch.exp(-rates * DT)
        return torch.bernoulli(torch.clamp(probs, 0.0, 1.0))

# --------------------------------------------------------------------------- 
# 2. Local TD-LTP Critic
# --------------------------------------------------------------------------- 
class LocalCritic:
    def __init__(self, n_input: int, device: torch.device, n_hidden: int = 512):
        self.device = device
        self.n_hidden = n_hidden
        self.w_h = torch.empty(n_input, n_hidden, device=device).normal_(0.0, 0.05)
        self.b_h = torch.zeros(n_hidden, device=device)
        self.w_val = torch.zeros(n_hidden, device=device)
        self.b_val = torch.zeros(1, device=device)
        self.w_var = torch.zeros(n_hidden, device=device)
        self.b_var = torch.zeros(1, device=device)
        self.mem_h = torch.zeros(n_hidden, device=device)
        self.decay_mem = math.exp(-DT / TAU_M)
        self.thresh = 1.0

    def reset_state(self):
        self.mem_h.zero_()

    def forward(self, input_spikes: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self.mem_h = self.mem_h * self.decay_mem + torch.matmul(input_spikes, self.w_h) + self.b_h
        spikes = (self.mem_h >= self.thresh).float()
        self.mem_h = self.mem_h * (1.0 - spikes)
        val = torch.dot(self.w_val, spikes) + self.b_val
        var = F.softplus(torch.dot(self.w_var, spikes) + self.b_var) + 1e-4
        return val, var, spikes

    def update(self, hidden_spikes: torch.Tensor, td_error: torch.Tensor, var: torch.Tensor, lr: float):
        delta_val = lr * td_error.abs().detach() * td_error.detach()
        self.w_val += delta_val * hidden_spikes
        self.b_val += delta_val
        if VAR_DECAY > 0:
            self.w_var *= (1.0 - VAR_DECAY)
            self.b_var *= (1.0 - VAR_DECAY)
        delta_var = lr * (td_error.detach().pow(2) - var.detach())
        self.w_var += delta_var * hidden_spikes
        self.b_var += delta_var
        for p in [self.w_val, self.b_val, self.w_var, self.b_var]: p.clamp_(-100.0, 100.0)

# --------------------------------------------------------------------------- 
# 3. Modulated Actor
# --------------------------------------------------------------------------- 
class ModulatedActor(nn.Module):
    def __init__(self, n_input: int, device: torch.device):
        super().__init__()
        self.device = device
        self.w = torch.empty(n_input, 4, device=device).normal_(0.0, 0.1)
        self.z_eps = torch.zeros(n_input, device=device)
        self.decay_eps = math.exp(-DT / TAU_M)

    def reset_state(self):
        self.z_eps.zero_()

    def forward(self, input_spikes: torch.Tensor, noise_scale: float) -> Tuple[int, torch.Tensor]:
        self.z_eps = self.z_eps * self.decay_eps + input_spikes
        v_mem = torch.matmul(self.z_eps, self.w) + torch.randn(4, device=self.device) * noise_scale
        rho = 100.0 * torch.exp(torch.clamp((v_mem - ACTOR_THETA)/2.0, -50.0, 50.0))
        spikes = torch.bernoulli(torch.clamp(1.0 - torch.exp(-rho * DT), 0.0, 1.0))
        if torch.sum(spikes) == 1: action = int(torch.argmax(spikes).item())
        else: action = int(torch.argmax(v_mem).item())
        return action, spikes

    def update(self, td_error: float, output_spikes: torch.Tensor, current_lr: float):
        self.w += current_lr * td_error * torch.outer(self.z_eps, output_spikes)
        self.w.clamp_(-20.0, 20.0)

def train(args):
    set_global_seed(args.seed)
    device = torch.device("cpu")
    env = gym.make("LunarLander-v3", render_mode="human" if args.render else None)
    encoder = PlaceCellEncoder(device)
    actor = ModulatedActor(encoder.n_neurons, device)
    critic = LocalCritic(encoder.n_neurons, device)

    avg_unexpected, avg_expected = 0.0, 0.0
    td_fast, td_slow, td_trace_inited = 0.0, 0.0, False
    reward_history, unexpected_history, expected_history, variance_history = [], [], [], []
    td_history, td2_history, actor_lr_history = [], [], []
    actor_lr_state = args.base_lr

    for ep in range(args.episodes):
        obs, _ = env.reset(seed=args.seed + ep if args.seed is not None else None)
        obs_t = torch.tensor(obs, dtype=torch.float32, device=device)
        actor.reset_state(); critic.reset_state()
        done, total_reward = False, 0.0
        td_sum, td_count, var_sum, td2_sum, alr_sum = 0.0, 0, 0.0, 0.0, 0.0

        current_ne = logistic_drive(args.ne_max, NE_K, NE_CENTER, avg_unexpected, args.base_noise)
        current_ach = logistic_drive(args.ach_max, ACH_K, ACH_CENTER, avg_expected, args.base_lr)

        while not done:
            spikes = encoder(obs_t)
            action_code, act_spikes = actor(spikes, noise_scale=current_ne)
            next_obs, reward, terminated, truncated, _ = env.step(action_code)
            done = terminated or truncated
            next_obs_t = torch.tensor(next_obs, dtype=torch.float32, device=device)

            # Reward Shaping
            shaped_reward = reward - (next_obs[4]**2)*2.0 - (next_obs[5]**2)*1.0
            v_curr, var_curr, h_spikes = critic.forward(spikes)
            if not done: v_next, _, _ = critic.forward(encoder(next_obs_t))
            else: v_next = torch.tensor([0.0], device=device)

            td_error = (shaped_reward * args.reward_scale) + GAMMA * v_next.detach() - v_curr.detach()
            td_error_val = float(td_error.item())
            critic.update(h_spikes, td_error, var_curr, lr=args.critic_base_lr)

            actor_lr_state = min(max(actor_lr_state * (1.0 - args.actor_lr_decay) + args.actor_lr_boost * current_ach, args.actor_lr_min), args.actor_lr_max)
            actor.update(td_error_val, act_spikes, current_lr=actor_lr_state)

            current_sigma = min(max(torch.sqrt(var_curr.detach()).item(), SURPRISE_EPS), 10.0)
            avg_expected = (1.0 - EXP_SURPRISE_DECAY) * avg_expected + EXP_SURPRISE_DECAY * current_sigma
            td_signal = min(max(abs(td_error_val) / (current_sigma ** SURPRISE_VARIANCE_WEIGHT), 0.0), TD_SIGNAL_CLIP)

            if not td_trace_inited: td_fast = td_slow = td_signal; td_trace_inited = True
            else:
                td_fast = (1.0 - args.td_fast_alpha) * td_fast + args.td_fast_alpha * td_signal
                td_slow = (1.0 - args.td_slow_alpha) * td_slow + args.td_slow_alpha * td_signal
            avg_unexpected = UNEXP_SURPRISE_DECAY * avg_unexpected + max(0.0, td_fast - td_slow)

            current_ne = logistic_drive(args.ne_max, NE_K, NE_CENTER, avg_unexpected, args.base_noise)
            current_ach = logistic_drive(args.ach_max, ACH_K, ACH_CENTER, avg_expected, args.base_lr)
            obs_t, total_reward = next_obs_t, total_reward + reward
            td_sum += abs(td_error_val); td_count += 1; var_sum += float(var_curr.item()); td2_sum += td_error_val**2; alr_sum += actor_lr_state

        reward_history.append(total_reward); unexpected_history.append(avg_unexpected); expected_history.append(avg_expected)
        variance_history.append(var_sum/td_count if td_count > 0 else 0); td_history.append(td_sum/td_count if td_count > 0 else 0)
        td2_history.append(td2_sum/td_count if td_count > 0 else 0); actor_lr_history.append(alr_sum/td_count if td_count > 0 else 0)

        if ep % 100 == 0:
            print(f"Ep {ep:4d} | R: {total_reward:3.0f} | Avg: {np.mean(reward_history[-20:]):4.1f} | NE: {current_ne:.2f} | ACh: {current_ach:.4f} | Unexpected: {avg_unexpected:.2f} | Expected: {avg_expected:.2f}")

    out_dir = f"ll/runs/{args.seed}_flagship" if args.seed is not None else "ll/runs/noseed_flagship"
    os.makedirs(out_dir, exist_ok=True)
    plt.figure(figsize=(10,6)); plt.plot(reward_history, label="Reward"); plt.plot(np.convolve(reward_history, np.ones(20)/20, mode="valid"), label="20-ep Avg")
    plt.legend(); plt.savefig(os.path.join(out_dir, "plot.png"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=20000)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--base_lr", type=float, default=BASE_LR)
    parser.add_argument("--base_noise", type=float, default=BASE_NOISE)
    parser.add_argument("--ne_max", type=float, default=NE_MAX)
    parser.add_argument("--ach_max", type=float, default=ACH_MAX)
    parser.add_argument("--critic_base_lr", type=float, default=CRITIC_BASE_LR)
    parser.add_argument("--actor_lr_decay", type=float, default=ACTOR_LR_DECAY)
    parser.add_argument("--actor_lr_boost", type=float, default=ACTOR_LR_BOOST)
    parser.add_argument("--actor_lr_min", type=float, default=ACTOR_LR_MIN)
    parser.add_argument("--actor_lr_max", type=float, default=ACTOR_LR_MAX)
    parser.add_argument("--reward_scale", type=float, default=REWARD_SCALE)
    parser.add_argument("--td_fast_alpha", type=float, default=TD_FAST_ALPHA)
    parser.add_argument("--td_slow_alpha", type=float, default=TD_SLOW_ALPHA)
    args = parser.parse_args(); train(args)
