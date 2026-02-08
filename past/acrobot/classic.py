#!/usr/bin/env python3
"""
Acrobot control with NE (noise) and ACh (learning rate) driven by surprise.

Differences from the CartPole version:
- Uses the six-dimensional Acrobot observation space.
- No control inversion/switching; the goal is simply to swing up.
- Actor outputs three discrete torques (-1, 0, +1) instead of binary forces.
"""

import argparse
import math
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from pathlib import Path
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
BASE_NOISE = 0.4

# Gains 
NE_GAIN = 0         # Boosts Noise
ACH_GAIN = 0        # Boosts Plasticity

SURPRISE_DECAY = 0.9

# --- SNN CRITIC PARAMETERS ---
CRITIC_TAU_M = 0.02
CRITIC_THRESH = 1.0

# Editable global seed (set to None for non-deterministic runs)
SEED = 5

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
        c1 = torch.linspace(-1.0, 1.0, 5, device=device)
        s1 = torch.linspace(-1.0, 1.0, 5, device=device)
        c2 = torch.linspace(-1.0, 1.0, 5, device=device)
        s2 = torch.linspace(-1.0, 1.0, 5, device=device)
        d1 = torch.linspace(-4.0, 4.0, 5, device=device)
        d2 = torch.linspace(-9.0, 9.0, 5, device=device)

        mesh = torch.meshgrid(c1, s1, c2, s2, d1, d2, indexing="ij")
        centers = torch.stack([x.flatten() for x in mesh], dim=1)
        self.register_buffer("centers", centers)

        spacing = torch.tensor(
            [
                (c1[1] - c1[0]) / 1.5,
                (s1[1] - s1[0]) / 1.5,
                (c2[1] - c2[0]) / 1.5,
                (s2[1] - s2[0]) / 1.5,
                (d1[1] - d1[0]) / 1.5,
                (d2[1] - d2[0]) / 1.5,
            ],
            device=device,
        ).unsqueeze(0)
        self.register_buffer("sigmas", spacing)
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
        self.w = torch.empty(n_input, 3, device=device).normal_(0.0, 0.05)
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

        if torch.sum(spikes) == 1:
            action = int(torch.argmax(spikes).item())
        else:
            dist = torch.softmax(v_mem, dim=0)
            action = int(torch.multinomial(dist, 1).item())

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
    # Seed global RNGs before creating the environment
    set_global_seed(SEED)
    env = gym.make("Acrobot-v1", render_mode="human" if render else None)
    if SEED is not None:
        try:
            env.reset(seed=SEED)
        except TypeError:
            try:
                env.seed(SEED)
            except Exception:
                pass
        try:
            env.action_space.seed(SEED)
        except Exception:
            pass
        try:
            env.observation_space.seed(SEED)
        except Exception:
            pass
    
    encoder = PlaceCellEncoder(device)
    actor = ModulatedActor(encoder.n_neurons, device)
    critic = SpikingCritic(encoder.n_neurons).to(device)
    
    critic_optim = optim.Adam(critic.parameters(), lr=1e-3)
    
    print(f"Start Acrobot (NE+ACh fixed + SNN Critic). Episodes: {episodes}")

    
    avg_surprise = 0.0
    
    reward_history = []
    td_history = []
    
    for ep in range(episodes):
        # Per-episode seeding for reproducibility (if SEED is set)
        if SEED is not None:
            try:
                obs, _ = env.reset(seed=SEED + ep)
            except TypeError:
                try:
                    env.seed(SEED + ep)
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
        
        # Determine Neuromodulator Levels 
        current_ne = BASE_NOISE * (1.0 + avg_surprise * NE_GAIN)
        current_ach = BASE_LR * (1.0 + avg_surprise * ACH_GAIN)
        current_ne = min(current_ne, 5.0) 
        
        td_sum = 0.0
        td_count = 0
        while not done:
            spikes = encoder(obs_t)
            action_code, act_spikes = actor(spikes, noise_scale=current_ne)

            next_obs, reward, terminated, truncated, _ = env.step(action_code)
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
            td_sum += abs(td_error_val)
            td_count += 1
            
            val_loss = td_error.pow(2)
            target_var = td_error.detach().pow(2) 
            var_loss = (var_curr - target_var).pow(2)
            
            critic_optim.zero_grad()
            (val_loss + var_loss).backward()
            critic_optim.step()
            
            # Actor Update
            actor.update(td_error_val, act_spikes, current_lr=current_ach)
            
            # --- SURPRISE CALCULATION ---
            current_sigma = torch.sqrt(var_curr.detach()).item()
            if current_sigma < 1e-3: current_sigma = 1e-3
            
            raw_surprise = abs(td_error_val) / current_sigma
            adjusted_surprise = max(0.0, raw_surprise - 1.0)
            
            avg_surprise = SURPRISE_DECAY * avg_surprise + (1.0 - SURPRISE_DECAY) * adjusted_surprise
            
            # Update dynamics for next step
            current_ne = BASE_NOISE * (1.0 + avg_surprise * NE_GAIN)
            current_ach = BASE_LR * (1.0 + avg_surprise * ACH_GAIN)
            current_ne = min(current_ne, 5.0)
            
            obs_t = next_obs_t
            total_reward += reward
            
        reward_history.append(total_reward)
        mean_abs_td = (td_sum / td_count) if td_count > 0 else 0.0
        td_history.append(mean_abs_td)
        avg_r = np.mean(reward_history[-20:]) if len(reward_history) > 0 else 0

        if ep % 100 == 0:
            print(
                f"Ep {ep:4d} | R: {total_reward:5.1f} | Avg: {avg_r:5.1f} | "
                f"NE: {current_ne:.2f} | ACh: {current_ach:.4f} | Surprise: {avg_surprise:.2f}"
            )
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(reward_history)), reward_history, linewidth=1, alpha=0.6)
    window_size = 20
    if len(reward_history) >= window_size:
        rolling_avg = np.convolve(reward_history, np.ones(window_size)/window_size, mode='valid')
        plt.plot(range(window_size-1, len(reward_history)), rolling_avg, linewidth=2, label=f'{window_size}-ep Avg')
    plt.xlabel('Episode')
    plt.ylabel('Episode Reward')
    plt.title('Acrobot - classic + SNN Critic')
    plt.legend()
    plt.grid(True, alpha=0.3)
    # Ensure output directory exists and save both CSV and PNG
    script_dir = Path(__file__).resolve().parent
    out_root = script_dir / "runs"
    out_root.mkdir(parents=True, exist_ok=True)
    run_name = f"{SEED}_classic_acrobot" if SEED is not None else "noseed_classic_acrobot"
    out_dir = out_root / run_name
    out_dir.mkdir(exist_ok=True)
    png_path = out_dir / f"{run_name}.png"
    plt.savefig(png_path, dpi=150, bbox_inches='tight')

    csv_path = out_dir / f"{run_name}.csv"
    with csv_path.open("w") as fh:
        fh.write("episode,reward,avg_surprise,mean_abs_td\n")
        for i, (r, td) in enumerate(zip(reward_history, td_history)):
            # classic does not store per-episode surprise history, use 0.0 placeholder
            fh.write(f"{i},{r},0.0,{td}\n")

    print(f"\nPlot saved as '{png_path}' and CSV saved as '{csv_path}'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=4000)
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()
    
    train(args.episodes, args.render)
