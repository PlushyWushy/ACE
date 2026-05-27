#!/usr/bin/env python3
"""
Surrogate-gradient baseline for the switch CartPole experiment.
Matches the exact 10,000 episode timescale and linear place-cell architecture of cartpole_successful/flagship.py.
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
# Constants (aligned with cartpole_successful/flagship.py)
# ---------------------------------------------------------------------------
DT = 0.02
RHO_PC = 50.0
TAU_M = 0.02
ACTOR_THETA = 2.0
GAMMA = 0.99

BASE_LR = 0.000055 
BASE_NOISE = 1

ACH_MAX = 1.0
ACH_K = 10
ACH_CENTER = 5

NE_MAX = 3
NE_K = 0.2
NE_CENTER = 15

EXP_SURPRISE_DECAY = 0.01  
UNEXP_SURPRISE_DECAY = 0.8 
SURPRISE_VARIANCE_WEIGHT = 0
SURPRISE_EPS = 1e-3

TD_SIGNAL_CLIP = 20.0
TD_FAST_ALPHA = 0.097663     # ~20-step timescale
TD_SLOW_ALPHA = 0.003642    # ~1000-step timescale
TD_NOVELTY_MARGIN = 0.0

CRITIC_BASE_LR = 0.001
SEED = 1234

# Toggle neuromodulation (False for baseline comparison)
USE_NEUROMOD = False

# ---------------------------------------------------------------------------
# Place Cell Encoder & Surrogate Actor/Critic
# ---------------------------------------------------------------------------
class PlaceCellEncoder(nn.Module):
    def __init__(self, device: torch.device):
        super().__init__()
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


class SurrogateCritic(nn.Module):
    """Linear critic matching the place-cell structure of flagship.py, updated via Adam."""
    def __init__(self, n_input: int):
        super().__init__()
        self.fc_val = nn.Linear(n_input, 1, bias=True)
        self.fc_var = nn.Linear(n_input, 1, bias=True)
        with torch.no_grad():
            self.fc_val.weight.zero_()
            self.fc_val.bias.zero_()
            self.fc_var.weight.zero_()
            self.fc_var.bias.zero_()

    def forward(self, input_spikes: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        val = self.fc_val(input_spikes)
        var = F.softplus(self.fc_var(input_spikes)) + 1e-4
        return val, var


class SurrogateActor(nn.Module):
    def __init__(self, n_input: int, device: torch.device):
        super().__init__()
        self.device = device
        self.w = nn.Parameter(torch.empty(n_input, 2, device=device).normal_(0.0, 0.1))
        self.z_eps = torch.zeros(n_input, device=device)
        self.decay_eps = math.exp(-DT / TAU_M)

    def reset_state(self):
        self.z_eps.zero_()

    def forward(self, input_spikes: torch.Tensor, noise_scale: float) -> Tuple[int, torch.Tensor]:
        self.z_eps = self.z_eps * self.decay_eps + input_spikes
        v_det = torch.matmul(self.z_eps, self.w)
        noise = torch.randn_like(v_det) * noise_scale
        v_mem = v_det + noise
        spike_input = v_mem - ACTOR_THETA

        # Custom surrogate heaviside autograd
        class CustomSurrogate(torch.autograd.Function):
            @staticmethod
            def forward(ctx, input_t):
                ctx.save_for_backward(input_t)
                return (input_t > 0).float()
            @staticmethod
            def backward(ctx, grad_output):
                (input_t,) = ctx.saved_tensors
                return grad_output / (1.0 + torch.abs(input_t) * 5.0).pow(2)

        spikes = CustomSurrogate.apply(spike_input)

        s_cpu = spikes.detach().cpu().numpy()
        if s_cpu[0] == 1 and s_cpu[1] == 0:
            action = 0
        elif s_cpu[0] == 0 and s_cpu[1] == 1:
            action = 1
        else:
            action = int(torch.argmax(v_mem).item())

        return action, spikes

    def surrogate_update(self, td_error: float, output_spikes: torch.Tensor, lr_scale: float, optimizer: torch.optim.Optimizer):
        eligibility = torch.outer(self.z_eps, output_spikes)
        coef = float(lr_scale * td_error)
        loss = -coef * torch.sum(self.w * eligibility)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            self.w.clamp_(-10.0, 10.0)


def set_global_seed(seed: int | None):
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train(episodes=10000, seed: int | None = None):
    device = torch.device("cpu")
    set_global_seed(seed)

    env = gym.make("CartPole-v1")

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

    encoder = PlaceCellEncoder(device)
    actor = SurrogateActor(encoder.n_neurons, device)
    critic = SurrogateCritic(encoder.n_neurons).to(device)

    # Actor optimizer SGD with lr=1.0, scaled by lr_scale inside update
    actor_optim = optim.SGD([actor.w], lr=1.0)
    critic_optim = optim.Adam(critic.parameters(), lr=CRITIC_BASE_LR)

    print(f"Start Surrogate Switch CartPole. Episodes: {episodes}")

    avg_surprise = 0.0
    td_fast = 0.0
    td_slow = 0.0
    td_trace_inited = False

    reward_history = []
    surprise_history = []
    td_history = []

    for ep in range(episodes):
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

        done = False
        total_reward = 0.0

        inverted = (ep >= 5000)

        if USE_NEUROMOD:
            current_ne = NE_MAX / (1.0 + math.exp(-NE_K * (avg_surprise - NE_CENTER))) + BASE_NOISE
            current_ach = ACH_MAX / (1.0 + math.exp(-ACH_K * (avg_surprise - ACH_CENTER))) + BASE_LR
            current_ne = min(current_ne, 5.0)
            current_ach = min(max(current_ach, 0.0), ACH_MAX)
        else:
            current_ne = BASE_NOISE
            current_ach = BASE_LR

        td_sum = 0.0
        td_count = 0

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
                    v_next, _ = critic(next_spikes)
                else:
                    v_next = torch.tensor([0.0], device=device)

            target = reward + GAMMA * v_next
            td_error = target - v_curr
            td_error_val = float(td_error.item())

            td_sum += abs(td_error_val)
            td_count += 1

            val_loss = td_error.pow(2)
            target_var = td_error.detach().pow(2)
            var_loss = (var_curr - target_var).pow(2)

            total_loss = val_loss + var_loss

            critic_optim.zero_grad()
            total_loss.backward()
            critic_optim.step()

            # surrogate actor update
            actor.surrogate_update(td_error_val, act_spikes, lr_scale=current_ach, optimizer=actor_optim)

            current_sigma = torch.sqrt(var_curr.detach()).item()
            current_sigma = max(current_sigma, SURPRISE_EPS)

            td_signal = abs(td_error_val) / (current_sigma ** SURPRISE_VARIANCE_WEIGHT)
            td_signal = float(min(max(td_signal, 0.0), TD_SIGNAL_CLIP))

            if not td_trace_inited:
                td_fast = td_signal
                td_slow = td_signal
                td_trace_inited = True
            else:
                td_fast = (1.0 - TD_FAST_ALPHA) * td_fast + TD_FAST_ALPHA * td_signal
                td_slow = (1.0 - TD_SLOW_ALPHA) * td_slow + TD_SLOW_ALPHA * td_signal

            td_novelty = max(0.0, td_fast - td_slow - TD_NOVELTY_MARGIN)
            avg_surprise = EXP_SURPRISE_DECAY * avg_surprise + (1.0 - EXP_SURPRISE_DECAY) * td_novelty

            obs_t = next_obs_t
            total_reward += float(reward)

        reward_history.append(total_reward)
        surprise_history.append(avg_surprise)
        mean_abs_td = (td_sum / td_count) if td_count > 0 else 0.0
        td_history.append(mean_abs_td)
        avg_r = float(np.mean(reward_history[-20:])) if len(reward_history) > 0 else 0.0

        if ep % 100 == 0:
            status = "NORMAL" if not inverted else "INVERTED"
            print(
                f"Ep {ep:4d} | {status} | R: {total_reward:3.0f} | Avg20: {avg_r:4.1f} | "
                f"Surprise: {avg_surprise:.2f}"
            )

    out_dir = f"surrogate_baseline/runs/{seed}_cartpole_surrogate" if seed is not None else "surrogate_baseline/runs/noseed_cartpole_surrogate"
    os.makedirs(out_dir, exist_ok=True)

    csv_path = os.path.join(out_dir, f"{seed}_cartpole_surrogate.csv")
    with open(csv_path, "w") as fh:
        fh.write("episode,reward,avg_surprise,mean_abs_td\n")
        for i, (r, s, td) in enumerate(zip(reward_history, surprise_history, td_history)):
            fh.write(f"{i},{r},{s},{td}\n")

    print(f"Saved CSV to '{csv_path}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed")
    args = parser.parse_args()

    train(args.episodes, seed=args.seed)
