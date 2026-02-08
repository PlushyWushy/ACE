#!/usr/bin/env python3
"""
Switch CartPole with decoupled uncertainty following Yu & Dayan's conjecture:
- NE (Noradrenaline) driven by *unexpected* uncertainty (fast-slow TD novelty)
- ACh (Acetylcholine) driven by *expected* uncertainty (critic's variance estimate)

This separates epistemic (model disagreement) from aleatoric (state stochasticity) signals
for more nuanced neuromodulation control.
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
BASE_NOISE = 0.1

# NE logistic mapping params (for unexpected uncertainty / novelty)
NE_MAX = 2.0
NE_K = 1
NE_CENTER = 1.2

# ACh logistic mapping params (for expected uncertainty / variance)
ACH_MAX = 5e-2
ACH_K = 8
ACH_CENTER = math.exp(1.0)  # e ~ 2.718

# Surprise EMA
SURPRISE_DECAY = 0.9997

# If =1.0 -> divide by sigma (z-ish). If =0.0 -> ignore variance term.
SURPRISE_VARIANCE_WEIGHT = 0
SURPRISE_EPS = 1e-3

# ---------------------------------------------------------------------------
# Fast–slow TD novelty (habituation / baseline subtraction) for UNEXPECTED uncertainty
# ---------------------------------------------------------------------------
# TD signal is |TD| / (sigma**SURPRISE_VARIANCE_WEIGHT), then clipped.
TD_SIGNAL_CLIP = 20.0

# Fast trace reacts quickly; slow trace is "what I'm used to".
TD_FAST_ALPHA = 0.05     # ~20-step timescale
TD_SLOW_ALPHA = 0.001    # ~1000-step timescale

# Faster traces for expected uncertainty (ACh)
EXP_FAST_ALPHA = 0.001    # ~50-step timescale
EXP_SLOW_ALPHA = 0.00001

# Extra deadzone after baseline subtraction (helps suppress tiny random novelty).
TD_NOVELTY_MARGIN = 0.0

# ---------------------------------------------------------------------------
# Critic dynamics params
# ---------------------------------------------------------------------------
CRITIC_TAU_M = 0.02
CRITIC_THRESH = 1.0

# ---------------------------------------------------------------------------
# Critic LR modulation params
# ---------------------------------------------------------------------------
CRITIC_BASE_LR = 1e-3
CRITIC_ACH_MIN_SCALE = 0.1
CRITIC_ACH_MAX_SCALE = 5.0

# Editable global seed (set to None for non-deterministic runs)
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


def logistic_drive(max_val: float, k: float, center: float, signal: float, base: float) -> float:
    # numerically-stable logistic: clip the exponent to avoid overflow
    z = k * (signal - center)
    # clip thresholds chosen to avoid math.exp overflow on most platforms
    if z >= 700:
        return max_val + base
    if z <= -700:
        return base
    return max_val / (1.0 + math.exp(-z)) + base

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
# 2. SNN Critic
# ---------------------------------------------------------------------------
class SurrogateHeaviside(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return (input > 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        (input,) = ctx.saved_tensors
        grad_input = grad_output / (1.0 + torch.abs(input) * 5.0).pow(2)
        return grad_input

def surrogate_spike(x: torch.Tensor) -> torch.Tensor:
    return SurrogateHeaviside.apply(x)

class SpikingCritic(nn.Module):
    def __init__(self, n_input: int, n_hidden: int = 256):
        super().__init__()
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
        self.w = torch.empty(n_input, 2, device=device).normal_(0.0, 0.01)
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
        rho = 100.0 * torch.exp((spike_input) / 2.0)
        probs = 1.0 - torch.exp(-rho * DT)
        spikes = torch.bernoulli(torch.clamp(probs, 0.0, 1.0))

        if torch.sum(spikes) == 1:
            action = int(torch.argmax(spikes).item())
        else:
            action = int(torch.argmax(v_mem).item())
        return action, spikes

    def update(self, td_error: float, output_spikes: torch.Tensor, current_lr: float):
        eligibility = torch.outer(self.z_eps, output_spikes)
        self.w += current_lr * td_error * eligibility
        self.w.clamp_(-10.0, 10.0)


def train(episodes: int = 4000, render: bool = False, seed: int | None = SEED):
    set_global_seed(seed)
    device = torch.device("cpu")

    env = gym.make("CartPole-v1", render_mode="human" if render else None)

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

    print(f"Start Switch CartPole (Decoupled Uncertainty: NE~Novelty, ACh~Variance). Episodes: {episodes}")

    # Traces for unexpected uncertainty (novelty)
    td_fast = 0.0
    td_slow = 0.0
    td_trace_inited = False
    
    # EMA of unexpected uncertainty (drives NE)
    avg_unexpected = 0.0
    
    # EMA of expected uncertainty (drives ACh)
    avg_expected = 0.0
    exp_fast = 0.0
    exp_slow = 0.0
    exp_trace_inited = False

    reward_history = []
    unexpected_history = []
    expected_history = []
    td_history = []

    max_critic_scale = 0.0

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
        critic.reset_state()

        done = False
        total_reward = 0.0

        inverted = (ep > 2000)

        # =====================================================================
        # DECOUPLED NEUROMODULATION
        # =====================================================================
        # NE driven by unexpected uncertainty (novelty)
        current_ne = logistic_drive(NE_MAX, NE_K, NE_CENTER, avg_unexpected, BASE_NOISE)
        current_ne = min(current_ne, 5.0)

        # ACh uses its own logistic (expected uncertainty -> learning rate)
        current_ach = logistic_drive(ACH_MAX, ACH_K, ACH_CENTER, avg_expected, BASE_LR)

        td_sum = 0.0
        td_count = 0

        # For printing/debug visibility
        last_td_signal = 0.0
        last_td_fast = td_fast
        last_td_slow = td_slow
        last_td_novelty = 0.0

        while not done:
            spikes = encoder(obs_t)
            action_code, act_spikes = actor(spikes, noise_scale=current_ne)

            real_action = 1 - action_code if inverted else action_code

            next_obs, reward, terminated, truncated, _ = env.step(real_action)
            done = terminated or truncated
            next_obs_t = torch.tensor(next_obs, dtype=torch.float32, device=device)

            # -------------------------------
            # Critic update
            # -------------------------------
            v_curr, var_curr = critic(spikes)

            with torch.no_grad():
                if not done:
                    next_spikes = encoder(next_obs_t)

                    mem_backup = critic.mem_hidden.clone()
                    syn_backup = critic.syn_val.clone()

                    v_next, _ = critic(next_spikes)

                    critic.mem_hidden.copy_(mem_backup)
                    critic.syn_val.copy_(syn_backup)
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

            # Critic ACh uses its own logistic (expected uncertainty -> learning rate)
            critic_lr = logistic_drive(ACH_MAX, ACH_K, ACH_CENTER, avg_expected, CRITIC_BASE_LR)
            for pg in critic_optim.param_groups:
                pg['lr'] = critic_lr

            total_loss = (val_loss + var_loss)

            critic_optim.zero_grad()
            total_loss.backward()
            critic_optim.step()

            # -------------------------------
            # Actor update
            # -------------------------------
            actor.update(td_error_val, act_spikes, current_lr=current_ach)

            # =====================================================================
            # EXPECTED UNCERTAINTY: Use critic's variance estimate directly
            # =====================================================================
            current_sigma = torch.sqrt(var_curr.detach()).item()
            current_sigma = max(current_sigma, SURPRISE_EPS)
            # Expected uncertainty: fast-slow novelty on critic's variance
            if not exp_trace_inited:
                exp_fast = current_sigma
                exp_slow = current_sigma
                exp_trace_inited = True
            else:
                exp_fast = (1.0 - EXP_FAST_ALPHA) * exp_fast + EXP_FAST_ALPHA * current_sigma
                exp_slow = (1.0 - EXP_SLOW_ALPHA) * exp_slow + EXP_SLOW_ALPHA * current_sigma
            exp_novelty = max(0.0, exp_fast - exp_slow)
            avg_expected = SURPRISE_DECAY * avg_expected + (1.0 - SURPRISE_DECAY) * exp_novelty

            # =====================================================================
            # UNEXPECTED UNCERTAINTY: Fast-slow TD novelty
            # =====================================================================
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
            avg_unexpected = SURPRISE_DECAY * avg_unexpected + (1.0 - SURPRISE_DECAY) * td_novelty

            # Update dynamics for next step
            current_ne = logistic_drive(NE_MAX, NE_K, NE_CENTER, avg_unexpected, BASE_NOISE)
            current_ne = min(current_ne, 5.0)
            current_ach = logistic_drive(ACH_MAX, ACH_K, ACH_CENTER, avg_expected, BASE_LR)

            obs_t = next_obs_t
            total_reward += float(reward)

            # keep last-step debug values for printouts
            last_td_signal = td_signal
            last_td_fast = td_fast
            last_td_slow = td_slow
            last_td_novelty = td_novelty

        reward_history.append(total_reward)
        unexpected_history.append(avg_unexpected)
        expected_history.append(avg_expected)
        mean_abs_td = (td_sum / td_count) if td_count > 0 else 0.0
        td_history.append(mean_abs_td)
        avg_r = float(np.mean(reward_history[-20:])) if len(reward_history) > 0 else 0.0

        if ep % 100 == 0:
            status = "NORMAL" if not inverted else "INVERTED"
            print(
                f"Ep {ep:4d} | {status} | R: {total_reward:3.0f} | Avg: {avg_r:4.1f} | "
                f"NE: {current_ne:.2f} | ACh: {current_ach:.4f} | "
                f"TDsig: {last_td_signal:.2f} | Unexpected: {avg_unexpected:.2f} | Expected: {avg_expected:.2f}"
            )

    print(f"Max Critic Scale: {max_critic_scale:.3f}")

    # -----------------------------------------------------------------------
    # Plot results
    # -----------------------------------------------------------------------
    fig, ax1 = plt.subplots(figsize=(10, 6))

    ax1.plot(range(len(reward_history)), reward_history, color="tab:blue", linewidth=1, alpha=0.6, label="Reward")
    window_size = 20
    if len(reward_history) >= window_size:
        rolling_avg = np.convolve(reward_history, np.ones(window_size) / window_size, mode="valid")
        ax1.plot(range(window_size - 1, len(reward_history)), rolling_avg, color="tab:blue", linewidth=2, label=f"{window_size}-ep Avg")
    ax1.axvline(x=2000, color="r", linestyle="--", label="Switch Point")
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Reward", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")

    ax2 = ax1.twinx()
    ax2.plot(range(len(unexpected_history)), unexpected_history, color="tab:orange", linewidth=1.5, alpha=0.7, label="Unexpected (Novelty)")
    ax2.plot(range(len(expected_history)), expected_history, color="tab:green", linewidth=1.5, alpha=0.7, label="Expected (Variance)")
    ax2.set_ylabel("Uncertainty", color="tab:orange")
    ax2.tick_params(axis="y", labelcolor="tab:orange")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    ax1.grid(True, alpha=0.3)
    fig.tight_layout()
    plt.title("Switch CartPole - Decoupled Uncertainty (NE: Novelty, ACh: Variance)")

    out_dir = f"cartpole/runs/{seed}_critic_ach_decoupled" if seed is not None else "cartpole/runs/noseed_critic_ach_decoupled"
    os.makedirs(out_dir, exist_ok=True)

    png_path = os.path.join(out_dir, f"{seed}_critic_ach_decoupled.png")
    fig.savefig(png_path, dpi=150, bbox_inches="tight")

    csv_path = os.path.join(out_dir, f"{seed}_critic_ach_decoupled.csv")
    with open(csv_path, "w") as fh:
        fh.write("episode,reward,unexpected_uncertainty,expected_uncertainty,mean_abs_td\n")
        for i, (r, u, e, td) in enumerate(zip(reward_history, unexpected_history, expected_history, td_history)):
            fh.write(f"{i},{r},{u},{e},{td}\n")

    print(f"\nPlot saved as '{png_path}' and CSV saved as '{csv_path}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=4000)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed (overrides top-level SEED)")
    args = parser.parse_args()

    train(args.episodes, args.render, seed=args.seed)
