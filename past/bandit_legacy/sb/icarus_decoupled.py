#!/usr/bin/env python3
"""
Switching bandit with decoupled uncertainty following Yu & Dayan's conjecture:
- NE (Noradrenaline) driven by *unexpected* uncertainty (fast-slow TD novelty)
- ACh (Acetylcholine) driven by *expected* uncertainty (critic's variance estimate)

This separates epistemic (model disagreement) from aleatoric (state stochasticity) signals.
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import os
from typing import Tuple

# ---------------------------------------------------------------------------
# Shared constants (match ne_ach base values)
# ---------------------------------------------------------------------------
DT = 0.02
TAU_M = 0.02
ACTOR_THETA = 2.0

BASE_LR = 1e-2
BASE_NOISE = 0.5
# Slightly lower decay to keep bursts of surprise high for longer
SURPRISE_DECAY = 0.9

NE_MAX = 5.0
NE_K = 1.0
NE_CENTER = 0.9

ACH_MAX = 1.0
ACH_K = 3.0
ACH_CENTER = 0.5

# Surprise trace hyperparameters (for unexpected uncertainty / novelty)
SURPRISE_VARIANCE_WEIGHT = 0.0
SURPRISE_EPS = 1e-3
TD_SIGNAL_CLIP = 20.0
TD_FAST_ALPHA = 1 
TD_SLOW_ALPHA = 0.0002
TD_NOVELTY_MARGIN = 0.0

# Faster traces for expected uncertainty (ACh)
EXP_FAST_ALPHA = 1.0
EXP_SLOW_ALPHA = 0.01

# Variance scaling parameters
VAR_MAX = 5.0  # Max value for expected uncertainty signal
VAR_K = 1.0    # Steepness of variance logistic
VAR_CENTER = 0.5  # Center of variance logistic curve

CRITIC_LR = 1e-2
SEED = 5

CRITIC_ACH_MIN_SCALE = 0.1
CRITIC_ACH_MAX_SCALE = 1 


def set_global_seed(seed: int | None):
    if seed is None:
        return
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class MetaCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )
        self.val_head = nn.Linear(64, 1)
        self.var_head = nn.Linear(64, 1)

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.trunk(state)
        val = self.val_head(x)
        var = F.softplus(self.var_head(x)) + 1e-4
        return val, var


class SNNActor(nn.Module):
    def __init__(self, device: torch.device):
        super().__init__()
        self.device = device
        self.w = torch.empty(1, 2, device=device).normal_(0.0, 0.1)
        self.z_eps = torch.zeros(1, device=device)
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
            action = int(torch.argmax(v_mem).item())
        return action, spikes

    def update(self, td_error: float, output_spikes: torch.Tensor, lr: float):
        eligibility = torch.outer(self.z_eps, output_spikes)
        self.w += lr * td_error * eligibility
        self.w.clamp_(-10.0, 10.0)


def logistic_drive(max_val: float, k: float, center: float, signal: float, base: float) -> float:
    """Generic logistic modulation: max_val / (1 + exp(-k*(signal - center))) + base"""
    return max_val / (1.0 + math.exp(-k * (signal - center))) + base


def train(episodes: int = 4000, seed: int | None = SEED):
    set_global_seed(seed)
    device = torch.device("cpu")

    actor = SNNActor(device)
    critic = MetaCritic().to(device)
    optim_critic = optim.Adam(critic.parameters(), lr=CRITIC_LR)

    # Traces for unexpected uncertainty (novelty)
    td_fast = 0.0
    td_slow = 0.0
    td_inited = False
    
    # EMA of unexpected uncertainty (drives NE)
    avg_unexpected = 0.0
    
    # EMA of expected uncertainty (drives ACh)
    avg_expected = 0.0
    exp_fast = 0.0
    exp_slow = 0.0
    exp_inited = False
    
    last_td_signal = 0.0

    input_state = torch.tensor([1.0], device=device)
    input_spikes = torch.tensor([1.0], device=device)

    reward_history = []
    unexpected_history = []
    expected_history = []

    max_critic_scale = 0.0

    for ep in range(1, episodes + 1):
        if ep <= 2000:
            prob = [1.0, 0.0]
            optimal = 0
        else:
            prob = [0.0, 1.0]
            optimal = 1

        # =====================================================================
        # DECOUPLED NEUROMODULATION
        # =====================================================================
        # NE driven by unexpected uncertainty (novelty)
        current_ne = logistic_drive(NE_MAX, NE_K, NE_CENTER, avg_unexpected, BASE_NOISE)
        current_ne = min(current_ne, 5.0)
        
        # ACh uses its own logistic (expected uncertainty -> learning rate)
        current_ach = logistic_drive(ACH_MAX, ACH_K, ACH_CENTER, avg_expected, BASE_LR)

        actor.reset_state()
        action, spikes = actor(input_spikes, noise_scale=current_ne)
        reward = 1.0 if np.random.rand() < prob[action] else -1.0

        v_curr, var_curr = critic(input_state)
        var_curr = torch.clamp(var_curr, min=0.01)
        td_error = reward - v_curr
        td_error_val = float(td_error.item())

        val_loss = td_error.pow(2)
        target_var = td_error.detach().pow(2)
        var_loss = (var_curr - target_var).pow(2)

        # Critic ACh uses its own logistic (expected uncertainty -> learning rate)
        critic_lr = logistic_drive(ACH_MAX, ACH_K, ACH_CENTER, avg_expected, CRITIC_LR)
        for pg in optim_critic.param_groups:
            pg['lr'] = critic_lr

        total_loss = val_loss + var_loss

        optim_critic.zero_grad()
        total_loss.backward()
        optim_critic.step()

        actor.update(td_error_val, spikes, lr=current_ach)

        # =====================================================================
        # EXPECTED UNCERTAINTY: Use critic's variance estimate directly
        # =====================================================================
        sigma = torch.sqrt(var_curr.detach()).item()
        sigma = max(sigma, SURPRISE_EPS)
        # Expected uncertainty: fast-slow novelty on critic's variance
        if not exp_inited:
            exp_fast = sigma
            exp_slow = sigma
            exp_inited = True
        else:
            exp_fast = (1.0 - EXP_FAST_ALPHA) * exp_fast + EXP_FAST_ALPHA * sigma
            exp_slow = (1.0 - EXP_SLOW_ALPHA) * exp_slow + EXP_SLOW_ALPHA * sigma
        exp_novelty = max(0.0, exp_fast - exp_slow)
        avg_expected = SURPRISE_DECAY * avg_expected + (1.0 - SURPRISE_DECAY) * exp_novelty

        # =====================================================================
        # UNEXPECTED UNCERTAINTY: Fast-slow TD novelty
        # =====================================================================
        td_signal = abs(td_error_val) / (sigma ** SURPRISE_VARIANCE_WEIGHT)
        td_signal = float(min(max(td_signal, 0.0), TD_SIGNAL_CLIP))

        if not td_inited:
            td_fast = td_signal
            td_slow = td_signal
            td_inited = True
        else:
            td_fast = (1.0 - TD_FAST_ALPHA) * td_fast + TD_FAST_ALPHA * td_signal
            td_slow = (1.0 - TD_SLOW_ALPHA) * td_slow + TD_SLOW_ALPHA * td_signal
        
        td_novelty = max(0.0, td_fast - td_slow - TD_NOVELTY_MARGIN)
        avg_unexpected = SURPRISE_DECAY * avg_unexpected + (1.0 - SURPRISE_DECAY) * td_novelty
        last_td_signal = td_signal

        reward_history.append(1 if action == optimal else 0)
        unexpected_history.append(avg_unexpected)
        expected_history.append(avg_expected)

        if ep % 50 == 0:
            accuracy = np.mean(reward_history[-200:]) * 100 if len(reward_history) >= 200 else np.mean(reward_history) * 100
            print(
                f"Ep {ep:4d} | Opt%: {accuracy:5.1f} | NE: {current_ne:.2f} | ACh: {current_ach:.4f} | "
                f"CriticScale: {critic_scale:.3f} | TDsig: {last_td_signal:.2f} | "
                f"Unexpected: {avg_unexpected:.3f} | Expected: {avg_expected:.3f}"
            )

    plot_episodes = list(range(50, episodes + 1, 50))
    reward_rates = []
    for i in plot_episodes:
        window = reward_history[i - 50:i]
        reward_rates.append(float(np.mean(window) * 100.0))

    plt.figure(figsize=(10, 6))
    plt.plot(plot_episodes, reward_rates, linewidth=2, color="tab:blue", label="Reward Rate")
    plt.axvline(x=2000, color="tab:red", linestyle="--", label="Switch")
    plt.xlabel("Episode")
    plt.ylabel("Reward Rate (%)")
    plt.ylim(0, 105)
    plt.title("Switch Bandit - Decoupled Uncertainty (NE: Novelty, ACh: Variance)")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="lower right")
    plt.savefig("switch_bandit_rstdp_decoupled.png", dpi=150, bbox_inches="tight")
    print("\nPlot saved as 'switch_bandit_rstdp_decoupled.png'")
    print(f"Max Critic Scale: {max_critic_scale:.3f}")
    
    # Ensure output directory exists and save both CSV and PNG in runs folder
    out_dir = f"switch_bandit_experiment/runs/{seed}_icarus_decoupled" if seed is not None else "switch_bandit_experiment/runs/noseed_icarus_decoupled"
    os.makedirs(out_dir, exist_ok=True)

    png_path = os.path.join(out_dir, f"{seed}_icarus_decoupled.png")
    plt.savefig(png_path, dpi=150, bbox_inches='tight')

    csv_path = os.path.join(out_dir, f"{seed}_icarus_decoupled.csv")
    with open(csv_path, "w") as fh:
        fh.write("episode,is_optimal,unexpected_uncertainty,expected_uncertainty\n")
        for i, (r, u, e) in enumerate(zip(reward_history, unexpected_history, expected_history)):
            fh.write(f"{i},{int(r)},{u:.6f},{e:.6f}\n")

    print(f"\nPlot saved as '{png_path}' and CSV saved as '{csv_path}'")
    print(f"Max Critic Scale: {max_critic_scale:.3f}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=None, help="Random seed (optional)")
    args = parser.parse_args()
    train(args.episodes, seed=args.seed)
