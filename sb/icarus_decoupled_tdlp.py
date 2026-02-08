#!/usr/bin/env python3
"""
Switching bandit with decoupled uncertainty following Yu & Dayan's conjecture:
- NE (Noradrenaline) driven by *unexpected* uncertainty (fast-slow TD novelty)
- ACh (Acetylcholine) driven by *expected* uncertainty (critic's variance estimate)

Critic uses a local TD-LTP style update with loss as a multiplicative factor.
"""

import math
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os

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


class LocalCritic:
    def __init__(self, device: torch.device):
        self.w_val = torch.zeros(1, device=device)
        self.b_val = torch.zeros(1, device=device)
        self.w_var = torch.zeros(1, device=device)
        self.b_var = torch.zeros(1, device=device)

    def forward(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        val = state * self.w_val + self.b_val
        var = F.softplus(state * self.w_var + self.b_var) + 1e-4
        return val, var

    def update(self, state: torch.Tensor, td_error: torch.Tensor, var: torch.Tensor, lr: float):
        # TD-LTP: local delta rule with loss as a multiplicative factor.
        val_loss = td_error.pow(2).detach()
        delta_val = lr * val_loss * td_error.detach()
        self.w_val += delta_val * state
        self.b_val += delta_val

        # Variance tracks squared TD error with a similar loss-scaled update.
        err_var = (td_error.detach().pow(2) - var.detach())
        var_loss = err_var.pow(2)
        delta_var = lr * var_loss * err_var
        self.w_var += delta_var * state
        self.b_var += delta_var


class SNNActor:
    def __init__(self, device: torch.device):
        self.device = device
        self.w = torch.empty(1, 2, device=device).normal_(0.0, 0.1)
        self.z_eps = torch.zeros(1, device=device)
        self.decay_eps = math.exp(-DT / TAU_M)

    def reset_state(self):
        self.z_eps.zero_()

    def forward(self, input_spikes: torch.Tensor, noise_scale: float) -> tuple[int, torch.Tensor]:
        self.z_eps = self.z_eps * self.decay_eps + input_spikes
        v_det = torch.matmul(self.z_eps, self.w)
        noise = torch.randn_like(v_det) * noise_scale
        v_mem = v_det + noise
        # Clamp to avoid inf/nan when exp overflows.
        exp_arg = torch.clamp((v_mem - ACTOR_THETA) / 2.0, min=-50.0, max=50.0)
        rho = 100.0 * torch.exp(exp_arg)
        probs = 1.0 - torch.exp(-rho * DT)
        probs = torch.nan_to_num(probs, nan=0.0, posinf=1.0, neginf=0.0)
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
    critic = LocalCritic(device)

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
        action, spikes = actor.forward(input_spikes, noise_scale=current_ne)
        reward = 1.0 if np.random.rand() < prob[action] else -1.0

        v_curr, var_curr = critic.forward(input_state)
        var_curr = torch.clamp(var_curr, min=0.01)
        td_error = reward - v_curr
        td_error_val = float(td_error.item())

        # Critic ACh uses its own logistic (expected uncertainty -> learning rate)
        critic_lr = logistic_drive(ACH_MAX, ACH_K, ACH_CENTER, avg_expected, CRITIC_LR)
        critic.update(input_state, td_error, var_curr, lr=critic_lr)

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
                f"CriticLR: {critic_lr:.4f} | TDsig: {last_td_signal:.2f} | "
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
    plt.savefig("switch_bandit_rstdp_decoupled_tdlp.png", dpi=150, bbox_inches="tight")
    print("\nPlot saved as 'switch_bandit_rstdp_decoupled_tdlp.png'")

    # Ensure output directory exists and save both CSV and PNG in runs folder
    out_dir = f"switch_bandit_experiment/runs/{seed}_icarus_decoupled_tdlp" if seed is not None else "switch_bandit_experiment/runs/noseed_icarus_decoupled_tdlp"
    os.makedirs(out_dir, exist_ok=True)

    png_path = os.path.join(out_dir, f"{seed}_icarus_decoupled_tdlp.png")
    plt.savefig(png_path, dpi=150, bbox_inches='tight')

    csv_path = os.path.join(out_dir, f"{seed}_icarus_decoupled_tdlp.csv")
    with open(csv_path, "w") as fh:
        fh.write("episode,is_optimal,unexpected_uncertainty,expected_uncertainty\n")
        for i, (r, u, e) in enumerate(zip(reward_history, unexpected_history, expected_history)):
            fh.write(f"{i},{int(r)},{u:.6f},{e:.6f}\n")

    print(f"\nPlot saved as '{png_path}' and CSV saved as '{csv_path}'")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=None, help="Random seed (optional)")
    args = parser.parse_args()
    train(args.episodes, seed=args.seed)
