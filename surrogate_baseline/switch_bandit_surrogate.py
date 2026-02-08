#!/usr/bin/env python3
"""
Surrogate-gradient baseline for the switch bandit experiment.
This is a copy of `switch_bandit_experiment/icarus.py` but the actor is implemented
with surrogate spikes and updated via optimizer steps (surrogate gradient-style),
rather than the original hard-threshold / Hebbian weight update.
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
ACTOR_THETA = 0.5

BASE_LR = 1e-2
BASE_NOISE = 0.5
SURPRISE_DECAY = 0.9

ACH_MAX = 1.0
ACH_K = 3
ACH_CENTER = 0.9

NE_MAX = 5.0
NE_K = 1.0
NE_CENTER = ACH_CENTER

SURPRISE_VARIANCE_WEIGHT = 0.0
SURPRISE_EPS = 1e-3
TD_SIGNAL_CLIP = 20.0
TD_FAST_ALPHA = 1
TD_SLOW_ALPHA = 0.0002
TD_NOVELTY_MARGIN = 0.0

CRITIC_LR = 1e-2
SEED = 5

CRITIC_ACH_MIN_SCALE = 0.1
CRITIC_ACH_MAX_SCALE = 1

# Toggle neuromodulation (if False -> NE and ACh are fixed at base values)
USE_NEUROMOD = False

# ---------------------------------------------------------------------------
# Utilities: surrogate spike function (simple differentiable Heaviside)
# ---------------------------------------------------------------------------
class SurrogateHeaviside(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return (input > 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        (input,) = ctx.saved_tensors
        # simple polynomial surrogate derivative
        grad_input = grad_output / (1.0 + torch.abs(input) * 5.0).pow(2)
        return grad_input


def surrogate_spike(x: torch.Tensor) -> torch.Tensor:
    return SurrogateHeaviside.apply(x)


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
    """Surrogate-gradient actor: weights are nn.Parameter and updates are
    performed through an optimizer step using a surrogate loss that reproduces
    the Hebbian-like td * eligibility update in a differentiable way.
    """

    def __init__(self, device: torch.device):
        super().__init__()
        self.device = device
        # register weights as parameters so optimizers can update them
        self.w = nn.Parameter(torch.empty(1, 2, device=device).normal_(0.0, 0.1))
        self.z_eps = torch.zeros(1, device=device)
        self.decay_eps = math.exp(-DT / TAU_M)

    def reset_state(self):
        self.z_eps.zero_()

    def forward(self, input_spikes: torch.Tensor, noise_scale: float) -> Tuple[int, torch.Tensor]:
        # update eligibility trace
        self.z_eps = self.z_eps * self.decay_eps + input_spikes
        v_det = torch.matmul(self.z_eps, self.w)
        noise = torch.randn_like(v_det) * noise_scale
        v_mem = v_det + noise
        # surrogate spike on membrane potential threshold
        spike_input = v_mem - ACTOR_THETA
        spikes = surrogate_spike(spike_input)

        if torch.sum(spikes) == 1:
            action = int(torch.argmax(spikes).item())
        else:
            # fallback to membrane potential argmax for deterministic choice
            action = int(torch.argmax(v_mem).item())
        return action, spikes

    def surrogate_update(self, td_error: float, output_spikes: torch.Tensor, lr_scale: float, optimizer: torch.optim.Optimizer):
        # build a differentiable surrogate loss whose gradient matches: dw ~ lr * td_error * eligibility
        # eligibility = outer(z_eps, output_spikes)
        eligibility = torch.outer(self.z_eps, output_spikes)
        # surrogate loss = - (lr_scale * td_error) * sum(w * eligibility) -> grad_w = -lr*td*eligibility
        # we want to *add* lr*td*eligibility to w (Hebbian), so minimize negative of that
        coef = float(lr_scale * td_error)
        loss = -coef * torch.sum(self.w * eligibility)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # clamp weights for stability
        with torch.no_grad():
            self.w.clamp_(-10.0, 10.0)


def logistic_drive(max_val: float, k: float, center: float, surprise: float, base: float) -> float:
    # numerically stable logistic used elsewhere
    z = k * (surprise - center)
    if z > 700:
        return max_val + base
    if z < -700:
        return base
    return max_val / (1.0 + math.exp(-z)) + base


def set_global_seed(seed: int | None):
    if seed is None:
        return
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train(episodes: int = 4000, seed: int | None = SEED):
    set_global_seed(seed)
    device = torch.device("cpu")

    actor = SNNActor(device)
    # actor optimizer: we set lr=1.0 and scale updates by lr_scale in loss
    actor_optim = optim.SGD([actor.w], lr=1.0)

    critic = MetaCritic().to(device)
    optim_critic = optim.Adam(critic.parameters(), lr=CRITIC_LR)

    td_fast = 0.0
    td_slow = 0.0
    td_inited = False
    avg_surprise = 0.0
    last_td_signal = 0.0

    input_state = torch.tensor([1.0], device=device)
    input_spikes = torch.tensor([1.0], device=device)

    reward_history = []
    surprise_history = []

    max_critic_scale = 0.0

    for ep in range(1, episodes + 1):
        if ep <= 2000:
            prob = [1.0, 0.0]
            optimal = 0
        else:
            prob = [0.0, 1.0]
            optimal = 1

        if USE_NEUROMOD:
            current_ne = logistic_drive(NE_MAX, NE_K, NE_CENTER, avg_surprise, BASE_NOISE)
            current_ne = min(current_ne, 5.0)
            current_ach = logistic_drive(ACH_MAX, ACH_K, ACH_CENTER, avg_surprise, BASE_LR)
            current_ach = min(max(current_ach, 0.0), ACH_MAX)
        else:
            # fixed baseline noise and learning rate
            current_ne = BASE_NOISE
            current_ach = BASE_LR

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

        if USE_NEUROMOD and ACH_MAX > 0:
            ach_norm = current_ach / ACH_MAX
        else:
            ach_norm = 0.0
        desired_scale = 1.0 + ach_norm * (CRITIC_ACH_MAX_SCALE - 1.0)
        critic_scale = max(CRITIC_ACH_MIN_SCALE, min(CRITIC_ACH_MAX_SCALE, desired_scale))

        max_critic_scale = max(max_critic_scale, critic_scale)

        total_loss = val_loss + var_loss
        scaled_loss = total_loss * critic_scale

        optim_critic.zero_grad()
        scaled_loss.backward()
        optim_critic.step()

        # surrogate-gradient actor update
        actor.surrogate_update(td_error_val, spikes, lr_scale=current_ach, optimizer=actor_optim)

        sigma = torch.sqrt(var_curr.detach()).item()
        sigma = max(sigma, SURPRISE_EPS)
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
        avg_surprise = SURPRISE_DECAY * avg_surprise + (1.0 - SURPRISE_DECAY) * td_novelty
        last_td_signal = td_signal

        reward_history.append(1 if action == optimal else 0)
        surprise_history.append(avg_surprise)

        if ep % 50 == 0:
            accuracy = np.mean(reward_history[-200:]) * 100 if len(reward_history) >= 200 else np.mean(reward_history) * 100
            print(
                f"Ep {ep:4d} | Opt%: {accuracy:5.1f} | NE: {current_ne:.2f} | ACh: {current_ach:.4f} | CriticScale: {critic_scale:.3f} | "
                f"TDsig: {last_td_signal:.2f} | Surprise: {avg_surprise:.3f}"
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
    plt.title("Switch Bandit - Surrogate Actor + Surrogate Critic-style")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="lower right")
    plt.savefig("surrogate_bandit_trace.png", dpi=150, bbox_inches="tight")
    print("\nPlot saved as 'surrogate_bandit_trace.png'")
    print(f"Max Critic Scale: {max_critic_scale:.3f}")

    out_dir = f"surrogate_baseline/runs/{seed}_bandit_surrogate" if seed is not None else "surrogate_baseline/runs/noseed_bandit_surrogate"
    os.makedirs(out_dir, exist_ok=True)

    png_path = os.path.join(out_dir, f"{seed}_bandit_surrogate.png")
    plt.savefig(png_path, dpi=150, bbox_inches='tight')

    csv_path = os.path.join(out_dir, f"{seed}_bandit_surrogate.csv")
    with open(csv_path, "w") as fh:
        fh.write("episode,is_optimal,surprise\n")
        for i, (r, s) in enumerate(zip(reward_history, surprise_history)):
            fh.write(f"{i},{int(r)},{s:.6f}\n")

    print(f"\nPlot saved as '{png_path}' and CSV saved as '{csv_path}'")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=None, help="Random seed (optional)")
    args = parser.parse_args()
    train(args.episodes, seed=args.seed)
