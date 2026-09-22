#!/usr/bin/env python3
"""
Multi-switch, multi-arm variant of classic.py:
- Adjustable number of arms (default 2) via --arms flag.
- Instead of a single fixed switch at ep 5000, the environment switches
  3 times total at random episodes (uniformly sampled).
- Each switch picks a different optimal arm from the available arms.
- Uses the Classic stateful Actor LR accumulator (decay + boost).
"""

import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import os
from typing import Tuple

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------
DT = 0.02
TAU_M = 0.02
ACTOR_THETA = 2.0

BASE_LR = 1e-2
BASE_NOISE = 1

# Surprises
EXP_SURPRISE_DECAY = 0.01
UNEXP_SURPRISE_DECAY = 0.8  # acts as leaky decay multiplier

# Logistic neuromodulator params
ACH_MAX = 1.0
ACH_K = 3.0
ACH_CENTER = 0.9

NE_MAX = 0
NE_K = 1.0
NE_CENTER = 1000000

# Surprise trace hyperparameters
SURPRISE_VARIANCE_WEIGHT = 0.0
SURPRISE_EPS = 1e-3
TD_SIGNAL_CLIP = 20.0
# Tuned for fast/slow tracking
TD_FAST_ALPHA = 0.097663
TD_SLOW_ALPHA = 0.1  # Optimized from 0.003642 for discrete bandit shock responses
TD_NOVELTY_MARGIN = 0.0

# Actor LR stateful constants
ACTOR_LR_DECAY = 0.1
ACTOR_LR_BOOST = 0  # scale of boost 
ACTOR_LR_MIN = BASE_LR
ACTOR_LR_MAX = 0.1

CRITIC_LR = 1e-2
SEED = 5

CRITIC_ACH_MIN_SCALE = 0.1
CRITIC_ACH_MAX_SCALE = 1.0

NUM_SWITCHES = 3  # Number of random environment switches
N_ARMS = 4        # Number of bandit arms

# --- DETERMINISTIC SWITCH SCHEDULE ---
# Episodes where the optimal arm changes. 
# The arm sequence will cycle through arms (e.g., 0 -> 1 -> 2 -> 3 -> 0...).
SWITCH_SCHEDULE = [15000, 30000, 45000]


def set_global_seed(seed: int | None):
    if seed is None:
        return
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class LocalCritic:
    def __init__(self, n_input: int, device: torch.device):
        self.w_val = torch.zeros(n_input, device=device)
        self.b_val = torch.zeros(1, device=device)
        self.w_var = torch.zeros(n_input, device=device)
        self.b_var = torch.zeros(1, device=device)

    def forward(self, input_spikes: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        val = torch.dot(self.w_val, input_spikes) + self.b_val
        var = F.softplus(torch.dot(self.w_var, input_spikes) + self.b_var) + 1e-4
        return val, var

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def update(self, input_spikes: torch.Tensor, td_error: torch.Tensor, var: torch.Tensor, lr: float):
        # TD-LTP local delta
        val_loss = td_error.abs().detach()
        delta_val = lr * val_loss * td_error.detach()
        self.w_val += delta_val * input_spikes
        self.b_val += delta_val

        # Variance TD-LTP
        target_var = td_error.detach().pow(2)
        err_var = target_var - var.detach()
        delta_var = lr * err_var    
        self.w_var += delta_var * input_spikes
        self.b_var += delta_var


class SNNActor(nn.Module):
    def __init__(self, device: torch.device, n_arms: int = 2):
        super().__init__()
        self.device = device
        self.n_arms = n_arms
        self.w = torch.empty(1, n_arms, device=device).normal_(0.0, 0.1)
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
    z = k * (signal - center)
    if z >= 700:
        return max_val + base
    if z <= -700:
        return base
    return max_val / (1.0 + math.exp(-z)) + base


def _get_deterministic_switches(n_arms: int) -> Tuple[list, list]:
    """Use the global SWITCH_SCHEDULE and generate a deterministic arm sequence.
    Sequence: 0 -> 1 -> 2 -> ... -> (N_ARMS-1) -> 0 ...
    """
    points = sorted(SWITCH_SCHEDULE)
    arm_sequence = []
    current_arm = 0
    for _ in points:
        current_arm = (current_arm + 1) % n_arms
        arm_sequence.append(current_arm)
    return points, arm_sequence

def train(episodes: int = 10000, seed: int | None = SEED, num_switches: int = NUM_SWITCHES, n_arms: int = N_ARMS, **kwargs):
    set_global_seed(seed)
    device = torch.device("cpu")

    # Use deterministic switch points and arm sequence
    switch_points, arm_sequence = _get_deterministic_switches(n_arms)
    switch_points_saved = list(switch_points)
    arm_sequence_saved = list(arm_sequence)

    # Explicit fallback dictionary for type safety and linter scoping
    defaults = {
        'BASE_LR': globals()['BASE_LR'], 'BASE_NOISE': globals()['BASE_NOISE'], 
        'EXP_SURPRISE_DECAY': globals()['EXP_SURPRISE_DECAY'],
        'UNEXP_SURPRISE_DECAY': globals()['UNEXP_SURPRISE_DECAY'], 
        'ACH_MAX': globals()['ACH_MAX'], 'ACH_K': globals()['ACH_K'], 'ACH_CENTER': globals()['ACH_CENTER'],
        'NE_MAX': globals()['NE_MAX'], 'NE_K': globals()['NE_K'], 'NE_CENTER': globals()['NE_CENTER'],
        'SURPRISE_VARIANCE_WEIGHT': globals()['SURPRISE_VARIANCE_WEIGHT'], 'SURPRISE_EPS': globals()['SURPRISE_EPS'],
        'TD_SIGNAL_CLIP': globals()['TD_SIGNAL_CLIP'], 'TD_FAST_ALPHA': globals()['TD_FAST_ALPHA'], 
        'TD_SLOW_ALPHA': globals()['TD_SLOW_ALPHA'],
        'TD_NOVELTY_MARGIN': globals()['TD_NOVELTY_MARGIN'], 'ACTOR_LR_DECAY': globals()['ACTOR_LR_DECAY'],
        'ACTOR_LR_BOOST': globals()['ACTOR_LR_BOOST'], 'ACTOR_LR_MIN': globals()['ACTOR_LR_MIN'], 
        'ACTOR_LR_MAX': globals()['ACTOR_LR_MAX'],
        'CRITIC_LR': globals()['CRITIC_LR'], 'CRITIC_ACH_MIN_SCALE': globals()['CRITIC_ACH_MIN_SCALE'], 
        'CRITIC_ACH_MAX_SCALE': globals()['CRITIC_ACH_MAX_SCALE']
    }

    # Local shadowing
    BASE_LR = kwargs.get('BASE_LR', defaults['BASE_LR'])
    BASE_NOISE = kwargs.get('BASE_NOISE', defaults['BASE_NOISE'])
    EXP_SURPRISE_DECAY = kwargs.get('EXP_SURPRISE_DECAY', defaults['EXP_SURPRISE_DECAY'])
    UNEXP_SURPRISE_DECAY = kwargs.get('UNEXP_SURPRISE_DECAY', defaults['UNEXP_SURPRISE_DECAY'])
    ACH_MAX = kwargs.get('ACH_MAX', defaults['ACH_MAX'])
    ACH_K = kwargs.get('ACH_K', defaults['ACH_K'])
    ACH_CENTER = kwargs.get('ACH_CENTER', defaults['ACH_CENTER'])
    NE_MAX = kwargs.get('NE_MAX', defaults['NE_MAX'])
    NE_K = kwargs.get('NE_K', defaults['NE_K'])
    NE_CENTER = kwargs.get('NE_CENTER', defaults['NE_CENTER'])
    SURPRISE_VARIANCE_WEIGHT = kwargs.get('SURPRISE_VARIANCE_WEIGHT', defaults['SURPRISE_VARIANCE_WEIGHT'])
    SURPRISE_EPS = kwargs.get('SURPRISE_EPS', defaults['SURPRISE_EPS'])
    TD_SIGNAL_CLIP = kwargs.get('TD_SIGNAL_CLIP', defaults['TD_SIGNAL_CLIP'])
    TD_FAST_ALPHA = kwargs.get('TD_FAST_ALPHA', defaults['TD_FAST_ALPHA'])
    TD_SLOW_ALPHA = kwargs.get('TD_SLOW_ALPHA', defaults['TD_SLOW_ALPHA'])
    TD_NOVELTY_MARGIN = kwargs.get('TD_NOVELTY_MARGIN', defaults['TD_NOVELTY_MARGIN'])
    ACTOR_LR_DECAY = kwargs.get('ACTOR_LR_DECAY', defaults['ACTOR_LR_DECAY'])
    ACTOR_LR_BOOST = kwargs.get('ACTOR_LR_BOOST', defaults['ACTOR_LR_BOOST'])
    ACTOR_LR_MIN = kwargs.get('ACTOR_LR_MIN', defaults['ACTOR_LR_MIN'])
    ACTOR_LR_MAX = kwargs.get('ACTOR_LR_MAX', defaults['ACTOR_LR_MAX'])
    CRITIC_LR = kwargs.get('CRITIC_LR', defaults['CRITIC_LR'])
    CRITIC_ACH_MIN_SCALE = kwargs.get('CRITIC_ACH_MIN_SCALE', defaults['CRITIC_ACH_MIN_SCALE'])
    CRITIC_ACH_MAX_SCALE = kwargs.get('CRITIC_ACH_MAX_SCALE', defaults['CRITIC_ACH_MAX_SCALE'])
    device = torch.device("cpu")

    # Quiet mode flag
    quiet = kwargs.get('quiet', False)

    if not quiet:
        print(f"Arms: {n_arms} | Switch points: {switch_points_saved}")
        print(f"Arm sequence: [0] -> {' -> '.join(f'[{a}]' for a in arm_sequence_saved)}")

    actor = SNNActor(device, n_arms=n_arms)
    critic = LocalCritic(1, device)

    # Actor LR Accumulator
    actor_lr_state = float(BASE_LR)

    # Surprise Tracking
    td_fast = 0.0
    td_slow = 0.0
    td_inited = False
    
    avg_expected = 0.0
    avg_unexpected = 0.0

    last_td_signal = 0.0

    input_state = torch.tensor([1.0], device=device)
    input_spikes = torch.tensor([1.0], device=device)

    reward_history = []
    unexpected_history = []
    expected_history = []

    max_critic_scale = 0.0

    # Determine optimal arm for each episode based on switch points
    optimal_arm = 0

    for ep in range(1, episodes + 1):
        # Check if we've hit a switch point
        if switch_points and ep == switch_points[0]:
            optimal_arm = arm_sequence.pop(0)
            switch_points.pop(0)

        # Build reward probabilities: 1.0 for optimal arm, 0.0 for all others
        prob = [0.0] * n_arms
        prob[optimal_arm] = 1.0
        optimal = optimal_arm

        # Decoupled Drives
        current_ne = logistic_drive(NE_MAX, NE_K, NE_CENTER, avg_unexpected, BASE_NOISE)
        current_ne = min(current_ne, 5.0)
        
        current_ach = logistic_drive(ACH_MAX, ACH_K, ACH_CENTER, avg_expected, 0.0)
        current_ach = min(max(current_ach, 0.0), ACH_MAX)
        
        # Stateful Actor LR
        actor_lr_state *= (1.0 - ACTOR_LR_DECAY)
        actor_lr_state += ACTOR_LR_BOOST * current_ach
        actor_lr_state = min(max(actor_lr_state, ACTOR_LR_MIN), ACTOR_LR_MAX)

        actor.reset_state()
        action, spikes = actor(input_spikes, noise_scale=current_ne)
        reward = 1.0 if np.random.rand() < prob[action] else -1.0

        v_curr, var_curr = critic(input_spikes)
        var_curr = torch.clamp(var_curr, min=0.01)
        td_error = reward - v_curr
        td_error_val = float(td_error.item())

        # Critic loss scaling by ACh (Expected uncertainty)
        if ACH_MAX > 0:
            ach_norm = current_ach / ACH_MAX
        else:
            ach_norm = 0.0
        desired_scale = 1.0 + ach_norm * (CRITIC_ACH_MAX_SCALE - 1.0)
        critic_scale = max(CRITIC_ACH_MIN_SCALE, min(CRITIC_ACH_MAX_SCALE, desired_scale))

        max_critic_scale = max(max_critic_scale, critic_scale)

        # Local TD-LTP Update
        critic.update(input_spikes, td_error, var_curr, lr=CRITIC_LR * critic_scale)

        actor.update(td_error_val, spikes, lr=actor_lr_state)

        # EXPECTED UNCERTAINTY (Variance bounds)
        current_sigma = torch.sqrt(var_curr.detach()).item()
        current_sigma = max(current_sigma, SURPRISE_EPS)
        avg_expected = (1.0 - EXP_SURPRISE_DECAY) * avg_expected + EXP_SURPRISE_DECAY * current_sigma

        # UNEXPECTED UNCERTAINTY (Fast-Slow TD)
        td_signal = abs(td_error_val) / (current_sigma ** SURPRISE_VARIANCE_WEIGHT)
        td_signal = float(min(max(td_signal, 0.0), TD_SIGNAL_CLIP))

        if not td_inited:
            td_fast = td_signal
            td_slow = td_signal
            td_inited = True
        else:
            td_fast = (1.0 - TD_FAST_ALPHA) * td_fast + TD_FAST_ALPHA * td_signal
            td_slow = (1.0 - TD_SLOW_ALPHA) * td_slow + TD_SLOW_ALPHA * td_signal
            
        td_novelty = max(0.0, td_fast - td_slow - TD_NOVELTY_MARGIN)
        avg_unexpected = UNEXP_SURPRISE_DECAY * avg_unexpected + td_novelty
        
        last_td_signal = td_signal

        reward_history.append(1 if action == optimal else 0)
        unexpected_history.append(avg_unexpected)
        expected_history.append(avg_expected)

        if ep % 50 == 0 and not quiet:
            accuracy = np.mean(reward_history[-200:]) * 100 if len(reward_history) >= 200 else np.mean(reward_history) * 100
            print(
                f"Ep {ep:4d} | Opt%: {accuracy:5.1f} | NE: {current_ne:.2f} | ACh: {current_ach:.4f} | LR: {actor_lr_state:.4f} | CriticScale: {critic_scale:.3f} | "
                f"Unexpected: {avg_unexpected:.3f} | Expected: {avg_expected:.3f}"
            )

    # Use the saved copy for plotting
    plot_switch_points = switch_points_saved

    plot_episodes = list(range(50, episodes + 1, 50))
    reward_rates = []
    for i in plot_episodes:
        window = reward_history[i - 50:i]
        reward_rates.append(float(np.mean(window) * 100.0))

    plt.figure(figsize=(10, 6))
    plt.plot(plot_episodes, reward_rates, linewidth=2, color="tab:blue", label="Reward Rate")
    colors = ["tab:red", "tab:orange", "tab:purple", "tab:green", "tab:brown", "tab:pink"]
    for idx, sp in enumerate(plot_switch_points):
        plt.axvline(x=sp, color=colors[idx % len(colors)], linestyle="--", 
                    label=f"Switch {idx+1} -> arm {arm_sequence_saved[idx]} (ep {sp})")
    plt.xlabel("Episode")
    plt.ylabel("Reward Rate (%)")
    plt.ylim(0, 105)
    plt.title(f"Multi-Switch Bandit ({n_arms} arms, {num_switches} switches) - Classic")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="lower right")
    
    out_dir = f"sb_multiswitch/runs/{seed}_classic_multiswitch" if seed is not None else "sb_multiswitch/runs/noseed_classic_multiswitch"
    os.makedirs(out_dir, exist_ok=True)

    png_path = os.path.join(out_dir, "plot.png")
    plt.savefig(png_path, dpi=150, bbox_inches='tight')

    csv_path = os.path.join(out_dir, "data.csv")
    with open(csv_path, "w") as fh:
        fh.write("episode,is_optimal,expected,unexpected\n")
        for i, (r, e, u) in enumerate(zip(reward_history, expected_history, unexpected_history)):
            fh.write(f"{i},{int(r)},{e:.6f},{u:.6f}\n")

    meta_path = os.path.join(out_dir, "switches.txt")
    with open(meta_path, "w") as fh:
        fh.write(f"arms: {n_arms}\n")
        fh.write(f"initial_optimal: 0\n")
        for sp, arm in zip(plot_switch_points, arm_sequence_saved):
            fh.write(f"ep {sp} -> arm {arm}\n")

    if not quiet:
        print(f"\nSwitch points: {plot_switch_points}")
        print(f"Plot saved to '{png_path}' and CSV to '{csv_path}'")
        print(f"Max Critic Scale: {max_critic_scale:.3f}")
    
    return np.mean(reward_history[-1000:])  # Return average accuracy of last 1000 episodes


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=60000)
    parser.add_argument("--seed", type=int, default=None, help="Random seed (optional)")
    parser.add_argument("--switches", type=int, default=NUM_SWITCHES, help="Number of random switches")
    parser.add_argument("--arms", type=int, default=N_ARMS, help="Number of bandit arms")
    args = parser.parse_args()
    train(args.episodes, seed=args.seed, num_switches=args.switches, n_arms=args.arms)
