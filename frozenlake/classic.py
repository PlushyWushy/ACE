#!/usr/bin/env python3
"""
Switch FrozenLake with decoupled uncertainty (Classic variant):
- NE (Noradrenaline) driven by *unexpected* uncertainty (fast-slow TD novelty)
- ACh (Acetylcholine) driven by *expected* uncertainty (fast-slow variance novelty)

Actor LR uses a stateful accumulator: decays each step, boosted by ACh.

At the switch point the map layout changes — holes move to new positions,
forcing the agent to discover a completely new path to the goal.
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
# Hyperparameters  (all editable here)
# ---------------------------------------------------------------------------

DT = 0.02
TAU_M = 0.02
ACTOR_THETA = 0.5
GAMMA = 0.99

# --- NEUROMODULATION PARAMETERS ---
BASE_LR = 0.05
BASE_NOISE = 1.8

# NE logistic mapping params (unexpected uncertainty / novelty)
NE_MAX = 2
NE_K = 2000.0
NE_CENTER = 550.5

# ACh logistic mapping params (expected uncertainty / variance)
ACH_MAX = 0.01
ACH_K = 10
ACH_CENTER = 0.5

# Surprise EMA
EXP_SURPRISE_DECAY = 0.05
UNEXP_SURPRISE_DECAY = 0.8

# Variance weighting of surprise signal
SURPRISE_VARIANCE_WEIGHT = 0
SURPRISE_EPS = 1e-3

# ---------------------------------------------------------------------------
# Fast-slow TD novelty (unexpected uncertainty)
# ---------------------------------------------------------------------------
TD_SIGNAL_CLIP = 20.0
TD_FAST_ALPHA = 0.097663
TD_SLOW_ALPHA = 0.003642
TD_NOVELTY_MARGIN = 0.0

# Faster traces for expected uncertainty (ACh)
EXP_FAST_ALPHA = 1
EXP_SLOW_ALPHA = 0.1

# ---------------------------------------------------------------------------
# Actor / Critic LR params
# ---------------------------------------------------------------------------
CRITIC_BASE_LR = 0.1
VAR_DECAY = 0
ACTOR_LR_DECAY = 0.05
ACTOR_LR_BOOST = 0.1
ACTOR_LR_MIN = 0.1
ACTOR_LR_MAX = 0.1

# ---------------------------------------------------------------------------
# Environment / experiment params
# ---------------------------------------------------------------------------
SWITCH_EP = 2000
TOTAL_EPISODES = 4000
SEED = 1234
IS_SLIPPERY = False

# ---------------------------------------------------------------------------
# Map definitions (4×4 FrozenLake)
# ---------------------------------------------------------------------------
MAP_A = [
    "SFFF",
    "FHFH",
    "FFFH",
    "HFFG"
]

MAP_B = [
    "SFFF",
    "FFFH",
    "FHFH",
    "FFFG"
]


def set_global_seed(seed: int | None):
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


def logistic_drive(max_val: float, k: float, center: float,
                   signal: float, base: float) -> float:
    z = k * (signal - center)
    if z >= 700:
        return max_val + base
    if z <= -700:
        return base
    return max_val / (1.0 + math.exp(-z)) + base


# ---------------------------------------------------------------------------
# 1. One-Hot Spike Encoder
# ---------------------------------------------------------------------------
class OneHotSpikeEncoder(nn.Module):
    """Deterministic one-hot encoder for discrete states."""

    def __init__(self, n_states: int, device: torch.device, **_kwargs):
        super().__init__()
        self.device = device
        self.n_states = n_states
        self.n_neurons = n_states

    def forward(self, state_idx: int) -> torch.Tensor:
        spikes = torch.zeros(self.n_neurons, device=self.device)
        spikes[state_idx] = 1.0
        return spikes


# ---------------------------------------------------------------------------
# 2. Local TD-LTP Critic (value + variance)
# ---------------------------------------------------------------------------
class LocalCritic:
    def __init__(self, n_input: int, device: torch.device):
        self.w_val = torch.zeros(n_input, device=device)
        self.b_val = torch.zeros(1, device=device)
        self.w_var = torch.zeros(n_input, device=device)
        self.b_var = torch.zeros(1, device=device)

    def reset_state(self):
        return

    def forward(self, input_spikes: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        val = torch.dot(self.w_val, input_spikes) + self.b_val
        var = F.softplus(torch.dot(self.w_var, input_spikes) + self.b_var) + 1e-4
        return val, var

    def update(self, input_spikes: torch.Tensor, td_error: torch.Tensor,
               var: torch.Tensor, lr: float):
        val_loss = td_error.abs().detach()
        delta_val = lr * val_loss * td_error.detach()
        self.w_val += delta_val * input_spikes
        self.b_val += delta_val

        if VAR_DECAY > 0.0:
            self.w_var *= (1.0 - VAR_DECAY)
            self.b_var *= (1.0 - VAR_DECAY)

        target_var = td_error.detach().pow(2)
        err_var = target_var - var.detach()
        delta_var = lr * err_var
        self.w_var += delta_var * input_spikes
        self.b_var += delta_var


# ---------------------------------------------------------------------------
# 3. Modulated Actor  (4 actions for FrozenLake)
# ---------------------------------------------------------------------------
class ModulatedActor(nn.Module):
    def __init__(self, n_input: int, n_actions: int, device: torch.device):
        super().__init__()
        self.device = device
        self.w = torch.empty(n_input, n_actions, device=device).normal_(0.0, 0.1)
        self.z_eps = torch.zeros(n_input, device=device)
        self.decay_eps = math.exp(-DT / TAU_M)

    def reset_state(self):
        self.z_eps.zero_()

    def forward(self, input_spikes: torch.Tensor,
                noise_scale: float) -> Tuple[int, torch.Tensor]:
        self.z_eps = self.z_eps * self.decay_eps + input_spikes
        v_det = torch.matmul(self.z_eps, self.w)
        noise = torch.randn_like(v_det) * noise_scale
        v_mem = v_det + noise

        spike_input = v_mem - ACTOR_THETA
        exp_arg = torch.clamp(spike_input / 2.0, min=-50.0, max=50.0)
        rho = 100.0 * torch.exp(exp_arg)
        probs = 1.0 - torch.exp(-rho * DT)
        probs = torch.nan_to_num(probs, nan=0.0, posinf=1.0, neginf=0.0)
        spikes = torch.bernoulli(torch.clamp(probs, 0.0, 1.0))

        if torch.sum(spikes) == 1:
            action = int(torch.argmax(spikes).item())
        else:
            action = int(torch.argmax(v_mem).item())
        return action, spikes

    def update(self, td_error: float, output_spikes: torch.Tensor,
               current_lr: float):
        eligibility = torch.outer(self.z_eps, output_spikes)
        self.w += current_lr * td_error * eligibility
        self.w.clamp_(-10.0, 10.0)


# ---------------------------------------------------------------------------
# 4. Map-switch helper
# ---------------------------------------------------------------------------
def make_env(map_desc, is_slippery, render):
    return gym.make("FrozenLake-v1",
                    desc=map_desc,
                    is_slippery=is_slippery,
                    render_mode="human" if render else None)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
def train(args):
    set_global_seed(args.seed)
    device = torch.device("cpu")

    env = make_env(MAP_A, args.slippery, args.render)

    if args.seed is not None:
        try:
            env.reset(seed=args.seed)
        except TypeError:
            pass
        try:
            env.action_space.seed(args.seed)
        except Exception:
            pass

    n_states = 16
    n_actions = 4

    encoder = OneHotSpikeEncoder(n_states, device)
    actor = ModulatedActor(encoder.n_neurons, n_actions=n_actions, device=device)
    critic = LocalCritic(encoder.n_neurons, device)

    print(f"FrozenLake Classic | Episodes: {args.episodes} | Switch at: {args.switch_ep}")
    print(f"Map A -> Map B | Slippery: {args.slippery}")
    print(f"Neurons: {encoder.n_neurons}")

    # Traces for unexpected uncertainty
    td_fast = 0.0
    td_slow = 0.0
    td_trace_inited = False
    avg_unexpected = 0.0

    # Traces for expected uncertainty (classic uses fast-slow on variance)
    avg_expected = 0.0
    exp_fast = 0.0
    exp_slow = 0.0
    exp_trace_inited = False

    reward_history = []
    unexpected_history = []
    expected_history = []
    variance_history = []
    actor_lr_history = []
    td_history = []

    actor_lr_state = args.base_lr
    switched = False

    for ep in range(args.episodes):
        # ---- SWITCH ----
        if ep == args.switch_ep and not switched:
            env.close()
            env = make_env(MAP_B, args.slippery, args.render)
            if args.seed is not None:
                try:
                    env.action_space.seed(args.seed)
                except Exception:
                    pass
            switched = True
            print(f">>> SWITCH at episode {ep}: map changed <<<")

        if args.seed is not None:
            try:
                obs, _ = env.reset(seed=args.seed + ep)
            except TypeError:
                obs, _ = env.reset()
        else:
            obs, _ = env.reset()

        actor.reset_state()
        critic.reset_state()

        done = False
        total_reward = 0.0

        # Neuromodulation
        current_ne = logistic_drive(args.ne_max, NE_K, NE_CENTER,
                                    avg_unexpected, args.base_noise)
        current_ne = min(current_ne, 5.0)
        current_ach = logistic_drive(args.ach_max, ACH_K, ACH_CENTER,
                                     avg_expected, args.base_lr)

        td_sum = 0.0
        td_count = 0
        var_sum = 0.0
        var_count = 0
        actor_lr_sum = 0.0
        actor_lr_count = 0

        while not done:
            spikes = encoder(int(obs))
            action_code, act_spikes = actor(spikes, noise_scale=current_ne)

            next_obs, reward, terminated, truncated, _ = env.step(action_code)
            done = terminated or truncated

            # Critic
            v_curr, var_curr = critic.forward(spikes)
            var_sum += float(var_curr.item())
            var_count += 1

            if not done:
                next_spikes = encoder(int(next_obs))
                v_next, _ = critic.forward(next_spikes)
            else:
                v_next = torch.tensor([0.0], device=device)

            target = reward + GAMMA * v_next
            td_error = target - v_curr
            td_error_val = float(td_error.item())
            td_sum += abs(td_error_val)
            td_count += 1

            critic.update(spikes, td_error, var_curr, lr=args.critic_base_lr)

            # Stateful Actor LR
            actor_lr_state *= (1.0 - args.actor_lr_decay)
            actor_lr_state += args.actor_lr_boost * current_ach
            actor_lr_state = min(max(actor_lr_state, args.actor_lr_min),
                                 args.actor_lr_max)
            actor_lr = actor_lr_state
            actor_lr_sum += actor_lr
            actor_lr_count += 1

            actor.update(td_error_val, act_spikes, current_lr=actor_lr)

            # =================================================================
            # EXPECTED UNCERTAINTY: fast-slow on critic's variance
            # =================================================================
            current_sigma = torch.sqrt(var_curr.detach()).item()
            current_sigma = max(current_sigma, SURPRISE_EPS)

            if not exp_trace_inited:
                exp_fast = current_sigma
                exp_slow = current_sigma
                exp_trace_inited = True
            else:
                exp_fast = ((1.0 - EXP_FAST_ALPHA) * exp_fast
                            + EXP_FAST_ALPHA * current_sigma)
                exp_slow = ((1.0 - EXP_SLOW_ALPHA) * exp_slow
                            + EXP_SLOW_ALPHA * current_sigma)
            exp_novelty = max(0.0, exp_fast)
            avg_expected = (EXP_SURPRISE_DECAY * avg_expected + exp_novelty)

            # =================================================================
            # UNEXPECTED UNCERTAINTY: fast-slow TD novelty
            # =================================================================
            td_signal = abs(td_error_val) / (current_sigma ** SURPRISE_VARIANCE_WEIGHT)
            td_signal = float(min(max(td_signal, 0.0), TD_SIGNAL_CLIP))

            if not td_trace_inited:
                td_fast = td_signal
                td_slow = td_signal
                td_trace_inited = True
            else:
                td_fast = ((1.0 - args.td_fast_alpha) * td_fast
                           + args.td_fast_alpha * td_signal)
                td_slow = ((1.0 - args.td_slow_alpha) * td_slow
                           + args.td_slow_alpha * td_signal)

            td_novelty = max(0.0, td_fast - td_slow - TD_NOVELTY_MARGIN)
            avg_unexpected = UNEXP_SURPRISE_DECAY * avg_unexpected + td_novelty

            current_ne = logistic_drive(args.ne_max, NE_K, NE_CENTER,
                                        avg_unexpected, args.base_noise)
            current_ne = min(current_ne, 5.0)
            current_ach = logistic_drive(args.ach_max, ACH_K, ACH_CENTER,
                                         avg_expected, args.base_lr)

            obs = next_obs
            total_reward += float(reward)

        reward_history.append(total_reward)
        unexpected_history.append(avg_unexpected)
        expected_history.append(avg_expected)
        mean_abs_td = (td_sum / td_count) if td_count > 0 else 0.0
        mean_var = (var_sum / var_count) if var_count > 0 else 0.0
        td_history.append(mean_abs_td)
        variance_history.append(mean_var)
        mean_actor_lr = (actor_lr_sum / actor_lr_count) if actor_lr_count > 0 else 0.0
        actor_lr_history.append(mean_actor_lr)
        avg_r = float(np.mean(reward_history[-100:])) if len(reward_history) > 0 else 0.0

        if ep % 200 == 0:
            status = "MAP_A" if not switched else "MAP_B"
            print(
                f"Ep {ep:5d} | {status:5s} | R: {total_reward:.1f} | "
                f"Avg100: {avg_r:.3f} | NE: {current_ne:.2f} | "
                f"ACh: {current_ach:.4f} | LR: {mean_actor_lr:.6f} | "
                f"Unexpected: {avg_unexpected:.2f} | Expected: {avg_expected:.2f}"
            )

    # -----------------------------------------------------------------------
    # Plot (3 separate subplots)
    # -----------------------------------------------------------------------
    out_dir = (f"frozenlake/runs/{args.seed}_classic" if args.seed is not None
               else "frozenlake/runs/noseed_classic")
    os.makedirs(out_dir, exist_ok=True)

    episodes_x = range(len(reward_history))
    window_size = 100

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
    fig.suptitle("Switch FrozenLake — Classic (Stateful Actor LR)", fontsize=14)

    # --- Subplot 1: Reward (rolling success rate) ---
    if len(reward_history) >= window_size:
        rolling = np.convolve(reward_history,
                              np.ones(window_size) / window_size, mode="valid")
        ax1.plot(range(window_size - 1, len(reward_history)), rolling,
                 color="tab:blue", linewidth=2, label=f"{window_size}-ep Success Rate")
    ax1.axvline(x=args.switch_ep, color="r", linestyle="--", alpha=0.7, label="Switch")
    ax1.set_ylabel("Success Rate")
    ax1.set_ylim(-0.05, 1.05)
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)

    # --- Subplot 2: Unexpected & Expected Uncertainty ---
    ax2.plot(episodes_x, unexpected_history,
             color="tab:orange", linewidth=1.5, alpha=0.8, label="Unexpected (Novelty)")
    ax2.plot(episodes_x, expected_history,
             color="tab:green", linewidth=1.5, alpha=0.8, label="Expected (Variance)")
    ax2.axvline(x=args.switch_ep, color="r", linestyle="--", alpha=0.7)
    ax2.set_ylabel("Uncertainty")
    ax2.legend(loc="upper left")
    ax2.grid(True, alpha=0.3)

    # --- Subplot 3: Critic Variance & Actor LR ---
    color_var = "tab:purple"
    ax3.plot(episodes_x, variance_history,
             color=color_var, linewidth=1.2, alpha=0.8, label="Critic Variance (Mean)")
    ax3.set_ylabel("Variance", color=color_var)
    ax3.tick_params(axis="y", labelcolor=color_var)
    ax3.axvline(x=args.switch_ep, color="r", linestyle="--", alpha=0.7)
    ax3.grid(True, alpha=0.3)

    ax3_lr = ax3.twinx()
    color_lr = "tab:gray"
    ax3_lr.plot(episodes_x, actor_lr_history,
                color=color_lr, linewidth=1.2, alpha=0.8, label="Actor LR (Mean)")
    ax3_lr.set_ylabel("Actor LR", color=color_lr)
    ax3_lr.tick_params(axis="y", labelcolor=color_lr)

    lines3a, labels3a = ax3.get_legend_handles_labels()
    lines3b, labels3b = ax3_lr.get_legend_handles_labels()
    ax3.legend(lines3a + lines3b, labels3a + labels3b, loc="upper left")

    ax3.set_xlabel("Episode")
    fig.tight_layout()

    png_path = os.path.join(out_dir, "plot.png")
    fig.savefig(png_path, dpi=150, bbox_inches="tight")

    csv_path = os.path.join(out_dir, "data.csv")
    with open(csv_path, "w") as fh:
        fh.write("episode,reward,unexpected,expected,variance,actor_lr,abs_td\n")
        for i, (r, u, e, v, alr, td) in enumerate(zip(
                reward_history, unexpected_history, expected_history,
                variance_history, actor_lr_history, td_history)):
            fh.write(f"{i},{r},{u},{e},{v},{alr},{td}\n")

    print(f"\nPlot: '{png_path}'  CSV: '{csv_path}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=TOTAL_EPISODES)
    parser.add_argument("--switch_ep", type=int, default=SWITCH_EP)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--slippery", action="store_true", default=IS_SLIPPERY)
    parser.add_argument("--base_lr", type=float, default=BASE_LR)
    parser.add_argument("--base_noise", type=float, default=BASE_NOISE)
    parser.add_argument("--ne_max", type=float, default=NE_MAX)
    parser.add_argument("--ach_max", type=float, default=ACH_MAX)
    parser.add_argument("--critic_base_lr", type=float, default=CRITIC_BASE_LR)
    parser.add_argument("--actor_lr_decay", type=float, default=ACTOR_LR_DECAY)
    parser.add_argument("--actor_lr_boost", type=float, default=ACTOR_LR_BOOST)
    parser.add_argument("--actor_lr_min", type=float, default=ACTOR_LR_MIN)
    parser.add_argument("--actor_lr_max", type=float, default=ACTOR_LR_MAX)
    parser.add_argument("--td_fast_alpha", type=float, default=TD_FAST_ALPHA)
    parser.add_argument("--td_slow_alpha", type=float, default=TD_SLOW_ALPHA)
    args = parser.parse_args()
    train(args)
