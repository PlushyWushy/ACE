#!/usr/bin/env python3
"""
Switch Acrobot with a fixed LR schedule that spikes at the switch point:
- Before the switch: constant base actor LR.
- At the switch: LR jumps to SWITCH_LR_PEAK.
- Holds at peak for SWITCH_LR_HOLD_EPS episodes.
- Linearly anneals back to base LR over SWITCH_LR_ANNEAL_EPS episodes.

No neuromodulation (NE/ACh) drives the actor LR — it's entirely
schedule-driven.  This serves as the "engineered upper-bound" baseline
for the Icarus neuromodulated variant.

At the switch point the physical link parameters are swapped, so the
torque that was applied to the "elbow" joint now effectively acts on the
"shoulder" joint.
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
RHO_PC = 50.0
TAU_M = 0.02
ACTOR_THETA = 2.0
GAMMA = 0.99

# --- FIXED BASE LR ---
BASE_LR = 0.001
BASE_NOISE = 2

# --- SPIKE SCHEDULE PARAMS ---
SWITCH_LR_PEAK = 0.1        # Peak LR right at the switch
SWITCH_LR_HOLD_EPS = 1       # Episodes to hold peak before annealing
SWITCH_LR_ANNEAL_EPS = 200   # Episodes to linearly anneal back to BASE_LR

# --- CRITIC ---
CRITIC_BASE_LR = 0.001
VAR_DECAY = 0

# --- SURPRISE TRACKING (still tracked for logging, not used for LR) ---
EXP_SURPRISE_DECAY = 0.01
UNEXP_SURPRISE_DECAY = 0.8
SURPRISE_VARIANCE_WEIGHT = 0
SURPRISE_EPS = 1e-3
TD_SIGNAL_CLIP = 20.0
TD_FAST_ALPHA = 0.097663
TD_SLOW_ALPHA = 0.003642
TD_NOVELTY_MARGIN = 0.0

# ---------------------------------------------------------------------------
# Environment / experiment params
# ---------------------------------------------------------------------------
SWITCH_EP = 5000
TOTAL_EPISODES = 10000
SEED = 1234

# ---------------------------------------------------------------------------
# Place Cell Encoder grid sizes (one per obs dim)
# ---------------------------------------------------------------------------
PC_GRID_SIZES = [5, 5, 5, 5, 4, 4]   # cos1, sin1, cos2, sin2, w1, w2


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
# Fixed LR spike schedule
# ---------------------------------------------------------------------------
def spike_lr(ep: int, switch_ep: int, base_lr: float, peak_lr: float,
             hold_eps: int, anneal_eps: int) -> float:
    """Return the actor LR for the given episode under the spike schedule.
    Spikes at ep=0 (initial learning) and at switch_ep."""
    # 1) Initial spike at episode 0
    if ep < anneal_eps:
        frac = 1.0 - (ep / float(anneal_eps))
        return base_lr + (peak_lr - base_lr) * frac
        
    # 2) Before the switch
    if ep < switch_ep:
        return base_lr
        
    # 3) Switch spike
    t = ep - switch_ep
    if t < hold_eps:
        return peak_lr
    ta = t - hold_eps
    if ta < anneal_eps:
        frac = 1.0 - (ta / float(anneal_eps))
        return base_lr + (peak_lr - base_lr) * frac
        
    return base_lr


# ---------------------------------------------------------------------------
# 1. Place Cell Encoder  (6-D for Acrobot)
# ---------------------------------------------------------------------------
class PlaceCellEncoder(nn.Module):
    """Radial-basis / place-cell encoder for Acrobot's 6-D observation.

    Observation layout:
        [cos(θ1), sin(θ1), cos(θ2), sin(θ2), ω1, ω2]
    Ranges:
        cos/sin: [-1, 1]
        ω1: [-4π, 4π] ≈ [-12.57, 12.57]
        ω2: [-9π, 9π] ≈ [-28.27, 28.27]
    """
    def __init__(self, device: torch.device, grid_sizes=None):
        super().__init__()
        self.device = device
        gs = grid_sizes or PC_GRID_SIZES

        lows  = [-1.0, -1.0, -1.0, -1.0, -4*math.pi, -9*math.pi]
        highs = [ 1.0,  1.0,  1.0,  1.0,  4*math.pi,  9*math.pi]

        linspaces = []
        for lo, hi, n in zip(lows, highs, gs):
            linspaces.append(torch.linspace(lo, hi, n, device=device))

        mesh = torch.meshgrid(*linspaces, indexing="ij")
        centers = torch.stack([x.flatten() for x in mesh], dim=1)
        self.register_buffer("centers", centers)

        sigma_list = []
        for ls in linspaces:
            if len(ls) > 1:
                sigma_list.append((ls[1] - ls[0]) / 1.5)
            else:
                sigma_list.append(torch.tensor(1.0, device=device))
        sigmas = torch.stack(sigma_list).unsqueeze(0)
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
# 3. Modulated Actor  (3 actions for Acrobot)
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
# 4. Joint-switch helper
# ---------------------------------------------------------------------------
def swap_link_params(env):
    """Halve the upper link and add the removed length to the lower link.
    Default: L1=1.0, L2=1.0  →  After: L1=0.5, L2=1.5
    This fundamentally changes the swing-up dynamics."""
    uw = env.unwrapped
    half = uw.LINK_LENGTH_1 / 2.0
    uw.LINK_LENGTH_2 += half
    uw.LINK_LENGTH_1 = half
    uw.LINK_COM_POS_1 = uw.LINK_LENGTH_1 / 2.0
    uw.LINK_COM_POS_2 = uw.LINK_LENGTH_2 / 2.0


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
def train(args):
    set_global_seed(args.seed)
    device = torch.device("cpu")

    env = gym.make("Acrobot-v1", render_mode="human" if args.render else None)

    if args.seed is not None:
        try:
            env.reset(seed=args.seed)
        except TypeError:
            try:
                env.seed(args.seed)
            except Exception:
                pass
        try:
            env.action_space.seed(args.seed)
        except Exception:
            pass
        try:
            env.observation_space.seed(args.seed)
        except Exception:
            pass

    encoder = PlaceCellEncoder(device)
    actor = ModulatedActor(encoder.n_neurons, n_actions=3, device=device)
    critic = LocalCritic(encoder.n_neurons, device)

    print(f"Acrobot Spike | Episodes: {args.episodes} | Switch at: {args.switch_ep}")
    print(f"LR schedule: base={args.base_lr}, peak={args.switch_lr_peak}, "
          f"hold={args.switch_lr_hold_eps}, anneal={args.switch_lr_anneal_eps}")
    print(f"Place-cell neurons: {encoder.n_neurons}")

    # Surprise traces (tracked for logging, not used for LR)
    td_fast = 0.0
    td_slow = 0.0
    td_trace_inited = False
    avg_unexpected = 0.0
    avg_expected = 0.0

    reward_history = []
    unexpected_history = []
    expected_history = []
    variance_history = []
    actor_lr_history = []
    td_history = []

    switched = False

    for ep in range(args.episodes):
        # ---- SWITCH ----
        if ep == args.switch_ep and not switched:
            swap_link_params(env)
            switched = True
            print(f">>> SWITCH at episode {ep}: link lengths changed (L1 halved, L2 extended) <<<")

        if args.seed is not None:
            try:
                obs, _ = env.reset(seed=args.seed + ep)
            except TypeError:
                try:
                    env.seed(args.seed + ep)
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

        # Fixed LR from spike schedule
        actor_lr = spike_lr(ep, args.switch_ep, args.base_lr,
                            args.switch_lr_peak, args.switch_lr_hold_eps,
                            args.switch_lr_anneal_eps)

        td_sum = 0.0
        td_count = 0
        var_sum = 0.0
        var_count = 0

        while not done:
            spikes = encoder(obs_t)
            action_code, act_spikes = actor(spikes, noise_scale=args.base_noise)

            next_obs, reward, terminated, truncated, _ = env.step(action_code)
            done = terminated or truncated
            next_obs_t = torch.tensor(next_obs, dtype=torch.float32, device=device)

            # Critic
            v_curr, var_curr = critic.forward(spikes)
            var_sum += float(var_curr.item())
            var_count += 1

            if not done:
                next_spikes = encoder(next_obs_t)
                v_next, _ = critic.forward(next_spikes)
            else:
                v_next = torch.tensor([0.0], device=device)

            target = reward + GAMMA * v_next
            td_error = target - v_curr
            td_error_val = float(td_error.item())
            td_sum += abs(td_error_val)
            td_count += 1

            critic.update(spikes, td_error, var_curr, lr=args.critic_base_lr)

            # Actor update with the fixed schedule LR
            actor.update(td_error_val, act_spikes, current_lr=actor_lr)

            # Track surprise signals (for logging / comparison only)
            current_sigma = torch.sqrt(var_curr.detach()).item()
            current_sigma = max(current_sigma, SURPRISE_EPS)
            avg_expected = ((1.0 - EXP_SURPRISE_DECAY) * avg_expected
                            + EXP_SURPRISE_DECAY * current_sigma)

            td_signal = abs(td_error_val) / (current_sigma ** SURPRISE_VARIANCE_WEIGHT)
            td_signal = float(min(max(td_signal, 0.0), TD_SIGNAL_CLIP))

            if not td_trace_inited:
                td_fast = td_signal
                td_slow = td_signal
                td_trace_inited = True
            else:
                td_fast = ((1.0 - TD_FAST_ALPHA) * td_fast
                           + TD_FAST_ALPHA * td_signal)
                td_slow = ((1.0 - TD_SLOW_ALPHA) * td_slow
                           + TD_SLOW_ALPHA * td_signal)

            td_novelty = max(0.0, td_fast - td_slow - TD_NOVELTY_MARGIN)
            avg_unexpected = UNEXP_SURPRISE_DECAY * avg_unexpected + td_novelty

            obs_t = next_obs_t
            total_reward += float(reward)

        reward_history.append(total_reward)
        unexpected_history.append(avg_unexpected)
        expected_history.append(avg_expected)
        mean_abs_td = (td_sum / td_count) if td_count > 0 else 0.0
        mean_var = (var_sum / var_count) if var_count > 0 else 0.0
        td_history.append(mean_abs_td)
        variance_history.append(mean_var)
        actor_lr_history.append(actor_lr)
        avg_r = float(np.mean(reward_history[-20:])) if len(reward_history) > 0 else 0.0

        if ep % 100 == 0:
            status = "NORMAL" if not switched or ep < args.switch_ep else "SWITCHED"
            print(
                f"Ep {ep:5d} | {status:8s} | R: {total_reward:7.1f} | "
                f"Avg20: {avg_r:7.1f} | LR: {actor_lr:.6f} | "
                f"Unexpected: {avg_unexpected:.2f} | Expected: {avg_expected:.2f}"
            )

    # -----------------------------------------------------------------------
    # Plot (3 separate subplots)
    # -----------------------------------------------------------------------
    out_dir = (f"acrobot/runs/{args.seed}_spike" if args.seed is not None
               else "acrobot/runs/noseed_spike")
    os.makedirs(out_dir, exist_ok=True)

    episodes_x = range(len(reward_history))
    window_size = 20

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
    fig.suptitle("Switch Acrobot — Spike (Fixed LR Schedule)", fontsize=14)

    # --- Subplot 1: Reward ---
    ax1.plot(episodes_x, reward_history,
             color="tab:blue", linewidth=0.8, alpha=0.5, label="Reward")
    if len(reward_history) >= window_size:
        rolling = np.convolve(reward_history,
                              np.ones(window_size) / window_size, mode="valid")
        ax1.plot(range(window_size - 1, len(reward_history)), rolling,
                 color="tab:blue", linewidth=2, label=f"{window_size}-ep Avg")
    ax1.axvline(x=args.switch_ep, color="r", linestyle="--", alpha=0.7, label="Switch")
    ax1.set_ylabel("Reward")
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
    color_lr = "tab:red"
    ax3_lr.plot(episodes_x, actor_lr_history,
                color=color_lr, linewidth=1.5, alpha=0.8, label="Actor LR (Spike)")
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
    parser.add_argument("--base_lr", type=float, default=BASE_LR)
    parser.add_argument("--base_noise", type=float, default=BASE_NOISE)
    parser.add_argument("--critic_base_lr", type=float, default=CRITIC_BASE_LR)
    parser.add_argument("--switch_lr_peak", type=float, default=SWITCH_LR_PEAK)
    parser.add_argument("--switch_lr_hold_eps", type=int, default=SWITCH_LR_HOLD_EPS)
    parser.add_argument("--switch_lr_anneal_eps", type=int, default=SWITCH_LR_ANNEAL_EPS)
    args = parser.parse_args()
    train(args)
