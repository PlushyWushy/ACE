#!/usr/bin/env python3
"""
Switch CartPole with decoupled uncertainty following Yu & Dayan's conjecture:
- NE (Noradrenaline) driven by *unexpected* uncertainty (fast-slow TD novelty)
- ACh (Acetylcholine) driven by *expected* uncertainty (critic's variance estimate)

Critic uses a local TD-LTP style update with loss as a multiplicative factor.
Actor LR decays each step and is boosted by ACh (not set equal to ACh).
"""

#TODO: Run with low ACh center with switch; run with high ACh center without switch.
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
GAMMA = 0.99

# --- NEUROMODULATION PARAMETERS ---
BASE_LR = 0.0000055 
BASE_NOISE = 1

# NE logistic mapping params (for unexpected uncertainty / novelty)
NE_MAX = 3
NE_K = 0.2
NE_CENTER = 15

# ACh logistic mapping params (for expected uncertainty / variance)
ACH_MAX = 1
ACH_K = 1 
ACH_CENTER = 12

# Surprise EMA
EXP_SURPRISE_DECAY = 0.01  
UNEXP_SURPRISE_DECAY = 0.8 

# If =1.0 -> divide by sigma (z-ish). If =0.0 -> ignore variance term.
SURPRISE_VARIANCE_WEIGHT = 0
SURPRISE_EPS = 1e-3

# --------------------------------------------------------------------------- 
# Fast–slow TD novelty (habituation / baseline subtraction) for UNEXPECTED uncertainty
# --------------------------------------------------------------------------- 
# TD signal is |TD| / (sigma**SURPRISE_VARIANCE_WEIGHT), then clipped.
TD_SIGNAL_CLIP = 20.0

# Fast trace reacts quickly; slow trace is "what I'm used to".
TD_FAST_ALPHA = 0.097663     # ~20-step timescale
TD_SLOW_ALPHA = 0.003642    # ~1000-step timescale

# Faster traces for expected uncertainty (ACh)
EXP_FAST_ALPHA = 1    # 
EXP_SLOW_ALPHA = 0.1

# Extra deadzone after baseline subtraction (helps suppress tiny random novelty).
TD_NOVELTY_MARGIN = 0.0

# --------------------------------------------------------------------------- 
# Actor LR modulation params
# --------------------------------------------------------------------------- 
CRITIC_BASE_LR = 0.001

VAR_DECAY = 0
ACTOR_LR_DECAY = 0.1  
ACTOR_LR_BOOST = 0.001
ACTOR_LR_MIN = 1e-4
ACTOR_LR_MAX = 0.1

# Editable global seed (set to None for non-deterministic runs)
SEED = 1234


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

    def update(self, input_spikes: torch.Tensor, td_error: torch.Tensor, var: torch.Tensor, lr: float):
        # TD-LTP: local delta rule with loss as a multiplicative factor.
        val_loss = td_error.abs().detach()
        delta_val = lr * val_loss * td_error.detach()
        self.w_val += delta_val * input_spikes
        self.b_val += delta_val

        if VAR_DECAY > 0.0:
            self.w_var *= (1.0 - VAR_DECAY)
            self.b_var *= (1.0 - VAR_DECAY)

        # Variance tracks squared TD error with a similar loss-scaled update.
        # err_var = (td_error.detach().abs() - var.detach())
        # var_loss = err_var.pow(2)
        # delta_var = lr * var_loss * err_var
        target_var = td_error.detach().pow(2)
        err_var = target_var - var.detach()
        delta_var = lr * err_var    
        self.w_var += delta_var * input_spikes
        self.b_var += delta_var


# --------------------------------------------------------------------------- 
# 3. Modulated Actor
# --------------------------------------------------------------------------- 
class ModulatedActor(nn.Module):
    def __init__(self, n_input: int, device: torch.device):
        super().__init__()
        self.device = device
        self.w = torch.empty(n_input, 2, device=device).normal_(0.0, 0.1)
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

    def update(self, td_error: float, output_spikes: torch.Tensor, current_lr: float):
        eligibility = torch.outer(self.z_eps, output_spikes)
        self.w += current_lr * td_error * eligibility
        self.w.clamp_(-10.0, 10.0)


def train(args):
    set_global_seed(args.seed)
    device = torch.device("cpu")

    env = gym.make("CartPole-v1", render_mode="human" if args.render else None)

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
    actor = ModulatedActor(encoder.n_neurons, device)
    critic = LocalCritic(encoder.n_neurons, device)

    print(f"Start Switch CartPole (Decoupled Uncertainty: NE~Novelty, ACh~Variance). Episodes: {args.episodes}")

    # Traces for unexpected uncertainty (novelty)
    td_fast = 0.0
    td_slow = 0.0
    td_trace_inited = False

    # EMA of unexpected uncertainty (drives NE)
    avg_unexpected = 0.0

    # EMA of expected uncertainty (drives ACh)
    avg_expected = 0.0

    reward_history = []
    unexpected_history = []
    expected_history = []
    variance_history = []
    td2_history = []
    delta_var_history = []
    actor_lr_history = []
    td_history = []

    max_critic_scale = 0.0
    actor_lr_state = args.base_lr

    for ep in range(args.episodes):
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

        switch_active = (ep > args.switch_ep)

        # =====================================================================
        # DECOUPLED NEUROMODULATION
        # =====================================================================
        # NE driven by unexpected uncertainty (novelty)
        current_ne = logistic_drive(args.ne_max, NE_K, NE_CENTER, avg_unexpected, args.base_noise)
        current_ne = min(current_ne, 5.0)

        # ACh uses its own logistic (expected uncertainty -> modulatory signal).
        current_ach = logistic_drive(args.ach_max, ACH_K, ACH_CENTER, avg_expected, args.base_lr)

        td_sum = 0.0
        td_count = 0
        var_sum = 0.0
        var_count = 0
        td2_sum = 0.0
        delta_var_sum = 0.0
        actor_lr_sum = 0.0
        actor_lr_count = 0

        # For printing/debug visibility
        last_td_signal = 0.0
        last_td_fast = td_fast
        last_td_slow = td_slow
        last_td_novelty = 0.0

        while not done:
            spikes = encoder(obs_t)
            action_code, act_spikes = actor(spikes, noise_scale=current_ne)

            real_action = action_code

            # --- ENV SWITCH DYNAMICS ---
            if switch_active:
                env.unwrapped.gravity = 9.8
                env.unwrapped.force_mag = 20.0 if real_action == 1 else 10.0
            else:
                env.unwrapped.gravity = 9.8
                env.unwrapped.force_mag = 10.0

            next_obs, reward, terminated, truncated, _ = env.step(real_action)

            done = terminated or truncated
            next_obs_t = torch.tensor(next_obs, dtype=torch.float32, device=device)

            # ------------------------------- 
            # Critic update
            # ------------------------------- 
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
            td_error_sq = td_error_val ** 2
            var_val = float(var_curr.item())
            err_var_val = abs(td_error_val) - var_val
            td2_sum += td_error_sq

            critic_lr = args.critic_base_lr
            critic.update(spikes, td_error, var_curr, lr=critic_lr)
            delta_var_val = critic_lr * abs(err_var_val) * err_var_val
            delta_var_sum += delta_var_val

            actor_lr_state *= (1.0 - args.actor_lr_decay)
            actor_lr_state += args.actor_lr_boost * current_ach
            actor_lr_state = min(max(actor_lr_state, args.actor_lr_min), args.actor_lr_max)
            actor_lr = actor_lr_state
            actor_lr_sum += actor_lr
            actor_lr_count += 1

            # ------------------------------- 
            # Actor update
            # ------------------------------- 
            actor.update(td_error_val, act_spikes, current_lr=actor_lr)

            # =====================================================================
            # EXPECTED UNCERTAINTY: Direct EMA of critic's variance estimate
            # =====================================================================
            current_sigma = var_curr.detach().item()#torch.sqrt(var_curr.detach()).item()
            current_sigma = max(current_sigma, SURPRISE_EPS)
            
            avg_expected = (1.0 - EXP_SURPRISE_DECAY) * avg_expected + EXP_SURPRISE_DECAY * current_sigma

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
                td_fast = (1.0 - args.td_fast_alpha) * td_fast + args.td_fast_alpha * td_signal
                td_slow = (1.0 - args.td_slow_alpha) * td_slow + args.td_slow_alpha * td_signal

            td_novelty = max(0.0, td_fast - td_slow - TD_NOVELTY_MARGIN)
            avg_unexpected = UNEXP_SURPRISE_DECAY * avg_unexpected + td_novelty #(1.0 - UNEXP_SURPRISE_DECAY) * td_novelty

            # Update dynamics for next step
            current_ne = logistic_drive(args.ne_max, NE_K, NE_CENTER, avg_unexpected, args.base_noise)
            current_ne = min(current_ne, 5.0)
            current_ach = logistic_drive(args.ach_max, ACH_K, ACH_CENTER, avg_expected, args.base_lr)

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
        mean_var = (var_sum / var_count) if var_count > 0 else 0.0
        mean_td2 = (td2_sum / td_count) if td_count > 0 else 0.0
        mean_delta_var = (delta_var_sum / td_count) if td_count > 0 else 0.0
        mean_actor_lr = (actor_lr_sum / actor_lr_count) if actor_lr_count > 0 else 0.0
        td_history.append(mean_abs_td)
        variance_history.append(mean_var)
        td2_history.append(mean_td2)
        delta_var_history.append(mean_delta_var)
        actor_lr_history.append(mean_actor_lr)
        avg_r = float(np.mean(reward_history[-20:])) if len(reward_history) > 0 else 0.0

        if ep % 100 == 0:
            status = "NORMAL" if not switch_active else "HEAVY+SENSITIVE"
            print(
                f"Ep {ep:4d} | {status} | R: {total_reward:3.0f} | Avg: {avg_r:4.1f} | "
                f"NE: {current_ne:.2f} | ACh: {current_ach:.4f} | LR: {mean_actor_lr:.6f} | "
                f"TDsig: {last_td_signal:.2f} | Unexpected: {avg_unexpected:.2f} | Expected: {avg_expected:.2f}"
            )

    print(f"Max Critic Scale: {max_critic_scale:.3f}")

    # -----------------------------------------------------------------------
    # Plot (3 separate subplots for readability)
    # -----------------------------------------------------------------------
    episodes_x = range(len(reward_history))
    window_size = 20

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
    fig.suptitle("CartPole Env Switch — Icarus (Decoupled Uncertainty)", fontsize=14)

    # --- Subplot 1: Reward ---
    ax1.plot(episodes_x, reward_history,
             color="tab:blue", linewidth=0.8, alpha=0.5, label="Reward")
    if len(reward_history) >= window_size:
        rolling = np.convolve(reward_history,
                              np.ones(window_size) / window_size, mode="valid")
        ax1.plot(range(window_size - 1, len(reward_history)), rolling,
                 color="tab:blue", linewidth=2, label=f"{window_size}-ep Avg")
    ax1.axvline(x=args.switch_ep, color="r", linestyle="--", alpha=0.7, label="Wind Start")
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

    out_dir = f"icarussecondpaper/runs/{args.seed}_flagship" if args.seed is not None else "cartpole/runs/noseed_flagship"
    os.makedirs(out_dir, exist_ok=True)

    png_path = os.path.join(out_dir, f"{args.seed}_flagship.png")
    fig.savefig(png_path, dpi=150, bbox_inches="tight")

    csv_path = os.path.join(out_dir, f"{args.seed}_flagship.csv")
    with open(csv_path, "w") as fh:
        fh.write("episode,reward,unexpected_uncertainty,expected_uncertainty,mean_variance,mean_td_error_sq,mean_delta_var,mean_actor_lr,mean_abs_td\n")
        for i, (r, u, e, v, td2, dv, alr, td) in enumerate(zip(reward_history, unexpected_history, expected_history, variance_history, td2_history, delta_var_history, actor_lr_history, td_history)):
            fh.write(f"{i},{r},{u},{e},{v},{td2},{dv},{alr},{td}\n")

    print(f"\nPlot saved as '{png_path}' and CSV saved as '{csv_path}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=15000)
    parser.add_argument("--switch_ep", type=int, default=5000)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed (overrides top-level SEED)")
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
