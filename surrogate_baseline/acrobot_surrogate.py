#!/usr/bin/env python3
"""
Surrogate-gradient baseline for the switch Acrobot experiment.
Matches the exact 15,000 episode timescale, physical parameters, and linear place-cell architecture of acrobot/flagship.py.
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

try:
    import gymnasium as gym
except ImportError:
    import gym

# ---------------------------------------------------------------------------
# Constants (aligned with acrobot/flagship.py)
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
TD_FAST_ALPHA = 0.097663   # ~20-step timescale
TD_SLOW_ALPHA = 0.003642   # ~1000-step timescale
TD_NOVELTY_MARGIN = 0.0

CRITIC_BASE_LR = 0.001
VAR_DECAY = 0
ACTOR_LR_DECAY = 0.1
ACTOR_LR_BOOST = 0.1
ACTOR_LR_MIN = 1e-4
ACTOR_LR_MAX = 0.1

SWITCH_EP = 7500
TOTAL_EPISODES = 15000
SEED = 2

PC_GRID_SIZES = [5, 5, 5, 5, 4, 4]   # cos1, sin1, cos2, sin2, w1, w2

# Toggle neuromodulation (False for baseline comparison)
USE_NEUROMOD = False

# ---------------------------------------------------------------------------
# Place Cell Encoder & Surrogate Actor/Critic
# ---------------------------------------------------------------------------
class PlaceCellEncoder(nn.Module):
    def __init__(self, device: torch.device):
        super().__init__()
        self.device = device
        gs = PC_GRID_SIZES

        lows  = [-1.0, -1.0, -1.0, -1.0, -4 * math.pi, -9 * math.pi]
        highs = [ 1.0,  1.0,  1.0,  1.0,  4 * math.pi,  9 * math.pi]

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


class SurrogateCritic(nn.Module):
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
    def __init__(self, n_input: int, n_actions: int, device: torch.device):
        super().__init__()
        self.device = device
        self.w = nn.Parameter(torch.empty(n_input, n_actions, device=device).normal_(0.0, 0.1))
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

        if torch.sum(spikes) == 1:
            action = int(torch.argmax(spikes).item())
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


def swap_link_params(env):
    uw = env.unwrapped
    half = uw.LINK_LENGTH_1 / 2.0
    uw.LINK_LENGTH_2 += half
    uw.LINK_LENGTH_1 = half
    uw.LINK_COM_POS_1 = uw.LINK_LENGTH_1 / 2.0
    uw.LINK_COM_POS_2 = uw.LINK_LENGTH_2 / 2.0


def set_global_seed(seed: int | None):
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train(episodes=TOTAL_EPISODES, seed: int | None = None, switch_ep=SWITCH_EP):
    device = torch.device("cpu")
    set_global_seed(seed)

    env = gym.make("Acrobot-v1")

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
    actor = SurrogateActor(encoder.n_neurons, n_actions=3, device=device)
    critic = SurrogateCritic(encoder.n_neurons).to(device)

    actor_optim = optim.SGD([actor.w], lr=1.0)
    critic_optim = optim.Adam(critic.parameters(), lr=CRICIT_BASE_LR if 'CRICIT_BASE_LR' in globals() else CRITIC_BASE_LR)

    print(f"Start Surrogate Switch Acrobot. Episodes: {episodes}")

    td_fast = 0.0
    td_slow = 0.0
    td_trace_inited = False
    avg_unexpected = 0.0
    avg_expected = 0.0

    reward_history = []
    unexpected_history = []
    expected_history = []
    variance_history = []
    td2_history = []
    delta_var_history = []
    actor_lr_history = []
    td_history = []

    actor_lr_state = BASE_LR
    switched = False

    for ep in range(episodes):
        if ep == switch_ep and not switched:
            swap_link_params(env)
            switched = True
            print(f">>> SWITCH at episode {ep}: link lengths changed (L1 halved, L2 extended) <<<")

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

        if USE_NEUROMOD:
            current_ne = logistic_drive(NE_MAX, NE_K, NE_CENTER, avg_unexpected, BASE_NOISE)
            current_ne = min(current_ne, 5.0)
            current_ach = logistic_drive(ACH_MAX, ACH_K, ACH_CENTER, avg_expected, BASE_LR)
        else:
            current_ne = BASE_NOISE
            current_ach = BASE_LR

        td_sum = 0.0
        td_count = 0
        var_sum = 0.0
        var_count = 0
        td2_sum = 0.0
        delta_var_sum = 0.0
        actor_lr_sum = 0.0
        actor_lr_count = 0

        while not done:
            spikes = encoder(obs_t)
            action_code, act_spikes = actor(spikes, noise_scale=current_ne)

            next_obs, reward, terminated, truncated, _ = env.step(action_code)
            done = terminated or truncated
            next_obs_t = torch.tensor(next_obs, dtype=torch.float32, device=device)

            v_curr, var_curr = critic(spikes)
            var_sum += float(var_curr.item())
            var_count += 1

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
            td_error_sq = td_error_val ** 2
            var_val = float(var_curr.item())
            err_var_val = abs(td_error_val) - var_val
            td2_sum += td_error_sq

            val_loss = td_error.pow(2)
            target_var = td_error.detach().pow(2)
            var_loss = (var_curr - target_var).pow(2)
            total_loss = val_loss + var_loss

            critic_optim.zero_grad()
            total_loss.backward()
            critic_optim.step()

            delta_var_val = CRITIC_BASE_LR * abs(err_var_val) * err_var_val
            delta_var_sum += delta_var_val

            actor_lr_state *= (1.0 - ACTOR_LR_DECAY)
            actor_lr_state += ACTOR_LR_BOOST * current_ach
            actor_lr_state = min(max(actor_lr_state, ACTOR_LR_MIN), ACTOR_LR_MAX)
            actor_lr = actor_lr_state
            actor_lr_sum += actor_lr
            actor_lr_count += 1

            actor.surrogate_update(td_error_val, act_spikes, lr_scale=actor_lr, optimizer=actor_optim)

            current_sigma = torch.sqrt(var_curr.detach()).item()
            current_sigma = max(current_sigma, SURPRISE_EPS)

            avg_expected = ((1.0 - EXP_SURPRISE_DECAY) * avg_expected + EXP_SURPRISE_DECAY * current_sigma)

            td_signal = abs(td_error_val) / (current_sigma ** SURPRISE_VARIANCE_WEIGHT)
            td_signal = float(min(max(td_signal, 0.0), TD_SIGNAL_CLIP))

            if not td_trace_inited:
                td_fast = td_signal
                td_slow = td_signal
                td_trace_inited = True
            else:
                td_fast = ((1.0 - TD_FAST_ALPHA) * td_fast + TD_FAST_ALPHA * td_signal)
                td_slow = ((1.0 - TD_SLOW_ALPHA) * td_slow + TD_SLOW_ALPHA * td_signal)

            td_novelty = max(0.0, td_fast - td_slow - TD_NOVELTY_MARGIN)
            avg_unexpected = UNEXP_SURPRISE_DECAY * avg_unexpected + td_novelty

            obs_t = next_obs_t
            total_reward += float(reward)

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
            status = "NORMAL" if not switched or ep < switch_ep else "SWITCHED"
            print(
                f"Ep {ep:5d} | {status:8s} | R: {total_reward:7.1f} | "
                f"Avg20: {avg_r:7.1f} | LR: {mean_actor_lr:.6f} | "
                f"Unexpected: {avg_unexpected:.2f} | Expected: {avg_expected:.2f}"
            )

    out_dir = f"surrogate_baseline/runs/{seed}_acrobot_surrogate" if seed is not None else "surrogate_baseline/runs/noseed_acrobot_surrogate"
    os.makedirs(out_dir, exist_ok=True)

    csv_path = os.path.join(out_dir, f"{seed}_acrobot_surrogate.csv")
    with open(csv_path, "w") as fh:
        fh.write("episode,reward,unexpected,expected,variance,td2,delta_var,actor_lr,abs_td\n")
        for i, (r, u, e, v, t2, dv, alr, td) in enumerate(zip(
                reward_history, unexpected_history, expected_history,
                variance_history, td2_history, delta_var_history,
                actor_lr_history, td_history)):
            fh.write(f"{i},{r},{u},{e},{v},{t2},{dv},{alr},{td}\n")

    print(f"Saved CSV to '{csv_path}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=TOTAL_EPISODES)
    parser.add_argument("--switch_ep", type=int, default=SWITCH_EP)
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed")
    args = parser.parse_args()

    train(args.episodes, seed=args.seed, switch_ep=args.switch_ep)
