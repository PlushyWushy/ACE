#!/usr/bin/env python3
"""
Switch Bandit trained with ACE: ACh sets the actor learning rate, NE sets the
action noise. Two arms, reward swaps at SWITCH_EP.

    python src/bandit/ace_sb.py --seed 1
"""

import math
import os
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DT = 0.02
TAU_M = 0.02
ACTOR_THETA = 2.0

BASE_LR = 1e-2
BASE_NOISE = 0.33               # sigma_NE floor

EXP_SURPRISE_DECAY = 1.0        # beta_e = 1: no smoothing
UNEXP_SURPRISE_DECAY = 0.8      # leak on the novelty signal

# Logistics mapping surprise to ACh (learning rate) and NE (noise).
# Centres sit mid-range of the observed signals: n_e in ~[0, 1], n_u in ~[0, 0.013].
ACH_MAX = 0.04
ACH_K = 30.0
ACH_CENTER = 0.5

NE_MAX = 0.4
NE_K = 340.0
NE_CENTER = 0.006

SURPRISE_VARIANCE_WEIGHT = 0.0
SURPRISE_EPS = 1e-3
TD_SIGNAL_CLIP = 20.0
TD_FAST_ALPHA = 0.097663
TD_SLOW_ALPHA = 0.1
TD_NOVELTY_MARGIN = 0.0

CRITIC_LR = 1e-2                # value head
CRITIC_VAR_LR = 2.0             # variance head
SWITCH_EP = 10000
SEED = 5

RUNS = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "results", "bandit", "runs"))


def set_global_seed(seed):
    if seed is None:
        return
    np.random.seed(seed)
    torch.manual_seed(seed)
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

    def update(self, input_spikes, td_error, var_raw, lr: float, lr_var: float = None):
        # var_raw is the unclamped variance; using the clamped one winds up the variance weights.
        val_loss = td_error.abs().detach()
        delta_val = lr * val_loss * td_error.detach()
        self.w_val += delta_val * input_spikes
        self.b_val += delta_val

        target_var = td_error.detach().pow(2)
        err_var = target_var - var_raw.detach()
        delta_var = (lr if lr_var is None else lr_var) * err_var
        self.w_var += delta_var * input_spikes
        self.b_var += delta_var


class SNNActor(nn.Module):
    def __init__(self, device: torch.device):
        super().__init__()
        self.device = device
        self.w = torch.empty(1, 2, device=device).normal_(0.0, 0.1)
        self.z_eps = torch.zeros(1, device=device)
        self.decay_eps = math.exp(-DT / TAU_M)

    def reset_state(self):
        self.z_eps.zero_()

    def forward(self, input_spikes: torch.Tensor, noise_scale: float):
        self.z_eps = self.z_eps * self.decay_eps + input_spikes
        v_det = torch.matmul(self.z_eps, self.w)
        noise = torch.randn_like(v_det) * noise_scale
        v_mem = v_det + noise
        rho = 100.0 * torch.exp((v_mem - ACTOR_THETA) / 2.0)
        probs = 1.0 - torch.exp(-rho * DT)
        spikes = torch.bernoulli(torch.clamp(probs, 0.0, 1.0))

        if torch.sum(spikes) == 1:
            action = int(torch.argmax(spikes).item())
            fallback = 0
        else:
            action = int(torch.argmax(v_mem).item())
            fallback = 1
        return action, spikes, fallback

    def update(self, td_error: float, output_spikes, lr: float):
        eligibility = torch.outer(self.z_eps, output_spikes)
        self.w += lr * td_error * eligibility
        self.w.clamp_(-10.0, 10.0)


def logistic_drive(max_val: float, k: float, center: float, signal: float, base: float) -> float:
    if max_val == 0.0:
        return base
    z = k * (signal - center)
    if z >= 700:
        return max_val + base
    if z <= -700:
        return base
    return max_val / (1.0 + math.exp(-z)) + base


def train(episodes=20000, seed=SEED, out_root=RUNS, tag="ace", quiet=False, **kw):
    set_global_seed(seed)
    device = torch.device("cpu")

    p = dict(
        BASE_LR=BASE_LR, BASE_NOISE=BASE_NOISE,
        ACH_MAX=ACH_MAX, ACH_K=ACH_K, ACH_CENTER=ACH_CENTER,
        NE_MAX=NE_MAX, NE_K=NE_K, NE_CENTER=NE_CENTER,
        EXP_SURPRISE_DECAY=EXP_SURPRISE_DECAY,
        UNEXP_SURPRISE_DECAY=UNEXP_SURPRISE_DECAY,
        TD_FAST_ALPHA=TD_FAST_ALPHA, TD_SLOW_ALPHA=TD_SLOW_ALPHA,
        TD_NOVELTY_MARGIN=TD_NOVELTY_MARGIN, TD_SIGNAL_CLIP=TD_SIGNAL_CLIP,
        SURPRISE_VARIANCE_WEIGHT=SURPRISE_VARIANCE_WEIGHT,
        SURPRISE_EPS=SURPRISE_EPS, CRITIC_LR=CRITIC_LR, CRITIC_VAR_LR=CRITIC_VAR_LR,
    )
    p.update({k: v for k, v in kw.items() if k in p})

    actor = SNNActor(device)
    critic = LocalCritic(1, device)

    td_fast = td_slow = 0.0
    td_inited = False
    avg_expected = avg_unexpected = 0.0

    input_spikes = torch.tensor([1.0], device=device)

    hist = dict(opt=[], exp=[], unexp=[], ach=[], ne=[], fb=[])

    for ep in range(1, episodes + 1):
        if ep <= SWITCH_EP:
            prob, optimal = [1.0, 0.0], 0
        else:
            prob, optimal = [0.0, 1.0], 1

        current_ne = logistic_drive(p['NE_MAX'], p['NE_K'], p['NE_CENTER'],
                                    avg_unexpected, p['BASE_NOISE'])
        current_ne = min(current_ne, 5.0)

        current_ach = logistic_drive(p['ACH_MAX'], p['ACH_K'], p['ACH_CENTER'],
                                     avg_expected, p['BASE_LR'])
        current_ach = min(max(current_ach, p['BASE_LR']), p['ACH_MAX'] + p['BASE_LR'])

        actor.reset_state()
        action, spikes, fallback = actor(input_spikes, noise_scale=current_ne)
        reward = 1.0 if np.random.rand() < prob[action] else -1.0

        v_curr, var_curr = critic(input_spikes)
        var_raw = var_curr
        var_curr = torch.clamp(var_curr, min=0.01)
        td_error = reward - v_curr
        td_error_val = float(td_error.item())

        critic.update(input_spikes, td_error, var_raw,
                      lr=p['CRITIC_LR'], lr_var=p['CRITIC_VAR_LR'])
        actor.update(td_error_val, spikes, lr=current_ach)

        # expected uncertainty (critic variance)
        current_sigma = max(torch.sqrt(var_curr.detach()).item(), p['SURPRISE_EPS'])
        avg_expected = ((1.0 - p['EXP_SURPRISE_DECAY']) * avg_expected
                        + p['EXP_SURPRISE_DECAY'] * current_sigma ** 2)

        # unexpected uncertainty (fast/slow TD novelty)
        td_signal = abs(td_error_val) / (current_sigma ** p['SURPRISE_VARIANCE_WEIGHT'])
        td_signal = float(min(max(td_signal, 0.0), p['TD_SIGNAL_CLIP']))
        if not td_inited:
            td_fast = td_slow = td_signal
            td_inited = True
        else:
            td_fast = (1.0 - p['TD_FAST_ALPHA']) * td_fast + p['TD_FAST_ALPHA'] * td_signal
            td_slow = (1.0 - p['TD_SLOW_ALPHA']) * td_slow + p['TD_SLOW_ALPHA'] * td_signal
        td_novelty = max(0.0, td_fast - td_slow - p['TD_NOVELTY_MARGIN'])
        avg_unexpected = p['UNEXP_SURPRISE_DECAY'] * avg_unexpected + td_novelty

        hist['opt'].append(1 if action == optimal else 0)
        hist['exp'].append(avg_expected)
        hist['unexp'].append(avg_unexpected)
        hist['ach'].append(current_ach)
        hist['ne'].append(current_ne)
        hist['fb'].append(fallback)

    out_dir = os.path.join(out_root, f"{seed}_{tag}" if seed is not None else f"noseed_{tag}")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "data.csv"), "w") as fh:
        fh.write("episode,is_optimal,expected,unexpected,ach,sigma_ne,argmax_fallback\n")
        for i in range(episodes):
            fh.write(f"{i},{hist['opt'][i]},{hist['exp'][i]:.6f},{hist['unexp'][i]:.6f},"
                     f"{hist['ach'][i]:.6f},{hist['ne'][i]:.6f},{hist['fb'][i]}\n")

    ach_a, ne_a = np.array(hist['ach']), np.array(hist['ne'])
    total = 2 * int(np.sum(hist['opt'])) - episodes
    post = 2 * int(np.sum(hist['opt'][SWITCH_EP:])) - (episodes - SWITCH_EP)
    diag = dict(
        seed=seed, tag=tag, total=total, post=post,
        ach_min=ach_a.min(), ach_max=ach_a.max(),
        ach_span=(ach_a.max() / ach_a.min() - 1) * 100 if ach_a.min() > 0 else float('nan'),
        ne_min=ne_a.min(), ne_max=ne_a.max(),
        ne_span=(ne_a.max() / ne_a.min() - 1) * 100 if ne_a.min() > 0 else float('nan'),
        fallback_rate=float(np.mean(hist['fb'])),
    )
    if not quiet:
        print(f"[{tag} seed {seed}] total={total:+7d} post={post:+7d} | "
              f"ACh {diag['ach_min']:.4f}-{diag['ach_max']:.4f} ({diag['ach_span']:+.1f}%) | "
              f"sigma_NE {diag['ne_min']:.4f}-{diag['ne_max']:.4f} ({diag['ne_span']:+.1f}%) | "
              f"argmax-fallback {diag['fallback_rate']:.1%}")
    return diag


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--tag", type=str, default="ace")
    ap.add_argument("--out_root", type=str, default=RUNS)
    ap.add_argument("--ne_max", type=float, default=NE_MAX)
    ap.add_argument("--ne_k", type=float, default=NE_K)
    ap.add_argument("--ne_center", type=float, default=NE_CENTER)
    ap.add_argument("--ach_max", type=float, default=ACH_MAX)
    ap.add_argument("--ach_k", type=float, default=ACH_K)
    ap.add_argument("--ach_center", type=float, default=ACH_CENTER)
    ap.add_argument("--base_noise", type=float, default=BASE_NOISE)
    ap.add_argument("--base_lr", type=float, default=BASE_LR)
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()
    train(a.episodes, seed=a.seed, tag=a.tag, out_root=a.out_root, quiet=a.quiet,
          NE_MAX=a.ne_max, NE_K=a.ne_k, NE_CENTER=a.ne_center,
          ACH_MAX=a.ach_max, ACH_K=a.ach_k, ACH_CENTER=a.ach_center,
          BASE_NOISE=a.base_noise, BASE_LR=a.base_lr)
