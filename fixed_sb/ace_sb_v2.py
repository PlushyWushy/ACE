#!/usr/bin/env python3
"""
ACE Switch Bandit, v2 — fast decay added to the EXPECTED (ACh) pathway.

Why
---
The oracle test (oracle_test.py) showed the mechanism works: an actor learning
rate that spikes ~x5 at the switch and decays with tau ~ 500 episodes takes the
bandit from 13,824 to 19,226.  It also showed the shape is what matters --
tau = 5000 (a sustained elevation) *loses* 2,741.

v1's ACh could not produce that shape.  Its expected pathway was

    avg_expected = sigma^2                 (EXP_SURPRISE_DECAY = 1, no smoothing)

which tracks the critic variance *level*.  The variance stays elevated for
thousands of episodes after the switch, so ACh inherited a ~2500-episode decay
-- the regime the oracle shows is useless.  Measured spike was only x1.34.

Change
------
The expected pathway now has the same fast/slow + leaky-decay structure the
unexpected pathway already uses, so its response is transient rather than a
level:

    e_fast  = (1-EXP_FAST_ALPHA) e_fast + EXP_FAST_ALPHA * sigma^2
    e_slow  = (1-EXP_SLOW_ALPHA) e_slow + EXP_SLOW_ALPHA * sigma^2
    e_nov   = max(0, e_fast - e_slow - EXP_MARGIN)
    n_e     = EXP_LEAK * n_e + e_nov

Set EXP_FAST_ALPHA = EXP_SLOW_ALPHA to recover v1-like level tracking.

Everything else -- actor, critic, TD-LTP rule, NE pathway, seeding -- is
unchanged from ace_sb.py.
"""

import math
import os

import numpy as np
import torch

from ace_sb import LocalCritic, SNNActor, logistic_drive, set_global_seed

DT, TAU_M = 0.02, 0.02
EPISODES, SWITCH = 20000, 10000

# --- NE pathway (unchanged from ace_sb.py) ---
NE_MAX, NE_K, NE_CENTER, BASE_NOISE = 0.4, 340.0, 0.006, 0.33
TD_FAST_ALPHA, TD_SLOW_ALPHA = 0.097663, 0.1
TD_NOVELTY_MARGIN, UNEXP_LEAK, TD_SIGNAL_CLIP = 0.0, 0.8, 20.0

# --- ACh pathway (new fast/slow + leak on the expected signal) ---
EXP_FAST_ALPHA, EXP_SLOW_ALPHA = 0.05, 0.002
EXP_MARGIN, EXP_LEAK = 0.0, 0.8
ACH_MAX, ACH_K, ACH_CENTER, BASE_LR = 0.05, 5.0, 0.10, 0.010

CRITIC_LR = 1e-2


def train(seed, quiet=True, **kw):
    P = dict(NE_MAX=NE_MAX, NE_K=NE_K, NE_CENTER=NE_CENTER, BASE_NOISE=BASE_NOISE,
             EXP_FAST_ALPHA=EXP_FAST_ALPHA, EXP_SLOW_ALPHA=EXP_SLOW_ALPHA,
             EXP_MARGIN=EXP_MARGIN, EXP_LEAK=EXP_LEAK,
             ACH_MAX=ACH_MAX, ACH_K=ACH_K, ACH_CENTER=ACH_CENTER, BASE_LR=BASE_LR)
    P.update({k: v for k, v in kw.items() if k in P})

    set_global_seed(seed)
    dev = torch.device("cpu")
    actor, critic = SNNActor(dev), LocalCritic(1, dev)
    x = torch.tensor([1.0], device=dev)

    td_f = td_s = 0.0
    e_f = e_s = 0.0
    inited = False
    n_u = n_e = 0.0
    opt = 0
    ach_hist, ne_hist = [], []

    for ep in range(1, EPISODES + 1):
        prob, optimal = ([1.0, 0.0], 0) if ep <= SWITCH else ([0.0, 1.0], 1)

        sig = logistic_drive(P['NE_MAX'], P['NE_K'], P['NE_CENTER'], n_u, P['BASE_NOISE'])
        sig = min(sig, 5.0)
        ach = logistic_drive(P['ACH_MAX'], P['ACH_K'], P['ACH_CENTER'], n_e, P['BASE_LR'])
        ach = min(max(ach, P['BASE_LR']), P['ACH_MAX'] + P['BASE_LR'])

        actor.reset_state()
        action, spikes, _ = actor(x, noise_scale=sig)
        reward = 1.0 if np.random.rand() < prob[action] else -1.0
        v, var = critic(x)
        var = torch.clamp(var, min=0.01)
        td = reward - v
        tdv = float(td.item())
        critic.update(x, td, var, lr=CRITIC_LR)
        actor.update(tdv, spikes, lr=ach)

        # ---- expected pathway: fast/slow novelty on the variance, then leak ----
        s2 = float(var.detach().item())
        if not inited:
            e_f = e_s = s2
        else:
            e_f = (1 - P['EXP_FAST_ALPHA']) * e_f + P['EXP_FAST_ALPHA'] * s2
            e_s = (1 - P['EXP_SLOW_ALPHA']) * e_s + P['EXP_SLOW_ALPHA'] * s2
        n_e = P['EXP_LEAK'] * n_e + max(0.0, e_f - e_s - P['EXP_MARGIN'])

        # ---- unexpected pathway (unchanged) ----
        u = float(min(max(abs(tdv), 0.0), TD_SIGNAL_CLIP))
        if not inited:
            td_f = td_s = u
            inited = True
        else:
            td_f = (1 - TD_FAST_ALPHA) * td_f + TD_FAST_ALPHA * u
            td_s = (1 - TD_SLOW_ALPHA) * td_s + TD_SLOW_ALPHA * u
        n_u = UNEXP_LEAK * n_u + max(0.0, td_f - td_s - TD_NOVELTY_MARGIN)

        opt += 1 if action == optimal else 0
        ach_hist.append(ach)
        ne_hist.append(sig)

    return dict(total=2 * opt - EPISODES, ach=np.array(ach_hist), ne=np.array(ne_hist))


def shape(ach):
    """peak multiplier over pre-switch baseline, and decay tau (episodes)."""
    base = float(np.median(ach[SWITCH - 2000:SWITCH]))
    post = ach[SWITCH:]
    peak = float(post.max())
    mult = peak / base if base > 0 else float('nan')
    # tau: episodes until the excess falls to 1/e of its peak
    exc = post - base
    thr = exc.max() / math.e
    idx = np.where(exc < thr)[0]
    start = int(np.argmax(exc))
    after = idx[idx > start]
    tau = int(after[0] - start) if len(after) else len(post)
    return base, peak, mult, tau
