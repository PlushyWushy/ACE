#!/usr/bin/env python3
"""
Gradual and multi-switch variants of the ACE Switch Bandit (same model as ace_sb.py).

  gradual      2 arms, 20k episodes, arms swap over 200 episodes starting at 10k
  multiswitch  4 arms, 60k episodes, correct arm advances at 15k / 30k / 45k

Ablations freeze each modulator at full ACE's mean value.

    python src/bandit/ace_variants.py --variant gradual
    python src/bandit/ace_variants.py --variant multiswitch
"""

import argparse
import concurrent.futures as cf
import math
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ace_sb import LocalCritic, logistic_drive, set_global_seed

DT, TAU_M, ACTOR_THETA = 0.02, 0.02, 2.0
BASE_LR, BASE_NOISE = 0.010, 0.33
ACH_MAX, ACH_K, ACH_CENTER = 0.04, 30.0, 0.5
NE_MAX, NE_K, NE_CENTER = 0.4, 340.0, 0.006
TD_FAST_ALPHA, TD_SLOW_ALPHA, TD_MARGIN = 0.097663, 0.1, 0.0
UNEXP_LEAK, TD_CLIP, EXP_BETA = 0.8, 20.0, 1.0
CRITIC_LR, CRITIC_VAR_LR = 1e-2, 2.0

VARIANTS = {
    "gradual":     dict(arms=2, episodes=20000, switches=[10000], anneal=200),
    "multiswitch": dict(arms=4, episodes=60000, switches=[15000, 30000, 45000], anneal=0),
}


class Actor(nn.Module):
    def __init__(self, n_actions, device):
        super().__init__()
        self.w = torch.empty(1, n_actions, device=device).normal_(0.0, 0.1)
        self.z = torch.zeros(1, device=device)
        self.decay = math.exp(-DT / TAU_M)

    def reset_state(self):
        self.z.zero_()

    def forward(self, x, noise_scale):
        self.z = self.z * self.decay + x
        v = torch.matmul(self.z, self.w) + torch.randn(self.w.shape[1]) * noise_scale
        rho = 100.0 * torch.exp((v - ACTOR_THETA) / 2.0)
        p = torch.clamp(1.0 - torch.exp(-rho * DT), 0.0, 1.0)
        sp = torch.bernoulli(p)
        a = int(torch.argmax(sp).item()) if torch.sum(sp) == 1 else int(torch.argmax(v).item())
        return a, sp

    def update(self, td, sp, lr):
        self.w += lr * td * torch.outer(self.z, sp)
        self.w.clamp_(-10.0, 10.0)


def env(ep, cfg):
    """Return (arm probabilities, optimal arm) for episode ep."""
    n, sw, an = cfg["arms"], cfg["switches"], cfg["anneal"]
    k = sum(1 for s in sw if ep > s)            # how many switches have passed
    cur = k % n
    prob = [0.0] * n
    prob[cur] = 1.0
    if an > 0 and k > 0:                        # gradual: anneal from the previous arm
        last = sw[k - 1]
        if ep <= last + an:
            t = (ep - last) / an
            prev = (k - 1) % n
            prob = [0.0] * n
            prob[prev], prob[cur] = 1.0 - t, t
    return prob, int(np.argmax(prob))


def run(a):
    seed, variant, tag, ach_max, ne_max, base_lr, base_noise = a
    cfg = VARIANTS[variant]
    set_global_seed(seed)
    dev = torch.device("cpu")
    actor, critic = Actor(cfg["arms"], dev), LocalCritic(1, dev)
    x = torch.tensor([1.0], device=dev)
    tdf = tds = 0.0
    inited = False
    n_u = n_e = 0.0
    opt = []
    ach_h, ne_h = [], []

    for ep in range(1, cfg["episodes"] + 1):
        prob, optimal = env(ep, cfg)
        sig = min(logistic_drive(ne_max, NE_K, NE_CENTER, n_u, base_noise), 5.0)
        ach = min(max(logistic_drive(ach_max, ACH_K, ACH_CENTER, n_e, base_lr), base_lr),
                  ach_max + base_lr)
        actor.reset_state()
        act, sp = actor(x, sig)
        rw = 1.0 if np.random.rand() < prob[act] else -1.0
        v, var_raw = critic(x)
        var = torch.clamp(var_raw, min=0.01)
        td = rw - v
        tdv = float(td.item())
        critic.update(x, td, var_raw, lr=CRITIC_LR, lr_var=CRITIC_VAR_LR)   # update on unclamped variance
        actor.update(tdv, sp, ach)

        n_e = (1.0 - EXP_BETA) * n_e + EXP_BETA * float(var.detach().item())
        u = float(min(max(abs(tdv), 0.0), TD_CLIP))
        if not inited:
            tdf = tds = u
            inited = True
        else:
            tdf = (1 - TD_FAST_ALPHA) * tdf + TD_FAST_ALPHA * u
            tds = (1 - TD_SLOW_ALPHA) * tds + TD_SLOW_ALPHA * u
        n_u = UNEXP_LEAK * n_u + max(0.0, tdf - tds - TD_MARGIN)

        opt.append(1 if act == optimal else 0)
        ach_h.append(ach)
        ne_h.append(sig)

    opt = np.array(opt)
    n = cfg["episodes"]
    post_from = cfg["switches"][0]
    total = 2 * int(opt.sum()) - n
    post = 2 * int(opt[post_from:].sum()) - (n - post_from)
    return total, post, float(np.mean(ach_h)), float(np.mean(ne_h))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=list(VARIANTS), required=True)
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 8)
    a = ap.parse_args()
    S = list(range(1, a.seeds + 1))
    V = a.variant
    from scipy import stats

    with cf.ProcessPoolExecutor(max_workers=a.workers) as ex:
        ace = list(ex.map(run, [(s, V, "ace", ACH_MAX, NE_MAX, BASE_LR, BASE_NOISE) for s in S]))
        ach_mean = float(np.mean([r[2] for r in ace]))
        ne_mean = float(np.mean([r[3] for r in ace]))
        cls = list(ex.map(run, [(s, V, "classic", 0.0, 0.0, BASE_LR, 1.0) for s in S]))
        one = list(ex.map(run, [(s, V, "only_ne", 0.0, NE_MAX, ach_mean, BASE_NOISE) for s in S]))
        oah = list(ex.map(run, [(s, V, "only_ach", ACH_MAX, 0.0, BASE_LR, ne_mean) for s in S]))

    res = {"classic": cls, "ace": ace, "only_ne": one, "only_ach": oah}
    print(f"\n=== {V.upper()}  ({a.seeds} seeds) ===")
    print(f"ACE realised: ACh mean {ach_mean:.4f}   sigma_NE mean {ne_mean:.4f}\n")
    print(f"{'condition':12s} {'total':>22s} {'post-switch':>22s}")
    for k, v in res.items():
        t = np.array([r[0] for r in v]); p = np.array([r[1] for r in v])
        print(f"  {k:11s} {t.mean():10.1f} +/- {t.std(ddof=0):8.1f} {p.mean():10.1f} +/- {p.std(ddof=0):8.1f}")
    print("\nPAIRED (ACE vs each):")
    A = np.array([r[0] for r in ace]); Ap = np.array([r[1] for r in ace])
    for k in ["classic", "only_ne", "only_ach"]:
        B = np.array([r[0] for r in res[k]]); Bp = np.array([r[1] for r in res[k]])
        for lbl, u, w in [("total", A, B), ("post ", Ap, Bp)]:
            _, pt = stats.ttest_rel(u, w)
            try:
                _, pw = stats.wilcoxon(u, w)
            except ValueError:
                pw = float("nan")
            print(f"  vs {k:9s} [{lbl}] diff {u.mean()-w.mean():+9.1f}  p_t={pt:.1e}  p_w={pw:.1e}")


if __name__ == "__main__":
    main()
