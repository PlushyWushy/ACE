#!/usr/bin/env python3
"""
Oracle test: hand-scripted learning-rate and noise dynamics on Switch Bandit.

Instead of deriving ACh and NE from surprise signals, this imposes them
directly -- both spike at the switch episode and decay exponentially back to
baseline:

    x(ep) = base                                        ep <  switch
    x(ep) = base + (peak - base) * exp(-(ep-switch)/tau) ep >= switch

This is an UPPER BOUND on what neuromodulation can buy on this task.  A
surprise-driven modulator has to *infer* the switch from the TD error; the
oracle is told exactly when it happened and reacts instantly and noiselessly.
So if a scripted spike-and-decay does not beat a constant baseline here, no
learned or signal-driven version can either -- the ceiling is the ceiling.

Each scripted condition is compared against TWO controls:

  * baseline-constant  -- held at the pre-switch base value.  Answers "does
                          spiking help at all?"
  * level-matched      -- held at the scripted schedule's own time-average.
                          Answers "does the TIMING help, or just the average
                          level?"  This is the control the original ablations
                          lacked, and the one that made the bandit result
                          evaporate.

    python3 oracle_test.py --phase lr      # spike the learning rate only
    python3 oracle_test.py --phase noise   # spike the exploration noise only
    python3 oracle_test.py --phase both    # spike both together, sweep tau
"""

import argparse
import concurrent.futures as cf

import numpy as np
import torch
from scipy import stats

from ace_sb import LocalCritic, SNNActor, set_global_seed

EPISODES = 20000
SWITCH = 10000
LR_BASE = 0.010          # a well-tuned constant learning rate for this task
SIG_BASE = 0.380         # a well-tuned constant exploration noise
SEEDS = list(range(1, 11))


def schedule(ep, base, peak, tau):
    """base before the switch; spike to peak at the switch, exp-decay back."""
    if ep < SWITCH or peak == base:
        return base
    return base + (peak - base) * np.exp(-(ep - SWITCH) / tau)


def run(args):
    seed, lr_peak, lr_tau, sig_peak, sig_tau = args
    set_global_seed(seed)
    device = torch.device("cpu")
    actor, critic = SNNActor(device), LocalCritic(1, device)
    x = torch.tensor([1.0], device=device)

    opt = 0
    lrs, sigs = [], []
    for ep in range(1, EPISODES + 1):
        prob, optimal = ([1.0, 0.0], 0) if ep <= SWITCH else ([0.0, 1.0], 1)
        lr = schedule(ep, LR_BASE, lr_peak, lr_tau)
        sig = schedule(ep, SIG_BASE, sig_peak, sig_tau)
        lrs.append(lr)
        sigs.append(sig)

        actor.reset_state()
        action, spikes, _ = actor(x, noise_scale=sig)
        reward = 1.0 if np.random.rand() < prob[action] else -1.0
        v, var = critic(x)
        var = torch.clamp(var, min=0.01)
        td = reward - v
        critic.update(x, td, var, lr=0.01)
        actor.update(float(td.item()), spikes, lr=lr)
        opt += 1 if action == optimal else 0

    total = 2 * opt - EPISODES
    return total, float(np.mean(lrs)), float(np.mean(sigs))


def evaluate(label, lr_peak, lr_tau, sig_peak, sig_tau, ex):
    """scripted condition + its two controls, paired over seeds"""
    scripted = list(ex.map(run, [(s, lr_peak, lr_tau, sig_peak, sig_tau) for s in SEEDS]))
    tot = np.array([r[0] for r in scripted])
    lr_mean, sig_mean = scripted[0][1], scripted[0][2]

    base = np.array([r[0] for r in ex.map(
        run, [(s, LR_BASE, 1.0, SIG_BASE, 1.0) for s in SEEDS])])
    matched = np.array([r[0] for r in ex.map(
        run, [(s, lr_mean, 1.0, sig_mean, 1.0) for s in SEEDS])])

    _, p_base = stats.ttest_rel(tot, base)
    _, p_match = stats.ttest_rel(tot, matched)
    print(f"  {label:26s} {tot.mean():8.0f} | vs base {tot.mean()-base.mean():+7.0f} "
          f"(p={p_base:.3f}) | vs level-matched {tot.mean()-matched.mean():+7.0f} (p={p_match:.3f})"
          f"   [mean lr {lr_mean:.4f}, sig {sig_mean:.3f}]")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["lr", "noise", "both"], default="lr")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()

    print(f"baseline: LR={LR_BASE}, sigma={SIG_BASE}, switch at {SWITCH}, "
          f"{len(SEEDS)} seeds, {EPISODES} episodes\n")
    with cf.ProcessPoolExecutor(max_workers=a.workers) as ex:
        if a.phase == "lr":
            print("PHASE A — spike the LEARNING RATE only (tau = 500 episodes)")
            for m in [1, 2, 5, 10, 20]:
                evaluate(f"LR x{m:<4g} at switch", LR_BASE * m, 500, SIG_BASE, 1.0, ex)
        elif a.phase == "noise":
            print("PHASE B — spike the NOISE only (tau = 500 episodes)")
            for m in [1, 1.5, 2, 3, 5]:
                evaluate(f"sigma x{m:<4g} at switch", LR_BASE, 1.0, SIG_BASE * m, 500, ex)
        else:
            print("PHASE C — spike BOTH, sweeping the decay constant")
            for tau in [100, 500, 2000, 5000]:
                evaluate(f"LR x5, sigma x2, tau={tau:<5d}", LR_BASE * 5, tau, SIG_BASE * 2, tau, ex)


if __name__ == "__main__":
    main()
