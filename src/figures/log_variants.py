#!/usr/bin/env python3
"""Re-run ace_variants.py's gradual/multiswitch arms, logging per-episode traces.

Replicates ace_variants.run() call-for-call (identical RNG consumption) so the
aggregate numbers reproduce exactly; the only addition is per-episode logging.
"""
import argparse, concurrent.futures as cf, os, sys
import numpy as np, torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "bandit"))
import ace_variants as AV
from ace_sb import LocalCritic, logistic_drive, set_global_seed

OUT = os.path.join(os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "results", "bandit")), "traj")


def run(a):
    seed, variant, tag, ach_max, ne_max, base_lr, base_noise = a
    cfg = AV.VARIANTS[variant]
    set_global_seed(seed)
    dev = torch.device("cpu")
    actor, critic = AV.Actor(cfg["arms"], dev), LocalCritic(1, dev)
    x = torch.tensor([1.0], device=dev)
    tdf = tds = 0.0
    inited = False
    n_u = n_e = 0.0
    opt, ach_h, ne_h = [], [], []

    for ep in range(1, cfg["episodes"] + 1):
        prob, optimal = AV.env(ep, cfg)
        sig = min(logistic_drive(ne_max, AV.NE_K, AV.NE_CENTER, n_u, base_noise), 5.0)
        ach = min(max(logistic_drive(ach_max, AV.ACH_K, AV.ACH_CENTER, n_e, base_lr), base_lr),
                  ach_max + base_lr)
        actor.reset_state()
        act, sp = actor(x, sig)
        rw = 1.0 if np.random.rand() < prob[act] else -1.0
        v, var_raw = critic(x)
        var = torch.clamp(var_raw, min=0.01)
        td = rw - v
        tdv = float(td.item())
        critic.update(x, td, var_raw, lr=AV.CRITIC_LR, lr_var=AV.CRITIC_VAR_LR)
        actor.update(tdv, sp, ach)

        n_e = (1.0 - AV.EXP_BETA) * n_e + AV.EXP_BETA * float(var.detach().item())
        u = float(min(max(abs(tdv), 0.0), AV.TD_CLIP))
        if not inited:
            tdf = tds = u
            inited = True
        else:
            tdf = (1 - AV.TD_FAST_ALPHA) * tdf + AV.TD_FAST_ALPHA * u
            tds = (1 - AV.TD_SLOW_ALPHA) * tds + AV.TD_SLOW_ALPHA * u
        n_u = AV.UNEXP_LEAK * n_u + max(0.0, tdf - tds - AV.TD_MARGIN)

        opt.append(1 if act == optimal else 0)
        ach_h.append(ach)
        ne_h.append(sig)

    opt = np.array(opt, dtype=np.int8)
    d = os.path.join(OUT, variant)
    os.makedirs(d, exist_ok=True)
    np.savez_compressed(os.path.join(d, f"{tag}_{seed}.npz"),
                        is_optimal=opt,
                        ach=np.array(ach_h, dtype=np.float32),
                        ne=np.array(ne_h, dtype=np.float32))
    n = cfg["episodes"]
    pf = cfg["switches"][0]
    return (2 * int(opt.sum()) - n, 2 * int(opt[pf:].sum()) - (n - pf),
            float(np.mean(ach_h)), float(np.mean(ne_h)))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=list(AV.VARIANTS), required=True)
    ap.add_argument("--seeds", type=int, default=20)
    a = ap.parse_args()
    S = list(range(1, a.seeds + 1))
    V = a.variant
    jobs = ([(s, V, "ace", AV.ACH_MAX, AV.NE_MAX, AV.BASE_LR, AV.BASE_NOISE) for s in S]
            + [(s, V, "classic", 0.0, 0.0, AV.BASE_LR, 1.0) for s in S])
    with cf.ProcessPoolExecutor(max_workers=os.cpu_count()) as ex:
        res = list(ex.map(run, jobs))
    ace, cls = res[:len(S)], res[len(S):]
    print(f"=== {V.upper()} ({a.seeds} seeds) ===")
    for k, v in [("classic", cls), ("ace", ace)]:
        t = np.array([r[0] for r in v]); p = np.array([r[1] for r in v])
        print(f"  {k:8s} total {t.mean():9.1f} +/- {t.std(ddof=0):8.1f}   post {p.mean():9.1f} +/- {p.std(ddof=0):8.1f}")
