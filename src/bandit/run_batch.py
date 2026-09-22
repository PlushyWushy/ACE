#!/usr/bin/env python3
"""
Run the corrected Switch Bandit experiment: ACE, classic, and two ablations.

The ablations are LEVEL-MATCHED, which the originals were not.  In
sb/run_ablations.py the "NE constant" arm used base_noise = 1.5 while full ACE
operated at sigma_NE ~ 0.365 — so that arm changed the *level* of exploration
4x as well as removing its adaptivity, and the two effects cannot be separated.

Here phase 1 runs full ACE and measures the mean ACh and mean sigma_NE it
actually used.  Phase 2 freezes each modulator at that measured mean.  The only
thing that then differs between full ACE and an ablation is whether the signal
adapts — which is the question the ablation is supposed to answer.

    python3 run_batch.py                # all four conditions, 20 seeds
    python3 run_batch.py --seeds 5      # quick smoke test
"""

import argparse
import concurrent.futures as cf
import json
import os

import numpy as np

import ace_sb

HERE = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "results", "bandit"))
RUNS = os.path.join(HERE, "runs")

# classic baseline: both modulators off.  BASE_LR 1e-2 is the value the paper
# reports for the bandit; BASE_NOISE 1.0 matches sb/classic.py, which the paper
# never states — see README.
CLASSIC_NOISE = 1.0
CLASSIC_LR = 1e-2


def _run(kwargs):
    return ace_sb.train(**kwargs)


def launch(jobs, workers):
    out = []
    with cf.ProcessPoolExecutor(max_workers=workers) as ex:
        for d in ex.map(_run, jobs):
            out.append(d)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--episodes", type=int, default=20000)
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 4)
    a = ap.parse_args()
    seeds = list(range(1, a.seeds + 1))
    common = dict(episodes=a.episodes, out_root=RUNS, quiet=False)

    print("=" * 78)
    print("PHASE 1  full ACE (both modulators adaptive)  +  classic (both off)")
    print("=" * 78)
    ace = launch([dict(common, seed=s, tag="ace") for s in seeds], a.workers)
    classic = launch([dict(common, seed=s, tag="classic",
                           NE_MAX=0.0, ACH_MAX=0.0,
                           BASE_NOISE=CLASSIC_NOISE, BASE_LR=CLASSIC_LR)
                      for s in seeds], a.workers)

    # measure ACE's realised operating points, per episode across all seeds
    import pandas as pd
    ach, ne = [], []
    for s in seeds:
        d = pd.read_csv(os.path.join(RUNS, f"{s}_ace", "data.csv"))
        ach.append(d["ach"].values)
        ne.append(d["sigma_ne"].values)
    ach_mean = float(np.mean(np.concatenate(ach)))
    ne_mean = float(np.mean(np.concatenate(ne)))
    ach_span = float(np.max(np.concatenate(ach)) / max(np.min(np.concatenate(ach)), 1e-12) - 1) * 100
    ne_span = float(np.max(np.concatenate(ne)) / max(np.min(np.concatenate(ne)), 1e-12) - 1) * 100

    print("\n" + "-" * 78)
    print(f"ACE realised modulation:   ACh mean {ach_mean:.4f} (range {ach_span:+.1f}%)"
          f"   sigma_NE mean {ne_mean:.4f} (range {ne_span:+.1f}%)")
    if ach_span < 5 or ne_span < 5:
        print("  !! WARNING: a modulator moved <5% — the logistic is still inert.")
        print("     Retune --ach_center / --ne_center to the middle of the signal range.")
    print("-" * 78 + "\n")

    print("=" * 78)
    print("PHASE 2  level-matched ablations (each modulator frozen at ACE's own mean)")
    print("=" * 78)
    only_ne = launch([dict(common, seed=s, tag="only_ne",
                           ACH_MAX=0.0, BASE_LR=ach_mean) for s in seeds], a.workers)
    only_ach = launch([dict(common, seed=s, tag="only_ach",
                            NE_MAX=0.0, BASE_NOISE=ne_mean) for s in seeds], a.workers)

    summary = dict(seeds=seeds, episodes=a.episodes,
                   ace_ach_mean=ach_mean, ace_ne_mean=ne_mean,
                   ace_ach_span_pct=ach_span, ace_ne_span_pct=ne_span,
                   classic_noise=CLASSIC_NOISE, classic_lr=CLASSIC_LR,
                   conditions={"ace": ace, "classic": classic,
                               "only_ne": only_ne, "only_ach": only_ach})
    with open(os.path.join(HERE, "run_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2, default=float)
    print(f"\nwrote {os.path.join(HERE, 'run_summary.json')}")
    print("next:  python3 aggregate.py")


if __name__ == "__main__":
    main()
