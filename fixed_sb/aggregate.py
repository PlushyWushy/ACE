#!/usr/bin/env python3
"""
Aggregate fixed_sb runs into paper-ready numbers.

Reports totals in the paper's +1/-1 convention (2 * optimal_count - n_episodes),
population SD (ddof=0, matching the paper), paired t-test and Wilcoxon p-values,
and — the part that was missing before — the realised modulation range of ACh
and sigma_NE in every condition.

    python3 aggregate.py
"""

import glob
import os

import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, "runs")
SWITCH_EP = 10000
CONDITIONS = ["classic", "ace", "only_ne", "only_ach"]


def load(tag):
    rows = {}
    for f in sorted(glob.glob(os.path.join(RUNS, f"*_{tag}", "data.csv"))):
        head = os.path.basename(os.path.dirname(f)).split("_")[0]
        if not head.isdigit():
            continue
        seed = int(head)
        d = pd.read_csv(f)
        n = len(d)
        rows[seed] = dict(
            total=2 * int(d["is_optimal"].sum()) - n,
            post=2 * int(d.loc[d["episode"] >= SWITCH_EP, "is_optimal"].sum()) - (n - SWITCH_EP),
            ach_min=d["ach"].min(), ach_max=d["ach"].max(),
            ne_min=d["sigma_ne"].min(), ne_max=d["sigma_ne"].max(),
            fallback=d["argmax_fallback"].mean(),
        )
    return pd.DataFrame(rows).T.sort_index()


def paired(a, b, label):
    d = a - b
    t, pt = stats.ttest_rel(a, b)
    try:
        w, pw = stats.wilcoxon(a, b)
    except ValueError:
        w, pw = np.nan, np.nan
    print(f"  {label:34s} diff {d.mean():+9.1f}   p_t={pt:.2e}   p_w={pw:.2e}"
          f"   all-same-sign={bool(np.all(d > 0) or np.all(d < 0))}")
    return pt, pw


def main():
    data = {c: load(c) for c in CONDITIONS}
    missing = [c for c, v in data.items() if v.empty]
    if missing:
        print(f"no runs found for: {missing}  — run  python3 run_batch.py  first")
        if len(missing) == len(CONDITIONS):
            return

    print("=" * 84)
    print("REALISED MODULATION  (this is the check that was missing before)")
    print("=" * 84)
    for c, v in data.items():
        if v.empty:
            continue
        aspan = (v["ach_max"].max() / max(v["ach_min"].min(), 1e-12) - 1) * 100
        nspan = (v["ne_max"].max() / max(v["ne_min"].min(), 1e-12) - 1) * 100
        flag = "  <-- INERT" if (aspan < 5 and nspan < 5) else ""
        print(f"  {c:10s} ACh {v['ach_min'].min():.4f}-{v['ach_max'].max():.4f} ({aspan:+8.1f}%)   "
              f"sigma_NE {v['ne_min'].min():.4f}-{v['ne_max'].max():.4f} ({nspan:+8.1f}%)   "
              f"argmax-fallback {v['fallback'].mean():.1%}{flag}")

    print("\n" + "=" * 84)
    print("REWARD  (paper convention: +1 optimal / -1 suboptimal; SD is ddof=0)")
    print("=" * 84)
    print(f"  {'condition':12s} {'total':>22s} {'post-switch':>22s}")
    for c, v in data.items():
        if v.empty:
            continue
        print(f"  {c:12s} {v['total'].mean():10.1f} +/- {v['total'].std(ddof=0):8.1f}"
              f"   {v['post'].mean():10.1f} +/- {v['post'].std(ddof=0):8.1f}")

    if not data["ace"].empty:
        idx = data["ace"].index
        print("\n" + "=" * 84)
        print("PAIRED COMPARISONS")
        print("=" * 84)
        for win in ["total", "post"]:
            print(f"\n [{win}]")
            for other in ["classic", "only_ne", "only_ach"]:
                if data[other].empty:
                    continue
                common = idx.intersection(data[other].index)
                paired(data["ace"].loc[common, win].values,
                       data[other].loc[common, win].values,
                       f"ACE vs {other}")

    out = os.path.join(HERE, "comparison.csv")
    pd.concat({c: v for c, v in data.items() if not v.empty},
              names=["condition", "seed"]).to_csv(out)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
