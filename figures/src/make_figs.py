#!/usr/bin/env python3
"""Generate the ACE paper figures: one per dataset, classic vs ACE in each."""
import csv, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "/Users/a../Desktop/Icarus"
TRAJ = os.path.join(os.path.dirname(os.path.abspath(__file__)), "traj")
OUT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # ../ = figures/
os.makedirs(OUT, exist_ok=True)

# entity -> colour, fixed across every figure (validated: dataviz six checks, light mode)
C_CLASSIC = "#2a78d6"   # slot 1 blue
C_ACE     = "#eb6834"   # slot 2 orange
C_ACH     = "#1baf7a"   # slot 3 aqua   (ACh-only)
C_NE      = "#4a3aa7"   # slot 7 violet (NE-only)
SURFACE   = "#fcfcfb"
INK       = "#0b0b0b"
INK2      = "#52514e"
GRID      = "#d8d7d2"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif"],
    "font.size": 7.5, "axes.labelsize": 7.5, "axes.titlesize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.linewidth": 0.5, "xtick.major.width": 0.5, "ytick.major.width": 0.5,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE, "text.color": INK,
    "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.edgecolor": INK2, "savefig.dpi": 400, "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02, "legend.frameon": False,
})

LW = 1.2
BAND = 0.16


def style(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, color=GRID, linewidth=0.4, alpha=0.9)
    ax.set_axisbelow(True)


def window(mat, w):
    """mat: (seeds, episodes) -> (x centres, mean over seeds, sd over seeds) of
    non-overlapping w-episode means."""
    n = mat.shape[1] // w
    b = mat[:, : n * w].reshape(mat.shape[0], n, w).mean(axis=2)
    x = (np.arange(n) + 0.5) * w
    return x, b.mean(axis=0), b.std(axis=0, ddof=0)


def band(ax, x, m, s, color, label, z=2):
    ax.fill_between(x, m - s, m + s, color=color, alpha=BAND, linewidth=0, zorder=z)
    ax.plot(x, m, color=color, linewidth=LW, label=label, zorder=z + 1, solid_capstyle="round")


def switch_line(ax, xs, label=True):
    for i, x in enumerate(np.atleast_1d(xs)):
        ax.axvline(x, color=INK2, linestyle=(0, (4, 3)), linewidth=0.7, zorder=1)
    if label:
        x0 = np.atleast_1d(xs)[0]
        ax.annotate("switch", xy=(x0, 1.005), xycoords=("data", "axes fraction"),
                    ha="center", va="bottom", fontsize=6.5, color=INK2)


# ---------------------------------------------------------------- loaders
def load_fixed_sb(cond, seeds=range(1, 21)):
    out = []
    for s in seeds:
        p = f"{ROOT}/fixed_sb/runs/{s}_{cond}/data.csv"
        out.append([int(r["is_optimal"]) for r in csv.DictReader(open(p))])
    return np.array(out, dtype=float)


def load_traj(variant, tag, seeds=range(1, 21)):
    return np.array([np.load(f"{TRAJ}/{variant}/{tag}_{s}.npz")["is_optimal"]
                     for s in seeds], dtype=float)


def load_cartpole(root, tag, seeds=range(1, 21), col="reward", need=10000):
    out = []
    for s in seeds:
        d = f"{root}/{s}_{tag}"
        p = f"{d}/{s}_{tag}.csv"
        if not os.path.exists(p):
            continue
        r = [float(x[col]) for x in csv.DictReader(open(p))]
        if len(r) >= need:
            out.append(r[:need])
    return np.array(out, dtype=float)


def load_agg(path, group):
    """pre-aggregated per-episode mean/std curve"""
    m, s = [], []
    for r in csv.DictReader(open(path)):
        if r["group"] == group:
            m.append(float(r["mean_reward"])); s.append(float(r["std_reward"]))
    return np.array(m), np.array(s)


def curve_from_agg(m, s, w):
    n = len(m) // w
    return ((np.arange(n) + 0.5) * w,
            m[: n * w].reshape(n, w).mean(axis=1),
            s[: n * w].reshape(n, w).mean(axis=1))


# ---------------------------------------------------------------- figures
def bandit_fig(name, cls, ace, switches, w, arms, anchor=(0.16, 0.02)):
    fig, ax = plt.subplots(figsize=(3.4, 2.0))
    for mat, c, lab in [(cls, C_CLASSIC, "Classic RSTDP"), (ace, C_ACE, "ACE")]:
        x, m, sd = window(mat, w)
        band(ax, x, 100 * m, 100 * sd, c, lab)
    switch_line(ax, switches)
    ax.axhline(100 / arms, color=INK2, linewidth=0.5, linestyle=(0, (1, 2)), zorder=1)
    style(ax)
    ax.set_xlabel("Episode"); ax.set_ylabel("Optimal-action rate (%)")
    ax.set_ylim(-2, 104); ax.set_xlim(0, cls.shape[1])
    ax.legend(loc="lower left", bbox_to_anchor=anchor, handlelength=1.4,
              borderaxespad=0.0, labelspacing=0.25)
    fig.savefig(f"{OUT}/{name}.png"); plt.close(fig)
    print("wrote", name)


def main():
    # 1 ── default switch bandit
    bandit_fig("fig_bandit_default", load_fixed_sb("classic"), load_fixed_sb("ace"),
               10000, 50, arms=2)

    # 2 ── gradual
    bandit_fig("fig_bandit_gradual", load_traj("gradual", "classic"),
               load_traj("gradual", "ace"), 10000, 50, arms=2)

    # 3 ── multi-switch
    bandit_fig("fig_bandit_multiswitch", load_traj("multiswitch", "classic"),
               load_traj("multiswitch", "ace"), [15000, 30000, 45000], 200, arms=4,
               anchor=(0.05, 0.02))

    # 4 ── switch cartpole
    rs = f"{ROOT}/cartpole_successful/runs_saved"
    cls = load_cartpole(rs, "classic"); ace = load_cartpole(rs, "flagship")
    fig, ax = plt.subplots(figsize=(3.4, 2.0))
    for mat, c, lab in [(cls, C_CLASSIC, "Classic RSTDP"), (ace, C_ACE, "ACE")]:
        x, m, s = window(mat, 50)
        band(ax, x, m, s, c, lab)
    switch_line(ax, 5000)
    style(ax)
    ax.set_xlabel("Episode"); ax.set_ylabel("Reward per episode")
    ax.set_xlim(0, 10000); ax.set_ylim(bottom=0)
    ax.legend(loc="upper left", handlelength=1.4, borderaxespad=0.3)
    fig.savefig(f"{OUT}/fig_cartpole.png"); plt.close(fig)
    print("wrote fig_cartpole", cls.shape, ace.shape)

    # 5 ── ablations, two panels
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.1))
    a = axes[0]
    for cond, c, lab in [("ace", C_ACE, "Full ACE"), ("only_ach", C_ACH, "ACh-only (NE frozen)"),
                         ("only_ne", C_NE, "NE-only (ACh frozen)")]:
        x, m, s = window(load_fixed_sb(cond), 50)
        band(a, x, 100 * m, 100 * s, c, lab)
    switch_line(a, 10000)
    a.axhline(50, color=INK2, linewidth=0.5, linestyle=(0, (1, 2)), zorder=1)
    style(a); a.set_xlabel("Episode"); a.set_ylabel("Optimal-action rate (%)")
    a.set_ylim(-2, 104); a.set_xlim(0, 20000)
    a.set_title("(a) Switch Bandit", loc="left", color=INK)
    a.legend(loc="lower left", bbox_to_anchor=(0.06, 0.02), handlelength=1.4,
             borderaxespad=0.0, labelspacing=0.25)

    b = axes[1]
    x, m, s = window(ace, 50)
    band(b, x, m, s, C_ACE, "Full ACE")
    agg = f"{ROOT}/cartpole_successful/ablation_comparison_stats.csv"
    for group, c, lab in [("ACE (Only ACh)", C_ACH, "ACh-only (NE frozen)"),
                          ("ACE (Only NE)", C_NE, "NE-only (ACh frozen)")]:
        mm, ss = load_agg(agg, group)
        xx, m2, s2 = curve_from_agg(mm, ss, 50)
        band(b, xx, m2, s2, c, lab)
    switch_line(b, 5000)
    style(b); b.set_xlabel("Episode"); b.set_ylabel("Reward per episode")
    b.set_xlim(0, 10000); b.set_ylim(bottom=0)
    b.set_title("(b) Switch CartPole", loc="left", color=INK)
    b.legend(loc="upper left", handlelength=1.4, borderaxespad=0.3)
    fig.tight_layout(pad=0.4)
    fig.savefig(f"{OUT}/fig_ablation.png"); plt.close(fig)
    print("wrote fig_ablation")

    # 6 ── hyperparameter sweeps
    def rows(p):
        return list(csv.DictReader(open(p)))
    cs = f"{ROOT}/cartpole_successful"
    ace_s = rows(f"{cs}/sweep_summary.csv")
    ace_t = rows(f"{cs}/top10_summary.csv")
    cls_s = rows(f"{cs}/classic_sweep_summary.csv")
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    ax.scatter([float(r["mean_total"]) for r in cls_s], [float(r["mean_post"]) for r in cls_s],
               s=14, facecolor="none", edgecolor=C_CLASSIC, linewidth=0.8,
               label="Classic sweep (25)", zorder=3)
    ax.scatter([float(r["mean_total"]) for r in ace_s], [float(r["mean_post"]) for r in ace_s],
               s=14, facecolor="none", edgecolor=C_ACE, linewidth=0.8,
               label="ACE sweep (100)", zorder=3)
    ax.scatter([float(r["mean_total"]) for r in ace_t], [float(r["mean_post"]) for r in ace_t],
               s=22, color=C_ACE, edgecolor=SURFACE, linewidth=0.6,
               label="ACE top-10 (5 seeds)", zorder=4)
    # classic top-5 revalidation: 5 configs x 5 seeds, aggregated per config
    import glob, collections
    cfg = collections.defaultdict(list)
    for f in sorted(glob.glob(f"{cs}/classic_top5_runs/*.csv")):
        name = os.path.basename(f)
        cid = name.split("_seed_")[0]
        r = [float(x["reward"]) for x in csv.DictReader(open(f))]
        cfg[cid].append((float(np.mean(r)), float(np.mean(r[5000:]))))
    ct = [(float(np.mean([v[0] for v in v_])), float(np.mean([v[1] for v in v_])))
          for v_ in cfg.values()]
    ax.scatter([c[0] for c in ct], [c[1] for c in ct], s=22, color=C_CLASSIC,
               edgecolor=SURFACE, linewidth=0.6, label="Classic top-5 (5 seeds)", zorder=4)
    ax.scatter([88.71], [9.825], s=46, marker="*", color=C_CLASSIC,
               edgecolor=SURFACE, linewidth=0.5, label="Classic baseline (main)", zorder=5)
    style(ax)
    ax.set_xlabel("Mean reward per episode, full run")
    ax.set_ylabel("Mean reward per episode, post-switch")
    ax.legend(loc="upper left", handlelength=1.0, borderaxespad=0.3, labelspacing=0.3)
    fig.savefig(f"{OUT}/fig_hyperparam.png"); plt.close(fig)
    print("wrote fig_hyperparam")


main()
