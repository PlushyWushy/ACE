#!/usr/bin/env python3
"""
Offline training visualization for icarus_decoupled_tdlp_ach_lr_decay.

Captures per-step state (including weights) and renders a single, dense summary
image per episode where each step is represented as a column.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import torch

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import icarus_decoupled_tdlp_ach_lr_decay as model


@dataclass
class ModulationState:
    td_fast: float = 0.0
    td_slow: float = 0.0
    td_trace_inited: bool = False
    avg_unexpected: float = 0.0
    avg_expected: float = 0.0
    exp_fast: float = 0.0
    exp_slow: float = 0.0
    exp_trace_inited: bool = False
    actor_lr_state: float = 0.0


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norm = np.zeros_like(matrix, dtype=np.float32)
    for i in range(matrix.shape[0]):
        row = matrix[i]
        min_v = float(np.min(row))
        max_v = float(np.max(row))
        if max_v - min_v < 1e-8:
            norm[i] = 0.5
        else:
            norm[i] = (row - min_v) / (max_v - min_v)
    return norm


def capture_episode(
    env,
    encoder: model.PlaceCellEncoder,
    actor: model.ModulatedActor,
    critic: model.LocalCritic,
    args: argparse.Namespace,
    ep_idx: int,
    mod_state: ModulationState,
    device: torch.device,
) -> Dict[str, np.ndarray]:
    if args.seed is not None:
        try:
            obs, _ = env.reset(seed=args.seed + ep_idx)
        except TypeError:
            try:
                env.seed(args.seed + ep_idx)
                obs, _ = env.reset()
            except Exception:
                obs, _ = env.reset()
    else:
        obs, _ = env.reset()

    obs_t = torch.tensor(obs, dtype=torch.float32, device=device)
    actor.reset_state()
    critic.reset_state()

    inverted = (ep_idx > args.switch_episode)

    current_ne = model.logistic_drive(
        args.ne_max, model.NE_K, model.NE_CENTER, mod_state.avg_unexpected, args.base_noise
    )
    current_ne = min(current_ne, 5.0)
    current_ach = model.logistic_drive(
        args.ach_max, model.ACH_K, model.ACH_CENTER, mod_state.avg_expected, args.base_lr
    )

    step = 0
    done = False
    stats_rows: List[List[float]] = []
    actor_w: List[np.ndarray] = []
    critic_w_val: List[np.ndarray] = []
    critic_w_var: List[np.ndarray] = []
    place_spikes: List[np.ndarray] = []

    while not done and step < args.max_steps:
        spikes = encoder(obs_t)
        action_code, act_spikes = actor(spikes, noise_scale=current_ne)
        real_action = 1 - action_code if inverted else action_code

        next_obs, reward, terminated, truncated, _ = env.step(real_action)
        done = terminated or truncated
        next_obs_t = torch.tensor(next_obs, dtype=torch.float32, device=device)

        v_curr, var_curr = critic.forward(spikes)
        if not done:
            next_spikes = encoder(next_obs_t)
            v_next, _ = critic.forward(next_spikes)
        else:
            v_next = torch.tensor([0.0], device=device)

        target = reward + model.GAMMA * v_next
        td_error = target - v_curr
        td_error_val = float(td_error.item())
        var_val = float(var_curr.item())

        critic_lr = args.critic_base_lr
        critic.update(spikes, td_error, var_curr, lr=critic_lr)

        mod_state.actor_lr_state *= (1.0 - args.actor_lr_decay)
        mod_state.actor_lr_state += args.actor_lr_boost * current_ach
        mod_state.actor_lr_state = min(
            max(mod_state.actor_lr_state, args.actor_lr_min), args.actor_lr_max
        )
        actor_lr = mod_state.actor_lr_state

        actor.update(td_error_val, act_spikes, current_lr=actor_lr)

        current_sigma = torch.sqrt(var_curr.detach()).item()
        current_sigma = max(current_sigma, model.SURPRISE_EPS)

        if not mod_state.exp_trace_inited:
            mod_state.exp_fast = current_sigma
            mod_state.exp_slow = current_sigma
            mod_state.exp_trace_inited = True
        else:
            mod_state.exp_fast = (
                (1.0 - model.EXP_FAST_ALPHA) * mod_state.exp_fast
                + model.EXP_FAST_ALPHA * current_sigma
            )
            mod_state.exp_slow = (
                (1.0 - model.EXP_SLOW_ALPHA) * mod_state.exp_slow
                + model.EXP_SLOW_ALPHA * current_sigma
            )
        exp_novelty = max(0.0, mod_state.exp_fast)
        mod_state.avg_expected = model.EXP_SURPRISE_DECAY * mod_state.avg_expected + exp_novelty

        td_signal = abs(td_error_val) / (current_sigma ** model.SURPRISE_VARIANCE_WEIGHT)
        td_signal = float(min(max(td_signal, 0.0), model.TD_SIGNAL_CLIP))

        if not mod_state.td_trace_inited:
            mod_state.td_fast = td_signal
            mod_state.td_slow = td_signal
            mod_state.td_trace_inited = True
        else:
            mod_state.td_fast = (
                (1.0 - args.td_fast_alpha) * mod_state.td_fast + args.td_fast_alpha * td_signal
            )
            mod_state.td_slow = (
                (1.0 - args.td_slow_alpha) * mod_state.td_slow + args.td_slow_alpha * td_signal
            )

        td_novelty = max(0.0, mod_state.td_fast - mod_state.td_slow - model.TD_NOVELTY_MARGIN)
        mod_state.avg_unexpected = model.UNEXP_SURPRISE_DECAY * mod_state.avg_unexpected + td_novelty

        current_ne = model.logistic_drive(
            args.ne_max, model.NE_K, model.NE_CENTER, mod_state.avg_unexpected, args.base_noise
        )
        current_ne = min(current_ne, 5.0)
        current_ach = model.logistic_drive(
            args.ach_max, model.ACH_K, model.ACH_CENTER, mod_state.avg_expected, args.base_lr
        )

        stats_rows.append(
            [
                float(step),
                float(obs[0]),
                float(obs[1]),
                float(obs[2]),
                float(obs[3]),
                float(action_code),
                float(real_action),
                float(reward),
                float(done),
                float(v_curr.item()),
                var_val,
                td_error_val,
                td_signal,
                mod_state.td_fast,
                mod_state.td_slow,
                td_novelty,
                mod_state.avg_unexpected,
                mod_state.exp_fast,
                mod_state.exp_slow,
                exp_novelty,
                mod_state.avg_expected,
                float(current_ne),
                float(current_ach),
                float(actor_lr),
                float(critic_lr),
            ]
        )

        actor_w.append(actor.w.detach().cpu().numpy().astype(np.float32).copy())
        critic_w_val.append(critic.w_val.detach().cpu().numpy().astype(np.float32).copy())
        critic_w_var.append(critic.w_var.detach().cpu().numpy().astype(np.float32).copy())
        if args.include_place_cells:
            place_spikes.append(spikes.detach().cpu().numpy().astype(np.uint8).copy())

        obs = next_obs
        obs_t = next_obs_t
        step += 1

    stats = np.array(stats_rows, dtype=np.float32)
    trace: Dict[str, np.ndarray] = {
        "actor_w": np.stack(actor_w, axis=0),
        "critic_w_val": np.stack(critic_w_val, axis=0),
        "critic_w_var": np.stack(critic_w_var, axis=0),
        "stats": stats,
        "stat_names": np.array(
            [
                "step",
                "obs_0",
                "obs_1",
                "obs_2",
                "obs_3",
                "action_code",
                "real_action",
                "reward",
                "done",
                "value",
                "variance",
                "td_error",
                "td_signal",
                "td_fast",
                "td_slow",
                "td_novelty",
                "avg_unexpected",
                "exp_fast",
                "exp_slow",
                "exp_novelty",
                "avg_expected",
                "ne",
                "ach",
                "actor_lr",
                "critic_lr",
            ],
            dtype=str,
        ),
    }
    if args.include_place_cells:
        trace["place_spikes"] = np.stack(place_spikes, axis=0)
    meta = {
        "episode": ep_idx,
        "seed": args.seed,
        "include_place_cells": args.include_place_cells,
        "switch_episode": args.switch_episode,
    }
    trace["meta_json"] = np.array([json.dumps(meta)], dtype=str)
    return trace


def write_stats_csv(path: str, stats: np.ndarray, stat_names: List[str], print_stats: bool) -> None:
    with open(path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(stat_names)
        if print_stats:
            print(",".join(stat_names))
        for row in stats:
            writer.writerow([f"{v:.6f}" for v in row])
            if print_stats:
                print(",".join([f"{v:.6f}" for v in row]))


def render_edge_graph(
    ax: plt.Axes, actor_w_step: np.ndarray, place_spikes_step: np.ndarray | None
) -> None:
    n_cells = actor_w_step.shape[0]
    y = np.linspace(0.0, 1.0, n_cells, dtype=np.float32)
    x_left = np.zeros_like(y)
    action_y = np.array([0.25, 0.75], dtype=np.float32)

    max_abs = float(np.max(np.abs(actor_w_step)))
    if max_abs < 1e-6:
        max_abs = 1.0

    segments0 = np.zeros((n_cells, 2, 2), dtype=np.float32)
    segments0[:, 0, 0] = x_left
    segments0[:, 0, 1] = y
    segments0[:, 1, 0] = 1.0
    segments0[:, 1, 1] = action_y[0]

    segments1 = np.zeros((n_cells, 2, 2), dtype=np.float32)
    segments1[:, 0, 0] = x_left
    segments1[:, 0, 1] = y
    segments1[:, 1, 0] = 1.0
    segments1[:, 1, 1] = action_y[1]

    norm = plt.Normalize(-max_abs, max_abs)
    lc0 = LineCollection(
        segments0,
        array=actor_w_step[:, 0],
        cmap="coolwarm",
        norm=norm,
        linewidths=0.4,
        alpha=0.6,
    )
    lc1 = LineCollection(
        segments1,
        array=actor_w_step[:, 1],
        cmap="coolwarm",
        norm=norm,
        linewidths=0.4,
        alpha=0.6,
    )
    ax.add_collection(lc0)
    ax.add_collection(lc1)

    if place_spikes_step is None:
        ax.scatter(x_left, y, s=2, c="#111111", alpha=0.3, linewidths=0)
    else:
        sizes = np.where(place_spikes_step > 0.0, 6.0, 2.0)
        colors = np.where(place_spikes_step > 0.0, "#111111", "#aaaaaa")
        ax.scatter(x_left, y, s=sizes, c=colors, alpha=0.7, linewidths=0)

    ax.scatter([1.0, 1.0], action_y, s=60, c=["#1565c0", "#c62828"], zorder=5)
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.05, 1.05)
    ax.axis("off")


def render_trace(
    trace_path: str, out_path: str, include_place_cells: bool, edge_graph_step: int, title: str
) -> None:
    data = np.load(trace_path, allow_pickle=True)
    actor_w = data["actor_w"]
    critic_w_val = data["critic_w_val"]
    critic_w_var = data["critic_w_var"]
    stats = data["stats"]
    stat_names = [str(s) for s in data["stat_names"]]
    place_spikes = data["place_spikes"] if include_place_cells and "place_spikes" in data else None
    show_place_cells = include_place_cells and place_spikes is not None

    steps = actor_w.shape[0]
    weight0 = actor_w[:, :, 0].T
    weight1 = actor_w[:, :, 1].T
    w_val = critic_w_val.T
    w_var = critic_w_var.T

    stat_matrix = stats.T
    stat_norm = normalize_rows(stat_matrix.astype(np.float32))

    rows = 5 + (1 if show_place_cells else 0)
    fig = plt.figure(figsize=(18, 12), dpi=150)
    gs = fig.add_gridspec(nrows=rows, ncols=2, width_ratios=[4.2, 1.2], wspace=0.15)

    ax_stats = fig.add_subplot(gs[0, 0])
    ax_stats.imshow(stat_norm, aspect="auto", interpolation="nearest", cmap="viridis")
    ax_stats.set_ylabel("Stats")
    ax_stats.set_yticks(range(len(stat_names)))
    ax_stats.set_yticklabels(stat_names, fontsize=7)
    ax_stats.set_xticks([])
    ax_stats.set_title("Per-step stats (normalized)")

    max_abs_actor = float(np.max(np.abs(actor_w)))
    if max_abs_actor < 1e-6:
        max_abs_actor = 1.0

    ax_w0 = fig.add_subplot(gs[1, 0])
    im0 = ax_w0.imshow(
        weight0,
        aspect="auto",
        interpolation="nearest",
        cmap="coolwarm",
        vmin=-max_abs_actor,
        vmax=max_abs_actor,
    )
    ax_w0.set_ylabel("Actor w (a0)")
    ax_w0.set_xticks([])
    fig.colorbar(im0, ax=ax_w0, fraction=0.02, pad=0.01)

    ax_w1 = fig.add_subplot(gs[2, 0])
    im1 = ax_w1.imshow(
        weight1,
        aspect="auto",
        interpolation="nearest",
        cmap="coolwarm",
        vmin=-max_abs_actor,
        vmax=max_abs_actor,
    )
    ax_w1.set_ylabel("Actor w (a1)")
    ax_w1.set_xticks([])
    fig.colorbar(im1, ax=ax_w1, fraction=0.02, pad=0.01)

    ax_cv = fig.add_subplot(gs[3, 0])
    im_cv = ax_cv.imshow(
        w_val,
        aspect="auto",
        interpolation="nearest",
        cmap="magma",
    )
    ax_cv.set_ylabel("Critic w_val")
    ax_cv.set_xticks([])
    fig.colorbar(im_cv, ax=ax_cv, fraction=0.02, pad=0.01)

    ax_cv_var = fig.add_subplot(gs[4, 0])
    im_cv_var = ax_cv_var.imshow(
        w_var,
        aspect="auto",
        interpolation="nearest",
        cmap="cividis",
    )
    ax_cv_var.set_ylabel("Critic w_var")
    ax_cv_var.set_xticks([])
    fig.colorbar(im_cv_var, ax=ax_cv_var, fraction=0.02, pad=0.01)

    if show_place_cells:
        ax_pc = fig.add_subplot(gs[5, 0])
        ax_pc.imshow(
            place_spikes.T,
            aspect="auto",
            interpolation="nearest",
            cmap="Greys",
        )
        ax_pc.set_ylabel("Place spikes")
        ax_pc.set_xlabel("Step")
    else:
        ax_cv_var.set_xlabel("Step")

    ax_edge = fig.add_subplot(gs[:, 1])
    if steps == 0:
        ax_edge.text(0.5, 0.5, "No steps captured", ha="center", va="center")
    else:
        if edge_graph_step < 0:
            edge_graph_step = steps - 1
        edge_graph_step = int(max(0, min(edge_graph_step, steps - 1)))
        spikes_step = None
        if show_place_cells and place_spikes is not None:
            spikes_step = place_spikes[edge_graph_step]
        render_edge_graph(ax_edge, actor_w[edge_graph_step], spikes_step)
        ax_edge.set_title(f"Edge map (step {edge_graph_step})")

    fig.suptitle(title or f"Training trace ({steps} steps)", fontsize=14)
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.97])
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=model.SEED)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--include-place-cells", action="store_true")
    parser.add_argument("--print-stats", action="store_true")
    parser.add_argument("--mode", choices=["both", "capture", "render"], default="both")
    parser.add_argument("--trace", type=str, default="")
    parser.add_argument("--out-dir", type=str, default="cartpole/vis_runs")
    parser.add_argument("--edge-graph-step", type=int, default=-1)
    parser.add_argument("--title", type=str, default="")

    parser.add_argument("--base_lr", type=float, default=model.BASE_LR)
    parser.add_argument("--base_noise", type=float, default=model.BASE_NOISE)
    parser.add_argument("--ne_max", type=float, default=model.NE_MAX)
    parser.add_argument("--ach_max", type=float, default=model.ACH_MAX)
    parser.add_argument("--critic_base_lr", type=float, default=model.CRITIC_BASE_LR)
    parser.add_argument("--actor_lr_decay", type=float, default=model.ACTOR_LR_DECAY)
    parser.add_argument("--actor_lr_boost", type=float, default=model.ACTOR_LR_BOOST)
    parser.add_argument("--actor_lr_min", type=float, default=model.ACTOR_LR_MIN)
    parser.add_argument("--actor_lr_max", type=float, default=model.ACTOR_LR_MAX)
    parser.add_argument("--td_fast_alpha", type=float, default=model.TD_FAST_ALPHA)
    parser.add_argument("--td_slow_alpha", type=float, default=model.TD_SLOW_ALPHA)
    parser.add_argument("--switch-episode", type=int, default=5000)
    args = parser.parse_args()

    device = torch.device("cpu")
    model.set_global_seed(args.seed)
    torch.set_grad_enabled(False)

    ensure_dir(args.out_dir)
    trace_paths: List[str] = []

    if args.mode in ("both", "capture"):
        try:
            import gymnasium as gym  # pylint: disable=import-outside-toplevel
        except ImportError:
            import gym  # type: ignore[no-redef] # pylint: disable=import-outside-toplevel

        env = gym.make("CartPole-v1", render_mode="human" if args.render else None)
        encoder = model.PlaceCellEncoder(device)
        actor = model.ModulatedActor(encoder.n_neurons, device)
        critic = model.LocalCritic(encoder.n_neurons, device)

        mod_state = ModulationState(actor_lr_state=args.base_lr)

        for ep in range(args.episodes):
            trace = capture_episode(
                env,
                encoder,
                actor,
                critic,
                args,
                ep,
                mod_state,
                device,
            )
            trace_path = os.path.join(args.out_dir, f"trace_ep{ep}.npz")
            np.savez_compressed(trace_path, **trace)
            trace_paths.append(trace_path)

            stats = trace["stats"]
            stat_names = [str(s) for s in trace["stat_names"]]
            csv_path = os.path.join(args.out_dir, f"stats_ep{ep}.csv")
            write_stats_csv(csv_path, stats, stat_names, args.print_stats)

        env.close()

    if args.mode in ("both", "render"):
        if args.trace:
            trace_paths = [args.trace]
        for trace_path in trace_paths:
            base = os.path.splitext(os.path.basename(trace_path))[0]
            out_png = os.path.join(args.out_dir, f"{base}_viz.png")
            render_trace(
                trace_path,
                out_png,
                args.include_place_cells,
                args.edge_graph_step,
                args.title,
            )


if __name__ == "__main__":
    main()
