#!/usr/bin/env python3
"""
Switch CartPole with NE (Noise) AND ACh (Learning Rate) + SNN Critic + SNN World Model
with COUNTERFACTUAL SURPRISE (action-remap detector) — FIXED + MORE SENSITIVE + STABLE.

Key fixes vs your prior counterfactual script:
1) TRUE counterfactuals for an SNN world model:
   - The world model has recurrent state (mem/syn). To compare action0 vs action1 fairly,
     we snapshot and restore the SNN internal state so both NLLs start from the same latent.
   - We also advance the world-model state exactly ONCE per env step (with the intended action),
     so time stays aligned.

2) Surprise is now a FAST–SLOW novelty filter on the counterfactual NLL gap:
     gap = max(0, NLL_intended - min(NLL0, NLL1))
     gap_fast tracks quickly, gap_slow tracks slowly
     novelty = ReLU(gap_fast - gap_slow)
   This makes surprise spike at regime changes (like action inversion) without
   normalizing the spike away inside the same episode.

3) Variance control:
   - Gaussian NLL models can “cheat” by inflating std so everything looks plausible.
   - We clamp std and add a small std penalty.

Run:
  python counterfactual_surprise_fixed.py --episodes 4000 --seed 6777
"""

import argparse
import math
import os
import random
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

try:
    import gymnasium as gym
except ImportError:
    import gym


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DT = 0.02
RHO_PC = 50.0
TAU_M = 0.02
ACTOR_THETA = 2.0
GAMMA = 0.99

# --- Exploration noise (NE) range ---
NOISE_MIN = 1.0
NOISE_MAX = 5.0

# --- Actor learning rate (ACh) range ---
ACTOR_LR_MIN = 5e-4
ACTOR_LR_MAX = 5e-3

# --- Neuromodulation logistic params ---
NE_K = 2.0
NE_CENTER = 1.0
ACH_K = 2.0
ACH_CENTER = 1.0

# Surprise EMA (for smoother modulation)
# NOTE: 0.98 is very sluggish; 0.95 responds more strongly at the switch.
SURPRISE_DECAY = 0.95

# --- Counterfactual novelty filter (stable + sensitive) ---
# fast reacts within ~1/alpha steps; slow reacts within ~1/alpha steps (much slower)
GAP_FAST_ALPHA = 0.05
GAP_SLOW_ALPHA = 0.001
SURPRISE_GAIN = 1.0
SURPRISE_CLIP = 10.0

# --- Critic dynamics params ---
CRITIC_TAU_M = 0.02
CRITIC_THRESH = 1.0

# Critic optimizer lr (base); effective step size comes from loss scaling
CRITIC_BASE_LR = 1e-3
CRITIC_ACH_MIN_SCALE = 0.1
CRITIC_ACH_MAX_SCALE = 5.0

# World model optimizer lr
WORLD_MODEL_BASE_LR = 1e-3

# World model variance control
WM_STD_MAX = 2.0
WM_STD_PENALTY = 1e-3

# Switch point (episode index)
SWITCH_EP = 2000

# Seed
SEED = 6777


# ---------------------------------------------------------------------------
# Seeding
# ---------------------------------------------------------------------------
def set_global_seed(seed: int | None):
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Helpers for gym/gymnasium compat
# ---------------------------------------------------------------------------
def env_reset(env, seed: int | None = None):
    if seed is None:
        out = env.reset()
    else:
        try:
            out = env.reset(seed=seed)
        except TypeError:
            try:
                env.seed(seed)
                out = env.reset()
            except Exception:
                out = env.reset()

    # gymnasium: (obs, info), gym: obs
    if isinstance(out, tuple) and len(out) >= 1:
        return out[0]
    return out


def env_step(env, action: int):
    out = env.step(action)
    # gymnasium: (obs, reward, terminated, truncated, info)
    if isinstance(out, tuple) and len(out) == 5:
        obs, reward, terminated, truncated, info = out
        done = bool(terminated or truncated)
        return obs, float(reward), done, info
    # gym: (obs, reward, done, info)
    obs, reward, done, info = out
    return obs, float(reward), bool(done), info


# ---------------------------------------------------------------------------
# 1. Place Cell Encoder
# ---------------------------------------------------------------------------
class PlaceCellEncoder(nn.Module):
    def __init__(self, device: torch.device):
        super().__init__()
        self.device = device
        m = torch.linspace(-2.5, 2.5, 6, device=device)
        n = torch.linspace(-2.0, 2.0, 6, device=device)
        p = torch.linspace(-0.25, 0.25, 8, device=device)
        q = torch.linspace(-2.0, 2.0, 8, device=device)

        mesh = torch.meshgrid(m, n, p, q, indexing="ij")
        centers = torch.stack([x.flatten() for x in mesh], dim=1)
        self.register_buffer("centers", centers)

        s1 = (m[1] - m[0]) / 1.5
        s2 = (n[1] - n[0]) / 1.5
        s3 = (p[1] - p[0]) / 1.5
        s4 = (q[1] - q[0]) / 1.5
        sigmas = torch.tensor([s1, s2, s3, s4], device=device).unsqueeze(0)
        self.register_buffer("sigmas", sigmas)

        self.n_neurons = centers.shape[0]

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        diff = (state.unsqueeze(0) - self.centers) ** 2
        exponent = -torch.sum(diff / (2 * self.sigmas ** 2), dim=1)
        rates = RHO_PC * torch.exp(exponent)
        probs = 1.0 - torch.exp(-rates * DT)
        spikes = torch.bernoulli(torch.clamp(probs, 0.0, 1.0))
        return spikes


# ---------------------------------------------------------------------------
# 2. Surrogate spike
# ---------------------------------------------------------------------------
class SurrogateHeaviside(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return (input > 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        (input,) = ctx.saved_tensors
        grad_input = grad_output / (1.0 + torch.abs(input) * 5.0).pow(2)
        return grad_input


def surrogate_spike(x: torch.Tensor) -> torch.Tensor:
    return SurrogateHeaviside.apply(x)


# ---------------------------------------------------------------------------
# 3. SNN Critic
# ---------------------------------------------------------------------------
class SpikingCritic(nn.Module):
    def __init__(self, n_input: int, n_hidden: int = 256):
        super().__init__()
        self.fc1 = nn.Linear(n_input, n_hidden)
        self.fc_val = nn.Linear(n_hidden, 1)
        self.fc_var = nn.Linear(n_hidden, 1)

        self.decay_mem = math.exp(-DT / CRITIC_TAU_M)
        self.decay_syn = math.exp(-DT / CRITIC_TAU_M)

        self.register_buffer("mem_hidden", torch.zeros(n_hidden))
        self.register_buffer("syn_hidden", torch.zeros(n_hidden))

    def reset_state(self):
        self.mem_hidden.zero_()
        self.syn_hidden.zero_()

    def forward(self, input_spikes: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        self.mem_hidden = self.mem_hidden.detach()
        self.syn_hidden = self.syn_hidden.detach()

        current_in = self.fc1(input_spikes)
        self.mem_hidden = self.mem_hidden * self.decay_mem + current_in

        spike_in = self.mem_hidden - CRITIC_THRESH
        h_spikes = surrogate_spike(spike_in)

        self.mem_hidden = self.mem_hidden * (1.0 - h_spikes.detach())
        self.syn_hidden = self.syn_hidden * self.decay_syn + h_spikes

        val = self.fc_val(self.syn_hidden)
        var = F.softplus(self.fc_var(self.syn_hidden)) + 1e-4
        return val, var


# ---------------------------------------------------------------------------
# 4. SNN World Model (probabilistic): predicts mean and std for next_state(4) + reward(1)
# ---------------------------------------------------------------------------
class SpikingWorldModel(nn.Module):
    def __init__(self, n_input: int, n_hidden: int = 256, out_dim: int = 5):
        super().__init__()
        self.out_dim = out_dim

        self.fc1 = nn.Linear(n_input, n_hidden)
        self.fc_out = nn.Linear(n_hidden, out_dim * 2)  # mean + log_std

        self.decay_mem = math.exp(-DT / TAU_M)
        self.decay_syn = math.exp(-DT / TAU_M)

        self.register_buffer("mem_hidden", torch.zeros(n_hidden))
        self.register_buffer("syn_hidden", torch.zeros(n_hidden))

    def reset_state(self):
        self.mem_hidden.zero_()
        self.syn_hidden.zero_()

    def forward(self, input_spikes: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        self.mem_hidden = self.mem_hidden.detach()
        self.syn_hidden = self.syn_hidden.detach()

        current_in = self.fc1(input_spikes)
        self.mem_hidden = self.mem_hidden * self.decay_mem + current_in

        spike_in = self.mem_hidden - 1.0
        h_spikes = surrogate_spike(spike_in)

        self.mem_hidden = self.mem_hidden * (1.0 - h_spikes.detach())
        self.syn_hidden = self.syn_hidden * self.decay_syn + h_spikes

        out = self.fc_out(self.syn_hidden)
        mu, log_std = torch.split(out, self.out_dim, dim=-1)

        std = F.softplus(log_std) + 1e-3
        std = torch.clamp(std, 1e-3, WM_STD_MAX)  # prevent “variance inflation” cheat

        return mu, std


def gaussian_nll(target: torch.Tensor, mu: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    var = std * std
    nll = 0.5 * ((target - mu) ** 2 / var + torch.log(var) + math.log(2.0 * math.pi))
    return nll.sum()


# ---------------------------------------------------------------------------
# 5. Universal Actor (Modulated)
# ---------------------------------------------------------------------------
class ModulatedActor(nn.Module):
    def __init__(self, n_input: int, device: torch.device):
        super().__init__()
        self.device = device
        self.w = torch.empty(n_input, 2, device=device).normal_(0.0, 0.05)
        self.z_eps = torch.zeros(n_input, device=device)
        self.decay_eps = math.exp(-DT / TAU_M)

    def reset_state(self):
        self.z_eps.zero_()

    def forward(self, input_spikes: torch.Tensor, noise_scale: float) -> Tuple[int, torch.Tensor]:
        self.z_eps = self.z_eps * self.decay_eps + input_spikes
        v_det = torch.matmul(self.z_eps, self.w)

        noise = torch.randn_like(v_det) * noise_scale
        v_mem = v_det + noise

        rho = 100.0 * torch.exp((v_mem - ACTOR_THETA) / 2.0)
        probs = 1.0 - torch.exp(-rho * DT)
        spikes = torch.bernoulli(torch.clamp(probs, 0.0, 1.0))

        s = spikes.detach().cpu().numpy()
        if s[0] == 1 and s[1] == 0:
            action = 0
        elif s[0] == 0 and s[1] == 1:
            action = 1
        else:
            action = int(torch.argmax(v_mem).item())

        return action, spikes

    def update(self, td_error: float, output_spikes: torch.Tensor, current_lr: float):
        eligibility = torch.outer(self.z_eps, output_spikes)
        dw = current_lr * td_error * eligibility
        self.w += dw
        self.w.clamp_(-10.0, 10.0)


# ---------------------------------------------------------------------------
# 6. Neuromodulation helpers
# ---------------------------------------------------------------------------
def logistic01(x: float, k: float, center: float) -> float:
    return 1.0 / (1.0 + math.exp(-k * (x - center)))


def compute_modulators(avg_surprise: float) -> Tuple[float, float, float, float]:
    """
    Returns:
      noise_scale (NE-driven),
      actor_lr (ACh-driven),
      critic_scale (ACh-driven),
      ach_level (0..1)
    """
    ne_level = logistic01(avg_surprise, NE_K, NE_CENTER)
    ach_level = logistic01(avg_surprise, ACH_K, ACH_CENTER)

    noise_scale = NOISE_MIN + ne_level * (NOISE_MAX - NOISE_MIN)
    actor_lr = ACTOR_LR_MIN + ach_level * (ACTOR_LR_MAX - ACTOR_LR_MIN)

    critic_scale = CRITIC_ACH_MIN_SCALE + ach_level * (CRITIC_ACH_MAX_SCALE - CRITIC_ACH_MIN_SCALE)
    critic_scale = float(max(CRITIC_ACH_MIN_SCALE, min(CRITIC_ACH_MAX_SCALE, critic_scale)))

    return float(noise_scale), float(actor_lr), float(critic_scale), float(ach_level)


# ---------------------------------------------------------------------------
# 7. Training Loop
# ---------------------------------------------------------------------------
def train(episodes: int = 4000, render: bool = False, seed: int | None = None):
    device = torch.device("cpu")
    set_global_seed(seed)

    env = gym.make("CartPole-v1", render_mode="human" if render else None)

    # Seed env spaces where supported
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
        try:
            env.observation_space.seed(seed)
        except Exception:
            pass

    encoder = PlaceCellEncoder(device)
    actor = ModulatedActor(encoder.n_neurons, device)
    critic = SpikingCritic(encoder.n_neurons).to(device)

    # World model input = place-cell spikes + intended action(2)
    world_model = SpikingWorldModel(encoder.n_neurons + 2, n_hidden=256, out_dim=5).to(device)

    critic_optim = optim.Adam(critic.parameters(), lr=CRITIC_BASE_LR)
    world_optim = optim.Adam(world_model.parameters(), lr=WORLD_MODEL_BASE_LR)

    print("Start Switch CartPole (NE+ACh + SNN Critic + Counterfactual WM Surprise FIXED). "
          f"Episodes: {episodes}")

    avg_surprise = 0.0

    # fast/slow novelty filter state
    gap_fast = 0.0
    gap_slow = 0.0
    gap_inited = False

    reward_history = []
    surprise_history = []
    td_history = []

    gap_history = []       # episode avg raw gap
    novelty_history = []   # episode avg novelty (fast-slow)
    p_int_history = []     # episode avg p(intended) under WM counterfactual softmax

    for ep in range(episodes):
        obs = env_reset(env, seed=(seed + ep) if seed is not None else None)
        obs_t = torch.tensor(obs, dtype=torch.float32, device=device)

        actor.reset_state()
        critic.reset_state()
        world_model.reset_state()

        done = False
        total_reward = 0.0

        inverted = (ep > SWITCH_EP)

        current_ne, current_lr, critic_scale, ach_level = compute_modulators(avg_surprise)

        td_sum = 0.0
        td_count = 0

        gap_sum = 0.0
        nov_sum = 0.0
        p_int_sum = 0.0
        step_count = 0

        while not done:
            # Encode current state to spikes (ONE sample per step; reuse for counterfactuals)
            s_spikes = encoder(obs_t)

            # Actor chooses intended action
            action_code, act_spikes = actor(s_spikes, noise_scale=current_ne)

            # Environment executes (maybe inverted)
            real_action = 1 - action_code if inverted else action_code
            next_obs, reward, done, _info = env_step(env, real_action)
            next_obs_t = torch.tensor(next_obs, dtype=torch.float32, device=device)

            # ------------------------------------------------------------
            # WORLD MODEL: TRUE COUNTERFACTUALS (STATE-SAFE) + ONE ADVANCE
            # ------------------------------------------------------------
            target = torch.cat([next_obs_t, torch.tensor([reward], device=device)], dim=0)

            # Snapshot world-model latent state ONCE per env step
            wm_mem_b = world_model.mem_hidden.clone()
            wm_syn_b = world_model.syn_hidden.clone()

            a0 = torch.tensor([1.0, 0.0], device=device)
            a1 = torch.tensor([0.0, 1.0], device=device)

            def nll_for_action(a_onehot: torch.Tensor) -> torch.Tensor:
                # restore so both action evaluations start from identical latent
                world_model.mem_hidden.copy_(wm_mem_b)
                world_model.syn_hidden.copy_(wm_syn_b)
                wm_in = torch.cat([s_spikes, a_onehot], dim=0)
                mu, std = world_model(wm_in)
                return gaussian_nll(target, mu, std)

            nll0 = nll_for_action(a0)
            nll1 = nll_for_action(a1)

            n0 = float(nll0.detach().item())
            n1 = float(nll1.detach().item())

            n_min = min(n0, n1)
            n_int = n0 if action_code == 0 else n1
            gap = max(0.0, n_int - n_min)

            # Soft evidence: p(intended) under a Boltzmann over negative NLL
            # (helps interpret whether WM thinks the other action explains the outcome)
            T = 1.0
            logits = torch.tensor([-n0 / T, -n1 / T], device=device)
            p = torch.softmax(logits, dim=0)
            p_intended = float(p[action_code].item())

            # FAST–SLOW novelty (stable spike at regime changes)
            if not gap_inited:
                gap_fast = gap
                gap_slow = gap
                gap_inited = True
            else:
                gap_fast = (1.0 - GAP_FAST_ALPHA) * gap_fast + GAP_FAST_ALPHA * gap
                gap_slow = (1.0 - GAP_SLOW_ALPHA) * gap_slow + GAP_SLOW_ALPHA * gap

            novelty = max(0.0, gap_fast - gap_slow)

            step_surprise = min(SURPRISE_CLIP, SURPRISE_GAIN * novelty)
            avg_surprise = SURPRISE_DECAY * avg_surprise + (1.0 - SURPRISE_DECAY) * step_surprise

            # Train + advance world model exactly ONCE with intended action
            world_model.mem_hidden.copy_(wm_mem_b)
            world_model.syn_hidden.copy_(wm_syn_b)
            a_int = a0 if action_code == 0 else a1
            wm_in_int = torch.cat([s_spikes, a_int], dim=0)
            mu_int, std_int = world_model(wm_in_int)
            nll_intended = gaussian_nll(target, mu_int, std_int)

            loss_wm = nll_intended + WM_STD_PENALTY * std_int.mean()
            world_optim.zero_grad()
            loss_wm.backward()
            world_optim.step()

            # ------------------------------------------------------------
            # CRITIC UPDATE
            # ------------------------------------------------------------
            v_curr, var_curr = critic(s_spikes)

            with torch.no_grad():
                if not done:
                    next_spikes = encoder(next_obs_t)

                    mem_b = critic.mem_hidden.clone()
                    syn_b = critic.syn_hidden.clone()

                    v_next, _ = critic(next_spikes)

                    critic.mem_hidden.copy_(mem_b)
                    critic.syn_hidden.copy_(syn_b)
                else:
                    v_next = torch.tensor([0.0], device=device)

            td_target = reward + GAMMA * v_next
            td_error = td_target - v_curr
            td_error_val = float(td_error.item())

            td_sum += abs(td_error_val)
            td_count += 1

            val_loss = td_error.pow(2)
            target_var = td_error.detach().pow(2)
            var_loss = (var_curr - target_var).pow(2)

            scaled_loss = (val_loss + var_loss) * critic_scale

            critic_optim.zero_grad()
            scaled_loss.backward()
            critic_optim.step()

            # ------------------------------------------------------------
            # ACTOR UPDATE
            # ------------------------------------------------------------
            actor.update(td_error_val, act_spikes, current_lr=current_lr)

            # Update modulators for next step
            current_ne, current_lr, critic_scale, ach_level = compute_modulators(avg_surprise)

            # Step forward
            obs_t = next_obs_t
            total_reward += reward

            # Episode stats
            gap_sum += gap
            nov_sum += novelty
            p_int_sum += p_intended
            step_count += 1

        reward_history.append(total_reward)
        surprise_history.append(avg_surprise)
        td_history.append((td_sum / td_count) if td_count > 0 else 0.0)

        ep_gap = (gap_sum / step_count) if step_count > 0 else 0.0
        ep_nov = (nov_sum / step_count) if step_count > 0 else 0.0
        ep_pint = (p_int_sum / step_count) if step_count > 0 else 0.5
        gap_history.append(ep_gap)
        novelty_history.append(ep_nov)
        p_int_history.append(ep_pint)

        avg_r = float(np.mean(reward_history[-20:])) if len(reward_history) > 0 else 0.0

        if ep % 100 == 0:
            status = "NORMAL" if not inverted else "INVERTED"
            print(
                f"Ep {ep:4d} | {status} | R: {total_reward:3.0f} | Avg: {avg_r:5.1f} | "
                f"NE: {current_ne:.2f} | AChLR: {current_lr:.6f} | AChLvl: {ach_level:.3f} | "
                f"CriticScale: {critic_scale:.3f} | Surprise: {avg_surprise:.2f} | "
                f"EpGap: {ep_gap:.3f} | EpNov: {ep_nov:.3f} | EpPint: {ep_pint:.3f}"
            )

    # -----------------------------------------------------------------------
    # Plot results
    # -----------------------------------------------------------------------
    fig, ax1 = plt.subplots(figsize=(10, 6))

    ax1.plot(range(len(reward_history)), reward_history, linewidth=1, alpha=0.6, label="Episode Reward")
    window_size = 20
    if len(reward_history) >= window_size:
        rolling_avg = np.convolve(reward_history, np.ones(window_size) / window_size, mode="valid")
        ax1.plot(range(window_size - 1, len(reward_history)), rolling_avg, linewidth=2, label=f"{window_size}-ep Avg")

    ax1.axvline(x=SWITCH_EP, color="r", linestyle="--", label="Switch Point")
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Episode Reward")
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(range(len(surprise_history)), surprise_history, linewidth=1.5, alpha=0.9, label="Avg Surprise")
    ax2.plot(range(len(td_history)), td_history, linewidth=1.2, alpha=0.9, label="Mean |TD|")
    ax2.plot(range(len(gap_history)), gap_history, linewidth=1.2, alpha=0.9, label="Ep Counterfactual Gap")
    ax2.plot(range(len(novelty_history)), novelty_history, linewidth=1.2, alpha=0.9, label="Ep Novelty (fast-slow)")
    ax2.plot(range(len(p_int_history)), p_int_history, linewidth=1.2, alpha=0.9, label="Ep P(intended)")
    ax2.set_ylabel("Surprise / TD / Gap / Novelty / P(intended)")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    fig.tight_layout()
    plt.title("Switch CartPole - NE+ACh + SNN Critic + Counterfactual WM Surprise (Fixed)")

    out_dir = f"runs/{seed}_counterfactual_surprise_fixed" if seed is not None else "runs/noseed_counterfactual_surprise_fixed"
    os.makedirs(out_dir, exist_ok=True)

    png_path = os.path.join(out_dir, f"{seed}_counterfactual_surprise_fixed.png")
    fig.savefig(png_path, dpi=150, bbox_inches="tight")

    csv_path = os.path.join(out_dir, f"{seed}_counterfactual_surprise_fixed.csv")
    with open(csv_path, "w") as fh:
        fh.write("episode,reward,avg_surprise,mean_abs_td,ep_gap,ep_novelty,ep_p_intended\n")
        for i in range(len(reward_history)):
            fh.write(f"{i},{reward_history[i]},{surprise_history[i]},{td_history[i]},{gap_history[i]},{novelty_history[i]},{p_int_history[i]}\n")

    print(f"\nPlot saved as '{png_path}' and CSV saved as '{csv_path}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=4000)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed (overrides top-level SEED)")
    args = parser.parse_args()

    train(args.episodes, args.render, seed=args.seed)
