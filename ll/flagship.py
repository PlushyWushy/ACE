#!/usr/bin/env python3
"""
Switch Lunar Lander with decoupled uncertainty following Yu & Dayan's conjecture:
- NE (Noradrenaline) driven by *unexpected* uncertainty (fast-slow TD novelty)
- ACh (Acetylcholine) driven by *expected* uncertainty (critic's variance estimate)

Critic uses a local TD-LTP style update with loss as a multiplicative factor.
Actor LR decays each step and is boosted by ACh (not set equal to ACh).
"""

#TODO: Run with low ACh center with switch; run with high ACh center without switch.
import argparse
import math
import numpy as np
import random
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
import matplotlib.pyplot as plt

try:
    import gymnasium as gym
except ImportError:
    import gym

# --------------------------------------------------------------------------- 
# Constants
# --------------------------------------------------------------------------- 

# --------------------------------------------------------------------------- 

DT = 0.02
RHO_PC = 50.0
TAU_M = 0.02
ACTOR_THETA = 2.0
GAMMA = 0.9995 # From tau_gamma = 2000ms: exp(-1/2000) approx 0.9995

# --- NEUROMODULATION PARAMETERS ---
BASE_LR = 6.25e-5 # From Table 1 (Actor LR)
BASE_NOISE = 1

# NE logistic mapping params (for unexpected uncertainty / novelty)
NE_MAX = 3
NE_K = 0.2
NE_CENTER = 15

# ACh logistic mapping params (for expected uncertainty / variance)
ACH_MAX = 1
ACH_K = 10 
ACH_CENTER = 5

# Surprise EMA
EXP_SURPRISE_DECAY = 0.01  
UNEXP_SURPRISE_DECAY = 0.8 

# If =1.0 -> divide by sigma (z-ish). If =0.0 -> ignore variance term.
SURPRISE_VARIANCE_WEIGHT = 0
SURPRISE_EPS = 1e-3

# --------------------------------------------------------------------------- 
# Fast–slow TD novelty (habituation / baseline subtraction) for UNEXPECTED uncertainty
# --------------------------------------------------------------------------- 
# TD signal is |TD| / (sigma**SURPRISE_VARIANCE_WEIGHT), then clipped.
TD_SIGNAL_CLIP = 20.0

# Fast trace reacts quickly; slow trace is "what I'm used to".
TD_FAST_ALPHA = 0.097663     # ~20-step timescale
TD_SLOW_ALPHA = 0.003642    # ~1000-step timescale

# Faster traces for expected uncertainty (ACh)
EXP_FAST_ALPHA = 1    # 
EXP_SLOW_ALPHA = 0.1

# Extra deadzone after baseline subtraction (helps suppress tiny random novelty).
TD_NOVELTY_MARGIN = 0.0

# --------------------------------------------------------------------------- 
# Hyperparameters Alignement (Chung & Kozma 2020)
# --------------------------------------------------------------------------- 
CRITIC_BASE_LR = 1.25e-4 # From Table 1
ADAM_BETA1 = 0.995       # From Table 1
ADAM_BETA2 = 0.99995     # From Table 1
REWARD_SCALE = 0.012     # From Table 1

VAR_DECAY = 0
ACTOR_LR_DECAY = 0.1  
ACTOR_LR_BOOST = 0.1
ACTOR_LR_MIN = 1e-6
ACTOR_LR_MAX = 0.1

# Editable global seed (set to None for non-deterministic runs)
SEED = 1234


def set_global_seed(seed: int | None):
    """Set seeds for python, numpy and torch for reproducibility."""
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


def logistic_drive(max_val: float, k: float, center: float, signal: float, base: float) -> float:
    # numerically-stable logistic: clip the exponent to avoid overflow
    z = k * (signal - center)
    # clip thresholds chosen to avoid math.exp overflow on most platforms
    if z >= 700:
        return max_val + base
    if z <= -700:
        return base
    return max_val / (1.0 + math.exp(-z)) + base


# --------------------------------------------------------------------------- 
# 1. Place Cell Encoder
# --------------------------------------------------------------------------- 
class PlaceCellEncoder(nn.Module):
    def __init__(self, device: torch.device):
        super().__init__()
        self.device = device
        
        # Lunar Lander specific ranges - Resolution 4 for continuous
        m0 = torch.linspace(-1.0, 1.0, 4, device=device) # x
        m1 = torch.linspace(-0.3, 1.2, 4, device=device) # y
        m2 = torch.linspace(-2.0, 2.0, 4, device=device) # vx
        m3 = torch.linspace(-2.0, 2.0, 4, device=device) # vy
        m4 = torch.linspace(-1.0, 1.0, 4, device=device) # angle
        m5 = torch.linspace(-1.0, 1.0, 4, device=device) # a_vel
        m6 = torch.linspace(0.0, 1.0, 2, device=device)  # touch 1
        m7 = torch.linspace(0.0, 1.0, 2, device=device)  # touch 2

        # Meshgrid in 8D: 4^6 * 2^2 = 16384 neurons
        mesh = torch.meshgrid(m0, m1, m2, m3, m4, m5, m6, m7, indexing="ij")
        centers = torch.stack([x.flatten() for x in mesh], dim=1)
        self.register_buffer("centers", centers)

        # Sigmas (width of place cells)
        s0 = (m0[1] - m0[0]) / 1.5
        s1 = (m1[1] - m1[0]) / 1.5
        s2 = (m2[1] - m2[0]) / 1.5
        s3 = (m3[1] - m3[0]) / 1.5
        s4 = (m4[1] - m4[0]) / 1.5
        s5 = (m5[1] - m5[0]) / 1.5
        s6 = 0.5 
        s7 = 0.5
        sigmas = torch.tensor([s0, s1, s2, s3, s4, s5, s6, s7], device=device).unsqueeze(0)
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
# 2. Local TD-LTP Critic (value + variance)
# --------------------------------------------------------------------------- 
class LocalCritic:
    def __init__(self, n_input: int, device: torch.device, n_hidden: int = 512):
        self.device = device
        self.n_hidden = n_hidden
        
        # Fixed random projection for the hidden layer
        self.w_h = torch.empty(n_input, n_hidden, device=device).normal_(0.0, 0.05)
        self.b_h = torch.zeros(n_hidden, device=device)
        
        # Output layers (trained manually via Adam)
        self.w_val = torch.zeros(n_hidden, device=device)
        self.b_val = torch.zeros(1, device=device)
        self.w_var = torch.zeros(n_hidden, device=device)
        self.b_var = torch.zeros(1, device=device)
        
        # Adam buffers
        self.m_val = torch.zeros_like(self.w_val)
        self.v_val = torch.zeros_like(self.w_val)
        self.m_b_val = torch.zeros_like(self.b_val)
        self.v_b_val = torch.zeros_like(self.b_val)
        
        self.m_var = torch.zeros_like(self.w_var)
        self.v_var = torch.zeros_like(self.w_var)
        self.m_b_var = torch.zeros_like(self.b_var)
        self.v_b_var = torch.zeros_like(self.b_var)
        
        self.t = 0 # Step counter for Adam bias correction
        
        self.mem_h = torch.zeros(n_hidden, device=device)
        self.decay_mem = math.exp(-DT / TAU_M)
        self.thresh = 1.0

    def reset_state(self):
        self.mem_h.zero_()

    def forward(self, input_spikes: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Hidden layer LIF dynamics
        self.mem_h = self.mem_h * self.decay_mem + torch.matmul(input_spikes, self.w_h) + self.b_h
        hidden_spikes = (self.mem_h >= self.thresh).float()
        self.mem_h = self.mem_h * (1.0 - hidden_spikes) # Reset
        
        val = torch.dot(self.w_val, hidden_spikes) + self.b_val
        var = F.softplus(torch.dot(self.w_var, hidden_spikes) + self.b_var) + 1e-4
        return val, var, hidden_spikes

    def update(self, hidden_spikes: torch.Tensor, td_error: torch.Tensor, var: torch.Tensor, lr: float):
        self.t += 1
        beta1 = ADAM_BETA1
        beta2 = ADAM_BETA2
        eps = 1e-8

        # --- Value Weights Update ---
        # gradient = - td_error * hidden_spikes (for MSE minimization)
        # However, our td_error = target - current, so grad is actually -td_error * hidden_spikes
        # But we want to step in the direction of the gradient: w = w - lr * grad
        # So w = w + lr * td_error * hidden_spikes
        g_w_val = - (td_error.detach() * hidden_spikes)
        g_b_val = - td_error.detach()

        # m = beta1 * m + (1-beta1) * g
        self.m_val = beta1 * self.m_val + (1.0 - beta1) * g_w_val
        self.m_b_val = beta1 * self.m_b_val + (1.0 - beta1) * g_b_val
        # v = beta2 * v + (1-beta2) * g^2
        self.v_val = beta2 * self.v_val + (1.0 - beta2) * g_w_val.pow(2)
        self.v_b_val = beta2 * self.v_b_val + (1.0 - beta2) * g_b_val.pow(2)

        # Bias correction
        m_corr = self.m_val / (1.0 - beta1**self.t)
        m_b_corr = self.m_b_val / (1.0 - beta1**self.t)
        v_corr = self.v_val / (1.0 - beta2**self.t)
        v_b_corr = self.v_b_val / (1.0 - beta2**self.t)

        self.w_val -= lr * m_corr / (torch.sqrt(v_corr) + eps)
        self.b_val -= lr * m_b_corr / (torch.sqrt(v_b_corr) + eps)

        # --- Variance Weights Update ---
        # target_var = td_error^2
        # variance tracks squared error
        target_var = td_error.detach().pow(2)
        err_var = target_var - var.detach()
        
        g_w_var = - (err_var * hidden_spikes)
        g_b_var = - err_var

        self.m_var = beta1 * self.m_var + (1.0 - beta1) * g_w_var
        self.m_b_var = beta1 * self.m_b_var + (1.0 - beta1) * g_b_var
        self.v_var = beta2 * self.v_var + (1.0 - beta2) * g_w_var.pow(2)
        self.v_b_var = beta2 * self.v_b_var + (1.0 - beta2) * g_b_var.pow(2)

        m_v_corr = self.m_var / (1.0 - beta1**self.t)
        m_bv_corr = self.m_b_var / (1.0 - beta1**self.t)
        v_v_corr = self.v_var / (1.0 - beta2**self.t)
        v_bv_corr = self.v_b_var / (1.0 - beta2**self.t)

        self.w_var -= lr * m_v_corr / (torch.sqrt(v_v_corr) + eps)
        self.b_var -= lr * m_bv_corr / (torch.sqrt(v_bv_corr) + eps)

        # Hard clamping for stability
        self.w_val.clamp_(-100.0, 100.0)
        self.b_val.clamp_(-100.0, 100.0)
        self.w_var.clamp_(-100.0, 100.0)
        self.b_var.clamp_(-100.0, 100.0)


# --------------------------------------------------------------------------- 
# 3. Modulated Actor (4 actions for Lunar Lander)
# --------------------------------------------------------------------------- 
class ModulatedActor(nn.Module):
    def __init__(self, n_input: int, device: torch.device):
        super().__init__()
        self.device = device
        # 4 actions: Nothing, Left, Main, Right
        # Initialize with small weights
        self.w = torch.empty(n_input, 4, device=device).normal_(0.0, 0.01)
        self.z_eps = torch.zeros(n_input, device=device)
        # Trace time constant matches their tau_q setup (approx 20-40ms)
        self.decay_eps = math.exp(-DT / TAU_M)
        
        # Gating traces for each action (q_ij in paper)
        self.q_trace = torch.zeros(n_input, 4, device=device)
        self.decay_q = math.exp(-DT / 0.04) # tau_q = 40ms

        # Firing rate tracking for regularization
        self.avg_firing_rate = torch.zeros(4, device=device) # per action
        self.target_rate = 0.05 # 50Hz target (approx)

    def reset_state(self):
        self.z_eps.zero_()
        self.q_trace.zero_()
        self.avg_firing_rate.zero_()

    def forward(self, input_spikes: torch.Tensor, noise_scale: float) -> Tuple[int, torch.Tensor, torch.Tensor]:
        # Update eligibility trace of inputs
        self.z_eps = self.z_eps * self.decay_eps + input_spikes
        
        # Action selection via Softmax on membrane potential (approximating their rate-based policy)
        # We treat w * z_eps as 'logits'
        logits = torch.matmul(self.z_eps, self.w)
        
        # Apply inverse temperature (alpha in paper) - higher means more deterministic
        # We modulate beta with NE: High NE (surprise) -> Lower beta (more random)
        base_beta = 15.0 
        beta = base_beta / (1.0 + noise_scale * 2.0) 
        
        probs = F.softmax(logits * beta, dim=0)
        
        # Sample action
        dist = torch.distributions.Categorical(probs)
        action_idx = dist.sample()
        action = int(action_idx.item())
        
        # Create "action spikes" (one-hot for the chosen action) - A_k in paper
        action_hot = torch.zeros(4, device=self.device)
        action_hot[action] = 1.0
        
        # Update average firing rates (EMA)
        self.avg_firing_rate = 0.999 * self.avg_firing_rate + 0.001 * action_hot
        
        return action, action_hot, probs

    def update(self, td_error: float, action_hot: torch.Tensor, probs: torch.Tensor, current_lr: float):
        # 1. Feedback Modulated Plasticity (Action Gating)
        # Delta w_ij ~ delta * (A_k - s_k) * z_ij
        # signal_k = A_k - s_k
        feedback_signal = action_hot - probs # Shape: [4]
        
        # Update gating trace q_ij
        # q_trace: [n_input, 4]
        # We broadcast feedback_signal to [1, 4] and z_eps to [n_input, 1]
        term1 = torch.outer(self.z_eps, feedback_signal)
        self.q_trace = self.q_trace * self.decay_q + term1
        
        # Main update: TD * q_trace
        delta_w = current_lr * td_error * self.q_trace
        
        # 2. Entropy Regularization
        # Helps prevent converging to "Do Nothing"
        # grad_H ~ - beta_H * sum (s * (1+log s) * (A-s))
        # This is complex to implement fully in 1-step, but adding a term that pushes 
        # probability towards uniform helps.
        # Simplified Exploration Bonus: Push weights of unchosen actions UP slightly if entropy is low?
        # Actually, the paper's formula simplifies to adding a bonus proportional to the eligibility trace.
        entropy_beta = 0.01
        # Inverse entropy force: (log(s_k) + 1)
        log_probs = torch.log(probs + 1e-6)
        entropy_term = -1.0 * (log_probs + 1.0) # Shape [4]
        # Gating for entropy: (A_k - s_k) part is handled by the q_trace logic implicitly if we add to feedback
        # But we can just add a direct exploration pressure:
        # If we just boost the weights of the CHOSEN action inversely to its probability?
        # Let's stick to their rule: w += eta * ( ... + beta_H * entropy_grad )
        # Their approx: eta * c_e * g_k * z_ij
        # It's safer to just use a fixed "Entropy Bonus" added to the reward if entropy is high? 
        # No, let's implement the weight decay they use which helps distribution.
        
        # 3. Weight Decay
        decay_rate = 1e-5
        delta_w -= current_lr * decay_rate * self.w
        
        # 4. Target Firing Rate Regularization (Homeostasis)
        # Penalize if average rate is too high
        # delta ~ - c_t * (rho_avg - rho_target) * z_ij
        homeo_beta = 0.05
        rate_error = self.avg_firing_rate - self.target_rate
        # Create a "homeostatic pressure" vector [4]
        homeo_pressure = -1.0 * rate_error 
        
        # Apply to weights proportional to input activity (z_eps)
        # We outer product z_eps [N] with homeo_pressure [4]
        delta_homeo = current_lr * homeo_beta * torch.outer(self.z_eps, homeo_pressure)
        
        # Combine
        self.w += delta_w + delta_homeo
        
        # Soft clamp to prevent absolute explosion, but let them grow larger than before
        self.w.clamp_(-20.0, 20.0)


def train(args):
    set_global_seed(args.seed)
    device = torch.device("cpu")

    env = gym.make("LunarLander-v3", render_mode="human" if args.render else None)

    if args.seed is not None:
        try:
            env.reset(seed=args.seed)
        except TypeError:
            try:
                env.seed(args.seed)
            except Exception:
                pass
        try:
            env.action_space.seed(args.seed)
        except Exception:
            pass
        try:
            env.observation_space.seed(args.seed)
        except Exception:
            pass

    encoder = PlaceCellEncoder(device)
    actor = ModulatedActor(encoder.n_neurons, device)
    # Critic uses manual Adam update (stylistic parity with Chung & Kozma)
    critic = LocalCritic(encoder.n_neurons, device)

    print(f"Start Switch Lunar Lander (Decoupled Uncertainty: NE~Novelty, ACh~Variance). Episodes: {args.episodes}")

    # Traces for unexpected uncertainty (novelty)
    td_fast = 0.0
    td_slow = 0.0
    td_trace_inited = False

    # EMA of unexpected uncertainty (drives NE)
    avg_unexpected = 0.0

    # EMA of expected uncertainty (drives ACh)
    avg_expected = 0.0

    reward_history = []
    unexpected_history = []
    expected_history = []
    variance_history = []
    td2_history = []
    delta_var_history = []
    actor_lr_history = []
    td_history = []

    max_critic_scale = 0.0
    actor_lr_state = args.base_lr

    for ep in range(args.episodes):
        if args.seed is not None:
            try:
                obs, _ = env.reset(seed=args.seed + ep)
            except TypeError:
                try:
                    env.seed(args.seed + ep)
                    obs, _ = env.reset()
                except Exception:
                    obs, _ = env.reset()
        else:
            obs, _ = env.reset()

        obs_t = torch.tensor(obs, dtype=torch.float32, device=device)
        actor.reset_state()
        critic.reset_state()

        done = False
        total_reward = 0.0
        step_count = 0

        inverted = (ep > 20000)

        # =====================================================================
        # DECOUPLED NEUROMODULATION
        # =====================================================================
        # NE driven by unexpected uncertainty (novelty)
        current_ne = logistic_drive(args.ne_max, NE_K, NE_CENTER, avg_unexpected, args.base_noise)
        current_ne = min(current_ne, 5.0)

        # ACh uses its own logistic (expected uncertainty -> modulatory signal). 
        current_ach = logistic_drive(args.ach_max, ACH_K, ACH_CENTER, avg_expected, args.base_lr)

        td_sum = 0.0
        td_count = 0
        var_sum = 0.0
        var_count = 0
        td2_sum = 0.0
        delta_var_sum = 0.0
        actor_lr_sum = 0.0
        actor_lr_count = 0

        # For printing/debug visibility
        last_td_signal = 0.0
        last_td_fast = td_fast
        last_td_slow = td_slow
        last_td_novelty = 0.0

        while not done:
            spikes = encoder(obs_t)
            
            # Action Sampling: Every 2 environment steps (Chung & Kozma parity)
            if step_count % 2 == 0:
                action_code, act_spikes, act_probs = actor(spikes, noise_scale=current_ne)

            # Control Inversion: Swap Left (1) and Right (3) thrusters
            if inverted:
                if action_code == 1: real_action = 3
                elif action_code == 3: real_action = 1
                else: real_action = action_code
            else:
                real_action = action_code

            next_obs, reward, terminated, truncated, _ = env.step(real_action)
            done = terminated or truncated
            next_obs_t = torch.tensor(next_obs, dtype=torch.float32, device=device)

            # --- REWARD SHAPING ---
            angle = next_obs[4]
            a_vel = next_obs[5]
            angle_bonus = - (angle ** 2) * 2.0
            stability_bonus = - (a_vel ** 2) * 1.0
            shaped_reward = reward + angle_bonus + stability_bonus

            # ------------------------------- 
            # Critic update (Manual Adam)
            # ------------------------------- 
            v_curr, var_curr, h_spikes = critic.forward(spikes)
            var_sum += float(var_curr.item())
            var_count += 1

            if not done:
                next_spikes = encoder(next_obs_t)
                v_next, _, _ = critic.forward(next_spikes)
            else:
                v_next = torch.tensor([0.0], device=device)

            # Reward Scaling (Parity with Table 1: 0.012)
            scaled_reward = shaped_reward * REWARD_SCALE
            target = scaled_reward + GAMMA * v_next.detach()
            
            td_error = target - v_curr.detach()
            td_error_val = float(td_error.item())
            
            # Manual Adam update call
            critic.update(h_spikes, td_error, var_curr, lr=args.critic_base_lr)

            td_sum += abs(td_error_val)
            td_count += 1
            td_error_sq = td_error_val ** 2
            var_val = float(var_curr.item())
            err_var_val = abs(td_error_val) - var_val
            td2_sum += td_error_sq
            
            delta_var_val = args.critic_base_lr * abs(err_var_val) * err_var_val # Approx
            delta_var_sum += delta_var_val

            actor_lr_state *= (1.0 - args.actor_lr_decay)
            actor_lr_state += args.actor_lr_boost * current_ach
            actor_lr_state = min(max(actor_lr_state, args.actor_lr_min), args.actor_lr_max)
            actor_lr = actor_lr_state
            actor_lr_sum += actor_lr
            actor_lr_count += 1

            # ------------------------------- 
            # Actor update (Modulated SGD)
            # ------------------------------- 
            # We pass prob distribution now
            actor.update(td_error_val, act_spikes, act_probs, current_lr=actor_lr)

            # =====================================================================
            # EXPECTED UNCERTAINTY: Direct EMA of critic's variance estimate
            # =====================================================================
            current_sigma = torch.sqrt(var_curr.detach()).item()
            current_sigma = min(max(current_sigma, SURPRISE_EPS), 10.0)
            
            avg_expected = (1.0 - EXP_SURPRISE_DECAY) * avg_expected + EXP_SURPRISE_DECAY * current_sigma

            # =====================================================================
            # UNEXPECTED UNCERTAINTY: Fast-slow TD novelty
            # =====================================================================
            td_signal = abs(td_error_val) / (current_sigma ** SURPRISE_VARIANCE_WEIGHT)
            td_signal = float(min(max(td_signal, 0.0), TD_SIGNAL_CLIP))

            if not td_trace_inited:
                td_fast = td_signal
                td_slow = td_signal
                td_trace_inited = True
            else:
                td_fast = (1.0 - args.td_fast_alpha) * td_fast + args.td_fast_alpha * td_signal
                td_slow = (1.0 - args.td_slow_alpha) * td_slow + args.td_slow_alpha * td_signal

            td_novelty = max(0.0, td_fast - td_slow - TD_NOVELTY_MARGIN)
            avg_unexpected = UNEXP_SURPRISE_DECAY * avg_unexpected + td_novelty #(1.0 - UNEXP_SURPRISE_DECAY) * td_novelty

            # Update dynamics for next step
            current_ne = logistic_drive(args.ne_max, NE_K, NE_CENTER, avg_unexpected, args.base_noise)
            current_ne = min(current_ne, 5.0)
            current_ach = logistic_drive(args.ach_max, ACH_K, ACH_CENTER, avg_expected, args.base_lr)

            obs_t = next_obs_t
            total_reward += float(reward)

            # keep last-step debug values for printouts
            last_td_signal = td_signal
            last_td_fast = td_fast
            last_td_slow = td_slow
            last_td_novelty = td_novelty

        reward_history.append(total_reward)
        unexpected_history.append(avg_unexpected)
        expected_history.append(avg_expected)
        mean_abs_td = (td_sum / td_count) if td_count > 0 else 0.0
        mean_var = (var_sum / var_count) if var_count > 0 else 0.0
        mean_td2 = (td2_sum / td_count) if td_count > 0 else 0.0
        mean_delta_var = (delta_var_sum / td_count) if td_count > 0 else 0.0
        mean_actor_lr = (actor_lr_sum / actor_lr_count) if actor_lr_count > 0 else 0.0
        td_history.append(mean_abs_td)
        variance_history.append(mean_var)
        td2_history.append(mean_td2)
        delta_var_history.append(mean_delta_var)
        actor_lr_history.append(mean_actor_lr)
        avg_r = float(np.mean(reward_history[-20:])) if len(reward_history) > 0 else 0.0

        if ep % 100 == 0:
            status = "NORMAL" if not inverted else "INVERTED"
            print(
                f"Ep {ep:4d} | {status} | R: {total_reward:3.0f} | Avg: {avg_r:4.1f} | "
                f"NE: {current_ne:.2f} | ACh: {current_ach:.4f} | LR: {mean_actor_lr:.6f} | "
                f"TDsig: {last_td_signal:.2f} | Unexpected: {avg_unexpected:.2f} | Expected: {avg_expected:.2f}"
            )

    print(f"Max Critic Scale: {max_critic_scale:.3f}")

    # ----------------------------------------------------------------------- 
    # Plot results
    # ----------------------------------------------------------------------- 
    fig, ax1 = plt.subplots(figsize=(10, 6))

    ax1.plot(range(len(reward_history)), reward_history, color="tab:blue", linewidth=1, alpha=0.6, label="Reward")
    window_size = 20
    if len(reward_history) >= window_size:
        rolling_avg = np.convolve(reward_history, np.ones(window_size) / window_size, mode="valid")
        ax1.plot(range(window_size - 1, len(reward_history)), rolling_avg, color="tab:blue", linewidth=2, label=f"{window_size}-ep Avg")
    ax1.axvline(x=5000, color="r", linestyle="--", label="Switch Point")
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Reward", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")

    ax2 = ax1.twinx()
    ax2.plot(range(len(unexpected_history)), unexpected_history, color="tab:orange", linewidth=1.5, alpha=0.7, label="Unexpected (Novelty)")
    ax2.plot(range(len(expected_history)), expected_history, color="tab:green", linewidth=1.5, alpha=0.7, label="Expected (Variance Novelty)")
    ax2.plot(range(len(variance_history)), variance_history, color="tab:purple", linewidth=1.2, alpha=0.6, label="Critic Variance (Mean)")
    ax2.plot(range(len(td2_history)), td2_history, color="tab:red", linewidth=1.0, alpha=0.6, label="TD Error^2 (Mean)")
    ax2.plot(range(len(delta_var_history)), delta_var_history, color="tab:brown", linewidth=1.0, alpha=0.6, label="Delta Var (Mean)")
    ax2.plot(range(len(actor_lr_history)), actor_lr_history, color="tab:gray", linewidth=1.0, alpha=0.6, label="Actor LR (Mean)")
    ax2.set_ylabel("Uncertainty / Variance", color="tab:orange")
    ax2.tick_params(axis="y", labelcolor="tab:orange")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    ax1.grid(True, alpha=0.3)
    fig.tight_layout()
    plt.title("Switch Lunar Lander - Decoupled Uncertainty (TD-LTP Critic, Actor LR Decay)")

    out_dir = f"ll/runs/{args.seed}_flagship" if args.seed is not None else "ll/runs/noseed_flagship"
    os.makedirs(out_dir, exist_ok=True)

    png_path = os.path.join(out_dir, f"{args.seed}_flagship.png")
    fig.savefig(png_path, dpi=150, bbox_inches="tight")

    csv_path = os.path.join(out_dir, f"{args.seed}_flagship.csv")
    with open(csv_path, "w") as fh:
        fh.write("episode,reward,unexpected_uncertainty,expected_uncertainty,mean_variance,mean_td_error_sq,mean_delta_var,mean_actor_lr,mean_abs_td\n")
        for i, (r, u, e, v, td2, dv, alr, td) in enumerate(zip(reward_history, unexpected_history, expected_history, variance_history, td2_history, delta_var_history, actor_lr_history, td_history)):
            fh.write(f"{i},{r},{u},{e},{v},{td2},{dv},{alr},{td}\n")

    print(f"\nPlot saved as '{png_path}' and CSV saved as '{csv_path}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=20000)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed (overrides top-level SEED)")
    parser.add_argument("--base_lr", type=float, default=BASE_LR)
    parser.add_argument("--base_noise", type=float, default=BASE_NOISE)
    parser.add_argument("--ne_max", type=float, default=NE_MAX)
    parser.add_argument("--ach_max", type=float, default=ACH_MAX)
    parser.add_argument("--critic_base_lr", type=float, default=CRITIC_BASE_LR)
    parser.add_argument("--actor_lr_decay", type=float, default=ACTOR_LR_DECAY)
    parser.add_argument("--actor_lr_boost", type=float, default=ACTOR_LR_BOOST)
    parser.add_argument("--actor_lr_min", type=float, default=ACTOR_LR_MIN)
    parser.add_argument("--actor_lr_max", type=float, default=ACTOR_LR_MAX)
    parser.add_argument("--td_fast_alpha", type=float, default=TD_FAST_ALPHA)
    parser.add_argument("--td_slow_alpha", type=float, default=TD_SLOW_ALPHA)
    args = parser.parse_args()

    train(args)
