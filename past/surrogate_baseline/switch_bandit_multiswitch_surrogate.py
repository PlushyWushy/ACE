#!/usr/bin/env python3
"""
Surrogate baseline for the multi-switch bandit experiment.
Standard softmax policy gradient (REINFORCE) with a linear critic, trained via Adam.
Matches the time scale of sb_multiswitch/icarus_upgraded_multiswitch.py
(60,000 episodes, 4 arms, deterministic switches at 15k/30k/45k).

Key differences from the spiking ACE model:
- Actor: softmax policy over 4 learned logits → REINFORCE update (vs Hebbian + neuromodulation)
- Critic: linear value head → MSE via Adam (vs local TD-LTP)
- Exploration: softmax entropy (vs stochastic Bernoulli spikes + NE-driven noise)
- No neuromodulation: fixed learning rates throughout
"""

import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
ACTOR_LR = 0.01
CRITIC_LR = 0.01
ENTROPY_COEF = 0.01

# Environment
EPISODES = 60000
N_ARMS = 4
SWITCH_SCHEDULE = [15000, 30000, 45000]

# Surprise trace params (for logging, not used for modulation)
TD_FAST_ALPHA = 0.097663
TD_SLOW_ALPHA = 0.1
UNEXP_SURPRISE_DECAY = 0.8
EXP_SURPRISE_DECAY = 1.0
SURPRISE_EPS = 1e-3
TD_SIGNAL_CLIP = 20.0

SEED = 5


def set_global_seed(seed: int | None):
    if seed is None:
        return
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _get_deterministic_switches(n_arms: int):
    """Same deterministic switch sequence as the spiking counterpart.
    Arm sequence: 0 → 1 → 2 → 3 → 0 ..."""
    points = sorted(SWITCH_SCHEDULE)
    arm_sequence = []
    current_arm = 0
    for _ in points:
        current_arm = (current_arm + 1) % n_arms
        arm_sequence.append(current_arm)
    return points, arm_sequence


class SoftmaxActor(nn.Module):
    """Softmax policy over n_actions. Standard ML policy gradient actor."""
    def __init__(self, n_actions: int = 4):
        super().__init__()
        self.logits = nn.Parameter(torch.zeros(n_actions))

    def forward(self):
        probs = F.softmax(self.logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action), dist.entropy()


class LinearCritic(nn.Module):
    """Linear value + variance critic for a single-state bandit."""
    def __init__(self):
        super().__init__()
        self.value = nn.Parameter(torch.zeros(1))
        self.log_var = nn.Parameter(torch.zeros(1))

    def forward(self):
        v = self.value
        var = F.softplus(self.log_var) + 1e-4
        return v, var


def train(episodes: int = EPISODES, seed: int | None = SEED, n_arms: int = N_ARMS):
    set_global_seed(seed)

    switch_points, arm_sequence = _get_deterministic_switches(n_arms)

    actor = SoftmaxActor(n_actions=n_arms)
    critic = LinearCritic()
    actor_optim = optim.Adam(actor.parameters(), lr=ACTOR_LR)
    critic_optim = optim.Adam(critic.parameters(), lr=CRITIC_LR)

    # Surprise traces (logged for comparison but don't affect learning)
    td_fast = 0.0
    td_slow = 0.0
    td_inited = False
    avg_unexpected = 0.0
    avg_expected = 0.0

    reward_history = []
    unexpected_history = []
    expected_history = []

    optimal_arm = 0

    for ep in range(1, episodes + 1):
        # Check for switch
        if switch_points and ep == switch_points[0]:
            optimal_arm = arm_sequence.pop(0)
            switch_points.pop(0)

        # Build reward probabilities: 1.0 for optimal arm, 0.0 for others
        prob = [0.0] * n_arms
        prob[optimal_arm] = 1.0
        optimal = optimal_arm

        # Forward pass
        action, log_prob, entropy = actor()
        reward = 1.0 if np.random.rand() < prob[action] else -1.0

        # Critic
        v, var = critic()
        td_error_val = reward - v.item()

        # Actor update: REINFORCE + entropy regularization
        actor_loss = -(td_error_val * log_prob) - ENTROPY_COEF * entropy
        actor_optim.zero_grad()
        actor_loss.backward()
        actor_optim.step()

        # Critic update: MSE for value + variance
        val_loss = (reward - v).pow(2)
        var_target = (reward - v.detach()).pow(2)
        var_loss = (var - var_target).pow(2)
        critic_loss = val_loss + var_loss
        critic_optim.zero_grad()
        critic_loss.backward()
        critic_optim.step()

        # Update surprise traces (for logging only)
        sigma = max(var.detach().sqrt().item(), SURPRISE_EPS)
        td_signal = min(max(abs(td_error_val), 0.0), TD_SIGNAL_CLIP)

        if not td_inited:
            td_fast = td_signal
            td_slow = td_signal
            td_inited = True
        else:
            td_fast = (1.0 - TD_FAST_ALPHA) * td_fast + TD_FAST_ALPHA * td_signal
            td_slow = (1.0 - TD_SLOW_ALPHA) * td_slow + TD_SLOW_ALPHA * td_signal

        td_novelty = max(0.0, td_fast - td_slow)
        avg_unexpected = UNEXP_SURPRISE_DECAY * avg_unexpected + td_novelty
        avg_expected = (1.0 - EXP_SURPRISE_DECAY) * avg_expected + EXP_SURPRISE_DECAY * sigma ** 2

        reward_history.append(1 if action == optimal else 0)
        unexpected_history.append(avg_unexpected)
        expected_history.append(avg_expected)

        if ep % 5000 == 0:
            recent = np.mean(reward_history[-500:]) * 100 if len(reward_history) >= 500 else np.mean(reward_history) * 100
            print(f"Ep {ep:6d} | Opt%: {recent:5.1f} | Optimal Arm: {optimal_arm} | Unexpected: {avg_unexpected:.3f} | Expected: {avg_expected:.4f}")

    # Save CSV
    out_dir = f"surrogate_baseline/runs/{seed}_bandit_multiswitch_surrogate" if seed is not None else "surrogate_baseline/runs/noseed_bandit_multiswitch_surrogate"
    os.makedirs(out_dir, exist_ok=True)

    csv_path = os.path.join(out_dir, f"{seed}_bandit_multiswitch_surrogate.csv")
    with open(csv_path, "w") as fh:
        fh.write("episode,is_optimal,expected,unexpected\n")
        for i, (r, e, u) in enumerate(zip(reward_history, expected_history, unexpected_history)):
            fh.write(f"{i+1},{int(r)},{e:.6f},{u:.6f}\n")

    print(f"Saved CSV to '{csv_path}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=EPISODES)
    parser.add_argument("--seed", type=int, default=None, help="Random seed (optional)")
    parser.add_argument("--arms", type=int, default=N_ARMS, help="Number of arms")
    args = parser.parse_args()
    train(args.episodes, seed=args.seed, n_arms=args.arms)
