#!/usr/bin/env python3
"""
Surrogate baseline for the switch bandit experiment.
Standard softmax policy gradient (REINFORCE) with a linear critic, trained via Adam.
Matches the time scale of sb/icarus_upgraded.py (20,000 episodes, switch at 10,000).

Key differences from the spiking ACE model:
- Actor: softmax policy over learned logits → REINFORCE update (vs Hebbian + neuromodulation)
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
EPISODES = 20000
SWITCH_EP = 10000

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


class SoftmaxActor(nn.Module):
    """Softmax policy over n_actions. Standard ML policy gradient actor."""
    def __init__(self, n_actions: int = 2):
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


def train(episodes: int = EPISODES, seed: int | None = SEED):
    set_global_seed(seed)

    actor = SoftmaxActor(n_actions=2)
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

    for ep in range(1, episodes + 1):
        # Environment: deterministic switch at SWITCH_EP
        if ep <= SWITCH_EP:
            prob = [1.0, 0.0]
            optimal = 0
        else:
            prob = [0.0, 1.0]
            optimal = 1

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

        if ep % 1000 == 0:
            recent = np.mean(reward_history[-200:]) * 100 if len(reward_history) >= 200 else np.mean(reward_history) * 100
            print(f"Ep {ep:6d} | Opt%: {recent:5.1f} | Unexpected: {avg_unexpected:.3f} | Expected: {avg_expected:.4f}")

    # Save CSV
    out_dir = f"surrogate_baseline/runs/{seed}_bandit_surrogate" if seed is not None else "surrogate_baseline/runs/noseed_bandit_surrogate"
    os.makedirs(out_dir, exist_ok=True)

    csv_path = os.path.join(out_dir, f"{seed}_bandit_surrogate.csv")
    with open(csv_path, "w") as fh:
        fh.write("episode,is_optimal,expected,unexpected\n")
        for i, (r, e, u) in enumerate(zip(reward_history, expected_history, unexpected_history)):
            fh.write(f"{i+1},{int(r)},{e:.6f},{u:.6f}\n")

    print(f"Saved CSV to '{csv_path}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=EPISODES)
    parser.add_argument("--seed", type=int, default=None, help="Random seed (optional)")
    args = parser.parse_args()
    train(args.episodes, seed=args.seed)
