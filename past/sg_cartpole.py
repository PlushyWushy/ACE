#!/usr/bin/env python3
"""
CartPole training with the same SNN architecture as rstdp.py but optimized via
surrogate-gradient backpropagation (policy gradient style).
"""

import argparse
import gym
import numpy as np
import torch
import torch.nn as nn


class SurrogateSpike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return (input > 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        (input,) = ctx.saved_tensors
        grad = grad_output / (1.0 + input.abs()) ** 2
        return grad


surrogate_spike = SurrogateSpike.apply


class LIFCell(nn.Module):
    def __init__(self, n_in, n_neurons, dt=1.0, tau=20.0, v_th=1.0, v_reset=0.0):
        super().__init__()
        self.n_in = n_in
        self.n_neurons = n_neurons
        self.alpha = float(np.exp(-dt / tau))
        self.v_th = v_th
        self.v_reset = v_reset
        self.fc = nn.Linear(n_in, n_neurons)

    def forward(self, x, v_prev):
        I = self.fc(x)
        v_next = self.alpha * v_prev + I
        m = v_next - self.v_th
        spike = surrogate_spike(m)
        v_next = v_next * (1.0 - spike) + self.v_reset * spike
        return spike, v_next


class SNNPolicy(nn.Module):
    def __init__(self, obs_dim, hidden_size=64, n_actions=2):
        super().__init__()
        self.lif = LIFCell(obs_dim, hidden_size)
        self.readout = nn.Linear(hidden_size, n_actions)

    def forward(self, obs, v_state):
        spike, v_next = self.lif(obs, v_state)
        logits = self.readout(spike)
        return logits, v_next, spike


def select_action(policy, obs, v_state):
    logits, v_next, _ = policy(obs, v_state)
    dist = torch.distributions.Categorical(logits=logits)
    action = dist.sample()
    log_prob = dist.log_prob(action)
    return action.item(), log_prob, v_next


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    env = gym.make("CartPole-v1")
    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    policy = SNNPolicy(obs_dim, hidden_size=args.hidden, n_actions=n_actions).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr)

    rewards_history = []
    for ep in range(1, args.episodes + 1):
        obs = env.reset()
        obs = obs[0] if isinstance(obs, tuple) else obs
        v_state = torch.zeros(1, policy.lif.n_neurons, device=device)

        log_probs = []
        rewards = []

        for _ in range(args.max_steps):
            obs_t = torch.from_numpy(obs).float().unsqueeze(0).to(device)
            action, log_prob, v_state = select_action(policy, obs_t, v_state)
            step_out = env.step(action)
            if isinstance(step_out, tuple) and len(step_out) == 5:
                obs_next, reward, terminated, truncated, _ = step_out
                done = bool(terminated or truncated)
            else:
                obs_next, reward, done, _ = step_out

            log_probs.append(log_prob)
            rewards.append(reward)
            obs = obs_next
            if done:
                break

        total_reward = sum(rewards)
        rewards_history.append(total_reward)

        returns = []
        G = 0.0
        for r in reversed(rewards):
            G = r + args.gamma * G
            returns.append(G)
        returns = torch.tensor(list(reversed(returns)), dtype=torch.float32, device=device)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        policy_loss = []
        for log_prob, Gt in zip(log_probs, returns):
            policy_loss.append(-log_prob * Gt)
        loss = torch.stack(policy_loss).sum()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if ep % args.log_interval == 0:
            avg_reward = np.mean(rewards_history[-args.log_interval:])
            print(f"Episode {ep}\tReward: {total_reward:.2f}\tAvgReward: {avg_reward:.2f}")

    torch.save(policy.state_dict(), args.save_path)
    env.close()


def main():
    parser = argparse.ArgumentParser(description="CartPole SNN with surrogate gradients")
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--save-path", type=str, default="snn_cartpole_surrogate.pt")
    parser.add_argument("--no-cuda", action="store_true")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
