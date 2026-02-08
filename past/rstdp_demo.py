#!/usr/bin/env python3
"""
rstdp_demo.py

Classic reward-modulated STDP demo on a tiny pattern-classification task.

Two prototype input patterns (A/B) drive Poisson spike trains on 20 input afferents.
Two output neurons compete to represent the class. R-STDP adjusts the synapses so
that each output fires preferentially for its corresponding pattern.
"""

import argparse
import math
import random
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn


class SimpleLIF(nn.Module):
    def __init__(
        self,
        n_inputs: int,
        n_outputs: int,
        dt: float = 1.0,
        tau: float = 10.0,
        v_th: float = 1.0,
        v_reset: float = 0.0,
    ):
        super().__init__()
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.dt = dt
        self.tau = tau
        self.alpha = math.exp(-dt / tau)
        self.v_th = v_th
        self.v_reset = v_reset
        self.fc = nn.Linear(n_inputs, n_outputs, bias=True)
        nn.init.normal_(self.fc.weight, mean=0.0, std=0.1)
        nn.init.constant_(self.fc.bias, 0.0)

    def forward_step(self, spikes_in: torch.Tensor, v_prev: torch.Tensor):
        """
        spikes_in: [batch, n_inputs] binary spike vector.
        v_prev: [batch, n_outputs]
        """
        I = self.fc(spikes_in)
        v_next = self.alpha * v_prev + I
        m = v_next - self.v_th
        spikes_out = (m > 0).float()
        v_next = v_next * (1.0 - spikes_out) + self.v_reset * spikes_out
        return spikes_out, v_next


def generate_poisson_pattern(
    base_rates: np.ndarray, timesteps: int, rng: random.Random
) -> np.ndarray:
    """Generate binary spike train from Bernoulli rates."""
    n_inputs = base_rates.shape[0]
    spikes = np.zeros((timesteps, n_inputs), dtype=np.float32)
    for t in range(timesteps):
        spikes[t] = rng.random() < base_rates
    return spikes


def build_patterns(n_inputs: int, rng: random.Random) -> Tuple[np.ndarray, np.ndarray]:
    """Create two prototype firing-rate patterns in [0,1]."""
    center_a = rng.uniform(0.2, 0.8)
    center_b = rng.uniform(0.2, 0.8)
    pattern_a = np.clip(
        rng.random() * 0.2 + center_a + 0.2 * (np.random.rand(n_inputs) - 0.5),
        0.05,
        0.95,
    )
    pattern_b = np.clip(
        rng.random() * 0.2 + center_b + 0.2 * (np.random.rand(n_inputs) - 0.5),
        0.05,
        0.95,
    )
    return pattern_a.astype(np.float32), pattern_b.astype(np.float32)


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    rng = random.Random(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    pattern_a, pattern_b = build_patterns(args.n_inputs, rng)
    patterns = {0: pattern_a, 1: pattern_b}

    snn = SimpleLIF(args.n_inputs, args.n_outputs).to(device)

    baseline = 0.0
    accuracy_history = []

    for epoch in range(1, args.epochs + 1):
        correct = 0
        total = 0

        for trial in range(args.trials_per_epoch):
            label = rng.choice([0, 1])
            rates = patterns[label]
            spikes = np.random.rand(args.timesteps, args.n_inputs) < rates
            spikes = torch.from_numpy(spikes.astype(np.float32)).to(device)

            v_state = torch.zeros(1, args.n_outputs, device=device)
            pre_trace = torch.zeros(args.n_inputs, device=device)
            post_trace = torch.zeros(args.n_outputs, device=device)
            elig_w = torch.zeros_like(snn.fc.weight)
            elig_b = torch.zeros_like(snn.fc.bias)
            spike_counts = torch.zeros(args.n_outputs, device=device)

            for t in range(args.timesteps):
                x_t = spikes[t].unsqueeze(0)
                out_spike, v_state = snn.forward_step(x_t, v_state)

                spike_counts += out_spike.squeeze(0)

                pre_trace = args.trace_decay * pre_trace + x_t.squeeze(0)
                post_trace = args.trace_decay * post_trace + out_spike.squeeze(0)

                elig_w = (
                    args.elig_decay * elig_w
                    + args.ltp_scale * torch.outer(out_spike.squeeze(0), pre_trace)
                    - args.ltd_scale * torch.outer(post_trace, x_t.squeeze(0))
                )
                elig_b = (
                    args.elig_decay * elig_b
                    + args.ltp_scale * out_spike.squeeze(0)
                    - args.ltd_scale * post_trace
                )

                if args.elig_clip > 0.0:
                    elig_w.clamp_(-args.elig_clip, args.elig_clip)
                    elig_b.clamp_(-args.elig_clip, args.elig_clip)

            pred = int(torch.argmax(spike_counts).item())
            reward = 1.0 if pred == label else 0.0
            correct += int(pred == label)
            total += 1

            baseline = (1.0 - args.baseline_beta) * baseline + args.baseline_beta * reward
            M_signal = reward - baseline

            with torch.no_grad():
                snn.fc.weight.data += args.lr * M_signal * elig_w
                snn.fc.bias.data += args.lr * M_signal * elig_b
                if args.weight_clip > 0.0:
                    snn.fc.weight.data.clamp_(-args.weight_clip, args.weight_clip)

        epoch_acc = correct / max(total, 1)
        accuracy_history.append(epoch_acc)

        if epoch % args.log_interval == 0:
            print(
                f"Epoch {epoch}\tAcc: {epoch_acc:.3f}\t"
                f"Baseline: {baseline:.3f}\t"
                f"Wnorm: {snn.fc.weight.data.norm():.3e}"
            )

    print("Training complete.")
    print(f"Final accuracy: {accuracy_history[-1]:.3f}")


def main():
    parser = argparse.ArgumentParser(description="Classic reward-modulated STDP demo")
    parser.add_argument("--n-inputs", type=int, default=20)
    parser.add_argument("--n-outputs", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--trials-per-epoch", type=int, default=200)
    parser.add_argument("--timesteps", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--trace-decay", type=float, default=0.9)
    parser.add_argument("--elig-decay", type=float, default=0.95)
    parser.add_argument("--ltp-scale", type=float, default=1.0)
    parser.add_argument("--ltd-scale", type=float, default=1.0)
    parser.add_argument("--elig-clip", type=float, default=5.0)
    parser.add_argument("--weight-clip", type=float, default=2.0)
    parser.add_argument("--baseline-beta", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--no-cuda", action="store_true")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
