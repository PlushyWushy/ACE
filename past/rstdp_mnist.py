#!/usr/bin/env python3
"""
rstdp_mnist.py

Supervised R-STDP training demo on MNIST.

We reuse the three-factor update scheme from the CartPole experiment,
but drive it with a supervised "reward": batches that improve accuracy
receive LTP, while those that underperform incur LTD. Eligibility traces
are clipped to mimic saturating synaptic tags.
"""

import argparse
import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# ---- Spiking neuron components ---------------------------------------------


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
        self.dt = dt
        self.tau = tau
        self.alpha = math.exp(-dt / tau)
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


class SNNClassifier(nn.Module):
    def __init__(self, input_dim, hidden_size=256, n_classes=10, device="cpu"):
        super().__init__()
        self.device = device
        self.lif = LIFCell(input_dim, hidden_size).to(device)
        self.readout = nn.Linear(hidden_size, n_classes).to(device)

    def forward(self, obs, v_prev):
        spike, v_next = self.lif(obs, v_prev)
        logits = self.readout(spike)
        return logits, v_next, spike


# ---- Data helpers ----------------------------------------------------------


def flatten_tensor(tensor):
    return tensor.view(-1)


def build_dataloaders(args) -> Tuple[DataLoader, DataLoader]:
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Lambda(flatten_tensor),  # flatten to 784
        ]
    )

    if args.fake_data:
        train_set = datasets.FakeData(
            size=60000,
            image_size=(1, 28, 28),
            num_classes=10,
            transform=transform,
        )
        test_set = datasets.FakeData(
            size=10000,
            image_size=(1, 28, 28),
            num_classes=10,
            transform=transform,
        )
    else:
        train_set = datasets.MNIST(
            args.data_dir, train=True, download=True, transform=transform
        )
        test_set = datasets.MNIST(
            args.data_dir, train=False, download=True, transform=transform
        )

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=not args.no_cuda,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=args.test_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=not args.no_cuda,
    )
    return train_loader, test_loader


# ---- Training --------------------------------------------------------------


def evaluate(policy, data_loader, device, timesteps):
    policy.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in data_loader:
            data = data.to(device)
            target = target.to(device)
            batch_size = data.size(0)
            v_state = torch.zeros(batch_size, policy.lif.n_neurons, device=device)
            logits_accum = torch.zeros(batch_size, policy.readout.out_features, device=device)
            for _ in range(timesteps):
                logits, v_state, _spike = policy(data, v_state)
                logits_accum += logits
            preds = torch.argmax(logits_accum, dim=1)
            correct += (preds == target).sum().item()
            total += batch_size
    policy.train()
    return correct / max(total, 1)


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    train_loader, test_loader = build_dataloaders(args)

    input_dim = 28 * 28
    n_classes = 10
    policy = SNNClassifier(input_dim, hidden_size=args.hidden, n_classes=n_classes, device=device)

    dopamine_threshold = (
        args.dopamine_threshold
        if args.dopamine_threshold is not None
        else args.default_dopamine_level
    )

    for epoch in range(1, args.epochs + 1):
        running_acc = 0.0
        batches = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            data = data.to(device)
            target = target.to(device)
            batch_size = data.size(0)

            v_state = torch.zeros(batch_size, policy.lif.n_neurons, device=device)
            logits_accum = torch.zeros(batch_size, n_classes, device=device)

            elig_w_out = torch.zeros_like(policy.readout.weight)
            elig_b_out = torch.zeros_like(policy.readout.bias)
            elig_w_in = torch.zeros_like(policy.lif.fc.weight)
            elig_b_in = torch.zeros_like(policy.lif.fc.bias)

            for _ in range(args.timesteps):
                noisy_input = data + args.input_noise * torch.randn_like(data)
                logits, v_state, spike = policy(noisy_input, v_state)
                logits_accum += logits

                pre_in = noisy_input
                post_hidden = spike
                pre_out = spike
                teacher = F.one_hot(target, num_classes=n_classes).float()

                elig_w_in = args.elig_decay * elig_w_in + torch.einsum(
                    "bi,bj->ij", post_hidden, pre_in
                ) / batch_size
                elig_b_in = args.elig_decay * elig_b_in + post_hidden.mean(dim=0)

                elig_w_out = args.elig_decay * elig_w_out + torch.einsum(
                    "bi,bj->ij", teacher, pre_out
                ) / batch_size
                elig_b_out = args.elig_decay * elig_b_out + teacher.mean(dim=0)

                if args.elig_clip > 0.0:
                    clamp_min, clamp_max = -args.elig_clip, args.elig_clip
                    elig_w_in.clamp_(clamp_min, clamp_max)
                    elig_b_in.clamp_(clamp_min, clamp_max)
                    elig_w_out.clamp_(clamp_min, clamp_max)
                    elig_b_out.clamp_(clamp_min, clamp_max)

            preds = torch.argmax(logits_accum, dim=1)
            reward = float((preds == target).float().mean().item())

            dopamine_level = reward
            if args.dopamine_threshold is None:
                dopamine_threshold = (
                    (1.0 - args.baseline_beta) * dopamine_threshold
                    + args.baseline_beta * dopamine_level
                )
            M_signal = dopamine_level - dopamine_threshold

            with torch.no_grad():
                delta_w_out = args.lr_readout * M_signal * elig_w_out
                delta_b_out = args.lr_readout * M_signal * elig_b_out
                delta_w_in = args.lr_hidden * M_signal * elig_w_in
                delta_b_in = args.lr_hidden * M_signal * elig_b_in

                policy.readout.weight.data += delta_w_out
                policy.readout.bias.data += delta_b_out
                policy.lif.fc.weight.data += delta_w_in
                policy.lif.fc.bias.data += delta_b_in

            running_acc += reward
            batches += 1

            if (batch_idx + 1) % args.log_interval == 0:
                print(
                    f"Epoch {epoch} Batch {batch_idx+1}/{len(train_loader)}\t"
                    f"TrainAcc: {reward:.3f}\t"
                    f"DAthr: {dopamine_threshold:.3f}\t"
                    f"M: {M_signal:.3f}\t"
                    f"W_in: {policy.lif.fc.weight.data.norm():.3e}\t"
                    f"W_out: {policy.readout.weight.data.norm():.3e}"
                )

        avg_acc = running_acc / max(batches, 1)
        test_acc = evaluate(policy, test_loader, device, args.timesteps)
        print(
            f"Epoch {epoch} Complete\tAvgTrainAcc: {avg_acc:.3f}\t"
            f"TestAcc: {test_acc:.3f}\tDAthr: {dopamine_threshold:.3f}"
        )

    torch.save(policy.state_dict(), args.save_path)
    print(f"Model saved to {args.save_path}")


def main():
    parser = argparse.ArgumentParser(description="Supervised R-STDP on MNIST")
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--test-batch-size", type=int, default=256)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--timesteps", type=int, default=5)
    parser.add_argument("--lr-hidden", type=float, default=1e-2)
    parser.add_argument("--lr-readout", type=float, default=1e-2)
    parser.add_argument("--elig-decay", type=float, default=0.9)
    parser.add_argument("--elig-clip", type=float, default=5.0)
    parser.add_argument("--input-noise", type=float, default=0.05)
    parser.add_argument("--baseline-beta", type=float, default=0.01)
    parser.add_argument("--dopamine-threshold", type=float, default=None)
    parser.add_argument("--default-dopamine-level", type=float, default=0.2)
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--save-path", type=str, default="snn_mnist_rstdp.pt")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--no-cuda", action="store_true", help="Force CPU even if CUDA is available")
    parser.add_argument(
        "--fake-data",
        action="store_true",
        help="Use torchvision FakeData to avoid downloads (useful for quick tests).",
    )
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
