#!/usr/bin/env python3
"""
Monolithic TD-STDP CartPole script.

This is a single-file copy of the current `td_stdp` system (main + util + bindsnet_add),
placed under `cartpole_successful/` for convenience.

Usage (same idea as `td_stdp/main.py`):
  python cartpole_successful/spike.py -c config_cp.ini

Config resolution:
- If `-c/--config` includes a path separator, it is treated as a path.
- Otherwise, it is loaded from `td_stdp/config/<name>`.
"""

from __future__ import annotations

import argparse
import collections
import configparser
import os
import pickle
import sys
from abc import ABC
from typing import Dict, Iterable, List, Optional, Sequence, Sized, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor

import gym

# ---------------------------------------------------------------------------
# Editable defaults (Flagship-style knobs)
# ---------------------------------------------------------------------------

# Variance head (critic uncertainty / ACh source)
DEFAULT_VAR_HEAD = True
DEFAULT_NUM_VARS_EXC = None  # None => match `num_critics_exc`
DEFAULT_VAR_LR_SCALE = 1.0  # var_lr = DEFAULT_VAR_LR_SCALE * critic_lr
DEFAULT_VAR_M = 1.0
DEFAULT_VAR_B = 0.0
DEFAULT_VAR_EPS = 1e-6

# Neuromodulation switches
DEFAULT_USE_NE_NOISE = True

# In this `spike_bind.py` variant, actor LR is *not* driven by expected uncertainty (variance / ACh).
# Instead, it's engineered to spike at the env switch point and anneal back to baseline.
DEFAULT_USE_ACH_LR_MOD = False
DEFAULT_USE_SWITCH_LR_SPIKE = True

# NE (noise) from TD novelty
DEFAULT_BASE_NOISE = 0.344308
DEFAULT_NE_MAX = 3.0
DEFAULT_NE_K = 0.2
DEFAULT_NE_CENTER = 15.0
DEFAULT_TD_SIGNAL_CLIP = 20.0
DEFAULT_TD_FAST_ALPHA = 0.097663 
DEFAULT_TD_SLOW_ALPHA = 0.003642
DEFAULT_UNEXP_DECAY = 0.8
DEFAULT_SURPRISE_VARIANCE_WEIGHT = 0.0
DEFAULT_TD_NOVELTY_MARGIN = 0.0

# ACh (LR drive) from expected uncertainty (variance)
DEFAULT_BASE_LR_MOD = 0.000055
DEFAULT_ACH_MAX = 0.004061
DEFAULT_ACH_K = 4.0
DEFAULT_ACH_CENTER = 0.06
DEFAULT_EXP_FAST_ALPHA = 1.0
DEFAULT_EXP_SLOW_ALPHA = 0.1
DEFAULT_EXP_DECAY = 0.5

# Actor LR modulation dynamics (only used if DEFAULT_USE_ACH_LR_MOD / use_ach_lr_mod enabled)
DEFAULT_ACTOR_LR_DECAY = 0.1
DEFAULT_ACTOR_LR_BOOST = 0.1
DEFAULT_ACTOR_LR_MIN = 0.01
DEFAULT_ACTOR_LR_MAX = 1

# Engineered actor LR spike at switch (only used if DEFAULT_USE_SWITCH_LR_SPIKE / use_switch_lr_spike enabled)
DEFAULT_SWITCH_LR_PEAK = 2
DEFAULT_SWITCH_LR_HOLD_EPS = 1
DEFAULT_SWITCH_LR_ANNEAL_EPS = 5 

# Logging/output (match `cartpole_successful/flagship.py` style)
DEFAULT_FLAGSHIP_LOGS = True
DEFAULT_PRINT_EVERY = 10
DEFAULT_OUT_BASEDIR = os.path.join("cartpole", "runs")

# Env switch (CartPole only): after N completed episodes, invert controls (a -> 1-a).
DEFAULT_SWITCH_AT = 150

# ---------------------------------------------------------------------------
# Gym API compatibility helpers
# ---------------------------------------------------------------------------


def _gym_reset(env):
    out = env.reset()
    if isinstance(out, tuple) and len(out) == 2:
        obs, _info = out
        return obs
    return out


def _gym_step(env, action):
    out = env.step(action)
    if isinstance(out, tuple) and len(out) == 5:
        obs, reward, terminated, truncated, info = out
        done = bool(terminated) or bool(truncated)
        info = dict(info)
        info.setdefault("TimeLimit.truncated", bool(truncated))
        return obs, reward, done, info
    return out


# ---------------------------------------------------------------------------
# Utility functions and encoders (from `td_stdp/util.py`, trimmed to what we use)
# ---------------------------------------------------------------------------


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def softmax(x):
    e_x = np.exp(x - np.amax(x, axis=-1)[..., np.newaxis])
    return e_x / e_x.sum(axis=-1)[..., np.newaxis]


def relu(x):
    y = np.copy(x)
    y[y < 0] = 0
    return y


def mv(a, n=1000):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1 :] / n


def logistic_drive(max_val: float, k: float, center: float, signal: float, base: float) -> float:
    z = k * (signal - center)
    if z >= 700:
        return max_val + base
    if z <= -700:
        return base
    return max_val / (1.0 + np.exp(-z)) + base


def real_to_bin(real, bin_min=0, bin_max=1, bin_num=10):
    bin_index = np.array((real - bin_min) / (bin_max - bin_min) * bin_num).astype(int)
    bin_index -= 1
    bin_index[bin_index >= bin_num] = bin_num - 1
    bin_index[bin_index < 0] = 0
    return bin_index


class uniform_policy:
    def __init__(self, batch_size, actions):
        self.actions = np.array(actions, dtype=int)
        self.batch_size = batch_size

    def __call__(self, state):
        return np.random.choice(self.actions, size=self.batch_size)


class State_to_spike_bin:
    def __init__(self, time, bin_min, bin_max, bin_num, basis=False, rep=1):
        self.time = time
        self.bin_min = bin_min
        self.bin_max = bin_max
        self.bin_num = bin_num
        self.basis = basis
        self.rep = rep
        self.out_shape = (rep, len(bin_min) * bin_num + (1 if self.basis else 0))

    def __call__(self, state, augment=None):
        b = real_to_bin(real=state, bin_min=self.bin_min, bin_max=self.bin_max, bin_num=self.bin_num)
        bins = np.zeros(state.shape + (self.bin_num,), dtype=int)
        bins[np.arange(state.shape[0])[..., np.newaxis], np.arange(state.shape[1]), b] = 1
        bins = np.reshape(bins, (state.shape[0], -1))
        if self.basis:
            bins = np.concatenate([bins, np.ones((state.shape[0], 1))], axis=-1)
        if augment is not None:
            bins = np.concatenate([bins, augment], axis=-1)
        bins = np.broadcast_to(bins, (self.time, self.rep) + bins.shape)
        bins = np.swapaxes(bins, 1, 2).reshape(self.time, state.shape[0], self.rep, -1)
        return torch.from_numpy(bins).float()


class State_to_spike_RBF:
    def __init__(self, time, bin_min, bin_max, bin_num, basis=False, rep=1):
        self.time = time
        self.bin_min = bin_min
        self.bin_max = bin_max
        self.bin_size = (bin_max - bin_min) / bin_num
        self.rbf_mean = bin_min[..., np.newaxis] + self.bin_size[..., np.newaxis] * (np.arange(bin_num) + 0.5)
        self.rbf_var = self.bin_size[..., np.newaxis] / 2
        self.basis = basis
        self.rep = rep
        self.out_shape = (rep, len(bin_min) * bin_num + (1 if self.basis else 0))

    def __call__(self, state, augment=None):
        state = np.clip(state, self.bin_min + self.bin_size / 2, self.bin_max - self.bin_size / 2)
        rbf_output = np.exp(-np.abs(state[..., np.newaxis] - self.rbf_mean) / self.rbf_var).reshape(state.shape[0], -1)
        if self.basis:
            rbf_output = np.concatenate([rbf_output, np.ones((state.shape[0], 1))], axis=-1)
        if augment is not None:
            rbf_output = np.concatenate([rbf_output, augment], axis=-1)
        rbf_spike = np.random.binomial(n=1, p=rbf_output, size=(self.time, self.rep) + rbf_output.shape)
        rbf_spike = np.swapaxes(rbf_spike, 1, 2).reshape(self.time, state.shape[0], self.rep, -1)
        return torch.from_numpy(rbf_spike).float()


class State_to_spike_Fourier:
    def __init__(self, time, bin_min, bin_max, k=2, basis=False, soft=False, cross_term=True, double=False, rep=1):
        n = len(bin_min)
        if type(k) != np.ndarray:
            if type(k) == list:
                k = np.array(k)
            else:
                k = np.array([k] * n)
        if cross_term:
            self._c = np.zeros((n, np.prod(k + 1)))
            for i in range(np.prod(k + 1)):
                l = i
                for j in range(n):
                    self._c[j, i] = l % (k[j] + 1)
                    l //= (k[j] + 1)
                    if l == 0:
                        break
        else:
            self._c = np.zeros((n, np.sum(k)))
            j, l = 0, 0
            for i in range(np.sum(k)):
                l += 1
                self._c[j, i] = l
                if l >= k[j]:
                    l = 0
                    j += 1
        self._k = k
        self._n = n
        self.mean_re = (bin_max + bin_min) / 2
        self.range_re = bin_max - bin_min
        self.time = time
        self.out_shape = (rep, (self._c.shape[-1]) * (2 if double else 1) + (1 if basis else 0))
        self.basis = basis
        self.soft = soft
        self.double = double
        self.rep = rep

    def __call__(self, state, augment=None):
        norm_state = (state - self.mean_re) / self.range_re + 0.5
        norm_state = np.clip(norm_state, 0, 1)
        if self.double:
            f_basis_p = relu(np.cos(np.pi * np.dot(norm_state, self._c)))
            f_basis_n = relu(-np.cos(np.pi * np.dot(norm_state, self._c)))
            f_basis = np.concatenate([f_basis_p, f_basis_n], -1)
        else:
            f_basis = (np.cos(np.pi * np.dot(norm_state, self._c)) + 1) / 2
        if self.basis:
            f_basis = np.concatenate([f_basis, np.ones((state.shape[0], 1))], axis=-1)
        if augment is not None:
            f_basis = np.concatenate([f_basis, augment], axis=-1)

        if self.soft:
            f_output = np.broadcast_to(f_basis[np.newaxis, np.newaxis, :, :], (self.time, self.rep) + f_basis.shape)
        else:
            f_output = np.random.binomial(n=1, p=f_basis, size=(self.time, self.rep) + f_basis.shape)
        f_output = np.swapaxes(f_output, 1, 2).reshape(self.time, state.shape[0], self.rep, -1)
        return torch.from_numpy(f_output).float()


class State_to_spike_PlaceCell_CartPole:
    """
    Place-cell population code for CartPole (6 x 6 x 8 x 8 = 2304 cells).
    Matches our `cartpole_successful` encoder; added to `td_stdp` as input_type=3.
    """

    def __init__(self, time, basis=False, rep=1, rho_pc=50.0, dt_s=1e-3):
        self.time = int(time)
        self.basis = bool(basis)
        self.rep = int(rep)
        self.rho_pc = float(rho_pc)
        self.dt_s = float(dt_s)

        m = torch.linspace(-2.5, 2.5, 6)
        n = torch.linspace(-2.0, 2.0, 6)
        p = torch.linspace(-0.25, 0.25, 8)
        q = torch.linspace(-2.0, 2.0, 8)

        mesh = torch.meshgrid(m, n, p, q, indexing="ij")
        centers = torch.stack([x.flatten() for x in mesh], dim=1)
        self.centers = centers

        s1 = (m[1] - m[0]) / 1.5
        s2 = (n[1] - n[0]) / 1.5
        s3 = (p[1] - p[0]) / 1.5
        s4 = (q[1] - q[0]) / 1.5
        self.sigmas = torch.tensor([s1, s2, s3, s4]).unsqueeze(0)

        self._clip_min = torch.tensor([m.min(), n.min(), p.min(), q.min()])
        self._clip_max = torch.tensor([m.max(), n.max(), p.max(), q.max()])

        n_cells = int(self.centers.shape[0])
        self.out_shape = (self.rep, n_cells + (1 if self.basis else 0))

    def __call__(self, state, augment=None):
        state_t = torch.as_tensor(state, dtype=torch.float32)
        if state_t.ndim != 2 or state_t.shape[1] != 4:
            raise ValueError(f"CartPole place-cell encoder expects state shape [batch, 4], got {tuple(state_t.shape)}")

        state_t = torch.clamp(state_t, self._clip_min, self._clip_max)
        diff = (state_t.unsqueeze(1) - self.centers.unsqueeze(0)) ** 2
        exponent = -torch.sum(diff / (2 * (self.sigmas**2)), dim=2)
        rates = self.rho_pc * torch.exp(exponent)
        probs = 1.0 - torch.exp(-rates * self.dt_s)
        probs = torch.clamp(probs, 0.0, 1.0)

        bsz, n_cells = probs.shape
        u = torch.rand(self.time, bsz, self.rep, n_cells, dtype=torch.float32)
        spikes = (u < probs.unsqueeze(0).unsqueeze(2)).to(torch.float32)

        extras = []
        if self.basis:
            extras.append(torch.ones(self.time, bsz, self.rep, 1, dtype=torch.float32))
        if augment is not None:
            aug_t = torch.as_tensor(augment, dtype=torch.float32)
            extras.append(aug_t.unsqueeze(0).unsqueeze(2).expand(self.time, bsz, self.rep, -1))
        if extras:
            spikes = torch.cat([spikes] + extras, dim=-1)

        return spikes


class batch_envs:
    def __init__(self, name, batch_size=1, rest_n=100, warm_n=100):
        self._name = name
        self._batch_size = int(batch_size)
        self._action = None
        self._reward = np.zeros(batch_size)
        self._isEnd = np.ones(batch_size, bool)
        self._truncatedEnd = np.zeros(batch_size, bool)
        self._rest = np.zeros(batch_size)
        self._warm = np.zeros(batch_size)

        self._env = [gym.make(name) for _ in range(batch_size)]
        obs0 = np.asarray(_gym_reset(self._env[0]))
        self._state = np.zeros((batch_size,) + obs0.shape, dtype=obs0.dtype)
        self._stateCode = np.zeros(batch_size, dtype=int)
        self._rest_n = rest_n
        self._warm_n = warm_n
        self.reset()

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def reward(self):
        return self._reward

    @property
    def action_space(self):
        return self._env[0].action_space

    @property
    def isEnd(self):
        return self._isEnd

    @property
    def stateCode(self):
        return self._stateCode

    @property
    def state(self):
        return self._state

    @property
    def info(self):
        return {"stateCode": self._stateCode, "truncatedEnd": self._truncatedEnd}

    def step(self, action):
        self._rest[self._isEnd] += 1
        self._warm += 1
        self.reset(self._rest > self._rest_n)
        isLive = np.logical_and(self._warm > self._warm_n, ~self._isEnd)
        self._reward[~isLive] = 0
        for i in isLive.nonzero()[0]:
            self._state[i], self._reward[i], self._isEnd[i], info = _gym_step(self._env[i], action[i])
            self._truncatedEnd[i] = info["TimeLimit.truncated"] if "TimeLimit.truncated" in info else False
            if self._truncatedEnd[i]:
                self._rest[i] = self._rest_n

        self._stateCode[isLive] = 0
        self._stateCode[self._rest >= 1] = 1
        self._stateCode[self._warm <= self._warm_n] = 3
        self._stateCode[self._warm == 0] = 2
        return self.state, self.reward, self._isEnd, self.info

    def reset(self, index=None):
        for i in range(self._batch_size) if index is None else index.nonzero()[0]:
            self._state[i] = _gym_reset(self._env[i])
            self._reward[i] = 0
        if index is None:
            index = slice(None)
        self._rest[index] = 0
        self._warm[index] = 0
        self._truncatedEnd[index] = False
        self._isEnd[index] = False
        return self._state


def plot_performance(
    performances: Dict[str, List[float]],
    fig=None,
    ax=None,
    figsize: Tuple[int, int] = (8, 6),
    title: str = "Estimated classification accuracy",
    xlabel: str = "No. of examples",
    ylabel: str = "Accuracy",
    ylim: Tuple[int, int] = None,
):
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        box = ax.get_position()
        ax.set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.9])

    ax.clear()
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if ylim is not None:
        ax.set_ylim(ylim)

    for scheme in performances:
        ax.plot(range(len(performances[scheme])), [p for p in performances[scheme]], label=scheme)

    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), fancybox=True, shadow=False, ncol=4)
    fig.canvas.draw()
    fig.show()
    return fig, ax


# ---------------------------------------------------------------------------
# BindsNET additions (from `td_stdp/bindsnet_add.py`, copied and minimally edited)
# ---------------------------------------------------------------------------

from bindsnet.learning import LearningRule  # noqa: E402
from bindsnet.network import Network  # noqa: E402
from bindsnet.network.nodes import AdaptiveLIFNodes, Input, LIFNodes  # noqa: E402
from bindsnet.network.topology import AbstractConnection, Connection, LocalConnection  # noqa: E402


class adam_optimizer:
    def __init__(self, learning_rate, beta_1=0.999, beta_2=0.99999, epsilon=1e-09):
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.learning_rate = learning_rate
        self._cache = {}

    def delta(self, grads, name="_", learning_rate=None):
        if name not in self._cache:
            self._cache[name] = [
                [torch.zeros_like(i).to(grads[0].device) for i in grads],
                [torch.zeros_like(i).to(grads[0].device) for i in grads],
                0,
            ]
        self._cache[name][2] += 1
        t = self._cache[name][2]
        deltas = []
        beta_1 = self.beta_1
        beta_2 = self.beta_2
        learning_rate = self.learning_rate if learning_rate is None else learning_rate
        for n, g in enumerate(grads):
            m = self._cache[name][0][n]
            v = self._cache[name][1][n]
            m = beta_1 * m + (1 - beta_1) * g
            v = beta_2 * v + (1 - beta_2) * (g**2)
            self._cache[name][0][n] = m
            self._cache[name][1][n] = v
            m_hat = m / (1 - np.power(beta_1, t).item())
            v_hat = v / (1 - np.power(beta_2, t).item())
            deltas.append(learning_rate * m_hat / (torch.sqrt(v_hat) + self.epsilon))

        return deltas


class FBTDSTDP(LearningRule):
    """
    Feedback-modulated TD-STDP.
    """

    def __init__(
        self,
        connection: AbstractConnection,
        nu: Optional[Union[float, Sequence[float]]] = None,
        reduction: Optional[callable] = None,
        weight_decay: float = 0.0,
        **kwargs,
    ) -> None:
        super().__init__(connection=connection, nu=nu, reduction=reduction, weight_decay=weight_decay, **kwargs)

        if isinstance(connection, (Connection, LocalConnection)):
            self.update = self._connection_update
        else:
            raise NotImplementedError("This learning rule is not supported for this Connection type.")
        self.tc_plus = torch.tensor(kwargs.get("tc_plus", 20.0))
        self.tc_minus = torch.tensor(kwargs.get("tc_minus", 20.0))
        self.tc_e_trace = torch.tensor(kwargs.get("tc_e_trace", 20.0))
        self.tc_a_trace = torch.tensor(kwargs.get("tc_a_trace", 20.0))
        self.fb_gate = kwargs.get("fb_gate", False)

        if self.fb_gate:
            self.num_actions = kwargs["num_actions"]
            assert self.target.n % self.num_actions == 0, "Number of node has to be divisible by number of actions"
            self.node_label = torch.repeat_interleave(
                torch.arange(start=0, end=self.num_actions), self.target.n // self.num_actions
            ).unsqueeze(0)
            if kwargs.get("gpu", False):
                self.node_label = self.node_label.cuda()

        self.targ_firing_r = torch.tensor(kwargs.get("targ_firing_r", 0))
        self.targ_firing_rate = torch.tensor(kwargs.get("targ_firing_rate", 0.2))
        self.inh = kwargs.get("inh", False)
        self.adam = kwargs.get("adam", False)
        adam_beta_1 = kwargs.get("adam_beta_1", 0.999)
        adam_beta_2 = kwargs.get("adam_beta_2", 0.99999)
        self.weight_reg = kwargs.get("weight_reg", 0)
        self.gpu = kwargs.get("gpu", False)

        if self.adam:
            self.optimzier = adam_optimizer(nu[1], beta_1=adam_beta_1, beta_2=adam_beta_2)

    def _connection_update(self, **kwargs) -> None:
        batch_size = self.source.batch_size
        gpu = kwargs.get("gpu", False)

        self.at_decay = torch.exp(-self.connection.dt / self.tc_a_trace)
        self.et_decay = torch.exp(-self.connection.dt / self.tc_e_trace)
        self.ltp_decay = torch.exp(-self.connection.dt / self.tc_plus)
        self.ltd_decay = torch.exp(-self.connection.dt / self.tc_minus)

        if not hasattr(self, "p_plus"):
            self.p_plus = torch.zeros(batch_size, int(np.prod(self.source.shape)))
            if gpu:
                self.p_plus = self.p_plus.cuda()
        if not hasattr(self, "p_minus"):
            self.p_minus = torch.zeros(batch_size, *self.target.shape)
            if gpu:
                self.p_minus = self.p_minus.cuda()
        if not hasattr(self, "eligibility"):
            self.eligibility = torch.zeros(batch_size, *self.connection.w.shape)
            if gpu:
                self.eligibility = self.eligibility.cuda()
        if not hasattr(self, "eligibility_trace"):
            self.eligibility_trace = torch.zeros(batch_size, *self.connection.w.shape)
            if gpu:
                self.eligibility_trace = self.eligibility_trace.cuda()
        if self.fb_gate:
            if not hasattr(self, "action_v"):
                self.action_v = torch.zeros(batch_size, *self.connection.w.shape)
                if gpu:
                    self.action_v = self.action_v.cuda()
            if not hasattr(self, "action_trace"):
                self.action_trace = torch.zeros(batch_size, *self.connection.w.shape)
                if gpu:
                    self.action_trace = self.action_trace.cuda()

        source_s = self.source.s.view(batch_size, -1).float()
        target_s = self.target.s.view(batch_size, -1).float()
        reward = kwargs["reward"][self.connection]

        if self.fb_gate:
            p = torch.repeat_interleave(kwargs["p"], self.target.n // self.num_actions, dim=-1).unsqueeze(1)
            a = (kwargs["action"].unsqueeze(-1) == self.node_label).unsqueeze(1).float()
            self.action_v = (a - p) * self.eligibility_trace
            self.action_trace *= self.at_decay
            self.action_trace += self.action_v / self.tc_a_trace
            update = reward * self.action_trace

            if kwargs.get("ent", None) is not None:
                ent = torch.repeat_interleave(kwargs["ent"], self.target.n // self.num_actions, dim=-1).unsqueeze(1)
                if self.inh:
                    update -= ent * self.eligibility_trace
                else:
                    update += ent * self.eligibility_trace
        else:
            update = reward * self.eligibility_trace

        if self.targ_firing_r != 0:
            targ_adj = (
                (self.targ_firing_rate - torch.mean(self.target.x, dim=1) / self.target.tc_trace) * self.targ_firing_r
            )
            update += targ_adj.unsqueeze(-1).unsqueeze(-1) * self.eligibility_trace

        if self.weight_reg != 0:
            update -= 0.5 * self.weight_reg * self.connection.w

        if self.adam:
            update = self.optimzier.delta([self.reduction(update, dim=0)])[0]
        else:
            update = self.reduction(update, dim=0)

        self.connection.w += update

        self.p_plus *= self.ltp_decay
        self.p_plus += self.nu[1] * source_s
        self.eligibility = torch.bmm(self.p_plus.unsqueeze(2), target_s.unsqueeze(1))

        if self.nu[0] != 0:
            self.p_minus *= self.ltd_decay
            self.p_minus += -self.nu[0] * target_s
            self.eligibility += torch.bmm(source_s.unsqueeze(2), self.p_minus.unsqueeze(1))

        self.eligibility_trace *= self.et_decay
        self.eligibility_trace += self.eligibility / self.tc_e_trace

        super().update()

    def reset_state_variables(self, index=slice(None)) -> None:
        if hasattr(self, "p_plus"):
            self.p_plus[index] = 0
        if hasattr(self, "p_minus"):
            self.p_minus[index] = 0
        if hasattr(self, "eligibility_trace"):
            self.eligibility_trace[index] = 0
        if self.fb_gate and hasattr(self, "action_trace"):
            self.action_trace[index] = 0

    def set_batch_size(self, batch_size) -> None:
        if hasattr(self, "p_plus"):
            delattr(self, "p_plus")
        if hasattr(self, "p_minus"):
            delattr(self, "p_minus")
        if hasattr(self, "eligibility"):
            delattr(self, "eligibility")
        if hasattr(self, "eligibility_trace"):
            delattr(self, "eligibility_trace")
        if hasattr(self, "action_v"):
            delattr(self, "action_v")
        if hasattr(self, "action_trace"):
            delattr(self, "action_trace")


class Input_add(Input):
    def reset_state_variables(self, index=slice(None)) -> None:
        super().reset_state_variables()


class LIFNodes_add(LIFNodes):
    def reset_state_variables(self, index=slice(None)) -> None:
        self.v[index] = self.rest
        self.refrac_count[index] = 0
        if self.traces:
            self.x[index] = 0
        if self.sum_input:
            self.summed[index] = 0


class Network_add(Network):
    def run(self, inputs: Dict[str, torch.Tensor], time: int, one_step=False, **kwargs) -> None:
        clamps = kwargs.get("clamp", {})
        unclamps = kwargs.get("unclamp", {})
        masks = kwargs.get("masks", {})
        injects_v = kwargs.get("injects_v", {})

        if inputs != {}:
            for key in inputs:
                if len(inputs[key].size()) == 1:
                    inputs[key] = inputs[key].unsqueeze(0).unsqueeze(0)
                elif len(inputs[key].size()) == 2:
                    inputs[key] = inputs[key].unsqueeze(1)

            for key in inputs:
                if inputs[key].size(1) != self.batch_size:
                    self.batch_size = inputs[key].size(1)

                    for l in self.layers:
                        self.layers[l].set_batch_size(self.batch_size)

                    for m in self.monitors:
                        self.monitors[m].reset_state_variables()

                break

        timesteps = int(time / self.dt)

        for t in range(timesteps):
            current_inputs = {}
            if not one_step:
                current_inputs.update(self._get_inputs())

            for l in self.layers:
                if l in inputs:
                    if l in current_inputs:
                        current_inputs[l] += inputs[l][t]
                    else:
                        current_inputs[l] = inputs[l][t]

                if one_step:
                    current_inputs.update(self._get_inputs(layers=[l]))

                self.layers[l].forward(x=current_inputs[l])

                clamp = clamps.get(l, None)
                if clamp is not None:
                    if clamp.ndimension() == 1:
                        self.layers[l].s[:, clamp] = 1
                    else:
                        self.layers[l].s[:, clamp[t]] = 1

                unclamp = unclamps.get(l, None)
                if unclamp is not None:
                    if unclamp.ndimension() == 1:
                        self.layers[l].s[unclamp] = 0
                    else:
                        self.layers[l].s[unclamp[t]] = 0

                inject_v = injects_v.get(l, None)
                if inject_v is not None:
                    if inject_v.ndimension() == 1:
                        self.layers[l].v += inject_v
                    else:
                        self.layers[l].v += inject_v[t]

            if self.reward_fn is not None:
                kwargs["reward"] = self.reward_fn.compute(self, t, **kwargs)

            for c in self.connections:
                self.connections[c].update(mask=masks.get(c, None), learning=self.learning, **kwargs)

            current_inputs.update(self._get_inputs())

            for m in self.monitors:
                self.monitors[m].record()

        for c in self.connections:
            self.connections[c].normalize()

    def reset_state_variables(self, index=slice(None)) -> None:
        for layer in self.layers:
            self.layers[layer].reset_state_variables(index=index)

        for connection in self.connections:
            self.connections[connection].update_rule.reset_state_variables(index=index)


def build_ac_network(
    input_shape,
    init_scale,
    num_critics,
    num_vars_exc,
    num_actors_pa,
    num_actions,
    critic_layer_param,
    critic_fd_conn_param,
    var_layer_param,
    var_fd_conn_param,
    actor_layer_param,
    actor_fd_conn_param,
    actor_active=True,
    var_head=False,
    gpu=False,
):
    network = Network_add(dt=1)
    input_layer = Input_add(n=np.prod(input_shape), shape=(np.prod(input_shape),), traces=True, tc_trace=25.0)
    network.add_layer(input_layer, name="H0")
    layers = [input_layer]
    in_n_neurons = int(np.prod(input_shape))

    if type(init_scale) != list:
        init_scale = [init_scale, init_scale]

    critic_layer_param.update({"n": num_critics[0], "shape": (num_critics[0],)})
    layers.append(LIFNodes_add(**critic_layer_param))
    network.add_layer(layers[-1], name="CE")
    fd_w = torch.rand(in_n_neurons, num_critics[0]) * init_scale[0]
    critic_fd_conn_param.update({"source": input_layer, "target": layers[-1], "w": fd_w, "inh": False, "gpu": gpu})
    critic_fd_conn = Connection(**critic_fd_conn_param)
    network.add_connection(critic_fd_conn, source="H0", target="CE")

    if num_critics[1] > 0:
        critic_layer_param.update({"n": num_critics[1], "shape": (num_critics[1],)})
        layers.append(LIFNodes_add(**critic_layer_param))
        network.add_layer(layers[-1], name="CN")
        fd_w = torch.rand(in_n_neurons, num_critics[1]) * init_scale[0]
        critic_fd_conn_param.update({"source": input_layer, "target": layers[-1], "w": fd_w, "inh": True, "gpu": gpu})
        critic_fd_conn = Connection(**critic_fd_conn_param)
        network.add_connection(critic_fd_conn, source="H0", target="CN")

    if var_head and int(num_vars_exc) > 0:
        var_layer_param.update({"n": int(num_vars_exc), "shape": (int(num_vars_exc),)})
        layers.append(LIFNodes_add(**var_layer_param))
        network.add_layer(layers[-1], name="CV")
        fd_w = torch.rand(in_n_neurons, int(num_vars_exc)) * init_scale[0]
        var_fd_conn_param.update({"source": input_layer, "target": layers[-1], "w": fd_w, "inh": False, "gpu": gpu})
        var_fd_conn = Connection(**var_fd_conn_param)
        network.add_connection(var_fd_conn, source="H0", target="CV")

    if actor_active:
        actor_layer_param.update({"n": num_actors_pa[0] * num_actions, "shape": (num_actors_pa[0] * num_actions,)})
        layers.append(LIFNodes_add(**actor_layer_param))
        network.add_layer(layers[-1], name="AE")
        fd_w = torch.rand(in_n_neurons, num_actors_pa[0] * num_actions) * init_scale[1]
        actor_fd_conn_param.update(
            {"source": input_layer, "target": layers[-1], "w": fd_w, "num_actions": num_actions, "inh": False, "gpu": gpu}
        )
        actor_fd_conn = Connection(**actor_fd_conn_param)
        network.add_connection(actor_fd_conn, source="H0", target="AE")

        if num_actors_pa[1] > 0:
            actor_layer_param.update({"n": num_actors_pa[1] * num_actions, "shape": (num_actors_pa[1] * num_actions,)})
            layers.append(LIFNodes_add(**actor_layer_param))
            network.add_layer(layers[-1], name="AN")
            fd_w = torch.rand(in_n_neurons, num_actors_pa[1] * num_actions) * init_scale[1]
            actor_fd_conn_param.update(
                {"source": input_layer, "target": layers[-1], "w": fd_w, "num_actions": num_actions, "inh": True, "gpu": gpu}
            )
            actor_fd_conn = Connection(**actor_fd_conn_param)
            network.add_connection(actor_fd_conn, source="H0", target="AN")

    if gpu:
        network.to("cuda")
    return network


class Reward_fn:
    def __init__(
        self,
        tc,
        network_steps,
        isEnd_zero,
        last_zero_n,
        batch_size,
        value_m,
        value_b,
        var_m=1.0,
        var_b=0.0,
        var_eps=1e-6,
        gpu=False,
    ):
        self.gpu = gpu
        self.isEnd_zero = isEnd_zero
        self.last_zero_n = last_zero_n
        self.batch_size = batch_size
        self.value_m = value_m
        self.value_b = value_b
        self.var_m = float(var_m)
        self.var_b = float(var_b)
        self.var_eps = float(var_eps)

        self.v_prev = torch.zeros(self.batch_size, dtype=torch.float32)
        self.tc = torch.tensor(tc).float()
        self.gamma = torch.tensor(np.exp(-1 / tc)).float()
        self.r_gamma = torch.sqrt(self.gamma)
        self.network_steps = network_steps

        if self.gpu:
            self.tc = self.tc.float()
            self.v_prev = self.v_prev.cuda()
            self.gamma = self.gamma.cuda()
            self.r_gamma = self.r_gamma.cuda()

        self.v_prev_rec = torch.zeros(network_steps, dtype=torch.float32)
        self.td_error_rec = torch.zeros(network_steps, dtype=torch.float32)
        self.reward_rec = torch.zeros(network_steps, dtype=torch.float32)
        self.isEnd_rec = torch.zeros(network_steps, dtype=torch.bool)
        self.var_rec = torch.zeros(network_steps, dtype=torch.float32)

    def compute(self, network, t, **kwargs):
        env_reward = torch.from_numpy(kwargs.get("env_reward", None)).float()
        stateCode = torch.from_numpy(kwargs["info"]["stateCode"]).float()
        truncatedEnd = torch.from_numpy(kwargs["info"]["truncatedEnd"])
        isEnd = torch.from_numpy(kwargs.get("isEnd", None))
        isEnd = isEnd & ~truncatedEnd
        critic_exc = network.layers["CE"]
        critic_inh = network.layers["CN"] if "CN" in network.layers else None

        if self.gpu:
            stateCode = stateCode.cuda()
            env_reward = env_reward.cuda()
            isEnd = isEnd.cuda()

        env_reward /= self.network_steps
        isLive = ~(stateCode == 1) if t < self.network_steps - self.last_zero_n else ~isEnd

        v = self.value_b + torch.mean(critic_exc.x, dim=1) / critic_exc.tc_trace * self.value_m
        if critic_inh is not None:
            v -= torch.mean(critic_inh.x, dim=1) / critic_inh.tc_trace * self.value_m

        td_error = torch.zeros(self.batch_size, dtype=torch.float32)
        if self.isEnd_zero:
            td_error = (self.r_gamma * env_reward + self.gamma * v * isLive.float()) - self.v_prev
        else:
            td_error = (self.r_gamma * env_reward + self.gamma * v) - self.v_prev
        td_error[stateCode >= 2] = 0.0
        self.v_prev = v

        td_error = td_error.unsqueeze(-1).unsqueeze(-1)
        rewards = {}
        rewards[network.connections[("H0", "CE")]] = td_error
        if "CN" in network.layers:
            rewards[network.connections[("H0", "CN")]] = -td_error
        if "AE" in network.layers:
            rewards[network.connections[("H0", "AE")]] = td_error
        if "AN" in network.layers:
            rewards[network.connections[("H0", "AN")]] = -td_error

        if "CV" in network.layers and ("H0", "CV") in network.connections:
            var_layer = network.layers["CV"]
            var_est = self.var_b + torch.mean(var_layer.x, dim=1) / var_layer.tc_trace * self.var_m
            var_est = torch.clamp(var_est, min=self.var_eps)
            td_scalar = td_error.squeeze(-1).squeeze(-1)
            var_reward = (td_scalar**2 - var_est).unsqueeze(-1).unsqueeze(-1)
            rewards[network.connections[("H0", "CV")]] = var_reward
            self.var_rec[t] = var_est[0].detach().cpu()

        self.td_error_rec[t] = td_error[0]
        self.v_prev_rec[t] = self.v_prev[0]
        self.reward_rec[t] = env_reward[0] if type(env_reward) == torch.Tensor else 0
        self.isEnd_rec[t] = isEnd[0]

        return rewards


class Snn_actor:
    def __init__(
        self,
        network,
        batch_size,
        state_to_spike,
        augment_state,
        env_dt,
        reward_adj,
        actor_m,
        num_actions,
        num_actors_pa,
        actor_active,
        entropy_reg,
        freeze_action,
        actor_lr_base,
        critic_lr_base,
        var_lr_base,
        use_ne_noise=True,
        use_ach_lr_mod=False,
        use_switch_lr_spike=False,
        switch_at=-1,
        switch_lr_peak=None,
        switch_lr_hold_eps=1,
        switch_lr_anneal_eps=200,
        base_noise=0.344308,
        ne_max=3.0,
        ne_k=0.2,
        ne_center=15.0,
        base_lr_mod=0.000055,
        ach_max=0.004061,
        ach_k=4.0,
        ach_center=8.0,
        td_signal_clip=20.0,
        td_fast_alpha=0.097663,
        td_slow_alpha=0.003642,
        unexp_decay=0.8,
        exp_fast_alpha=1.0,
        exp_slow_alpha=0.1,
        exp_decay=0.5,
        surprise_variance_weight=0.0,
        td_novelty_margin=0.0,
        actor_lr_decay=0.0,
        actor_lr_boost=0.0,
        actor_lr_min=None,
        actor_lr_max=None,
        stat_rec=False,
        gpu=False,
    ):
        self.network = network
        self.state_to_spike = state_to_spike
        self.augment_state = augment_state
        self.env_dt = env_dt
        self.reward_adj = reward_adj
        self.actor_m = actor_m
        self.num_actions = num_actions
        self.num_actors_exc_pa = num_actors_pa[0]
        self.num_actors_inh_pa = num_actors_pa[1]
        self.actor_active = actor_active
        self.entropy_reg = entropy_reg
        self.freeze_action = freeze_action
        self.stat_rec = stat_rec
        self.gpu = gpu

        self.actor_lr_base = float(actor_lr_base)
        self.critic_lr_base = float(critic_lr_base)
        self.var_lr_base = float(var_lr_base)

        self.use_ne_noise = bool(use_ne_noise)
        self.use_ach_lr_mod = bool(use_ach_lr_mod)
        self.use_switch_lr_spike = bool(use_switch_lr_spike)

        self.switch_at = int(switch_at)
        self.switch_lr_peak = float(self.actor_lr_base if switch_lr_peak is None else switch_lr_peak)
        self.switch_lr_hold_eps = int(switch_lr_hold_eps)
        self.switch_lr_anneal_eps = int(switch_lr_anneal_eps)

        self.base_noise = float(base_noise)
        self.ne_max = float(ne_max)
        self.ne_k = float(ne_k)
        self.ne_center = float(ne_center)

        self.base_lr_mod = float(base_lr_mod)
        self.ach_max = float(ach_max)
        self.ach_k = float(ach_k)
        self.ach_center = float(ach_center)

        self.td_signal_clip = float(td_signal_clip)
        self.td_fast_alpha = float(td_fast_alpha)
        self.td_slow_alpha = float(td_slow_alpha)
        self.unexp_decay = float(unexp_decay)
        self.exp_fast_alpha = float(exp_fast_alpha)
        self.exp_slow_alpha = float(exp_slow_alpha)
        self.exp_decay = float(exp_decay)
        self.surprise_variance_weight = float(surprise_variance_weight)
        self.td_novelty_margin = float(td_novelty_margin)

        self.actor_lr_decay = float(actor_lr_decay)
        self.actor_lr_boost = float(actor_lr_boost)
        self.actor_lr_min = float(self.actor_lr_base if actor_lr_min is None else actor_lr_min)
        self.actor_lr_max = float(self.actor_lr_base if actor_lr_max is None else actor_lr_max)
        self.actor_lr_state = float(self.actor_lr_base)

        self.td_fast = 0.0
        self.td_slow = 0.0
        self.td_trace_inited = False
        self.avg_unexpected = 0.0

        self.exp_fast = 0.0
        self.exp_slow = 0.0
        self.exp_trace_inited = False
        self.avg_expected = 0.0

        self.current_ne = float(self.base_noise)
        self.current_ach = float(self.base_lr_mod)
        self.last_metrics: Dict[str, float] = {}
        self.last_td_signal = 0.0
        self.last_td_fast = 0.0
        self.last_td_slow = 0.0
        self.last_td_novelty = 0.0

        self.p = torch.full((batch_size, num_actions), 1 / num_actions)
        self.ent_eye = torch.eye(num_actions)
        self.action = torch.distributions.Categorical(self.p).sample()
        self.freeze_timer = torch.zeros(batch_size)

        if gpu:
            self.p = self.p.cuda()
            self.ent_eye = self.ent_eye.cuda()
            self.action = self.action.cuda()
            self.freeze_timer = self.freeze_timer.cuda()

        if not actor_active:
            self.policy = uniform_policy(batch_size=batch_size, actions=[i for i in range(num_actions)])

        qlen, dlen = int(3000), int(3000 / env_dt)
        self.stat = {
            "est_v": collections.deque(maxlen=qlen),
            "td_error": collections.deque(maxlen=qlen),
            "reward": collections.deque(maxlen=qlen),
            "isEnd": collections.deque(maxlen=qlen),
            "v_exc_rate": collections.deque(maxlen=dlen),
            "ne": collections.deque(maxlen=dlen),
            "ach": collections.deque(maxlen=dlen),
            "actor_lr": collections.deque(maxlen=dlen),
            "var_est": collections.deque(maxlen=dlen),
            "abs_td": collections.deque(maxlen=dlen),
        }
        if "CN" in self.network.layers:
            self.stat.update({"v_inh_rate": collections.deque(maxlen=dlen)})

        if self.actor_active:
            self.stat.update(
                {"a_exc_rate": [collections.deque(maxlen=dlen) for _ in range(num_actions)], "p": [collections.deque(maxlen=dlen) for _ in range(num_actions)]}
            )
            if "AN" in self.network.layers:
                self.stat.update({"a_inh_rate": [collections.deque(maxlen=dlen) for _ in range(num_actions)]})

    def _set_connection_lr(self, key: Tuple[str, str], lr: float) -> None:
        if key not in self.network.connections:
            return
        rule = self.network.connections[key].update_rule
        rule.nu = (0.0, float(lr))
        if getattr(rule, "adam", False) and hasattr(rule, "optimzier"):
            rule.optimzier.learning_rate = float(lr)

    def __call__(self, state, reward, isEnd, info, episode_counts=None):
        network = self.network

        stateCode = info["stateCode"]
        if np.any(stateCode == 2):
            network.reset_state_variables(stateCode == 2)

        # Apply learning rates for weight updates happening inside `network.run`.
        actor_lr_used = self.actor_lr_base
        if self.use_switch_lr_spike and episode_counts is not None and self.switch_at >= 0:
            ep0 = int(np.asarray(episode_counts).reshape(-1)[0])
            if ep0 < self.switch_at:
                actor_lr_used = self.actor_lr_base
            else:
                t = ep0 - self.switch_at
                if t < max(self.switch_lr_hold_eps, 0):
                    actor_lr_used = self.switch_lr_peak
                else:
                    ta = t - max(self.switch_lr_hold_eps, 0)
                    if ta < max(self.switch_lr_anneal_eps, 0):
                        frac = 1.0 - (ta / max(float(self.switch_lr_anneal_eps), 1.0))
                        actor_lr_used = self.actor_lr_base + (self.switch_lr_peak - self.actor_lr_base) * frac
                    else:
                        actor_lr_used = self.actor_lr_base
            self.actor_lr_state = float(actor_lr_used)
        elif self.use_ach_lr_mod:
            actor_lr_used = self.actor_lr_state
        self._set_connection_lr(("H0", "CE"), self.critic_lr_base)
        self._set_connection_lr(("H0", "AE"), actor_lr_used)
        self._set_connection_lr(("H0", "CV"), self.var_lr_base)

        p = self.p
        ent = (
            -self.entropy_reg
            * torch.sum(((p * (torch.log(p) + 1))[:, np.newaxis, :] * (self.ent_eye - p[..., np.newaxis])), axis=-1)
            if self.entropy_reg > 0
            else None
        )
        augment = np.concatenate([isEnd[:, np.newaxis], (stateCode == 1)[:, np.newaxis]], axis=-1) if self.augment_state else None
        state_spike = self.state_to_spike(state, augment)
        if self.gpu:
            state_spike = state_spike.cuda()

        network.run(
            inputs={"H0": state_spike.reshape(state_spike.shape[:2] + (-1,))},
            time=self.env_dt,
            input_time_dim=1,
            action=self.action,
            p=self.p,
            ent=ent,
            env_reward=self.reward_adj * reward,
            info=info,
            isEnd=isEnd,
            gpu=self.gpu,
        )

        # Compute NE / ACh signals from TD error + variance head.
        td = network.reward_fn.td_error_rec.detach().float().cpu()
        abs_td = float(td.abs().mean().item())
        td2 = float((td**2).mean().item())
        var_est = float(network.reward_fn.var_rec.detach().float().cpu().mean().item()) if hasattr(network.reward_fn, "var_rec") else 0.0
        sigma = float(np.sqrt(max(var_est, 1e-6)))
        sigma = max(sigma, 1e-6)

        if not self.exp_trace_inited:
            self.exp_fast = sigma
            self.exp_slow = sigma
            self.exp_trace_inited = True
        else:
            self.exp_fast = (1.0 - self.exp_fast_alpha) * self.exp_fast + self.exp_fast_alpha * sigma
            self.exp_slow = (1.0 - self.exp_slow_alpha) * self.exp_slow + self.exp_slow_alpha * sigma
        exp_novelty = max(0.0, self.exp_fast)
        self.avg_expected = self.exp_decay * self.avg_expected + exp_novelty
        self.current_ach = float(logistic_drive(self.ach_max, self.ach_k, self.ach_center, self.avg_expected, self.base_lr_mod))

        td_signal = abs_td / (sigma ** self.surprise_variance_weight)
        td_signal = float(min(max(td_signal, 0.0), self.td_signal_clip))
        self.last_td_signal = float(td_signal)
        if not self.td_trace_inited:
            self.td_fast = td_signal
            self.td_slow = td_signal
            self.td_trace_inited = True
        else:
            self.td_fast = (1.0 - self.td_fast_alpha) * self.td_fast + self.td_fast_alpha * td_signal
            self.td_slow = (1.0 - self.td_slow_alpha) * self.td_slow + self.td_slow_alpha * td_signal
        td_novelty = max(0.0, self.td_fast - self.td_slow - self.td_novelty_margin)
        self.last_td_fast = float(self.td_fast)
        self.last_td_slow = float(self.td_slow)
        self.last_td_novelty = float(td_novelty)
        self.avg_unexpected = self.unexp_decay * self.avg_unexpected + td_novelty
        self.current_ne = float(logistic_drive(self.ne_max, self.ne_k, self.ne_center, self.avg_unexpected, self.base_noise))
        self.current_ne = float(min(self.current_ne, 5.0))

        if self.use_ach_lr_mod and (not self.use_switch_lr_spike):
            self.actor_lr_state *= (1.0 - self.actor_lr_decay)
            self.actor_lr_state += self.actor_lr_boost * self.current_ach
            self.actor_lr_state = float(min(max(self.actor_lr_state, self.actor_lr_min), self.actor_lr_max))

        # Always expose last-step metrics (episode logger uses this even when `stat_rec` is False).
        self.last_metrics = {
            "abs_td": abs_td,
            "td2": td2,
            "var_est": var_est,
            "var_err": float(td2 - var_est),
            "ne": float(self.current_ne),
            "ach": float(self.current_ach),
            "actor_lr_used": float(actor_lr_used),
            "actor_lr_state": float(self.actor_lr_state),
        }

        if self.actor_active:
            actor_exc = network.layers["AE"]
            a_exc_rate = torch.mean(actor_exc.x.view(-1, self.num_actions, self.num_actors_exc_pa), dim=-1) / actor_exc.tc_trace
            if "AN" in network.layers:
                actor_inh = network.layers["AN"]
                a_inh_rate = torch.mean(actor_inh.x.view(-1, self.num_actions, self.num_actors_inh_pa), dim=-1) / actor_inh.tc_trace
                a_rate = self.actor_m * (a_exc_rate - a_inh_rate)
            else:
                a_rate = self.actor_m * a_exc_rate

            logits = a_rate
            if self.use_ne_noise:
                logits = logits + torch.randn_like(logits) * float(self.current_ne)

            self.p = torch.nn.functional.softmax(logits, dim=1)
            self.freeze_timer += self.env_dt
            new_b = self.freeze_timer >= self.freeze_action
            if new_b.any().detach().cpu().numpy():
                self.action[new_b] = torch.distributions.Categorical(logits=logits[new_b]).sample()
                self.freeze_timer[new_b] = 0
        else:
            self.action = torch.from_numpy(self.policy(state))

        if self.stat_rec:
            self.stat["est_v"].extend(network.reward_fn.v_prev_rec.detach().cpu().numpy().tolist())
            self.stat["td_error"].extend(network.reward_fn.td_error_rec.detach().cpu().numpy().tolist())
            self.stat["reward"].extend(network.reward_fn.reward_rec.detach().cpu().numpy().tolist())
            self.stat["isEnd"].extend(network.reward_fn.isEnd_rec.detach().cpu().numpy().tolist())
            v_exc_rate_0 = torch.mean(network.layers["CE"].x[0], dim=-1) / network.layers["CE"].tc_trace
            self.stat["v_exc_rate"].append(v_exc_rate_0.detach().cpu().numpy())
            if "CN" in network.layers:
                v_inh_rate_0 = torch.mean(network.layers["CN"].x[0], dim=-1) / network.layers["CN"].tc_trace
                self.stat["v_inh_rate"].append(v_inh_rate_0.detach().cpu().numpy())
            if self.actor_active:
                for i in range(self.num_actions):
                    self.stat["a_exc_rate"][i].append(a_exc_rate[0, i].detach().cpu().numpy())
                    if "AN" in network.layers:
                        self.stat["a_inh_rate"][i].append(a_inh_rate[0, i].detach().cpu().numpy())
                    self.stat["p"][i].append(self.p[0, i].detach().cpu().numpy())

            self.stat["ne"].append(self.current_ne)
            self.stat["ach"].append(self.current_ach)
            self.stat["actor_lr"].append(self.actor_lr_state)
            self.stat["var_est"].append(var_est)
            self.stat["abs_td"].append(abs_td)

        return (self.action.detach().cpu().numpy(), self.stat)


class Stat_plotter:
    def __init__(self, len_adj):
        self.pers, self.axs = [None] * 5, [None] * 5
        self.len_adj = len_adj

    def __call__(self, stat, eps_ret, eps_len):
        d = {"Return of Episode": np.array(eps_ret)}
        if len(eps_ret) >= 100:
            d.update({"Trailing avg return of Episode": np.concatenate([np.full((99,), np.nan), mv(eps_ret, 100)])})
        self.pers[0], self.axs[0] = plot_performance(
            d, title="Learning Curve", xlabel="Episodes", ylabel="Return of Episode", fig=self.pers[0], ax=self.axs[0]
        )
        d = {"Length of Episode": np.array(eps_len) + self.len_adj}
        if len(eps_ret) >= 100:
            d.update(
                {"Trailing avg length of Episode": np.concatenate([np.full((99,), np.nan), mv(eps_len, 100)]) + self.len_adj}
            )
        self.pers[1], self.axs[1] = plot_performance(
            d, title="Learning Curve", xlabel="Episodes", ylabel="Lengths (Seconds)", fig=self.pers[1], ax=self.axs[1]
        )

        d = {"Exc Value Neuron": stat["v_exc_rate"]}
        if "v_inh_rate" in stat:
            d.update({"Inh Value Neuron": stat["v_inh_rate"]})
        if "a_exc_rate" in stat:
            for i in range(len(stat["a_exc_rate"])):
                d.update({f"Exc Actor {i} Neuron": stat["a_exc_rate"][i]})
        if "a_inh_rate" in stat:
            for i in range(len(stat["a_inh_rate"])):
                d.update({f"Inh Actor {i} Neuron": stat["a_inh_rate"][i]})
        self.pers[2], self.axs[2] = plot_performance(
            d, title="Neuron firing rate", xlabel="Time", ylabel="Firing Rate", fig=self.pers[2], ax=self.axs[2], ylim=(0, 1)
        )
        if "p" in stat:
            d = {f"Action {i}": stat["p"][i] for i in range(len(stat["p"]))}
            self.pers[3], self.axs[3] = plot_performance(
                d, title="Action probability", xlabel="Time", ylabel="Prob.", fig=self.pers[3], ax=self.axs[3], ylim=(0, 1)
            )
        d = {"TD error": stat["td_error"]}
        self.pers[4], self.axs[4] = plot_performance(
            d, title="TD error", xlabel="Time", ylabel="Error", fig=self.pers[4], ax=self.axs[4]
        )


# ---------------------------------------------------------------------------
# Main training/eval driver (from `td_stdp/main.py`, copied and path-adjusted)
# ---------------------------------------------------------------------------


def _resolve_config_path(cfg_arg: str) -> str:
    if os.path.sep in cfg_arg or (os.path.altsep and os.path.altsep in cfg_arg):
        return cfg_arg
    repo_root = os.path.dirname(os.path.dirname(__file__))
    return os.path.join(repo_root, "td_stdp", "config", cfg_arg)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", required=False, default="config_cp_400_1run.ini", help="config file name or path")
    args = ap.parse_args()

    cfg_path = _resolve_config_path(args.config)
    print(f"Loading config from {cfg_path}")

    config = configparser.ConfigParser(inline_comment_prefixes="#")
    config.read(cfg_path)

    gpu = config.getboolean("USER", "gpu")
    gpu_requested = bool(gpu)
    gpu = bool(gpu and torch.cuda.is_available())
    if gpu_requested and not gpu:
        print("CUDA not available; falling back to CPU (set `gpu = False` in config to silence this).")

    min_eps = config.getint("USER", "min_eps")
    max_eps = config.getint("USER", "max_eps")
    n_run = config.getint("USER", "n_run")

    batch_size = config.getint("USER", "batch_size")
    input_type = config.getint("USER", "input_type")
    augment_state = config.getboolean("USER", "augment_state")
    rep = config.getint("USER", "rep")
    basis = config.getboolean("USER", "basis")

    bin_num = config.getint("USER", "bin_num")

    forier_order = config.getint("USER", "forier_order")
    forier_cross_term = config.getboolean("USER", "forier_cross_term")
    forier_double = config.getboolean("USER", "forier_double")

    critic_lr = config.getfloat("USER", "critic_lr")
    actor_lr = config.getfloat("USER", "actor_lr")
    init_scale = config.getfloat("USER", "init_scale")
    adam = config.getboolean("USER", "adam")
    adam_beta_1 = config.getfloat("USER", "adam_beta_1")
    adam_beta_2 = config.getfloat("USER", "adam_beta_2")

    env_name = config.get("USER", "env_name")
    env_tc = config.getfloat("USER", "env_tc")
    env_dt = config.getint("USER", "env_dt")
    reward_adj = config.getfloat("USER", "reward_adj")
    freeze_action = config.getint("USER", "freeze_action")
    isEnd_zero = config.getboolean("USER", "isEnd_zero")
    last_zero_n = config.getint("USER", "last_zero_n")
    rest_n = config.getint("USER", "rest_n")
    warm_n = config.getint("USER", "warm_n")

    actor_active = config.getboolean("USER", "actor_active")
    num_critics_exc = config.getint("USER", "num_critics_exc")
    num_critics_inh = config.getint("USER", "num_critics_inh")
    num_actors_exc_pa = config.getint("USER", "num_actors_exc_pa")
    num_actors_inh_pa = config.getint("USER", "num_actors_inh_pa")
    entropy_reg = config.getfloat("USER", "entropy_reg")
    weight_reg = config.getfloat("USER", "weight_reg")
    value_m = config.getfloat("USER", "value_m")
    value_b = config.getfloat("USER", "value_b")
    actor_m = config.getfloat("USER", "actor_m")
    tau_n = config.getfloat("USER", "tau_n")
    tau_plus = config.getfloat("USER", "tau_plus")
    tau_z = config.getfloat("USER", "tau_z")
    tau_q = config.getfloat("USER", "tau_q")
    tau_v = config.getfloat("USER", "tau_v")
    targ_firing_r = config.getfloat("USER", "targ_firing_r")
    targ_firing_rate = config.getfloat("USER", "targ_firing_rate")

    # Variance head (critic uncertainty) + neuromodulation (optional; defaults chosen to be safe).
    var_head = config.getboolean("USER", "var_head", fallback=DEFAULT_VAR_HEAD)
    num_vars_exc = config.getint(
        "USER",
        "num_vars_exc",
        fallback=(num_critics_exc if DEFAULT_NUM_VARS_EXC is None else int(DEFAULT_NUM_VARS_EXC)),
    )
    var_m = config.getfloat("USER", "var_m", fallback=DEFAULT_VAR_M)
    var_b = config.getfloat("USER", "var_b", fallback=DEFAULT_VAR_B)
    var_eps = config.getfloat("USER", "var_eps", fallback=DEFAULT_VAR_EPS)
    var_lr = config.getfloat("USER", "var_lr", fallback=(DEFAULT_VAR_LR_SCALE * critic_lr))

    use_ne_noise = config.getboolean("USER", "use_ne_noise", fallback=DEFAULT_USE_NE_NOISE)
    use_ach_lr_mod = config.getboolean("USER", "use_ach_lr_mod", fallback=DEFAULT_USE_ACH_LR_MOD)
    use_switch_lr_spike = config.getboolean("USER", "use_switch_lr_spike", fallback=DEFAULT_USE_SWITCH_LR_SPIKE)

    base_noise = config.getfloat("USER", "base_noise", fallback=DEFAULT_BASE_NOISE)
    ne_max = config.getfloat("USER", "ne_max", fallback=DEFAULT_NE_MAX)
    ne_k = config.getfloat("USER", "ne_k", fallback=DEFAULT_NE_K)
    ne_center = config.getfloat("USER", "ne_center", fallback=DEFAULT_NE_CENTER)

    base_lr_mod = config.getfloat("USER", "base_lr_mod", fallback=DEFAULT_BASE_LR_MOD)
    ach_max = config.getfloat("USER", "ach_max", fallback=DEFAULT_ACH_MAX)
    ach_k = config.getfloat("USER", "ach_k", fallback=DEFAULT_ACH_K)
    ach_center = config.getfloat("USER", "ach_center", fallback=DEFAULT_ACH_CENTER)

    td_signal_clip = config.getfloat("USER", "td_signal_clip", fallback=DEFAULT_TD_SIGNAL_CLIP)
    td_fast_alpha = config.getfloat("USER", "td_fast_alpha", fallback=DEFAULT_TD_FAST_ALPHA)
    td_slow_alpha = config.getfloat("USER", "td_slow_alpha", fallback=DEFAULT_TD_SLOW_ALPHA)
    unexp_decay = config.getfloat("USER", "unexp_decay", fallback=DEFAULT_UNEXP_DECAY)
    exp_fast_alpha = config.getfloat("USER", "exp_fast_alpha", fallback=DEFAULT_EXP_FAST_ALPHA)
    exp_slow_alpha = config.getfloat("USER", "exp_slow_alpha", fallback=DEFAULT_EXP_SLOW_ALPHA)
    exp_decay = config.getfloat("USER", "exp_decay", fallback=DEFAULT_EXP_DECAY)
    surprise_variance_weight = config.getfloat("USER", "surprise_variance_weight", fallback=DEFAULT_SURPRISE_VARIANCE_WEIGHT)
    td_novelty_margin = config.getfloat("USER", "td_novelty_margin", fallback=DEFAULT_TD_NOVELTY_MARGIN)

    actor_lr_decay = config.getfloat("USER", "actor_lr_decay", fallback=DEFAULT_ACTOR_LR_DECAY)
    actor_lr_boost = config.getfloat("USER", "actor_lr_boost", fallback=DEFAULT_ACTOR_LR_BOOST)
    actor_lr_min = config.getfloat("USER", "actor_lr_min", fallback=(actor_lr if DEFAULT_ACTOR_LR_MIN is None else DEFAULT_ACTOR_LR_MIN))
    actor_lr_max = config.getfloat("USER", "actor_lr_max", fallback=(actor_lr if DEFAULT_ACTOR_LR_MAX is None else DEFAULT_ACTOR_LR_MAX))

    switch_lr_peak = config.getfloat("USER", "switch_lr_peak", fallback=DEFAULT_SWITCH_LR_PEAK)
    switch_lr_hold_eps = config.getint("USER", "switch_lr_hold_eps", fallback=DEFAULT_SWITCH_LR_HOLD_EPS)
    switch_lr_anneal_eps = config.getint("USER", "switch_lr_anneal_eps", fallback=DEFAULT_SWITCH_LR_ANNEAL_EPS)

    # Flagship-style logging/output (CSV + PNG under cartpole/runs/*).
    flagship_logs = config.getboolean("USER", "flagship_logs", fallback=DEFAULT_FLAGSHIP_LOGS)
    print_every = config.getint("USER", "print_every", fallback=DEFAULT_PRINT_EVERY)
    out_basedir = config.get("USER", "out_basedir", fallback=DEFAULT_OUT_BASEDIR)
    switch_at = config.getint("USER", "switch_at", fallback=DEFAULT_SWITCH_AT)

    name = config.get("USER", "name")
    test = config.getboolean("USER", "test")
    checkpoint = config.get("USER", "checkpoint")
    test_eps = config.getint("USER", "test_eps")
    test_vis = config.getboolean("USER", "test_vis")

    results = []
    plot_stat = False
    p_eps_n = 1 if test else 10
    batch_size = 1 if test and test_vis else batch_size

    state_lim = {
        "CartPole-v1": [np.array([-2.4, -3.2, -np.pi / 12.0, -3.2]), np.array([2.4, 3.2, np.pi / 12.0, 3.2])],
        "LunarLander-v2": [np.array([-1, -0.2, -1, -1, -1, -1, 0, 0]), np.array([+1, +2, +1, +1, +1, +1, 1, 1])],
    }

    solve_def = {"CartPole-v1": (100, 500), "LunarLander-v2": (100, 200)}

    env = batch_envs(name=env_name, batch_size=batch_size, rest_n=rest_n, warm_n=warm_n)
    num_actions = env.action_space.n
    bin_min, bin_max = state_lim[env_name]
    switch_enabled = bool((env_name == "CartPole-v1") and (int(switch_at) >= 0))

    if input_type == 0:
        state_to_spike = State_to_spike_bin(env_dt, bin_min, bin_max, bin_num, basis=basis, rep=rep)
    elif input_type == 1:
        state_to_spike = State_to_spike_RBF(env_dt, bin_min, bin_max, bin_num, basis=basis, rep=rep)
    elif input_type == 2:
        state_to_spike = State_to_spike_Fourier(
            env_dt, bin_min, bin_max, k=forier_order, basis=basis, soft=False, cross_term=forier_cross_term, double=forier_double, rep=rep
        )
    elif input_type == 3:
        if env_name != "CartPole-v1":
            raise ValueError(f"Place-cell input_type=3 is only implemented for CartPole-v1 (got {env_name})")
        state_to_spike = State_to_spike_PlaceCell_CartPole(env_dt, basis=basis, rep=rep)
    else:
        raise ValueError(f"Unknown input_type: {input_type}")

    input_shape = state_to_spike.out_shape
    if augment_state:
        input_shape = (rep, input_shape[1] + 2)
    print("Input shape:", input_shape)

    critic_layer_param = {"traces": True, "traces_additive": True, "tc_trace": tau_n, "refrac": 0, "tc_decay": tau_v}
    actor_layer_param = critic_layer_param.copy()
    actor_layer_param.update({"tc_trace": tau_plus})
    var_layer_param = critic_layer_param.copy()
    var_layer_param.update({"tc_trace": tau_n})

    critic_fd_conn_param = {
        "update_rule": FBTDSTDP,
        "nu": (0, critic_lr),
        "wmin": -np.inf,
        "wmax": np.inf,
        "norm": None,
        "tc_plus": tau_plus,
        "tc_e_trace": tau_z,
        "fb_gate": False,
        "adam": adam,
        "adam_beta_1": adam_beta_1,
        "adam_beta_2": adam_beta_2,
        "weight_reg": weight_reg,
    }
    var_fd_conn_param = critic_fd_conn_param.copy()
    var_fd_conn_param.update(
        {
            "update_rule": FBTDSTDP,
            "nu": (0, var_lr),
            "fb_gate": False,
            "adam": adam,
            "adam_beta_1": adam_beta_1,
            "adam_beta_2": adam_beta_2,
            "weight_reg": weight_reg,
        }
    )
    actor_fd_conn_param = critic_fd_conn_param.copy()
    actor_fd_conn_param.update(
        {
            "update_rule": FBTDSTDP,
            "nu": (0, actor_lr),
            "tc_plus": tau_plus,
            "tc_e_trace": tau_z,
            "tc_a_trace": tau_q,
            "fb_gate": True,
            "targ_firing_r": targ_firing_r,
            "targ_firing_rate": targ_firing_rate,
            "adam": adam,
        }
    )

    stat_plotter = Stat_plotter(len_adj=-(rest_n - warm_n) * env_dt)

    repo_root = os.path.dirname(os.path.dirname(__file__))
    td_root = os.path.join(repo_root, "td_stdp")
    result_dir = os.path.join(td_root, "result")
    model_dir = os.path.join(td_root, "model")
    os.makedirs(result_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    for n in range(1 if test else n_run):
        env.reset()
        if test:
            print(f"Loading checkpoint from {checkpoint}..")
            f_name = os.path.join(model_dir, checkpoint)
            try:
                if hasattr(torch, "serialization") and hasattr(torch.serialization, "add_safe_globals"):
                    torch.serialization.add_safe_globals([Network_add])
            except Exception:
                pass
            with open(f_name, "rb") as f:
                try:
                    network = torch.load(f, map_location="cpu", weights_only=False)
                except TypeError:
                    network = torch.load(f, map_location="cpu")
            if gpu:
                network.to("cuda")
            network.learning = False
        else:
            network = build_ac_network(
                input_shape=input_shape,
                init_scale=init_scale,
                num_critics=(num_critics_exc, num_critics_inh),
                num_vars_exc=num_vars_exc,
                num_actors_pa=(num_actors_exc_pa, num_actors_inh_pa),
                num_actions=num_actions,
                critic_layer_param=critic_layer_param,
                critic_fd_conn_param=critic_fd_conn_param,
                var_layer_param=var_layer_param,
                var_fd_conn_param=var_fd_conn_param,
                actor_layer_param=actor_layer_param,
                actor_fd_conn_param=actor_fd_conn_param,
                actor_active=actor_active,
                var_head=var_head,
                gpu=gpu,
            )
            print(f"Initializing {name} network..")

        for _k, v in network.layers.items():
            v.set_batch_size(batch_size)
        for _k, v in network.connections.items():
            v.update_rule.set_batch_size(batch_size)

        network.reward_fn = Reward_fn(
            tc=env_tc,
            network_steps=env_dt,
            isEnd_zero=isEnd_zero,
            last_zero_n=last_zero_n,
            batch_size=batch_size,
            value_m=value_m,
            value_b=value_b,
            var_m=var_m,
            var_b=var_b,
            var_eps=var_eps,
            gpu=gpu,
        )

        snn_actor = Snn_actor(
            network=network,
            batch_size=batch_size,
            state_to_spike=state_to_spike,
            augment_state=augment_state,
            env_dt=env_dt,
            reward_adj=reward_adj,
            actor_m=actor_m,
            num_actions=num_actions,
            num_actors_pa=(num_actors_exc_pa, num_actors_inh_pa),
            actor_active=actor_active,
            entropy_reg=entropy_reg,
            freeze_action=freeze_action,
            actor_lr_base=actor_lr,
            critic_lr_base=critic_lr,
            var_lr_base=var_lr,
            use_ne_noise=use_ne_noise,
            use_ach_lr_mod=use_ach_lr_mod,
            use_switch_lr_spike=use_switch_lr_spike,
            switch_at=switch_at,
            switch_lr_peak=switch_lr_peak,
            switch_lr_hold_eps=switch_lr_hold_eps,
            switch_lr_anneal_eps=switch_lr_anneal_eps,
            base_noise=base_noise,
            ne_max=ne_max,
            ne_k=ne_k,
            ne_center=ne_center,
            base_lr_mod=base_lr_mod,
            ach_max=ach_max,
            ach_k=ach_k,
            ach_center=ach_center,
            td_signal_clip=td_signal_clip,
            td_fast_alpha=td_fast_alpha,
            td_slow_alpha=td_slow_alpha,
            unexp_decay=unexp_decay,
            exp_fast_alpha=exp_fast_alpha,
            exp_slow_alpha=exp_slow_alpha,
            exp_decay=exp_decay,
            surprise_variance_weight=surprise_variance_weight,
            td_novelty_margin=td_novelty_margin,
            actor_lr_decay=actor_lr_decay,
            actor_lr_boost=actor_lr_boost,
            actor_lr_min=actor_lr_min,
            actor_lr_max=actor_lr_max,
            stat_rec=plot_stat,
            gpu=gpu,
        )

        # Flagship-style per-episode logs (single-env only).
        log_enabled = bool(flagship_logs and (batch_size == 1) and (not test))
        suppress_legacy_prints = bool(flagship_logs and (batch_size == 1) and (not test))
        if flagship_logs and batch_size != 1 and (not test):
            print("Note: `flagship_logs=True` only logs env[0]; set `batch_size=1` for clean per-episode CSVs.")

        reward_history: List[float] = []
        unexpected_history: List[float] = []
        expected_history: List[float] = []
        variance_history: List[float] = []
        td2_history: List[float] = []
        delta_var_history: List[float] = []
        actor_lr_history: List[float] = []
        td_history: List[float] = []

        ep_abs_td_sum = 0.0
        ep_td2_sum = 0.0
        ep_var_sum = 0.0
        ep_var_err_sum = 0.0
        ep_actor_lr_sum = 0.0
        ep_step_count = 0
        ep_idx = 0

        eps_ret, eps_len = [], []
        c_eps_ret = np.zeros(batch_size)
        c_eps_len = np.zeros(batch_size)
        ep_counts = np.zeros(batch_size, dtype=np.int64)
        step, p_eps = 0, 0
        f_perfect, solved = False, False

        state = env.reset()
        reward = env.reward
        isEnd = env.isEnd
        info = env.info

        while True:
            step += 1
            action, stat = snn_actor(state, reward, isEnd, info, episode_counts=ep_counts)

            if log_enabled and snn_actor.last_metrics:
                ep_abs_td_sum += float(snn_actor.last_metrics.get("abs_td", 0.0))
                ep_td2_sum += float(snn_actor.last_metrics.get("td2", 0.0))
                ep_var_sum += float(snn_actor.last_metrics.get("var_est", 0.0))
                ep_var_err_sum += abs(float(snn_actor.last_metrics.get("var_err", 0.0)))
                ep_actor_lr_sum += float(snn_actor.last_metrics.get("actor_lr_used", actor_lr))
                ep_step_count += 1

            action_arr = np.asarray(action)
            if action_arr.shape == ():
                action_arr = action_arr.reshape(1)

            if switch_enabled:
                inverted_mask = ep_counts >= int(switch_at)
                if np.any(inverted_mask):
                    real_action = action_arr.copy()
                    real_action[inverted_mask] = 1 - real_action[inverted_mask]
                else:
                    real_action = action_arr
            else:
                real_action = action_arr

            state, reward, _isEnd, info = env.step(real_action)
            stateCode = info["stateCode"]

            c_eps_ret += reward
            c_eps_len += env_dt
            new_end = np.logical_and(isEnd == False, _isEnd == True)

            if np.any(new_end):
                if log_enabled and bool(new_end[0]):
                    ep_idx += 1
                    total_reward = float(c_eps_ret[0])
                    reward_history.append(total_reward)

                    mean_abs_td = (ep_abs_td_sum / ep_step_count) if ep_step_count > 0 else 0.0
                    mean_td2 = (ep_td2_sum / ep_step_count) if ep_step_count > 0 else 0.0
                    mean_var = (ep_var_sum / ep_step_count) if ep_step_count > 0 else 0.0
                    mean_delta_var = (var_lr * (ep_var_err_sum / ep_step_count)) if ep_step_count > 0 else 0.0
                    mean_actor_lr = (ep_actor_lr_sum / ep_step_count) if ep_step_count > 0 else actor_lr

                    variance_history.append(mean_var)
                    td2_history.append(mean_td2)
                    delta_var_history.append(mean_delta_var)
                    actor_lr_history.append(mean_actor_lr)
                    td_history.append(mean_abs_td)
                    unexpected_history.append(float(snn_actor.avg_unexpected))
                    expected_history.append(float(snn_actor.avg_expected))

                    if ep_idx % max(int(print_every), 1) == 0:
                        avg_r = float(np.mean(reward_history[-20:])) if reward_history else 0.0
                        status = "INVERTED" if (switch_enabled and (int(ep_counts[0]) >= int(switch_at))) else "NORMAL"
                        print(
                            f"Ep {ep_idx:4d} | {status} | R: {total_reward:3.0f} | Avg: {avg_r:4.1f} | "
                            f"NE: {snn_actor.current_ne:.2f} | ACh: {snn_actor.current_ach:.4f} | LR: {mean_actor_lr:.6f} | "
                            f"TDsig: {snn_actor.last_td_signal:.2f} | Unexpected: {snn_actor.avg_unexpected:.2f} | Expected: {snn_actor.avg_expected:.2f}"
                        )

                    ep_abs_td_sum = 0.0
                    ep_td2_sum = 0.0
                    ep_var_sum = 0.0
                    ep_var_err_sum = 0.0
                    ep_actor_lr_sum = 0.0
                    ep_step_count = 0

                ep_counts[new_end] += 1
                eps_ret.extend(c_eps_ret[new_end].tolist())
                eps_len.extend(c_eps_len[new_end].tolist())
                c_eps_ret[new_end] = 0.0
                c_eps_len[new_end] = 0.0

                while len(eps_ret) >= p_eps + p_eps_n:
                    p_eps += p_eps_n
                    if not log_enabled:
                        print(
                            "%d: Return of Episode: %.2f; Last 100 Avg. Return of Episode: %.2f; Last 100 Avg. Length of Episode: %.2f Solved: %s"
                            % (
                                p_eps,
                                eps_ret[p_eps - 1],
                                np.average(eps_ret[p_eps - 100 : p_eps]),
                                np.average(eps_len[p_eps - 100 : p_eps]),
                                "Y" if solved else "N",
                            )
                        )

                if not test and env_name in solve_def:
                    avg_n = solve_def[env_name][0]
                    p_score = solve_def[env_name][1]
                    if not f_perfect and np.amax(eps_ret) >= p_score:
                        r = np.argmax(np.array(eps_ret) >= p_score, axis=-1) + 1
                        print("%d: First perfect. Eps required: %d" % (n, r))
                        f_perfect = True
                    if not solved and len(eps_ret) > avg_n and np.amax(mv(eps_ret, avg_n)) >= p_score:
                        r = np.argmax(np.array(mv(eps_ret, avg_n)) >= p_score, axis=-1) + avg_n - 1
                        solved = True
                        f_name = os.path.join(model_dir, f"model_{name}_{n}.pt")
                        print("%d: Solved. Eps required: %d. Model saved to %s" % (n, r, f_name))
                        network.save(f_name)

                if test:
                    if len(eps_ret) >= test_eps:
                        break
                else:
                    if (solved and len(eps_ret) >= min_eps) or len(eps_ret) >= max_eps:
                        break

            isEnd = np.copy(_isEnd)
            if step % 50 == 0 and plot_stat:
                stat_plotter(stat, eps_ret, eps_len)

        results.append([eps_ret, eps_len])
        if not suppress_legacy_prints:
            print("Average return : %f (%d episodes)" % (np.average(eps_ret), len(eps_ret)))

        if log_enabled and reward_history:
            run_tag = f"{n}_{name}_spike_bind"
            out_dir = os.path.join(out_basedir, run_tag)
            os.makedirs(out_dir, exist_ok=True)

            png_path = os.path.join(out_dir, f"{run_tag}.png")
            csv_path = os.path.join(out_dir, f"{run_tag}.csv")

            with open(csv_path, "w") as fh:
                fh.write(
                    "episode,reward,unexpected_uncertainty,expected_uncertainty,mean_variance,mean_td_error_sq,mean_delta_var,mean_actor_lr,mean_abs_td\n"
                )
                for i in range(len(reward_history)):
                    fh.write(
                        f"{i},{reward_history[i]},{unexpected_history[i]},{expected_history[i]},{variance_history[i]},"
                        f"{td2_history[i]},{delta_var_history[i]},{actor_lr_history[i]},{td_history[i]}\n"
                    )

            fig, ax1 = plt.subplots(figsize=(10, 6))
            ax1.plot(reward_history, label="Reward", color="tab:blue")
            ax1.set_xlabel("Episode")
            ax1.set_ylabel("Reward", color="tab:blue")
            ax1.tick_params(axis="y", labelcolor="tab:blue")
            ax1.grid(True, alpha=0.3)

            ax2 = ax1.twinx()
            ax2.plot(unexpected_history, label="Unexpected (NE drive)", color="tab:orange", alpha=0.6)
            ax2.plot(expected_history, label="Expected (ACh drive)", color="tab:green", alpha=0.6)
            ax2.plot(actor_lr_history, label="Actor LR", color="tab:red", alpha=0.6)
            ax2.set_ylabel("Modulators", color="tab:gray")
            ax2.tick_params(axis="y", labelcolor="tab:gray")

            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
            fig.tight_layout()
            fig.savefig(png_path, dpi=150, bbox_inches="tight")
            print(f"\nPlot saved as '{png_path}' and CSV saved as '{csv_path}'")

        if not test:
            f_name = os.path.join(result_dir, f"rewards_{name}_{n}.pkl")
            with open(f_name, "wb") as f:
                pickle.dump(results[-1], f)

    if not test:
        f_name = os.path.join(result_dir, f"rewards_{name}_all.pkl")
        with open(f_name, "wb") as f:
            pickle.dump(results, f)

    if (not suppress_legacy_prints) and (not test) and env_name in solve_def:
        print("Stat on t_f (first eps. having perfect score) and t_s (eps. required to solve):")
        avg_n = solve_def[env_name][0]
        p_score = solve_def[env_name][1]
        t_f, t_s = [], []
        for i in results:
            rets = np.array(i[0])
            if (rets >= p_score).any():
                t_f.append(np.argmax(rets >= p_score, axis=-1) + 1)
            else:
                t_f.append(-1)
            if len(rets) >= avg_n and (mv(rets, avg_n) >= p_score).any():
                t_s.append(np.argmax(mv(rets, avg_n) >= p_score, axis=-1) + avg_n)
            else:
                t_s.append(-1)

        def print_t(lbl, t):
            t = np.array(t)
            print("%s: %d / %d achieved." % (lbl, np.sum(t > -1), len(t)))
            t_filter = t[t > -1]
            if len(t_filter) > 0:
                print(
                    "%s: avg. %.2f median %.2f min %.2f max %.2f std %.2f"
                    % (lbl, np.average(t_filter), np.median(t_filter), np.amin(t_filter), np.amax(t_filter), np.std(t_filter))
                )
            print("%s: " % lbl, t)

        print_t("t_f", t_f)
        print_t("t_s", t_s)

    r = np.array([i[0][:min_eps] for i in results])
    r_avg = np.average(r, axis=0)
    r_std = np.std(r, axis=0)
    vis_len, sel = r.shape[1], 1 if r.shape[1] < 500 else 5
    ind = (np.arange(vis_len) % sel) == 0
    plt.figure(figsize=(8, 4))
    x = np.arange(vis_len)[ind]
    plt.plot(x, r_avg[ind], label="average reward")
    plt.fill_between(x, (r_avg - r_std)[ind], (r_avg + r_std)[ind], alpha=0.2)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title(name)
    f_name = os.path.join(result_dir, f"fig_{name}_{'test' if test else 'train'}.png")
    plt.savefig(f_name)
    if not suppress_legacy_prints:
        print(f"Plot saved to {f_name}")
        plt.show()


if __name__ == "__main__":
    main()
