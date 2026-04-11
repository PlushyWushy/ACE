import argparse
import collections
import os
import pickle
import sys
from abc import ABC
from typing import Dict, List, Optional, Sequence, Union, Iterable, Tuple

import gym
import matplotlib.lines as mlines
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch
from bindsnet.learning import LearningRule
from bindsnet.network import Network
from bindsnet.network.topology import AbstractConnection, Connection, Conv2dConnection, LocalConnection
from bindsnet.network.nodes import Input, LIFNodes, AdaptiveLIFNodes
from matplotlib.axes import Axes
from matplotlib.collections import PathCollection
from matplotlib.figure import Figure
from matplotlib.image import AxesImage
from mpl_toolkits.axes_grid1 import make_axes_locatable
from torch import Tensor
from torch.nn.modules.utils import _pair

# ==========================================
# 1. Gym Helpers for v26+ API Compatibility
# ==========================================

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

# ==========================================
# 2. Utilities
# ==========================================

def relu(x):
    y = np.copy(x)
    y[y < 0] = 0
    return y

def mv(a, n=1000):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def logistic_drive(max_val: float, k: float, center: float, signal: float, base: float) -> float:
    z = k * (signal - center)
    if z >= 700:
        return max_val + base
    if z <= -700:
        return base
    import math
    return max_val / (1.0 + math.exp(-z)) + base

def plot_performance(
    performances: Dict[str, List[float]],    
    fig: Optional[Figure] = None,
    ax: Optional[Axes] = None,
    figsize: Tuple[int, int] = (8, 6),
    title: str = "Estimated classification accuracy",
    xlabel: str = "No. of examples",
    ylabel: str = "Accuracy",
    ylim: Tuple[int, int] = None,
) -> Axes:
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)         
        box = ax.get_position()
        ax.set_position([box.x0, box.y0 + box.height * 0.1,
                       box.width, box.height * 0.9])      
    
    ax.clear()
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if ylim is not None:
        ax.set_ylim(ylim)

    for scheme in performances:
        ax.plot(
            range(len(performances[scheme])),
            [p for p in performances[scheme]],
            label=scheme,
        )

    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1),
                fancybox=True, shadow=False, ncol=4)
    fig.canvas.draw()
    fig.show()
    return fig, ax

# ==========================================
# 3. State to Spike Encoders
# ==========================================

class State_to_spike_Fourier():  
    def __init__(self, time, bin_min, bin_max, k=2, basis=False, soft=False, cross_term=True, double=False, rep=1):
        n = len(bin_min)
        if type(k) != np.ndarray:
            if type(k) == list:
                k = np.array(k)
            else:
                k = np.array([k]*n)
        if cross_term:
            self._c = np.zeros((n, np.prod(k+1)))
            for i in range(np.prod(k+1)):
                l = i
                for j in range(n):    
                    self._c[j, i] = l % (k[j]+1)
                    l //= (k[j]+1)
                    if l == 0: break
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
        self.out_shape = (rep, (self._c.shape[-1])*(2 if double else 1)+(1 if basis else 0))
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
            f_basis = (np.cos(np.pi * np.dot(norm_state, self._c))+1)/2 
        if self.basis: f_basis = np.concatenate([f_basis, np.ones((state.shape[0], 1))], axis=-1)
        if augment is not None: f_basis = np.concatenate([f_basis, augment], axis=-1)        
          
        if self.soft: 
            f_output = np.broadcast_to(f_basis[np.newaxis, np.newaxis, :, :], (self.time, self.rep) + f_basis.shape)      
        else:
            f_output = np.random.binomial(n=1,p=f_basis, size=(self.time, self.rep,)+f_basis.shape)      
        f_output = np.swapaxes(f_output, 1, 2).reshape(self.time, state.shape[0], self.rep, -1) 
        return torch.from_numpy(f_output).float()     

class uniform_policy():
    def __init__(self, batch_size, actions):
        self.actions = np.array(actions, dtype=int)
        self.batch_size = batch_size
      
    def __call__(self, state):
        return np.random.choice(self.actions, size=self.batch_size)  

# ==========================================
# 4. Environment Wrapper
# ==========================================

class batch_envs():
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
    def name(self): return self._name
    @property
    def batch_size(self): return self._batch_size
    @property
    def reward(self): return self._reward
    @property
    def action(self): return self._action
    @property
    def action_space(self): return self._env[0].action_space
    @property
    def isEnd(self): return self._isEnd
    @property
    def stateCode(self): return self._stateCode
    @property
    def state(self): return self._state    
    @property
    def info(self): return {"stateCode": self._stateCode, "truncatedEnd": self._truncatedEnd}
    
    def step(self, action):
        self._rest[self._isEnd] += 1
        self._warm += 1
        self.reset(self._rest>self._rest_n)      
        isLive = np.logical_and(self._warm>self._warm_n, ~self._isEnd)
        self._reward[~isLive] = 0
        for i in isLive.nonzero()[0]:   
            self._state[i], self._reward[i], self._isEnd[i], info = _gym_step(self._env[i], action[i])  
            self._truncatedEnd[i] = info['TimeLimit.truncated'] if 'TimeLimit.truncated' in info else False
            if self._truncatedEnd[i]: self._rest[i] = self._rest_n
        self._stateCode[isLive] = 0            
        self._stateCode[self._rest>=1] = 1
        self._stateCode[self._warm<= self._warm_n] = 3    
        self._stateCode[self._warm==0] = 2
        return self.state, self.reward, self._isEnd, self.info

    def reset(self, index=None):    
        for i in range(self._batch_size) if index is None else index.nonzero()[0]:
            self._state[i] = _gym_reset(self._env[i])
            self._reward[i] = 0
        if index is None: index = slice(None)  
        self._rest[index] = 0
        self._warm[index] = 0
        self._truncatedEnd[index] = False
        self._isEnd[index] = False
        return self._state

# ==========================================
# 5. BindsNET Tuning & Additions
# ==========================================

class adam_optimizer():
    def __init__(self, learning_rate, beta_1=0.999, beta_2=0.99999, epsilon=1e-09):
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.learning_rate = learning_rate
        self._cache = {}        
    
    def delta(self, grads, name="_", learning_rate=None):
        if name not in self._cache:
            self._cache[name] = [[torch.zeros_like(i).to(grads[0].device) for i in grads],
                                 [torch.zeros_like(i).to(grads[0].device) for i in grads],
                                 0]
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
            v = beta_2 * v + (1 - beta_2) * (g ** 2)
            self._cache[name][0][n] = m
            self._cache[name][1][n] = v            
            m_hat = m / (1 - np.power(beta_1, t).item())
            v_hat = v / (1 - np.power(beta_2, t).item())
            deltas.append(learning_rate * m_hat / (torch.sqrt(v_hat) + self.epsilon))
        return deltas   

class FBTDSTDP(LearningRule):
    def __init__(self, connection: AbstractConnection, nu: Optional[Union[float, Sequence[float]]] = None, reduction: Optional[callable] = None, weight_decay: float = 0.0, **kwargs) -> None:
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
            self.node_label = torch.repeat_interleave(torch.arange(start=0, end=self.num_actions), self.target.n//self.num_actions).unsqueeze(0)     
            if kwargs["gpu"]: self.node_label = self.node_label.cuda()
        self.targ_firing_r = torch.tensor(kwargs.get("targ_firing_r", 0))        
        self.targ_firing_rate = torch.tensor(kwargs.get("targ_firing_rate", 0.2))        
        self.inh = kwargs.get("inh", False)
        self.adam = kwargs.get("adam", False)
        self.weight_reg = kwargs.get("weight_reg", 0)
        self.gpu = kwargs.get("gpu", False)
        if self.adam: self.optimzier = adam_optimizer(nu[1])

    def _connection_update(self, **kwargs) -> None:
        batch_size = self.source.batch_size
        gpu = kwargs.get("gpu", False)
        self.at_decay = torch.exp(-self.connection.dt / self.tc_a_trace)
        self.et_decay = torch.exp(-self.connection.dt / self.tc_e_trace)
        self.ltp_decay = torch.exp(-self.connection.dt / self.tc_plus)
        self.ltd_decay = torch.exp(-self.connection.dt / self.tc_minus)
        if not hasattr(self, "p_plus"):
            self.p_plus = torch.zeros(batch_size, int(np.prod(self.source.shape)))
            if gpu: self.p_plus = self.p_plus.cuda()
        if not hasattr(self, "p_minus"):
            self.p_minus = torch.zeros(batch_size, *self.target.shape)
            if gpu: self.p_minus = self.p_minus.cuda()
        if not hasattr(self, "eligibility"):
            self.eligibility = torch.zeros(batch_size, *self.connection.w.shape)
            if gpu: self.eligibility = self.eligibility.cuda()
        if not hasattr(self, "eligibility_trace"):
            self.eligibility_trace = torch.zeros(batch_size, *self.connection.w.shape)
            if gpu: self.eligibility_trace = self.eligibility_trace.cuda() 
        if self.fb_gate:     
            if not hasattr(self, "action_v"):
                self.action_v = torch.zeros(batch_size, *self.connection.w.shape)
                if gpu: self.action_v = self.action_v.cuda()            
            if not hasattr(self, "action_trace"):
                self.action_trace = torch.zeros(batch_size, *self.connection.w.shape)
                if gpu: self.action_trace = self.action_trace.cuda()
        source_s = self.source.s.view(batch_size, -1).float()
        target_s = self.target.s.view(batch_size, -1).float()        
        reward = kwargs["reward"][self.connection]
        if self.fb_gate:   
            p = torch.repeat_interleave(kwargs["p"], self.target.n//self.num_actions, dim=-1).unsqueeze(1)            
            a = (kwargs["action"].unsqueeze(-1) == self.node_label).unsqueeze(1).float()     
            self.action_v = (a - p) * self.eligibility_trace
            self.action_trace *= self.at_decay
            self.action_trace += self.action_v / self.tc_a_trace
            update = reward * self.action_trace
            if kwargs.get("ent", None) is not None:
                ent = torch.repeat_interleave(kwargs["ent"], self.target.n//self.num_actions, dim=-1).unsqueeze(1)
                update += -ent*self.eligibility_trace if self.inh else ent*self.eligibility_trace            
        else:
            update = reward * self.eligibility_trace
        if self.targ_firing_r != 0:
            targ_adj = (self.targ_firing_rate - torch.mean(self.target.x, dim=1)/self.target.tc_trace)*self.targ_firing_r
            update += targ_adj.unsqueeze(-1).unsqueeze(-1)*self.eligibility_trace
        if self.weight_reg != 0: update -= 0.5 * self.weight_reg * self.connection.w
        if self.adam: update = self.optimzier.delta([self.reduction(update, dim=0)])[0]
        else: update = self.reduction(update, dim=0)
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
        if hasattr(self, "p_plus"):  self.p_plus[index] = 0
        if hasattr(self, "p_minus"):  self.p_minus[index] = 0
        if hasattr(self, "eligibility_trace"):  self.eligibility_trace[index] = 0
        if self.fb_gate and hasattr(self, "action_trace"): self.action_trace[index] = 0

    def set_batch_size(self, batch_size) -> None:
        for var in ["p_plus", "p_minus", "eligibility", "eligibility_trace", "action_v", "action_trace"]:
            if hasattr(self, var): delattr(self, var)

class Input_add(Input):
    def reset_state_variables(self, index=slice(None)) -> None: super().reset_state_variables()

class LIFNodes_add(LIFNodes):
    def reset_state_variables(self, index=slice(None)) -> None:
        self.v[index] = self.rest
        self.refrac_count[index] = 0
        if self.traces: self.x[index] = 0
        if self.sum_input: self.summed[index] = 0

class Network_add(Network):
    def run(self, inputs: Dict[str, torch.Tensor], time: int, one_step=False, **kwargs) -> None:
        clamps, unclamps, masks, injects_v = kwargs.get("clamp", {}), kwargs.get("unclamp", {}), kwargs.get("masks", {}), kwargs.get("injects_v", {})
        if inputs != {}:
            for key in inputs:
                if len(inputs[key].size()) == 1: inputs[key] = inputs[key].unsqueeze(0).unsqueeze(0)
                elif len(inputs[key].size()) == 2: inputs[key] = inputs[key].unsqueeze(1)
            for key in inputs:
                if inputs[key].size(1) != self.batch_size:
                    self.batch_size = inputs[key].size(1)
                    for l in self.layers: self.layers[l].set_batch_size(self.batch_size)
                    for m in self.monitors: self.monitors[m].reset_state_variables()
                break
        timesteps = int(time / self.dt)
        for t in range(timesteps):
            current_inputs = {} if one_step else self._get_inputs()
            for l in self.layers:
                if l in inputs: current_inputs[l] = current_inputs.get(l, 0) + inputs[l][t]
                if one_step: current_inputs.update(self._get_inputs(layers=[l]))
                self.layers[l].forward(x=current_inputs[l])
                for d, val in [(clamps, 1), (unclamps, 0)]:
                    clamp = d.get(l, None)
                    if clamp is not None: self.layers[l].s[:, clamp if clamp.ndimension() == 1 else clamp[t]] = val
                inject_v = injects_v.get(l, None)
                if inject_v is not None: self.layers[l].v += inject_v if inject_v.ndimension() == 1 else inject_v[t]
            if self.reward_fn is not None: kwargs["reward"] = self.reward_fn.compute(self, t, **kwargs)
            for c in self.connections: self.connections[c].update(mask=masks.get(c, None), learning=self.learning, **kwargs)
            current_inputs.update(self._get_inputs())
            for m in self.monitors: self.monitors[m].record()
        for c in self.connections: self.connections[c].normalize()

    def reset_state_variables(self, index=slice(None)) -> None:
        for layer in self.layers: self.layers[layer].reset_state_variables(index=index)
        for connection in self.connections: self.connections[connection].update_rule.reset_state_variables(index=index)

# ==========================================
# 6. Network Builder
# ==========================================

def build_ac_network(input_shape, init_scale, num_critics, num_vars_exc, num_actors_pa, num_actions, critic_layer_param, critic_fd_conn_param, var_layer_param, var_fd_conn_param, actor_layer_param, actor_fd_conn_param, actor_active=True, var_head=False, gpu=False):
    network = Network_add(dt=1)
    input_layer = Input_add(n=np.prod(input_shape), shape=(np.prod(input_shape),), traces=True, tc_trace=25.0)    
    network.add_layer(input_layer, name="H0")
    layers = [input_layer]
    in_n_neurons = int(np.prod(input_shape))
    if type(init_scale) != list: init_scale = [init_scale, init_scale]
    
    critic_layer_param.update({"n": num_critics[0], "shape": (num_critics[0],)})
    layers.append(LIFNodes_add(**critic_layer_param))    
    network.add_layer(layers[-1], name="CE")
    fd_w = torch.rand(in_n_neurons, num_critics[0]) * init_scale[0]
    critic_fd_conn_param.update({"source": input_layer, "target": layers[-1], "w": fd_w, "inh": False, "gpu": gpu})
    network.add_connection(Connection(**critic_fd_conn_param), source="H0", target="CE")         

    if num_critics[1] > 0:
        critic_layer_param.update({"n": num_critics[1], "shape": (num_critics[1],)})
        layers.append(LIFNodes_add(**critic_layer_param))
        network.add_layer(layers[-1], name="CN")
        fd_w = torch.rand(in_n_neurons, num_critics[1]) * init_scale[0]
        critic_fd_conn_param.update({"source": input_layer, "target": layers[-1], "w": fd_w, "inh": True, "gpu": gpu})
        network.add_connection(Connection(**critic_fd_conn_param), source="H0", target="CN")         

    if var_head and int(num_vars_exc) > 0:
        var_layer_param.update({"n": int(num_vars_exc), "shape": (int(num_vars_exc),)})
        layers.append(LIFNodes_add(**var_layer_param))
        network.add_layer(layers[-1], name="CV")
        fd_w = torch.rand(in_n_neurons, int(num_vars_exc)) * init_scale[0]
        var_fd_conn_param.update({"source": input_layer, "target": layers[-1], "w": fd_w, "inh": False, "gpu": gpu})
        network.add_connection(Connection(**var_fd_conn_param), source="H0", target="CV")

    if actor_active:                  
        actor_layer_param.update({"n": num_actors_pa[0]*num_actions, "shape": (num_actors_pa[0]*num_actions,)})
        layers.append(LIFNodes_add(**actor_layer_param))      
        network.add_layer(layers[-1], name="AE")      
        fd_w = torch.rand(in_n_neurons, num_actors_pa[0]*num_actions) * init_scale[1]  
        actor_fd_conn_param.update({"source": input_layer, "target": layers[-1], "w": fd_w, "num_actions": num_actions, "inh": False, "gpu": gpu})
        network.add_connection(Connection(**actor_fd_conn_param), source="H0", target="AE")

        if num_actors_pa[1] > 0:
            actor_layer_param.update({"n": num_actors_pa[1]*num_actions, "shape": (num_actors_pa[1]*num_actions,)})
            layers.append(LIFNodes_add(**actor_layer_param))
            network.add_layer(layers[-1], name="AN") 
            fd_w = torch.rand(in_n_neurons, num_actors_pa[1]*num_actions) * init_scale[1]
            actor_fd_conn_param.update({"source": input_layer, "target": layers[-1], "w": fd_w, "num_actions": num_actions, "inh": True, "gpu": gpu})        
            network.add_connection(Connection(**actor_fd_conn_param), source="H0", target="AN")
    if gpu: network.to("cuda")
    return network        

# ==========================================
# 7. Reward Function & Actor
# ==========================================

class Reward_fn():
    def __init__(self, tc, network_steps, isEnd_zero, last_zero_n, batch_size, value_m, value_b, var_m=1.0, var_b=0.0, var_eps=1e-6, gpu=False):
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
        self.gamma = torch.tensor(np.exp(-1/tc)).float()    
        self.r_gamma = torch.sqrt(self.gamma)    
        self.network_steps = network_steps    
        if self.gpu: 
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
        v = self.value_b + torch.mean(critic_exc.x, dim=1)/critic_exc.tc_trace*self.value_m
        if critic_inh is not None: v -= torch.mean(critic_inh.x, dim=1)/critic_inh.tc_trace*self.value_m    
        td_error = (self.r_gamma * env_reward + self.gamma * v * isLive.float() if self.isEnd_zero else self.r_gamma * env_reward + self.gamma * v) - self.v_prev    
        td_error[stateCode >= 2] = 0.
        self.v_prev = v
        td_error = td_error.unsqueeze(-1).unsqueeze(-1)    
        rewards = {network.connections[("H0","CE")]: td_error}        
        if "CN" in network.layers: rewards[network.connections[("H0","CN")]] = -td_error
        if "AE" in network.layers: rewards[network.connections[("H0","AE")]] = td_error
        if "AN" in network.layers: rewards[network.connections[("H0","AN")]] = -td_error    
        
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
        self.isEnd_rec[t]= isEnd[0]
        return rewards    

class Snn_actor():
    def __init__(self, network, batch_size, state_to_spike, augment_state, env_dt, reward_adj, actor_m, num_actions, num_actors_pa, actor_active, entropy_reg, freeze_action, var_lr_base=0.000125, 
                 use_ach_lr_mod=False, use_ne_noise=False, base_lr=0.0000025, actor_lr_min=1e-4, actor_lr_max=0.001, actor_lr_decay=0.1, actor_lr_boost=0.1, 
                 base_noise=1.0, ne_max=3.0, ne_k=0.2, ne_center=15.0, ach_max=1.0, ach_k=10.0, ach_center=5.0, 
                 td_signal_clip=20.0, td_fast_alpha=0.097663, td_slow_alpha=0.003642, unexp_decay=0.8, 
                 exp_fast_alpha=1.0, exp_slow_alpha=0.1, exp_decay=0.01, surprise_variance_weight=0.0, td_novelty_margin=0.0,
                 stat_rec=False, gpu=False):
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
        self.var_lr_base = float(var_lr_base)
        
        self.use_ach_lr_mod = use_ach_lr_mod
        self.use_ne_noise = use_ne_noise
        self.base_lr = base_lr
        self.actor_lr_min = actor_lr_min
        self.actor_lr_max = actor_lr_max
        self.actor_lr_decay = actor_lr_decay
        self.actor_lr_boost = actor_lr_boost
        
        self.base_noise = base_noise
        self.ne_max = ne_max
        self.ne_k = ne_k
        self.ne_center = ne_center
        
        self.ach_max = ach_max
        self.ach_k = ach_k
        self.ach_center = ach_center
        
        self.td_signal_clip = td_signal_clip
        self.td_fast_alpha = td_fast_alpha
        self.td_slow_alpha = td_slow_alpha
        self.unexp_decay = unexp_decay
        
        self.exp_fast_alpha = exp_fast_alpha
        self.exp_slow_alpha = exp_slow_alpha
        self.exp_decay = exp_decay
        self.surprise_variance_weight = surprise_variance_weight
        self.td_novelty_margin = td_novelty_margin
        
        self.actor_lr_state = float(self.base_lr)
        self.td_fast = 0.0
        self.td_slow = 0.0
        self.td_trace_inited = False
        self.avg_unexpected = 0.0
        
        self.exp_fast = 0.0
        self.exp_slow = 0.0
        self.exp_trace_inited = False
        self.avg_expected = 0.0
        
        self.current_ne = float(self.base_noise)
        self.current_ach = 0.0  # Or another baseline
        
        self.last_metrics = {}
        self.last_td_signal = 0.0
        self.last_td_fast = 0.0
        self.last_td_slow = 0.0
        self.last_td_novelty = 0.0
        
        self.stat_rec = stat_rec 
        self.gpu = gpu
        self.p = torch.full((batch_size, num_actions), 1/num_actions)
        self.ent_eye = torch.eye(num_actions)      
        self.action = torch.distributions.Categorical(self.p).sample()
        self.freeze_timer = torch.zeros(batch_size)        
        if gpu: 
            self.p, self.ent_eye, self.action, self.freeze_timer = self.p.cuda(), self.ent_eye.cuda(), self.action.cuda(), self.freeze_timer.cuda()
        if not actor_active: self.policy = uniform_policy(batch_size=batch_size, actions=list(range(num_actions)))    
        qlen, dlen = 3000, int(3000/env_dt)
        self.stat = {"est_v": collections.deque(maxlen=qlen), "td_error": collections.deque(maxlen=qlen), "reward": collections.deque(maxlen=qlen), "isEnd": collections.deque(maxlen=qlen), "v_exc_rate": collections.deque(maxlen=dlen)}
        if "CN" in self.network.layers: self.stat["v_inh_rate"] = collections.deque(maxlen=dlen)
        if self.actor_active:
            self.stat.update({"a_exc_rate": [collections.deque(maxlen=dlen) for _ in range(num_actions)], "p": [collections.deque(maxlen=dlen) for _ in range(num_actions)]})
            if "AN" in self.network.layers: self.stat["a_inh_rate"] = [collections.deque(maxlen=dlen) for _ in range(num_actions)]
          
    def _set_connection_lr(self, key: Tuple[str, str], lr: float) -> None:
        if key not in self.network.connections: return
        rule = self.network.connections[key].update_rule
        rule.nu = (0.0, float(lr))
        if getattr(rule, "adam", False) and hasattr(rule, "optimzier"):
            rule.optimzier.learning_rate = float(lr)

    def __call__(self, state, reward, isEnd, info):    
        network = self.network
        if np.any(info["stateCode"]==2): network.reset_state_variables(info["stateCode"]==2)    
        
        self._set_connection_lr(("H0", "CV"), self.var_lr_base)
        actor_lr_used = self.actor_lr_state if self.use_ach_lr_mod else self.base_lr
        self._set_connection_lr(("H0", "AE"), actor_lr_used)

        p = self.p
        ent = (-self.entropy_reg * torch.sum(((p * (torch.log(p) + 1))[:, np.newaxis, :] * (self.ent_eye - p[..., np.newaxis])), axis=-1) if self.entropy_reg > 0 else None)    
        augment = np.concatenate([isEnd[:, np.newaxis], (info["stateCode"]==1)[:,np.newaxis]], axis=-1) if self.augment_state else None    
        state_spike = self.state_to_spike(state, augment)  
        if self.gpu: state_spike = state_spike.cuda()
        network.run(inputs={"H0": state_spike.reshape(state_spike.shape[:2]+(-1,))}, time=self.env_dt, input_time_dim=1, action=self.action, p=self.p, ent=ent, env_reward=self.reward_adj*reward, info=info, isEnd=isEnd, gpu=self.gpu)    
        
        # Extract Modulators
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
        self.current_ach = float(logistic_drive(self.ach_max, self.ach_k, self.ach_center, self.avg_expected, 0.0))

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

        if self.use_ach_lr_mod:
            self.actor_lr_state *= (1.0 - self.actor_lr_decay)
            self.actor_lr_state += self.actor_lr_boost * self.current_ach
            self.actor_lr_state = float(min(max(self.actor_lr_state, self.actor_lr_min), self.actor_lr_max))

        if self.actor_active:
            actor_exc = network.layers["AE"]
            a_exc_rate = torch.mean(actor_exc.x.view(-1, self.num_actions, self.num_actors_exc_pa), dim=-1)/actor_exc.tc_trace
            a_rate = self.actor_m * (a_exc_rate - torch.mean(network.layers["AN"].x.view(-1, self.num_actions, self.num_actors_inh_pa), dim=-1)/network.layers["AN"].tc_trace) if "AN" in network.layers else self.actor_m * a_exc_rate
            
            logits = a_rate
            if self.use_ne_noise:
                logits = logits + torch.randn_like(logits) * float(self.current_ne)
            
            self.p = torch.nn.functional.softmax(logits, dim=1)
            self.freeze_timer += self.env_dt
            new_b = self.freeze_timer >= self.freeze_action      
            if new_b.any().detach().cpu().numpy():        
                self.action[new_b] = torch.distributions.Categorical(logits=a_rate[new_b]).sample()       
                self.freeze_timer[new_b] = 0
        else: self.action = torch.from_numpy(self.policy(state))
        if self.stat_rec:
            for k in ["est_v", "td_error", "reward", "isEnd"]: self.stat[k].extend(getattr(network.reward_fn, k+"_rec").detach().cpu().numpy().tolist())
            self.stat["v_exc_rate"].append((torch.mean(network.layers["CE"].x[0], dim=-1)/network.layers["CE"].tc_trace).detach().cpu().numpy())
            if "CN" in network.layers: self.stat["v_inh_rate"].append((torch.mean(network.layers["CN"].x[0], dim=-1)/network.layers["CN"].tc_trace).detach().cpu().numpy())              
            if hasattr(self, "current_ne"): self.stat["ne"].append(self.current_ne)
            if hasattr(self, "current_ach"): self.stat["ach"].append(self.current_ach)
            if hasattr(self, "actor_lr_state"): self.stat["actor_lr"].append(self.actor_lr_state)
            
            if self.actor_active:
                for i in range(self.num_actions): 
                    self.stat["a_exc_rate"][i].append(a_exc_rate[0, i].detach().cpu().numpy())
                    if "AN" in network.layers: self.stat["a_inh_rate"][i].append((torch.mean(network.layers["AN"].x[0].view(-1, self.num_actions, self.num_actors_inh_pa), dim=-1)/network.layers["AN"].tc_trace)[0, i].detach().cpu().numpy())
                    self.stat["p"][i].append(self.p[0, i].detach().cpu().numpy())             
        return (self.action.detach().cpu().numpy(), self.stat)
class Stat_plotter():  
    def __init__(self, len_adj): self.pers, self.axs, self.len_adj = [None]*5, [None]*5, len_adj
    def __call__(self, stat, eps_ret, eps_len):
        d = {"Return of Episode": np.array(eps_ret)}
        if len(eps_ret) >= 100: d["Trailing avg return of Episode"] = np.concatenate([np.full((99,), np.nan,), mv(eps_ret, 100)])
        self.pers[0], self.axs[0] = plot_performance(d, title="Learning Curve", xlabel="Episodes", ylabel="Return of Episode", fig=self.pers[0], ax=self.axs[0])
        d = {"Length of Episode": np.array(eps_len)+self.len_adj}
        if len(eps_ret) >= 100: d["Trailing avg length of Episode"] = np.concatenate([np.full((99,), np.nan,), mv(eps_len, 100)])+self.len_adj
        self.pers[1], self.axs[1] = plot_performance(d, title="Learning Curve", xlabel="Episodes", ylabel="Lengths (Seconds)", fig=self.pers[1], ax=self.axs[1])   
        d = {"Exc Value Neuron": stat["v_exc_rate"]}           
        if "v_inh_rate" in stat: d["Inh Value Neuron"] = stat["v_inh_rate"]
        if "a_exc_rate" in stat: d.update({"Exc Actor %d Neuron"% i: stat["a_exc_rate"][i] for i in range(len(stat["a_exc_rate"]))})
        if "a_inh_rate" in stat: d.update({"Inh Actor %d Neuron"% i: stat["a_inh_rate"][i] for i in range(len(stat["a_inh_rate"]))})  
        self.pers[2], self.axs[2] = plot_performance(d, title="Firing Rate", xlabel="Step", ylabel="Firing Rate", fig=self.pers[2], ax=self.axs[2], ylim=[0, 1])
        if "p" in stat:
            self.pers[3], self.axs[3] = plot_performance({ "Action %d"% i: stat["p"][i] for i in range(len(stat["p"]))}, title="Probability of Action", xlabel="Step", ylabel="p", fig=self.pers[3], ax=self.axs[3], ylim=[0, 1])
        self.pers[4], self.axs[4] = plot_performance({"Value from Network": np.array(stat["est_v"]), "TD Error": np.array(stat["td_error"]), "Reward": np.array(stat["reward"]), "isEnd": np.array(stat["isEnd"])*0.05}, title="Value Stat.", xlabel="Step", ylabel="", fig=self.pers[4], ax=self.axs[4])

# ==========================================
# 8. Main Configuration & Training Loop
# ==========================================

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    result_dir = os.path.join(script_dir, "result")
    model_dir = os.path.join(script_dir, "model")
    os.makedirs(result_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    
    # Setup internal file logging
    log_path_env = os.environ.get("SWEEP_LOG_PATH")
    if log_path_env:
        log_file = sys.stdout
    else:
        log_path = os.path.join(result_dir, "training_log.txt")
        log_file = open(log_path, "w", buffering=1)
        print(f"Logging episode data to: {log_path} (Terminal output suppressed for episodes)")

    # Parameters from config_ll.ini
    gpu_flag = True
    min_eps = 3000
    max_eps = 2000
    n_run = 1
    batch_size = 16
    input_type = 2 # Fourier
    augment_state = False
    rep = 16
    basis = True
    forier_order = 1
    forier_cross_term = False
    forier_double = True
    critic_lr = 0.000125
    actor_lr = 0.0000625
    init_scale = 0.125
    adam = True
    adam_beta_1 = 0.995
    adam_beta_2 = 0.99995
    env_name = "LunarLander-v2"
    env_tc = 2000
    env_dt = 20
    
    # Variance head parameters
    var_head = True
    num_vars_exc = 128
    var_m = 0.01
    var_b = 0.0
    var_eps = 1e-6
    var_lr = 0.0002
    
    # Icarus Neuromodulation Parameters
    use_ach_lr_mod = True
    use_ne_noise = True
    
    base_lr = 0.0000625
    actor_lr_min = 0.0000625  # Replaced from 1e-4 to allow bottoming out at base
    actor_lr_max = 0.001
    actor_lr_decay = 0.1
    
    actor_lr_boost = 0.1
    
    base_noise = 1.0
    ne_max = 3.0
    ne_k = 0.2
    ne_center = 15.0
    
    ach_max = 1
    ach_k = 1000.0
    ach_center = 0.08
    
    td_signal_clip = 20.0
    td_fast_alpha = 0.097663
    td_slow_alpha = 0.003642
    unexp_decay = 0.8
    
    exp_fast_alpha = 1.0
    exp_slow_alpha = 0.01 # Approximately 5 episodes of memory (1000 steps)
    exp_decay = 0.01
    surprise_variance_weight = 0.0
    td_novelty_margin = 0.0
    reward_adj = 0.012
    freeze_action = 40
    isEnd_zero = True
    last_zero_n = 2
    rest_n = 0
    warm_n = 5
    actor_active = True
    num_critics_exc = 128
    num_critics_inh = 128
    num_actors_exc_pa = 32
    num_actors_inh_pa = 32
    entropy_reg = 0.0001
    weight_reg = 0.00000001
    value_m = 4
    value_b = -2
    actor_m = 15
    tau_n = 20.0
    tau_plus = 20.0
    tau_z = 20.0
    tau_q = 20.0
    tau_v = 100.0
    targ_firing_r = 0.0
    targ_firing_rate = 0.0
    name = "ll_std"
    test = False
    checkpoint = "model_ll_std_0.pt"
    test_eps = 100
    test_vis = False

    # GPU Check Safeguard
    gpu_requested = bool(gpu_flag)
    gpu = bool(gpu_flag and torch.cuda.is_available())
    if gpu_requested and not gpu:
        print("CUDA not available; falling back to CPU.")

    state_lim = {
        "LunarLander-v2": [np.array([-1, -0.2, -1, -1, -1, -1, 0, 0]),
                           np.array([+1, +2, +1, +1, +1, +1, 1, 1])]
    }
    solve_def = {
        "LunarLander-v2": (100, 200)
    }

    env = batch_envs(name=env_name, batch_size=batch_size, rest_n=rest_n, warm_n=warm_n)  
    state_size = env.state.shape[1]
    num_actions = env.action_space.n 
    bin_min, bin_max = state_lim[env_name]

    if input_type != 2:
        raise ValueError("Only input_type=2 (Fourier) is implemented in this combined script for LunarLander.")

    state_to_spike = State_to_spike_Fourier(
        env_dt, bin_min, bin_max,  
        k=forier_order, basis=basis, soft=False,
        cross_term=forier_cross_term, double=forier_double, rep=rep
    )

    input_shape = state_to_spike.out_shape
    if augment_state: input_shape = (rep, input_shape[1]+2)
    print("Input shape:", input_shape)

    critic_layer_param = {
        "traces": True,             
        "traces_additive": True,   
        "tc_trace": tau_n,           
        "refrac": 0,
        "tc_decay": tau_v
    }
    actor_layer_param = critic_layer_param.copy()
    actor_layer_param.update({"tc_trace": tau_plus})     
                      
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
    }
    actor_fd_conn_param = critic_fd_conn_param.copy()
    actor_fd_conn_param.update({
        "update_rule": FBTDSTDP,
        "nu": (0, actor_lr), 
        "tc_plus": tau_plus,
        "tc_e_trace": tau_z,
        "tc_a_trace": tau_q,
        "fb_gate": True,
        "targ_firing_r": targ_firing_r,
        "targ_firing_rate": targ_firing_rate,
        "adam": adam
    })     

    stat_plotter = Stat_plotter(len_adj=-(rest_n-warm_n)*env_dt)
                             
    results = []
    plot_stat = False

    for n in range(1 if test else n_run):  
        env.reset()
        var_layer_param = critic_layer_param.copy()
        var_layer_param.update({"tc_trace": tau_n})
        var_fd_conn_param = critic_fd_conn_param.copy()
        var_fd_conn_param.update(
            {
                "update_rule": FBTDSTDP,
                "nu": (0, var_lr),
                "wmin": 0.0,
                "fb_gate": False,
                "adam": adam,
                "adam_beta_1": adam_beta_1,
                "adam_beta_2": adam_beta_2,
                "weight_reg": weight_reg,
            }
        )

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
            gpu=gpu
        )
        print("Initializing %s network.." % name)
      
        for k, v in network.layers.items(): v.set_batch_size(batch_size)
        for k, v in network.connections.items(): v.update_rule.set_batch_size(batch_size)

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
            gpu=gpu
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
            var_lr_base=var_lr,
            use_ach_lr_mod=use_ach_lr_mod,
            use_ne_noise=use_ne_noise,
            base_lr=base_lr,
            actor_lr_min=actor_lr_min,
            actor_lr_max=actor_lr_max,
            actor_lr_decay=actor_lr_decay,
            actor_lr_boost=actor_lr_boost,
            base_noise=base_noise,
            ne_max=ne_max,
            ne_k=ne_k,
            ne_center=ne_center,
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
            stat_rec=plot_stat, 
            gpu=gpu
        )
        snn_actor._set_connection_lr(("H0", "CV"), var_lr)

        eps_ret, eps_len = [], []  
        c_eps_ret = np.zeros(batch_size)
        c_eps_len = np.zeros(batch_size)
        step, p_eps = 0, 0
        plot_episodes, plot_rewards, plot_avg_rewards, plot_variances, plot_lrs = [], [], [], [], []
        f_perfect, solved = False, False

        state = env.reset()
        reward = env.reward
        isEnd = env.isEnd
        info = env.info
        
        while True:
            step += 1
            action, stat = snn_actor(state, reward, isEnd, info)

            action_arr = np.asarray(action)
            if action_arr.shape == ():
                action_arr = action_arr.reshape(-1)

            # Invert controls for LunarLander at eps >= 1000 (Swap Left and Right)
            real_action = action_arr.copy()
            if len(eps_ret) >= 1000:
                mask_1 = (action_arr == 1)
                mask_3 = (action_arr == 3)
                real_action[mask_1] = 3
                real_action[mask_3] = 1

            state, reward, _isEnd, info = env.step(real_action)    
            stateCode = info['stateCode']

            c_eps_ret += reward
            c_eps_len += env_dt
            new_end = np.logical_and(isEnd == False, _isEnd==True)
            if np.any(new_end):
                eps_ret.extend(c_eps_ret[new_end].tolist())
                eps_len.extend(c_eps_len[new_end].tolist())      
                c_eps_ret[new_end] = 0.
                c_eps_len[new_end] = 0.    
                while len(eps_ret) >= p_eps + 1:      
                    p_eps += 1
                    mean_var = 0.0
                    if "CV" in network.layers:
                         mean_var = torch.mean(network.reward_fn.var_rec).item()
                    curr_ne = getattr(snn_actor, "current_ne", 0.0)
                    curr_ach = getattr(snn_actor, "current_ach", 0.0)
                    curr_lr = getattr(snn_actor, "actor_lr_state", base_lr)
                    avg_ret_100 = np.average(eps_ret[max(0, p_eps-100):p_eps])
                    log_str = "%d: Ret: %.2f; Last 100 Avg Ret: %.2f; Var: %.8f; NE: %.3f; ACh: %.3f; ActLR: %.8f; Solved: %s" % (
                            p_eps, eps_ret[p_eps-1], avg_ret_100, mean_var, curr_ne, curr_ach, curr_lr, "Y" if solved else "N")
                    log_file.write(log_str + "\n")
                    # print(log_str) # Suppressed as per user request
                    plot_episodes.append(p_eps)
                    plot_rewards.append(eps_ret[p_eps-1])
                    plot_avg_rewards.append(avg_ret_100)
                    plot_variances.append(mean_var)
                    plot_lrs.append(curr_lr)            
                if env_name in solve_def:          
                    avg_n, p_score = solve_def[env_name]
                    if not f_perfect and np.amax(eps_ret) >= p_score: f_perfect = True           
                    if not solved and len(eps_ret) > avg_n and np.amax(mv(eps_ret, avg_n)) >= p_score:
                        solved = True             
                        f_save = os.path.join(model_dir, "model_%s_%d.pt" %(name, n))  
                        print("%d: Solved. Model saved to %s" % (n, f_save))
                        network.save(f_save)            
                if len(eps_ret) >= max_eps or (solved and len(eps_ret) >= min_eps): break
            isEnd = np.copy(_isEnd)
        results.append([eps_ret, eps_len])
        print("Average return : %f (%d episodes)" % (np.average(eps_ret), len(eps_ret)))
        f_save = os.path.join(result_dir, "rewards_%s_%d.pkl" %(name, n))
        with open(f_save, 'wb') as f: pickle.dump(results[-1], f)

    f_save = os.path.join(result_dir, "rewards_%s_all.pkl" %(name))
    with open(f_save, 'wb') as f: pickle.dump(results, f)    
    print("Training Complete.")

    # --- Plot Reward, Variance, Learning Rate ---
    if len(plot_episodes) > 0:
        fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

        axes[0].plot(plot_episodes, plot_rewards, color='lightgray', alpha=0.6, label='Return')
        axes[0].plot(plot_episodes, plot_avg_rewards, color='blue', linewidth=2, label='Last 100 Avg Return')
        axes[0].set_ylabel('Reward')
        axes[0].set_title('Reward over Episodes')
        axes[0].legend()
        axes[0].grid(True)

        axes[1].plot(plot_episodes, plot_variances, color='orange')
        axes[1].set_ylabel('Variance')
        axes[1].set_title('Variance over Episodes')
        axes[1].grid(True)

        axes[2].plot(plot_episodes, plot_lrs, color='green')
        axes[2].set_ylabel('Learning Rate')
        axes[2].set_xlabel('Episode')
        axes[2].set_title('Actor Learning Rate over Episodes')
        axes[2].grid(True)

        plt.tight_layout()
        
        # Save to dynamic path for sweeps
        plot_dir = os.environ.get("SWEEP_PLOT_DIR", result_dir)
        os.makedirs(plot_dir, exist_ok=True)
        plot_name = os.environ.get("SWEEP_PLOT_NAME", "training_metrics.png")
        plot_path = os.path.join(plot_dir, plot_name)
        
        plt.savefig(plot_path)
        print(f"Plot saved to {plot_path}")
