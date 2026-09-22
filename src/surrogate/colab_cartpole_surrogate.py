import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "gymnasium[classic-control]"], check=True)

import math, random, os
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

# Config
SEEDS          = [1]            # list(range(1, 21)) for full 20-seed batch
TOTAL_EPISODES = 10000
SWITCH_EP      = 5000           # action mapping inverts at this episode
OUTPUT_DIR     = "/content/cartpole_surrogate_runs"

DT            = 0.02
RHO_PC        = 50.0
TAU_M         = 0.02
ACTOR_THETA   = 2.0
GAMMA         = 0.99
BASE_LR       = 0.000055
BASE_NOISE    = 1.0

ACTOR_LR_DECAY = 0.1
ACTOR_LR_BOOST = 0.1
ACTOR_LR_MIN   = 0.003
ACTOR_LR_MAX   = 0.05

CRITIC_BASE_LR = 0.001

# Target critic for TD targets, soft-updated once per episode (0 = frozen, 1 = no target).
TARGET_CRITIC_TAU = 0.01        # per-episode soft-update rate

# Surprise signal (kept for logging; neuromod is off)
TD_SIGNAL_CLIP     = 20.0
TD_FAST_ALPHA      = 0.097663
TD_SLOW_ALPHA      = 0.003642
EXP_SURPRISE_DECAY = 0.01
SURPRISE_EPS       = 1e-3

os.makedirs(OUTPUT_DIR, exist_ok=True)


class PlaceCellEncoder(nn.Module):
    def __init__(self, device):
        super().__init__()
        m = torch.linspace(-2.5,  2.5, 6, device=device)
        n = torch.linspace(-2.0,  2.0, 6, device=device)
        p = torch.linspace(-0.25, 0.25, 8, device=device)
        q = torch.linspace(-2.0,  2.0, 8, device=device)
        mesh = torch.meshgrid(m, n, p, q, indexing="ij")
        centers = torch.stack([x.flatten() for x in mesh], dim=1)
        self.register_buffer("centers", centers)
        sigmas = torch.tensor(
            [(m[1]-m[0])/1.5, (n[1]-n[0])/1.5, (p[1]-p[0])/1.5, (q[1]-q[0])/1.5],
            device=device).unsqueeze(0)
        self.register_buffer("sigmas", sigmas)
        self.n_neurons = centers.shape[0]

    def forward(self, state):
        diff  = (state.unsqueeze(0) - self.centers) ** 2
        rates = RHO_PC * torch.exp(-torch.sum(diff / (2 * self.sigmas**2), dim=1))
        probs = 1.0 - torch.exp(-rates * DT)
        return torch.bernoulli(torch.clamp(probs, 0.0, 1.0))


class SurrogateCritic(nn.Module):
    def __init__(self, n_input):
        super().__init__()
        self.fc_val = nn.Linear(n_input, 1, bias=True)
        self.fc_var = nn.Linear(n_input, 1, bias=True)
        with torch.no_grad():
            self.fc_val.weight.zero_(); self.fc_val.bias.zero_()
            self.fc_var.weight.zero_(); self.fc_var.bias.zero_()

    def forward(self, spikes):
        return self.fc_val(spikes), F.softplus(self.fc_var(spikes)) + 1e-4


class SurrogateActor(nn.Module):
    def __init__(self, n_input, device):
        super().__init__()
        self.w         = nn.Parameter(torch.empty(n_input, 2, device=device).normal_(0.0, 0.1))
        self.z_eps     = torch.zeros(n_input, device=device)
        self.decay_eps = math.exp(-DT / TAU_M)

    def reset_state(self):
        self.z_eps.zero_()

    def forward(self, spikes, noise_scale):
        self.z_eps = self.z_eps * self.decay_eps + spikes
        v_det = torch.matmul(self.z_eps, self.w)
        v_mem = v_det + torch.randn_like(v_det) * noise_scale
        spike_input = v_mem - ACTOR_THETA

        class Surrogate(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x):
                ctx.save_for_backward(x)
                return (x > 0).float()
            @staticmethod
            def backward(ctx, grad):
                (x,) = ctx.saved_tensors
                return grad / (1.0 + torch.abs(x) * 5.0).pow(2)

        out_spikes = Surrogate.apply(spike_input)
        s = out_spikes.detach().cpu().numpy()
        if   s[0] == 1 and s[1] == 0: action = 0
        elif s[0] == 0 and s[1] == 1: action = 1
        else:                          action = int(torch.argmax(v_mem).item())
        return action, out_spikes

    def surrogate_update(self, td_error, out_spikes, lr_scale, optimizer):
        loss = -float(lr_scale * td_error) * torch.sum(self.w * torch.outer(self.z_eps, out_spikes))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            self.w.clamp_(-10.0, 10.0)


def soft_update(source, target, tau):
    with torch.no_grad():
        for sp, tp in zip(source.parameters(), target.parameters()):
            tp.data.copy_(tau * sp.data + (1.0 - tau) * tp.data)


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def train(seed, episodes=TOTAL_EPISODES):
    device = torch.device("cpu")
    set_seed(seed)

    env = gym.make("CartPole-v1")
    try:    env.reset(seed=seed)
    except: pass
    try:    env.action_space.seed(seed)
    except: pass

    encoder       = PlaceCellEncoder(device)
    actor         = SurrogateActor(encoder.n_neurons, device)
    critic        = SurrogateCritic(encoder.n_neurons).to(device)
    target_critic = SurrogateCritic(encoder.n_neurons).to(device)
    target_critic.load_state_dict(critic.state_dict())

    actor_optim  = optim.SGD([actor.w], lr=1.0)
    critic_optim = optim.Adam(critic.parameters(), lr=CRITIC_BASE_LR)

    avg_surprise   = 0.0
    td_fast = td_slow = 0.0
    td_inited      = False
    actor_lr_state = ACTOR_LR_MIN
    reward_history = []

    print(f"[Seed {seed}] CartPole surrogate | target_tau={TARGET_CRITIC_TAU} | "
          f"LR_floor={ACTOR_LR_MIN} | {episodes} eps | switch@{SWITCH_EP}")

    for ep in range(episodes):
        inverted = (ep >= SWITCH_EP)

        try:    obs, _ = env.reset(seed=seed + ep)
        except: obs, _ = env.reset()

        obs_t = torch.tensor(obs, dtype=torch.float32)
        actor.reset_state()
        done = False
        total_reward = 0.0

        while not done:
            spikes = encoder(obs_t)
            action_code, act_spikes = actor(spikes, noise_scale=BASE_NOISE)
            real_action = 1 - action_code if inverted else action_code

            next_obs, reward, terminated, truncated, _ = env.step(real_action)
            done = terminated or truncated
            next_obs_t = torch.tensor(next_obs, dtype=torch.float32)

            v_curr, var_curr = critic(spikes)

            # TD target from the target critic
            with torch.no_grad():
                v_next = target_critic(encoder(next_obs_t))[0] if not done else torch.zeros(1)

            td_error = (reward + GAMMA * v_next) - v_curr
            td_val   = float(td_error.item())

            # Critic trains on full unclipped error
            critic_optim.zero_grad()
            (td_error.pow(2) + (var_curr - td_error.detach().pow(2)).pow(2)).backward()
            critic_optim.step()

            actor_lr_state = actor_lr_state * (1.0 - ACTOR_LR_DECAY) + ACTOR_LR_BOOST * BASE_LR
            actor_lr_state = min(max(actor_lr_state, ACTOR_LR_MIN), ACTOR_LR_MAX)
            actor.surrogate_update(td_val, act_spikes, actor_lr_state, actor_optim)

            # Surprise signal (logged only)
            td_sig = float(min(max(abs(td_val), 0.0), TD_SIGNAL_CLIP))
            if not td_inited:
                td_fast = td_slow = td_sig; td_inited = True
            else:
                td_fast = (1 - TD_FAST_ALPHA) * td_fast + TD_FAST_ALPHA * td_sig
                td_slow = (1 - TD_SLOW_ALPHA) * td_slow + TD_SLOW_ALPHA * td_sig
            avg_surprise = EXP_SURPRISE_DECAY * avg_surprise + (1 - EXP_SURPRISE_DECAY) * max(0.0, td_fast - td_slow)

            obs_t = next_obs_t
            total_reward += float(reward)

        reward_history.append(total_reward)

        # Soft-update target critic once per episode
        soft_update(critic, target_critic, TARGET_CRITIC_TAU)

        if ep % 500 == 0:
            avg20 = float(np.mean(reward_history[-20:]))
            print(f"  Ep {ep:5d} | {'INVERTED' if inverted else 'NORMAL  '} | "
                  f"R: {total_reward:5.0f} | Avg20: {avg20:5.1f} | "
                  f"LR: {actor_lr_state:.5f} | Surprise: {avg_surprise:.3f}")

    env.close()

    run_dir  = os.path.join(OUTPUT_DIR, f"{seed}_cartpole_surrogate")
    os.makedirs(run_dir, exist_ok=True)
    csv_path = os.path.join(run_dir, f"{seed}_cartpole_surrogate.csv")
    with open(csv_path, "w") as fh:
        fh.write("episode,reward\n")
        for i, r in enumerate(reward_history):
            fh.write(f"{i},{r}\n")
    print(f"[Seed {seed}] Saved → {csv_path}")
    return reward_history


# Run
all_rewards = {}
for s in SEEDS:
    all_rewards[s] = train(s)

# Plot
fig, ax = plt.subplots(figsize=(12, 4))
W = 100
for s, rewards in all_rewards.items():
    r = np.array(rewards, dtype=float)
    ax.plot(np.convolve(r, np.ones(W)/W, mode='valid'), alpha=0.8, label=f"seed {s}")
ax.axvline(SWITCH_EP - W//2, color='red', linestyle='--', label='Switch (ep 5000)')
ax.set_xlabel("Episode"); ax.set_ylabel(f"Reward ({W}-ep smoothed)")
ax.set_title("CartPole Surrogate — Switch Experiment")
ax.legend(); ax.set_ylim(bottom=0)
plot_path = os.path.join(OUTPUT_DIR, "rewards.png")
plt.savefig(plot_path, dpi=120, bbox_inches='tight')
plt.show()

# Download (Colab only)
try:
    import shutil
    from google.colab import files
    zip_path = "/content/cartpole_surrogate_results"
    shutil.make_archive(zip_path, 'zip', OUTPUT_DIR)
    files.download(zip_path + ".zip")
except ImportError:
    pass
