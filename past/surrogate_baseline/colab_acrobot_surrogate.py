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

# ── Config ────────────────────────────────────────────────────────────────────
SEEDS          = [1]            # swap for list(range(1, 21)) to run 20 seeds
TOTAL_EPISODES = 15000
SWITCH_EP      = 7500           # link lengths swap at this episode
OUTPUT_DIR     = "/content/acrobot_surrogate_runs"
# ─────────────────────────────────────────────────────────────────────────────

DT                 = 0.02
RHO_PC             = 50.0
TAU_M              = 0.02
ACTOR_THETA        = 2.0
GAMMA              = 0.99
BASE_LR            = 0.000055
BASE_NOISE         = 1
TD_SIGNAL_CLIP     = 20.0
TD_FAST_ALPHA      = 0.097663
TD_SLOW_ALPHA      = 0.003642
TD_NOVELTY_MARGIN  = 0.0
SURPRISE_EPS       = 1e-3
EXP_SURPRISE_DECAY = 0.01
UNEXP_SURPRISE_DECAY = 0.8
CRITIC_BASE_LR     = 0.001
ACTOR_LR_DECAY     = 0.1
ACTOR_LR_BOOST     = 0.1
ACTOR_LR_MIN       = 1e-4
ACTOR_LR_MAX       = 0.1

PC_GRID_SIZES = [5, 5, 5, 5, 4, 4]   # cos1, sin1, cos2, sin2, w1, w2

os.makedirs(OUTPUT_DIR, exist_ok=True)


class PlaceCellEncoder(nn.Module):
    def __init__(self, device):
        super().__init__()
        gs    = PC_GRID_SIZES
        lows  = [-1.0, -1.0, -1.0, -1.0, -4*math.pi, -9*math.pi]
        highs = [ 1.0,  1.0,  1.0,  1.0,  4*math.pi,  9*math.pi]
        linspaces = [torch.linspace(lo, hi, n, device=device) for lo, hi, n in zip(lows, highs, gs)]
        mesh    = torch.meshgrid(*linspaces, indexing="ij")
        centers = torch.stack([x.flatten() for x in mesh], dim=1)
        self.register_buffer("centers", centers)
        sigma_list = [(ls[1]-ls[0])/1.5 if len(ls) > 1 else torch.tensor(1.0, device=device)
                      for ls in linspaces]
        self.register_buffer("sigmas", torch.stack(sigma_list).unsqueeze(0))
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
    def __init__(self, n_input, n_actions, device):
        super().__init__()
        self.w         = nn.Parameter(torch.empty(n_input, n_actions, device=device).normal_(0.0, 0.1))
        self.z_eps     = torch.zeros(n_input, device=device)
        self.decay_eps = math.exp(-DT / TAU_M)

    def reset_state(self):
        self.z_eps.zero_()

    def forward(self, spikes, noise_scale):
        self.z_eps = self.z_eps * self.decay_eps + spikes
        v_mem = torch.matmul(self.z_eps, self.w) + torch.randn(self.w.shape[1]) * noise_scale
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
        if torch.sum(out_spikes) == 1:
            action = int(torch.argmax(out_spikes).item())
        else:
            action = int(torch.argmax(v_mem).item())
        return action, out_spikes

    def surrogate_update(self, td_error, out_spikes, lr_scale, optimizer):
        loss = -float(lr_scale * td_error) * torch.sum(self.w * torch.outer(self.z_eps, out_spikes))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            self.w.clamp_(-10.0, 10.0)


def swap_link_params(env):
    uw = env.unwrapped
    half = uw.LINK_LENGTH_1 / 2.0
    uw.LINK_LENGTH_2 += half
    uw.LINK_LENGTH_1  = half
    uw.LINK_COM_POS_1 = uw.LINK_LENGTH_1 / 2.0
    uw.LINK_COM_POS_2 = uw.LINK_LENGTH_2 / 2.0


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def train(seed, episodes=TOTAL_EPISODES):
    device = torch.device("cpu")
    set_seed(seed)

    env = gym.make("Acrobot-v1")
    try:    env.reset(seed=seed)
    except: pass
    try:    env.action_space.seed(seed)
    except: pass

    encoder      = PlaceCellEncoder(device)
    actor        = SurrogateActor(encoder.n_neurons, n_actions=3, device=device)
    critic       = SurrogateCritic(encoder.n_neurons).to(device)
    actor_optim  = optim.SGD([actor.w], lr=1.0)
    critic_optim = optim.Adam(critic.parameters(), lr=CRITIC_BASE_LR)

    td_fast = td_slow = 0.0
    td_inited       = False
    avg_unexpected  = 0.0
    avg_expected    = 0.0
    actor_lr_state  = BASE_LR
    switched        = False
    reward_history  = []

    print(f"[Seed {seed}] Acrobot surrogate — {episodes} episodes, switch at {SWITCH_EP}")

    for ep in range(episodes):
        if ep == SWITCH_EP and not switched:
            swap_link_params(env)
            switched = True
            print(f"  >>> SWITCH at episode {ep}: link lengths changed <<<")

        try:    obs, _ = env.reset(seed=seed + ep)
        except: obs, _ = env.reset()

        obs_t = torch.tensor(obs, dtype=torch.float32)
        actor.reset_state()
        done = False
        total_reward = 0.0

        while not done:
            spikes = encoder(obs_t)
            action, act_spikes = actor(spikes, noise_scale=BASE_NOISE)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            next_obs_t = torch.tensor(next_obs, dtype=torch.float32)

            v_curr, var_curr = critic(spikes)
            with torch.no_grad():
                v_next = critic(encoder(next_obs_t))[0] if not done else torch.tensor([0.0])

            td_error = (reward + GAMMA * v_next) - v_curr
            td_val   = float(td_error.item())

            critic_optim.zero_grad()
            (td_error.pow(2) + (var_curr - td_error.detach().pow(2)).pow(2)).backward()
            critic_optim.step()

            # Actor LR adapts per-step
            actor_lr_state *= (1.0 - ACTOR_LR_DECAY)
            actor_lr_state += ACTOR_LR_BOOST * BASE_LR
            actor_lr_state  = min(max(actor_lr_state, ACTOR_LR_MIN), ACTOR_LR_MAX)

            actor.surrogate_update(td_val, act_spikes, actor_lr_state, actor_optim)

            sigma = max(torch.sqrt(var_curr.detach()).item(), SURPRISE_EPS)
            avg_expected = (1 - EXP_SURPRISE_DECAY) * avg_expected + EXP_SURPRISE_DECAY * sigma

            td_sig = float(min(max(abs(td_val), 0.0), TD_SIGNAL_CLIP))
            if not td_inited:
                td_fast = td_slow = td_sig; td_inited = True
            else:
                td_fast = (1 - TD_FAST_ALPHA) * td_fast + TD_FAST_ALPHA * td_sig
                td_slow = (1 - TD_SLOW_ALPHA) * td_slow + TD_SLOW_ALPHA * td_sig
            avg_unexpected = UNEXP_SURPRISE_DECAY * avg_unexpected + max(0.0, td_fast - td_slow)

            obs_t = next_obs_t
            total_reward += float(reward)

        reward_history.append(total_reward)
        if ep % 500 == 0:
            avg20  = float(np.mean(reward_history[-20:]))
            status = "SWITCHED" if switched and ep >= SWITCH_EP else "NORMAL  "
            print(f"  Ep {ep:5d} | {status} | R: {total_reward:7.1f} | Avg20: {avg20:7.1f} | "
                  f"Unexpected: {avg_unexpected:.3f}")

    env.close()

    run_dir  = os.path.join(OUTPUT_DIR, f"{seed}_acrobot_surrogate")
    os.makedirs(run_dir, exist_ok=True)
    csv_path = os.path.join(run_dir, f"{seed}_acrobot_surrogate.csv")
    with open(csv_path, "w") as fh:
        fh.write("episode,reward\n")
        for i, r in enumerate(reward_history):
            fh.write(f"{i},{r}\n")
    print(f"[Seed {seed}] Saved → {csv_path}")
    return reward_history


# ── Run ───────────────────────────────────────────────────────────────────────
all_rewards = {}
for s in SEEDS:
    all_rewards[s] = train(s)

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 4))
W = 100
for s, rewards in all_rewards.items():
    r = np.array(rewards, dtype=float)
    ax.plot(np.convolve(r, np.ones(W)/W, mode='valid'), alpha=0.8, label=f"seed {s}")
ax.axvline(SWITCH_EP - W//2, color='red', linestyle='--', label='Switch (ep 7500)')
ax.set_xlabel("Episode"); ax.set_ylabel(f"Reward ({W}-ep smoothed)")
ax.set_title("Acrobot Surrogate — Switch Experiment")
ax.legend()
plot_path = os.path.join(OUTPUT_DIR, "rewards.png")
plt.savefig(plot_path, dpi=120, bbox_inches='tight')
plt.show()
print(f"Plot saved → {plot_path}")

# ── Download (Colab only) ─────────────────────────────────────────────────────
try:
    import shutil
    from google.colab import files
    zip_path = "/content/acrobot_surrogate_results"
    shutil.make_archive(zip_path, 'zip', OUTPUT_DIR)
    files.download(zip_path + ".zip")
except ImportError:
    pass
