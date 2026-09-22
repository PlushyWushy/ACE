import os
import re

spike_path = "/Users/a../Desktop/Icarus/cartpole_successful/spike.py"
tdstdpll_path = "/Users/a../Desktop/Icarus/ll/tdstdpll.py"
out_path = "/Users/a../Desktop/Icarus/ll/spike_ll.py"

with open(spike_path, "r") as f:
    spike_code = f.read()

with open(tdstdpll_path, "r") as f:
    tdstdpll_code = f.read()

# 1. State_to_spike_Fourier extraction from tdstdpll_code
match = re.search(r'class State_to_spike_Fourier.*?Return true if f_output', tdstdpll_code, re.DOTALL)
if not match:
    # Try different regex to capture class State_to_spike_Fourier
    match = re.search(r'(class State_to_spike_Fourier.*?)\nclass uniform_policy', tdstdpll_code, re.DOTALL)
fourier_code = match.group(1) if match else ""

# Extract batch_envs from tdstdpll.py
match_env = re.search(r'(class batch_envs.*?\n)(?=def plot_performance|# =|class |def )', tdstdpll_code.split("# 5. Main")[0], re.DOTALL)
if not match_env:
    match_env = re.search(r'(class batch_envs.*?)\n# ==========================================\n# 5.', tdstdpll_code, re.DOTALL)
env_code = match_env.group(1) if match_env else ""

# Now build the new script by modifying spike_code
new_code = spike_code.replace("env_name = config.get(\"USER\", \"env_name\")", """env_name = config.get("USER", "env_name", fallback="LunarLander-v2")""")
# Replace State_to_spike_RBF or bin
# Actually, it's easier to just find State_to_spike_bin and State_to_spike_RBF and insert State_to_spike_Fourier
new_code = re.sub(r'class State_to_spike_bin.*?(?=class State_to_spike_RBF)', fourier_code + '\n\n', new_code, flags=re.DOTALL)

# Replace batch_envs class in spike.py
new_code = re.sub(r'class batch_envs.*?def reset\(self, index=None\).*?return self._state\n\n\n', env_code + '\n\n', new_code, flags=re.DOTALL)

# Default to False for switch spike
new_code = new_code.replace("DEFAULT_USE_SWITCH_LR_SPIKE = True", "DEFAULT_USE_SWITCH_LR_SPIKE = False")
new_code = new_code.replace('env = batch_envs("CartPole-v1"', 'env = batch_envs(env_name')

new_code = new_code.replace('print("Start generating state encoder...")', """print("Start generating state encoder...")
    
    bin_min = np.array([-1, -1, -1, -1, -1, -1, 0, 0])
    bin_max = np.array([1, 1, 1, 1, 1, 1, 1, 1])
    state_to_spike = State_to_spike_Fourier(
        time=config.getint('USER', 'time'),
        bin_min=bin_min,
        bin_max=bin_max,
        k=config.getint('USER', 'forier_order'),
        basis=config.getboolean('USER', 'basis'),
        soft=config.getboolean('USER', 'input_type') == 1,
        cross_term=config.getboolean('USER', 'forier_cross_term'),
        double=config.getboolean('USER', 'forier_double'),
        rep=config.getint('USER', 'rep')
    )""")

# Remove the old state_to_spike lines
new_code = re.sub(r'if input_type == 0:.*?else:.*?rep=rep\n    \)', '', new_code, flags=re.DOTALL)

# Also fix the num_actions logic
new_code = new_code.replace("num_actions = 2", "num_actions = 4")

# Also the switch logic in spike.py 
# if (eps + 1) == switch_at: actions = [1, 0]... we can comment out the cartpole switch logic
new_code = re.sub(r'# --- ENVIRONMENT SWITCH \(CartPole\) ---.*?(?=\n\s*(?:if eps % print_every == 0|eps \+= 1))', '', new_code, flags=re.DOTALL)
new_code = new_code.replace('actions = [0, 1]', 'actions = [0, 1, 2, 3]')

with open(out_path, "w") as f:
    f.write(new_code)
print("Merge script completed!")
