# Icarus

Research project investigating neuromodulated learning in spiking neural networks for reinforcement learning tasks.
## Updates:
There have been multiple upgrades made recently, mainly to make the tests more biologically plausible. For both switch-bandit and switch-cartpole, the ACh and NE have been decoupled from the original surprise signal and now rely on variance and TD-error as expected and unexpected surprise signals, respectively. Furthermore, the critic is now trained using TD-stdp for both tasks. The new scripts to run are cartpole_successful/flagship.py and sb/icarus_upgraded.py.


## Experiments

### CartPole Switch Task
### Switch Bandit Task
### Switch Lunar Landing Task (still in progress)



## Results

Results are saved to `cartpole/runs/SEED_EXPERIMENT/`:
- `*.png` - Training plots
- `*.csv` - Episode rewards and metrics
