# ACE

Research project investigating neuromodulated learning in spiking neural networks for reinforcement learning tasks.

Abstract: 
The current standard biologically plausible training rule for spiking network reinforcement learning (RL), reward-modulated spike-timing-dependent plasticity (RSTDP), often has inferior performance compared to biologically implausible methods. In addition, RSTDP is slow, purely local, and fragile in non-stationary environments. This paper presents ACE (the first three letters of acetylcholine), a novel, biologically inspired learning rule that extends RSTDP's three factors to five. ACE introduces modulating factors based on neuromodulators acetylcholine (ACh) and norepinephrine (NE), which modulate learning rate and random noise, respectively. Actor-critic systems trained using ACE achieve higher reward over the fixed window following a task switch than a hyperparameter-tuned classic RSTDP baseline. On the Switch Bandit tasks, ACE recovers better than the baseline; on Switch CartPole, the classic baseline fails to re-adapt within the training horizon while ACE recovers control. ACE does not match non-local surrogate-gradient training but still outperforms classic RSTDP in the tested benchmarks. ACE demonstrates that biologically-inspired neuromodulation increases the adaptability of SNNs in the tested non-stationary environments.


## Usage

```bash
pip install -r requirements.txt

# Switch Bandit (20k episodes, switch at 10k) -> results/bandit/runs/1_ace/
python src/bandit/ace_sb.py --seed 1

# Switch CartPole (10k episodes, controls invert at 5k) -> results/cartpole/runs/1_flagship/
python src/cartpole/flagship.py --seed 1
```

Classic RSTDP baselines: `src/cartpole/classic.py`, or `src/bandit/run_batch.py` for all bandit conditions. Paper figures: `python src/figures/make_figs.py`.



