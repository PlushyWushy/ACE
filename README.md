# RSTDP - Reward-modulated Spike-Timing-Dependent Plasticity

Research project investigating neuromodulated learning in spiking neural networks for reinforcement learning tasks.

## Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/RSTDP.git
cd RSTDP
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run Experiments

**Classic CartPole (SNN):**
```bash
python3 cartpole/classic.py --episodes 4000
```

**MLP CartPole:**
```bash
python3 cartpole/mlp_classic.py --episodes 4000
```

## Project Structure

```
RSTDP/
├── cartpole/           # CartPole experiments
│   ├── classic.py      # SNN with R-STDP
│   ├── mlp_classic.py  # MLP baseline
│   └── ...             # Other variants
└── README.md
```

## Experiments

### CartPole Switch Task
- Episodes 0-500: Normal controls
- Episodes 500+: Inverted controls (tests adaptability)

## Workflow Across Computers

```bash
# On Computer A
git pull                    # Get latest changes
# ... make changes ...
git add .
git commit -m "message"
git push

# On Computer B
git pull                    # Sync changes
# ... continue work ...
```

## Results

Results are saved to `cartpole/runs/SEED_EXPERIMENT/`:
- `*.png` - Training plots
- `*.csv` - Episode rewards and metrics
