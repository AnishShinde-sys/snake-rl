# Snake-RL: Deep Reinforcement Learning for Snake Game

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Implementation of Proximal Policy Optimization (PPO) to master Snake on a 10×10 grid. Current best: **91/97 score** (93.8% optimal play).

## Features

- 30-dimensional feature-engineered state space
- Relative action space (Straight, Right, Left)
- Action masking with tail reachability (BFS)
- 2-step lookahead for trap avoidance
- Separate Actor-Critic networks
- Automatic checkpoint management

## Installation

```bash
git clone git@github.com:AnishShinde-sys/snake-rl.git
cd snake-rl
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage

```bash
# Train from scratch
python train.py --episodes 10000

# Train with visualization
python train.py --render --episodes 1000

# Continue from checkpoint
python train.py --load checkpoints/run_005_final.pt --episodes 5000

# Prevent sleep during training (macOS)
caffeinate -i python train.py --episodes 50000
```

## Results

| Metric | Value |
|--------|-------|
| Max Score | 91/97 (93.8%) |
| Average Score | ~33-34 |
| Training Episodes | 15,000+ |
| Training Time | ~25 minutes |

## Architecture

### Neural Network

```
ACTOR:  Input(30) → Linear(256) → Tanh → Linear(256) → Tanh → Linear(3) → Softmax
CRITIC: Input(30) → Linear(256) → ReLU → Linear(256) → ReLU → Linear(1)
```

### State Representation (30 Features)

- Danger signals (3): Immediate collision per action
- Food direction (3): Relative food position
- Food distance (1): Normalized Manhattan distance
- Wall distance (3): Distance to walls
- Obstacle depth (3): Depth perception
- Action previews (9): Would die/eat/distance change
- Snake state (2): Length, adjacent body
- Lookahead (6): Min/max safe moves 2 steps ahead

### Hyperparameters

```python
Learning Rate:     3e-3
Gamma:             0.9
Epsilon Clip:      0.2
K-Epochs:          4
Entropy Coeff:     0.05
```

## Key Innovations

**Relative Action Space**: Actions are relative to current direction (Straight/Right/Left), creating direct mapping between danger signals and actions.

**Tail Reachability**: BFS-based check to ensure snake maintains path to tail at high scores, preventing self-trapping.

**Action Masking**: Only considers safe actions that don't cause immediate death and maintain escape routes.

**Separate Actor-Critic**: Independent networks prevent critic gradients from dominating policy learning.

## Repository Structure

```
snake-rl/
├── snake_env.py      # RL environment (30-feature state)
├── ppo_agent.py      # PPO agent + ActorCritic network
├── train.py          # Training loop with checkpointing
├── snake.py          # Original human-playable game
├── test_env.py       # Environment testing
├── requirements.txt  # Dependencies
└── checkpoints/      # Model weights
```

## Reward Shaping

| Event | Reward |
|-------|--------|
| Eat Food | +10 |
| Move Closer | +1.0 |
| Move Away | -1.5 |
| Death | -10 |
| Tail Reachable (len>30) | +2.0 |
| Tail Unreachable (len>30) | -5.0 |

## References

- [PPO Paper](https://arxiv.org/abs/1707.06347) - Proximal Policy Optimization Algorithms (Schulman et al., 2017)

## License

MIT License
