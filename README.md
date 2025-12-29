# Snake Reinforcement Learning with PPO

A reinforcement learning implementation for the classic Snake game using Proximal Policy Optimization (PPO).

## Features

- **10x10 Grid Environment**: Simplified Snake game on a 10x10 grid
- **PPO Algorithm**: State-of-the-art policy gradient method
- **100-Dimensional State Space**: Flattened grid representation (0=empty, 1=body, 2=head, 3=food)
- **Actor-Critic Architecture**: Neural network with shared layers and separate policy/value heads
- **Episode Logging**: Comprehensive training statistics logged to file
- **Model Checkpointing**: Automatic saving every N episodes
- **Graceful Shutdown**: SIGINT (Ctrl+C) handling with model saving
- **Render Mode**: Optional visualization during training

## Installation

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Training from Scratch

```bash
python train.py
```

### Training with Rendering

```bash
python train.py --render
```

### Continue Training from Checkpoint

```bash
python train.py --load checkpoints/latest_model.pt
```

### Custom Training Parameters

```bash
python train.py --episodes 5000 --update-timestep 1000 --save-interval 50
```

### Play Original Snake Game (Human)

```bash
python snake.py
```

## Command Line Arguments

- `--render`: Enable game visualization during training
- `--load PATH`: Load model weights from checkpoint
- `--episodes N`: Maximum number of episodes to train (default: 10000)
- `--update-timestep N`: Update policy every N timesteps (default: 2000)
- `--save-interval N`: Save model every N episodes (default: 100)

## Project Structure

```
snake-rl/
├── snake.py           # Original Snake game (10x10 grid)
├── snake_env.py       # RL environment wrapper
├── ppo_agent.py       # PPO agent and neural network
├── train.py           # Main training script
├── requirements.txt   # Python dependencies
├── checkpoints/       # Saved model weights
└── training_log.txt   # Episode statistics
```

## Environment Details

### State Space
- 100-dimensional vector (10x10 grid flattened)
- Values: 0 (empty), 1 (snake body), 2 (snake head), 3 (food)

### Action Space
- 4 discrete actions: UP (0), RIGHT (1), DOWN (2), LEFT (3)
- 180-degree turns are prevented

### Rewards
- +10: Eating food
- +0.1: Surviving each step
- -10: Collision (wall or self)
- -5: Timeout (stuck in loop)

### Episode Termination
- Wall collision
- Self collision
- Maximum steps reached (grid_size² × 10)

## Neural Network Architecture

```
Input (100) → Shared Layer (128) → Shared Layer (128)
                                    ├→ Actor (64) → Actions (4)
                                    └→ Critic (64) → Value (1)
```

## Training Tips

1. **Start without rendering** for faster training
2. **Monitor the logs** to track progress
3. **Use checkpoints** to resume training
4. **Adjust hyperparameters** if agent gets stuck
5. **Press Ctrl+C** to stop and save model anytime

## Troubleshooting

### Agent keeps hitting walls
- Increase training episodes
- Adjust reward structure
- Lower learning rate

### Training is slow
- Disable rendering (`--render` flag)
- Increase update timestep
- Use GPU if available

### Model not improving
- Check if agent is exploring enough
- Adjust epsilon clipping parameter
- Increase network capacity

## License

MIT License

