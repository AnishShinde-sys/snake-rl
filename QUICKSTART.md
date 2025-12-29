# Quick Start Guide

## Setup (One-time)

```bash
# Activate virtual environment
source venv/bin/activate

# Verify installation
python test_env.py
```

## Training Commands

### Basic Training (No Rendering - Fast)
```bash
python train.py
```

### Training with Visualization
```bash
python train.py --render
```

### Resume from Checkpoint
```bash
python train.py --load checkpoints/latest_model.pt
```

### Quick Test (5 episodes)
```bash
python train.py --episodes 5
```

### Long Training Session
```bash
python train.py --episodes 50000 --save-interval 500
```

## Monitoring Progress

### Watch Training Log in Real-time
```bash
tail -f training_log.txt
```

### Check Latest Episode Stats
```bash
tail -20 training_log.txt
```

## Tips

1. **Start without --render** for faster training (10-100x speedup)
2. **Use Ctrl+C** to stop training anytime - model will be saved automatically
3. **Check checkpoints/** folder for saved models
4. **Monitor avg score** - it should increase over time
5. **Be patient** - RL training can take thousands of episodes

## Expected Progress

- Episodes 1-100: Snake learns to avoid walls
- Episodes 100-500: Snake starts moving toward food
- Episodes 500-2000: Snake occasionally eats food
- Episodes 2000+: Snake consistently scores 5-10+ points

## Troubleshooting

**Import Error**: Make sure venv is activated
```bash
source venv/bin/activate
```

**Training too slow**: Remove --render flag

**Model not improving**: Try training longer (10000+ episodes)

