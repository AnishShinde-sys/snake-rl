#!/usr/bin/env python3
"""
Snake RL Training Script with GRPO

Features:
- TensorBoard logging for all metrics
- --render flag to visualize training
- --load flag to load existing model weights
- Graceful SIGINT handling (Ctrl+C) to save model before exit

Usage:
    python train.py                    # Train from scratch, no rendering
    python train.py --render           # Train with game visualization
    python train.py --load weights.pt  # Continue training from checkpoint
    python train.py --render --load weights.pt
"""
import argparse
import signal
import sys
import os
from datetime import datetime

import torch
from torch.utils.tensorboard import SummaryWriter

from snake_env import SnakeEnv
from model import SnakeNet, GRPOTrainer


# Global flag for graceful shutdown
shutdown_requested = False


def signal_handler(signum, frame):
    """Handle SIGINT (Ctrl+C) for graceful shutdown"""
    global shutdown_requested
    print("\n[SIGINT] Shutdown requested. Finishing current episode and saving model...")
    shutdown_requested = True


def parse_args():
    parser = argparse.ArgumentParser(description='Train Snake RL Agent with GRPO')
    parser.add_argument('--render', action='store_true',
                        help='Render the game during training')
    parser.add_argument('--load', type=str, default=None,
                        help='Path to model weights to load')
    parser.add_argument('--episodes', type=int, default=10000,
                        help='Number of episodes to train (default: 10000)')
    parser.add_argument('--lr', type=float, default=3e-4,
                        help='Learning rate (default: 3e-4)')
    parser.add_argument('--group-size', type=int, default=8,
                        help='GRPO group size (default: 8)')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='Discount factor (default: 0.99)')
    parser.add_argument('--save-dir', type=str, default='checkpoints',
                        help='Directory to save model checkpoints')
    parser.add_argument('--log-dir', type=str, default='logs',
                        help='Directory for TensorBoard logs')
    parser.add_argument('--save-interval', type=int, default=100,
                        help='Save model every N episodes (default: 100)')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device to use: cpu, cuda, mps, or auto (default: auto)')
    return parser.parse_args()


def get_device(device_arg):
    """Get the best available device"""
    if device_arg == 'auto':
        if torch.cuda.is_available():
            return 'cuda'
        elif torch.backends.mps.is_available():
            return 'mps'
        else:
            return 'cpu'
    return device_arg


def train(args):
    global shutdown_requested
    
    # Set up signal handler
    signal.signal(signal.SIGINT, signal_handler)
    
    # Get device
    device = get_device(args.device)
    print(f"Using device: {device}")
    
    # Create directories
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    # Initialize environment
    env = SnakeEnv(render_mode=args.render)
    
    # Initialize or load model
    if args.load and os.path.exists(args.load):
        print(f"Loading model from {args.load}")
        model = SnakeNet.load(args.load, device=device)
    else:
        print("Initializing new model")
        model = SnakeNet(input_size=100, hidden_size=256, output_size=4)
        model.to(device)
    
    # Initialize trainer
    trainer = GRPOTrainer(
        model=model,
        lr=args.lr,
        gamma=args.gamma,
        group_size=args.group_size,
        device=device
    )
    
    # Set up TensorBoard
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    writer = SummaryWriter(os.path.join(args.log_dir, f'snake_rl_{timestamp}'))
    
    # Training metrics
    episode_count = 0
    total_steps = 0
    best_score = 0
    
    # Rolling averages
    recent_scores = []
    recent_rewards = []
    recent_lengths = []
    window_size = 100
    
    # Death reason tracking
    death_reasons = {'wall': 0, 'self': 0, 'timeout': 0, 'win': 0, 'unknown': 0}
    
    print(f"\nStarting training...")
    print(f"Episodes: {args.episodes}")
    print(f"Group size: {args.group_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Rendering: {args.render}")
    print(f"Press Ctrl+C to stop and save model\n")
    
    try:
        while episode_count < args.episodes and not shutdown_requested:
            # Collect group of trajectories
            group = trainer.collect_group(env, render_fn=env.render if args.render else None)
            
            # Update policy
            loss, entropy = trainer.update(group)
            
            # Log metrics for each trajectory in the group
            for traj in group:
                episode_count += 1
                total_steps += traj['length']
                
                # Track scores
                score = traj['score']
                total_reward = traj['total_reward']
                ep_length = traj['length']
                reason = traj['reason']
                
                recent_scores.append(score)
                recent_rewards.append(total_reward)
                recent_lengths.append(ep_length)
                
                # Keep rolling window
                if len(recent_scores) > window_size:
                    recent_scores.pop(0)
                    recent_rewards.pop(0)
                    recent_lengths.pop(0)
                
                # Track death reasons
                death_reasons[reason] = death_reasons.get(reason, 0) + 1
                
                # Update best score
                if score > best_score:
                    best_score = score
                    # Save best model
                    best_path = os.path.join(args.save_dir, 'best_model.pt')
                    model.save(best_path)
                
                # Log to TensorBoard
                writer.add_scalar('Episode/Score', score, episode_count)
                writer.add_scalar('Episode/TotalReward', total_reward, episode_count)
                writer.add_scalar('Episode/Length', ep_length, episode_count)
                writer.add_scalar('Episode/BestScore', best_score, episode_count)
                
                # Log rolling averages
                if len(recent_scores) >= 10:
                    avg_score = sum(recent_scores) / len(recent_scores)
                    avg_reward = sum(recent_rewards) / len(recent_rewards)
                    avg_length = sum(recent_lengths) / len(recent_lengths)
                    
                    writer.add_scalar('Average/Score', avg_score, episode_count)
                    writer.add_scalar('Average/Reward', avg_reward, episode_count)
                    writer.add_scalar('Average/Length', avg_length, episode_count)
                
                # Log death reasons periodically
                if episode_count % 10 == 0:
                    total_deaths = sum(death_reasons.values())
                    if total_deaths > 0:
                        for reason_name, count in death_reasons.items():
                            writer.add_scalar(f'DeathReason/{reason_name}', 
                                            count / total_deaths * 100, episode_count)
                
                # Print progress
                if episode_count % 10 == 0:
                    avg_score_str = f"{sum(recent_scores)/len(recent_scores):.2f}" if recent_scores else "N/A"
                    print(f"Episode {episode_count}/{args.episodes} | "
                          f"Score: {score} | Best: {best_score} | "
                          f"Avg(100): {avg_score_str} | "
                          f"Loss: {loss:.4f} | Entropy: {entropy:.4f}")
            
            # Log training metrics
            writer.add_scalar('Training/Loss', loss, episode_count)
            writer.add_scalar('Training/Entropy', entropy, episode_count)
            writer.add_scalar('Training/TotalSteps', total_steps, episode_count)
            
            # Periodic save
            if episode_count % args.save_interval == 0:
                checkpoint_path = os.path.join(args.save_dir, f'checkpoint_ep{episode_count}.pt')
                model.save(checkpoint_path)
                
                # Also save latest
                latest_path = os.path.join(args.save_dir, 'latest_model.pt')
                model.save(latest_path)
            
            # Handle rendering events
            if args.render:
                import pygame
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        shutdown_requested = True
    
    except Exception as e:
        print(f"\nError during training: {e}")
        raise
    
    finally:
        # Save final model
        print("\nSaving final model...")
        final_path = os.path.join(args.save_dir, 'final_model.pt')
        model.save(final_path)
        
        # Also save as latest
        latest_path = os.path.join(args.save_dir, 'latest_model.pt')
        model.save(latest_path)
        
        # Log final stats
        print(f"\n{'='*50}")
        print(f"Training Complete!")
        print(f"{'='*50}")
        print(f"Total Episodes: {episode_count}")
        print(f"Total Steps: {total_steps}")
        print(f"Best Score: {best_score}")
        print(f"Final Avg Score (last 100): {sum(recent_scores)/len(recent_scores):.2f}" if recent_scores else "N/A")
        print(f"\nDeath Reasons:")
        total_deaths = sum(death_reasons.values())
        for reason, count in death_reasons.items():
            pct = count / total_deaths * 100 if total_deaths > 0 else 0
            print(f"  {reason}: {count} ({pct:.1f}%)")
        print(f"\nModels saved to: {args.save_dir}")
        print(f"TensorBoard logs: {args.log_dir}")
        print(f"Run 'tensorboard --logdir {args.log_dir}' to view training curves")
        
        # Close resources
        writer.close()
        env.close()


def main():
    args = parse_args()
    train(args)


if __name__ == '__main__':
    main()

