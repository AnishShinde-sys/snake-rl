import argparse
import signal
import sys
import os
import glob
import torch
import numpy as np
from datetime import datetime
from snake_env import SnakeEnv
from ppo_agent import PPOAgent


class Trainer:
    """Main training class with logging and graceful shutdown"""
    
    def __init__(self, render=False, load_model=None, max_episodes=10000, 
                 update_frequency=10, save_interval=100):
        self.render = render
        self.max_episodes = max_episodes
        self.update_frequency = update_frequency  # Update every N episodes
        self.save_interval = save_interval
        self.should_exit = False
        
        # Setup signal handler for graceful shutdown
        signal.signal(signal.SIGINT, self.signal_handler)
        
        # Create environment
        self.env = SnakeEnv(grid_size=10, render_mode=render)
        
        # Setup device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Create agent with better hyperparameters for Snake
        state_dim = self.env.observation_space
        action_dim = self.env.action_space
        self.agent = PPOAgent(
            state_dim, 
            action_dim, 
            lr=3e-3,  # Higher learning rate for faster learning
            gamma=0.9,  # Lower gamma for faster credit assignment
            eps_clip=0.2,
            k_epochs=4,  # Fewer epochs to avoid overfitting
            device=self.device
        )
        
        # Load model if specified
        if load_model:
            if os.path.exists(load_model):
                print(f"Loading model from {load_model}")
                self.agent.load(load_model)
            else:
                print(f"Warning: Model file {load_model} not found. Starting fresh.")
        
        # Create checkpoint directory
        os.makedirs("checkpoints", exist_ok=True)
        
        # Determine run number (find highest existing run and increment)
        self.run_number = self._get_next_run_number()
        print(f"Starting Run #{self.run_number:03d}")
        
        # Training stats
        self.episode_rewards = []
        self.episode_scores = []
        self.timestep = 0
        self.episode_count = 0
        self.start_time = datetime.now()
    
    def _get_next_run_number(self):
        """Find the next available run number"""
        existing_runs = glob.glob("checkpoints/run_*_*.pt")
        if not existing_runs:
            return 1
        
        # Extract run numbers from filenames
        run_numbers = []
        for f in existing_runs:
            basename = os.path.basename(f)
            try:
                # Format: run_XXX_episode_YYYY.pt or run_XXX_final.pt
                parts = basename.split('_')
                if len(parts) >= 2:
                    run_num = int(parts[1])
                    run_numbers.append(run_num)
            except (ValueError, IndexError):
                continue
        
        if run_numbers:
            return max(run_numbers) + 1
        return 1
    
    def signal_handler(self, sig, frame):
        """Handle SIGINT (Ctrl+C) for graceful shutdown"""
        print("\n\n[SIGINT] Received interrupt signal. Saving model and exiting gracefully...")
        self.should_exit = True
    
    def save_model(self, episode, is_final=False):
        """Save model checkpoint with run number"""
        if is_final:
            filepath = f"checkpoints/run_{self.run_number:03d}_final.pt"
        else:
            filepath = f"checkpoints/run_{self.run_number:03d}_episode_{episode}.pt"
        self.agent.save(filepath)
        # Also save as latest
        latest_path = "checkpoints/latest_model.pt"
        self.agent.save(latest_path)
        print(f"Model saved to {filepath}")
    
    def log_episode(self, episode, score, reward, steps, reason=""):
        """Log episode statistics"""
        avg_reward = np.mean(self.episode_rewards[-100:]) if self.episode_rewards else 0
        avg_score = np.mean(self.episode_scores[-100:]) if self.episode_scores else 0
        max_score = max(self.episode_scores) if self.episode_scores else 0
        
        elapsed = (datetime.now() - self.start_time).total_seconds()
        
        log_msg = (f"Episode: {episode:5d} | "
                  f"Score: {score:3d} | "
                  f"Max: {max_score:3d} | "
                  f"Avg (100): {avg_score:5.2f} | "
                  f"Time: {elapsed:.1f}s")
        
        if reason:
            log_msg += f" | {reason}"
        
        print(log_msg)
        
        # Log to file
        with open("training_log.txt", "a") as f:
            f.write(log_msg + "\n")
    
    def train(self):
        """Main training loop"""
        print("=" * 80)
        print("Starting Snake RL Training with PPO")
        print("=" * 80)
        print(f"Max Episodes: {self.max_episodes}")
        print(f"Update Frequency: Every {self.update_frequency} episodes")
        print(f"Save Interval: {self.save_interval}")
        print(f"Render Mode: {self.render}")
        print("=" * 80)
        print("\nPress Ctrl+C to stop training and save model\n")
        
        # Clear previous log
        with open("training_log.txt", "w") as f:
            f.write(f"Training started at {self.start_time}\n")
            f.write("=" * 80 + "\n")
        
        for episode in range(1, self.max_episodes + 1):
            if self.should_exit:
                break
            
            state = self.env.reset()
            episode_reward = 0
            steps = 0
            done = False
            end_reason = ""
            
            while not done and not self.should_exit:
                # Get action mask (safe actions) from environment
                action_mask = self.env.get_action_mask()
                
                # Select action with mask - prevents obviously bad moves
                action = self.agent.select_action(state, action_mask)
                
                # Take action
                next_state, reward, done, info = self.env.step(action)
                
                # Store transition
                self.agent.store_transition(reward, done)
                
                # Update state and stats
                state = next_state
                episode_reward += reward
                steps += 1
                self.timestep += 1
                
                # Render if enabled
                if self.render:
                    self.env.render()
                
                if done:
                    end_reason = info.get("reason", "")
            
            # Store episode stats
            score = info.get("score", 0)
            self.episode_rewards.append(episode_reward)
            self.episode_scores.append(score)
            self.episode_count += 1
            
            # Update policy after collecting N episodes
            if self.episode_count % self.update_frequency == 0:
                self.agent.update()
            
            # Log episode
            self.log_episode(episode, score, episode_reward, steps, end_reason)
            
            # Save model periodically
            if episode % self.save_interval == 0:
                self.save_model(episode)
            
            # Check for exit signal
            if self.should_exit:
                break
        
        # Final save
        print("\n" + "=" * 80)
        print(f"Training completed or interrupted - Run #{self.run_number:03d}")
        print("=" * 80)
        self.save_model(episode, is_final=True)
        
        # Print final statistics
        if self.episode_scores:
            print(f"\nFinal Statistics:")
            print(f"Total Episodes: {len(self.episode_scores)}")
            print(f"Average Score: {np.mean(self.episode_scores):.2f}")
            print(f"Max Score: {np.max(self.episode_scores)}")
            print(f"Average Reward: {np.mean(self.episode_rewards):.2f}")
            print(f"Total Time: {(datetime.now() - self.start_time).total_seconds():.1f}s")
        
        # Close environment
        self.env.close()
        print("\nExiting gracefully. Model saved.")


def main():
    parser = argparse.ArgumentParser(description="Train Snake RL agent with PPO")
    parser.add_argument("--render", action="store_true", 
                       help="Render the game during training")
    parser.add_argument("--load", type=str, default=None,
                       help="Path to model checkpoint to load")
    parser.add_argument("--episodes", type=int, default=10000,
                       help="Maximum number of episodes to train")
    parser.add_argument("--update-frequency", type=int, default=10,
                       help="Update policy every N episodes")
    parser.add_argument("--save-interval", type=int, default=1000,
                       help="Save model every N episodes (milestones: 1000, 2000, etc.)")
    
    args = parser.parse_args()
    
    # Create and run trainer
    trainer = Trainer(
        render=args.render,
        load_model=args.load,
        max_episodes=args.episodes,
        update_frequency=args.update_frequency,
        save_interval=args.save_interval
    )
    
    trainer.train()


if __name__ == "__main__":
    main()

