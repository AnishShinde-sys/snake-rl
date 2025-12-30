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
                 update_frequency=10, save_interval=100, use_mcts=False,
                 mcts_threshold=50, mcts_simulations=100, label=None, lr=None):
        self.render = render
        self.max_episodes = max_episodes
        self.update_frequency = update_frequency
        self.save_interval = save_interval
        self.should_exit = False
        self.use_mcts = use_mcts
        self.mcts_threshold = mcts_threshold
        self.mcts_simulations = mcts_simulations
        self.label = label
        
        # Setup signal handler for graceful shutdown
        signal.signal(signal.SIGINT, self.signal_handler)
        
        # Create environment
        self.env = SnakeEnv(grid_size=10, render_mode=render)
        
        # Setup device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Create agent
        state_dim = self.env.observation_space
        action_dim = self.env.action_space
        learning_rate = lr if lr is not None else 3e-3
        self.ppo_agent = PPOAgent(
            state_dim, 
            action_dim, 
            lr=learning_rate,
            gamma=0.9,
            eps_clip=0.2,
            k_epochs=4,
            device=self.device
        )
        
        # Load model if specified
        if load_model:
            if os.path.exists(load_model):
                print(f"Loading model from {load_model}")
                self.ppo_agent.load(load_model)
            else:
                print(f"Warning: Model file {load_model} not found. Starting fresh.")
        
        # Create hybrid agent if MCTS enabled
        if use_mcts:
            from mcts import HybridAgent
            self.agent = HybridAgent(
                ppo_agent=self.ppo_agent,
                mcts_threshold=mcts_threshold,
                num_simulations=mcts_simulations
            )
            print(f"MCTS enabled: threshold={mcts_threshold}, simulations={mcts_simulations}")
        else:
            self.agent = self.ppo_agent
        
        # Create checkpoint directory
        os.makedirs("checkpoints", exist_ok=True)
        
        # Determine run number
        self.run_number = self._get_next_run_number()
        if self.label:
            print(f"Starting training with label: {self.label}")
        else:
            print(f"Starting Run #{self.run_number:03d}")
        
        # Training stats
        self.episode_rewards = []
        self.episode_scores = []
        self.timestep = 0
        self.episode_count = 0
        self.start_time = datetime.now()
        self.best_score = 0  # Track best score for saving best models
    
    def _get_next_run_number(self):
        """Find the next available run number"""
        existing_runs = glob.glob("checkpoints/run_*_*.pt")
        if not existing_runs:
            return 1
        
        run_numbers = []
        for f in existing_runs:
            basename = os.path.basename(f)
            try:
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
    
    def save_model(self, episode, is_final=False, is_best=False):
        """Save model checkpoint with run number or custom label"""
        if self.label:
            # Use custom label
            if is_final:
                filepath = f"checkpoints/{self.label}_final.pt"
            elif is_best:
                filepath = f"checkpoints/{self.label}_best.pt"
            else:
                filepath = f"checkpoints/{self.label}_episode_{episode}.pt"
        else:
            # Use automatic run numbering
            if is_final:
                filepath = f"checkpoints/run_{self.run_number:03d}_final.pt"
            elif is_best:
                filepath = f"checkpoints/run_{self.run_number:03d}_best.pt"
            else:
                filepath = f"checkpoints/run_{self.run_number:03d}_episode_{episode}.pt"
        self.agent.save(filepath)
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
        
        # Add MCTS stats if enabled
        if self.use_mcts and hasattr(self.agent, 'get_stats'):
            if episode % 100 == 0:  # Log MCTS stats every 100 episodes
                log_msg += f" | {self.agent.get_stats()}"
        
        print(log_msg)
        
        with open("training_log.txt", "a") as f:
            f.write(log_msg + "\n")
    
    def train(self):
        """Main training loop"""
        print("=" * 80)
        if self.use_mcts:
            if self.mcts_threshold <= 1:
                print("Starting Snake RL Training with MCTS (AlphaZero-style)")
                print("WARNING: This is SLOW (~50ms/move) but generates high-quality data")
            else:
                print("Starting Snake RL Training with PPO + MCTS")
        else:
            print("Starting Snake RL Training with PPO")
        print("=" * 80)
        print(f"Max Episodes: {self.max_episodes}")
        print(f"Update Frequency: Every {self.update_frequency} episodes")
        print(f"Save Interval: {self.save_interval}")
        print(f"Render Mode: {self.render}")
        if self.use_mcts:
            print(f"MCTS Threshold: {self.mcts_threshold}")
            print(f"MCTS Simulations: {self.mcts_simulations}")
        print("=" * 80)
        print("\nPress Ctrl+C to stop training and save model\n")
        
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
                # Get action mask
                action_mask = self.env.get_action_mask()
                
                # Select action - different signature for hybrid agent
                if self.use_mcts:
                    action = self.agent.select_action(state, self.env, action_mask)
                else:
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
                
                if self.render:
                    self.env.render()
                
                if done:
                    end_reason = info.get("reason", "")
            
            # Store episode stats
            score = info.get("score", 0)
            self.episode_rewards.append(episode_reward)
            self.episode_scores.append(score)
            self.episode_count += 1
            
            # Update policy
            if self.episode_count % self.update_frequency == 0:
                self.agent.update()
            
            self.log_episode(episode, score, episode_reward, steps, end_reason)
            
            # Save best model if score improved
            if score > self.best_score:
                self.best_score = score
                self.save_model(episode, is_best=True)
                print(f"  *** NEW BEST SCORE: {score} ***")
            
            # Adaptive save interval: save more frequently early, less later
            # For very long runs (1M+ episodes), save every 10K after first 100K
            if self.max_episodes > 100000:
                if episode <= 100000:
                    save_interval = self.save_interval  # Use configured interval
                else:
                    save_interval = 10000  # Save every 10K episodes after 100K
            else:
                save_interval = self.save_interval
            
            if episode % save_interval == 0:
                self.save_model(episode)
            
            if self.should_exit:
                break
        
        print("\n" + "=" * 80)
        if self.label:
            print(f"Training completed or interrupted - Label: {self.label}")
        else:
            print(f"Training completed or interrupted - Run #{self.run_number:03d}")
        print("=" * 80)
        self.save_model(episode, is_final=True)
        
        if self.episode_scores:
            print(f"\nFinal Statistics:")
            print(f"Total Episodes: {len(self.episode_scores)}")
            print(f"Average Score: {np.mean(self.episode_scores):.2f}")
            print(f"Max Score: {np.max(self.episode_scores)}")
            print(f"Average Reward: {np.mean(self.episode_rewards):.2f}")
            print(f"Total Time: {(datetime.now() - self.start_time).total_seconds():.1f}s")
            if self.use_mcts and hasattr(self.agent, 'get_stats'):
                print(f"MCTS Usage: {self.agent.get_stats()}")
        
        self.env.close()
        print("\nExiting gracefully. Model saved.")


def main():
    parser = argparse.ArgumentParser(description="Train Snake RL agent with PPO (+ optional MCTS)")
    parser.add_argument("--render", action="store_true", 
                       help="Render the game during training")
    parser.add_argument("--load", type=str, default=None,
                       help="Path to model checkpoint to load")
    parser.add_argument("--episodes", type=int, default=10000,
                       help="Maximum number of episodes to train")
    parser.add_argument("--update-frequency", type=int, default=10,
                       help="Update policy every N episodes")
    parser.add_argument("--save-interval", type=int, default=1000,
                       help="Save model every N episodes")
    parser.add_argument("--label", type=str, default=None,
                       help="Custom label for saved checkpoints (e.g., 'mcts_training', 'refined_v2')")
    parser.add_argument("--lr", type=float, default=None,
                       help="Learning rate (default: 3e-3, use 1e-3 for long training)")
    
    # MCTS options
    parser.add_argument("--mcts", action="store_true",
                       help="Enable MCTS for high-score situations")
    parser.add_argument("--mcts-threshold", type=int, default=50,
                       help="Snake length threshold to activate MCTS (default: 50)")
    parser.add_argument("--mcts-simulations", type=int, default=100,
                       help="Number of MCTS simulations per move (default: 100)")
    parser.add_argument("--mcts-always", action="store_true",
                       help="Always use MCTS (AlphaZero-style training, SLOW but high quality)")
    
    args = parser.parse_args()
    
    # --mcts-always sets threshold to 1 (MCTS from the start)
    mcts_threshold = 1 if args.mcts_always else args.mcts_threshold
    use_mcts = args.mcts or args.mcts_always
    
    trainer = Trainer(
        render=args.render,
        load_model=args.load,
        max_episodes=args.episodes,
        update_frequency=args.update_frequency,
        save_interval=args.save_interval,
        use_mcts=use_mcts,
        mcts_threshold=mcts_threshold,
        mcts_simulations=args.mcts_simulations,
        label=args.label,
        lr=args.lr
    )
    
    trainer.train()


if __name__ == "__main__":
    main()
