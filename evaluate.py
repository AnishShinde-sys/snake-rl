"""
Evaluate a trained PPO model with optional MCTS enhancement.

Usage:
    # Just PPO (fast)
    python evaluate.py --load checkpoints/latest_model.pt --episodes 10 --render
    
    # PPO + MCTS (slower but smarter at high scores)
    python evaluate.py --load checkpoints/latest_model.pt --mcts --episodes 10 --render
    
    # MCTS from the start (slowest, most accurate)
    python evaluate.py --load checkpoints/latest_model.pt --mcts --mcts-threshold 1 --render
"""

import argparse
import torch
import numpy as np
from snake_env import SnakeEnv
from ppo_agent import PPOAgent
from mcts import MCTS


def evaluate_ppo(model_path, episodes=10, render=True):
    """Evaluate using pure PPO"""
    env = SnakeEnv(grid_size=10, render_mode=render)
    agent = PPOAgent(state_dim=30, action_dim=3)
    agent.load(model_path)
    
    scores = []
    for ep in range(1, episodes + 1):
        state = env.reset()
        done = False
        while not done:
            mask = env.get_action_mask()
            action = agent.select_action(state, mask)
            state, reward, done, info = env.step(action)
            if render:
                env.render()
        
        scores.append(info['score'])
        print(f"Episode {ep}: Score = {info['score']}")
        agent.clear_memory()
    
    env.close()
    return scores


def evaluate_mcts(model_path, episodes=10, render=True, mcts_threshold=50, num_simulations=100):
    """Evaluate using MCTS (with PPO network as guide)"""
    env = SnakeEnv(grid_size=10, render_mode=render)
    agent = PPOAgent(state_dim=30, action_dim=3)
    agent.load(model_path)
    
    # Create MCTS with trained policy network
    mcts = MCTS(
        policy_network=agent.policy,
        grid_size=10,
        num_simulations=num_simulations,
        device=agent.device
    )
    
    scores = []
    mcts_used = 0
    ppo_used = 0
    
    for ep in range(1, episodes + 1):
        state = env.reset()
        done = False
        ep_mcts = 0
        ep_ppo = 0
        
        while not done:
            snake_len = len(env.snake)
            
            if snake_len >= mcts_threshold:
                # Use MCTS for careful planning
                action = mcts.select_action(env, temperature=0.1)  # Low temp = deterministic
                ep_mcts += 1
            else:
                # Use PPO for fast play
                mask = env.get_action_mask()
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(state).to(agent.device)
                    action_probs, _ = agent.policy(state_tensor)
                    if mask is not None:
                        mask_tensor = torch.FloatTensor(mask).to(agent.device)
                        action_probs = action_probs * mask_tensor
                        action_probs = action_probs / action_probs.sum()
                    action = torch.argmax(action_probs).item()  # Deterministic
                ep_ppo += 1
            
            state, reward, done, info = env.step(action)
            if render:
                env.render()
        
        scores.append(info['score'])
        mcts_used += ep_mcts
        ppo_used += ep_ppo
        print(f"Episode {ep}: Score = {info['score']} (PPO: {ep_ppo}, MCTS: {ep_mcts})")
    
    env.close()
    print(f"\nTotal: PPO={ppo_used}, MCTS={mcts_used}")
    return scores


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained Snake RL model")
    parser.add_argument("--load", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--episodes", type=int, default=10, help="Number of episodes")
    parser.add_argument("--render", action="store_true", help="Render the game")
    parser.add_argument("--mcts", action="store_true", help="Use MCTS for evaluation")
    parser.add_argument("--mcts-threshold", type=int, default=50, help="Snake length to activate MCTS")
    parser.add_argument("--mcts-simulations", type=int, default=100, help="MCTS simulations per move")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print(f"Evaluating model: {args.load}")
    print(f"Mode: {'MCTS (threshold={})'.format(args.mcts_threshold) if args.mcts else 'PPO only'}")
    print("=" * 60)
    
    if args.mcts:
        scores = evaluate_mcts(
            args.load, 
            episodes=args.episodes, 
            render=args.render,
            mcts_threshold=args.mcts_threshold,
            num_simulations=args.mcts_simulations
        )
    else:
        scores = evaluate_ppo(args.load, episodes=args.episodes, render=args.render)
    
    print("\n" + "=" * 60)
    print("Results:")
    print(f"  Episodes: {len(scores)}")
    print(f"  Average Score: {np.mean(scores):.2f}")
    print(f"  Max Score: {max(scores)}")
    print(f"  Min Score: {min(scores)}")
    print("=" * 60)


if __name__ == "__main__":
    main()

