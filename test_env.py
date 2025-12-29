"""Test script to verify the Snake RL environment with rich features"""
import numpy as np
from snake_env import SnakeEnv

def test_environment():
    print("Testing Snake RL Environment with Rich Features...")
    print("=" * 60)
    
    # Create environment
    env = SnakeEnv(grid_size=10, render_mode=False)
    print("✓ Environment created successfully")
    print(f"  State space: {env.observation_space} features")
    print(f"  Action space: {env.action_space} actions")
    
    # Test reset
    state = env.reset()
    print(f"\n✓ Reset works. State shape: {state.shape}")
    assert state.shape == (35,), f"Expected state shape (35,), got {state.shape}"
    
    # Explain state features
    print("\n--- State Feature Breakdown ---")
    idx = 0
    
    print(f"Direction one-hot [0-3]:      {state[idx:idx+4]}")
    idx += 4
    
    print(f"Danger (str/right/left) [4-6]: {state[idx:idx+3]}")
    idx += 3
    
    print(f"Food direction (U/R/D/L) [7-10]: {state[idx:idx+4]}")
    idx += 4
    
    print(f"Food distance (dx, dy) [11-12]: {state[idx:idx+2]}")
    idx += 2
    
    print(f"Distance to wall [13-16]:   {state[idx:idx+4]}")
    idx += 4
    
    print(f"Depth to obstacle [17-20]:   {state[idx:idx+4]}")
    idx += 4
    
    print(f"Action preview - would die [21-24]: {state[idx:idx+4]}")
    idx += 4
    
    print(f"Action preview - would eat [25-28]: {state[idx:idx+4]}")
    idx += 4
    
    print(f"Action preview - dist change [29-32]: {state[idx:idx+4]}")
    idx += 4
    
    print(f"Snake length (normalized) [33]: {state[idx]:.3f}")
    idx += 1
    
    print(f"Adjacent body segments [34]: {state[idx]:.3f}")
    
    # Test step
    print("\n--- Testing Steps ---")
    state, reward, done, info = env.step(1)  # Move right
    print(f"✓ Step works. Reward: {reward}, Done: {done}")
    
    # Test multiple steps with action preview awareness
    print("\n--- Running Episode ---")
    state = env.reset()
    total_reward = 0
    steps = 0
    
    for _ in range(100):
        # Use action preview to make "smart" random choices
        # Avoid actions that would cause death
        would_die = state[21:25]  # Action preview - would die
        
        # Find safe actions
        safe_actions = [i for i in range(4) if would_die[i] == 0]
        
        if safe_actions:
            action = np.random.choice(safe_actions)
        else:
            action = np.random.randint(0, 4)  # All actions lead to death
        
        state, reward, done, info = env.step(action)
        total_reward += reward
        steps += 1
        
        if done:
            break
    
    print(f"✓ Ran {steps} steps with 'smart' random agent")
    print(f"  Total reward: {total_reward:.2f}")
    print(f"  Final score: {info.get('score', 0)}")
    print(f"  End reason: {info.get('reason', 'N/A')}")
    
    # Compare with pure random
    print("\n--- Comparing with Pure Random ---")
    state = env.reset()
    total_reward_random = 0
    steps_random = 0
    
    for _ in range(100):
        action = np.random.randint(0, 4)
        state, reward, done, info = env.step(action)
        total_reward_random += reward
        steps_random += 1
        if done:
            break
    
    print(f"Pure random: {steps_random} steps, reward: {total_reward_random:.2f}")
    
    # Close environment
    env.close()
    print("\n✓ Environment closed successfully")
    
    print("\n" + "=" * 60)
    print("All tests passed! ✓")
    print("\nFeature Summary:")
    print("  - 4 direction features (one-hot)")
    print("  - 3 relative danger features (straight/right/left)")
    print("  - 4 food direction features")
    print("  - 2 food distance features (normalized)")
    print("  - 4 distance to wall features (explicit wall distance)")
    print("  - 4 depth to obstacle features (wall or body)")
    print("  - 12 action preview features (4 actions × 3 outcomes)")
    print("  - 2 snake state features (length, adjacent body)")
    print("  Total: 35 features")

if __name__ == "__main__":
    test_environment()
