"""Test script to verify the Snake RL environment with RELATIVE actions"""
import numpy as np
from snake_env import SnakeEnv

def test_environment():
    print("Testing Snake RL Environment with RELATIVE Actions...")
    print("=" * 60)
    
    # Create environment
    env = SnakeEnv(grid_size=10, render_mode=False)
    print("✓ Environment created successfully")
    print(f"  State space: {env.observation_space} features")
    print(f"  Action space: {env.action_space} actions (RELATIVE: straight/right/left)")
    
    # Test reset
    state = env.reset()
    print(f"\n✓ Reset works. State shape: {state.shape}")
    assert state.shape == (24,), f"Expected state shape (24,), got {state.shape}"
    
    # Explain state features
    print("\n--- State Feature Breakdown (ALL RELATIVE) ---")
    idx = 0
    
    print(f"Danger (str/right/left) [0-2]:   {state[idx:idx+3]}")
    idx += 3
    
    print(f"Food direction rel [3-5]:        {state[idx:idx+3]}")
    idx += 3
    
    print(f"Food distance (normalized) [6]:  {state[idx]:.3f}")
    idx += 1
    
    print(f"Wall distance rel [7-9]:         {state[idx:idx+3]}")
    idx += 3
    
    print(f"Depth to obstacle rel [10-12]:   {state[idx:idx+3]}")
    idx += 3
    
    print(f"Would die (per action) [13-15]:  {state[idx:idx+3]} <- DIRECT MAPPING!")
    idx += 3
    
    print(f"Would eat (per action) [16-18]:  {state[idx:idx+3]}")
    idx += 3
    
    print(f"Dist change (per action) [19-21]:{state[idx:idx+3]}")
    idx += 3
    
    print(f"Snake length (normalized) [22]:  {state[idx]:.3f}")
    idx += 1
    
    print(f"Adjacent body segments [23]:     {state[idx]:.3f}")
    
    # Test step with RELATIVE actions
    print("\n--- Testing Steps with RELATIVE Actions ---")
    state, reward, done, info = env.step(0)  # Go straight
    print(f"✓ Action 0 (straight): Reward: {reward}, Done: {done}")
    
    state, reward, done, info = env.step(1)  # Turn right
    print(f"✓ Action 1 (turn right): Reward: {reward}, Done: {done}")
    
    state, reward, done, info = env.step(2)  # Turn left
    print(f"✓ Action 2 (turn left): Reward: {reward}, Done: {done}")
    
    # Test "smart" agent using DIRECT danger mapping
    print("\n--- Running Episode with Direct Danger Mapping ---")
    state = env.reset()
    total_reward = 0
    steps = 0
    
    for _ in range(200):
        # DIRECT MAPPING: action i is safe if danger[i] == 0
        # danger is at indices 0,1,2 (same as would_die at 13,14,15)
        danger = state[0:3]  # straight, right, left
        
        # Find safe actions (DIRECT mapping!)
        safe_actions = [i for i in range(3) if danger[i] == 0]
        
        if safe_actions:
            action = np.random.choice(safe_actions)
        else:
            action = np.random.randint(0, 3)  # All actions dangerous
        
        state, reward, done, info = env.step(action)
        total_reward += reward
        steps += 1
        
        if done:
            break
    
    print(f"✓ Ran {steps} steps with 'danger-aware' agent")
    print(f"  Total reward: {total_reward:.2f}")
    print(f"  Final score: {info.get('score', 0)}")
    print(f"  End reason: {info.get('reason', 'N/A')}")
    
    # Compare with pure random
    print("\n--- Comparing with Pure Random ---")
    state = env.reset()
    total_reward_random = 0
    steps_random = 0
    
    for _ in range(200):
        action = np.random.randint(0, 3)  # 3 relative actions
        state, reward, done, info = env.step(action)
        total_reward_random += reward
        steps_random += 1
        if done:
            break
    
    print(f"Pure random: {steps_random} steps, reward: {total_reward_random:.2f}")
    print(f"Score: {info.get('score', 0)}")
    
    # Close environment
    env.close()
    print("\n✓ Environment closed successfully")
    
    print("\n" + "=" * 60)
    print("All tests passed! ✓")
    print("\nNEW DESIGN: RELATIVE ACTIONS + DIRECT MAPPING")
    print("  Actions: 0=straight, 1=right, 2=left")
    print("  danger[i] maps DIRECTLY to action i")
    print("  would_die[i] maps DIRECTLY to action i")
    print("  This makes learning MUCH easier!")
    print("\nFeatures (24 total, all relative):")
    print("  - 3 danger features (straight/right/left)")
    print("  - 3 relative food direction features")
    print("  - 1 food distance feature")
    print("  - 3 wall distance features (relative)")
    print("  - 3 depth to obstacle features (relative)")
    print("  - 9 action preview features (3 actions × 3 outcomes)")
    print("  - 2 snake state features (length, adjacent body)")

if __name__ == "__main__":
    test_environment()
