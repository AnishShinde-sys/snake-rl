"""Quick test script to verify the environment works"""
import numpy as np
from snake_env import SnakeEnv

def test_environment():
    print("Testing Snake RL Environment...")
    print("=" * 50)
    
    # Create environment
    env = SnakeEnv(grid_size=10, render_mode=False)
    print("✓ Environment created successfully")
    
    # Test reset
    state = env.reset()
    print(f"✓ Reset works. State shape: {state.shape}")
    assert state.shape == (100,), f"Expected state shape (100,), got {state.shape}"
    
    # Test step
    state, reward, done, info = env.step(1)  # Move right
    print(f"✓ Step works. Reward: {reward}, Done: {done}")
    
    # Test multiple steps
    total_reward = 0
    steps = 0
    for _ in range(50):
        action = np.random.randint(0, 4)
        state, reward, done, info = env.step(action)
        total_reward += reward
        steps += 1
        if done:
            break
    
    print(f"✓ Ran {steps} steps. Total reward: {total_reward:.2f}")
    print(f"  Final score: {info.get('score', 0)}")
    print(f"  End reason: {info.get('reason', 'N/A')}")
    
    # Close environment
    env.close()
    print("✓ Environment closed successfully")
    
    print("=" * 50)
    print("All tests passed! ✓")

if __name__ == "__main__":
    test_environment()

