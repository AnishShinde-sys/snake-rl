"""
Snake RL Environment
Wraps the snake game for reinforcement learning
"""
import numpy as np
from snake_game import SnakeGame, Direction


class SnakeEnv:
    """
    RL Environment wrapper for the Snake game
    
    State Space: 100 values (10x10 grid)
        - 0 = empty
        - 1 = snake body
        - 2 = snake head
        - 3 = food
    
    Action Space: 4 discrete actions
        - 0 = UP
        - 1 = RIGHT
        - 2 = DOWN
        - 3 = LEFT
    
    Reward Structure:
        - +10 for eating food
        - -10 for collision (wall or self)
        - -5 for timeout (prevents infinite loops)
        - -0.01 per step (encourages efficiency)
        - +100 for winning (filling entire grid)
    
    Handling self-collision and trapped states:
        The agent learns to avoid these through negative rewards.
        GRPO helps by comparing trajectories - actions that lead to
        collisions will have lower returns compared to the group,
        resulting in decreased probability.
    """
    
    def __init__(self, render_mode=False):
        self.game = SnakeGame(render_mode=render_mode)
        self.render_mode = render_mode
        
        # Environment properties
        self.observation_space_size = 100  # 10x10 grid
        self.action_space_size = 4  # UP, RIGHT, DOWN, LEFT
        self.grid_size = 10
    
    def reset(self):
        """Reset the environment and return initial state"""
        state = self.game.reset()
        return np.array(state, dtype=np.float32)
    
    def step(self, action):
        """
        Take a step in the environment
        
        Args:
            action: int in [0, 3] representing direction
        
        Returns:
            state: numpy array of shape (100,)
            reward: float
            done: bool
            info: dict with additional info
        """
        state, reward, done, info = self.game.step(action)
        return np.array(state, dtype=np.float32), reward, done, info
    
    def render(self):
        """Render the current state"""
        if self.render_mode:
            self.game.render()
    
    def close(self):
        """Clean up resources"""
        self.game.close()
    
    def get_valid_actions(self):
        """
        Get list of actions that won't immediately cause collision
        This can be used for action masking if needed
        """
        valid = []
        head_x, head_y = self.game.snake[0]
        
        # Check each direction
        directions = {
            0: (head_x, head_y - 1),  # UP
            1: (head_x + 1, head_y),  # RIGHT
            2: (head_x, head_y + 1),  # DOWN
            3: (head_x - 1, head_y),  # LEFT
        }
        
        # Current direction's opposite
        opposite = {
            Direction.UP: Direction.DOWN,
            Direction.DOWN: Direction.UP,
            Direction.LEFT: Direction.RIGHT,
            Direction.RIGHT: Direction.LEFT
        }
        
        current_opposite = opposite[self.game.direction].value
        
        for action, (nx, ny) in directions.items():
            # Skip 180-degree turn (will be ignored anyway)
            if action == current_opposite:
                continue
            
            # Check bounds
            if nx < 0 or nx >= self.grid_size or ny < 0 or ny >= self.grid_size:
                continue
            
            # Check self-collision (exclude tail as it will move)
            snake_body = list(self.game.snake)
            if len(snake_body) > 1:
                snake_body = snake_body[:-1]  # Exclude tail
            
            if (nx, ny) in snake_body:
                continue
            
            valid.append(action)
        
        return valid if valid else list(range(4))  # Return all if none valid
    
    def get_state_features(self):
        """
        Get additional state features that might be useful
        Returns dict with computed features
        """
        head_x, head_y = self.game.snake[0]
        food_x, food_y = self.game.food if self.game.food else (0, 0)
        
        # Distance to food (Manhattan)
        food_dist = abs(head_x - food_x) + abs(head_y - food_y)
        
        # Distance to walls
        dist_to_walls = {
            'up': head_y,
            'down': self.grid_size - 1 - head_y,
            'left': head_x,
            'right': self.grid_size - 1 - head_x
        }
        
        # Snake length
        snake_length = len(self.game.snake)
        
        # Direction to food
        food_dir = {
            'up': food_y < head_y,
            'down': food_y > head_y,
            'left': food_x < head_x,
            'right': food_x > head_x
        }
        
        return {
            'food_distance': food_dist,
            'dist_to_walls': dist_to_walls,
            'snake_length': snake_length,
            'food_direction': food_dir,
            'valid_actions': self.get_valid_actions()
        }


def test_env():
    """Test the environment"""
    env = SnakeEnv(render_mode=True)
    
    state = env.reset()
    print(f"Initial state shape: {state.shape}")
    print(f"State: {state.reshape(10, 10)}")
    
    done = False
    total_reward = 0
    steps = 0
    
    import pygame
    
    while not done:
        # Handle pygame events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
                break
        
        if done:
            break
        
        # Random action
        action = np.random.randint(0, 4)
        
        state, reward, done, info = env.step(action)
        total_reward += reward
        steps += 1
        
        env.render()
        pygame.time.delay(100)
        
        if done:
            print(f"Episode ended: {info}")
            print(f"Total reward: {total_reward}, Steps: {steps}")
    
    env.close()


if __name__ == '__main__':
    test_env()

