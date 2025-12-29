import numpy as np
import pygame
import random
from collections import deque


class SnakeEnv:
    """Snake game environment for reinforcement learning"""
    
    def __init__(self, grid_size=10, render_mode=False):
        self.grid_size = grid_size
        self.render_mode = render_mode
        self.cell_size = 50
        self.window_size = self.grid_size * self.cell_size
        
        # Action space: 0=UP, 1=RIGHT, 2=DOWN, 3=LEFT
        self.action_space = 4
        # State space: 10x10 grid = 100 features
        self.observation_space = grid_size * grid_size
        
        if self.render_mode:
            pygame.init()
            self.screen = pygame.display.set_mode((self.window_size, self.window_size))
            pygame.display.set_caption("Snake RL")
            self.clock = pygame.time.Clock()
        else:
            self.screen = None
            self.clock = None
            
        self.reset()
    
    def reset(self):
        """Reset the game to initial state"""
        # Start snake in the middle
        self.snake = deque([(self.grid_size // 2, self.grid_size // 2)])
        self.direction = 1  # Start moving right
        self.food = self._place_food()
        self.score = 0
        self.steps = 0
        self.max_steps = self.grid_size * self.grid_size * 10  # Prevent infinite loops
        return self._get_state()
    
    def _place_food(self):
        """Place food at random empty position"""
        while True:
            food = (random.randint(0, self.grid_size - 1), 
                   random.randint(0, self.grid_size - 1))
            if food not in self.snake:
                return food
    
    def _get_state(self):
        """
        Get current state as 100-dimensional vector (10x10 grid)
        0 = empty, 1 = snake body, 2 = snake head, 3 = food
        """
        state = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        
        # Mark snake body
        for pos in list(self.snake)[1:]:
            if 0 <= pos[0] < self.grid_size and 0 <= pos[1] < self.grid_size:
                state[pos[1], pos[0]] = 1
        
        # Mark snake head
        head = self.snake[0]
        if 0 <= head[0] < self.grid_size and 0 <= head[1] < self.grid_size:
            state[head[1], head[0]] = 2
        
        # Mark food
        state[self.food[1], self.food[0]] = 3
        
        return state.flatten()
    
    def step(self, action):
        """
        Take action and return (state, reward, done, info)
        Actions: 0=UP, 1=RIGHT, 2=DOWN, 3=LEFT
        """
        self.steps += 1
        
        # Prevent 180-degree turns
        if (action == 0 and self.direction == 2) or \
           (action == 2 and self.direction == 0) or \
           (action == 1 and self.direction == 3) or \
           (action == 3 and self.direction == 1):
            action = self.direction
        
        self.direction = action
        
        # Calculate new head position
        head = self.snake[0]
        if action == 0:  # UP
            new_head = (head[0], head[1] - 1)
        elif action == 1:  # RIGHT
            new_head = (head[0] + 1, head[1])
        elif action == 2:  # DOWN
            new_head = (head[0], head[1] + 1)
        else:  # LEFT (action == 3)
            new_head = (head[0] - 1, head[1])
        
        # Check for collisions
        done = False
        reward = 0
        
        # Wall collision
        if (new_head[0] < 0 or new_head[0] >= self.grid_size or 
            new_head[1] < 0 or new_head[1] >= self.grid_size):
            done = True
            reward = -10
            return self._get_state(), reward, done, {"score": self.score, "reason": "wall"}
        
        # Self collision
        if new_head in self.snake:
            done = True
            reward = -10
            return self._get_state(), reward, done, {"score": self.score, "reason": "self"}
        
        # Move snake
        self.snake.appendleft(new_head)
        
        # Check if food eaten
        if new_head == self.food:
            self.score += 1
            reward = 10
            self.food = self._place_food()
        else:
            # Remove tail if no food eaten
            self.snake.pop()
            # Small reward for surviving
            reward = 0.1
        
        # Check if max steps reached (stuck in loop)
        if self.steps >= self.max_steps:
            done = True
            reward = -5
            return self._get_state(), reward, done, {"score": self.score, "reason": "timeout"}
        
        return self._get_state(), reward, done, {"score": self.score}
    
    def render(self):
        """Render the game if render_mode is True"""
        if not self.render_mode or self.screen is None:
            return
        
        # Handle pygame events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
        
        # Clear screen
        self.screen.fill((0, 0, 0))
        
        # Draw grid
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                rect = pygame.Rect(x * self.cell_size, y * self.cell_size, 
                                  self.cell_size - 2, self.cell_size - 2)
                pygame.draw.rect(self.screen, (40, 40, 40), rect)
        
        # Draw snake
        for i, pos in enumerate(self.snake):
            rect = pygame.Rect(pos[0] * self.cell_size, pos[1] * self.cell_size,
                             self.cell_size - 2, self.cell_size - 2)
            if i == 0:
                # Head
                pygame.draw.rect(self.screen, (0, 255, 0), rect)
            else:
                # Body
                pygame.draw.rect(self.screen, (0, 200, 0), rect)
        
        # Draw food
        food_rect = pygame.Rect(self.food[0] * self.cell_size, self.food[1] * self.cell_size,
                               self.cell_size - 2, self.cell_size - 2)
        pygame.draw.rect(self.screen, (255, 0, 0), food_rect)
        
        pygame.display.flip()
        if self.clock:
            self.clock.tick(10)  # 10 FPS for rendering
    
    def close(self):
        """Close the environment"""
        if self.render_mode and pygame.get_init():
            pygame.quit()

