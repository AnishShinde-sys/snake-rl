import numpy as np
import pygame
import random
from collections import deque


class SnakeEnv:
    """Snake game environment for reinforcement learning with rich feature state"""
    
    # Direction vectors: 0=UP, 1=RIGHT, 2=DOWN, 3=LEFT
    DIRECTIONS = [(0, -1), (1, 0), (0, 1), (-1, 0)]
    
    def __init__(self, grid_size=10, render_mode=False):
        self.grid_size = grid_size
        self.render_mode = render_mode
        self.cell_size = 50
        self.window_size = self.grid_size * self.cell_size
        
        # Action space: 0=UP, 1=RIGHT, 2=DOWN, 3=LEFT
        self.action_space = 4
        
        # State space: Rich feature representation
        # 4: current direction (one-hot)
        # 3: danger straight/right/left (relative to current direction)
        # 4: food direction (up/right/down/left)
        # 2: normalized food distance (dx, dy)
        # 4: depth to obstacle in each direction
        # 4: action preview - would die for each action
        # 4: action preview - would eat for each action  
        # 4: action preview - distance change for each action
        # 1: normalized snake length
        # 1: body segments adjacent to head
        # Total: 31 features
        self.observation_space = 31
        
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
    
    def _is_collision(self, pos):
        """Check if position causes collision with wall or snake body"""
        # Wall collision
        if pos[0] < 0 or pos[0] >= self.grid_size or pos[1] < 0 or pos[1] >= self.grid_size:
            return True
        # Body collision (exclude tail since it will move unless eating)
        if pos in list(self.snake)[:-1]:
            return True
        return False
    
    def _is_collision_full(self, pos):
        """Check collision including tail (for depth calculation)"""
        if pos[0] < 0 or pos[0] >= self.grid_size or pos[1] < 0 or pos[1] >= self.grid_size:
            return True
        if pos in self.snake:
            return True
        return False
    
    def _get_depth(self, start_pos, direction_idx):
        """Get distance to nearest obstacle in a direction (normalized 0-1)"""
        dx, dy = self.DIRECTIONS[direction_idx]
        x, y = start_pos
        distance = 0
        
        while True:
            x += dx
            y += dy
            distance += 1
            
            # Check for wall
            if x < 0 or x >= self.grid_size or y < 0 or y >= self.grid_size:
                break
            # Check for body
            if (x, y) in self.snake:
                break
        
        # Normalize by grid size
        return distance / self.grid_size
    
    def _get_next_pos(self, action):
        """Get the position after taking an action"""
        head = self.snake[0]
        dx, dy = self.DIRECTIONS[action]
        return (head[0] + dx, head[1] + dy)
    
    def _manhattan_distance(self, pos1, pos2):
        """Calculate manhattan distance between two positions"""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
    
    def _get_relative_direction(self, base_dir, turn):
        """
        Get absolute direction based on relative turn
        turn: 0=straight, 1=right, 2=left
        """
        if turn == 0:  # straight
            return base_dir
        elif turn == 1:  # right
            return (base_dir + 1) % 4
        else:  # left
            return (base_dir - 1) % 4
    
    def _get_state(self):
        """
        Get current state as feature vector
        
        Features:
        - Current direction (4): one-hot encoded
        - Danger relative (3): straight, right, left from current direction
        - Food direction (4): is food up/right/down/left
        - Food distance (2): normalized dx, dy to food
        - Depth to obstacle (4): in each absolute direction
        - Action preview - would die (4): for each action
        - Action preview - would eat (4): for each action
        - Action preview - distance change (4): for each action
        - Snake length (1): normalized
        - Adjacent body count (1): body segments next to head
        
        Total: 31 features
        """
        head = self.snake[0]
        features = []
        
        # 1. Current direction one-hot (4 features)
        dir_one_hot = [0, 0, 0, 0]
        dir_one_hot[self.direction] = 1
        features.extend(dir_one_hot)
        
        # 2. Danger in relative directions (3 features)
        # Straight, Right, Left from current direction
        for turn in [0, 1, 2]:  # straight, right, left
            abs_dir = self._get_relative_direction(self.direction, turn)
            next_pos = self._get_next_pos(abs_dir)
            danger = 1 if self._is_collision(next_pos) else 0
            features.append(danger)
        
        # 3. Food direction (4 features)
        food_up = 1 if self.food[1] < head[1] else 0
        food_right = 1 if self.food[0] > head[0] else 0
        food_down = 1 if self.food[1] > head[1] else 0
        food_left = 1 if self.food[0] < head[0] else 0
        features.extend([food_up, food_right, food_down, food_left])
        
        # 4. Normalized food distance (2 features)
        food_dx = (self.food[0] - head[0]) / self.grid_size
        food_dy = (self.food[1] - head[1]) / self.grid_size
        features.extend([food_dx, food_dy])
        
        # 5. Depth to obstacle in each direction (4 features)
        for direction_idx in range(4):
            depth = self._get_depth(head, direction_idx)
            features.append(depth)
        
        # 6-8. Action preview for each possible action (4 actions × 3 features = 12)
        current_dist = self._manhattan_distance(head, self.food)
        
        for action in range(4):
            next_pos = self._get_next_pos(action)
            
            # Would die?
            would_die = 1 if self._is_collision(next_pos) else 0
            features.append(would_die)
        
        for action in range(4):
            next_pos = self._get_next_pos(action)
            
            # Would eat?
            would_eat = 1 if next_pos == self.food else 0
            features.append(would_eat)
        
        for action in range(4):
            next_pos = self._get_next_pos(action)
            
            # Distance change (negative = getting closer, positive = getting farther)
            if self._is_collision(next_pos):
                dist_change = 1  # Penalize collision paths
            else:
                new_dist = self._manhattan_distance(next_pos, self.food)
                # Normalize: -1 (getting much closer) to +1 (getting much farther)
                dist_change = (new_dist - current_dist) / (2 * self.grid_size)
            features.append(dist_change)
        
        # 9. Normalized snake length (1 feature)
        max_possible_length = self.grid_size * self.grid_size
        snake_length = len(self.snake) / max_possible_length
        features.append(snake_length)
        
        # 10. Body segments adjacent to head (1 feature)
        adjacent_body = 0
        for direction_idx in range(4):
            dx, dy = self.DIRECTIONS[direction_idx]
            neighbor = (head[0] + dx, head[1] + dy)
            if neighbor in list(self.snake)[1:]:  # Exclude head itself
                adjacent_body += 1
        features.append(adjacent_body / 4)  # Normalize by max possible (4)
        
        return np.array(features, dtype=np.float32)
    
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
        dx, dy = self.DIRECTIONS[action]
        new_head = (head[0] + dx, head[1] + dy)
        
        # Check for collisions
        done = False
        reward = 0
        
        # Wall collision
        if (new_head[0] < 0 or new_head[0] >= self.grid_size or 
            new_head[1] < 0 or new_head[1] >= self.grid_size):
            done = True
            reward = -10
            return self._get_state(), reward, done, {"score": self.score, "reason": "wall"}
        
        # Self collision (check against body excluding tail which will move)
        body_without_tail = list(self.snake)[:-1]
        if new_head in body_without_tail:
            done = True
            reward = -10
            return self._get_state(), reward, done, {"score": self.score, "reason": "self"}
        
        # Move snake
        self.snake.appendleft(new_head)
        
        # Check if food eaten
        if new_head == self.food:
            self.score += 1
            reward = 50  # High reward for eating food
            self.food = self._place_food()
        else:
            # Remove tail if no food eaten
            self.snake.pop()
            # Reward for getting closer to food, penalize for getting farther
            old_dist = self._manhattan_distance(head, self.food)
            new_dist = self._manhattan_distance(new_head, self.food)
            if new_dist < old_dist:
                reward = 0.1  # Getting closer
            else:
                reward = -0.1  # Getting farther
        
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
