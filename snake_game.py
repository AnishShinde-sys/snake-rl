"""
Simple Snake Game - 10x10 Grid
Can be played standalone or used by the RL environment
"""
import pygame
import random
from enum import Enum
from collections import deque

# Constants
GRID_SIZE = 10
CELL_SIZE = 40
WINDOW_SIZE = GRID_SIZE * CELL_SIZE

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREEN = (0, 200, 0)
DARK_GREEN = (0, 150, 0)
RED = (200, 0, 0)
GRAY = (40, 40, 40)


class Direction(Enum):
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3


class SnakeGame:
    def __init__(self, render_mode=False):
        self.render_mode = render_mode
        self.grid_size = GRID_SIZE
        
        if self.render_mode:
            pygame.init()
            self.screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
            pygame.display.set_caption('Snake RL - 10x10')
            self.clock = pygame.time.Clock()
            self.font = pygame.font.Font(None, 36)
        
        self.reset()
    
    def reset(self):
        """Reset the game state"""
        # Start snake in the middle
        center = self.grid_size // 2
        self.snake = deque([(center, center)])
        self.direction = Direction.RIGHT
        self.food = None
        self.spawn_food()
        self.score = 0
        self.steps = 0
        self.game_over = False
        return self.get_state()
    
    def spawn_food(self):
        """Spawn food in a random empty cell"""
        empty_cells = []
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                if (x, y) not in self.snake:
                    empty_cells.append((x, y))
        
        if empty_cells:
            self.food = random.choice(empty_cells)
        else:
            # Snake fills the entire grid - win condition
            self.food = None
    
    def get_state(self):
        """
        Return the state as a flat array of 100 values (10x10 grid)
        0 = empty, 1 = snake body, 2 = snake head, 3 = food
        """
        state = [[0] * self.grid_size for _ in range(self.grid_size)]
        
        # Mark snake body
        for i, (x, y) in enumerate(self.snake):
            if i == 0:
                state[y][x] = 2  # Head
            else:
                state[y][x] = 1  # Body
        
        # Mark food
        if self.food:
            fx, fy = self.food
            state[fy][fx] = 3
        
        # Flatten to 1D array
        flat_state = []
        for row in state:
            flat_state.extend(row)
        
        return flat_state
    
    def step(self, action):
        """
        Take a step in the game
        action: 0=STRAIGHT, 1=RIGHT, 2=LEFT (relative to current direction)
        Returns: (state, reward, done, info)
        """
        if self.game_over:
            return self.get_state(), 0, True, {'score': self.score}
        
        self.steps += 1
        
        # Update direction based on relative action
        # 0 = STRAIGHT (no change), 1 = turn RIGHT, 2 = turn LEFT
        if action == 1:  # Turn right
            turn_right = {
                Direction.UP: Direction.RIGHT,
                Direction.RIGHT: Direction.DOWN,
                Direction.DOWN: Direction.LEFT,
                Direction.LEFT: Direction.UP
            }
            self.direction = turn_right[self.direction]
        elif action == 2:  # Turn left
            turn_left = {
                Direction.UP: Direction.LEFT,
                Direction.LEFT: Direction.DOWN,
                Direction.DOWN: Direction.RIGHT,
                Direction.RIGHT: Direction.UP
            }
            self.direction = turn_left[self.direction]
        # action == 0: continue straight, no direction change
        
        # Calculate new head position
        head_x, head_y = self.snake[0]
        
        if self.direction == Direction.UP:
            new_head = (head_x, head_y - 1)
        elif self.direction == Direction.DOWN:
            new_head = (head_x, head_y + 1)
        elif self.direction == Direction.LEFT:
            new_head = (head_x - 1, head_y)
        else:  # RIGHT
            new_head = (head_x + 1, head_y)
        
        # Check for collisions
        reward = 0
        done = False
        
        # Wall collision
        if (new_head[0] < 0 or new_head[0] >= self.grid_size or
            new_head[1] < 0 or new_head[1] >= self.grid_size):
            self.game_over = True
            done = True
            reward = -10
            return self.get_state(), reward, done, {'score': self.score, 'reason': 'wall'}
        
        # Self collision (check against body, excluding tail which will move)
        # We need to check all body parts except the tail if we're not eating
        will_eat = new_head == self.food
        snake_body = list(self.snake)
        if not will_eat and len(snake_body) > 1:
            # Tail will move, so exclude it from collision check
            snake_body = snake_body[:-1]
        
        if new_head in snake_body:
            self.game_over = True
            done = True
            reward = -10
            return self.get_state(), reward, done, {'score': self.score, 'reason': 'self'}
        
        # Move snake
        self.snake.appendleft(new_head)
        
        # Check food
        if will_eat:
            self.score += 1
            reward = 10
            self.spawn_food()
            
            # Check win condition
            if self.food is None:
                done = True
                reward = 100
                return self.get_state(), reward, done, {'score': self.score, 'reason': 'win'}
        else:
            self.snake.pop()
            # Small negative reward to encourage efficiency
            reward = -0.01
        
        # Step limit to prevent infinite loops
        max_steps = self.grid_size * self.grid_size * 4
        if self.steps >= max_steps:
            done = True
            reward = -5
            return self.get_state(), reward, done, {'score': self.score, 'reason': 'timeout'}
        
        return self.get_state(), reward, done, {'score': self.score}
    
    def render(self):
        """Render the game state"""
        if not self.render_mode:
            return
        
        self.screen.fill(BLACK)
        
        # Draw grid
        for x in range(self.grid_size + 1):
            pygame.draw.line(self.screen, GRAY, 
                           (x * CELL_SIZE, 0), 
                           (x * CELL_SIZE, WINDOW_SIZE))
        for y in range(self.grid_size + 1):
            pygame.draw.line(self.screen, GRAY, 
                           (0, y * CELL_SIZE), 
                           (WINDOW_SIZE, y * CELL_SIZE))
        
        # Draw snake
        for i, (x, y) in enumerate(self.snake):
            color = GREEN if i == 0 else DARK_GREEN
            rect = pygame.Rect(x * CELL_SIZE + 1, y * CELL_SIZE + 1, 
                              CELL_SIZE - 2, CELL_SIZE - 2)
            pygame.draw.rect(self.screen, color, rect)
        
        # Draw food
        if self.food:
            fx, fy = self.food
            rect = pygame.Rect(fx * CELL_SIZE + 1, fy * CELL_SIZE + 1, 
                              CELL_SIZE - 2, CELL_SIZE - 2)
            pygame.draw.rect(self.screen, RED, rect)
        
        # Draw score
        score_text = self.font.render(f'Score: {self.score}', True, WHITE)
        self.screen.blit(score_text, (5, 5))
        
        pygame.display.flip()
    
    def handle_events(self):
        """Handle pygame events for manual play
        Returns: 0=STRAIGHT, 1=RIGHT, 2=LEFT, False=quit, None=no input
        """
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP or event.key == pygame.K_w:
                    return 0  # STRAIGHT
                elif event.key == pygame.K_RIGHT or event.key == pygame.K_d:
                    return 1  # Turn RIGHT
                elif event.key == pygame.K_LEFT or event.key == pygame.K_a:
                    return 2  # Turn LEFT
        return None
    
    def close(self):
        """Clean up pygame"""
        if self.render_mode:
            pygame.quit()


def play_manual():
    """Play the game manually
    Controls: UP/W = go straight, RIGHT/D = turn right, LEFT/A = turn left
    """
    game = SnakeGame(render_mode=True)
    game.reset()
    
    print("Controls: UP/W = go straight, RIGHT/D = turn right, LEFT/A = turn left")
    
    running = True
    action = 0  # Default: go straight
    
    while running:
        # Handle events
        result = game.handle_events()
        if result is False:
            running = False
            continue
        elif result is not None:
            action = result
        else:
            action = 0  # Default to straight if no input
        
        # Take step
        state, reward, done, info = game.step(action)
        
        if done:
            print(f"Game Over! Score: {info['score']}, Reason: {info.get('reason', 'unknown')}")
            game.reset()
        
        game.render()
        game.clock.tick(8)  # 8 FPS for playability
    
    game.close()


if __name__ == '__main__':
    play_manual()

