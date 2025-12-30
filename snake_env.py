import numpy as np
import pygame
import random
from collections import deque
import json

# #region agent log
_DEBUG_LOG_PATH = "/Users/anishshinde/snake-rl/.cursor/debug.log"
def _debug_log(hyp, loc, msg, data):
    with open(_DEBUG_LOG_PATH, "a") as f:
        f.write(json.dumps({"hypothesisId": hyp, "location": loc, "message": msg, "data": data, "timestamp": __import__('time').time()}) + "\n")
# #endregion


class SnakeEnv:
    """Snake game environment for reinforcement learning with rich feature state"""
    
    # Direction vectors: 0=UP, 1=RIGHT, 2=DOWN, 3=LEFT
    DIRECTIONS = [(0, -1), (1, 0), (0, 1), (-1, 0)]
    
    def __init__(self, grid_size=10, render_mode=False):
        self.grid_size = grid_size
        self.render_mode = render_mode
        self.cell_size = 50
        self.window_size = self.grid_size * self.cell_size
        
        # Action space: RELATIVE actions - 0=STRAIGHT, 1=RIGHT, 2=LEFT
        # This directly maps to danger signals for easier learning!
        self.action_space = 3
        
        # State space: Rich feature representation (RELATIVE to direction)
        # 3: danger straight/right/left (relative to current direction)
        # 3: food direction relative (straight/right/left)
        # 1: normalized food distance
        # 3: distance to wall (straight/right/left)
        # 3: depth to obstacle (straight/right/left)
        # 3: action preview - would die (straight/right/left) - DIRECT mapping!
        # 3: action preview - would eat (straight/right/left)
        # 3: action preview - distance change (straight/right/left)
        # 1: normalized snake length
        # 1: body segments adjacent to head
        # Total: 24 features
        self.observation_space = 30  # 24 base + 6 lookahead features (min/max safe for each action)
        
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
        self.max_steps = self.grid_size * self.grid_size * 20  # Allow more steps for complete game
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
    
    def _get_wall_distance(self, start_pos, direction_idx):
        """Get distance to wall in a direction (normalized 0-1)"""
        dx, dy = self.DIRECTIONS[direction_idx]
        x, y = start_pos
        distance = 0
        
        while True:
            x += dx
            y += dy
            distance += 1
            
            # Check for wall only
            if x < 0 or x >= self.grid_size or y < 0 or y >= self.grid_size:
                break
        
        # Normalize by grid size
        return distance / self.grid_size
    
    def _get_body_distance(self, start_pos, direction_idx):
        """
        Get distance to nearest body segment in a direction (normalized 0-1)
        Returns distance to body, or 1.0 if no body found (wall reached first)
        """
        dx, dy = self.DIRECTIONS[direction_idx]
        x, y = start_pos
        distance = 0
        
        while True:
            x += dx
            y += dy
            distance += 1
            
            # Check for wall first - if wall hit before body, return max distance
            if x < 0 or x >= self.grid_size or y < 0 or y >= self.grid_size:
                return 1.0  # No body found, wall reached first
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
    
    def _is_pos_safe(self, pos, snake_body):
        """Check if a position is safe (not wall, not body)"""
        return (0 <= pos[0] < self.grid_size and 
                0 <= pos[1] < self.grid_size and 
                pos not in snake_body)
    
    def _count_safe_moves(self, head_pos, direction, snake_body):
        """
        Count how many of the 3 relative moves are safe from this position/direction.
        Returns 0-3.
        """
        safe_count = 0
        for turn in [0, 1, 2]:  # straight, right, left
            if turn == 0:
                new_dir = direction
            elif turn == 1:
                new_dir = (direction + 1) % 4
            else:
                new_dir = (direction - 1) % 4
            
            dx, dy = self.DIRECTIONS[new_dir]
            new_pos = (head_pos[0] + dx, head_pos[1] + dy)
            if self._is_pos_safe(new_pos, snake_body):
                safe_count += 1
        return safe_count
    
    def _lookahead_safe_moves(self, action_dir, depth=2):
        """
        Look ahead N moves and return (min_safe_moves, max_safe_moves) across all paths.
        This tells the agent: "If I take this action, what's the worst/best case for escape routes?"
        
        Returns tuple: (min_options_at_any_point, total_safe_paths_found)
        """
        head = self.snake[0]
        dx, dy = self.DIRECTIONS[action_dir]
        next_head = (head[0] + dx, head[1] + dy)
        
        # Check if first move is even valid
        snake_set = set(self.snake)
        if not self._is_pos_safe(next_head, snake_set):
            return (0, 0)  # Dead immediately
        
        # Simulate taking the action
        new_snake = list(self.snake)
        new_snake.insert(0, next_head)
        if next_head != self.food:
            new_snake.pop()  # Remove tail
        new_snake_set = set(new_snake)
        
        # Count safe moves after first action
        safe_after_first = self._count_safe_moves(next_head, action_dir, new_snake_set)
        
        if depth <= 1:
            return (safe_after_first, safe_after_first)
        
        # Lookahead depth 2: For each safe move after first, count safe moves after that
        min_safe = 3
        max_safe = 0
        paths_found = 0
        
        for turn in [0, 1, 2]:
            if turn == 0:
                dir2 = action_dir
            elif turn == 1:
                dir2 = (action_dir + 1) % 4
            else:
                dir2 = (action_dir - 1) % 4
            
            dx2, dy2 = self.DIRECTIONS[dir2]
            pos2 = (next_head[0] + dx2, next_head[1] + dy2)
            
            if self._is_pos_safe(pos2, new_snake_set):
                # Simulate this second move
                snake2 = list(new_snake)
                snake2.insert(0, pos2)
                if pos2 != self.food:
                    snake2.pop()
                snake2_set = set(snake2)
                
                safe_after_second = self._count_safe_moves(pos2, dir2, snake2_set)
                min_safe = min(min_safe, safe_after_second)
                max_safe = max(max_safe, safe_after_second)
                paths_found += 1
        
        if paths_found == 0:
            return (0, 0)  # No valid paths after first move
        
        return (min_safe, max_safe)
    
    def _can_reach_tail(self, head_pos, snake_body_set, tail_pos):
        """
        BFS to check if head can reach tail position.
        This is THE KEY to never getting trapped - if you can always reach your tail,
        you can never box yourself in!
        """
        if head_pos == tail_pos:
            return True
        
        visited = set()
        queue = [head_pos]
        visited.add(head_pos)
        
        while queue:
            pos = queue.pop(0)
            for dx, dy in self.DIRECTIONS:
                new_pos = (pos[0] + dx, pos[1] + dy)
                if new_pos == tail_pos:
                    return True  # Can reach tail!
                if (new_pos not in visited and
                    0 <= new_pos[0] < self.grid_size and
                    0 <= new_pos[1] < self.grid_size and
                    new_pos not in snake_body_set):
                    visited.add(new_pos)
                    queue.append(new_pos)
        
        return False  # Cannot reach tail - this move would trap us!
    
    def get_action_mask(self):
        """
        Return a mask of safe actions. 
        An action is "safe" if it doesn't lead to immediate death AND 
        leaves at least one escape route after taking it.
        
        CRITICAL: When snake is long (>30), also check if we can still reach our tail.
        This prevents boxing ourselves in!
        
        EXCEPTION: Always allow eating food if adjacent - eating changes tail position!
        
        Returns: list of 3 bools [straight_safe, right_safe, left_safe]
        """
        head = self.snake[0]
        tail = self.snake[-1]
        snake_set = set(self.snake)
        snake_len = len(self.snake)
        mask = [False, False, False]
        best_options = [-1, -1, -1]  # Track safety score for each action
        can_reach_tail_score = [0, 0, 0]  # Bonus for actions that maintain tail access
        food_adjacent = [False, False, False]  # Track which actions eat food
        
        for action in range(3):
            # Get absolute direction for this action
            if action == 0:
                abs_dir = self.direction
            elif action == 1:
                abs_dir = (self.direction + 1) % 4
            else:
                abs_dir = (self.direction - 1) % 4
            
            # Check if this action leads to immediate death
            dx, dy = self.DIRECTIONS[abs_dir]
            next_pos = (head[0] + dx, head[1] + dy)
            
            if not self._is_pos_safe(next_pos, snake_set):
                best_options[action] = -100  # Definitely bad
                continue  # Can't take this action, immediate death
            
            # Simulate taking this action
            new_snake_list = [next_pos] + list(self.snake)
            eating_food = (next_pos == self.food)
            food_adjacent[action] = eating_food
            if not eating_food:
                new_snake_list.pop()  # Remove tail
            new_snake_set = set(new_snake_list)
            new_tail = new_snake_list[-1]
            
            # Count safe moves after this action
            safe_after = self._count_safe_moves(next_pos, abs_dir, new_snake_set)
            best_options[action] = safe_after
            
            # EATING FOOD IS ALWAYS GOOD (if safe) - don't penalize!
            if eating_food:
                best_options[action] += 200  # Huge bonus for food!
                can_reach_tail_score[action] = 100  # Treat eating as "safe" for tail
            elif snake_len > 30:
                # Check if we can still reach our tail after this move
                # (exclude tail from body check since that's our target)
                body_without_tail = new_snake_set - {new_tail}
                if self._can_reach_tail(next_pos, body_without_tail, new_tail):
                    can_reach_tail_score[action] = 100  # Big bonus!
                    best_options[action] += 100
                else:
                    # Can't reach tail - very dangerous! (but still allow if only option)
                    best_options[action] -= 50
            
            # Action is safe if it has at least 1 escape route after
            if safe_after >= 1:
                mask[action] = True
        
        # ALWAYS allow eating food if it's safe (has escape routes)
        for action in range(3):
            if food_adjacent[action] and best_options[action] > 0:
                mask[action] = True
        
        # When snake is long and NOT eating food, prefer tail-reachable moves
        if snake_len > 30 and not any(food_adjacent):
            tail_reachable_mask = [can_reach_tail_score[i] > 0 for i in range(3)]
            if any(tail_reachable_mask):
                # Override mask - only allow tail-reachable moves
                mask = tail_reachable_mask
        
        # If no safe actions, pick the one with best options (or least bad)
        if not any(mask):
            best_idx = max(range(3), key=lambda i: best_options[i])
            mask[best_idx] = True  # Allow the least bad option
        
        return mask
    
    def _get_state(self):
        """
        Get current state as feature vector - ALL RELATIVE to current direction!
        
        Features (24 total):
        - Danger (3): straight, right, left - DIRECT mapping to actions!
        - Food direction relative (3): is food straight/right/left from me
        - Food distance (1): normalized manhattan distance
        - Wall distance relative (3): distance to wall straight/right/left
        - Depth to obstacle relative (3): depth straight/right/left
        - Would die (3): for each action - DIRECT mapping to actions!
        - Would eat (3): for each action
        - Distance change (3): for each action
        - Snake length (1)
        - Adjacent body count (1)
        - Lookahead (6): for each action, (min_safe_moves, max_safe_moves) after 2-step lookahead
        
        Total: 30 features
        """
        head = self.snake[0]
        features = []
        
        # Get the three relative directions
        dir_straight = self.direction
        dir_right = (self.direction + 1) % 4
        dir_left = (self.direction - 1) % 4
        relative_dirs = [dir_straight, dir_right, dir_left]
        
        # 1. Danger in relative directions (3 features) - DIRECT ACTION MAPPING
        # danger[0] = danger if action 0 (straight)
        # danger[1] = danger if action 1 (right)
        # danger[2] = danger if action 2 (left)
        for abs_dir in relative_dirs:
            next_pos = self._get_next_pos(abs_dir)
            danger = 1 if self._is_collision(next_pos) else 0
            features.append(danger)
        
        # 2. Food direction RELATIVE to current direction (3 features)
        # Is food in the direction I would go if I took each action?
        food_dx = self.food[0] - head[0]
        food_dy = self.food[1] - head[1]
        
        for abs_dir in relative_dirs:
            dir_dx, dir_dy = self.DIRECTIONS[abs_dir]
            # Dot product > 0 means food is in this direction
            dot = food_dx * dir_dx + food_dy * dir_dy
            food_in_dir = 1 if dot > 0 else 0
            features.append(food_in_dir)
        
        # 3. Food distance normalized (1 feature)
        food_dist = self._manhattan_distance(head, self.food) / (2 * self.grid_size)
        features.append(food_dist)
        
        # 4. Wall distance in relative directions (3 features)
        for abs_dir in relative_dirs:
            wall_dist = self._get_wall_distance(head, abs_dir)
            features.append(wall_dist)
        
        # 5. Depth to obstacle in relative directions (3 features)
        for abs_dir in relative_dirs:
            depth = self._get_depth(head, abs_dir)
            features.append(depth)
        
        # 6. Would die for each RELATIVE action (3 features) - DIRECT ACTION MAPPING
        for abs_dir in relative_dirs:
            next_pos = self._get_next_pos(abs_dir)
            would_die = 1 if self._is_collision(next_pos) else 0
            features.append(would_die)
        
        # 7. Would eat for each RELATIVE action (3 features)
        for abs_dir in relative_dirs:
            next_pos = self._get_next_pos(abs_dir)
            would_eat = 1 if next_pos == self.food else 0
            features.append(would_eat)
        
        # 8. Distance change for each RELATIVE action (3 features)
        current_dist = self._manhattan_distance(head, self.food)
        for abs_dir in relative_dirs:
            next_pos = self._get_next_pos(abs_dir)
            if self._is_collision(next_pos):
                dist_change = 1  # Penalize collision paths
            else:
                new_dist = self._manhattan_distance(next_pos, self.food)
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
            if neighbor in list(self.snake)[1:]:
                adjacent_body += 1
        features.append(adjacent_body / 4)
        
        # 11. LOOKAHEAD - For each action: min safe moves and max safe moves after 2-step lookahead
        # This tells the agent: "If I go this way, how trapped will I be?"
        # 6 features: (min_safe, max_safe) for each of 3 actions
        for abs_dir in relative_dirs:
            min_safe, max_safe = self._lookahead_safe_moves(abs_dir, depth=2)
            features.append(min_safe / 3.0)  # Normalize to 0-1
            features.append(max_safe / 3.0)  # Normalize to 0-1
        
        return np.array(features, dtype=np.float32)
    
    def step(self, action):
        """
        Take action and return (state, reward, done, info)
        RELATIVE Actions: 0=STRAIGHT, 1=TURN RIGHT, 2=TURN LEFT
        Direct mapping to danger signals!
        """
        self.steps += 1
        
        # Convert RELATIVE action to ABSOLUTE direction
        # action 0 = go straight (keep direction)
        # action 1 = turn right (direction + 1)
        # action 2 = turn left (direction - 1)
        if action == 0:
            new_direction = self.direction  # straight
        elif action == 1:
            new_direction = (self.direction + 1) % 4  # right
        else:  # action == 2
            new_direction = (self.direction - 1) % 4  # left
        
        # #region agent log
        if self.steps <= 3:  # Log first 3 steps of each episode
            pre_state = self._get_state()
            _debug_log("H5", "snake_env.py:285", "STEP_START", {
                "step": int(self.steps), "action": int(action), "prev_direction": int(self.direction),
                "new_direction": int(new_direction),
                "head": [int(x) for x in self.snake[0]], "food": [int(x) for x in self.food],
                "danger": [float(pre_state[0]), float(pre_state[1]), float(pre_state[2])],
                "would_die": [float(pre_state[13]), float(pre_state[14]), float(pre_state[15])]
            })
        # #endregion
        
        self.direction = new_direction
        
        # Calculate new head position
        head = self.snake[0]
        dx, dy = self.DIRECTIONS[new_direction]
        new_head = (head[0] + dx, head[1] + dy)
        
        # Check for collisions
        done = False
        reward = 0
        
        # Wall collision
        if (new_head[0] < 0 or new_head[0] >= self.grid_size or 
            new_head[1] < 0 or new_head[1] >= self.grid_size):
            done = True
            reward = -10  # Death penalty
            # #region agent log
            pre_state = self._get_state()
            # DIRECT MAPPING: action index = danger/would_die index!
            danger_signal = pre_state[action]  # danger at 0,1,2 = straight,right,left
            would_die_signal = pre_state[13 + action]  # would_die at 13,14,15
            _debug_log("H1", "snake_env.py:320", "WALL_DEATH", {
                "action": int(action), "direction": int(self.direction),
                "danger_signal": float(danger_signal),
                "would_die_signal": float(would_die_signal),
                "predictable": bool(danger_signal == 1),
                "head": [int(x) for x in head], "new_head": [int(x) for x in new_head]
            })
            # #endregion
            return pre_state, reward, done, {"score": self.score, "reason": "wall"}
        
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
            reward = 10  # Reward for eating food
            self.food = self._place_food()
        else:
            # Remove tail if no food eaten
            self.snake.pop()
            # STRONGER reward for getting closer to food
            old_dist = self._manhattan_distance(head, self.food)
            new_dist = self._manhattan_distance(new_head, self.food)
            if new_dist < old_dist:
                reward = 1  # Getting closer
            else:
                reward = -1.5  # Getting farther
            
            # SURVIVAL BONUS: Small reward for staying alive when snake is long
            snake_len = len(self.snake)
            if snake_len > 10:
                reward += 0.1 * (snake_len / 10)  # Bonus scales with length
            
            # TRAP PENALTY: Penalize moves that leave few escape options
            # Count safe moves available from current position
            snake_set = set(self.snake)
            safe_moves = self._count_safe_moves(new_head, new_direction, snake_set)
            if safe_moves == 0:
                reward -= 5  # About to die, big penalty (shouldn't happen, would be caught earlier)
            elif safe_moves == 1 and snake_len > 5:
                reward -= 2  # Only one escape route - very dangerous!
            elif safe_moves == 2 and snake_len > 10:
                reward -= 0.5  # Two escape routes when long - be careful
            
            # TAIL REACHABILITY BONUS: Critical for beating the game!
            # When snake is long, reward maintaining access to tail
            if snake_len > 30:
                tail = self.snake[-1]
                body_without_tail = snake_set - {tail}
                if self._can_reach_tail(new_head, body_without_tail, tail):
                    reward += 2.0  # Big reward for keeping escape route!
                else:
                    reward -= 5.0  # Big penalty for boxing ourselves in!
        
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
