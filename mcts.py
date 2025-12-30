"""
AlphaZero-style Monte Carlo Tree Search for Snake

This MCTS implementation uses:
1. Neural network policy as prior for action selection (PUCT)
2. Neural network value for leaf evaluation
3. Full game simulation for accurate state transitions

Key insight: Snake is deterministic, so we can perfectly simulate future states.
This makes MCTS extremely powerful for endgame situations.
"""

import numpy as np
import torch
import copy
from collections import deque
import math


class MCTSNode:
    """Node in the MCTS tree"""
    
    def __init__(self, state, parent=None, action=None, prior=0.0):
        self.state = state  # Game state dict
        self.parent = parent
        self.action = action  # Action that led to this node
        self.prior = prior  # Prior probability from policy network
        
        self.children = {}  # action -> MCTSNode
        self.visit_count = 0
        self.value_sum = 0.0
        self.is_terminal = False
        self.terminal_value = 0.0
    
    @property
    def value(self):
        """Average value of this node"""
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count
    
    def is_expanded(self):
        """Check if node has been expanded"""
        return len(self.children) > 0
    
    def select_child(self, c_puct=1.5):
        """
        Select child using PUCT formula (AlphaZero style)
        PUCT = Q(s,a) + c_puct * P(s,a) * sqrt(N(s)) / (1 + N(s,a))
        """
        best_score = -float('inf')
        best_action = None
        best_child = None
        
        sqrt_total = math.sqrt(self.visit_count + 1)
        
        for action, child in self.children.items():
            # UCB score with prior
            q_value = child.value if child.visit_count > 0 else 0.0
            exploration = c_puct * child.prior * sqrt_total / (1 + child.visit_count)
            score = q_value + exploration
            
            if score > best_score:
                best_score = score
                best_action = action
                best_child = child
        
        return best_action, best_child
    
    def expand(self, action_priors, next_states, terminals, terminal_values):
        """
        Expand node with all possible actions
        
        Args:
            action_priors: dict {action: prior_probability}
            next_states: dict {action: state_dict}
            terminals: dict {action: is_terminal}
            terminal_values: dict {action: terminal_reward}
        """
        for action, prior in action_priors.items():
            if action not in self.children:
                child = MCTSNode(
                    state=next_states[action],
                    parent=self,
                    action=action,
                    prior=prior
                )
                child.is_terminal = terminals[action]
                child.terminal_value = terminal_values[action]
                self.children[action] = child
    
    def backpropagate(self, value):
        """Backpropagate value up the tree"""
        node = self
        while node is not None:
            node.visit_count += 1
            node.value_sum += value
            node = node.parent


class SnakeSimulator:
    """
    Lightweight Snake simulator for MCTS
    Maintains game state without pygame overhead
    """
    
    DIRECTIONS = [(0, -1), (1, 0), (0, 1), (-1, 0)]  # UP, RIGHT, DOWN, LEFT
    
    def __init__(self, grid_size=10):
        self.grid_size = grid_size
    
    def get_state_dict(self, env):
        """Extract state from SnakeEnv for simulation"""
        return {
            'snake': list(env.snake),
            'food': env.food,
            'direction': env.direction,
            'score': env.score,
            'steps': env.steps,
            'max_steps': env.max_steps
        }
    
    def simulate_action(self, state_dict, action):
        """
        Simulate taking an action from a state
        Returns: (new_state_dict, reward, done, info)
        """
        snake = deque(state_dict['snake'])
        food = state_dict['food']
        direction = state_dict['direction']
        score = state_dict['score']
        steps = state_dict['steps'] + 1
        max_steps = state_dict['max_steps']
        
        # Convert relative action to absolute direction
        if action == 0:
            new_direction = direction
        elif action == 1:
            new_direction = (direction + 1) % 4
        else:
            new_direction = (direction - 1) % 4
        
        # Calculate new head position
        head = snake[0]
        dx, dy = self.DIRECTIONS[new_direction]
        new_head = (head[0] + dx, head[1] + dy)
        
        # Check for collisions
        done = False
        reward = 0
        info = {'score': score, 'reason': ''}
        
        # Wall collision
        if (new_head[0] < 0 or new_head[0] >= self.grid_size or
            new_head[1] < 0 or new_head[1] >= self.grid_size):
            done = True
            reward = -10
            info['reason'] = 'wall'
            return state_dict, reward, done, info
        
        # Self collision
        body_without_tail = list(snake)[:-1]
        if new_head in body_without_tail:
            done = True
            reward = -10
            info['reason'] = 'self'
            return state_dict, reward, done, info
        
        # Move snake
        new_snake = deque([new_head] + list(snake))
        
        # Check food
        if new_head == food:
            score += 1
            reward = 10
            # Place new food (simplified - random position not in snake)
            new_food = self._place_food(new_snake)
        else:
            new_snake.pop()
            new_food = food
            # Distance reward
            old_dist = abs(head[0] - food[0]) + abs(head[1] - food[1])
            new_dist = abs(new_head[0] - food[0]) + abs(new_head[1] - food[1])
            if new_dist < old_dist:
                reward = 1
            else:
                reward = -1.5
        
        # Timeout
        if steps >= max_steps:
            done = True
            reward = -5
            info['reason'] = 'timeout'
        
        info['score'] = score
        
        new_state = {
            'snake': list(new_snake),
            'food': new_food,
            'direction': new_direction,
            'score': score,
            'steps': steps,
            'max_steps': max_steps
        }
        
        return new_state, reward, done, info
    
    def _place_food(self, snake):
        """Place food at random empty position"""
        snake_set = set(snake)
        empty = [(x, y) for x in range(self.grid_size) 
                 for y in range(self.grid_size) if (x, y) not in snake_set]
        if empty:
            return empty[np.random.randint(len(empty))]
        return None
    
    def get_features(self, state_dict):
        """
        Get feature vector from state dict (same as SnakeEnv._get_state)
        Returns 30-dimensional feature vector
        """
        snake = deque(state_dict['snake'])
        food = state_dict['food']
        direction = state_dict['direction']
        head = snake[0]
        
        features = []
        
        # Relative directions
        dir_straight = direction
        dir_right = (direction + 1) % 4
        dir_left = (direction - 1) % 4
        relative_dirs = [dir_straight, dir_right, dir_left]
        
        # 1. Danger signals (3)
        for abs_dir in relative_dirs:
            dx, dy = self.DIRECTIONS[abs_dir]
            next_pos = (head[0] + dx, head[1] + dy)
            danger = self._is_collision(next_pos, snake)
            features.append(1 if danger else 0)
        
        # 2. Food direction relative (3)
        food_dx = food[0] - head[0]
        food_dy = food[1] - head[1]
        for abs_dir in relative_dirs:
            dir_dx, dir_dy = self.DIRECTIONS[abs_dir]
            dot = food_dx * dir_dx + food_dy * dir_dy
            features.append(1 if dot > 0 else 0)
        
        # 3. Food distance (1)
        food_dist = (abs(head[0] - food[0]) + abs(head[1] - food[1])) / (2 * self.grid_size)
        features.append(food_dist)
        
        # 4. Wall distance (3)
        for abs_dir in relative_dirs:
            wall_dist = self._get_wall_distance(head, abs_dir)
            features.append(wall_dist)
        
        # 5. Depth to obstacle (3)
        for abs_dir in relative_dirs:
            depth = self._get_depth(head, abs_dir, snake)
            features.append(depth)
        
        # 6. Would die (3)
        for abs_dir in relative_dirs:
            dx, dy = self.DIRECTIONS[abs_dir]
            next_pos = (head[0] + dx, head[1] + dy)
            features.append(1 if self._is_collision(next_pos, snake) else 0)
        
        # 7. Would eat (3)
        for abs_dir in relative_dirs:
            dx, dy = self.DIRECTIONS[abs_dir]
            next_pos = (head[0] + dx, head[1] + dy)
            features.append(1 if next_pos == food else 0)
        
        # 8. Distance change (3)
        current_dist = abs(head[0] - food[0]) + abs(head[1] - food[1])
        for abs_dir in relative_dirs:
            dx, dy = self.DIRECTIONS[abs_dir]
            next_pos = (head[0] + dx, head[1] + dy)
            if self._is_collision(next_pos, snake):
                features.append(1)
            else:
                new_dist = abs(next_pos[0] - food[0]) + abs(next_pos[1] - food[1])
                features.append((new_dist - current_dist) / (2 * self.grid_size))
        
        # 9. Snake length (1)
        features.append(len(snake) / (self.grid_size * self.grid_size))
        
        # 10. Adjacent body (1)
        adjacent = 0
        for dir_idx in range(4):
            dx, dy = self.DIRECTIONS[dir_idx]
            neighbor = (head[0] + dx, head[1] + dy)
            if neighbor in list(snake)[1:]:
                adjacent += 1
        features.append(adjacent / 4)
        
        # 11. Lookahead features (6) - simplified for MCTS
        for abs_dir in relative_dirs:
            min_safe, max_safe = self._lookahead(state_dict, abs_dir)
            features.append(min_safe / 3.0)
            features.append(max_safe / 3.0)
        
        return np.array(features, dtype=np.float32)
    
    def _is_collision(self, pos, snake):
        """Check if position causes collision"""
        if pos[0] < 0 or pos[0] >= self.grid_size or pos[1] < 0 or pos[1] >= self.grid_size:
            return True
        if pos in list(snake)[:-1]:
            return True
        return False
    
    def _get_wall_distance(self, pos, direction):
        """Get normalized distance to wall"""
        dx, dy = self.DIRECTIONS[direction]
        x, y = pos
        distance = 0
        while True:
            x += dx
            y += dy
            distance += 1
            if x < 0 or x >= self.grid_size or y < 0 or y >= self.grid_size:
                break
        return distance / self.grid_size
    
    def _get_depth(self, pos, direction, snake):
        """Get depth to nearest obstacle"""
        dx, dy = self.DIRECTIONS[direction]
        x, y = pos
        distance = 0
        while True:
            x += dx
            y += dy
            distance += 1
            if x < 0 or x >= self.grid_size or y < 0 or y >= self.grid_size:
                break
            if (x, y) in snake:
                break
        return distance / self.grid_size
    
    def _lookahead(self, state_dict, action_dir):
        """Simple 1-step lookahead for safe moves"""
        # Simulate taking action
        test_state, _, done, _ = self.simulate_action(state_dict, 
            0 if action_dir == state_dict['direction'] else 
            1 if action_dir == (state_dict['direction'] + 1) % 4 else 2)
        
        if done:
            return 0, 0
        
        # Count safe moves after
        safe = 0
        snake = deque(test_state['snake'])
        head = snake[0]
        new_dir = test_state['direction']
        
        for turn in [0, 1, 2]:
            if turn == 0:
                d = new_dir
            elif turn == 1:
                d = (new_dir + 1) % 4
            else:
                d = (new_dir - 1) % 4
            
            dx, dy = self.DIRECTIONS[d]
            next_pos = (head[0] + dx, head[1] + dy)
            if not self._is_collision(next_pos, snake):
                safe += 1
        
        return safe, safe


class MCTS:
    """
    Monte Carlo Tree Search with neural network guidance
    """
    
    def __init__(self, policy_network, grid_size=10, c_puct=1.5, 
                 num_simulations=100, device='cpu'):
        self.policy_network = policy_network
        self.grid_size = grid_size
        self.c_puct = c_puct
        self.num_simulations = num_simulations
        self.device = device
        self.simulator = SnakeSimulator(grid_size)
    
    def get_action_probs(self, env, temperature=1.0):
        """
        Run MCTS and return action probabilities
        
        Args:
            env: SnakeEnv instance
            temperature: Controls exploration (higher = more random)
        
        Returns:
            action_probs: numpy array of shape (3,) with probabilities
            best_action: action with highest visit count
        """
        # Create root node
        state_dict = self.simulator.get_state_dict(env)
        root = MCTSNode(state=state_dict)
        
        # Run simulations
        for _ in range(self.num_simulations):
            node = root
            search_path = [node]
            
            # Selection: traverse tree until unexpanded node
            while node.is_expanded() and not node.is_terminal:
                action, node = node.select_child(self.c_puct)
                search_path.append(node)
            
            # If terminal, backpropagate terminal value
            if node.is_terminal:
                value = node.terminal_value
            else:
                # Expansion: expand node and evaluate with neural network
                value = self._expand_node(node)
            
            # Backpropagation
            for node in reversed(search_path):
                node.visit_count += 1
                node.value_sum += value
                value = -value  # Flip value for opponent (not needed in Snake, but good practice)
        
        # Extract action probabilities from visit counts
        visits = np.array([root.children[a].visit_count if a in root.children else 0 
                          for a in range(3)])
        
        if temperature == 0:
            # Deterministic: choose best action
            action_probs = np.zeros(3)
            action_probs[np.argmax(visits)] = 1.0
        else:
            # Stochastic: sample proportionally to visits
            visits_temp = visits ** (1.0 / temperature)
            action_probs = visits_temp / visits_temp.sum()
        
        best_action = np.argmax(visits)
        
        return action_probs, best_action
    
    def _expand_node(self, node):
        """
        Expand a node and return its value estimate
        """
        state_dict = node.state
        
        # Get neural network predictions
        features = self.simulator.get_features(state_dict)
        with torch.no_grad():
            features_tensor = torch.FloatTensor(features).to(self.device)
            action_probs, value = self.policy_network(features_tensor)
            action_probs = action_probs.cpu().numpy()
            value = value.item()
        
        # Simulate each action
        next_states = {}
        terminals = {}
        terminal_values = {}
        action_priors = {}
        
        for action in range(3):
            next_state, reward, done, info = self.simulator.simulate_action(state_dict, action)
            next_states[action] = next_state
            terminals[action] = done
            terminal_values[action] = reward if done else 0
            action_priors[action] = action_probs[action]
        
        # Expand node
        node.expand(action_priors, next_states, terminals, terminal_values)
        
        return value
    
    def select_action(self, env, temperature=1.0):
        """
        Select action using MCTS
        
        Args:
            env: SnakeEnv instance
            temperature: Controls exploration
        
        Returns:
            action: selected action (0, 1, or 2)
        """
        action_probs, best_action = self.get_action_probs(env, temperature)
        
        if temperature == 0:
            return best_action
        else:
            return np.random.choice(3, p=action_probs)


class HybridAgent:
    """
    Hybrid agent that uses PPO for normal play and MCTS for high scores
    
    Strategy:
    - Snake length < mcts_threshold: Use PPO (fast)
    - Snake length >= mcts_threshold: Use MCTS (accurate)
    
    IMPORTANT: Even when using MCTS, we still record to PPO buffers for training.
    """
    
    def __init__(self, ppo_agent, mcts_threshold=50, num_simulations=100):
        self.ppo_agent = ppo_agent
        self.mcts_threshold = mcts_threshold
        self.mcts = MCTS(
            policy_network=ppo_agent.policy,
            grid_size=10,
            num_simulations=num_simulations,
            device=ppo_agent.device
        )
        self.use_mcts_count = 0
        self.use_ppo_count = 0
    
    def select_action(self, state, env, action_mask=None):
        """
        Select action using either PPO or MCTS depending on snake length.
        ALWAYS records to PPO buffers for consistent training.
        """
        snake_length = len(env.snake)
        
        if snake_length >= self.mcts_threshold:
            # Use MCTS for careful planning at high scores
            self.use_mcts_count += 1
            action = self.mcts.select_action(env, temperature=0.5)
            # CRITICAL: Still record to PPO buffers so tensor sizes match!
            # We use PPO's select_action but override the action choice
            self._record_action_to_ppo(state, action, action_mask)
            return action
        else:
            # Use PPO for fast exploration
            self.use_ppo_count += 1
            return self.ppo_agent.select_action(state, action_mask)
    
    def _record_action_to_ppo(self, state, mcts_action, action_mask=None):
        """
        Record MCTS-selected action to PPO buffers for training consistency.
        This ensures tensor sizes match during PPO update.
        """
        import torch
        from torch.distributions import Categorical
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).to(self.ppo_agent.device)
            action_probs, state_value = self.ppo_agent.policy_old(state_tensor)
            
            # Apply mask if provided
            if action_mask is not None:
                mask_tensor = torch.FloatTensor(action_mask).to(self.ppo_agent.device)
                masked_probs = action_probs * mask_tensor
                if masked_probs.sum() > 0:
                    masked_probs = masked_probs / masked_probs.sum()
                else:
                    masked_probs = action_probs
                action_probs_to_use = masked_probs
            else:
                action_probs_to_use = action_probs
            
            dist = Categorical(action_probs_to_use)
            action_tensor = torch.tensor(mcts_action)
            action_logprob = dist.log_prob(action_tensor)
        
        # Store to PPO buffers
        self.ppo_agent.states.append(state_tensor)
        self.ppo_agent.actions.append(mcts_action)
        self.ppo_agent.logprobs.append(action_logprob)
        self.ppo_agent.state_values.append(state_value)
    
    def store_transition(self, reward, is_terminal):
        """Store transition for PPO training"""
        self.ppo_agent.store_transition(reward, is_terminal)
    
    def update(self):
        """Update PPO policy"""
        self.ppo_agent.update()
    
    def save(self, filepath):
        """Save model"""
        self.ppo_agent.save(filepath)
    
    def load(self, filepath):
        """Load model"""
        self.ppo_agent.load(filepath)
    
    def get_stats(self):
        """Get usage statistics"""
        total = self.use_mcts_count + self.use_ppo_count
        if total == 0:
            return "No actions taken yet"
        return f"PPO: {self.use_ppo_count} ({100*self.use_ppo_count/total:.1f}%), MCTS: {self.use_mcts_count} ({100*self.use_mcts_count/total:.1f}%)"

