"""
Neural Network for Snake RL Agent using GRPO
Input: 100 values (10x10 grid state)
Output: 3 action probabilities (UP, RIGHT, LEFT)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import os


class SnakeNet(nn.Module):
    """
    Policy network for Snake game
    Uses a simple MLP architecture suitable for the 10x10 grid input
    """
    def __init__(self, input_size=100, hidden_size=256, output_size=3):
        super(SnakeNet, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Network layers
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc4 = nn.Linear(hidden_size // 2, output_size)
        
        # Layer normalization for stability
        self.ln1 = nn.LayerNorm(hidden_size)
        self.ln2 = nn.LayerNorm(hidden_size)
        self.ln3 = nn.LayerNorm(hidden_size // 2)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Xavier initialization"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x):
        """
        Forward pass
        x: batch of states, shape (batch_size, 100)
        Returns: action logits, shape (batch_size, 3)
        """
        x = F.relu(self.ln1(self.fc1(x)))
        x = F.relu(self.ln2(self.fc2(x)))
        x = F.relu(self.ln3(self.fc3(x)))
        logits = self.fc4(x)
        return logits
    
    def get_action(self, state, deterministic=False):
        """
        Get action from state
        state: numpy array or tensor of shape (100,) or (batch_size, 100)
        Returns: action (int), log_prob (tensor)
        """
        if not isinstance(state, torch.Tensor):
            state = torch.FloatTensor(state)
        
        if state.dim() == 1:
            state = state.unsqueeze(0)
        
        logits = self.forward(state)
        probs = F.softmax(logits, dim=-1)
        
        if deterministic:
            action = torch.argmax(probs, dim=-1)
        else:
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
        
        log_prob = F.log_softmax(logits, dim=-1)
        action_log_prob = log_prob.gather(1, action.unsqueeze(-1)).squeeze(-1)
        
        return action.item() if action.numel() == 1 else action, action_log_prob
    
    def get_log_probs(self, states, actions):
        """
        Get log probabilities for given state-action pairs
        states: tensor of shape (batch_size, 100)
        actions: tensor of shape (batch_size,)
        Returns: log_probs tensor of shape (batch_size,)
        """
        logits = self.forward(states)
        log_probs = F.log_softmax(logits, dim=-1)
        action_log_probs = log_probs.gather(1, actions.unsqueeze(-1)).squeeze(-1)
        return action_log_probs
    
    def save(self, path):
        """Save model weights"""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        torch.save({
            'model_state_dict': self.state_dict(),
            'input_size': self.input_size,
            'hidden_size': self.hidden_size,
            'output_size': self.output_size,
        }, path)
        print(f"Model saved to {path}")
    
    @classmethod
    def load(cls, path, device='cpu'):
        """Load model weights"""
        if not os.path.exists(path):
            raise FileNotFoundError(f"No model found at {path}")
        
        checkpoint = torch.load(path, map_location=device, weights_only=True)
        model = cls(
            input_size=checkpoint['input_size'],
            hidden_size=checkpoint['hidden_size'],
            output_size=checkpoint['output_size']
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        print(f"Model loaded from {path}")
        return model


class GRPOTrainer:
    """
    Group Relative Policy Optimization (GRPO) Trainer
    
    GRPO works by:
    1. Collecting groups of trajectories
    2. Computing advantages relative to the group mean
    3. Optimizing policy to increase probability of better-than-average actions
    
    Key insight for Snake: The issue of the snake hitting itself or getting trapped
    is addressed by the reward structure (-10 for collision) and by the GRPO
    algorithm naturally learning to avoid these situations through relative comparison.
    """
    def __init__(self, model, lr=3e-4, gamma=0.99, group_size=8, 
                 clip_epsilon=0.2, entropy_coef=0.01, device='cpu'):
        self.model = model
        self.model.to(device)
        self.device = device
        
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.gamma = gamma
        self.group_size = group_size  # Number of trajectories to compare
        self.clip_epsilon = clip_epsilon
        self.entropy_coef = entropy_coef
        
        # For storing old policy for GRPO update
        self.old_model = None
    
    def compute_returns(self, rewards, dones):
        """Compute discounted returns for a trajectory"""
        returns = []
        G = 0
        for r, done in zip(reversed(rewards), reversed(dones)):
            if done:
                G = 0
            G = r + self.gamma * G
            returns.insert(0, G)
        return returns
    
    def collect_trajectory(self, env, max_steps=400):
        """Collect a single trajectory"""
        states = []
        actions = []
        rewards = []
        log_probs = []
        dones = []
        
        state = env.reset()
        done = False
        steps = 0
        
        while not done and steps < max_steps:
            state_tensor = torch.FloatTensor(state).to(self.device)
            
            with torch.no_grad():
                action, log_prob = self.model.get_action(state_tensor)
            
            states.append(state)
            actions.append(action)
            log_probs.append(log_prob.item())
            
            state, reward, done, info = env.step(action)
            
            rewards.append(reward)
            dones.append(done)
            steps += 1
        
        return {
            'states': states,
            'actions': actions,
            'rewards': rewards,
            'log_probs': log_probs,
            'dones': dones,
            'total_reward': sum(rewards),
            'score': info.get('score', 0),
            'length': steps,
            'reason': info.get('reason', 'unknown')
        }
    
    def collect_group(self, env, render_fn=None):
        """Collect a group of trajectories for GRPO"""
        group = []
        for i in range(self.group_size):
            traj = self.collect_trajectory(env)
            group.append(traj)
            if render_fn:
                render_fn()
        return group
    
    def update(self, group):
        """
        GRPO update: optimize policy based on group-relative advantages
        """
        # Compute returns for each trajectory
        for traj in group:
            traj['returns'] = self.compute_returns(traj['rewards'], traj['dones'])
        
        # Compute group statistics
        all_returns = []
        for traj in group:
            all_returns.extend(traj['returns'])
        
        mean_return = sum(all_returns) / len(all_returns) if all_returns else 0
        std_return = (sum((r - mean_return) ** 2 for r in all_returns) / len(all_returns)) ** 0.5 if all_returns else 1
        std_return = max(std_return, 1e-8)  # Prevent division by zero
        
        # Prepare batch
        all_states = []
        all_actions = []
        all_advantages = []
        all_old_log_probs = []
        
        for traj in group:
            for i, (state, action, ret, old_log_prob) in enumerate(
                zip(traj['states'], traj['actions'], traj['returns'], traj['log_probs'])
            ):
                all_states.append(state)
                all_actions.append(action)
                # Group-relative advantage
                advantage = (ret - mean_return) / std_return
                all_advantages.append(advantage)
                all_old_log_probs.append(old_log_prob)
        
        if not all_states:
            return 0, 0
        
        # Convert to tensors
        states = torch.FloatTensor(all_states).to(self.device)
        actions = torch.LongTensor(all_actions).to(self.device)
        advantages = torch.FloatTensor(all_advantages).to(self.device)
        old_log_probs = torch.FloatTensor(all_old_log_probs).to(self.device)
        
        # Compute new log probs
        new_log_probs = self.model.get_log_probs(states, actions)
        
        # GRPO/PPO-style clipped objective
        ratio = torch.exp(new_log_probs - old_log_probs)
        clipped_ratio = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon)
        
        policy_loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()
        
        # Entropy bonus for exploration
        logits = self.model(states)
        probs = F.softmax(logits, dim=-1)
        entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1).mean()
        entropy_loss = -self.entropy_coef * entropy
        
        # Total loss
        loss = policy_loss + entropy_loss
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
        self.optimizer.step()
        
        return loss.item(), entropy.item()

