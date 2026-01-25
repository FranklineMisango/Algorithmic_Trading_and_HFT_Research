"""
Reinforcement Learning Agent for Pairs Trading

Deep Q-Network (DQN) agent that learns to trade pairs based on profit/loss rewards.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
from typing import Tuple, List
import yaml


class DQN(nn.Module):
    """Deep Q-Network for state-action value estimation."""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dims: List[int] = [128, 64]):
        """
        Initialize DQN architecture.
        
        Args:
            state_dim: Dimension of state vector
            action_dim: Number of actions (3: buy, sell, hold)
            hidden_dims: List of hidden layer dimensions
        """
        super(DQN, self).__init__()
        
        layers = []
        input_dim = state_dim
        
        # Build hidden layers
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            input_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(input_dim, action_dim))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, state):
        """Forward pass through network."""
        return self.network(state)


class ReplayBuffer:
    """Experience replay buffer for DQN training."""
    
    def __init__(self, capacity: int = 10000):
        """Initialize buffer with max capacity."""
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        """Add experience to buffer."""
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int):
        """Sample random batch from buffer."""
        batch = random.sample(self.buffer, batch_size)
        
        states, actions, rewards, next_states, dones = zip(*batch)
        
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones)
        )
    
    def __len__(self):
        """Return current buffer size."""
        return len(self.buffer)


class PairsTradingEnv:
    """Trading environment for RL agent."""
    
    def __init__(self,
                 states: pd.DataFrame,
                 price1: pd.Series,
                 price2: pd.Series,
                 initial_capital: float = 100000):
        """
        Initialize trading environment.
        
        Args:
            states: State features DataFrame
            price1, price2: Price series for the pair
            initial_capital: Starting capital
        """
        self.states = states.values
        self.price1 = price1.loc[states.index].values
        self.price2 = price2.loc[states.index].values
        self.initial_capital = initial_capital
        
        self.n_steps = len(self.states)
        self.current_step = 0
        
        # Position: 0 = no position, 1 = long pair, -1 = short pair
        self.position = 0
        self.capital = initial_capital
        self.entry_price1 = 0
        self.entry_price2 = 0
        
        # Performance tracking
        self.portfolio_values = []
        self.trades = []
        
    def reset(self):
        """Reset environment to initial state."""
        self.current_step = 0
        self.position = 0
        self.capital = self.initial_capital
        self.entry_price1 = 0
        self.entry_price2 = 0
        self.portfolio_values = []
        self.trades = []
        
        return self.states[0]
    
    def step(self, action):
        """
        Execute action and return (next_state, reward, done, info).
        
        Args:
            action: 0 = buy pair, 1 = sell pair, 2 = hold
            
        Returns:
            (next_state, reward, done, info)
        """
        # Get current prices
        curr_price1 = self.price1[self.current_step]
        curr_price2 = self.price2[self.current_step]
        
        reward = 0
        
        # Execute action
        if action == 0:  # Buy pair (long stock1, short stock2)
            if self.position == 0:
                # Open long position
                self.position = 1
                self.entry_price1 = curr_price1
                self.entry_price2 = curr_price2
                
        elif action == 1:  # Sell pair (short stock1, long stock2)
            if self.position == 0:
                # Open short position
                self.position = -1
                self.entry_price1 = curr_price1
                self.entry_price2 = curr_price2
                
        elif action == 2:  # Hold or close
            if self.position != 0:
                # Close position and calculate PnL
                pnl = self._calculate_pnl(curr_price1, curr_price2)
                reward = pnl / self.initial_capital  # Normalized reward
                
                self.capital += pnl
                self.trades.append({
                    'step': self.current_step,
                    'pnl': pnl,
                    'position': self.position
                })
                
                # Reset position
                self.position = 0
        
        # Move to next step
        self.current_step += 1
        done = self.current_step >= self.n_steps - 1
        
        # Get next state
        next_state = self.states[self.current_step] if not done else self.states[-1]
        
        # Track portfolio value
        current_value = self.capital
        if self.position != 0:
            current_value += self._calculate_pnl(curr_price1, curr_price2)
        
        self.portfolio_values.append(current_value)
        
        info = {
            'portfolio_value': current_value,
            'position': self.position,
            'num_trades': len(self.trades)
        }
        
        return next_state, reward, done, info
    
    def _calculate_pnl(self, curr_price1, curr_price2):
        """Calculate current PnL for open position."""
        if self.position == 0:
            return 0
        
        # Dollar-neutral sizing (equal $ amounts)
        position_size = self.initial_capital * 0.1  # 10% per leg
        
        # Calculate returns
        ret1 = (curr_price1 - self.entry_price1) / self.entry_price1
        ret2 = (curr_price2 - self.entry_price2) / self.entry_price2
        
        if self.position == 1:
            # Long stock1, short stock2
            pnl = position_size * (ret1 - ret2)
        else:
            # Short stock1, long stock2
            pnl = position_size * (ret2 - ret1)
        
        return pnl


class DQNAgent:
    """DQN agent for pairs trading."""
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize agent with configuration."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.rl_config = self.config['rl_agent']
        
        # Action space
        self.actions = self.rl_config['actions']
        self.n_actions = len(self.actions)
        
        # Exploration parameters
        self.epsilon = self.rl_config['epsilon_start']
        self.epsilon_end = self.rl_config['epsilon_end']
        self.epsilon_decay = self.rl_config['epsilon_decay']
        
        # Learning parameters
        self.gamma = self.rl_config['gamma']
        self.lr = self.rl_config['learning_rate']
        self.batch_size = self.rl_config['batch_size']
        
        # Networks (will be initialized after knowing state_dim)
        self.policy_net = None
        self.target_net = None
        self.optimizer = None
        
        # Experience replay
        self.memory = ReplayBuffer(self.rl_config['memory_size'])
        
        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def initialize_networks(self, state_dim: int):
        """Initialize DQN networks after knowing state dimension."""
        self.policy_net = DQN(state_dim, self.n_actions).to(self.device)
        self.target_net = DQN(state_dim, self.n_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)
    
    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """
        Select action using epsilon-greedy policy.
        
        Args:
            state: Current state vector
            training: Whether in training mode (use exploration)
            
        Returns:
            Action index
        """
        if training and random.random() < self.epsilon:
            # Explore: random action
            return random.randrange(self.n_actions)
        else:
            # Exploit: use policy network
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.policy_net(state_tensor)
                return q_values.argmax().item()
    
    def train_step(self):
        """Perform one training step with experience replay."""
        if len(self.memory) < self.batch_size:
            return
        
        # Sample batch from memory
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Current Q values
        current_q = self.policy_net(states).gather(1, actions.unsqueeze(1))
        
        # Next Q values from target network
        with torch.no_grad():
            next_q = self.target_net(next_states).max(1)[0]
            target_q = rewards + self.gamma * next_q * (1 - dones)
        
        # Compute loss
        loss = nn.MSELoss()(current_q.squeeze(), target_q)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        
        return loss.item()
    
    def update_target_network(self):
        """Update target network with policy network weights."""
        self.target_net.load_state_dict(self.policy_net.state_dict())
    
    def decay_epsilon(self):
        """Decay exploration rate."""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
    
    def save(self, path: str):
        """Save agent weights."""
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, path)
    
    def load(self, path: str):
        """Load agent weights."""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']


if __name__ == "__main__":
    # Test RL agent
    from data_acquisition import DataAcquisition
    from feature_engineering import FeatureEngineer
    
    # Fetch data
    data_acq = DataAcquisition()
    dataset = data_acq.fetch_full_dataset()
    train_prices, _ = data_acq.split_train_test(dataset['prices'])
    
    # Create features
    feature_eng = FeatureEngineer()
    states = feature_eng.create_state_vector(
        train_prices['MSFT'],
        train_prices['GOOGL']
    )
    states = feature_eng.normalize_features(states)
    
    # Create environment
    env = PairsTradingEnv(
        states,
        train_prices['MSFT'],
        train_prices['GOOGL']
    )
    
    # Create agent
    agent = DQNAgent()
    agent.initialize_networks(states.shape[1])
    
    # Test single episode
    state = env.reset()
    total_reward = 0
    
    for _ in range(100):
        action = agent.select_action(state)
        next_state, reward, done, info = env.step(action)
        
        total_reward += reward
        state = next_state
        
        if done:
            break
    
    print(f"Test episode: Total reward = {total_reward:.4f}")
    print(f"Final portfolio value: ${info['portfolio_value']:,.2f}")
