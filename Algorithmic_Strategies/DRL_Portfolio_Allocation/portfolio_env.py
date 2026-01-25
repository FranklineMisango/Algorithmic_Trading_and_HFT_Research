"""
Portfolio Gym Environment for DRL

Custom Gymnasium environment for portfolio allocation.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import yaml
from typing import Optional, Dict, Tuple


class PortfolioEnv(gym.Env):
    """
    Custom Gymnasium environment for portfolio allocation.
    
    State: Portfolio weights + market features
    Action: Target portfolio weights (continuous)
    Reward: Log return - volatility penalty - transaction costs
    """
    
    metadata = {'render_modes': ['human']}
    
    def __init__(
        self,
        prices: pd.DataFrame,
        returns: pd.DataFrame,
        config_path: str = "config.yaml"
    ):
        """
        Initialize environment.
        
        Args:
            prices: Price DataFrame
            returns: Returns DataFrame
            config_path: Path to config file
        """
        super().__init__()
        
        # Load config
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.prices = prices
        self.returns = returns
        self.n_assets = len(prices.columns)
        
        # Extract config
        env_config = self.config['environment']
        self.lookback = self.config['data']['lookback_window']
        
        self.min_weight = env_config['constraints']['min_weight']
        self.max_weight = env_config['constraints']['max_weight']
        self.max_leverage = env_config['constraints']['max_leverage']
        
        reward_config = env_config['reward']
        self.return_weight = reward_config['weights'][0]
        self.volatility_weight = reward_config['weights'][1]
        self.cost_weight = reward_config['weights'][3]
        
        backtest_config = self.config['backtest']
        self.commission_bps = backtest_config['costs']['commission_bps']
        self.slippage_bps = backtest_config['costs']['slippage_bps']
        self.min_trade_size = backtest_config['rebalancing']['min_trade_size']
        
        # Calculate state dimension
        # [current_weights(4), returns(4), volatility(4), momentum(4), correlations(6)] = 22
        self.state_dim = (
            self.n_assets +          # Current weights
            self.n_assets +          # Returns
            self.n_assets +          # Volatility
            self.n_assets +          # Momentum
            int(self.n_assets * (self.n_assets - 1) / 2)  # Correlation upper triangle
        )
        
        # Define action and observation spaces
        self.action_space = spaces.Box(
            low=self.min_weight,
            high=self.max_weight,
            shape=(self.n_assets,),
            dtype=np.float32
        )
        
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.state_dim,),
            dtype=np.float32
        )
        
        # Initialize state
        self.current_step = 0
        self.current_weights = None
        self.portfolio_value = 1.0
        self.initial_value = 1.0
        
        # Track history
        self.portfolio_history = []
        self.weights_history = []
        self.returns_history = []
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Dict]:
        """
        Reset environment to initial state.
        
        Returns:
            observation, info dict
        """
        super().reset(seed=seed)
        
        # Start after lookback period
        self.current_step = self.lookback
        
        # Initialize with equal weights
        self.current_weights = np.ones(self.n_assets) / self.n_assets
        self.portfolio_value = 1.0
        self.initial_value = 1.0
        
        # Reset history
        self.portfolio_history = [self.portfolio_value]
        self.weights_history = [self.current_weights.copy()]
        self.returns_history = []
        
        # Get initial observation
        observation = self._get_observation()
        info = {}
        
        return observation, info
    
    def step(
        self,
        action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Take a step in the environment.
        
        Args:
            action: Target portfolio weights
            
        Returns:
            observation, reward, terminated, truncated, info
        """
        # Normalize action to sum to 1.0 (softmax)
        action = np.clip(action, self.min_weight, self.max_weight)
        action = action / (np.sum(action) + 1e-8)
        
        # Calculate transaction costs
        weight_change = np.abs(action - self.current_weights)
        
        # Only count trades above minimum size
        significant_trades = weight_change > self.min_trade_size
        cost = np.sum(weight_change[significant_trades]) * (
            (self.commission_bps + self.slippage_bps) / 10000
        )
        
        # Get returns for current period
        period_returns = self.returns.iloc[self.current_step].values
        
        # Calculate portfolio return
        portfolio_return = np.dot(action, period_returns)
        
        # Calculate portfolio volatility (recent std)
        recent_returns = self.returns.iloc[
            max(0, self.current_step - self.lookback):self.current_step
        ]
        portfolio_volatility = np.std(
            recent_returns.values @ action
        ) * np.sqrt(252)
        
        # Calculate reward
        reward = (
            self.return_weight * portfolio_return -
            self.volatility_weight * portfolio_volatility -
            self.cost_weight * cost
        )
        
        # Update portfolio value
        self.portfolio_value *= (1 + portfolio_return - cost)
        
        # Update weights
        self.current_weights = action
        
        # Store history
        self.portfolio_history.append(self.portfolio_value)
        self.weights_history.append(self.current_weights.copy())
        self.returns_history.append(portfolio_return)
        
        # Move to next step
        self.current_step += 1
        
        # Check if episode is done
        terminated = self.current_step >= len(self.returns) - 1
        
        # Check circuit breaker (15% drawdown)
        max_value = max(self.portfolio_history)
        drawdown = (max_value - self.portfolio_value) / max_value
        truncated = drawdown >= 0.15
        
        # Get new observation
        observation = self._get_observation()
        
        info = {
            'portfolio_value': self.portfolio_value,
            'portfolio_return': portfolio_return,
            'volatility': portfolio_volatility,
            'cost': cost,
            'drawdown': drawdown
        }
        
        return observation, reward, terminated, truncated, info
    
    def _get_observation(self) -> np.ndarray:
        """
        Get current state observation.
        
        Returns:
            State vector
        """
        # Get price and return data up to current step
        prices_slice = self.prices.iloc[:self.current_step+1]
        returns_slice = self.returns.iloc[:self.current_step+1]
        
        # Calculate rolling features
        rolling_returns = prices_slice.pct_change(self.lookback).iloc[-1].values
        
        recent_returns = returns_slice.iloc[-self.lookback:]
        rolling_vol = recent_returns.std().values * np.sqrt(252)
        
        momentum = (prices_slice.iloc[-1] / prices_slice.iloc[-self.lookback] - 1).values
        
        # Correlation matrix
        corr_matrix = recent_returns.corr().values
        corr_flat = corr_matrix[np.triu_indices_from(corr_matrix, k=1)]
        
        # Combine features
        state = np.concatenate([
            self.current_weights,
            rolling_returns,
            rolling_vol,
            momentum,
            corr_flat
        ])
        
        # Normalize
        state = (state - np.mean(state)) / (np.std(state) + 1e-8)
        
        return state.astype(np.float32)
    
    def render(self):
        """Render environment state."""
        if len(self.portfolio_history) > 0:
            print(f"Step: {self.current_step}")
            print(f"Portfolio Value: ${self.portfolio_value:.4f}")
            print(f"Weights: {self.current_weights}")


# Test code
if __name__ == "__main__":
    from data_acquisition import DataAcquisition
    
    data_acq = DataAcquisition('config.yaml')
    dataset = data_acq.fetch_full_dataset()
    
    # Create environment
    env = PortfolioEnv(
        prices=dataset['train']['prices'],
        returns=dataset['train']['returns']
    )
    
    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")
    
    # Test episode
    obs, info = env.reset()
    print(f"\nInitial observation shape: {obs.shape}")
    
    for i in range(5):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Step {i+1}: Reward={reward:.6f}, Value={info['portfolio_value']:.4f}")
        
        if terminated or truncated:
            break
