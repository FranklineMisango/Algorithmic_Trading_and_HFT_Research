"""
Benchmark Strategies for Portfolio Allocation

Implements traditional portfolio optimization baselines.
"""

import numpy as np
import pandas as pd
import yaml
from typing import Dict, Optional
import warnings
warnings.filterwarnings('ignore')

try:
    import cvxpy as cp
except ImportError:
    print("Warning: cvxpy not installed")


class MarkowitzOptimizer:
    """
    Mean-Variance Optimization (Markowitz).
    Maximizes Sharpe ratio subject to constraints.
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize Markowitz optimizer."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        bench_config = self.config['benchmarks']['markowitz']
        self.estimation_window = bench_config['estimation_window']
        self.risk_free_rate = bench_config['risk_free_rate']
    
    def optimize(
        self,
        returns: pd.DataFrame,
        min_weight: float = 0.0,
        max_weight: float = 0.4
    ) -> np.ndarray:
        """
        Find optimal portfolio weights.
        
        Args:
            returns: Historical returns
            min_weight: Minimum asset weight
            max_weight: Maximum asset weight
            
        Returns:
            Optimal weights
        """
        # Use recent window for estimation
        recent_returns = returns.iloc[-self.estimation_window:]
        
        # Calculate expected returns and covariance
        mu = recent_returns.mean().values * 252  # Annualized
        Sigma = recent_returns.cov().values * 252  # Annualized
        
        n_assets = len(mu)
        
        # Define optimization problem
        w = cp.Variable(n_assets)
        
        # Expected return
        expected_return = mu @ w
        
        # Volatility
        volatility = cp.quad_form(w, Sigma)
        
        # Sharpe ratio (maximize return/risk)
        # We minimize -Sharpe, or equivalently minimize -return/sqrt(risk)
        objective = cp.Maximize(expected_return - 0.5 * volatility)
        
        # Constraints
        constraints = [
            cp.sum(w) == 1.0,           # Fully invested
            w >= min_weight,             # No shorts (or min weight)
            w <= max_weight              # Max position size
        ]
        
        # Solve
        problem = cp.Problem(objective, constraints)
        
        try:
            problem.solve(solver=cp.ECOS)
            
            if w.value is not None:
                weights = w.value
                # Ensure weights sum to 1.0 and are valid
                weights = np.clip(weights, min_weight, max_weight)
                weights = weights / np.sum(weights)
                return weights
            else:
                # Fallback to equal weights
                return np.ones(n_assets) / n_assets
        
        except Exception as e:
            print(f"Optimization failed: {e}")
            return np.ones(n_assets) / n_assets


class Classic6040:
    """
    Classic 60/40 portfolio (60% stocks, 40% bonds).
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize 60/40 strategy."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Assuming first asset is stocks (SPY), second is bonds (AGG)
        self.weights = np.array([0.6, 0.4, 0.0, 0.0])
    
    def get_weights(self) -> np.ndarray:
        """Return static weights."""
        return self.weights


class EqualWeight:
    """
    Equal weight portfolio (1/N).
    """
    
    def __init__(self, n_assets: int = 4):
        """Initialize equal weight strategy."""
        self.n_assets = n_assets
        self.weights = np.ones(n_assets) / n_assets
    
    def get_weights(self) -> np.ndarray:
        """Return equal weights."""
        return self.weights


class BenchmarkBacktester:
    """Backtest benchmark strategies."""
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize backtester."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        backtest_config = self.config['backtest']
        self.commission_bps = backtest_config['costs']['commission_bps']
        self.slippage_bps = backtest_config['costs']['slippage_bps']
    
    def backtest_markowitz(
        self,
        prices: pd.DataFrame,
        returns: pd.DataFrame,
        rebalance_freq: int = 21  # Monthly
    ) -> Dict:
        """
        Backtest Markowitz strategy.
        
        Args:
            prices: Price DataFrame
            returns: Returns DataFrame
            rebalance_freq: Rebalancing frequency (days)
            
        Returns:
            Backtest results
        """
        optimizer = MarkowitzOptimizer()
        
        portfolio_value = 1.0
        weights = np.ones(len(prices.columns)) / len(prices.columns)
        
        portfolio_history = [portfolio_value]
        weights_history = [weights.copy()]
        
        lookback = self.config['benchmarks']['markowitz']['estimation_window']
        
        for i in range(lookback, len(returns)):
            # Rebalance periodically
            if i % rebalance_freq == 0:
                # Optimize weights
                recent_returns = returns.iloc[:i]
                new_weights = optimizer.optimize(recent_returns)
                
                # Calculate transaction costs
                cost = np.sum(np.abs(new_weights - weights)) * (
                    (self.commission_bps + self.slippage_bps) / 10000
                )
                
                weights = new_weights
                portfolio_value *= (1 - cost)
            
            # Calculate portfolio return
            period_return = np.dot(weights, returns.iloc[i].values)
            portfolio_value *= (1 + period_return)
            
            portfolio_history.append(portfolio_value)
            weights_history.append(weights.copy())
        
        return {
            'portfolio_history': portfolio_history,
            'weights_history': weights_history,
            'final_value': portfolio_value
        }
    
    def backtest_static(
        self,
        prices: pd.DataFrame,
        returns: pd.DataFrame,
        strategy: object,
        rebalance_freq: int = 63  # Quarterly
    ) -> Dict:
        """
        Backtest static strategy (60/40 or equal weight).
        
        Args:
            prices: Price DataFrame
            returns: Returns DataFrame
            strategy: Strategy object with get_weights()
            rebalance_freq: Rebalancing frequency
            
        Returns:
            Backtest results
        """
        portfolio_value = 1.0
        weights = strategy.get_weights()
        
        portfolio_history = [portfolio_value]
        weights_history = [weights.copy()]
        
        for i in range(len(returns)):
            # Rebalance periodically (incur costs)
            if i % rebalance_freq == 0 and i > 0:
                cost = (self.commission_bps + self.slippage_bps) / 10000
                portfolio_value *= (1 - cost * 0.1)  # Smaller cost for rebalancing
            
            # Calculate portfolio return
            period_return = np.dot(weights, returns.iloc[i].values)
            portfolio_value *= (1 + period_return)
            
            portfolio_history.append(portfolio_value)
            weights_history.append(weights.copy())
        
        return {
            'portfolio_history': portfolio_history,
            'weights_history': weights_history,
            'final_value': portfolio_value
        }


# Test code
if __name__ == "__main__":
    from data_acquisition import DataAcquisition
    
    data_acq = DataAcquisition('config.yaml')
    dataset = data_acq.fetch_full_dataset()
    
    test_returns = dataset['test']['returns']
    test_prices = dataset['test']['prices']
    
    backtester = BenchmarkBacktester()
    
    # Test Markowitz
    print("Testing Markowitz...")
    mvo_results = backtester.backtest_markowitz(test_prices, test_returns)
    print(f"Final value: ${mvo_results['final_value']:.4f}")
    
    # Test 60/40
    print("\nTesting 60/40...")
    classic = Classic6040()
    classic_results = backtester.backtest_static(test_prices, test_returns, classic)
    print(f"Final value: ${classic_results['final_value']:.4f}")
    
    # Test Equal Weight
    print("\nTesting Equal Weight...")
    equal = EqualWeight(n_assets=4)
    equal_results = backtester.backtest_static(test_prices, test_returns, equal)
    print(f"Final value: ${equal_results['final_value']:.4f}")
