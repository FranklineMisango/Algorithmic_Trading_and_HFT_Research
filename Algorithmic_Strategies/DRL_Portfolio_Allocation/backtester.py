"""
Backtester for DRL Portfolio Allocation

Evaluates agent performance and compares to benchmarks.
"""

import numpy as np
import pandas as pd
import yaml
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

try:
    from scipy import stats
except ImportError:
    print("Warning: scipy not installed")


class Backtester:
    """
    Backtest DRL agent and calculate performance metrics.
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize backtester."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        backtest_config = self.config['backtest']
        self.commission_bps = backtest_config['costs']['commission_bps']
        self.slippage_bps = backtest_config['costs']['slippage_bps']
        self.min_trade_size = backtest_config['rebalancing']['min_trade_size']
        self.max_drawdown_threshold = backtest_config['risk']['max_drawdown_threshold']
    
    def run_backtest(
        self,
        agent,
        prices: pd.DataFrame,
        returns: pd.DataFrame,
        initial_capital: float = 1.0
    ) -> Dict:
        """
        Run backtest with trained agent.
        
        Args:
            agent: Trained DRL agent
            prices: Price DataFrame
            returns: Returns DataFrame
            initial_capital: Starting capital
            
        Returns:
            Backtest results dict
        """
        from portfolio_env import PortfolioEnv
        
        # Create environment
        env = PortfolioEnv(prices, returns)
        
        # Initialize
        obs, _ = env.reset()
        done = False
        
        portfolio_history = []
        weights_history = []
        returns_history = []
        dates = []
        
        step = 0
        
        while not done:
            # Get action from agent
            action = agent.predict(obs, deterministic=True)
            
            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Store results
            portfolio_history.append(info['portfolio_value'])
            weights_history.append(env.current_weights.copy())
            returns_history.append(info['portfolio_return'])
            dates.append(returns.index[env.current_step])
            
            done = terminated or truncated
            step += 1
            
            # Check circuit breaker
            if info['drawdown'] >= self.max_drawdown_threshold:
                print(f"Circuit breaker triggered at step {step}")
                break
        
        # Create results DataFrame
        results_df = pd.DataFrame({
            'portfolio_value': portfolio_history,
            'returns': returns_history
        }, index=dates)
        
        # Calculate metrics
        metrics = self.calculate_metrics(results_df, initial_capital)
        
        return {
            'results_df': results_df,
            'weights_history': weights_history,
            'metrics': metrics
        }
    
    def calculate_metrics(
        self,
        results_df: pd.DataFrame,
        initial_capital: float = 1.0
    ) -> Dict:
        """
        Calculate performance metrics.
        
        Args:
            results_df: Results DataFrame
            initial_capital: Initial capital
            
        Returns:
            Metrics dict
        """
        portfolio_values = results_df['portfolio_value'].values
        returns = results_df['returns'].values
        
        # Total return
        total_return = (portfolio_values[-1] - initial_capital) / initial_capital
        
        # Annualized return (CAGR)
        n_days = len(portfolio_values)
        n_years = n_days / 252
        annualized_return = (portfolio_values[-1] / initial_capital) ** (1 / n_years) - 1
        
        # Volatility (annualized)
        volatility = np.std(returns) * np.sqrt(252)
        
        # Sharpe ratio
        sharpe_ratio = annualized_return / (volatility + 1e-8)
        
        # Sortino ratio (downside deviation)
        downside_returns = returns[returns < 0]
        downside_std = np.std(downside_returns) * np.sqrt(252) if len(downside_returns) > 0 else 1e-8
        sortino_ratio = annualized_return / downside_std
        
        # Maximum drawdown
        cummax = np.maximum.accumulate(portfolio_values)
        drawdowns = (cummax - portfolio_values) / cummax
        max_drawdown = np.max(drawdowns)
        
        # Calmar ratio
        calmar_ratio = annualized_return / (max_drawdown + 1e-8)
        
        # Win rate
        win_rate = np.sum(returns > 0) / len(returns)
        
        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'n_days': n_days
        }
    
    def compare_strategies(
        self,
        results: Dict[str, Dict]
    ) -> pd.DataFrame:
        """
        Compare multiple strategies.
        
        Args:
            results: Dict of strategy results
            
        Returns:
            Comparison DataFrame
        """
        comparison = {}
        
        for strategy_name, result in results.items():
            metrics = result['metrics']
            comparison[strategy_name] = {
                'Total Return': f"{metrics['total_return']:.2%}",
                'Annual Return': f"{metrics['annualized_return']:.2%}",
                'Volatility': f"{metrics['volatility']:.2%}",
                'Sharpe Ratio': f"{metrics['sharpe_ratio']:.3f}",
                'Sortino Ratio': f"{metrics['sortino_ratio']:.3f}",
                'Calmar Ratio': f"{metrics['calmar_ratio']:.3f}",
                'Max Drawdown': f"{metrics['max_drawdown']:.2%}",
                'Win Rate': f"{metrics['win_rate']:.2%}"
            }
        
        return pd.DataFrame(comparison).T
    
    def statistical_tests(
        self,
        strategy_returns: np.ndarray,
        benchmark_returns: np.ndarray
    ) -> Dict:
        """
        Perform statistical tests.
        
        Args:
            strategy_returns: Strategy returns
            benchmark_returns: Benchmark returns
            
        Returns:
            Test results
        """
        # T-test for mean difference
        t_stat, p_value = stats.ttest_ind(strategy_returns, benchmark_returns)
        
        # Check if difference is significant
        is_significant = p_value < 0.05
        
        return {
            't_statistic': t_stat,
            'p_value': p_value,
            'is_significant': is_significant,
            'mean_diff': np.mean(strategy_returns) - np.mean(benchmark_returns)
        }


# Test code
if __name__ == "__main__":
    from data_acquisition import DataAcquisition
    from rl_agent import DRLAgent
    from portfolio_env import PortfolioEnv
    
    data_acq = DataAcquisition('config.yaml')
    dataset = data_acq.fetch_full_dataset()
    
    # Create environment and agent
    test_prices = dataset['test']['prices']
    test_returns = dataset['test']['returns']
    
    env = PortfolioEnv(test_prices, test_returns)
    agent = DRLAgent(env, algorithm='ppo')
    
    # Run backtest
    backtester = Backtester()
    results = backtester.run_backtest(agent, test_prices, test_returns)
    
    print("Backtest Metrics:")
    for metric, value in results['metrics'].items():
        if isinstance(value, float):
            print(f"  {metric}: {value:.4f}")
        else:
            print(f"  {metric}: {value}")
