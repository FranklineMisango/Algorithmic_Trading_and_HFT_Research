"""
Backtester for Jump-Adjusted Portfolio Strategy
Tests portfolio performance with monthly rebalancing
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from datetime import datetime
import logging

from portfolio_optimizer import PortfolioOptimizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class JumpRiskBacktester:
    """
    Backtest jump-adjusted portfolio strategy
    """
    
    def __init__(self, config: Dict):
        """
        Initialize backtester
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.backtest_config = config['backtesting']
        self.initial_capital = self.backtest_config['initial_capital']
        self.rebalance_freq = self.backtest_config['rebalancing_frequency']
        
        self.optimizer = PortfolioOptimizer(config)
        
    def backtest_strategy(
        self, 
        df_with_jumps: pd.DataFrame,
        strategy_name: str = 'jump_adjusted'
    ) -> pd.DataFrame:
        """
        Run backtest for a single strategy
        
        Args:
            df_with_jumps: DataFrame with returns and jumps
            strategy_name: Name of strategy
            
        Returns:
            DataFrame with portfolio values over time
        """
        logger.info(f"Backtesting {strategy_name} strategy...")
        
        # Prepare data
        returns_df = df_with_jumps.pivot(
            index='date', columns='asset', values='returns'
        ).dropna()
        
        jump_returns_df = df_with_jumps.pivot(
            index='date', columns='asset', values='jump_size'
        ).fillna(0)
        
        # Align dates
        common_dates = returns_df.index.intersection(jump_returns_df.index)
        returns_df = returns_df.loc[common_dates]
        jump_returns_df = jump_returns_df.loc[common_dates]
        
        # Initialize portfolio
        portfolio_value = self.initial_capital
        portfolio_values = []
        weights_history = []
        rebalance_dates = []
        
        # Get rebalancing dates
        rebalance_schedule = self._get_rebalance_schedule(returns_df.index)
        
        current_weights = None
        
        for date in returns_df.index:
            # Rebalance if needed
            if date in rebalance_schedule:
                # Get historical data up to this point
                historical_returns = returns_df.loc[:date]
                historical_jumps = jump_returns_df.loc[:date]
                
                # Optimize portfolio
                if strategy_name == 'jump_adjusted':
                    opt_result = self.optimizer.optimize_portfolio(
                        historical_returns, historical_jumps
                    )
                    current_weights = opt_result['optimal_weights']
                elif strategy_name == 'standard_minvar':
                    opt_result = self.optimizer.optimize_portfolio(
                        historical_returns, historical_jumps
                    )
                    current_weights = opt_result['standard_weights']
                elif strategy_name == 'equal_weight':
                    current_weights = {asset: 1.0 / len(returns_df.columns) 
                                     for asset in returns_df.columns}
                elif strategy_name == 'btc_eth_6040':
                    current_weights = {asset: 0 for asset in returns_df.columns}
                    if 'BTC' in current_weights:
                        current_weights['BTC'] = 0.6
                    if 'ETH' in current_weights:
                        current_weights['ETH'] = 0.4
                
                rebalance_dates.append(date)
                weights_history.append({
                    'date': date,
                    'weights': current_weights.copy()
                })
                
                logger.info(f"  Rebalanced on {date.strftime('%Y-%m-%d')}")
            
            # Apply returns with current weights
            if current_weights is not None:
                weight_array = np.array([current_weights.get(asset, 0) 
                                       for asset in returns_df.columns])
                
                day_return = (returns_df.loc[date].values @ weight_array)
                
                # Apply transaction costs on rebalance days
                if date in rebalance_schedule:
                    transaction_cost = self.backtest_config['transaction_costs']['commission']
                    slippage = self.backtest_config['transaction_costs']['slippage']
                    total_cost = transaction_cost + slippage
                    
                    day_return -= total_cost
                
                # Update portfolio value
                portfolio_value *= (1 + day_return)
            
            portfolio_values.append({
                'date': date,
                'portfolio_value': portfolio_value
            })
        
        # Create results DataFrame
        results_df = pd.DataFrame(portfolio_values)
        results_df['strategy'] = strategy_name
        results_df['returns'] = results_df['portfolio_value'].pct_change()
        
        logger.info(f"  Final value: ${portfolio_value:,.2f} "
                   f"(Return: {(portfolio_value/self.initial_capital - 1)*100:.2f}%)")
        
        return results_df
    
    def _get_rebalance_schedule(self, date_index: pd.DatetimeIndex) -> List[datetime]:
        """
        Get rebalancing dates based on frequency
        
        Args:
            date_index: Full date index
            
        Returns:
            List of rebalancing dates
        """
        if self.rebalance_freq == 'monthly':
            # First day of each month
            rebalance_dates = date_index[date_index.is_month_start]
        elif self.rebalance_freq == 'quarterly':
            # First day of each quarter
            rebalance_dates = date_index[date_index.is_quarter_start]
        else:
            # Default to monthly
            rebalance_dates = date_index[date_index.is_month_start]
        
        # Always include first date
        if date_index[0] not in rebalance_dates:
            rebalance_dates = rebalance_dates.insert(0, date_index[0])
        
        return list(rebalance_dates)
    
    def backtest_all_strategies(self, df_with_jumps: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Backtest all strategies (jump-adjusted + benchmarks)
        
        Args:
            df_with_jumps: DataFrame with returns and jumps
            
        Returns:
            Dictionary of strategy results
        """
        logger.info("Running backtests for all strategies...")
        
        strategies = [
            'jump_adjusted',
            'standard_minvar',
            'equal_weight',
            'btc_eth_6040'
        ]
        
        results = {}
        
        for strategy in strategies:
            try:
                strategy_results = self.backtest_strategy(df_with_jumps, strategy)
                results[strategy] = strategy_results
            except Exception as e:
                logger.error(f"Failed to backtest {strategy}: {e}")
        
        return results
    
    def combine_results(self, results: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Combine all strategy results into single DataFrame
        
        Args:
            results: Dictionary of strategy results
            
        Returns:
            Combined DataFrame
        """
        combined = pd.concat([df for df in results.values()], ignore_index=True)
        return combined
    
    def calculate_drawdowns(self, results_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate drawdowns for each strategy
        
        Args:
            results_df: Results DataFrame
            
        Returns:
            DataFrame with drawdown information
        """
        drawdown_results = []
        
        for strategy in results_df['strategy'].unique():
            strategy_data = results_df[results_df['strategy'] == strategy].copy()
            
            # Calculate cumulative max
            strategy_data['cummax'] = strategy_data['portfolio_value'].cummax()
            
            # Calculate drawdown
            strategy_data['drawdown'] = (
                strategy_data['portfolio_value'] / strategy_data['cummax'] - 1
            )
            
            # Find max drawdown
            max_drawdown = strategy_data['drawdown'].min()
            max_dd_date = strategy_data[strategy_data['drawdown'] == max_drawdown]['date'].iloc[0]
            
            drawdown_results.append({
                'strategy': strategy,
                'max_drawdown': max_drawdown,
                'max_dd_date': max_dd_date
            })
        
        return pd.DataFrame(drawdown_results)


def run_backtest(df_with_jumps: pd.DataFrame, config: Dict) -> Dict:
    """
    Convenience function to run complete backtest
    
    Args:
        df_with_jumps: DataFrame with jump indicators
        config: Configuration dictionary
        
    Returns:
        Dictionary with all backtest results
    """
    backtester = JumpRiskBacktester(config)
    
    # Run all strategies
    results = backtester.backtest_all_strategies(df_with_jumps)
    
    # Combine results
    combined_results = backtester.combine_results(results)
    
    # Calculate drawdowns
    drawdowns = backtester.calculate_drawdowns(combined_results)
    
    return {
        'individual_results': results,
        'combined_results': combined_results,
        'drawdowns': drawdowns
    }


if __name__ == '__main__':
    import yaml
    from pathlib import Path
    from data_loader import load_and_prepare_data
    from jump_detector import detect_and_analyze_jumps
    
    # Load config
    config_path = Path(__file__).parent / 'config.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Load data and detect jumps
    data_splits = load_and_prepare_data(config)
    test_df = data_splits['test']  # Use test set
    df_with_jumps, metrics, cojump_df = detect_and_analyze_jumps(test_df, config)
    
    # Run backtest
    backtest_results = run_backtest(df_with_jumps, config)
    
    print("\n=== Backtest Complete ===")
    for strategy, result_df in backtest_results['individual_results'].items():
        final_value = result_df['portfolio_value'].iloc[-1]
        total_return = (final_value / 100000 - 1) * 100
        print(f"{strategy}: ${final_value:,.2f} ({total_return:+.2f}%)")
    
    print("\nMax Drawdowns:")
    print(backtest_results['drawdowns'])
