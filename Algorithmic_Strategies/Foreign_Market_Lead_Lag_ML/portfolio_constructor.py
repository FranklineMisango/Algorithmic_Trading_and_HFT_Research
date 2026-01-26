"""
Portfolio construction module for Foreign Market Lead-Lag ML Strategy.
Implements daily long/short portfolio based on predicted returns.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)


class PortfolioConstructor:
    """Constructs long/short portfolio based on ML predictions."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.long_percentile = config['portfolio']['long_percentile']
        self.short_percentile = config['portfolio']['short_percentile']
        self.equal_weight = config['portfolio']['equal_weight']
        self.dollar_neutral = config['portfolio']['dollar_neutral']
        self.max_position_size = config['portfolio']['max_position_size']
        
    def rank_stocks(self, predictions: pd.Series) -> pd.Series:
        """Rank stocks by predicted returns."""
        return predictions.rank(pct=True) * 100
    
    def select_long_short(self, predictions: pd.Series) -> Tuple[List[str], List[str]]:
        """
        Select long and short positions based on percentile thresholds.
        
        Args:
            predictions: Series of predicted returns indexed by stock ticker
            
        Returns:
            Tuple of (long_tickers, short_tickers)
        """
        # Rank stocks
        ranks = self.rank_stocks(predictions)
        
        # Select top and bottom percentiles
        long_tickers = ranks[ranks >= self.long_percentile].index.tolist()
        short_tickers = ranks[ranks <= self.short_percentile].index.tolist()
        
        logger.debug(f"Selected {len(long_tickers)} long, {len(short_tickers)} short positions")
        
        return long_tickers, short_tickers
    
    def calculate_position_weights(self, long_tickers: List[str], 
                                   short_tickers: List[str]) -> pd.Series:
        """
        Calculate position weights for long/short portfolio.
        
        Args:
            long_tickers: List of tickers for long positions
            short_tickers: List of tickers for short positions
            
        Returns:
            Series of position weights (positive for long, negative for short)
        """
        weights = pd.Series(dtype=float)
        
        if self.equal_weight:
            # Equal weight within each leg
            if len(long_tickers) > 0:
                long_weight = 1.0 / len(long_tickers)
                for ticker in long_tickers:
                    weights[ticker] = min(long_weight, self.max_position_size)
            
            if len(short_tickers) > 0:
                short_weight = -1.0 / len(short_tickers)
                for ticker in short_tickers:
                    weights[ticker] = max(short_weight, -self.max_position_size)
        
        # Normalize to dollar-neutral if required
        if self.dollar_neutral:
            long_sum = weights[weights > 0].sum()
            short_sum = abs(weights[weights < 0].sum())
            
            if long_sum > 0 and short_sum > 0:
                # Scale to make dollar-neutral
                scale_factor = min(long_sum, short_sum)
                weights[weights > 0] = weights[weights > 0] / long_sum * scale_factor
                weights[weights < 0] = weights[weights < 0] / short_sum * scale_factor
        
        return weights
    
    def construct_portfolio(self, predictions: pd.Series) -> pd.Series:
        """
        Construct complete portfolio from predictions.
        
        Args:
            predictions: Series of predicted returns indexed by stock ticker
            
        Returns:
            Series of position weights
        """
        # Select long and short positions
        long_tickers, short_tickers = self.select_long_short(predictions)
        
        # Calculate weights
        weights = self.calculate_position_weights(long_tickers, short_tickers)
        
        return weights
    
    def calculate_portfolio_return(self, weights: pd.Series, 
                                   returns: pd.Series) -> float:
        """
        Calculate portfolio return for a single period.
        
        Args:
            weights: Position weights
            returns: Realized returns for the period
            
        Returns:
            Portfolio return
        """
        # Align weights and returns
        common_tickers = weights.index.intersection(returns.index)
        
        if len(common_tickers) == 0:
            return 0.0
        
        portfolio_return = (weights[common_tickers] * returns[common_tickers]).sum()
        
        return portfolio_return
    
    def calculate_turnover(self, old_weights: pd.Series, 
                          new_weights: pd.Series) -> float:
        """
        Calculate portfolio turnover.
        
        Args:
            old_weights: Previous period weights
            new_weights: Current period weights
            
        Returns:
            Turnover (sum of absolute weight changes)
        """
        # Align indices
        all_tickers = old_weights.index.union(new_weights.index)
        old_weights_aligned = old_weights.reindex(all_tickers, fill_value=0)
        new_weights_aligned = new_weights.reindex(all_tickers, fill_value=0)
        
        # Calculate turnover
        turnover = abs(new_weights_aligned - old_weights_aligned).sum()
        
        return turnover
    
    def apply_transaction_costs(self, portfolio_return: float, 
                                turnover: float) -> float:
        """
        Apply transaction costs to portfolio return.
        
        Args:
            portfolio_return: Gross portfolio return
            turnover: Portfolio turnover
            
        Returns:
            Net portfolio return after costs
        """
        commission = self.config['costs']['commission']
        slippage = self.config['costs']['slippage']
        
        total_cost = (commission + slippage) * turnover
        net_return = portfolio_return - total_cost
        
        return net_return


class PortfolioSimulator:
    """Simulates portfolio performance over time."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.constructor = PortfolioConstructor(config)
        self.initial_capital = config['backtesting']['initial_capital']
        
    def simulate(self, predictions_df: pd.DataFrame, 
                returns_df: pd.DataFrame) -> pd.DataFrame:
        """
        Simulate portfolio performance.
        
        Args:
            predictions_df: DataFrame of predictions (dates x stocks)
            returns_df: DataFrame of realized returns (dates x stocks)
            
        Returns:
            DataFrame with portfolio metrics over time
        """
        logger.info("Starting portfolio simulation...")
        
        results = []
        portfolio_value = self.initial_capital
        previous_weights = pd.Series(dtype=float)
        
        # Align dates
        common_dates = predictions_df.index.intersection(returns_df.index)
        
        for date in common_dates:
            # Get predictions for this date
            predictions = predictions_df.loc[date].dropna()
            
            if len(predictions) == 0:
                continue
            
            # Construct portfolio
            weights = self.constructor.construct_portfolio(predictions)
            
            # Calculate turnover
            turnover = self.constructor.calculate_turnover(previous_weights, weights)
            
            # Get realized returns (next day)
            next_date_idx = returns_df.index.get_loc(date) + 1
            if next_date_idx >= len(returns_df):
                break
            
            next_date = returns_df.index[next_date_idx]
            realized_returns = returns_df.loc[next_date]
            
            # Calculate portfolio return
            gross_return = self.constructor.calculate_portfolio_return(weights, realized_returns)
            net_return = self.constructor.apply_transaction_costs(gross_return, turnover)
            
            # Update portfolio value
            portfolio_value *= (1 + net_return)
            
            # Store results
            results.append({
                'date': next_date,
                'gross_return': gross_return,
                'net_return': net_return,
                'turnover': turnover,
                'portfolio_value': portfolio_value,
                'num_long': len(weights[weights > 0]),
                'num_short': len(weights[weights < 0])
            })
            
            previous_weights = weights
        
        results_df = pd.DataFrame(results).set_index('date')
        
        logger.info(f"Simulation complete: {len(results_df)} periods")
        
        return results_df


if __name__ == "__main__":
    import yaml
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Load config
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Test portfolio construction
    constructor = PortfolioConstructor(config)
    
    # Create dummy predictions
    np.random.seed(42)
    tickers = [f'STOCK{i}' for i in range(100)]
    predictions = pd.Series(np.random.randn(100), index=tickers)
    
    # Construct portfolio
    weights = constructor.construct_portfolio(predictions)
    
    print("\nPortfolio Construction Test:")
    print(f"Total positions: {len(weights)}")
    print(f"Long positions: {len(weights[weights > 0])}")
    print(f"Short positions: {len(weights[weights < 0])}")
    print(f"Long exposure: {weights[weights > 0].sum():.4f}")
    print(f"Short exposure: {weights[weights < 0].sum():.4f}")
    print(f"Net exposure: {weights.sum():.4f}")
