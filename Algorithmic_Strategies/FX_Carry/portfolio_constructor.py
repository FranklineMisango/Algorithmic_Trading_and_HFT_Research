"""
FX Carry Strategy - Portfolio Construction Module

Implements:
1. Inverse volatility weighting
2. Position size limits
3. Target volatility scaling
4. Weekly rebalancing
"""

import pandas as pd
import numpy as np
import yaml
from typing import Tuple


class PortfolioConstructor:
    """Construct carry portfolio with risk management"""
    
    def __init__(self, config_path: str = 'config.yaml'):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.weighting_scheme = self.config['portfolio']['weighting_scheme']
        self.vol_lookback = self.config['portfolio']['vol_lookback']
        self.target_vol = self.config['strategy']['target_volatility']
        self.max_position = self.config['portfolio']['max_position_size']
        self.rebalance_freq = self.config['strategy']['rebalance_frequency']
        
    def calculate_volatility(self, returns_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate rolling volatility for each pair
        
        Args:
            returns_df: Daily returns
            
        Returns:
            Rolling volatility estimates
        """
        vol = returns_df.rolling(window=self.vol_lookback, min_periods=20).std()
        vol = vol * np.sqrt(252)  # Annualize
        
        return vol
    
    def inverse_volatility_weights(self, returns_df: pd.DataFrame, 
                                   signals_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate inverse volatility weights for active positions
        
        Args:
            returns_df: Daily returns
            signals_df: Trading signals {-1, 0, 1}
            
        Returns:
            Portfolio weights (sum to 1 for long, -1 for short)
        """
        print("\n" + "="*60)
        print("PORTFOLIO CONSTRUCTION")
        print("="*60)
        print(f"Weighting scheme: {self.weighting_scheme}")
        print(f"Volatility lookback: {self.vol_lookback} days")
        
        # Calculate volatility
        vol = self.calculate_volatility(returns_df)
        
        # Inverse volatility scores (1/vol)
        inv_vol = 1 / vol
        inv_vol = inv_vol.replace([np.inf, -np.inf], np.nan)
        
        # Only weight active positions
        inv_vol_masked = inv_vol * signals_df
        
        # Normalize weights separately for longs and shorts
        weights = pd.DataFrame(0.0, index=signals_df.index, columns=signals_df.columns)
        
        for date in signals_df.index:
            long_mask = signals_df.loc[date] == 1
            short_mask = signals_df.loc[date] == -1
            
            # Long weights sum to 1
            if long_mask.sum() > 0:
                long_scores = inv_vol_masked.loc[date, long_mask]
                long_scores = long_scores.fillna(0)
                if long_scores.sum() > 0:
                    weights.loc[date, long_mask] = long_scores / long_scores.sum()
            
            # Short weights sum to -1
            if short_mask.sum() > 0:
                short_scores = inv_vol_masked.loc[date, short_mask]
                short_scores = short_scores.fillna(0)
                if short_scores.sum() > 0:
                    weights.loc[date, short_mask] = -short_scores / short_scores.sum()
        
        # Apply position size limits
        weights = weights.clip(lower=-self.max_position, upper=self.max_position)
        
        # Renormalize after clipping
        for date in weights.index:
            long_sum = weights.loc[date][weights.loc[date] > 0].sum()
            short_sum = weights.loc[date][weights.loc[date] < 0].sum()
            
            if long_sum > 0:
                weights.loc[date][weights.loc[date] > 0] /= long_sum
            if abs(short_sum) > 0:
                weights.loc[date][weights.loc[date] < 0] /= abs(short_sum)
        
        print(f"Average number of positions: {(weights != 0).sum(axis=1).mean():.1f}")
        print(f"Max position size: {weights.abs().max().max():.2%}")
        print(f"Average gross exposure: {weights.abs().sum(axis=1).mean():.2f}")
        
        return weights
    
    def apply_rebalancing(self, weights: pd.DataFrame) -> pd.DataFrame:
        """
        Apply rebalancing frequency (weekly)
        
        Args:
            weights: Daily portfolio weights
            
        Returns:
            Weights that only change on rebalance days
        """
        if self.rebalance_freq == 'daily':
            return weights
        
        print(f"\nApplying {self.rebalance_freq} rebalancing...")
        
        rebalanced_weights = weights.copy()
        
        # Find rebalance days (Fridays for weekly)
        if self.rebalance_freq == 'weekly':
            # 4 = Friday
            rebalance_days = rebalanced_weights.index[rebalanced_weights.index.dayofweek == 4]
        else:
            rebalance_days = rebalanced_weights.index
        
        # Forward fill weights from rebalance days
        last_weights = None
        for date in rebalanced_weights.index:
            if date in rebalance_days:
                last_weights = rebalanced_weights.loc[date]
            elif last_weights is not None:
                rebalanced_weights.loc[date] = last_weights
        
        print(f"Number of rebalances: {len(rebalance_days)}")
        
        return rebalanced_weights
    
    def scale_to_target_volatility(self, returns: pd.Series) -> pd.Series:
        """
        Scale portfolio returns to target volatility
        
        Args:
            returns: Portfolio returns
            
        Returns:
            Scaled returns
        """
        realized_vol = returns.rolling(window=63, min_periods=20).std() * np.sqrt(252)
        realized_vol = realized_vol.fillna(method='bfill').fillna(self.target_vol)
        
        # Scaling factor
        scale = self.target_vol / realized_vol
        
        # Limit scaling to avoid excessive leverage
        scale = scale.clip(upper=3.0)
        
        scaled_returns = returns * scale.shift(1)  # Lag by 1 day
        
        return scaled_returns
    
    def construct_portfolio(self, returns_df: pd.DataFrame, 
                           signals_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Complete portfolio construction pipeline
        
        Returns:
            weights_df, portfolio_returns
        """
        # Calculate weights
        weights = self.inverse_volatility_weights(returns_df, signals_df)
        
        # Apply rebalancing
        weights = self.apply_rebalancing(weights)
        
        # Calculate portfolio returns
        portfolio_returns = (weights.shift(1) * returns_df).sum(axis=1)
        
        # Scale to target volatility
        scaled_returns = self.scale_to_target_volatility(portfolio_returns)
        
        print(f"\nPortfolio statistics:")
        print(f"  Realized volatility: {portfolio_returns.std() * np.sqrt(252):.2%}")
        print(f"  Scaled volatility: {scaled_returns.std() * np.sqrt(252):.2%}")
        print(f"  Mean daily return: {scaled_returns.mean() * 252:.2%}")
        print(f"  Sharpe ratio: {scaled_returns.mean() / scaled_returns.std() * np.sqrt(252):.2f}")
        
        print("\n✓ Portfolio construction complete")
        
        return weights, scaled_returns


if __name__ == "__main__":
    # Example usage
    from data_acquisition import FXDataAcquisition
    from signal_generator import CarrySignalGenerator
    from factor_models import FXFactorModel
    
    # Load data
    fx_data = FXDataAcquisition()
    spots, rates, carry, factors = fx_data.load_data()
    
    # Generate signals
    signal_gen = CarrySignalGenerator()
    zscores, signals, returns = signal_gen.run_signal_generation(carry, spots)
    
    # Neutralize factors
    factor_model = FXFactorModel()
    factor_returns = factor_model.calculate_factor_returns(factors)
    neutral_returns = factor_model.neutralize_returns(returns, factor_returns)
    
    # Construct portfolio
    portfolio = PortfolioConstructor()
    weights, pf_returns = portfolio.construct_portfolio(neutral_returns, signals)
    
    # Plot cumulative returns
    import matplotlib.pyplot as plt
    (1 + pf_returns).cumprod().plot(figsize=(12, 6), title='FX Carry Portfolio')
    plt.ylabel('Cumulative Return')
    plt.show()
