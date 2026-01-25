"""
Rates Carry Strategy - Portfolio Construction Module
"""

import pandas as pd
import numpy as np
import yaml


class RatesPortfolioConstructor:
    """Construct rates carry portfolio with duration limits"""
    
    def __init__(self, config_path: str = 'config.yaml'):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.vol_lookback = self.config['portfolio']['vol_lookback']
        self.target_vol = self.config['strategy']['target_volatility']
        self.max_duration = self.config['portfolio']['max_duration']
        
    def inverse_volatility_weights(self, returns_df: pd.DataFrame, signals_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate inverse vol weights with duration constraint"""
        print("\n" + "="*60)
        print("PORTFOLIO CONSTRUCTION")
        print("="*60)
        
        vol = returns_df.rolling(window=self.vol_lookback, min_periods=20).std() * np.sqrt(252)
        inv_vol = 1 / vol
        inv_vol = inv_vol.replace([np.inf, -np.inf], np.nan)
        
        weights = pd.DataFrame(0.0, index=signals_df.index, columns=signals_df.columns)
        
        for date in signals_df.index:
            long_mask = signals_df.loc[date] == 1
            short_mask = signals_df.loc[date] == -1
            
            if long_mask.sum() > 0:
                long_scores = inv_vol.loc[date, long_mask].fillna(0)
                if long_scores.sum() > 0:
                    weights.loc[date, long_mask] = long_scores / long_scores.sum()
            
            if short_mask.sum() > 0:
                short_scores = inv_vol.loc[date, short_mask].fillna(0)
                if short_scores.sum() > 0:
                    weights.loc[date, short_mask] = -short_scores / short_scores.sum()
        
        print(f"Average positions: {(weights != 0).sum(axis=1).mean():.1f}")
        return weights
    
    def construct_portfolio(self, returns_df: pd.DataFrame, signals_df: pd.DataFrame):
        """Complete portfolio construction"""
        weights = self.inverse_volatility_weights(returns_df, signals_df)
        portfolio_returns = (weights.shift(1) * returns_df).sum(axis=1)
        
        # Vol scaling
        realized_vol = portfolio_returns.rolling(63).std() * np.sqrt(252)
        scale = (self.target_vol / realized_vol.fillna(self.target_vol)).clip(upper=3.0)
        scaled_returns = portfolio_returns * scale.shift(1)
        
        print(f"\nScaled volatility: {scaled_returns.std() * np.sqrt(252):.2%}")
        print(f"Sharpe ratio: {scaled_returns.mean() / scaled_returns.std() * np.sqrt(252):.2f}")
        print("✓ Portfolio construction complete")
        
        return weights, scaled_returns
