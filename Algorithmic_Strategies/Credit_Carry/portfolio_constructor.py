"""
Credit Carry Strategy - Portfolio Construction Module
"""

import pandas as pd
import numpy as np
import yaml


class CreditPortfolioConstructor:
    """Construct credit carry portfolio"""
    
    def __init__(self, config_path: str = 'config.yaml'):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.vol_lookback = self.config['portfolio']['vol_lookback']
        self.target_vol = self.config['strategy']['target_volatility']
    
    def construct_portfolio(self, returns_df: pd.DataFrame, signals_df: pd.DataFrame):
        """Build portfolio with inverse vol weighting"""
        vol = returns_df.rolling(window=self.vol_lookback).std() * np.sqrt(252)
        inv_vol = 1 / vol.replace([np.inf, -np.inf], np.nan)
        
        weights = pd.DataFrame(0.0, index=signals_df.index, columns=signals_df.columns)
        
        for date in signals_df.index:
            active = signals_df.loc[date] != 0
            if active.sum() > 0:
                scores = inv_vol.loc[date, active] * signals_df.loc[date, active]
                scores = scores.fillna(0)
                if scores.abs().sum() > 0:
                    weights.loc[date, active] = scores / scores.abs().sum()
        
        portfolio_returns = (weights.shift(1) * returns_df).sum(axis=1)
        
        # Vol scaling
        realized_vol = portfolio_returns.rolling(63).std() * np.sqrt(252)
        scale = (self.target_vol / realized_vol.fillna(self.target_vol)).clip(upper=3.0)
        scaled_returns = portfolio_returns * scale.shift(1)
        
        return weights, scaled_returns
