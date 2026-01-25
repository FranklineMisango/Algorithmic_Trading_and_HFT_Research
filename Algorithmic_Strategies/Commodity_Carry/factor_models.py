"""
Commodity Carry Strategy - Factor Models Module
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import yaml
import yfinance as yf


class CommodityFactorModel:
    """Neutralize commodity risk factors"""
    
    def __init__(self, config_path: str = 'config.yaml'):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.neutralize_factors = self.config['signals']['neutralize_factors']
    
    def fetch_factors(self, start_date, end_date) -> pd.DataFrame:
        """Fetch dollar, equity, and energy factors"""
        factors = pd.DataFrame()
        
        if 'dollar_index' in self.neutralize_factors:
            dxy = yf.download('DX-Y.NYB', start=start_date, end=end_date, progress=False)['Adj Close']
            factors['dollar_index'] = dxy.pct_change()
        
        if 'equity_market' in self.neutralize_factors:
            spy = yf.download('SPY', start=start_date, end=end_date, progress=False)['Adj Close']
            factors['equity_market'] = spy.pct_change()
        
        if 'energy_sector' in self.neutralize_factors:
            xle = yf.download('XLE', start=start_date, end=end_date, progress=False)['Adj Close']
            factors['energy_sector'] = xle.pct_change()
        
        return factors.fillna(0)
    
    def neutralize_returns(self, returns_df: pd.DataFrame, factors: pd.DataFrame) -> pd.DataFrame:
        """Neutralize factor exposures"""
        common_dates = returns_df.index.intersection(factors.index)
        returns_aligned = returns_df.loc[common_dates]
        factors_aligned = factors.loc[common_dates]
        
        neutralized_returns = pd.DataFrame(index=returns_aligned.index, columns=returns_aligned.columns)
        
        for col in returns_aligned.columns:
            y = returns_aligned[col].dropna().values.reshape(-1, 1)
            X = factors_aligned.loc[returns_aligned[col].dropna().index].values
            
            if len(y) > 50:
                model = LinearRegression()
                model.fit(X, y)
                predictions = model.predict(factors_aligned.values)
                neutralized_returns[col] = returns_aligned[col].values - predictions.flatten()
            else:
                neutralized_returns[col] = returns_aligned[col]
        
        return neutralized_returns
