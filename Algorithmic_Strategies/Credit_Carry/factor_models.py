"""
Credit Carry Strategy - Factor Models Module
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import yaml
import yfinance as yf


class CreditFactorModel:
    """Neutralize credit risk factors"""
    
    def __init__(self, config_path: str = 'config.yaml'):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.neutralize_factors = self.config['signals']['neutralize_factors']
    
    def fetch_factors(self, start_date, end_date) -> pd.DataFrame:
        """Fetch equity and rates factors"""
        factors = pd.DataFrame()
        
        if 'equity_market' in self.neutralize_factors:
            spy = yf.download('SPY', start=start_date, end=end_date, progress=False)['Adj Close']
            factors['equity_market'] = spy.pct_change()
        
        if 'high_yield_beta' in self.neutralize_factors:
            hyg = yf.download('HYG', start=start_date, end=end_date, progress=False)['Adj Close']
            factors['high_yield_beta'] = hyg.pct_change()
        
        if 'rates' in self.neutralize_factors:
            tlt = yf.download('TLT', start=start_date, end=end_date, progress=False)['Adj Close']
            factors['rates'] = tlt.pct_change()
        
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
