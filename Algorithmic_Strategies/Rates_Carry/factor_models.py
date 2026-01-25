"""
Rates Carry Strategy - Factor Models Module
Neutralize duration, curve slope, and flight-to-quality factors
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import yaml


class RatesFactorModel:
    """Neutralize systematic rates risk factors"""
    
    def __init__(self, config_path: str = 'config.yaml'):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.neutralize_factors = self.config['signals']['neutralize_factors']
    
    def construct_factors(self, yields_df: pd.DataFrame) -> pd.DataFrame:
        """Construct factor time series from yield curve"""
        print("\nConstructing rate factors...")
        
        factors = pd.DataFrame(index=yields_df.index)
        
        # Duration factor: parallel shift in yields (average yield change)
        if 'duration' in self.neutralize_factors:
            factors['duration'] = yields_df.mean(axis=1).diff()
        
        # Curve slope: 10Y - 2Y spread change
        if 'curve_slope' in self.neutralize_factors:
            if 'US_10Y' in yields_df.columns and 'US_2Y' in yields_df.columns:
                slope = yields_df['US_10Y'] - yields_df['US_2Y']
                factors['curve_slope'] = slope.diff()
        
        # Flight-to-quality: VIX or Treasury-Bund spread
        if 'flight_to_quality' in self.neutralize_factors:
            import yfinance as yf
            try:
                vix = yf.download('^VIX', start=yields_df.index[0], end=yields_df.index[-1], progress=False)['Adj Close']
                factors['flight_to_quality'] = vix.pct_change()
            except:
                factors['flight_to_quality'] = 0
        
        print(f"Constructed {len(factors.columns)} factors")
        return factors.fillna(0)
    
    def neutralize_returns(self, returns_df: pd.DataFrame, factors: pd.DataFrame) -> pd.DataFrame:
        """Remove factor exposures via regression"""
        print("\n" + "="*60)
        print("FACTOR NEUTRALIZATION")
        print("="*60)
        
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
                residuals = returns_aligned[col].values.flatten() - predictions.flatten()
                neutralized_returns[col] = residuals
            else:
                neutralized_returns[col] = returns_aligned[col]
        
        print("✓ Factor neutralization complete")
        return neutralized_returns


if __name__ == "__main__":
    from data_acquisition import RatesDataAcquisition
    from signal_generator import RatesSignalGenerator
    
    rates_data = RatesDataAcquisition()
    yields, rolldown = rates_data.load_data()
    
    signal_gen = RatesSignalGenerator()
    zscores = signal_gen.calculate_zscore(rolldown)
    signals = signal_gen.generate_signals(zscores)
    returns = signal_gen.calculate_returns(yields, signals)
    
    factor_model = RatesFactorModel()
    factors = factor_model.construct_factors(yields)
    neutral_returns = factor_model.neutralize_returns(returns, factors)
