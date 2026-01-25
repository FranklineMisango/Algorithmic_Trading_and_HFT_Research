"""
FX Carry Strategy - Factor Models Module

Implements factor neutralization for FX carry:
1. Dollar index neutralization
2. Safe-haven factor neutralization
3. Commodity FX neutralization
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import yaml
from typing import Dict, Tuple


class FXFactorModel:
    """Neutralize systematic FX risk factors"""
    
    def __init__(self, config_path: str = 'config.yaml'):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.neutralize_factors = self.config['signals']['neutralize_factors']
        
    def calculate_factor_returns(self, factors: Dict[str, pd.Series]) -> pd.DataFrame:
        """
        Calculate daily returns for each factor
        
        Args:
            factors: Dictionary of factor level series
            
        Returns:
            DataFrame of factor returns
        """
        factor_returns = {}
        
        for name, series in factors.items():
            if name in self.neutralize_factors:
                factor_returns[name] = series.pct_change()
        
        return pd.DataFrame(factor_returns)
    
    def neutralize_returns(self, returns_df: pd.DataFrame, 
                          factor_returns: pd.DataFrame) -> pd.DataFrame:
        """
        Remove factor exposures from strategy returns via regression
        
        Args:
            returns_df: Strategy returns for each pair
            factor_returns: Factor returns
            
        Returns:
            Factor-neutralized returns
        """
        print("\n" + "="*60)
        print("FACTOR NEUTRALIZATION")
        print("="*60)
        
        # Align dates
        common_dates = returns_df.index.intersection(factor_returns.index)
        returns_aligned = returns_df.loc[common_dates]
        factors_aligned = factor_returns.loc[common_dates]
        
        # Remove NaNs
        factors_aligned = factors_aligned.fillna(0)
        
        print(f"Neutralizing against {len(self.neutralize_factors)} factors:")
        for factor in self.neutralize_factors:
            print(f"  - {factor}")
        
        # Regress each currency pair's returns on factors
        neutralized_returns = pd.DataFrame(
            index=returns_aligned.index,
            columns=returns_aligned.columns
        )
        
        factor_betas = {}
        
        for pair in returns_aligned.columns:
            pair_returns = returns_aligned[pair].values.reshape(-1, 1)
            
            # Remove NaNs
            valid_idx = ~np.isnan(pair_returns.flatten())
            y = pair_returns[valid_idx]
            X = factors_aligned.loc[valid_idx].values
            
            if len(y) > 50:  # Need minimum observations
                # Fit regression
                model = LinearRegression()
                model.fit(X, y)
                
                # Calculate residuals (factor-neutral returns)
                predictions = model.predict(factors_aligned.values)
                residuals = returns_aligned[pair].values.flatten() - predictions.flatten()
                
                neutralized_returns[pair] = residuals
                
                # Store betas for analysis
                factor_betas[pair] = dict(zip(factors_aligned.columns, model.coef_[0]))
            else:
                # Not enough data, use raw returns
                neutralized_returns[pair] = returns_aligned[pair]
        
        # Report average factor exposures
        print("\nAverage factor betas (before neutralization):")
        avg_betas = pd.DataFrame(factor_betas).T.mean()
        for factor, beta in avg_betas.items():
            print(f"  {factor}: {beta:.3f}")
        
        # Check that neutralization worked
        post_betas = {}
        for pair in neutralized_returns.columns:
            y = neutralized_returns[pair].dropna().values.reshape(-1, 1)
            X = factors_aligned.loc[neutralized_returns[pair].dropna().index].values
            
            if len(y) > 50:
                model = LinearRegression()
                model.fit(X, y)
                post_betas[pair] = dict(zip(factors_aligned.columns, model.coef_[0]))
        
        print("\nAverage factor betas (after neutralization):")
        avg_post_betas = pd.DataFrame(post_betas).T.mean()
        for factor, beta in avg_post_betas.items():
            print(f"  {factor}: {beta:.3f}")
        
        print("\n✓ Factor neutralization complete")
        
        return neutralized_returns
    
    def portfolio_factor_exposure(self, weights: pd.DataFrame, 
                                  pair_betas: Dict) -> pd.DataFrame:
        """
        Calculate portfolio-level factor exposures over time
        
        Args:
            weights: Portfolio weights for each pair
            pair_betas: Dictionary of factor betas for each pair
            
        Returns:
            Time series of portfolio factor exposures
        """
        portfolio_betas = pd.DataFrame(
            index=weights.index,
            columns=self.neutralize_factors
        )
        
        for date in weights.index:
            for factor in self.neutralize_factors:
                # Weighted average of pair betas
                factor_exposure = sum(
                    weights.loc[date, pair] * pair_betas.get(pair, {}).get(factor, 0)
                    for pair in weights.columns
                )
                portfolio_betas.loc[date, factor] = factor_exposure
        
        return portfolio_betas


if __name__ == "__main__":
    # Example usage
    from data_acquisition import FXDataAcquisition
    from signal_generator import CarrySignalGenerator
    
    # Load data
    fx_data = FXDataAcquisition()
    spots, rates, carry, factors = fx_data.load_data()
    
    # Generate signals and returns
    signal_gen = CarrySignalGenerator()
    zscores, signals, returns = signal_gen.run_signal_generation(carry, spots)
    
    # Neutralize factors
    factor_model = FXFactorModel()
    factor_returns = factor_model.calculate_factor_returns(factors)
    neutral_returns = factor_model.neutralize_returns(returns, factor_returns)
    
    print(f"\nOriginal returns std: {returns.std().mean():.4f}")
    print(f"Neutralized returns std: {neutral_returns.std().mean():.4f}")
