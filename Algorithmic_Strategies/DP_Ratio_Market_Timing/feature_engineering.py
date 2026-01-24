"""
Feature Engineering Module for Dividend-Price Ratio Strategy

This module implements the key signal:
    Signal_t = log(D/P_t) - log(D/P_t-1) = Δlog(D/P)

This is the change in the logarithm of the dividend-price ratio,
which the research paper found to be a statistically significant
predictor of next-month S&P 500 returns.
"""

import numpy as np
import pandas as pd
from typing import Tuple


class DPRatioFeatureEngineer:
    """
    Engineer features for the D/P ratio predictive model.
    """
    
    def __init__(self, config: dict):
        """
        Initialize feature engineer.
        
        Parameters
        ----------
        config : dict
            Configuration dictionary
        """
        self.config = config
        
    def calculate_log_dp_ratio(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate log(D/P ratio).
        
        Parameters
        ----------
        data : pd.DataFrame
            Data with dp_ratio column
            
        Returns
        -------
        pd.DataFrame
            Data with log_dp column
        """
        print("Calculating log(D/P)...")
        
        # Log transformation
        data['log_dp'] = np.log(data['dp_ratio'])
        
        # Check for infinities
        if np.isinf(data['log_dp']).any():
            print("Warning: Infinite values in log(D/P). Removing...")
            data = data[~np.isinf(data['log_dp'])]
        
        return data
    
    def calculate_delta_log_dp(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate the change in log(D/P) - THE KEY SIGNAL.
        
        Formula: Δlog(D/P)_t = log(D/P)_t - log(D/P)_t-1
        
        This is equivalent to: log(D/P_t / D/P_t-1)
        
        Parameters
        ----------
        data : pd.DataFrame
            Data with log_dp column
            
        Returns
        -------
        pd.DataFrame
            Data with delta_log_dp column (the trading signal)
        """
        print("Calculating Δlog(D/P) - THE TRADING SIGNAL...")
        
        # Calculate change in log D/P
        data['delta_log_dp'] = data['log_dp'].diff()
        
        print(f"  Mean: {data['delta_log_dp'].mean():.6f}")
        print(f"  Std: {data['delta_log_dp'].std():.6f}")
        print(f"  Min: {data['delta_log_dp'].min():.6f}")
        print(f"  Max: {data['delta_log_dp'].max():.6f}")
        
        return data
    
    def create_target_variable(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create target variable: next month's return.
        
        The model predicts Return_t using Signal_t-1 (lagged signal).
        
        Parameters
        ----------
        data : pd.DataFrame
            Data with monthly_return column
            
        Returns
        -------
        pd.DataFrame
            Data with next_month_return column
        """
        print("Creating target variable (next month's return)...")
        
        # Shift returns forward by 1 period
        # Row t has: delta_log_dp_t-1 (predictor) and return_t (target)
        data['next_month_return'] = data['monthly_return'].shift(-1)
        
        return data
    
    def create_lagged_signal(self, data: pd.DataFrame, lags: int = 1) -> pd.DataFrame:
        """
        Create lagged versions of the signal for robustness checks.
        
        Parameters
        ----------
        data : pd.DataFrame
            Data with delta_log_dp column
        lags : int
            Number of lags to create
            
        Returns
        -------
        pd.DataFrame
            Data with lagged signal columns
        """
        for lag in range(1, lags + 1):
            col_name = f'delta_log_dp_lag{lag}'
            data[col_name] = data['delta_log_dp'].shift(lag)
            print(f"Created {col_name}")
        
        return data
    
    def calculate_additional_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate additional features for robustness analysis.
        
        These are NOT part of the core single-factor model but useful
        for understanding what the model is missing.
        
        Parameters
        ----------
        data : pd.DataFrame
            Data with price and return information
            
        Returns
        -------
        pd.DataFrame
            Data with additional features
        """
        print("Calculating additional features (for analysis only)...")
        
        # 1. D/P ratio level (not just change)
        data['dp_ratio_zscore'] = (data['dp_ratio'] - data['dp_ratio'].mean()) / data['dp_ratio'].std()
        
        # 2. Price momentum (12-month return)
        data['momentum_12m'] = data['Close'].pct_change(12)
        
        # 3. Volatility (12-month rolling std of returns)
        data['volatility_12m'] = data['monthly_return'].rolling(12).std()
        
        # 4. Trend in D/P (12-month change)
        data['dp_trend_12m'] = data['dp_ratio'].pct_change(12)
        
        # 5. Earnings yield proxy (inverse P/E approximation)
        # Note: D/P is related to E/P via payout ratio
        # This is speculative without actual earnings data
        
        print("  Additional features created (not used in core model)")
        
        return data
    
    def prepare_model_dataset(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Prepare final dataset for modeling.
        
        Returns
        -------
        Tuple[pd.DataFrame, pd.DataFrame]
            (Features DataFrame, Target DataFrame)
        """
        print("\nPreparing model dataset...")
        
        # Core model uses only delta_log_dp as predictor
        predictor_col = self.config['model']['predictor_variable']
        target_col = self.config['model']['target_variable']
        
        # Remove rows with NaN in predictor or target
        model_data = data[[predictor_col, target_col]].copy()
        initial_len = len(model_data)
        model_data = model_data.dropna()
        final_len = len(model_data)
        
        if initial_len != final_len:
            print(f"  Removed {initial_len - final_len} rows with NaN")
        
        # Separate features and target
        X = model_data[[predictor_col]]
        y = model_data[target_col]
        
        print(f"  Final dataset: {len(X)} observations")
        print(f"  Predictor: {predictor_col}")
        print(f"  Target: {target_col}")
        
        return X, y
    
    def engineer_all_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Run all feature engineering steps.
        
        Parameters
        ----------
        data : pd.DataFrame
            Raw monthly data with dp_ratio and monthly_return
            
        Returns
        -------
        pd.DataFrame
            Data with all engineered features
        """
        print("="*60)
        print("FEATURE ENGINEERING")
        print("="*60)
        
        # 1. Calculate log(D/P)
        data = self.calculate_log_dp_ratio(data)
        
        # 2. Calculate Δlog(D/P) - THE KEY SIGNAL
        data = self.calculate_delta_log_dp(data)
        
        # 3. Create target variable (next month's return)
        data = self.create_target_variable(data)
        
        # 4. Create additional features (optional)
        data = self.calculate_additional_features(data)
        
        # Summary statistics
        print("\nSignal (Δlog D/P) Summary:")
        print(data['delta_log_dp'].describe())
        
        print("\nTarget (Next Month Return) Summary:")
        print(data['next_month_return'].describe())
        
        # Correlation analysis
        corr = data[['delta_log_dp', 'next_month_return']].corr().iloc[0, 1]
        print(f"\nCorrelation(Signal, Target): {corr:.4f}")
        
        print("="*60)
        
        return data


def visualize_signal(data: pd.DataFrame):
    """
    Visualize the trading signal and its relationship with returns.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: D/P Ratio over time
    ax1 = axes[0, 0]
    data['dp_ratio'].plot(ax=ax1, color='steelblue', linewidth=1)
    ax1.set_title('Dividend-Price Ratio Over Time', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('D/P Ratio')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Δlog(D/P) signal over time
    ax2 = axes[0, 1]
    data['delta_log_dp'].plot(ax=ax2, color='coral', linewidth=1)
    ax2.axhline(0, color='black', linestyle='--', alpha=0.5)
    ax2.set_title('Δlog(D/P) Trading Signal', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Δlog(D/P)')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Signal distribution
    ax3 = axes[1, 0]
    data['delta_log_dp'].hist(bins=50, ax=ax3, color='steelblue', alpha=0.7, edgecolor='black')
    ax3.axvline(0, color='red', linestyle='--', linewidth=2, label='Zero')
    ax3.set_title('Signal Distribution', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Δlog(D/P)')
    ax3.set_ylabel('Frequency')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Signal vs Next Month Return (scatter)
    ax4 = axes[1, 1]
    valid_data = data[['delta_log_dp', 'next_month_return']].dropna()
    ax4.scatter(valid_data['delta_log_dp'], valid_data['next_month_return'], 
                alpha=0.5, s=20, color='steelblue')
    
    # Add regression line
    from scipy.stats import linregress
    slope, intercept, r_value, p_value, std_err = linregress(
        valid_data['delta_log_dp'], valid_data['next_month_return']
    )
    x_line = np.array([valid_data['delta_log_dp'].min(), valid_data['delta_log_dp'].max()])
    y_line = intercept + slope * x_line
    ax4.plot(x_line, y_line, 'r-', linewidth=2, label=f'R²={r_value**2:.4f}')
    
    ax4.set_title('Signal vs Next Month Return', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Δlog(D/P) Signal')
    ax4.set_ylabel('Next Month Return')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/signal_analysis.png', dpi=150, bbox_inches='tight')
    print("Signal visualization saved to results/signal_analysis.png")
    plt.show()


def main():
    """
    Test feature engineering module.
    """
    import yaml
    from data_acquisition import DividendPriceDataFetcher
    
    # Load config
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Fetch data
    fetcher = DividendPriceDataFetcher(config)
    data = fetcher.fetch_and_prepare_data()
    
    # Engineer features
    engineer = DPRatioFeatureEngineer(config)
    data = engineer.engineer_all_features(data)
    
    # Prepare model dataset
    X, y = engineer.prepare_model_dataset(data)
    
    print("\nModel Dataset:")
    print(f"  X shape: {X.shape}")
    print(f"  y shape: {y.shape}")
    print(f"\nX (first 5):")
    print(X.head())
    print(f"\ny (first 5):")
    print(y.head())
    
    # Visualize
    visualize_signal(data)
    
    # Save engineered data
    data.to_csv('results/engineered_features.csv')
    print("\nEngineered features saved to results/engineered_features.csv")


if __name__ == "__main__":
    main()
