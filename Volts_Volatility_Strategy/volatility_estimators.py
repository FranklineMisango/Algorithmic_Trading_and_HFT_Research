"""
Historical Volatility Estimators Module

Implements various historical volatility estimators for the Volts strategy:
- Parkinson (1980): Uses high-low range
- Garman-Klass (1980): Incorporates OHLC
- Rogers-Satchell (1991): Accounts for drift
- Yang-Zhang (2000): Robust estimator with opening jumps

All estimators return annualized volatility estimates.
"""

import numpy as np
import pandas as pd
from typing import Union, List


class VolatilityEstimator:
    """
    Calculate historical volatility using multiple estimators.
    """
    
    def __init__(self, annualization_factor: int = 252):
        """
        Initialize the volatility estimator.
        
        Parameters:
        -----------
        annualization_factor : int
            Number of trading periods in a year (default: 252 for daily data)
        """
        self.annualization_factor = annualization_factor
    
    def parkinson(self, df: pd.DataFrame) -> pd.Series:
        """
        Parkinson (1980) volatility estimator.
        Uses the high-low range. Simple but misses overnight gaps.
        
        Formula: σ² = (1/(4*ln(2))) * ln(High/Low)²
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame with 'High' and 'Low' columns
            
        Returns:
        --------
        pd.Series : Annualized volatility estimates
        """
        hl_ratio = np.log(df['High'] / df['Low'])
        variance = (1 / (4 * np.log(2))) * (hl_ratio ** 2)
        volatility = np.sqrt(variance * self.annualization_factor)
        return volatility
    
    def garman_klass(self, df: pd.DataFrame) -> pd.Series:
        """
        Garman-Klass (1980) volatility estimator.
        Extends Parkinson by incorporating open and close prices.
        Assumes log-normal distribution without drift.
        
        Formula: σ² = 0.5 * ln(High/Low)² - (2*ln(2)-1) * ln(Close/Open)²
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame with 'Open', 'High', 'Low', 'Close' columns
            
        Returns:
        --------
        pd.Series : Annualized volatility estimates
        """
        hl_ratio = np.log(df['High'] / df['Low'])
        co_ratio = np.log(df['Close'] / df['Open'])
        
        variance = 0.5 * (hl_ratio ** 2) - (2 * np.log(2) - 1) * (co_ratio ** 2)
        volatility = np.sqrt(variance * self.annualization_factor)
        return volatility
    
    def rogers_satchell(self, df: pd.DataFrame) -> pd.Series:
        """
        Rogers-Satchell (1991) volatility estimator.
        Better for assets with drift; uses OHLC.
        
        Formula: σ² = ln(High/Close) * ln(High/Open) + ln(Low/Close) * ln(Low/Open)
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame with 'Open', 'High', 'Low', 'Close' columns
            
        Returns:
        --------
        pd.Series : Annualized volatility estimates
        """
        hc_ratio = np.log(df['High'] / df['Close'])
        ho_ratio = np.log(df['High'] / df['Open'])
        lc_ratio = np.log(df['Low'] / df['Close'])
        lo_ratio = np.log(df['Low'] / df['Open'])
        
        variance = hc_ratio * ho_ratio + lc_ratio * lo_ratio
        volatility = np.sqrt(variance * self.annualization_factor)
        return volatility
    
    def yang_zhang(self, df: pd.DataFrame, window: int = 20) -> pd.Series:
        """
        Yang-Zhang (2000) volatility estimator.
        Robust estimator that accounts for opening jumps and drift.
        
        Formula: σ² = σ²_open + k*σ²_close + (1-k)*σ²_RS
        where k is a constant based on window size.
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame with 'Open', 'High', 'Low', 'Close' columns
        window : int
            Rolling window size for calculation
            
        Returns:
        --------
        pd.Series : Annualized volatility estimates
        """
        # Calculate log returns
        close = df['Close']
        open_price = df['Open']
        
        # Opening jump (overnight return)
        log_oc = np.log(open_price / close.shift(1))
        
        # Close-to-close return
        log_cc = np.log(close / close.shift(1))
        
        # Rogers-Satchell component
        rs = self.rogers_satchell(df) ** 2 / self.annualization_factor
        
        # Calculate k (constant based on window size)
        k = 0.34 / (1.34 + (window + 1) / (window - 1))
        
        # Variance components (rolling)
        var_open = log_oc.rolling(window).var()
        var_close = log_cc.rolling(window).var()
        var_rs = rs.rolling(window).mean()
        
        # Yang-Zhang variance
        variance = var_open + k * var_close + (1 - k) * var_rs
        volatility = np.sqrt(variance * self.annualization_factor)
        
        return volatility
    
    def calculate_all(
        self, 
        df: pd.DataFrame, 
        rolling_window: int = 20
    ) -> pd.DataFrame:
        """
        Calculate all volatility estimators for the given data.
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame with 'Open', 'High', 'Low', 'Close' columns
        rolling_window : int
            Window size for rolling calculations
            
        Returns:
        --------
        pd.DataFrame : DataFrame with all volatility estimates
        """
        result = pd.DataFrame(index=df.index)
        
        # Calculate each estimator
        result['parkinson'] = self.parkinson(df).rolling(rolling_window).mean()
        result['garman_klass'] = self.garman_klass(df).rolling(rolling_window).mean()
        result['rogers_satchell'] = self.rogers_satchell(df).rolling(rolling_window).mean()
        result['yang_zhang'] = self.yang_zhang(df, window=rolling_window)
        
        return result
    
    def get_estimator(
        self, 
        df: pd.DataFrame, 
        method: str = 'yang_zhang',
        rolling_window: int = 20
    ) -> pd.Series:
        """
        Get a specific volatility estimator.
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame with OHLC data
        method : str
            Estimator method ('parkinson', 'garman_klass', 'rogers_satchell', 'yang_zhang')
        rolling_window : int
            Window size for rolling calculations
            
        Returns:
        --------
        pd.Series : Volatility estimates
        """
        method_map = {
            'parkinson': lambda: self.parkinson(df).rolling(rolling_window).mean(),
            'garman_klass': lambda: self.garman_klass(df).rolling(rolling_window).mean(),
            'rogers_satchell': lambda: self.rogers_satchell(df).rolling(rolling_window).mean(),
            'yang_zhang': lambda: self.yang_zhang(df, window=rolling_window)
        }
        
        if method not in method_map:
            raise ValueError(f"Unknown method: {method}. Choose from {list(method_map.keys())}")
        
        return method_map[method]()


def calculate_volatility_for_assets(
    data_dict: dict,
    estimator: str = 'yang_zhang',
    rolling_window: int = 20,
    annualization_factor: int = 252
) -> pd.DataFrame:
    """
    Calculate volatility for multiple assets.
    
    Parameters:
    -----------
    data_dict : dict
        Dictionary of {ticker: DataFrame} with OHLC data
    estimator : str
        Volatility estimator to use
    rolling_window : int
        Rolling window for calculation
    annualization_factor : int
        Annualization factor
        
    Returns:
    --------
    pd.DataFrame : DataFrame with volatility for each asset (columns)
    """
    vol_estimator = VolatilityEstimator(annualization_factor=annualization_factor)
    
    volatilities = {}
    for ticker, df in data_dict.items():
        try:
            vol = vol_estimator.get_estimator(df, method=estimator, rolling_window=rolling_window)
            if isinstance(vol, pd.Series) and len(vol) > 0:
                volatilities[ticker] = vol
            else:
                print(f"Warning: Skipping {ticker} - no valid volatility data")
        except Exception as e:
            print(f"Warning: Error calculating volatility for {ticker}: {e}")
    
    if not volatilities:
        raise ValueError("No valid volatility data calculated for any asset")
    
    result = pd.DataFrame(volatilities)
    return result


if __name__ == "__main__":
    # Example usage
    import yfinance as yf
    
    # Download sample data
    ticker = "AAPL"
    df = yf.download(ticker, start="2023-01-01", end="2023-12-31", progress=False)
    
    # Initialize estimator
    estimator = VolatilityEstimator(annualization_factor=252)
    
    # Calculate all estimators
    all_vols = estimator.calculate_all(df, rolling_window=20)
    
    print(f"Volatility estimates for {ticker}:")
    print(all_vols.tail())
    
    # Compare estimators
    print(f"\nMean volatility by estimator:")
    print(all_vols.mean())
