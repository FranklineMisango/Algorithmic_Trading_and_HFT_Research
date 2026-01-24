"""
Data Processor for Deep Learning Momentum Strategy

Handles data acquisition, filtering, feature engineering, and cross-sectional standardization.
"""

import numpy as np
import pandas as pd
import yfinance as yf
from typing import List, Dict, Tuple, Optional
from datetime import datetime, timedelta
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


class DataProcessor:
    """
    Process stock data for deep learning momentum strategy.
    
    Key Features:
    - Filter stocks by minimum price
    - Engineer 33 momentum features
    - Apply cross-sectional z-score standardization
    - Generate binary classification labels
    """
    
    def __init__(self, config: Dict):
        """
        Initialize data processor with configuration.
        
        Parameters:
        -----------
        config : Dict
            Configuration dictionary with data parameters
        """
        self.config = config
        self.min_price = config['data']['min_price']
        self.start_date = config['data']['start_date']
        self.end_date = config['data']['end_date']
        
        # Feature configuration
        self.long_term_months = config['features']['long_term_lookback_months']
        self.short_term_days = config['features']['short_term_lookback_days']
        self.include_january = config['features']['include_january_effect']
        
    def download_stock_universe(
        self,
        tickers: List[str],
        progress: bool = True
    ) -> Dict[str, pd.DataFrame]:
        """
        Download stock data for given tickers.
        
        Parameters:
        -----------
        tickers : List[str]
            List of stock ticker symbols
        progress : bool
            Show progress bar
            
        Returns:
        --------
        Dict[str, pd.DataFrame] : Dictionary of ticker -> OHLCV DataFrame
        """
        data_dict = {}
        
        iterator = tqdm(tickers, desc="Downloading stock data") if progress else tickers
        
        for ticker in iterator:
            try:
                df = yf.download(
                    ticker,
                    start=self.start_date,
                    end=self.end_date,
                    progress=False
                )
                
                if len(df) > 0:
                    # Fix MultiIndex columns
                    if isinstance(df.columns, pd.MultiIndex):
                        df.columns = df.columns.get_level_values(0)
                    
                    data_dict[ticker] = df
                    
            except Exception as e:
                if progress:
                    print(f"  Failed to download {ticker}: {e}")
                continue
        
        return data_dict
    
    def filter_by_price(
        self,
        data_dict: Dict[str, pd.DataFrame]
    ) -> Dict[str, pd.DataFrame]:
        """
        Filter stocks that don't meet minimum price requirement.
        
        Parameters:
        -----------
        data_dict : Dict[str, pd.DataFrame]
            Dictionary of stock data
            
        Returns:
        --------
        Dict[str, pd.DataFrame] : Filtered data dictionary
        """
        filtered_dict = {}
        
        for ticker, df in data_dict.items():
            # Check if stock price is above minimum most of the time
            valid_days = (df['Close'] >= self.min_price).sum()
            total_days = len(df)
            
            if valid_days / total_days > 0.8:  # 80% of days above threshold
                filtered_dict[ticker] = df
        
        return filtered_dict
    
    def calculate_long_term_features(
        self,
        prices: pd.Series,
        months: List[int]
    ) -> pd.DataFrame:
        """
        Calculate long-term momentum features (monthly returns).
        
        Parameters:
        -----------
        prices : pd.Series
            Price series
        months : List[int]
            List of month lookbacks (e.g., [2, 3, ..., 13])
            
        Returns:
        --------
        pd.DataFrame : Long-term momentum features
        """
        features = pd.DataFrame(index=prices.index)
        
        for month in months:
            days = month * 21  # Approximate trading days per month
            feature_name = f'ret_m{month}'
            
            # Cumulative return looking back 'month' months
            features[feature_name] = prices.pct_change(periods=days)
        
        return features
    
    def calculate_short_term_features(
        self,
        prices: pd.Series,
        days: int = 20
    ) -> pd.DataFrame:
        """
        Calculate short-term momentum features (daily returns from recent month).
        
        Parameters:
        -----------
        prices : pd.Series
            Price series
        days : int
            Number of days to look back (default: 20 trading days)
            
        Returns:
        --------
        pd.DataFrame : Short-term momentum features
        """
        features = pd.DataFrame(index=prices.index)
        
        for day in range(1, days + 1):
            feature_name = f'ret_d{day}'
            
            # Daily return looking back 'day' days
            features[feature_name] = prices.pct_change(periods=day)
        
        return features
    
    def calculate_anomaly_features(
        self,
        index: pd.DatetimeIndex
    ) -> pd.DataFrame:
        """
        Calculate anomaly features (January Effect).
        
        Parameters:
        -----------
        index : pd.DatetimeIndex
            Date index
            
        Returns:
        --------
        pd.DataFrame : Anomaly features
        """
        features = pd.DataFrame(index=index)
        
        if self.include_january:
            # Binary dummy: 1 if next month is January, 0 otherwise
            features['january_dummy'] = (index.month == 12).astype(int)
        
        return features
    
    def engineer_features(
        self,
        data_dict: Dict[str, pd.DataFrame]
    ) -> pd.DataFrame:
        """
        Engineer all 33 features for all stocks.
        
        Parameters:
        -----------
        data_dict : Dict[str, pd.DataFrame]
            Dictionary of stock data
            
        Returns:
        --------
        pd.DataFrame : Feature matrix with MultiIndex (date, ticker)
        """
        all_features = []
        
        for ticker, df in tqdm(data_dict.items(), desc="Engineering features"):
            prices = df['Close']
            
            # Long-term features (12)
            long_term = self.calculate_long_term_features(prices, self.long_term_months)
            
            # Short-term features (20)
            short_term = self.calculate_short_term_features(prices, self.short_term_days)
            
            # Anomaly features (1)
            anomaly = self.calculate_anomaly_features(prices.index)
            
            # Combine all features
            features = pd.concat([long_term, short_term, anomaly], axis=1)
            features['ticker'] = ticker
            features['date'] = features.index
            
            all_features.append(features)
        
        # Combine all tickers
        feature_df = pd.concat(all_features, axis=0)
        feature_df = feature_df.set_index(['date', 'ticker'])
        
        return feature_df
    
    def apply_cross_sectional_zscore(
        self,
        feature_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Apply cross-sectional z-score standardization.
        
        CRITICAL: For each day, standardize each feature across all stocks.
        This puts everything on a level playing field.
        
        Parameters:
        -----------
        feature_df : pd.DataFrame
            Feature matrix with MultiIndex (date, ticker)
            
        Returns:
        --------
        pd.DataFrame : Z-score standardized features
        """
        print("Applying cross-sectional z-score standardization...")
        
        # Group by date and apply z-score normalization
        standardized = feature_df.groupby(level='date').apply(
            lambda x: (x - x.mean()) / (x.std() + 1e-8)
        )
        
        return standardized
    
    def generate_labels(
        self,
        data_dict: Dict[str, pd.DataFrame],
        holding_period: int = 21  # 1 month
    ) -> pd.Series:
        """
        Generate binary classification labels.
        
        Label 1: Stock return > median return next month
        Label 0: Stock return <= median return next month
        
        Parameters:
        -----------
        data_dict : Dict[str, pd.DataFrame]
            Dictionary of stock data
        holding_period : int
            Holding period in days (default: 21 for 1 month)
            
        Returns:
        --------
        pd.Series : Binary labels with MultiIndex (date, ticker)
        """
        all_returns = []
        
        for ticker, df in data_dict.items():
            # Calculate forward returns
            forward_ret = df['Close'].pct_change(periods=holding_period).shift(-holding_period)
            
            forward_ret_df = pd.DataFrame({
                'date': df.index,
                'ticker': ticker,
                'forward_return': forward_ret
            })
            
            all_returns.append(forward_ret_df)
        
        # Combine all returns
        returns_df = pd.concat(all_returns, axis=0)
        returns_df = returns_df.set_index(['date', 'ticker'])
        
        # For each date, calculate median and create binary label
        labels = returns_df.groupby(level='date')['forward_return'].transform(
            lambda x: (x > x.median()).astype(int)
        )
        
        return labels
    
    def prepare_dataset(
        self,
        tickers: List[str]
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Complete pipeline: download, filter, engineer features, and generate labels.
        
        Parameters:
        -----------
        tickers : List[str]
            List of stock tickers
            
        Returns:
        --------
        Tuple[pd.DataFrame, pd.Series] : (Features, Labels)
        """
        print("="*80)
        print("DATA PREPARATION PIPELINE")
        print("="*80)
        
        # Step 1: Download data
        print("\nStep 1: Downloading stock data...")
        data_dict = self.download_stock_universe(tickers)
        print(f"  Downloaded {len(data_dict)} stocks")
        
        # Step 2: Filter by price
        print(f"\nStep 2: Filtering stocks (price > ${self.min_price})...")
        data_dict = self.filter_by_price(data_dict)
        print(f"  Remaining: {len(data_dict)} stocks")
        
        # Step 3: Engineer features
        print("\nStep 3: Engineering 33 features...")
        features = self.engineer_features(data_dict)
        print(f"  Features shape: {features.shape}")
        
        # Step 4: Cross-sectional standardization
        print("\nStep 4: Applying cross-sectional z-score...")
        features = self.apply_cross_sectional_zscore(features)
        
        # Step 5: Generate labels
        print("\nStep 5: Generating binary labels...")
        labels = self.generate_labels(data_dict)
        print(f"  Labels shape: {labels.shape}")
        
        # Align features and labels
        common_idx = features.index.intersection(labels.index)
        features = features.loc[common_idx]
        labels = labels.loc[common_idx]
        
        # Remove NaN values
        valid_mask = ~(features.isna().any(axis=1) | labels.isna())
        features = features[valid_mask]
        labels = labels[valid_mask]
        
        print(f"\nFinal dataset shape: {features.shape}")
        print(f"Label distribution: {labels.value_counts().to_dict()}")
        print("\n" + "="*80)
        
        return features, labels


if __name__ == "__main__":
    import yaml
    
    # Load configuration
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Example usage
    processor = DataProcessor(config)
    
    # Test with a small set of tickers
    test_tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA']
    features, labels = processor.prepare_dataset(test_tickers)
    
    print(f"\nFeatures:\n{features.head()}")
    print(f"\nLabels:\n{labels.head()}")
