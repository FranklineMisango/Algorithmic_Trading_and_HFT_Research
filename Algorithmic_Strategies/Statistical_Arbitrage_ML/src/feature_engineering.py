"""
Feature Engineering Module for Statistical Arbitrage Strategy

Calculates momentum, mean reversion, and volume features as described in the strategy.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional
from loguru import logger


class FeatureEngineer:
    """
    Generates features for the ML model based on price and volume data.
    
    Features include:
    - Momentum: Rate of change over various periods
    - Mean Reversion: Distance from moving averages
    - Volume: Relative volume indicators
    """
    
    def __init__(
        self,
        momentum_periods: Optional[List[int]] = None,
        ma_periods: Optional[List[int]] = None,
        volume_period: int = 126  # ~6 months of trading days
    ):
        """
        Initialize feature engineer.
        
        Args:
            momentum_periods: List of lookback periods for momentum (days)
            ma_periods: List of periods for moving averages (days)
            volume_period: Period for volume comparison (days)
        """
        # Default momentum periods: short, medium, long-term (up to 1 year)
        self.momentum_periods = momentum_periods or [5, 10, 20, 60, 126, 252]
        
        # Default MA periods: various timeframes up to 1 year
        self.ma_periods = ma_periods or [10, 20, 50, 100, 200, 252]
        
        self.volume_period = volume_period
        
        logger.info(
            f"FeatureEngineer initialized with "
            f"{len(self.momentum_periods)} momentum periods, "
            f"{len(self.ma_periods)} MA periods"
        )
    
    def calculate_log_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate log returns for each ticker.
        
        Args:
            df: DataFrame with 'Close' prices, multi-indexed by (date, ticker)
            
        Returns:
            DataFrame with log returns column added
        """
        df = df.copy()
        
        # Calculate log returns for each ticker separately
        df['log_return'] = df.groupby('ticker')['Close'].transform(
            lambda x: np.log(x / x.shift(1))
        )
        
        return df
    
    def calculate_forward_returns(
        self,
        df: pd.DataFrame,
        horizons: List[int] = [2, 3]
    ) -> pd.DataFrame:
        """
        Calculate forward-looking returns (targets for ML model).
        
        Args:
            df: DataFrame with price data
            horizons: List of forward-looking periods (days)
            
        Returns:
            DataFrame with forward return columns
        """
        df = df.copy()
        
        for horizon in horizons:
            col_name = f'forward_return_{horizon}d'
            
            # Calculate forward returns for each ticker
            df[col_name] = df.groupby('ticker')['Close'].transform(
                lambda x: np.log(x.shift(-horizon) / x)
            )
        
        return df
    
    def calculate_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate momentum features (rate of change).
        
        Args:
            df: DataFrame with 'Close' prices
            
        Returns:
            DataFrame with momentum feature columns added
        """
        df = df.copy()
        
        for period in self.momentum_periods:
            col_name = f'momentum_{period}d'
            
            # Rate of change: log(price_t / price_{t-period})
            df[col_name] = df.groupby('ticker')['Close'].transform(
                lambda x: np.log(x / x.shift(period))
            )
        
        logger.debug(f"Calculated {len(self.momentum_periods)} momentum features")
        return df
    
    def calculate_mean_reversion_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate mean reversion features (distance from moving average).
        
        Args:
            df: DataFrame with 'Close' prices
            
        Returns:
            DataFrame with mean reversion feature columns added
        """
        df = df.copy()
        
        for period in self.ma_periods:
            ma_col = f'ma_{period}d'
            distance_col = f'distance_from_ma_{period}d'
            
            # Calculate moving average
            df[ma_col] = df.groupby('ticker')['Close'].transform(
                lambda x: x.rolling(window=period, min_periods=int(period * 0.5)).mean()
            )
            
            # Calculate distance from MA as percentage
            df[distance_col] = (df['Close'] - df[ma_col]) / df[ma_col]
            
            # Drop intermediate MA column
            df = df.drop(ma_col, axis=1)
        
        logger.debug(f"Calculated {len(self.ma_periods)} mean reversion features")
        return df
    
    def calculate_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate volume-based features.
        
        Args:
            df: DataFrame with 'Volume' data
            
        Returns:
            DataFrame with volume feature columns added
        """
        df = df.copy()
        
        # Average volume over period
        df['avg_volume'] = df.groupby('ticker')['Volume'].transform(
            lambda x: x.rolling(window=self.volume_period, min_periods=int(self.volume_period * 0.5)).mean()
        )
        
        # Relative volume (current volume / average volume)
        df['relative_volume'] = df['Volume'] / df['avg_volume']
        
        # Volume momentum (change in volume over shorter period)
        df['volume_momentum_20d'] = df.groupby('ticker')['Volume'].transform(
            lambda x: x.rolling(window=20).mean() / x.rolling(window=60).mean()
        )
        
        # Drop intermediate avg_volume column
        df = df.drop('avg_volume', axis=1)
        
        logger.debug("Calculated volume features")
        return df
    
    def calculate_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate volatility-based features.
        
        Args:
            df: DataFrame with price data
            
        Returns:
            DataFrame with volatility feature columns added
        """
        df = df.copy()
        
        # Calculate log returns if not already present
        if 'log_return' not in df.columns:
            df = self.calculate_log_returns(df)
        
        # Historical volatility (std of returns)
        for period in [20, 60]:
            col_name = f'volatility_{period}d'
            df[col_name] = df.groupby('ticker')['log_return'].transform(
                lambda x: x.rolling(window=period, min_periods=int(period * 0.5)).std() * np.sqrt(252)
            )
        
        # High-Low range (proxy for intraday volatility)
        df['hl_range'] = (df['High'] - df['Low']) / df['Close']
        df['hl_range_ma20'] = df.groupby('ticker')['hl_range'].transform(
            lambda x: x.rolling(window=20).mean()
        )
        
        logger.debug("Calculated volatility features")
        return df
    
    def calculate_all_features(
        self,
        df: pd.DataFrame,
        target_horizons: List[int] = [3]
    ) -> pd.DataFrame:
        """
        Calculate all features for the strategy.
        
        Args:
            df: Raw OHLCV dataframe
            target_horizons: Forward return periods to calculate (targets)
            
        Returns:
            DataFrame with all features and targets
        """
        logger.info("Starting feature calculation...")
        
        # Ensure we have required columns
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Calculate features step by step
        df = self.calculate_log_returns(df)
        df = self.calculate_momentum_features(df)
        df = self.calculate_mean_reversion_features(df)
        df = self.calculate_volume_features(df)
        df = self.calculate_volatility_features(df)
        
        # Calculate target variables (forward returns)
        df = self.calculate_forward_returns(df, horizons=target_horizons)
        
        logger.info(f"Feature calculation complete. Total features: {len(df.columns)}")
        
        return df
    
    def get_feature_columns(self) -> List[str]:
        """
        Get list of feature column names (excluding targets and OHLCV).
        
        Returns:
            List of feature column names
        """
        feature_cols = []
        
        # Momentum features
        feature_cols.extend([f'momentum_{p}d' for p in self.momentum_periods])
        
        # Mean reversion features
        feature_cols.extend([f'distance_from_ma_{p}d' for p in self.ma_periods])
        
        # Volume features
        feature_cols.extend(['relative_volume', 'volume_momentum_20d'])
        
        # Volatility features
        feature_cols.extend(['volatility_20d', 'volatility_60d', 'hl_range', 'hl_range_ma20'])
        
        return feature_cols
    
    def prepare_ml_dataset(
        self,
        df: pd.DataFrame,
        target_col: str = 'forward_return_3d',
        drop_na: bool = True
    ) -> tuple[pd.DataFrame, pd.Series, list]:
        """
        Prepare final dataset for ML model training.
        
        Args:
            df: DataFrame with all features calculated
            target_col: Name of target variable column
            drop_na: Whether to drop rows with NaN values
            
        Returns:
            Tuple of (X_features, y_target, feature_names)
        """
        feature_cols = self.get_feature_columns()
        
        # Ensure all feature columns exist
        missing_features = [col for col in feature_cols if col not in df.columns]
        if missing_features:
            raise ValueError(f"Missing feature columns: {missing_features}")
        
        # Extract features and target
        X = df[feature_cols].copy()
        y = df[target_col].copy()
        
        if drop_na:
            # Create combined dataframe to ensure aligned dropping
            combined = pd.concat([X, y], axis=1)
            combined = combined.dropna()
            
            X = combined[feature_cols]
            y = combined[target_col]
            
            logger.info(f"Dropped NaN values. Remaining samples: {len(X)}")
        
        logger.info(
            f"ML dataset prepared: {len(X)} samples, "
            f"{len(feature_cols)} features, "
            f"target: {target_col}"
        )
        
        return X, y, feature_cols
    
    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add additional technical indicators (optional enhancement).
        
        Args:
            df: DataFrame with price data
            
        Returns:
            DataFrame with technical indicators added
        """
        df = df.copy()
        
        # RSI (Relative Strength Index)
        def calculate_rsi(prices, period=14):
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi
        
        df['rsi_14'] = df.groupby('ticker')['Close'].transform(
            lambda x: calculate_rsi(x, period=14)
        )
        
        # MACD
        df['ema_12'] = df.groupby('ticker')['Close'].transform(
            lambda x: x.ewm(span=12, adjust=False).mean()
        )
        df['ema_26'] = df.groupby('ticker')['Close'].transform(
            lambda x: x.ewm(span=26, adjust=False).mean()
        )
        df['macd'] = df['ema_12'] - df['ema_26']
        df['macd_signal'] = df.groupby('ticker')['macd'].transform(
            lambda x: x.ewm(span=9, adjust=False).mean()
        )
        df['macd_diff'] = df['macd'] - df['macd_signal']
        
        # Clean up intermediate columns
        df = df.drop(['ema_12', 'ema_26'], axis=1)
        
        logger.debug("Calculated additional technical indicators")
        return df


if __name__ == "__main__":
    # Example usage
    import sys
    sys.path.append('..')
    from data_acquisition import DataAcquisitionEngine
    from datetime import datetime, timedelta
    
    # Get sample data
    engine = DataAcquisitionEngine()
    universe = engine.get_russell_3000_universe()[:10]  # Sample 10 tickers
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    
    df = engine.get_training_data(universe, start_date, end_date, apply_filters=False)
    
    # Calculate features
    feature_engineer = FeatureEngineer()
    df_features = feature_engineer.calculate_all_features(df, target_horizons=[3])
    
    # Prepare ML dataset
    X, y = feature_engineer.prepare_ml_dataset(df_features, target_col='forward_return_3d')
    
    print(f"\nFeature matrix shape: {X.shape}")
    print(f"Target vector shape: {y.shape}")
    print(f"\nFeature columns ({len(X.columns)}):")
    print(X.columns.tolist())
    print(f"\nSample features:\n{X.head()}")
    print(f"\nTarget statistics:\n{y.describe()}")
