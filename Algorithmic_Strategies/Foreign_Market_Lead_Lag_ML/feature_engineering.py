"""
Feature engineering module for Foreign Market Lead-Lag ML Strategy.
Creates lagged weekly return features with cross-sectional standardization.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import logging
from scipy import stats

logger = logging.getLogger(__name__)


class FeatureEngineering:
    """Creates predictive features from foreign market returns."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.lags = config['features']['lags']
        self.standardize = config['features']['standardize']
        self.winsorize = config['features']['winsorize']
        self.winsorize_limits = config['features']['winsorize_limits']
        
    def create_lagged_features(self, weekly_returns: pd.DataFrame) -> pd.DataFrame:
        """
        Create lagged return features for each foreign market.
        
        Args:
            weekly_returns: DataFrame of weekly returns (markets as columns)
            
        Returns:
            DataFrame with lagged features (188 columns for 47 markets * 4 lags)
        """
        logger.info(f"Creating lagged features with lags: {self.lags}")
        
        lagged_features = pd.DataFrame(index=weekly_returns.index)
        
        for market in weekly_returns.columns:
            for lag in self.lags:
                feature_name = f"{market}_lag{lag}"
                lagged_features[feature_name] = weekly_returns[market].shift(lag)
        
        logger.info(f"Created {len(lagged_features.columns)} lagged features")
        return lagged_features
    
    def standardize_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """
        Cross-sectional standardization of features.
        Prevents models from biasing toward high-volatility markets.
        """
        if not self.standardize:
            return features
        
        logger.info("Applying cross-sectional standardization...")
        
        # Standardize each row (cross-sectional)
        standardized = features.apply(
            lambda row: (row - row.mean()) / row.std() if row.std() > 0 else row,
            axis=1
        )
        
        return standardized
    
    def winsorize_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Winsorize extreme values to reduce outlier impact."""
        if not self.winsorize:
            return features
        
        logger.info(f"Winsorizing features at {self.winsorize_limits}...")
        
        winsorized = features.apply(
            lambda col: pd.Series(
                stats.mstats.winsorize(col.dropna(), limits=self.winsorize_limits),
                index=col.dropna().index
            )
        )
        
        return winsorized
    
    def align_features_with_targets(self, features: pd.DataFrame, 
                                     daily_returns: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Align weekly features with daily target returns.
        Each weekly feature value is used for all days in that week.
        """
        logger.info("Aligning weekly features with daily targets...")
        
        # Forward fill weekly features to daily frequency
        features_daily = features.reindex(daily_returns.index, method='ffill')
        
        # Remove rows with missing features
        valid_idx = features_daily.dropna().index
        features_aligned = features_daily.loc[valid_idx]
        targets_aligned = daily_returns.loc[valid_idx]
        
        logger.info(f"Aligned data: {len(features_aligned)} days")
        return features_aligned, targets_aligned
    
    def create_target_variable(self, daily_returns: pd.DataFrame) -> pd.DataFrame:
        """
        Create next-day return targets for each stock.
        
        Args:
            daily_returns: DataFrame of daily returns (stocks as columns)
            
        Returns:
            DataFrame of next-day returns (shifted forward by 1 day)
        """
        logger.info("Creating next-day return targets...")
        
        # Shift returns forward by 1 day (today's features predict tomorrow's returns)
        targets = daily_returns.shift(-1)
        
        return targets
    
    def prepare_training_data(self, foreign_weekly_returns: pd.DataFrame,
                             sp500_daily_returns: pd.DataFrame,
                             stock_ticker: str) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare complete training dataset for a single stock.
        
        Args:
            foreign_weekly_returns: Weekly returns from foreign markets
            sp500_daily_returns: Daily returns for S&P 500 stocks
            stock_ticker: Target stock ticker
            
        Returns:
            Tuple of (features, target) aligned and ready for modeling
        """
        # Create lagged features
        lagged_features = self.create_lagged_features(foreign_weekly_returns)
        
        # Winsorize
        lagged_features = self.winsorize_features(lagged_features)
        
        # Standardize
        lagged_features = self.standardize_features(lagged_features)
        
        # Create target variable
        targets = self.create_target_variable(sp500_daily_returns)
        
        # Align features with targets
        features_aligned, targets_aligned = self.align_features_with_targets(
            lagged_features, targets
        )
        
        # Extract target stock
        if stock_ticker not in targets_aligned.columns:
            logger.warning(f"Stock {stock_ticker} not found in data")
            return pd.DataFrame(), pd.Series()
        
        target_stock = targets_aligned[stock_ticker]
        
        # Remove rows with missing targets
        valid_idx = target_stock.dropna().index
        X = features_aligned.loc[valid_idx]
        y = target_stock.loc[valid_idx]
        
        logger.info(f"Prepared {len(X)} samples for {stock_ticker}")
        return X, y
    
    def prepare_all_stocks(self, foreign_weekly_returns: pd.DataFrame,
                          sp500_daily_returns: pd.DataFrame) -> Dict[str, Tuple[pd.DataFrame, pd.Series]]:
        """
        Prepare training data for all S&P 500 stocks.
        
        Returns:
            Dictionary mapping stock ticker to (features, target) tuple
        """
        logger.info("Preparing data for all stocks...")
        
        # Create lagged features once
        lagged_features = self.create_lagged_features(foreign_weekly_returns)
        lagged_features = self.winsorize_features(lagged_features)
        lagged_features = self.standardize_features(lagged_features)
        
        # Create targets
        targets = self.create_target_variable(sp500_daily_returns)
        
        # Align
        features_aligned, targets_aligned = self.align_features_with_targets(
            lagged_features, targets
        )
        
        # Prepare data for each stock
        stock_data = {}
        for stock in targets_aligned.columns:
            target_stock = targets_aligned[stock]
            valid_idx = target_stock.dropna().index
            
            if len(valid_idx) > 0:
                X = features_aligned.loc[valid_idx]
                y = target_stock.loc[valid_idx]
                stock_data[stock] = (X, y)
        
        logger.info(f"Prepared data for {len(stock_data)} stocks")
        return stock_data


if __name__ == "__main__":
    import yaml
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Load config
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Load data
    sp500_returns = pd.read_csv('data/sp500_daily_returns.csv', index_col=0, parse_dates=True)
    foreign_returns = pd.read_csv('data/foreign_weekly_returns.csv', index_col=0, parse_dates=True)
    
    # Create features
    feature_eng = FeatureEngineering(config)
    
    # Test with single stock
    test_stock = sp500_returns.columns[0]
    X, y = feature_eng.prepare_training_data(foreign_returns, sp500_returns, test_stock)
    
    print(f"\nFeature Engineering Summary for {test_stock}:")
    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    print(f"\nFeature columns (first 10): {X.columns[:10].tolist()}")
