"""
Feature Engineering Module for Deep Learning Options Trading Strategy

Creates the parsimonious feature set for LSTM model training:
- Moneyness
- Time to expiration
- Option premium normalized by underlying
- Additional market features
"""

import pandas as pd
import numpy as np
from datetime import datetime
import logging
from pathlib import Path
import yaml

class OptionsFeatureEngineer:
    """
    Engineered features for LSTM-based options trading model.
    Focuses on parsimonious feature set as described in the research.
    """

    def __init__(self, config_path: str = "config.yaml"):
        """Initialize with configuration."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.logger = logging.getLogger(__name__)
        self._setup_logging()

    def _setup_logging(self):
        """Setup logging configuration."""
        logging.basicConfig(
            level=getattr(logging, self.config['logging']['level']),
            format=self.config['logging']['format'],
            handlers=[
                logging.FileHandler(self.config['logging']['file']),
                logging.StreamHandler()
            ]
        )

    def calculate_moneyness(self, options_df: pd.DataFrame) -> pd.Series:
        """
        Calculate moneyness ratio: strike_price / underlying_price

        Args:
            options_df: DataFrame with options data

        Returns:
            Series with moneyness values
        """
        return options_df['strike'] / options_df['spot_price']

    def calculate_time_to_expiry(self, options_df: pd.DataFrame) -> pd.Series:
        """
        Calculate time to expiration in years.

        Args:
            options_df: DataFrame with options data

        Returns:
            Series with time to expiry in years
        """
        time_diff = (options_df['expiry'] - options_df['date']).dt.days
        return time_diff / 365.0

    def normalize_option_premium(self, options_df: pd.DataFrame) -> pd.Series:
        """
        Normalize straddle premium by underlying stock price.

        Args:
            options_df: DataFrame with options data

        Returns:
            Series with normalized premium
        """
        return options_df['straddle_price'] / options_df['spot_price']

    def calculate_underlying_volatility(self, prices_df: pd.DataFrame,
                                      window: int = 30) -> pd.Series:
        """
        Calculate rolling volatility of underlying returns.

        Args:
            prices_df: DataFrame with price data
            window: Rolling window in days

        Returns:
            Series with volatility values
        """
        returns = prices_df.groupby('ticker')['return_1d']
        volatility = returns.rolling(window=window).std() * np.sqrt(252)  # Annualized
        return volatility

    def create_feature_matrix(self, options_df: pd.DataFrame,
                            prices_df: pd.DataFrame) -> pd.DataFrame:
        """
        Create the complete feature matrix for model training.

        Args:
            options_df: Options data
            prices_df: Underlying price data

        Returns:
            DataFrame with engineered features
        """
        self.logger.info("Creating feature matrix")

        # Basic options features
        features_df = options_df.copy()

        features_df['moneyness'] = self.calculate_moneyness(features_df)
        features_df['time_to_expiry'] = self.calculate_time_to_expiry(features_df)
        features_df['premium_normalized'] = self.normalize_option_premium(features_df)

        # Merge with underlying data
        features_df = features_df.merge(
            prices_df.reset_index()[['ticker', 'Date', 'return_1d', 'Adj Close']],
            left_on=['ticker', 'date'],
            right_on=['ticker', 'Date'],
            how='left'
        ).drop('Date', axis=1)

        # Calculate underlying volatility
        features_df['underlying_volatility_30d'] = (
            features_df.groupby('ticker')['return_1d']
            .rolling(window=30).std().reset_index(0, drop=True) * np.sqrt(252)
        )

        # Additional features
        features_df['implied_volatility'] = options_df['implied_vol']

        # Handle missing values
        features_df = self._handle_missing_values(features_df)

        # Select final feature set
        feature_columns = self.config['features']['feature_list']
        available_features = [col for col in feature_columns if col in features_df.columns]

        if len(available_features) != len(feature_columns):
            missing = set(feature_columns) - set(available_features)
            self.logger.warning(f"Missing features: {missing}")

        final_features = features_df[['date', 'ticker', 'straddle_price'] + available_features].copy()

        self.logger.info(f"Created feature matrix with {len(final_features)} rows and {len(available_features)} features")
        return final_features

    def create_sequential_data(self, features_df: pd.DataFrame,
                             lookback_window: int = None) -> tuple:
        """
        Create sequential data for LSTM input.

        Args:
            features_df: Feature matrix
            lookback_window: Number of historical days to include

        Returns:
            Tuple of (X, y) for model training
        """
        lookback = lookback_window or self.config['features']['lookback_window']
        self.logger.info(f"Creating sequential data with {lookback}-day lookback")

        # Sort by ticker and date
        features_df = features_df.sort_values(['ticker', 'date'])

        feature_cols = [col for col in features_df.columns
                       if col not in ['date', 'ticker', 'straddle_price']]

        X_sequences = []
        y_values = []
        metadata = []

        for ticker in features_df['ticker'].unique():
            ticker_data = features_df[features_df['ticker'] == ticker].copy()

            if len(ticker_data) < lookback + 1:
                continue

            # Create rolling windows
            for i in range(lookback, len(ticker_data)):
                X_seq = ticker_data.iloc[i-lookback:i][feature_cols].values
                y_val = ticker_data.iloc[i]['straddle_price']

                X_sequences.append(X_seq)
                y_values.append(y_val)
                metadata.append({
                    'ticker': ticker,
                    'date': ticker_data.iloc[i]['date'],
                    'straddle_price': y_val
                })

        X = np.array(X_sequences)
        y = np.array(y_values)

        self.logger.info(f"Created {len(X)} sequences with shape {X.shape}")
        return X, y, metadata

    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in feature matrix."""
        # Forward fill within ticker groups
        df = df.groupby('ticker').fillna(method='ffill')

        # Fill remaining NaNs with median values
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

        return df

    def save_features(self, features_df: pd.DataFrame, filepath: str = "data/features/features.csv"):
        """Save engineered features to disk."""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        features_df.to_csv(filepath, index=False)
        self.logger.info(f"Features saved to {filepath}")


if __name__ == "__main__":
    # Example usage
    engineer = OptionsFeatureEngineer()

    # Load sample data (would normally come from data_acquisition.py)
    try:
        options_df = pd.read_csv("data/options_data/options_data.csv")
        prices_df = pd.read_csv("data/underlying_prices/underlying_prices.csv")
        prices_df['Date'] = pd.to_datetime(prices_df['Date'])
        prices_df = prices_df.set_index(['ticker', 'Date'])

        features = engineer.create_feature_matrix(options_df, prices_df)
        X, y, metadata = engineer.create_sequential_data(features)

        print(f"Created {len(X)} training sequences")

    except FileNotFoundError:
        print("Data files not found. Run data_acquisition.py first.")