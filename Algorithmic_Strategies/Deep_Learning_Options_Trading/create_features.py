"""
Create Features from Raw Databento Options Data

This script processes raw options data and underlying prices into engineered
features suitable for LSTM training.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_raw_data():
    """Load raw data files"""
    logger.info("Loading raw data...")
    
    # Load underlying prices
    prices = pd.read_csv('data/underlying_prices/underlying_prices.csv')
    prices['date'] = pd.to_datetime(prices['date'])
    logger.info(f"Loaded {len(prices)} underlying price records")
    
    # Load options
    options = pd.read_csv('data/options_data/options_data.csv')
    options['date'] = pd.to_datetime(options['date'])
    
    # Remove timezone if present (to match underlying prices)
    if hasattr(options['date'].dtype, 'tz') and options['date'].dtype.tz is not None:
        options['date'] = options['date'].dt.tz_localize(None)
    
    logger.info(f"Loaded {len(options)} option records")
    
    return prices, options


def create_features(prices_df, options_df):
    """Create engineered features from raw data"""
    logger.info("Engineering features...")
    
    # Merge options with spot prices
    merged = options_df.merge(
        prices_df[['date', 'ticker', 'close', 'return_1d', 'volatility_30d']],
        on=['date', 'ticker'],
        how='left',
        suffixes=('_option', '_underlying')
    )
    
    logger.info(f"Merged data: {len(merged)} records")
    
    # Create features
    features = pd.DataFrame()
    features['date'] = merged['date']
    features['ticker'] = merged['ticker']
    
    # Option price features (using close as proxy for premium)
    features['option_price'] = merged['close_option']
    features['option_volume'] = merged['volume']
    
    # Normalized option premium (option price / underlying price)
    features['option_premium_normalized'] = merged['close_option'] / merged['close_underlying']
    
    # Underlying features
    features['underlying_price'] = merged['close_underlying']
    features['underlying_return_1d'] = merged['return_1d']
    features['underlying_volatility_30d'] = merged['volatility_30d']
    
    # Add option symbol info if available
    if 'symbol' in merged.columns:
        features['option_symbol'] = merged['symbol']
    
    # Calculate additional rolling features
    logger.info("Calculating rolling features...")
    
    for ticker in features['ticker'].unique():
        mask = features['ticker'] == ticker
        ticker_data = features[mask].copy()
        
        # 5-day rolling stats for option prices
        features.loc[mask, 'option_price_ma5'] = ticker_data['option_price'].rolling(5, min_periods=1).mean()
        features.loc[mask, 'option_price_std5'] = ticker_data['option_price'].rolling(5, min_periods=1).std()
        
        # Volume rolling stats
        features.loc[mask, 'volume_ma5'] = ticker_data['option_volume'].rolling(5, min_periods=1).mean()
    
    # Fill NaN values
    features = features.fillna(method='ffill').fillna(method='bfill')
    
    logger.info(f"Created {len(features)} feature records with {features.shape[1]} columns")
    
    return features


def create_sequences(features_df, lookback_window=30):
    """Create sequential data for LSTM (rolling windows)"""
    logger.info(f"Creating sequences with {lookback_window}-day lookback...")
    
    # Feature columns (exclude date, ticker, identifiers)
    exclude_cols = ['date', 'ticker', 'option_symbol']
    feature_cols = [col for col in features_df.columns if col not in exclude_cols]
    
    sequences = []
    targets = []
    metadata = []
    
    # Sort by ticker and date
    features_df = features_df.sort_values(['ticker', 'date'])
    
    for ticker in features_df['ticker'].unique():
        ticker_data = features_df[features_df['ticker'] == ticker].copy()
        
        if len(ticker_data) < lookback_window + 1:
            logger.warning(f"Skipping {ticker}: only {len(ticker_data)} records (need {lookback_window + 1})")
            continue
        
        # Create rolling windows
        for i in range(lookback_window, len(ticker_data)):
            # Sequence: past lookback_window days
            seq = ticker_data.iloc[i-lookback_window:i][feature_cols].values
            
            # Target: next day's option price
            target = ticker_data.iloc[i]['option_price']
            
            sequences.append(seq)
            targets.append(target)
            metadata.append({
                'ticker': ticker,
                'date': ticker_data.iloc[i]['date'],
                'underlying_price': ticker_data.iloc[i]['underlying_price']
            })
    
    X = np.array(sequences)
    y = np.array(targets)
    
    logger.info(f"Created {len(X)} sequences with shape {X.shape}")
    
    return X, y, metadata


def save_processed_data(features_df, X, y, metadata):
    """Save processed features and sequences"""
    logger.info("Saving processed data...")
    
    # Create directories
    Path('data/processed').mkdir(parents=True, exist_ok=True)
    
    # Save features
    features_file = 'data/processed/features.csv'
    features_df.to_csv(features_file, index=False)
    logger.info(f"Saved features to {features_file}")
    
    # Save sequences for LSTM
    np.save('data/processed/X_sequences.npy', X)
    np.save('data/processed/y_targets.npy', y)
    
    # Save metadata
    meta_df = pd.DataFrame(metadata)
    meta_df.to_csv('data/processed/metadata.csv', index=False)
    
    logger.info("Saved sequences and metadata")
    
    return features_file


def main():
    """Run feature engineering pipeline"""
    print("="*70)
    print("Feature Engineering Pipeline")
    print("="*70)
    
    try:
        # Load raw data
        prices, options = load_raw_data()
        
        # Create features
        features = create_features(prices, options)
        
        # Create sequences for LSTM
        X, y, metadata = create_sequences(features, lookback_window=30)
        
        # Save everything
        features_file = save_processed_data(features, X, y, metadata)
        
        print("\n" + "="*70)
        print("SUCCESS")
        print("="*70)
        print(f"Features saved: {features_file}")
        print(f"  - Feature records: {len(features)}")
        print(f"  - Feature columns: {features.shape[1]}")
        print(f"  - LSTM sequences: {len(X)}")
        print(f"  - Sequence shape: {X.shape}")
        print("\nYou can now run notebook 02_feature_analysis.ipynb")
        print("="*70)
        
        return 0
        
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit(main())
