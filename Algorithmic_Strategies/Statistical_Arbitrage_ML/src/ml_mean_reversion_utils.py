"""
ML Mean Reversion Strategy Utilities
Implements QPI calculation and feature engineering for Russell 3000 mean reversion
"""

import numpy as np
import pandas as pd


def calculate_qpi_3day(df):
    """
    Calculate 3-day Quantitative Pressure Index (QPI)
    
    QPI measures short-term oversold/overbought conditions
    Range: 0-100 (lower = more oversold, higher = more overbought)
    
    Args:
        df: DataFrame with OHLCV data
        
    Returns:
        DataFrame with qpi_3day column added
    """
    df = df.copy()
    
    # Price momentum components
    df['ret_1d'] = df['Close'].pct_change(1)
    df['ret_3d'] = df['Close'].pct_change(3)
    
    # Volume pressure
    df['vol_ratio'] = df['Volume'] / df['Volume'].rolling(20).mean()
    
    # Volatility-adjusted pressure
    df['volatility'] = df['ret_1d'].rolling(20).std()
    
    # QPI formula: normalized pressure index (0-100 scale)
    raw_qpi = 50 + (df['ret_3d'] / (df['volatility'] + 1e-6)) * 10 - (df['vol_ratio'] - 1) * 5
    df['qpi_3day'] = raw_qpi.clip(0, 100)
    
    return df


def calculate_rsi(series, period=14):
    """Calculate Relative Strength Index"""
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / (loss + 1e-6)
    return 100 - (100 / (1 + rs))


def create_ml_features(df):
    """
    Create ML features for mean reversion prediction
    
    Features:
    - qpi_3day: Quantitative Pressure Index
    - rsi_14: Relative Strength Index
    - bb_position: Bollinger Band position
    - volume_surge: Volume relative to recent average
    - mom_5, mom_10, mom_20: Momentum over different periods
    - vol_ratio: Volume ratio
    
    Args:
        df: DataFrame with OHLCV data
        
    Returns:
        DataFrame with feature columns added
    """
    df = calculate_qpi_3day(df)
    
    # Price features
    df['rsi_14'] = calculate_rsi(df['Close'], 14)
    sma_20 = df['Close'].rolling(20).mean()
    std_20 = df['Close'].rolling(20).std()
    df['bb_position'] = (df['Close'] - sma_20) / (std_20 + 1e-6)
    
    # Volume features
    df['volume_surge'] = df['Volume'] / df['Volume'].rolling(5).mean()
    
    # Momentum features
    for period in [5, 10, 20]:
        df[f'mom_{period}'] = df['Close'].pct_change(period)
    
    return df


def create_target(df, horizon=6):
    """
    Create binary target for ML training
    
    Args:
        df: DataFrame with price data
        horizon: Forward-looking period (default 6 days)
        
    Returns:
        DataFrame with target columns:
        - target_long: 1 if positive return in next horizon days
        - target_short: 1 if negative return in next horizon days
    """
    df['forward_return'] = df['Close'].pct_change(horizon).shift(-horizon)
    df['target_long'] = (df['forward_return'] > 0).astype(int)
    df['target_short'] = (df['forward_return'] < 0).astype(int)
    return df


def calculate_vix_regime(vix_series, window=15, threshold_multiplier=1.15):
    """
    Calculate VIX regime (bull/bear market signal)
    
    Bear market signal: VIX > SMA(15) * 1.15
    
    Args:
        vix_series: Series of VIX closing prices
        window: SMA window (default 15)
        threshold_multiplier: Multiplier for threshold (default 1.15)
        
    Returns:
        Series of boolean values (True = bear market)
    """
    vix_sma = vix_series.rolling(window).mean()
    threshold = vix_sma * threshold_multiplier
    return vix_series > threshold


def get_feature_columns():
    """Return list of feature columns used in ML model"""
    return [
        'qpi_3day',
        'rsi_14',
        'bb_position',
        'volume_surge',
        'mom_5',
        'mom_10',
        'mom_20',
        'vol_ratio'
    ]


def validate_signal(qpi, ml_probability, qpi_threshold=15, prob_threshold=0.60):
    """
    Validate entry signal
    
    Long signal: QPI < 15 AND ML_Probability > 0.60
    Short signal: QPI > 85 AND ML_Probability > 0.60
    
    Args:
        qpi: Current QPI value
        ml_probability: ML model probability
        qpi_threshold: QPI threshold (default 15)
        prob_threshold: Probability threshold (default 0.60)
        
    Returns:
        Tuple (is_long_signal, is_short_signal)
    """
    is_long = (qpi < qpi_threshold) and (ml_probability > prob_threshold)
    is_short = (qpi > (100 - qpi_threshold)) and (ml_probability > prob_threshold)
    return is_long, is_short


def calculate_position_size(capital, num_positions, allocation_multiplier):
    """
    Calculate position size
    
    Args:
        capital: Total capital
        num_positions: Number of positions
        allocation_multiplier: Allocation multiplier (e.g., 1.1 for 110% long)
        
    Returns:
        Position size per stock
    """
    if num_positions == 0:
        return 0
    return (capital * allocation_multiplier) / num_positions


def apply_liquidity_filter(df, min_price=1.0, min_dollar_volume=1000000):
    """
    Apply liquidity filters
    
    Args:
        df: DataFrame with price and volume data
        min_price: Minimum price (default $1.00)
        min_dollar_volume: Minimum average daily dollar volume
        
    Returns:
        Filtered DataFrame
    """
    df['dollar_volume'] = df['Close'] * df['Volume']
    avg_dollar_volume = df['dollar_volume'].rolling(20).mean()
    
    mask = (df['Close'] >= min_price) & (avg_dollar_volume >= min_dollar_volume)
    return df[mask]
