#!/usr/bin/env python3
"""
ML Mean Reversion Strategy - Complete Pipeline
Trains models and prepares for LEAN backtesting
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
import joblib
from ml_mean_reversion_utils import (
    create_ml_features,
    create_target,
    get_feature_columns
)

def download_data(symbols, start_date, end_date):
    """Download historical data for training"""
    print(f"Downloading data for {len(symbols)} symbols...")
    all_data = []
    
    for i, symbol in enumerate(symbols):
        try:
            df = yf.download(symbol, start=start_date, end=end_date, progress=False)
            if not df.empty and len(df) > 30:
                df = create_ml_features(df)
                df = create_target(df, horizon=6)
                df['symbol'] = symbol
                all_data.append(df)
                print(f"  [{i+1}/{len(symbols)}] {symbol}: {len(df)} rows")
        except Exception as e:
            print(f"  [{i+1}/{len(symbols)}] {symbol}: FAILED - {e}")
    
    return pd.concat(all_data) if all_data else pd.DataFrame()

def train_models(df, feature_cols):
    """Train long and short models"""
    print("\nPreparing training data...")
    train_df = df.dropna(subset=feature_cols + ['target_long', 'target_short'])
    
    X = train_df[feature_cols]
    y_long = train_df['target_long']
    y_short = train_df['target_short']
    
    # 80/20 split
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_long_train, y_long_test = y_long[:split_idx], y_long[split_idx:]
    y_short_train, y_short_test = y_short[:split_idx], y_short[split_idx:]
    
    print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")
    
    # Train Long Model
    print("\nTraining Long Model...")
    model_long = GradientBoostingClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        random_state=42
    )
    model_long.fit(X_train, y_long_train)
    
    y_long_pred_proba = model_long.predict_proba(X_test)[:, 1]
    y_long_pred = (y_long_pred_proba > 0.6).astype(int)
    
    print(f"  Accuracy (>0.6 threshold): {accuracy_score(y_long_test, y_long_pred):.3f}")
    print(f"  AUC: {roc_auc_score(y_long_test, y_long_pred_proba):.3f}")
    
    # Train Short Model
    print("\nTraining Short Model...")
    model_short = GradientBoostingClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        random_state=42
    )
    model_short.fit(X_train, y_short_train)
    
    y_short_pred_proba = model_short.predict_proba(X_test)[:, 1]
    y_short_pred = (y_short_pred_proba > 0.6).astype(int)
    
    print(f"  Accuracy (>0.6 threshold): {accuracy_score(y_short_test, y_short_pred):.3f}")
    print(f"  AUC: {roc_auc_score(y_short_test, y_short_pred_proba):.3f}")
    
    return model_long, model_short

def save_models(model_long, model_short, feature_cols):
    """Save models for LEAN"""
    os.makedirs('models', exist_ok=True)
    
    joblib.dump(model_long, 'models/ml_mean_reversion_long.pkl')
    joblib.dump(model_short, 'models/ml_mean_reversion_short.pkl')
    joblib.dump(feature_cols, 'models/feature_columns.pkl')
    
    print("\n✓ Models saved to models/ directory")
    print("  - ml_mean_reversion_long.pkl")
    print("  - ml_mean_reversion_short.pkl")
    print("  - feature_columns.pkl")

def main():
    # Sample Russell 3000 stocks (liquid large/mid caps)
    symbols = [
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META',
        'JPM', 'BAC', 'WFC', 'C', 'GS',
        'WMT', 'TGT', 'COST', 'HD', 'LOW',
        'XOM', 'CVX', 'COP', 'SLB',
        'PFE', 'JNJ', 'UNH', 'ABBV',
        'DIS', 'NFLX', 'CMCSA',
        'BA', 'CAT', 'DE', 'GE'
    ]
    
    start_date = '2018-01-01'
    end_date = '2023-12-31'
    
    print("=" * 60)
    print("ML Mean Reversion Strategy - Training Pipeline")
    print("=" * 60)
    
    # Download data
    df = download_data(symbols, start_date, end_date)
    
    if df.empty:
        print("\nERROR: No data downloaded. Check internet connection.")
        return
    
    print(f"\nTotal samples collected: {len(df)}")
    
    # Train models
    feature_cols = get_feature_columns()
    model_long, model_short = train_models(df, feature_cols)
    
    # Save models
    save_models(model_long, model_short, feature_cols)
    
    print("\n" + "=" * 60)
    print("Training complete! Next steps:")
    print("=" * 60)
    print("1. Review models in notebooks/ml_mean_reversion_training.ipynb")
    print("2. Run LEAN backtest:")
    print("   cd lean")
    print("   lean backtest ml_mean_reversion_main.py")
    print("=" * 60)

if __name__ == '__main__':
    main()
