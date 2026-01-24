"""
Data Processor
==============

Handles data loading, validation, cleaning, and preprocessing for order book data.
Fixes data leakage issues with proper time-aware processing.
"""

import pandas as pd
import numpy as np
import ast
import logging
from typing import Tuple, Dict, Any, Optional
from pathlib import Path


class DataProcessor:
    """
    Processes order book data with comprehensive validation and cleaning.
    
    Ensures no data leakage through proper time-aware splits and validation.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize DataProcessor.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = logging.getLogger("FuturesPrediction.DataProcessor")
        self.data_config = config.get("data", {})
        self.df = None
        
    def load_data(self, file_path: Optional[str] = None) -> pd.DataFrame:
        """
        Load order book data from CSV file.
        
        Args:
            file_path: Path to CSV file. If None, uses config default.
            
        Returns:
            Loaded DataFrame
            
        Raises:
            FileNotFoundError: If file doesn't exist
        """
        if file_path is None:
            file_path = self.data_config.get("file_path", "live_order_book_data.csv")
        
        file_path = Path(file_path)
        if not file_path.exists():
            self.logger.error(f"Data file not found: {file_path}")
            raise FileNotFoundError(f"Data file not found: {file_path}")
        
        self.logger.info(f"Loading data from {file_path}")
        df = pd.read_csv(file_path)
        self.logger.info(f"Loaded {len(df)} records")
        
        # Parse timestamp
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Sort by timestamp to ensure chronological order
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        # Parse string columns back to lists
        list_columns = ['bid_prices', 'bid_quantities', 'ask_prices', 'ask_quantities']
        for col in list_columns:
            if col in df.columns and df[col].dtype == 'object':
                df[col] = df[col].apply(ast.literal_eval)
        
        self.df = df
        return df
    
    def validate_data(self, df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Comprehensive data validation with multiple checks.
        
        Args:
            df: DataFrame to validate. If None, uses self.df
            
        Returns:
            Validated and cleaned DataFrame
        """
        if df is None:
            df = self.df
        
        if df is None or df.empty:
            raise ValueError("No data to validate. Load data first.")
        
        self.logger.info("Starting data validation...")
        initial_count = len(df)
        
        # 1. Check for duplicate timestamps
        duplicates = df.duplicated(subset=['timestamp']).sum()
        if duplicates > 0:
            self.logger.warning(f"Removing {duplicates} duplicate timestamps")
            df = df.drop_duplicates(subset=['timestamp'])
        
        # 2. Validate order book structure
        valid_mask = df.apply(self._validate_order_book_row, axis=1)
        invalid_count = (~valid_mask).sum()
        if invalid_count > 0:
            self.logger.warning(f"Removing {invalid_count} invalid order book entries")
            df = df[valid_mask].reset_index(drop=True)
        
        # 3. Outlier detection (IQR method on mid-price)
        df['temp_mid_price'] = df.apply(
            lambda row: (row['bid_prices'][0] + row['ask_prices'][0]) / 2, axis=1
        )
        
        q1 = df['temp_mid_price'].quantile(0.25)
        q3 = df['temp_mid_price'].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        outliers = ((df['temp_mid_price'] < lower_bound) | 
                   (df['temp_mid_price'] > upper_bound)).sum()
        
        if outliers > 0:
            self.logger.warning(f"Removing {outliers} price outliers")
            df = df[(df['temp_mid_price'] >= lower_bound) & 
                   (df['temp_mid_price'] <= upper_bound)].reset_index(drop=True)
        
        df = df.drop(columns=['temp_mid_price'])
        
        final_count = len(df)
        self.logger.info(f"Validation complete: {initial_count} -> {final_count} records "
                        f"({initial_count - final_count} removed)")
        
        self.df = df
        return df
    
    def _validate_order_book_row(self, row: pd.Series) -> bool:
        """
        Validate a single order book row.
        
        Args:
            row: DataFrame row
            
        Returns:
            True if valid, False otherwise
        """
        try:
            # Check if lists are valid
            if not all(isinstance(row[col], list) for col in 
                      ['bid_prices', 'bid_quantities', 'ask_prices', 'ask_quantities']):
                return False
            
            # Check if lists are non-empty
            if any(len(row[col]) == 0 for col in 
                  ['bid_prices', 'bid_quantities', 'ask_prices', 'ask_quantities']):
                return False
            
            # Check monotonicity (bids descending, asks ascending)
            if row['bid_prices'] != sorted(row['bid_prices'], reverse=True):
                return False
            
            if row['ask_prices'] != sorted(row['ask_prices']):
                return False
            
            # Check that bid prices < ask prices (no crossed book)
            if row['bid_prices'][0] >= row['ask_prices'][0]:
                return False
            
            # Check for negative quantities
            if any(q < 0 for q in row['bid_quantities'] + row['ask_quantities']):
                return False
            
            return True
        except Exception:
            return False
    
    def create_targets(self, df: Optional[pd.DataFrame] = None, 
                       look_ahead: int = 1) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
        """
        Create target variables WITHOUT data leakage.
        
        CRITICAL: Uses forward-looking target creation with proper alignment.
        The target at time t is the price change from t to t+look_ahead.
        After creating targets, we must drop the last look_ahead rows to avoid using future data.
        
        Args:
            df: DataFrame with features. If None, uses self.df
            look_ahead: Number of periods to look ahead for target (default: 1)
            
        Returns:
            Tuple of (df_clean, y_regression, y_classification)
        """
        if df is None:
            df = self.df
        
        if df is None or df.empty:
            raise ValueError("No data to create targets. Load data first.")
        
        # Calculate mid-price if not already present
        if 'mid_price' not in df.columns:
            df['mid_price'] = df.apply(
                lambda row: (row['bid_prices'][0] + row['ask_prices'][0]) / 2, axis=1
            )
        
        # Create price change target: change from t to t+look_ahead
        # This is the ONLY place we use future data, and we handle it explicitly
        df['price_change'] = df['mid_price'].shift(-look_ahead) - df['mid_price']
        
        # Drop the last look_ahead rows where we don't have a target
        # This prevents data leakage
        df_clean = df.iloc[:-look_ahead].copy()
        
        # Create regression and classification targets
        y_regression = df_clean['price_change'].copy()
        y_classification = (y_regression > 0).astype(int)
        
        # Remove target from features
        df_clean = df_clean.drop(columns=['price_change'], errors='ignore')
        
        self.logger.info(f"Created targets with look_ahead={look_ahead}")
        self.logger.info(f"Target statistics - Mean: {y_regression.mean():.6f}, "
                        f"Std: {y_regression.std():.6f}")
        self.logger.info(f"Classification distribution:\n{y_classification.value_counts()}")
        
        return df_clean, y_regression, y_classification
    
    def time_aware_split(self, df: pd.DataFrame, y_reg: pd.Series, y_clf: pd.Series, 
                        test_size: float = None) -> Tuple:
        """
        Create time-aware train/test split.
        
        CRITICAL: No shuffling - maintains temporal order to prevent data leakage.
        
        Args:
            df: Feature DataFrame
            y_reg: Regression target
            y_clf: Classification target
            test_size: Proportion for test set. If None, uses config.
            
        Returns:
            Tuple of (X_train, X_test, y_train_reg, y_test_reg, y_train_clf, y_test_clf, split_idx)
        """
        if test_size is None:
            test_size = self.data_config.get("test_split", 0.2)
        
        split_idx = int(len(df) * (1 - test_size))
        
        X_train = df.iloc[:split_idx].copy()
        X_test = df.iloc[split_idx:].copy()
        
        y_train_reg = y_reg.iloc[:split_idx].copy()
        y_test_reg = y_reg.iloc[split_idx:].copy()
        
        y_train_clf = y_clf.iloc[:split_idx].copy()
        y_test_clf = y_clf.iloc[split_idx:].copy()
        
        self.logger.info(f"Time-aware split: Train={len(X_train)}, Test={len(X_test)}")
        self.logger.info(f"Split index: {split_idx}")
        
        return X_train, X_test, y_train_reg, y_test_reg, y_train_clf, y_test_clf, split_idx
