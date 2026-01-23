"""
Feature Engine
==============

Vectorized feature engineering from order book data.
Avoids data leakage with proper rolling window handling.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, List, Optional, Tuple
from sklearn.feature_selection import mutual_info_regression, SelectKBest
from imblearn.over_sampling import SMOTE


class FeatureEngine:
    """
    Efficient, vectorized feature engineering for order book data.
    
    CRITICAL: All rolling features use only past data (no future leakage).
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize FeatureEngine.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = logging.getLogger("FuturesPrediction.FeatureEngine")
        self.feature_config = config.get("features", {})
        self.K = self.feature_config.get("top_levels", 10)
        self.MAX_LEVELS = self.feature_config.get("max_levels", 20)
        self.rolling_window = self.feature_config.get("rolling_window", 10)
        self.selector = None
        self.selected_features = None
        
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create all features from order book data using vectorized operations.
        
        Args:
            df: DataFrame with order book columns
            
        Returns:
            DataFrame with engineered features
        """
        self.logger.info("Engineering features...")
        df = df.copy()
        
        # Basic price features (vectorized)
        df['mid_price'] = df.apply(
            lambda row: (row['bid_prices'][0] + row['ask_prices'][0]) / 2, axis=1
        )
        df['spread'] = df.apply(
            lambda row: row['ask_prices'][0] - row['bid_prices'][0], axis=1
        )
        
        # Volume-based features (optimized)
        df['bid_sum_k'] = df['bid_quantities'].apply(lambda x: self._safe_slice_sum(x, self.K))
        df['ask_sum_k'] = df['ask_quantities'].apply(lambda x: self._safe_slice_sum(x, self.K))
        
        # Imbalance
        df['imbalance'] = (df['bid_sum_k'] - df['ask_sum_k']) / (
            df['bid_sum_k'] + df['ask_sum_k']
        ).replace(0, np.nan)
        df['imbalance'] = df['imbalance'].fillna(0)
        
        # Weighted mid-price
        df['bid_weighted'] = df.apply(
            lambda row: self._safe_weighted_avg(row['bid_prices'], row['bid_quantities'], self.K),
            axis=1
        )
        df['ask_weighted'] = df.apply(
            lambda row: self._safe_weighted_avg(row['ask_prices'], row['ask_quantities'], self.K),
            axis=1
        )
        df['weighted_mid_price'] = (df['bid_weighted'] + df['ask_weighted']) / 2
        
        # Cumulative volumes
        df['bid_cum_volume'] = df['bid_sum_k'].copy()
        df['ask_cum_volume'] = df['ask_sum_k'].copy()
        
        # Price slopes (trend)
        df['bid_slope'] = df['bid_prices'].apply(lambda x: self._slope_linregress(x, self.K))
        df['ask_slope'] = df['ask_prices'].apply(lambda x: self._slope_linregress(x, self.K))
        
        # Depth and liquidity
        df['depth'] = df['bid_sum_k'] + df['ask_sum_k']
        df['liquidity_ratio'] = df['bid_sum_k'] / df['ask_sum_k'].replace(0, np.nan)
        df['liquidity_ratio'] = df['liquidity_ratio'].fillna(0)
        
        # CRITICAL: Rolling volatility with min_periods to avoid looking ahead
        # This ensures the first rolling_window-1 values are NaN, which we'll handle
        df['mid_price_volatility'] = df['mid_price'].rolling(
            window=self.rolling_window, 
            min_periods=self.rolling_window  # Ensures no forward bias
        ).std()
        
        # Additional rolling features (past-only)
        df['mid_price_ma'] = df['mid_price'].rolling(
            window=self.rolling_window,
            min_periods=self.rolling_window
        ).mean()
        
        df['spread_ma'] = df['spread'].rolling(
            window=self.rolling_window,
            min_periods=self.rolling_window
        ).mean()
        
        # Price momentum (past-only)
        df['price_momentum'] = df['mid_price'] - df['mid_price'].shift(self.rolling_window)
        
        # Order book image for CNN (if needed)
        df['order_book_image'] = df.apply(
            lambda row: self._create_order_book_image(
                row['bid_quantities'], row['ask_quantities']
            ),
            axis=1
        )
        
        # Clean up intermediate columns
        df = df.drop(columns=['bid_sum_k', 'ask_sum_k', 'bid_weighted', 'ask_weighted'],
                    errors='ignore')
        
        self.logger.info(f"Feature engineering complete. Shape: {df.shape}")
        return df
    
    def select_features(self, X_train: pd.DataFrame, y_train: pd.Series, 
                       X_test: pd.DataFrame, k: Optional[int] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Select top-k features using mutual information.
        
        CRITICAL: Fit only on training data to avoid leakage.
        
        Args:
            X_train: Training features
            y_train: Training target
            X_test: Test features
            k: Number of features to select. If None, uses config.
            
        Returns:
            Tuple of (X_train_selected, X_test_selected)
        """
        if k is None:
            k = self.feature_config.get("feature_selection_k", 8)
        
        # Get numerical features only
        numeric_features = X_train.select_dtypes(include=[np.number]).columns.tolist()
        X_train_numeric = X_train[numeric_features].copy()
        X_test_numeric = X_test[numeric_features].copy()
        
        # Handle NaN values (from rolling features)
        X_train_numeric = X_train_numeric.fillna(0)
        X_test_numeric = X_test_numeric.fillna(0)
        
        # Feature selection
        self.selector = SelectKBest(score_func=mutual_info_regression, k=min(k, len(numeric_features)))
        X_train_selected = self.selector.fit_transform(X_train_numeric, y_train)
        
        # Get selected feature names
        selected_indices = self.selector.get_support(indices=True)
        self.selected_features = [numeric_features[i] for i in selected_indices]
        
        self.logger.info(f"Selected {len(self.selected_features)} features: {self.selected_features}")
        
        # Transform test set
        X_test_selected = self.selector.transform(X_test_numeric)
        
        # Convert back to DataFrame
        X_train_selected = pd.DataFrame(X_train_selected, columns=self.selected_features,
                                       index=X_train.index)
        X_test_selected = pd.DataFrame(X_test_selected, columns=self.selected_features,
                                      index=X_test.index)
        
        return X_train_selected, X_test_selected
    
    def handle_class_imbalance(self, X_train: pd.DataFrame, y_train: pd.Series,
                               use_smote: Optional[bool] = None) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Handle class imbalance with SMOTE if needed.
        
        Args:
            X_train: Training features
            y_train: Training classification target
            use_smote: Whether to use SMOTE. If None, uses config.
            
        Returns:
            Tuple of (X_train_resampled, y_train_resampled)
        """
        if use_smote is None:
            use_smote = self.feature_config.get("use_smote", True)
        
        if not use_smote:
            return X_train, y_train
        
        # Check imbalance ratio
        value_counts = y_train.value_counts()
        imbalance_ratio = value_counts.min() / value_counts.max()
        threshold = self.feature_config.get("smote_imbalance_threshold", 0.5)
        
        if imbalance_ratio < threshold:
            self.logger.info(f"Class imbalance detected (ratio: {imbalance_ratio:.2f}). Applying SMOTE...")
            smote = SMOTE(random_state=self.config.get("random_seed", 42))
            X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
            
            self.logger.info(f"SMOTE applied: {len(X_train)} -> {len(X_train_resampled)} samples")
            
            # Convert back to DataFrame/Series
            X_train_resampled = pd.DataFrame(X_train_resampled, columns=X_train.columns)
            y_train_resampled = pd.Series(y_train_resampled, name=y_train.name)
            
            return X_train_resampled, y_train_resampled
        else:
            self.logger.info(f"Class balance acceptable (ratio: {imbalance_ratio:.2f}). Skipping SMOTE.")
            return X_train, y_train
    
    def create_sequences(self, X: pd.DataFrame, y: pd.Series, 
                        sequence_length: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for LSTM/RNN models WITHOUT data leakage.
        
        Args:
            X: Feature DataFrame
            y: Target Series
            sequence_length: Sequence length. If None, uses config.
            
        Returns:
            Tuple of (X_sequences, y_sequences)
        """
        if sequence_length is None:
            sequence_length = self.feature_config.get("sequence_length", 20)
        
        X_seq = []
        y_seq = []
        
        # Create sequences using only past data
        for i in range(sequence_length, len(X)):
            X_seq.append(X.iloc[i-sequence_length:i].values)
            y_seq.append(y.iloc[i])
        
        X_seq = np.array(X_seq)
        y_seq = np.array(y_seq)
        
        self.logger.info(f"Created {len(X_seq)} sequences of length {sequence_length}")
        
        return X_seq, y_seq
    
    # Helper methods
    def _safe_slice_sum(self, lst: List, k: int) -> float:
        """Safely sum first k elements of list."""
        return sum(lst[:k]) if len(lst) >= k else sum(lst)
    
    def _safe_weighted_avg(self, prices: List, quantities: List, k: int) -> float:
        """Calculate weighted average of top k levels."""
        if len(prices) < k or len(quantities) < k:
            k = min(len(prices), len(quantities))
        if k == 0:
            return 0
        
        total_vol = sum(quantities[:k])
        if total_vol == 0:
            return prices[0] if prices else 0
        
        return sum(p * q for p, q in zip(prices[:k], quantities[:k])) / total_vol
    
    def _slope_linregress(self, prices: List, k: int) -> float:
        """Calculate slope of price levels using linear regression."""
        if len(prices) < k or k < 2:
            return 0
        
        x = np.arange(k)
        y = np.array(prices[:k])
        
        if len(np.unique(y)) < 2:
            return 0
        
        slope = np.polyfit(x, y, 1)[0]
        return slope
    
    def _create_order_book_image(self, bids: List, asks: List) -> np.ndarray:
        """Create 2D order book image for CNN."""
        bids_arr = np.array(bids[:self.MAX_LEVELS])
        asks_arr = np.array(asks[:self.MAX_LEVELS])
        
        # Pad if necessary
        if len(bids_arr) < self.MAX_LEVELS:
            bids_arr = np.pad(bids_arr, (0, self.MAX_LEVELS - len(bids_arr)), constant_values=0)
        if len(asks_arr) < self.MAX_LEVELS:
            asks_arr = np.pad(asks_arr, (0, self.MAX_LEVELS - len(asks_arr)), constant_values=0)
        
        return np.stack([bids_arr, asks_arr], axis=-1)
