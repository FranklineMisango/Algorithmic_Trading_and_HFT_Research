"""
Feature Engineering for Music Royalties Strategy
Calculates stability ratio, catalog age features, and price multipliers
"""

import pandas as pd
import numpy as np
from typing import Dict, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RoyaltyFeatureEngine:
    """
    Feature engineering for music royalty assets
    """
    
    def __init__(self, config: Dict):
        """
        Initialize feature engine
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer all features for the dataset
        
        Args:
            df: Input DataFrame with raw transaction data
            
        Returns:
            DataFrame with engineered features
        """
        df = df.copy()
        
        # Calculate stability ratio
        df = self._calculate_stability_ratio(df)
        
        # Calculate price multiplier (target variable)
        df = self._calculate_price_multiplier(df)
        
        # Engineer age features
        df = self._engineer_age_features(df)
        
        # Stability deviation feature
        df = self._calculate_stability_deviation(df)
        
        # Revenue features
        df = self._engineer_revenue_features(df)
        
        # Time-based features
        df = self._engineer_time_features(df)
        
        logger.info(f"Engineered {len(df.columns)} total features")
        
        return df
    
    def _calculate_stability_ratio(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Stability Ratio = Revenue_LTM / Revenue_LTY
        
        Interpretation: Ratio close to 1.0 indicates consistent revenue
        Market pays premium for stability
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with stability_ratio column
        """
        df['stability_ratio'] = df['revenue_ltm'] / df['revenue_lty']
        
        # Handle edge cases
        df['stability_ratio'] = df['stability_ratio'].replace([np.inf, -np.inf], np.nan)
        
        # Log statistics
        mean_stability = df['stability_ratio'].mean()
        median_stability = df['stability_ratio'].median()
        logger.info(f"Stability Ratio - Mean: {mean_stability:.3f}, Median: {median_stability:.3f}")
        
        return df
    
    def _calculate_price_multiplier(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Price Multiplier = Transaction_Price / Revenue_LTM
        
        This is the target variable for model training
        Represents how many years of revenue the market is willing to pay
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with price_multiplier column
        """
        df['price_multiplier'] = df['transaction_price'] / df['revenue_ltm']
        
        # Handle edge cases
        df['price_multiplier'] = df['price_multiplier'].replace([np.inf, -np.inf], np.nan)
        
        # Log statistics
        mean_mult = df['price_multiplier'].mean()
        median_mult = df['price_multiplier'].median()
        logger.info(f"Price Multiplier - Mean: {mean_mult:.3f}, Median: {median_mult:.3f}")
        
        return df
    
    def _engineer_age_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer catalog age features
        
        Older catalogs command premium (proven survivorship through tech changes)
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with age features
        """
        # Raw age
        df['catalog_age'] = df['catalog_age'].astype(float)
        
        # Age bins (young, mature, classic, vintage)
        df['age_category'] = pd.cut(
            df['catalog_age'],
            bins=[0, 10, 30, 50, 100],
            labels=['young', 'mature', 'classic', 'vintage']
        )
        
        # Age squared (non-linear age premium)
        df['catalog_age_squared'] = df['catalog_age'] ** 2
        
        # Log age (diminishing returns to age)
        df['catalog_age_log'] = np.log1p(df['catalog_age'])
        
        logger.info(f"Age distribution - Mean: {df['catalog_age'].mean():.1f} years")
        
        return df
    
    def _calculate_stability_deviation(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate deviation from ideal stability (1.0)
        
        Market penalizes volatility in either direction
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with stability deviation features
        """
        target_stability = self.config['features']['stability']['target_value']
        
        # Absolute deviation from target
        df['stability_deviation'] = np.abs(df['stability_ratio'] - target_stability)
        
        # Squared deviation (quadratic penalty)
        df['stability_deviation_squared'] = (df['stability_ratio'] - target_stability) ** 2
        
        # Stability quality flag (1 if close to target, 0 if far)
        df['is_stable'] = (df['stability_deviation'] < 0.2).astype(int)
        
        logger.info(f"{df['is_stable'].sum()} assets ({df['is_stable'].mean()*100:.1f}%) have high stability")
        
        return df
    
    def _engineer_revenue_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer revenue-based features
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with revenue features
        """
        # Log revenue (for modeling)
        df['revenue_ltm_log'] = np.log1p(df['revenue_ltm'])
        df['revenue_lty_log'] = np.log1p(df['revenue_lty'])
        
        # Revenue size categories
        df['revenue_size'] = pd.cut(
            df['revenue_ltm'],
            bins=[0, 10000, 50000, 100000, np.inf],
            labels=['small', 'medium', 'large', 'very_large']
        )
        
        # Revenue growth rate (LTM vs LTY)
        df['revenue_growth'] = (df['revenue_ltm'] / df['revenue_lty']) - 1.0
        
        logger.info(f"Revenue range: ${df['revenue_ltm'].min():.0f} - ${df['revenue_ltm'].max():.0f}")
        
        return df
    
    def _engineer_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer time-based features
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with time features
        """
        df['transaction_date'] = pd.to_datetime(df['transaction_date'])
        
        # Year and month
        df['transaction_year'] = df['transaction_date'].dt.year
        df['transaction_month'] = df['transaction_date'].dt.month
        df['transaction_quarter'] = df['transaction_date'].dt.quarter
        
        # Days since start of study
        min_date = df['transaction_date'].min()
        df['days_since_start'] = (df['transaction_date'] - min_date).dt.days
        
        return df
    
    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create interaction features between key variables
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with interaction features
        """
        df = df.copy()
        
        # Stability × Age interaction
        df['stability_age_interaction'] = df['stability_ratio'] * df['catalog_age']
        
        # Stability × Revenue size
        df['stability_revenue_interaction'] = df['stability_ratio'] * df['revenue_ltm_log']
        
        # Age × Revenue size
        df['age_revenue_interaction'] = df['catalog_age'] * df['revenue_ltm_log']
        
        logger.info("Created interaction features")
        
        return df
    
    def get_model_features(self) -> List[str]:
        """
        Get list of features to use in model
        
        Returns:
            List of feature column names
        """
        base_features = self.config['model']['features']
        
        # Could add more features here if needed
        additional_features = []
        
        all_features = base_features + additional_features
        
        return all_features
    
    def get_feature_statistics(self, df: pd.DataFrame) -> Dict:
        """
        Calculate statistics for all features
        
        Args:
            df: Input DataFrame
            
        Returns:
            Dictionary of feature statistics
        """
        stats = {}
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            stats[col] = {
                'mean': df[col].mean(),
                'median': df[col].median(),
                'std': df[col].std(),
                'min': df[col].min(),
                'max': df[col].max(),
                'missing': df[col].isna().sum()
            }
        
        return stats
    
    def validate_features(self, df: pd.DataFrame) -> bool:
        """
        Validate that required features are present and valid
        
        Args:
            df: Input DataFrame
            
        Returns:
            True if validation passes
            
        Raises:
            ValueError if validation fails
        """
        required = ['stability_ratio', 'catalog_age', 'price_multiplier']
        
        # Check presence
        missing = [col for col in required if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required features: {missing}")
        
        # Check for excessive NaNs
        for col in required:
            nan_pct = df[col].isna().mean()
            if nan_pct > 0.1:  # More than 10% missing
                logger.warning(f"{col} has {nan_pct*100:.1f}% missing values")
        
        # Check for infinite values
        for col in required:
            if np.isinf(df[col]).any():
                raise ValueError(f"{col} contains infinite values")
        
        logger.info("Feature validation passed")
        return True


def engineer_all_features(df: pd.DataFrame, config: Dict, 
                          include_interactions: bool = False) -> pd.DataFrame:
    """
    Convenience function to engineer all features
    
    Args:
        df: Input DataFrame
        config: Configuration dictionary
        include_interactions: Whether to include interaction features
        
    Returns:
        DataFrame with all engineered features
    """
    engine = RoyaltyFeatureEngine(config)
    
    # Engineer base features
    df = engine.engineer_features(df)
    
    # Add interactions if requested
    if include_interactions:
        df = engine.create_interaction_features(df)
    
    # Validate
    engine.validate_features(df)
    
    return df


if __name__ == '__main__':
    import yaml
    from pathlib import Path
    from data_loader import load_and_prepare_data
    
    # Load config
    config_path = Path(__file__).parent / 'config.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Load data
    data_splits = load_and_prepare_data(config)
    
    # Engineer features on training data
    train_df = engineer_all_features(data_splits['train'], config, include_interactions=True)
    
    print("\n=== Feature Engineering Complete ===")
    print(f"Total features: {len(train_df.columns)}")
    print(f"\nSample features:")
    print(train_df[['stability_ratio', 'catalog_age', 'price_multiplier']].head())
    print(f"\nFeature statistics:")
    engine = RoyaltyFeatureEngine(config)
    stats = engine.get_feature_statistics(train_df)
    for feat in ['stability_ratio', 'catalog_age', 'price_multiplier']:
        print(f"{feat}: mean={stats[feat]['mean']:.2f}, std={stats[feat]['std']:.2f}")
