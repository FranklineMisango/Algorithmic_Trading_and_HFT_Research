"""
Data Loader for Music Royalties Strategy
Loads and validates transaction data from Royalty Exchange or similar platforms
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict, List
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RoyaltyDataLoader:
    """
    Loads and preprocesses music royalty transaction data
    """
    
    def __init__(self, config: Dict):
        """
        Initialize data loader
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.required_fields = config['data']['required_fields']
        self.start_date = pd.to_datetime(config['data']['start_date'])
        self.end_date = pd.to_datetime(config['data']['end_date'])
        
    def load_data(self, filepath: Optional[str] = None) -> pd.DataFrame:
        """
        Load royalty transaction data from CSV
        
        Args:
            filepath: Path to data file. If None, generates synthetic data
            
        Returns:
            DataFrame with transaction data
        """
        if filepath and Path(filepath).exists():
            logger.info(f"Loading data from {filepath}")
            df = pd.read_csv(filepath)
            df['transaction_date'] = pd.to_datetime(df['transaction_date'])
        else:
            logger.warning("No data file provided. Generating synthetic data for demonstration")
            df = self._generate_synthetic_data()
        
        # Validate required fields
        self._validate_fields(df)
        
        # Filter date range
        df = df[(df['transaction_date'] >= self.start_date) & 
                (df['transaction_date'] <= self.end_date)]
        
        logger.info(f"Loaded {len(df)} transactions from {df['transaction_date'].min()} to {df['transaction_date'].max()}")
        
        return df
    
    def _validate_fields(self, df: pd.DataFrame) -> None:
        """
        Validate that DataFrame contains required fields
        
        Args:
            df: Input DataFrame
            
        Raises:
            ValueError if required fields are missing
        """
        missing_fields = [field for field in self.required_fields if field not in df.columns]
        if missing_fields:
            raise ValueError(f"Missing required fields: {missing_fields}")
        
        logger.info("All required fields present")
    
    def _generate_synthetic_data(self) -> pd.DataFrame:
        """
        Generate synthetic music royalty transaction data for demonstration
        Based on characteristics from the study
        
        Returns:
            Synthetic DataFrame
        """
        np.random.seed(42)
        
        n_transactions = 1000
        
        # Generate dates uniformly across study period
        date_range = pd.date_range(self.start_date, self.end_date, freq='D')
        transaction_dates = np.random.choice(date_range, size=n_transactions)
        
        # Generate catalog ages (mix of old and new)
        # Older catalogs should be more common (survivorship)
        catalog_ages = np.random.choice(
            np.arange(1, 71),  # 1 to 70 years
            size=n_transactions,
            p=self._age_distribution()
        )
        
        # Generate revenue (LTM and LTY)
        # LTM: $1,000 to $500,000 (log-normal distribution)
        revenue_ltm = np.random.lognormal(mean=9.0, sigma=1.5, size=n_transactions)
        revenue_ltm = np.clip(revenue_ltm, 1000, 500000)
        
        # LTY: Similar to LTM but with some variation
        # Stability ratio ~ 1.0 for most assets
        stability_noise = np.random.normal(1.0, 0.2, size=n_transactions)
        stability_noise = np.clip(stability_noise, 0.5, 2.0)
        revenue_lty = revenue_ltm / stability_noise
        
        # Generate price multipliers based on stability and age
        # Model: Multiplier = β₀ + β₁(StabilityRatio) + β₂(CatalogAge) + ε
        stability_ratio = revenue_ltm / revenue_lty
        
        # Coefficients (illustrative, similar to study findings)
        beta_0 = 5.0  # Base multiplier
        beta_1 = 3.0  # Stability premium (higher when ratio ~ 1.0)
        beta_2 = 0.05  # Age premium (older = higher)
        
        # Stability effect (quadratic penalty for deviation from 1.0)
        stability_effect = beta_1 * (1 - np.abs(stability_ratio - 1.0))
        age_effect = beta_2 * catalog_ages
        noise = np.random.normal(0, 2.0, size=n_transactions)
        
        price_multipliers = beta_0 + stability_effect + age_effect + noise
        price_multipliers = np.clip(price_multipliers, 2, 20)  # Reasonable range
        
        # Calculate transaction prices
        transaction_prices = revenue_ltm * price_multipliers
        
        # Generate contract types (80% LOR, 20% 10-Year as per study)
        contract_types = np.random.choice(
            ['LOR', '10-Year Term'],
            size=n_transactions,
            p=[0.8, 0.2]
        )
        
        # Generate genres
        genres = np.random.choice(
            ['Pop', 'Rock', 'Country', 'Hip-Hop', 'R&B', 'Electronic', 'Jazz', 'Classical'],
            size=n_transactions,
            p=[0.25, 0.20, 0.15, 0.15, 0.10, 0.08, 0.04, 0.03]
        )
        
        # Create DataFrame
        df = pd.DataFrame({
            'asset_id': [f'ASSET_{i:04d}' for i in range(n_transactions)],
            'transaction_date': transaction_dates,
            'transaction_price': transaction_prices,
            'revenue_ltm': revenue_ltm,
            'revenue_lty': revenue_lty,
            'catalog_age': catalog_ages,
            'contract_type': contract_types,
            'genre': genres
        })
        
        # Sort by date
        df = df.sort_values('transaction_date').reset_index(drop=True)
        
        logger.info(f"Generated {n_transactions} synthetic transactions")
        
        return df
    
    def _age_distribution(self) -> np.ndarray:
        """
        Generate age distribution favoring older catalogs
        (Older catalogs have proven survivorship)
        
        Returns:
            Probability distribution over ages 1-70
        """
        ages = np.arange(1, 71)
        # Exponential decay with floor
        probs = np.exp(-0.03 * ages) + 0.01
        probs = probs / probs.sum()
        return probs
    
    def filter_lor_only(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Filter to Life of Rights contracts only
        (10-Year Term contracts suffer guaranteed capital depreciation)
        
        Args:
            df: Input DataFrame
            
        Returns:
            Filtered DataFrame
        """
        original_count = len(df)
        df_filtered = df[df['contract_type'] == 'LOR'].copy()
        removed = original_count - len(df_filtered)
        
        logger.info(f"Filtered to LOR contracts: {len(df_filtered)} kept, {removed} removed")
        
        return df_filtered
    
    def clean_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove extreme outliers in stability ratio
        
        Args:
            df: Input DataFrame
            
        Returns:
            Cleaned DataFrame
        """
        # Calculate stability ratio if not present
        if 'stability_ratio' not in df.columns:
            df['stability_ratio'] = df['revenue_ltm'] / df['revenue_lty']
        
        min_stability = self.config['features']['stability']['min_threshold']
        max_stability = self.config['features']['stability']['max_threshold']
        
        original_count = len(df)
        df_clean = df[
            (df['stability_ratio'] >= min_stability) &
            (df['stability_ratio'] <= max_stability)
        ].copy()
        removed = original_count - len(df_clean)
        
        logger.info(f"Removed {removed} outliers with extreme stability ratios")
        
        return df_clean
    
    def split_train_val_test(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Split data into train/validation/test sets based on dates
        
        Args:
            df: Input DataFrame
            
        Returns:
            Dictionary with 'train', 'validation', 'test' DataFrames
        """
        train_start = pd.to_datetime(self.config['data']['train_start'])
        train_end = pd.to_datetime(self.config['data']['train_end'])
        val_start = pd.to_datetime(self.config['data']['validation_start'])
        val_end = pd.to_datetime(self.config['data']['validation_end'])
        test_start = pd.to_datetime(self.config['data']['test_start'])
        test_end = pd.to_datetime(self.config['data']['test_end'])
        
        train_df = df[(df['transaction_date'] >= train_start) & 
                      (df['transaction_date'] <= train_end)].copy()
        
        val_df = df[(df['transaction_date'] >= val_start) & 
                    (df['transaction_date'] <= val_end)].copy()
        
        test_df = df[(df['transaction_date'] >= test_start) & 
                     (df['transaction_date'] <= test_end)].copy()
        
        logger.info(f"Train: {len(train_df)} | Validation: {len(val_df)} | Test: {len(test_df)}")
        
        return {
            'train': train_df,
            'validation': val_df,
            'test': test_df
        }
    
    def get_summary_statistics(self, df: pd.DataFrame) -> Dict:
        """
        Calculate summary statistics for the dataset
        
        Args:
            df: Input DataFrame
            
        Returns:
            Dictionary of summary statistics
        """
        if 'stability_ratio' not in df.columns:
            df['stability_ratio'] = df['revenue_ltm'] / df['revenue_lty']
        
        if 'price_multiplier' not in df.columns:
            df['price_multiplier'] = df['transaction_price'] / df['revenue_ltm']
        
        stats = {
            'n_transactions': len(df),
            'date_range': f"{df['transaction_date'].min()} to {df['transaction_date'].max()}",
            'total_transaction_volume': df['transaction_price'].sum(),
            'avg_transaction_price': df['transaction_price'].mean(),
            'median_transaction_price': df['transaction_price'].median(),
            'avg_revenue_ltm': df['revenue_ltm'].mean(),
            'avg_catalog_age': df['catalog_age'].mean(),
            'avg_stability_ratio': df['stability_ratio'].mean(),
            'median_stability_ratio': df['stability_ratio'].median(),
            'avg_price_multiplier': df['price_multiplier'].mean(),
            'median_price_multiplier': df['price_multiplier'].median(),
            'contract_type_counts': df['contract_type'].value_counts().to_dict(),
            'genre_distribution': df['genre'].value_counts().to_dict() if 'genre' in df.columns else {}
        }
        
        return stats


def load_and_prepare_data(config: Dict, filepath: Optional[str] = None) -> Dict[str, pd.DataFrame]:
    """
    Convenience function to load and prepare data in one step
    
    Args:
        config: Configuration dictionary
        filepath: Optional path to data file
        
    Returns:
        Dictionary with 'train', 'validation', 'test' DataFrames
    """
    loader = RoyaltyDataLoader(config)
    
    # Load raw data
    df = loader.load_data(filepath)
    
    # Filter to LOR only
    df = loader.filter_lor_only(df)
    
    # Clean outliers
    df = loader.clean_outliers(df)
    
    # Get summary stats
    stats = loader.get_summary_statistics(df)
    logger.info(f"Data Summary: {stats['n_transactions']} transactions, "
                f"${stats['total_transaction_volume']:,.0f} total volume")
    
    # Split into train/val/test
    splits = loader.split_train_val_test(df)
    
    return splits


if __name__ == '__main__':
    import yaml
    
    # Load config
    config_path = Path(__file__).parent / 'config.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Load and prepare data
    data_splits = load_and_prepare_data(config)
    
    print("\n=== Data Loading Complete ===")
    print(f"Train: {len(data_splits['train'])} transactions")
    print(f"Validation: {len(data_splits['validation'])} transactions")
    print(f"Test: {len(data_splits['test'])} transactions")
