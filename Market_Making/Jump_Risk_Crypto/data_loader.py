"""
Data Loader for Crypto Jump Risk Analysis
Loads crypto price and volume data for jump detection
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict, List
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CryptoDataLoader:
    """
    Loads and preprocesses cryptocurrency price/volume data
    """
    
    def __init__(self, config: Dict):
        """
        Initialize data loader
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.assets = config['data']['assets']['major'] + config['data']['assets']['altcoins']
        self.start_date = pd.to_datetime(config['data']['start_date'])
        self.end_date = pd.to_datetime(config['data']['end_date'])
        
    def load_data(self, filepath: Optional[str] = None) -> pd.DataFrame:
        """
        Load crypto data from file or generate synthetic data
        
        Args:
            filepath: Path to CSV file. If None, generates synthetic data
            
        Returns:
            DataFrame with date, asset, close, volume, returns
        """
        if filepath and Path(filepath).exists():
            logger.info(f"Loading data from {filepath}")
            df = pd.read_csv(filepath)
            df['date'] = pd.to_datetime(df['date'])
        else:
            logger.warning("No data file provided. Generating synthetic crypto data")
            df = self._generate_synthetic_data()
        
        # Calculate returns
        df = self._calculate_returns(df)
        
        # Filter date range
        df = df[(df['date'] >= self.start_date) & (df['date'] <= self.end_date)]
        
        logger.info(f"Loaded {len(df)} observations for {df['asset'].nunique()} assets")
        logger.info(f"Date range: {df['date'].min()} to {df['date'].max()}")
        
        return df
    
    def _generate_synthetic_data(self) -> pd.DataFrame:
        """
        Generate synthetic crypto price data with jump dynamics
        
        Returns:
            DataFrame with synthetic data
        """
        np.random.seed(42)
        
        date_range = pd.date_range(self.start_date, self.end_date, freq='D')
        n_days = len(date_range)
        
        all_data = []
        
        for asset in self.assets:
            # Asset-specific parameters
            if asset in ['BTC', 'ETH']:
                # Major coins: fewer but bigger jumps
                base_vol = 0.04  # 4% daily volatility
                jump_intensity = 0.05  # 5% chance of jump per day
                jump_size_mean = 0.10  # 10% average jump
            else:
                # Altcoins: more frequent but smaller jumps
                base_vol = 0.06  # 6% daily volatility
                jump_intensity = 0.10  # 10% chance of jump per day
                jump_size_mean = 0.06  # 6% average jump
            
            # Generate price series
            price = 100.0  # Start at $100
            prices = [price]
            volumes = []
            dates = [date_range[0]]
            
            for i in range(1, n_days):
                # Continuous component (drift + diffusion)
                drift = 0.0003  # Small positive drift
                diffusion = np.random.normal(0, base_vol)
                
                # Jump component
                is_jump = np.random.random() < jump_intensity
                if is_jump:
                    jump_direction = np.random.choice([-1, 1])
                    jump_size = np.random.exponential(jump_size_mean)
                    jump = jump_direction * jump_size
                else:
                    jump = 0
                
                # Total return
                total_return = drift + diffusion + jump
                
                # Update price
                price = price * (1 + total_return)
                prices.append(price)
                dates.append(date_range[i])
                
                # Volume (correlated with volatility and jumps)
                base_volume = 1000000  # Base volume
                volume_multiplier = 1 + abs(total_return) * 10  # Higher on big moves
                volume = base_volume * volume_multiplier * np.random.lognormal(0, 0.3)
                volumes.append(volume)
            
            # Create asset DataFrame
            asset_df = pd.DataFrame({
                'date': dates,
                'asset': asset,
                'close': prices,
                'volume': [volumes[0]] + volumes  # Pad first day
            })
            
            all_data.append(asset_df)
        
        # Combine all assets
        df = pd.concat(all_data, ignore_index=True)
        
        # Add some cross-asset jump contagion
        df = self._add_jump_contagion(df, date_range)
        
        logger.info(f"Generated synthetic data for {len(self.assets)} assets")
        
        return df
    
    def _add_jump_contagion(self, df: pd.DataFrame, date_range: pd.DatetimeIndex) -> pd.DataFrame:
        """
        Add correlated jumps across assets to simulate contagion
        
        Args:
            df: Input DataFrame
            date_range: Date range
            
        Returns:
            DataFrame with contagion effects
        """
        # Identify random systemic jump days (market-wide events)
        n_systemic_jumps = int(len(date_range) * 0.02)  # 2% of days
        systemic_jump_dates = np.random.choice(date_range, n_systemic_jumps, replace=False)
        
        for jump_date in systemic_jump_dates:
            # All assets jump on this day
            mask = df['date'] == jump_date
            
            # Common jump direction
            direction = np.random.choice([-1, 1])
            
            for asset in self.assets:
                asset_mask = mask & (df['asset'] == asset)
                if asset_mask.sum() > 0:
                    # Apply correlated jump
                    current_price = df.loc[asset_mask, 'close'].values[0]
                    jump_size = np.random.uniform(0.05, 0.15)  # 5-15% jump
                    new_price = current_price * (1 + direction * jump_size)
                    df.loc[asset_mask, 'close'] = new_price
                    
                    # Increase volume on jump days
                    df.loc[asset_mask, 'volume'] *= np.random.uniform(2, 5)
        
        return df
    
    def _calculate_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate log returns for each asset
        
        Args:
            df: Input DataFrame with prices
            
        Returns:
            DataFrame with returns column
        """
        df = df.sort_values(['asset', 'date'])
        df['returns'] = df.groupby('asset')['close'].transform(
            lambda x: np.log(x / x.shift(1))
        )
        
        return df
    
    def split_train_test(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Split data into train and test sets
        
        Args:
            df: Input DataFrame
            
        Returns:
            Dictionary with 'train' and 'test' DataFrames
        """
        train_start = pd.to_datetime(self.config['data']['train_start'])
        train_end = pd.to_datetime(self.config['data']['train_end'])
        test_start = pd.to_datetime(self.config['data']['test_start'])
        test_end = pd.to_datetime(self.config['data']['test_end'])
        
        train_df = df[(df['date'] >= train_start) & (df['date'] <= train_end)].copy()
        test_df = df[(df['date'] >= test_start) & (df['date'] <= test_end)].copy()
        
        logger.info(f"Train: {len(train_df)} obs | Test: {len(test_df)} obs")
        
        return {'train': train_df, 'test': test_df}
    
    def get_summary_statistics(self, df: pd.DataFrame) -> Dict:
        """
        Calculate summary statistics for the dataset
        
        Args:
            df: Input DataFrame
            
        Returns:
            Dictionary of summary statistics
        """
        stats = {}
        
        for asset in df['asset'].unique():
            asset_data = df[df['asset'] == asset].copy()
            
            stats[asset] = {
                'n_obs': len(asset_data),
                'mean_return': asset_data['returns'].mean(),
                'volatility': asset_data['returns'].std(),
                'min_return': asset_data['returns'].min(),
                'max_return': asset_data['returns'].max(),
                'skewness': asset_data['returns'].skew(),
                'kurtosis': asset_data['returns'].kurtosis(),
                'avg_volume': asset_data['volume'].mean()
            }
        
        return stats
    
    def pivot_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Pivot data to wide format (dates × assets)
        
        Args:
            df: Input DataFrame in long format
            
        Returns:
            Wide DataFrame with returns
        """
        returns_wide = df.pivot(index='date', columns='asset', values='returns')
        return returns_wide.dropna()
    
    def pivot_volumes(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Pivot volume data to wide format
        
        Args:
            df: Input DataFrame in long format
            
        Returns:
            Wide DataFrame with volumes
        """
        volumes_wide = df.pivot(index='date', columns='asset', values='volume')
        return volumes_wide.dropna()


def load_and_prepare_data(config: Dict, filepath: Optional[str] = None) -> Dict[str, pd.DataFrame]:
    """
    Convenience function to load and prepare data
    
    Args:
        config: Configuration dictionary
        filepath: Optional path to data file
        
    Returns:
        Dictionary with 'train' and 'test' DataFrames
    """
    loader = CryptoDataLoader(config)
    
    # Load data
    df = loader.load_data(filepath)
    
    # Get summary stats
    stats = loader.get_summary_statistics(df)
    logger.info("\nSummary Statistics:")
    for asset, asset_stats in list(stats.items())[:3]:  # Show first 3
        logger.info(f"  {asset}: vol={asset_stats['volatility']*100:.2f}%, "
                   f"skew={asset_stats['skewness']:.2f}, "
                   f"kurtosis={asset_stats['kurtosis']:.2f}")
    
    # Split train/test
    splits = loader.split_train_test(df)
    
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
    print(f"Train: {len(data_splits['train'])} observations")
    print(f"Test: {len(data_splits['test'])} observations")
    print(f"Assets: {data_splits['train']['asset'].nunique()}")
