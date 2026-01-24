"""
Data Acquisition Module for Statistical Arbitrage Strategy

Handles downloading and preparing survivorship-bias-free data for the Russell 3000 universe.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Optional, Tuple
import yfinance as yf
from loguru import logger
import pickle
from pathlib import Path


class DataAcquisitionEngine:
    """
    Manages data acquisition for the statistical arbitrage strategy.
    
    Responsibilities:
    - Download historical price and volume data
    - Ensure survivorship-bias-free data handling
    - Cache data locally for efficient access
    - Handle data quality checks
    """
    
    def __init__(self, data_dir: str = "./data", cache_enabled: bool = True):
        """
        Initialize data acquisition engine.
        
        Args:
            data_dir: Directory for storing cached data
            cache_enabled: Whether to use local caching
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.cache_enabled = cache_enabled
        
        logger.info(f"DataAcquisitionEngine initialized with cache at {self.data_dir}")
    
    def get_russell_3000_universe(self, as_of_date: Optional[datetime] = None) -> List[str]:
        """
        Get Russell 3000 constituents as of a specific date.
        
        Note: This is a simplified implementation. In production, use:
        - Norgate Data API for historical constituents
        - Bloomberg Terminal for accurate historical composition
        - FTSE Russell official data
        
        Args:
            as_of_date: Date for universe composition (default: current date)
            
        Returns:
            List of ticker symbols
        """
        if as_of_date is None:
            as_of_date = datetime.now()
        
        # TODO: Implement actual Russell 3000 API integration
        # For now, return an expanded sample universe (placeholder)
        logger.warning("Using sample universe. Implement actual Russell 3000 API for production.")
        
        # Expanded sample universe with 250+ liquid stocks across market caps
        sample_universe = [
            # Mega cap tech
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'AVGO', 'ORCL', 'ADBE',
            'CRM', 'CSCO', 'INTC', 'AMD', 'QCOM', 'TXN', 'AMAT', 'LRCX', 'KLAC', 'SNPS',
            'CDNS', 'MRVL', 'NXPI', 'MCHP', 'ANSS', 'INTU', 'PANW', 'CRWD', 'ZS', 'NET',
            
            # Financial services
            'BRK.B', 'JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'USB', 'PNC', 'TFC',
            'BK', 'STT', 'SCHW', 'AXP', 'COF', 'DFS', 'SYF', 'BLK', 'TROW', 'IVZ',
            'V', 'MA', 'PYPL', 'SQ', 'FIS', 'FISV', 'ADP', 'PAYX', 'BR', 'MMC',
            
            # Healthcare & Pharma
            'JNJ', 'UNH', 'LLY', 'ABBV', 'MRK', 'PFE', 'TMO', 'ABT', 'DHR', 'BMY',
            'AMGN', 'GILD', 'VRTX', 'REGN', 'BIIB', 'ISRG', 'CVS', 'CI', 'HUM', 'ELV',
            'MCK', 'COR', 'CAH', 'ZTS', 'DXCM', 'IDXX', 'IQV', 'SYK', 'BSX', 'EW',
            
            # Consumer discretionary
            'AMZN', 'TSLA', 'HD', 'MCD', 'NKE', 'SBUX', 'LOW', 'TJX', 'BKNG', 'ABNB',
            'MAR', 'GM', 'F', 'CMG', 'YUM', 'ROST', 'ORLY', 'AZO', 'DG', 'DLTR',
            
            # Consumer staples
            'WMT', 'PG', 'COST', 'KO', 'PEP', 'PM', 'MO', 'CL', 'MDLZ', 'GIS',
            'KHC', 'K', 'HSY', 'STZ', 'TAP', 'EL', 'CLX', 'SJM', 'CPB', 'CAG',
            
            # Energy
            'XOM', 'CVX', 'COP', 'SLB', 'EOG', 'MPC', 'PSX', 'VLO', 'OXY', 'HES',
            'HAL', 'BKR', 'FANG', 'DVN', 'PXD', 'MRO', 'APA', 'KMI', 'WMB', 'OKE',
            
            # Industrials
            'BA', 'HON', 'UNP', 'RTX', 'UPS', 'CAT', 'DE', 'LMT', 'GE', 'MMM',
            'GD', 'NOC', 'ITW', 'EMR', 'ETN', 'PH', 'FDX', 'CSX', 'NSC', 'WM',
            
            # Materials
            'LIN', 'APD', 'ECL', 'SHW', 'FCX', 'NEM', 'DD', 'DOW', 'PPG', 'NUE',
            'VMC', 'MLM', 'BALL', 'AVY', 'PKG', 'IP', 'EMN', 'CE', 'CF', 'MOS',
            
            # Real estate & Utilities
            'AMT', 'PLD', 'CCI', 'EQIX', 'PSA', 'SPG', 'O', 'WELL', 'DLR', 'AVB',
            'NEE', 'DUK', 'SO', 'D', 'AEP', 'SRE', 'EXC', 'XEL', 'ED', 'PEG',
            
            # Communication services
            'GOOGL', 'META', 'DIS', 'NFLX', 'CMCSA', 'VZ', 'T', 'TMUS', 'CHTR', 'ATVI',
            'EA', 'TTWO', 'MTCH', 'PARA', 'WBD', 'FOXA', 'NWSA', 'IPG', 'OMC', 'DISH',
            
            # Additional growth & mid-caps
            'SHOP', 'UBER', 'LYFT', 'SPOT', 'ROKU', 'SNAP', 'PINS', 'TWLO', 'DDOG', 'MDB',
            'SNOW', 'PLTR', 'COIN', 'SQ', 'RBLX', 'U', 'DASH', 'RIVN', 'LCID', 'NIO',
            'ZM', 'DOCU', 'OKTA', 'WDAY', 'NOW', 'TEAM', 'SPLK', 'ESTC', 'HUBS', 'ZI'
        ]
        
        return sample_universe
    
    def download_historical_data(
        self,
        tickers: List[str],
        start_date: datetime,
        end_date: datetime,
        interval: str = '1d'
    ) -> pd.DataFrame:
        """
        Download historical OHLCV data for given tickers.
        
        Args:
            tickers: List of ticker symbols
            start_date: Start date for historical data
            end_date: End date for historical data
            interval: Data interval (1d, 1h, etc.)
            
        Returns:
            DataFrame with multi-index (date, ticker) and OHLCV columns
        """
        cache_file = self.data_dir / f"cache_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.pkl"
        
        # Try to load from cache
        if self.cache_enabled and cache_file.exists():
            logger.info(f"Loading cached data from {cache_file}")
            return pd.read_pickle(cache_file)
        
        logger.info(f"Downloading data for {len(tickers)} tickers from {start_date} to {end_date}")
        
        all_data = []
        failed_tickers = []
        
        for ticker in tickers:
            try:
                df = yf.download(
                    ticker,
                    start=start_date,
                    end=end_date,
                    interval=interval,
                    progress=False,
                    auto_adjust=True  # Adjust for splits and dividends
                )
                
                if df.empty:
                    logger.warning(f"No data returned for {ticker}")
                    failed_tickers.append(ticker)
                    continue
                
                # Handle MultiIndex columns from yfinance
                if isinstance(df.columns, pd.MultiIndex):
                    # Flatten multi-index columns (e.g., ('Close', 'AAPL') -> 'Close')
                    df.columns = df.columns.get_level_values(0)
                
                df['ticker'] = ticker
                all_data.append(df)
                
            except Exception as e:
                logger.error(f"Failed to download {ticker}: {str(e)}")
                failed_tickers.append(ticker)
        
        if not all_data:
            raise ValueError("No data was successfully downloaded for any ticker")
        
        # Combine all data
        combined_df = pd.concat(all_data)
        combined_df = combined_df.reset_index()
        combined_df = combined_df.set_index(['Date', 'ticker'])
        combined_df = combined_df.sort_index()
        
        # Save to cache
        if self.cache_enabled:
            combined_df.to_pickle(cache_file)
            logger.info(f"Cached data saved to {cache_file}")
        
        if failed_tickers:
            logger.warning(f"Failed to download {len(failed_tickers)} tickers: {failed_tickers[:10]}...")
        
        return combined_df
    
    def get_volume_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract volume data from OHLCV dataframe.
        
        Args:
            df: OHLCV dataframe with multi-index (date, ticker)
            
        Returns:
            DataFrame with volume data
        """
        if 'Volume' not in df.columns:
            raise ValueError("Volume column not found in dataframe")
        
        return df[['Volume']].copy()
    
    def filter_liquid_stocks(
        self,
        df: pd.DataFrame,
        min_avg_volume: float = 1_000_000,
        min_price: float = 5.0,
        max_price: Optional[float] = None,
        lookback_days: int = 20
    ) -> List[str]:
        """
        Filter stocks based on liquidity criteria.
        
        Args:
            df: OHLCV dataframe
            min_avg_volume: Minimum average daily volume
            min_price: Minimum price (to filter penny stocks)
            max_price: Maximum price (optional)
            lookback_days: Period for calculating average volume
            
        Returns:
            List of tickers that meet liquidity criteria
        """
        # Get latest data for each ticker
        latest_date = df.index.get_level_values(0).max()
        start_date = latest_date - pd.Timedelta(days=lookback_days)
        
        recent_data = df.loc[(df.index.get_level_values(0) >= start_date) & 
                             (df.index.get_level_values(0) <= latest_date)]
        
        # Calculate average volume and price
        avg_volume = recent_data.groupby('ticker')['Volume'].mean()
        avg_close = recent_data.groupby('ticker')['Close'].mean()
        
        # Apply filters
        liquid_tickers = avg_volume[avg_volume >= min_avg_volume].index.tolist()
        price_filtered = avg_close[(avg_close >= min_price)]
        
        if max_price is not None:
            price_filtered = price_filtered[price_filtered <= max_price]
        
        valid_tickers = set(liquid_tickers).intersection(set(price_filtered.index.tolist()))
        
        logger.info(f"Filtered to {len(valid_tickers)} liquid stocks from {len(avg_volume)} total")
        
        return list(valid_tickers)
    
    def exclude_high_risk_stocks(
        self,
        tickers: List[str],
        exclude_sectors: Optional[List[str]] = None
    ) -> List[str]:
        """
        Exclude high-risk categories (biotech, meme stocks, etc.).
        
        Args:
            tickers: List of ticker symbols
            exclude_sectors: Sectors to exclude (e.g., ['Biotechnology'])
            
        Returns:
            Filtered list of tickers
        """
        if exclude_sectors is None:
            exclude_sectors = ['Biotechnology', 'Biopharmaceuticals']
        
        # TODO: Implement sector/industry lookup via yfinance or other API
        # For now, simple keyword-based exclusion
        excluded_keywords = ['BIOTECH', 'BIO', 'THERAPEUTICS', 'PHARMA']
        
        filtered_tickers = []
        for ticker in tickers:
            # Simple heuristic: exclude tickers with bio/pharma keywords
            if not any(keyword in ticker.upper() for keyword in excluded_keywords):
                filtered_tickers.append(ticker)
            else:
                logger.debug(f"Excluded {ticker} based on keyword filter")
        
        logger.info(f"Excluded {len(tickers) - len(filtered_tickers)} high-risk stocks")
        
        return filtered_tickers
    
    def validate_data_quality(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
        """
        Perform data quality checks and cleaning.
        
        Args:
            df: OHLCV dataframe
            
        Returns:
            Tuple of (cleaned_df, quality_report)
        """
        report = {
            'total_rows': len(df),
            'null_values': {},
            'outliers_removed': 0,
            'tickers_removed': []
        }
        
        # Check for null values
        for col in df.columns:
            null_count = df[col].isnull().sum()
            if null_count > 0:
                report['null_values'][col] = null_count
                logger.warning(f"Found {null_count} null values in {col}")
        
        # Identify which price columns exist
        price_columns = [col for col in ['Open', 'High', 'Low', 'Close'] if col in df.columns]
        
        if not price_columns:
            logger.error(f"No price columns found in dataframe. Available columns: {df.columns.tolist()}")
            raise ValueError(f"Missing price columns. Available: {df.columns.tolist()}")
        
        # Remove rows with null prices
        initial_len = len(df)
        df = df.dropna(subset=price_columns)
        report['outliers_removed'] = initial_len - len(df)
        
        # Check for unrealistic prices (negative, zero) only for columns that exist
        if 'Close' in df.columns and 'Open' in df.columns:
            invalid_prices = df[(df['Close'] <= 0) | (df['Open'] <= 0)]
            if len(invalid_prices) > 0:
                logger.warning(f"Found {len(invalid_prices)} rows with invalid prices")
                # Build condition for all price columns that exist
                condition = (df['Close'] > 0) & (df['Open'] > 0)
                if 'High' in df.columns:
                    condition &= (df['High'] > 0)
                if 'Low' in df.columns:
                    condition &= (df['Low'] > 0)
                df = df[condition]
        
        # Check for unrealistic price movements (>50% in one day)
        df['price_change_pct'] = df.groupby('ticker')['Close'].pct_change()
        extreme_moves = df[abs(df['price_change_pct']) > 0.5]
        
        if len(extreme_moves) > 0:
            logger.warning(f"Found {len(extreme_moves)} extreme price movements (>50%)")
            # Don't remove these automatically - could be legitimate (splits, etc.)
        
        df = df.drop('price_change_pct', axis=1)
        
        logger.info(f"Data validation complete. Report: {report}")
        
        return df, report
    
    def get_training_data(
        self,
        tickers: List[str],
        start_date: datetime,
        end_date: datetime,
        apply_filters: bool = True
    ) -> pd.DataFrame:
        """
        Get cleaned and validated training data.
        
        Args:
            tickers: List of ticker symbols
            start_date: Start date
            end_date: End date
            apply_filters: Whether to apply liquidity and risk filters
            
        Returns:
            Clean OHLCV dataframe ready for feature engineering
        """
        # Download raw data
        df = self.download_historical_data(tickers, start_date, end_date)
        
        # Validate data quality
        df, quality_report = self.validate_data_quality(df)
        
        # Apply filters if requested
        if apply_filters:
            liquid_tickers = self.filter_liquid_stocks(df)
            safe_tickers = self.exclude_high_risk_stocks(liquid_tickers)
            
            # Filter dataframe to only include valid tickers
            df = df[df.index.get_level_values('ticker').isin(safe_tickers)]
            
            logger.info(f"Final dataset: {len(safe_tickers)} tickers, {len(df)} rows")
        
        return df


if __name__ == "__main__":
    # Example usage
    engine = DataAcquisitionEngine()
    
    # Get universe
    universe = engine.get_russell_3000_universe()
    print(f"Universe size: {len(universe)}")
    
    # Download data
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365 * 2)  # 2 years
    
    df = engine.get_training_data(
        tickers=universe,
        start_date=start_date,
        end_date=end_date,
        apply_filters=True
    )
    
    print(f"\nData shape: {df.shape}")
    print(f"\nColumns: {df.columns.tolist()}")
    print(f"\nDate range: {df.index.get_level_values(0).min()} to {df.index.get_level_values(0).max()}")
    print(f"\nSample data:\n{df.head()}")
