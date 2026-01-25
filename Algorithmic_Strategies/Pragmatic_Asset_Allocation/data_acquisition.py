"""
Pragmatic Asset Allocation Model - Data Acquisition Module
Fetches historical price data for risky assets, hedging assets, and macroeconomic indicators.
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import requests
import json
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import yaml
import logging

warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PragmaticAssetAllocationData:
    """
    Data acquisition for Pragmatic Asset Allocation Model.
    Handles fetching of asset prices and macroeconomic data.
    """

    def __init__(self, config_path: str = "config.yaml"):
        """Initialize with configuration."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.assets = self.config['assets']
        self.data_dir = Path("data")
        self.data_dir.mkdir(exist_ok=True)

        # Asset ticker mappings for historical data
        self.ticker_mappings = {
            # Risky assets - use appropriate historical proxies
            "QQQ": "QQQ",      # NASDAQ 100 ETF (available from 1999)
            "URTH": "URTH",    # MSCI World ETF (available from 2012, use SPY for earlier)
            "EEM": "EEM",      # MSCI EM ETF (available from 2003)

            # Hedging assets
            "IEF": "IEF",      # 7-10 Year Treasury ETF (available from 2002)
            "GLD": "GLD",      # Gold ETF (available from 2004)

            # Additional proxies for long-term data
            "NASDAQ-100": "^NDX",
            "MSCI-World": "SPY",    # S&P 500 as world proxy pre-1980s
            "MSCI-EM": "EEM",
            "US-10Y-Treasury": "^TNX",
            "GOLD": "GC=F"
        }

    def fetch_asset_data(self, ticker: str, start_date: str, end_date: str,
                        use_proxy: bool = True) -> pd.DataFrame:
        """
        Fetch historical price data for a single asset.

        Args:
            ticker: Asset ticker symbol
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            use_proxy: Whether to use historical proxy if direct data unavailable

        Returns:
            DataFrame with OHLCV data
        """
        try:
            # Try direct ticker first
            data = yf.download(ticker, start=start_date, end=end_date, progress=False, auto_adjust=True)

            if data.empty and use_proxy:
                # Try proxy ticker for historical data
                proxy_ticker = self.ticker_mappings.get(ticker, ticker)
                if proxy_ticker != ticker:
                    logger.info(f"Trying proxy ticker {proxy_ticker} for {ticker}")
                    data = yf.download(proxy_ticker, start=start_date, end=end_date, progress=False, auto_adjust=True)

            if data.empty:
                logger.warning(f"No data found for {ticker}")
                return pd.DataFrame()

            # Clean and format data - handle MultiIndex columns
            data = data.dropna()
            data.index = pd.to_datetime(data.index)

            # Flatten MultiIndex columns if they exist
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.droplevel(1)

            # Calculate returns using Close price (already adjusted)
            if 'Close' in data.columns:
                data['Returns'] = data['Close'].pct_change()
                data['Total_Return'] = (1 + data['Returns']).cumprod()

            return data

        except Exception as e:
            logger.error(f"Error fetching data for {ticker}: {str(e)}")
            return pd.DataFrame()

    def fetch_macroeconomic_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetch macroeconomic data including yield curve spreads.

        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)

        Returns:
            DataFrame with macroeconomic indicators
        """
        try:
            # FRED API for Treasury yields (if available)
            # For now, use yfinance proxies
            short_term_proxy = "^IRX"  # 13-week T-bill rate
            long_term_proxy = "^TNX"   # 10-year treasury yield

            short_term_data = yf.download(short_term_proxy, start=start_date, end=end_date, progress=False, auto_adjust=True)
            long_term_data = yf.download(long_term_proxy, start=start_date, end=end_date, progress=False, auto_adjust=True)

            if short_term_data.empty or long_term_data.empty:
                logger.warning("Macroeconomic data fetch failed")
                return pd.DataFrame()

            # Handle MultiIndex columns
            if isinstance(short_term_data.columns, pd.MultiIndex):
                short_term_data.columns = short_term_data.columns.droplevel(1)
            if isinstance(long_term_data.columns, pd.MultiIndex):
                long_term_data.columns = long_term_data.columns.droplevel(1)

            # Combine and calculate spread
            macro_data = pd.DataFrame(index=short_term_data.index)
            macro_data['3M_Treasury_Yield'] = short_term_data['Close'] / 100  # Convert to decimal
            macro_data['10Y_Treasury_Yield'] = long_term_data['Close'] / 100
            macro_data['Yield_Curve_Spread'] = macro_data['10Y_Treasury_Yield'] - macro_data['3M_Treasury_Yield']
            macro_data['Yield_Curve_Inverted'] = macro_data['Yield_Curve_Spread'] < 0

            return macro_data.dropna()

        except Exception as e:
            logger.error(f"Error fetching macroeconomic data: {str(e)}")
            return pd.DataFrame()

    def calculate_momentum_signals(self, price_data: pd.DataFrame,
                                 lookback_months: int = 12) -> pd.DataFrame:
        """
        Calculate momentum signals for assets.

        Args:
            price_data: DataFrame with price data
            lookback_months: Lookback period for momentum calculation

        Returns:
            DataFrame with momentum rankings
        """
        try:
            # Calculate rolling returns
            lookback_days = lookback_months * 21  # Approximate trading days per month

            momentum_data = pd.DataFrame(index=price_data.index)

            for asset in price_data.columns.levels[0]:
                asset_prices = price_data[asset]['Close']
                momentum_data[f'{asset}_Momentum'] = (
                    asset_prices / asset_prices.shift(lookback_days) - 1
                )

            # Calculate rankings (higher momentum = better rank)
            momentum_cols = [col for col in momentum_data.columns if 'Momentum' in col]
            rankings = momentum_data[momentum_cols].rank(axis=1, ascending=False, method='dense')

            # Add ranking columns to the data
            for col in rankings.columns:
                momentum_data[col.replace('Momentum', 'Rank')] = rankings[col]

            return momentum_data

        except Exception as e:
            logger.error(f"Error calculating momentum signals: {str(e)}")
            return pd.DataFrame()

    def calculate_trend_signals(self, price_data: pd.DataFrame,
                               lookback_months: int = 12) -> pd.DataFrame:
        """
        Calculate trend filter signals using moving averages.

        Args:
            price_data: DataFrame with price data
            lookback_months: Lookback period for moving average

        Returns:
            DataFrame with trend signals
        """
        try:
            lookback_days = lookback_months * 21
            trend_data = pd.DataFrame(index=price_data.index)

            for asset in price_data.columns.levels[0]:
                asset_prices = price_data[asset]['Close']
                ma = asset_prices.rolling(window=lookback_days).mean()
                trend_data[f'{asset}_SMA_{lookback_months}M'] = ma
                trend_data[f'{asset}_Trend_Up'] = asset_prices > ma

            return trend_data

        except Exception as e:
            logger.error(f"Error calculating trend signals: {str(e)}")
            return pd.DataFrame()

    def fetch_all_data(self, start_date: str = None, end_date: str = None) -> Dict[str, pd.DataFrame]:
        """
        Fetch all required data for the strategy.

        Args:
            start_date: Start date (defaults to config)
            end_date: End date (defaults to config)

        Returns:
            Dictionary containing all dataframes
        """
        if start_date is None:
            start_date = self.config['backtest']['start_date']
        if end_date is None:
            end_date = self.config['backtest']['end_date']

        logger.info(f"Fetching data from {start_date} to {end_date}")

        all_data = {}

        # Fetch risky assets
        risky_tickers = [asset['ticker'] for asset in self.assets['risky']]
        risky_data = {}
        for ticker in risky_tickers:
            logger.info(f"Fetching {ticker}...")
            data = self.fetch_asset_data(ticker, start_date, end_date)
            if not data.empty:
                risky_data[ticker] = data

        if risky_data:
            all_data['risky_assets'] = pd.concat(risky_data, axis=1, keys=risky_data.keys())

        # Fetch hedging assets
        hedging_tickers = [asset['ticker'] for asset in self.assets['hedging']]
        hedging_data = {}
        for ticker in hedging_tickers:
            logger.info(f"Fetching {ticker}...")
            data = self.fetch_asset_data(ticker, start_date, end_date)
            if not data.empty:
                hedging_data[ticker] = data

        if hedging_data:
            all_data['hedging_assets'] = pd.concat(hedging_data, axis=1, keys=hedging_data.keys())

        # Fetch macroeconomic data
        logger.info("Fetching macroeconomic data...")
        macro_data = self.fetch_macroeconomic_data(start_date, end_date)
        if not macro_data.empty:
            all_data['macroeconomic'] = macro_data

        # Calculate signals
        if 'risky_assets' in all_data:
            logger.info("Calculating momentum signals...")
            momentum_signals = self.calculate_momentum_signals(
                all_data['risky_assets'],
                self.config['signals']['momentum']['lookback_months']
            )
            all_data['momentum_signals'] = momentum_signals

            logger.info("Calculating trend signals...")
            trend_signals = self.calculate_trend_signals(
                all_data['risky_assets'],
                self.config['signals']['trend_filter']['lookback_months']
            )
            all_data['trend_signals'] = trend_signals

        # Save data
        self.save_data(all_data)

        logger.info("Data acquisition complete")
        return all_data

    def save_data(self, data_dict: Dict[str, pd.DataFrame]) -> None:
        """Save fetched data to disk."""
        for data_type, df in data_dict.items():
            if not df.empty:
                filename = f"{data_type}_{datetime.now().strftime('%Y%m%d')}.csv"
                filepath = self.data_dir / filename
                df.to_csv(filepath)
                logger.info(f"Saved {data_type} data to {filepath}")

    def load_cached_data(self) -> Dict[str, pd.DataFrame]:
        """Load previously cached data if available."""
        cached_data = {}
        for file in self.data_dir.glob("*.csv"):
            data_type = file.stem.split('_')[0]
            cached_data[data_type] = pd.read_csv(file, index_col=0, parse_dates=True)

        return cached_data

    def validate_data_quality(self, data_dict: Dict[str, pd.DataFrame]) -> Dict[str, bool]:
        """
        Validate data quality and completeness.

        Args:
            data_dict: Dictionary of dataframes to validate

        Returns:
            Dictionary of validation results
        """
        validation_results = {}

        for data_type, df in data_dict.items():
            if df.empty:
                validation_results[data_type] = False
                continue

            # Check completeness
            completeness = df.dropna().shape[0] / df.shape[0]
            threshold = self.config['data']['quality_checks']['data_completeness_threshold']

            # Check for outliers (simple z-score method)
            if 'Close' in str(df.columns):
                # Calculate z-scores for price data
                price_cols = [col for col in df.columns if 'Close' in str(col)]
                for col in price_cols:
                    try:
                        z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
                        outlier_pct = (z_scores > 3).sum() / len(z_scores)
                        if outlier_pct > 0.05:  # More than 5% outliers
                            logger.warning(f"High outlier percentage in {col}: {outlier_pct:.1%}")
                    except:
                        pass

            validation_results[data_type] = completeness >= threshold

        return validation_results