"""
Data Acquisition for Crypto Macro-Fundamental Strategy

Fetches cryptocurrency prices, stablecoin market caps, treasury yields, and VIX data.
"""

import pandas as pd
import numpy as np
import yaml
import os
import warnings
warnings.filterwarnings('ignore')

from typing import Dict, Optional
from datetime import datetime, timedelta

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

try:
    import yfinance as yf
except ImportError:
    print("Warning: yfinance not installed")

try:
    from fredapi import Fred
except ImportError:
    print("Warning: fredapi not installed")

try:
    import requests
except ImportError:
    print("Warning: requests not installed")


class DataAcquisition:
    """Handles data fetching for crypto macro-fundamental strategy."""
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize with configuration."""
        if load_dotenv is not None:
            load_dotenv()

        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.start_date = self.config['data']['start_date']
        self.end_date = self.config['data']['end_date']
        self.binance_api_key = os.getenv('BINANCE_API_KEY')
        self.fred_api_key = os.getenv('FRED_API_KEY')
    
    def fetch_crypto_prices(self) -> pd.DataFrame:
        """
        Fetch Bitcoin and Ethereum prices from Binance (fallback: Yahoo Finance).
        
        Returns:
            DataFrame with BTC-USD and ETH-USD prices
        """
        symbols = [self.config['assets']['primary']] + self.config['assets']['secondary']
        
        prices = {}
        
        for symbol in symbols:
            print(f"Fetching {symbol}...")
            
            try:
                series = self._fetch_symbol_price_series(symbol)
                if series is not None and not series.empty:
                    prices[symbol] = series
                else:
                    print(f"Warning: No close price column found for {symbol}. Using placeholder data.")
                    prices[symbol] = self._generate_placeholder_asset_price(symbol)
            
            except Exception as e:
                print(f"Error fetching {symbol}: {e}")
                print(f"Generating placeholder data for {symbol}...")
                prices[symbol] = self._generate_placeholder_asset_price(symbol)
        
        # Combine into single DataFrame
        if not prices:
            # Ensure downstream alignment always has a valid date index.
            for symbol in symbols:
                prices[symbol] = self._generate_placeholder_asset_price(symbol)

        df = pd.DataFrame(prices)
        df = df.sort_index()
        df = df.ffill()  # Forward fill missing values
        
        print(f"Fetched crypto prices: {len(df)} days")
        return df
    
    def fetch_treasury_yield(self, api_key: Optional[str] = None) -> pd.Series:
        """
        Fetch US 2-Year Treasury Yield from FRED.
        
        Args:
            api_key: FRED API key (or set FRED_API_KEY env var)
        
        Returns:
            Series with daily treasury yields
        """
        try:
            if 'Fred' not in globals():
                raise ImportError("fredapi is not installed")

            resolved_api_key = api_key or self.fred_api_key
            fred = Fred(api_key=resolved_api_key)
            
            print("Fetching US 2-Year Treasury Yield (DGS2)...")
            
            series = fred.get_series(
                'DGS2',
                observation_start=self.start_date,
                observation_end=self.end_date
            )
            
            # Convert to numeric and handle missing values
            series = pd.to_numeric(series, errors='coerce')
            series = series.ffill()  # Forward fill weekends/holidays
            
            print(f"Fetched treasury yields: {len(series)} days")
            return series
        
        except Exception as e:
            print(f"Error fetching treasury yield: {e}")
            print("Generating placeholder data...")
            return self._generate_placeholder_treasury()
    
    def fetch_vix(self) -> pd.Series:
        """
        Fetch VIX Index from Yahoo Finance.
        
        Returns:
            Series with daily VIX closing values
        """
        print("Fetching VIX Index...")
        
        try:
            data = yf.download(
                '^VIX',
                start=self.start_date,
                end=self.end_date,
                progress=False
            )
            
            vix = self._extract_close_series(data, '^VIX')
            if vix is None or vix.empty:
                raise ValueError("No close price column found for ^VIX")
            vix = vix.ffill()
            
            print(f"Fetched VIX: {len(vix)} days")
            return vix
        
        except Exception as e:
            print(f"Error fetching VIX: {e}")
            print("Generating placeholder data...")
            return self._generate_placeholder_vix()
    
    def fetch_stablecoin_market_caps(self) -> pd.DataFrame:
        """
        Fetch stablecoin market capitalizations.
        
        Note: This is a placeholder. In production, use CoinMarketCap API
        or scrape from coinmarketcap.com
        
        Returns:
            DataFrame with stablecoin market caps by date
        """
        print("Fetching stablecoin market caps...")
        print("Warning: Using placeholder data. Implement CoinMarketCap API for production.")
        
        # Generate placeholder data
        return self._generate_placeholder_stablecoin_mcap()
    
    def fetch_crypto_total_market_cap(self) -> pd.Series:
        """
        Fetch total cryptocurrency market capitalization.
        
        Note: Placeholder. Use CoinMarketCap API in production.
        
        Returns:
            Series with total crypto market cap
        """
        print("Fetching total crypto market cap...")
        print("Warning: Using placeholder data. Implement CoinMarketCap API for production.")
        
        return self._generate_placeholder_crypto_mcap()
    
    def fetch_institutional_events(self) -> pd.DataFrame:
        """
        Load institutional event calendar.
        
        Returns:
            DataFrame with date, description, and impact
        """
        events = self.config['features']['institutional_validation']['events']
        
        df = pd.DataFrame(events)
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date')
        
        print(f"Loaded {len(df)} institutional events")
        return df
    
    def fetch_full_dataset(self) -> Dict[str, pd.DataFrame]:
        """
        Fetch complete dataset.
        
        Returns:
            Dict with all data components
        """
        # Fetch crypto prices
        crypto_prices = self.fetch_crypto_prices()
        
        # Fetch treasury yield
        treasury = self.fetch_treasury_yield()
        
        # Fetch VIX
        vix = self.fetch_vix()
        
        # Fetch stablecoin market caps
        stablecoin_mcap = self.fetch_stablecoin_market_caps()
        
        # Fetch total crypto market cap
        crypto_mcap = self.fetch_crypto_total_market_cap()
        
        # Fetch institutional events
        events = self.fetch_institutional_events()
        
        if crypto_prices.empty:
            raise ValueError("Crypto prices are empty after fetch and fallback.")

        # Align all data to common index (crypto prices dates)
        aligned_data = pd.DataFrame(index=crypto_prices.index)
        
        # Add crypto prices
        for col in crypto_prices.columns:
            aligned_data[col] = crypto_prices[col]
        
        # Add treasury yield
        aligned_data['DGS2'] = treasury.reindex(aligned_data.index, method='ffill')
        
        # Add VIX
        aligned_data['VIX'] = vix.reindex(aligned_data.index, method='ffill')
        
        # Add stablecoin market caps
        for col in stablecoin_mcap.columns:
            aligned_data[col] = stablecoin_mcap[col].reindex(aligned_data.index, method='ffill')
        
        # Add total crypto market cap
        aligned_data['Total_Crypto_MCap'] = crypto_mcap.reindex(aligned_data.index, method='ffill')
        
        # Drop any remaining NaN rows
        aligned_data = aligned_data.dropna()
        
        print(f"\nFinal aligned dataset: {len(aligned_data)} days")
        print(f"Date range: {aligned_data.index[0]} to {aligned_data.index[-1]}")
        
        return {
            'prices': aligned_data,
            'events': events
        }
    
    def _extract_close_series(self, data: pd.DataFrame, symbol: str) -> Optional[pd.Series]:
        """Extract adjusted/regular close from yfinance output across column formats."""
        if data is None or data.empty:
            return None

        preferred_fields = ('Adj Close', 'Close')

        if isinstance(data.columns, pd.MultiIndex):
            level0 = data.columns.get_level_values(0)
            for field in preferred_fields:
                if field in level0:
                    subset = data[field]
                    if isinstance(subset, pd.DataFrame):
                        if symbol in subset.columns:
                            series = subset[symbol]
                        else:
                            series = subset.iloc[:, 0]
                    else:
                        series = subset
                    return pd.to_numeric(series, errors='coerce').rename(symbol)
            return None

        for field in preferred_fields:
            if field in data.columns:
                return pd.to_numeric(data[field], errors='coerce').rename(symbol)

        if len(data.columns) == 1:
            return pd.to_numeric(data.iloc[:, 0], errors='coerce').rename(symbol)

        return None

    def _fetch_symbol_price_series(self, symbol: str) -> Optional[pd.Series]:
        """Fetch symbol close series from Binance first, then fallback to Yahoo Finance."""
        pair = self._to_binance_pair(symbol)

        if pair is not None:
            binance_series = self._fetch_binance_daily_close(pair, symbol)
            if binance_series is not None and not binance_series.empty:
                return binance_series

        data = yf.download(
            symbol,
            start=self.start_date,
            end=self.end_date,
            progress=False
        )
        return self._extract_close_series(data, symbol)

    def _to_binance_pair(self, symbol: str) -> Optional[str]:
        """Map project symbols to Binance spot pairs."""
        mapping = {
            'BTC-USD': 'BTCUSDT',
            'ETH-USD': 'ETHUSDT',
            'BTC-USDT': 'BTCUSDT',
            'ETH-USDT': 'ETHUSDT',
        }
        return mapping.get(symbol)

    def _fetch_binance_daily_close(self, pair: str, output_name: str) -> Optional[pd.Series]:
        """Fetch daily close prices from Binance klines endpoint with pagination."""
        if 'requests' not in globals():
            return None

        print(f"Fetching {output_name} from Binance ({pair})...")

        base_url = 'https://api.binance.com/api/v3/klines'
        start_ts = int(pd.Timestamp(self.start_date, tz='UTC').timestamp() * 1000)
        end_ts = int((pd.Timestamp(self.end_date, tz='UTC') + pd.Timedelta(days=1)).timestamp() * 1000) - 1

        all_rows = []
        cursor = start_ts
        headers = {'X-MBX-APIKEY': self.binance_api_key} if self.binance_api_key else {}

        while cursor <= end_ts:
            params = {
                'symbol': pair,
                'interval': '1d',
                'startTime': cursor,
                'endTime': end_ts,
                'limit': 1000,
            }

            response = requests.get(base_url, params=params, headers=headers, timeout=30)
            response.raise_for_status()
            rows = response.json()

            if not rows:
                break

            all_rows.extend(rows)
            last_open_time = int(rows[-1][0])
            next_cursor = last_open_time + 86_400_000

            if next_cursor <= cursor:
                break
            cursor = next_cursor

            if len(rows) < 1000:
                break

        if not all_rows:
            return None

        df = pd.DataFrame(all_rows)
        series = pd.to_numeric(df[4], errors='coerce')
        index = pd.to_datetime(df[0].astype('int64'), unit='ms', utc=True).tz_localize(None)
        series.index = index
        series.name = output_name

        return series.sort_index()

    # Placeholder data generators (for testing without API keys)

    def _generate_placeholder_asset_price(self, symbol: str) -> pd.Series:
        """Generate synthetic daily price path for BTC/ETH when market data is unavailable."""
        dates = pd.date_range(self.start_date, self.end_date, freq='D')

        seed_map = {
            'BTC-USD': 101,
            'ETH-USD': 102,
        }
        start_map = {
            'BTC-USD': 9000.0,
            'ETH-USD': 700.0,
        }
        drift_map = {
            'BTC-USD': 0.00045,
            'ETH-USD': 0.00055,
        }
        vol_map = {
            'BTC-USD': 0.035,
            'ETH-USD': 0.045,
        }

        np.random.seed(seed_map.get(symbol, 199))
        n = len(dates)
        drift = drift_map.get(symbol, 0.0004)
        vol = vol_map.get(symbol, 0.03)
        log_returns = drift + np.random.normal(0, vol, n)

        price = np.empty(n)
        price[0] = start_map.get(symbol, 1000.0)
        for i in range(1, n):
            price[i] = max(price[i - 1] * np.exp(log_returns[i]), 1.0)

        # Inject known drawdown windows so stress tests remain informative.
        dates_index = pd.DatetimeIndex(dates)
        covid_mask = (dates_index >= '2020-03-01') & (dates_index <= '2020-03-31')
        bear_2022_mask = (dates_index >= '2022-01-01') & (dates_index <= '2022-12-31')
        ftx_mask = (dates_index >= '2022-11-01') & (dates_index <= '2022-11-30')

        price[covid_mask] *= 0.78
        price[bear_2022_mask] *= 0.70
        price[ftx_mask] *= 0.82

        return pd.Series(price, index=dates, name=symbol)
    
    def _generate_placeholder_treasury(self) -> pd.Series:
        """Generate synthetic treasury yield data."""
        dates = pd.date_range(self.start_date, self.end_date, freq='D')
        
        # Simulate treasury yield: trending upward in 2022 (Fed tightening)
        np.random.seed(42)
        base = 1.5  # Start around 1.5%
        trend = np.linspace(0, 3.0, len(dates))  # Rise to ~4.5% by end
        noise = np.random.normal(0, 0.1, len(dates))
        
        yield_data = base + trend + noise
        
        return pd.Series(yield_data, index=dates, name='DGS2')
    
    def _generate_placeholder_vix(self) -> pd.Series:
        """Generate synthetic VIX data."""
        dates = pd.date_range(self.start_date, self.end_date, freq='D')
        
        # Simulate VIX: spikes during crisis periods
        np.random.seed(43)
        base = 15  # Normal VIX around 15
        noise = np.random.normal(0, 3, len(dates))
        
        vix_data = base + noise
        
        # Add spikes for known crisis periods
        # March 2020 COVID
        covid_mask = (dates >= '2020-03-01') & (dates <= '2020-03-31')
        vix_data[covid_mask] += 30
        
        # 2022 volatility
        volatility_2022 = (dates >= '2022-01-01') & (dates <= '2022-12-31')
        vix_data[volatility_2022] += 10
        
        # FTX collapse
        ftx_mask = (dates >= '2022-11-01') & (dates <= '2022-11-30')
        vix_data[ftx_mask] += 15
        
        return pd.Series(vix_data, index=dates, name='VIX')
    
    def _generate_placeholder_stablecoin_mcap(self) -> pd.DataFrame:
        """Generate synthetic stablecoin market cap data."""
        dates = pd.date_range(self.start_date, self.end_date, freq='D')
        
        np.random.seed(44)
        
        # USDT: Largest, growing
        usdt = np.linspace(10e9, 80e9, len(dates)) + np.random.normal(0, 1e9, len(dates))
        
        # USDC: Second largest
        usdc = np.linspace(5e9, 50e9, len(dates)) + np.random.normal(0, 0.5e9, len(dates))
        
        # BUSD: Smaller
        busd = np.linspace(2e9, 20e9, len(dates)) + np.random.normal(0, 0.3e9, len(dates))
        
        # DAI: DeFi stablecoin
        dai = np.linspace(1e9, 10e9, len(dates)) + np.random.normal(0, 0.2e9, len(dates))
        
        # Spikes during crisis (capital fleeing to stablecoins)
        covid_mask = (dates >= '2020-03-01') & (dates <= '2020-03-31')
        usdt[covid_mask] *= 1.2
        usdc[covid_mask] *= 1.3
        
        ftx_mask = (dates >= '2022-11-01') & (dates <= '2022-11-30')
        usdc[ftx_mask] *= 1.5  # Flight to quality
        
        df = pd.DataFrame({
            'USDT_MCap': usdt,
            'USDC_MCap': usdc,
            'BUSD_MCap': busd,
            'DAI_MCap': dai
        }, index=dates)
        
        # Total stablecoin market cap
        df['Total_Stablecoin_MCap'] = df.sum(axis=1)
        
        return df
    
    def _generate_placeholder_crypto_mcap(self) -> pd.Series:
        """Generate synthetic total crypto market cap."""
        dates = pd.date_range(self.start_date, self.end_date, freq='D')
        
        np.random.seed(45)
        
        # Simulate crypto market: volatile, bull run 2020-2021, crash 2022
        base = 200e9  # $200B start
        
        # Create trend
        trend = np.zeros(len(dates))
        
        for i, date in enumerate(dates):
            if date < pd.Timestamp('2020-03-01'):
                trend[i] = 0
            elif date < pd.Timestamp('2021-11-01'):
                # Bull run to $3T
                progress = (date - pd.Timestamp('2020-03-01')).days / 600
                trend[i] = progress * 2800e9
            elif date < pd.Timestamp('2023-01-01'):
                # Crash back to $1T
                progress = (date - pd.Timestamp('2021-11-01')).days / 425
                trend[i] = 2800e9 - progress * 1800e9
            else:
                # Recovery
                trend[i] = 1000e9
        
        noise = np.random.normal(0, 50e9, len(dates))
        
        mcap = base + trend + noise
        mcap = np.maximum(mcap, 100e9)  # Floor at $100B
        
        return pd.Series(mcap, index=dates, name='Total_Crypto_MCap')


# Test code
if __name__ == "__main__":
    data_acq = DataAcquisition('config.yaml')
    
    dataset = data_acq.fetch_full_dataset()
    
    print(f"\nDataset shape: {dataset['prices'].shape}")
    print(f"\nColumns: {list(dataset['prices'].columns)}")
    print(f"\nFirst few rows:")
    print(dataset['prices'].head())
    
    print(f"\nEvents:")
    print(dataset['events'])
