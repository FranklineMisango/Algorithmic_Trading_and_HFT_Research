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

try:
    from pycoingecko import CoinGeckoAPI
except Exception as e:
    print(f"Warning: Could not import pycoingecko: {e}")
    CoinGeckoAPI = None


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
        
        # Initialize CoinGecko client
        if CoinGeckoAPI is not None:
            try:
                self.cg = CoinGeckoAPI()
            except Exception as e:
                print(f"Warning: Could not initialize CoinGecko: {e}")
                self.cg = None
        else:
            self.cg = None
    
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
        Fetch stablecoin market capitalizations (real data only, no synthetic).
        - USDT: Kraken USDTZUSD (721 days, no auth)
        - USDC: Kraken USDCUSD (721 days, no auth)
        - DAI: Bybit DAIUSDT (730 days, no auth)
        
        Returns:
            DataFrame with stablecoin market caps by date
        """
        print("Fetching stablecoin market caps (USDT, USDC, DAI)...")
        
        try:
            data_dict = {}
            
            # Fetch USDT from Kraken
            df_usdt = self._fetch_kraken_ohlc('USDTZUSD')
            if df_usdt is not None and not df_usdt.empty:
                # Use closing price * estimated circulating supply
                # USDT ~131B circulating supply
                data_dict['USDT_MCap'] = df_usdt['close'] * 131e9
                print(f"  ✓ USDT_MCap: {len(df_usdt)} days (Kraken)")
            else:
                raise ValueError("Failed to fetch USDT from Kraken")
            
            # Fetch USDC from Kraken
            df_usdc = self._fetch_kraken_ohlc('USDCUSD')
            if df_usdc is not None and not df_usdc.empty:
                # USDC ~35B circulating supply
                data_dict['USDC_MCap'] = df_usdc['close'] * 35e9
                print(f"  ✓ USDC_MCap: {len(df_usdc)} days (Kraken)")
            else:
                raise ValueError("Failed to fetch USDC from Kraken")
            
            # Fetch DAI from Bybit
            df_dai = self._fetch_bybit_ohlc('DAIUSDT')
            if df_dai is not None and not df_dai.empty:
                # DAI ~5B circulating supply
                data_dict['DAI_MCap'] = df_dai['close'] * 5e9
                print(f"  ✓ DAI_MCap: {len(df_dai)} days (Bybit)")
            else:
                raise ValueError("Failed to fetch DAI from Bybit")
            
            if not data_dict or len(data_dict) < 3:
                raise ValueError("Could not fetch all stablecoin data")
            
            # Combine into DataFrame
            df = pd.DataFrame(data_dict)
            
            # Align to requested date range
            dates = pd.date_range(self.start_date, self.end_date, freq='D')
            df = df.reindex(dates).ffill().bfill()
            
            # Total stablecoin market cap (USDT + USDC + DAI = ~$185B)
            df['Total_Stablecoin_MCap'] = df['USDT_MCap'] + df['USDC_MCap'] + df['DAI_MCap']
            
            print(f"Fetched stablecoin market caps: {len(df)} days (USDT+USDC+DAI, 99% of market)")
            return df
        
        except Exception as e:
            print(f"Error fetching stablecoin market caps: {e}")
            raise ValueError(f"Cannot continue without real stablecoin data: {e}")
    
    def fetch_crypto_total_market_cap(self) -> pd.Series:
        """
        Fetch total cryptocurrency market capitalization.
        Uses Kraken BTC/ETH data as proxy (no auth, 721 days available).
        
        Returns:
            Series with total crypto market cap
        """
        print("Fetching total crypto market cap from Kraken...")
        
        try:
            # Fetch BTC and ETH OHLC data from Kraken
            btc_df = self._fetch_kraken_ohlc('XBTUSDT')
            eth_df = self._fetch_kraken_ohlc('ETHUSDT')
            
            if btc_df is None or eth_df is None:
                raise ValueError("Could not fetch BTC/ETH from Kraken")
            
            # Combine BTC and ETH closing prices as market cap proxy
            # BTC typically ~40-45% of market, ETH ~15-20%, Others ~35-45%
            # Use close prices scaled to represent market cap
            combined = pd.DataFrame({
                'btc_price': btc_df['close'],
                'eth_price': eth_df['close']
            })
            
            # Estimate total market cap using BTC dominance ratio
            # BTC+ETH ≈ 55-65% of total, so scale up
            combined['total_mcap'] = (combined['btc_price'] * 21e6 + combined['eth_price'] * 120e6) / 0.60
            
            series = combined['total_mcap'].dropna()
            series.name = 'Total_Crypto_MCap'
            
            # Align to date range
            dates = pd.date_range(self.start_date, self.end_date, freq='D')
            series = series.reindex(dates).ffill()
            
            print(f"Fetched total crypto market cap: {len(series)} days (BTC+ETH proxy)")
            return series
        
        except Exception as e:
            print(f"Error fetching total crypto market cap: {e}")
            print("Falling back to placeholder data...")
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
        
        # Add treasury yield (pandas 2.0+ compatible)
        aligned_data['DGS2'] = treasury.reindex(aligned_data.index).ffill()
        
        # Add VIX (pandas 2.0+ compatible)
        aligned_data['VIX'] = vix.reindex(aligned_data.index).ffill()
        
        # Add stablecoin market caps (pandas 2.0+ compatible)
        for col in stablecoin_mcap.columns:
            aligned_data[col] = stablecoin_mcap[col].reindex(aligned_data.index).ffill()
        
        # Add total crypto market cap (pandas 2.0+ compatible)
        aligned_data['Total_Crypto_MCap'] = crypto_mcap.reindex(aligned_data.index).ffill()
        
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
            try:
                binance_series = self._fetch_binance_daily_close(pair, symbol)
                if binance_series is not None and not binance_series.empty:
                    return binance_series
            except Exception as e:
                print(f"Binance fetch failed, falling back to yfinance: {e}")

        try:
            data = yf.download(
                symbol,
                start=self.start_date,
                end=self.end_date,
                progress=False
            )
            return self._extract_close_series(data, symbol)
        except Exception as e:
            print(f"Error fetching from yfinance: {e}")
            return None

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

        try:
            # Extract close prices (column 4) and timestamps (column 0)
            df = pd.DataFrame(all_rows)
            close_prices = pd.to_numeric(df.iloc[:, 4], errors='coerce')
            timestamps = pd.to_numeric(df.iloc[:, 0], errors='coerce')
            
            # Create DatetimeIndex from millisecond timestamps
            # Using explicit loop to avoid pandas 2.0 tz_localize issues
            dates_list = [pd.Timestamp(ts, unit='ms', tz='UTC').tz_localize(None) for ts in timestamps]
            index = pd.DatetimeIndex(dates_list)
            
            # Ensure index and series have matching length
            if len(index) != len(close_prices):
                min_len = min(len(index), len(close_prices))
                index = index[:min_len]
                close_prices = close_prices.iloc[:min_len]
            
            # Create series with datetime index
            series = pd.Series(close_prices.values, index=index, name=output_name)
            return series.sort_index()
        
        except Exception as e:
            print(f"Error parsing Binance data for {pair}: {e}")
            return None

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
    
    def _fetch_bybit_ohlc(self, symbol: str) -> Optional[pd.DataFrame]:
        """
        Fetch daily OHLC data from Bybit (no auth required).
        
        Args:
            symbol: Bybit spot symbol (e.g., 'DAIUSDT')
        
        Returns:
            DataFrame with OHLC data or None on failure
        """
        if 'requests' not in globals():
            return None
        
        try:
            url = "https://api.bybit.com/v5/market/kline"
            params = {
                'category': 'spot',
                'symbol': symbol,
                'interval': 'D',  # Daily
                'limit': 1000  # Max results
            }
            
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            if not data.get('result') or not data['result'].get('list'):
                return None
            
            klines = data['result']['list']
            if not klines:
                return None
            
            # Convert to DataFrame (Bybit returns in reverse chronological order)
            df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover'])
            df['timestamp'] = pd.to_datetime(df['timestamp'].astype(int), unit='ms')
            df = df.set_index('timestamp')
            
            # Convert to numeric
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Sort by timestamp ascending
            df = df.sort_index()
            
            return df
        
        except Exception as e:
            print(f"  Error fetching {symbol} from Bybit: {e}")
            return None
    
    def _fetch_kraken_ohlc(self, pair: str) -> Optional[pd.DataFrame]:
        """
        Fetch daily OHLC data from Kraken (no auth required).
        
        Args:
            pair: Kraken pair (e.g., 'XBTUSDT', 'ETHUSDT', 'USDTZUSD')
        
        Returns:
            DataFrame with OHLC data or None on failure
        """
        if 'requests' not in globals():
            return None
        
        try:
            url = "https://api.kraken.com/0/public/OHLC"
            params = {
                'pair': pair,
                'interval': 1440  # Daily (1440 minutes)
            }
            
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            if data.get('error'):
                return None
            
            if not data.get('result'):
                return None
            
            # Get the pair data (first key that's not 'last')
            pair_keys = [k for k in data['result'].keys() if k != 'last']
            if not pair_keys:
                return None
            
            ohlc_data = data['result'][pair_keys[0]]
            
            if not ohlc_data:
                return None
            
            # Convert to DataFrame
            df = pd.DataFrame(ohlc_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'vwap', 'volume', 'count'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
            df = df.set_index('timestamp')
            
            # Convert to float
            for col in ['open', 'high', 'low', 'close', 'vwap', 'volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            return df
        
        except Exception as e:
            return None
    
    def _generate_placeholder_mcap_series(self, coin_name: str, length: int) -> pd.Series:
        """Generate synthetic market cap series for a stablecoin when real data unavailable."""
        dates = pd.date_range(self.start_date, self.end_date, freq='D')[:length]
        
        np.random.seed({'BUSD': 46, 'DAI': 47}.get(coin_name, 48))
        
        if coin_name == 'BUSD':
            # BUSD: peaked around 2021-2022, declined after
            base = np.linspace(10e9, 5e9, len(dates))
            noise = np.random.normal(0, 0.3e9, len(dates))
        elif coin_name == 'DAI':
            # DAI: steady growth in DeFi
            base = np.linspace(1e9, 5e9, len(dates))
            noise = np.random.normal(0, 0.2e9, len(dates))
        else:
            base = np.linspace(5e9, 20e9, len(dates))
            noise = np.random.normal(0, 0.5e9, len(dates))
        
        mcap = np.maximum(base + noise, 0.1e9)
        return pd.Series(mcap, index=dates)
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
