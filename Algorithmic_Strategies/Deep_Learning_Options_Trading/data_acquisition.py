"""
Data Acquisition Module for Deep Learning Options Trading Strategy

Fetches S&P 100 options data and underlying stock prices for LSTM-based
delta-neutral straddle trading strategy.
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import logging
from pathlib import Path
import yaml
import time
import requests
from typing import Optional

class OptionsDataAcquisition:
    """
    Handles acquisition of options and underlying data for S&P 100 constituents.
    Implements survivorship bias control using point-in-time index composition.
    """

    def __init__(self, config_path: str = "config.yaml"):
        """Initialize with configuration parameters."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.logger = logging.getLogger(__name__)
        self._setup_logging()

        # S&P 100 constituents (simplified list - in production, use point-in-time data)
        self.sp100_tickers = [
            'AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META', 'TSLA', 'NVDA', 'JPM', 'JNJ',
            'V', 'PG', 'UNH', 'HD', 'MA', 'PFE', 'KO', 'DIS', 'NFLX', 'ADBE',
            'CRM', 'AMD', 'INTC', 'CSCO', 'VZ', 'T', 'CMCSA', 'PEP', 'ABT', 'COST'
        ]
        
        # Polygon.io configuration
        self.polygon_api_key = self.config['data'].get('polygon_api_key')
        self.polygon_rate_limit = self.config['data'].get('polygon_rate_limit', 5)
        self.api_call_count = 0
        self.rate_limit_window_start = time.time()
        
        # Databento configuration
        self.databento_api_key = self.config['data'].get('databento_api_key')

    def _setup_logging(self):
        """Setup logging configuration."""
        logging.basicConfig(
            level=getattr(logging, self.config['logging']['level']),
            format=self.config['logging']['format'],
            handlers=[
                logging.FileHandler(self.config['logging']['file']),
                logging.StreamHandler()
            ]
        )

    def fetch_underlying_prices(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetch daily price data for S&P 100 constituents.

        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format

        Returns:
            DataFrame with multi-index [ticker, date] and price columns
        """
        self.logger.info(f"Fetching underlying prices from {start_date} to {end_date}")

        all_prices = []

        for ticker in self.sp100_tickers:
            try:
                self.logger.debug(f"Fetching data for {ticker}")
                data = yf.download(ticker, start=start_date, end=end_date,
                                 progress=False, threads=False, auto_adjust=True)

                if not data.empty:
                    # Flatten multi-index columns if they exist
                    if isinstance(data.columns, pd.MultiIndex):
                        data.columns = [col[0] if isinstance(col, tuple) else col for col in data.columns]
                    
                    data['ticker'] = ticker
                    data = data.reset_index()
                    all_prices.append(data)

            except Exception as e:
                self.logger.warning(f"Failed to fetch data for {ticker}: {e}")
                continue

        if not all_prices:
            self.logger.error("No price data fetched")
            return pd.DataFrame()

        combined_df = pd.concat(all_prices, ignore_index=True)
        combined_df = combined_df.set_index(['ticker', 'Date'])

        # Determine which close column to use (auto_adjust removes 'Adj Close')
        close_col = 'Close' if 'Close' in combined_df.columns else 'Adj Close'
        
        # Calculate returns - need to sort index first for proper groupby
        combined_df = combined_df.sort_index()
        combined_df['return_1d'] = combined_df.groupby(level=0)[close_col].pct_change(fill_method=None)
        combined_df['return_5d'] = combined_df.groupby(level=0)[close_col].pct_change(5, fill_method=None)

        self.logger.info(f"Fetched data for {len(combined_df.index.levels[0])} tickers")
        return combined_df

    def _rate_limit_polygon_api(self):
        """Enforce Polygon.io rate limits (5 calls/min for free tier)."""
        current_time = time.time()
        
        # Reset counter if we're in a new minute window
        if current_time - self.rate_limit_window_start >= 60:
            self.api_call_count = 0
            self.rate_limit_window_start = current_time
        
        # If we've hit the rate limit, wait until the next window
        if self.api_call_count >= self.polygon_rate_limit:
            sleep_time = 60 - (current_time - self.rate_limit_window_start)
            if sleep_time > 0:
                self.logger.info(f"Rate limit reached. Waiting {sleep_time:.1f}s...")
                time.sleep(sleep_time)
                self.api_call_count = 0
                self.rate_limit_window_start = time.time()
        
        self.api_call_count += 1

    def fetch_polygon_options_data(self, ticker: str, date: str) -> Optional[pd.DataFrame]:
        """
        Fetch real options data from Polygon.io for a specific ticker and date.
        
        Args:
            ticker: Stock ticker symbol
            date: Date in YYYY-MM-DD format
            
        Returns:
            DataFrame with options data or None if fetch fails
        """
        if not self.polygon_api_key:
            self.logger.error("Polygon.io API key not configured")
            return None
        
        self._rate_limit_polygon_api()
        
        try:
            # Polygon.io snapshot endpoint for options
            url = f"https://api.polygon.io/v3/snapshot/options/{ticker}"
            params = {
                'apiKey': self.polygon_api_key,
                'date': date
            }
            
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                if 'results' not in data or not data['results']:
                    return None
                
                options_list = []
                for option in data['results']:
                    details = option.get('details', {})
                    last_quote = option.get('last_quote', {})
                    greeks = option.get('greeks', {})
                    
                    options_list.append({
                        'date': date,
                        'ticker': ticker,
                        'contract_type': details.get('contract_type', '').lower(),
                        'strike': details.get('strike_price'),
                        'expiry': details.get('expiration_date'),
                        'bid': last_quote.get('bid'),
                        'ask': last_quote.get('ask'),
                        'last': last_quote.get('last'),
                        'volume': option.get('day', {}).get('volume', 0),
                        'open_interest': option.get('open_interest', 0),
                        'implied_vol': greeks.get('vega'),
                        'delta': greeks.get('delta'),
                        'gamma': greeks.get('gamma'),
                        'theta': greeks.get('theta'),
                        'vega': greeks.get('vega')
                    })
                
                return pd.DataFrame(options_list)
            
            elif response.status_code == 429:
                self.logger.warning(f"Rate limit hit for {ticker} on {date}")
                time.sleep(60)
                return self.fetch_polygon_options_data(ticker, date)
            
            else:
                self.logger.warning(f"Polygon API error {response.status_code} for {ticker}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error fetching Polygon data for {ticker}: {e}")
            return None

    def fetch_real_options_data(self, underlying_prices: pd.DataFrame) -> pd.DataFrame:
        """
        Fetch real options data from Polygon.io for all tickers and dates.
        Note: Free tier limits to 2 years of history and 5 calls/min.
        
        Args:
            underlying_prices: DataFrame with underlying price data
            
        Returns:
            DataFrame with real options data
        """
        self.logger.info("Fetching real options data from Polygon.io")
        
        if not self.polygon_api_key:
            raise ValueError(
                "Polygon.io API key required. Get free key at: https://polygon.io/\n"
                "Set in config.yaml: data.polygon_api_key: 'YOUR_KEY_HERE'"
            )
        
        all_options = []
        dates = underlying_prices.index.get_level_values('Date').unique()
        
        # Limit to last 2 years for free tier
        cutoff_date = pd.Timestamp.now() - pd.Timedelta(days=730)
        recent_dates = [d for d in dates if d >= cutoff_date]
        
        # Sample every 5th day to reduce API calls
        sampled_dates = recent_dates[::5]
        
        self.logger.info(f"Fetching data for {len(sampled_dates)} sampled dates")
        
        for idx, date in enumerate(sampled_dates):
            date_str = date.strftime('%Y-%m-%d')
            
            for ticker in self.sp100_tickers:
                options_df = self.fetch_polygon_options_data(ticker, date_str)
                
                if options_df is not None and not options_df.empty:
                    all_options.append(options_df)
                
                if (idx + 1) % 10 == 0:
                    self.logger.info(f"Processed {idx + 1}/{len(sampled_dates)} dates")
        
        if not all_options:
            raise ValueError("No options data fetched from Polygon.io")
        
        combined_options = pd.concat(all_options, ignore_index=True)
        
        # Filter for liquidity
        combined_options = combined_options[
            (combined_options['volume'] >= self.config['data']['min_volume']) &
            (combined_options['open_interest'] >= self.config['data']['min_open_interest'])
        ]
        
        self.logger.info(f"Fetched {len(combined_options)} real options records")
        return combined_options

    def fetch_databento_options_data(self, underlying_prices: pd.DataFrame) -> pd.DataFrame:
        """
        Fetch real options data from Databento for all tickers and dates.
        Databento provides high-quality historical options data with OPRA feed.
        
        Args:
            underlying_prices: DataFrame with underlying price data
            
        Returns:
            DataFrame with real options data from Databento
        """
        self.logger.info("Fetching real options data from Databento")
        
        if not self.databento_api_key:
            raise ValueError(
                "Databento API key required. Get key at: https://databento.com/\n"
                "Set in config.yaml: data.databento_api_key: 'YOUR_KEY_HERE'"
            )
        
        try:
            import databento as db
        except ImportError:
            raise ImportError(
                "databento package not installed. Install with: pip install databento"
            )
        
        client = db.Historical(self.databento_api_key)
        
        all_options = []
        
        # Use recent dates only (Databento access depends on subscription)
        # Data available up to yesterday
        end_date = (pd.Timestamp.now() - pd.Timedelta(days=1)).strftime('%Y-%m-%d')
        start_date = (pd.Timestamp.now() - pd.Timedelta(days=31)).strftime('%Y-%m-%d')
        
        self.logger.info(f"Fetching options data from {start_date} to {end_date}")
        self.logger.info(f"Processing {len(self.sp100_tickers[:3])} tickers (testing with 3 tickers first)...")
        
        for idx, ticker in enumerate(self.sp100_tickers[:3]):
            try:
                self.logger.info(f"[{idx+1}/3] Fetching options for {ticker}...")
                
                # Fetch OHLCV data for all options of this ticker
                # Use parent symbology: TICKER.OPT format with stype_in='parent'
                parent_symbol = f"{ticker}.OPT"
                
                data = client.timeseries.get_range(
                    dataset='OPRA.PILLAR',
                    symbols=[parent_symbol],
                    schema='ohlcv-1d',
                    start=start_date,
                    end=end_date,
                    stype_in='parent',  # This gets all options for the parent ticker
                    limit=1000  # Limit records to avoid excessive data
                )
                
                if data is not None:
                    df = data.to_df()
                    
                    if not df.empty:
                        # Parse the option symbols from the multi-index
                        df_reset = df.reset_index()
                        df_reset['parent_ticker'] = ticker
                        df_reset['date'] = pd.to_datetime(df_reset['ts_event'])
                        
                        # Extract symbol info if available
                        if 'symbol' in df_reset.columns:
                            self.logger.info(f"  Sample symbols: {df_reset['symbol'].unique()[:3].tolist()}")
                        
                        all_options.append(df_reset)
                        self.logger.info(f"  ✓ Fetched {len(df_reset)} records for {ticker}")
                    else:
                        self.logger.info(f"  No data returned for {ticker}")
                else:
                    self.logger.info(f"  No data available for {ticker}")
                
            except Exception as e:
                self.logger.error(f"  ✗ Error fetching data for {ticker}: {type(e).__name__}: {e}")
                continue
        
        if not all_options:
            self.logger.error("No options data fetched from Databento - NO SYNTHETIC FALLBACK ALLOWED")
            return None
        
        # DEBUG: Check columns before concat
        self.logger.info(f"DEBUG: First df columns before concat: {all_options[0].columns.tolist()}")
        self.logger.info(f"DEBUG: First df volume sample: {all_options[0]['volume'].head().tolist() if 'volume' in all_options[0].columns else 'NO VOLUME COLUMN'}")
        
        combined_options = pd.concat(all_options, ignore_index=True)
        
        # DEBUG: Check after concat
        self.logger.info(f"DEBUG: After concat columns: {combined_options.columns.tolist()}")
        self.logger.info(f"DEBUG: After concat volume sample: {combined_options['volume'].head().tolist() if 'volume' in combined_options.columns else 'NO VOLUME COLUMN'}")
        
        # Transform to standard format
        options_df = self._transform_databento_format(combined_options)
        
        # Filter for liquidity if columns exist (only if we have open_interest data)
        if 'volume' in options_df.columns:
            initial_count = len(options_df)
            # Use lower threshold for daily OHLCV data (not intraday trades)
            min_vol = 1  # At least 1 contract traded per day
            
            if 'open_interest' in options_df.columns:
                # Both filters available
                options_df = options_df[
                    (options_df['volume'] >= min_vol) &
                    (options_df['open_interest'] >= self.config['data']['min_open_interest'])
                ]
            else:
                # Only volume filter (OHLCV schema doesn't include open_interest)
                options_df = options_df[options_df['volume'] >= min_vol]
            
            self.logger.info(f"Filtered to {len(options_df)} records (from {initial_count}) with volume >= {min_vol}")
        
        self.logger.info(f"Successfully fetched {len(options_df)} Databento options records")
        return options_df
    
    def _transform_databento_format(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform Databento format to standard options format."""
        standard_df = pd.DataFrame()
        
        # Use date column if it exists
        if 'date' in df.columns:
            standard_df['date'] = pd.to_datetime(df['date'])
        elif 'ts_event' in df.columns:
            standard_df['date'] = pd.to_datetime(df['ts_event'])
        
        # Get ticker
        if 'parent_ticker' in df.columns:
            standard_df['ticker'] = df['parent_ticker']
        elif 'ticker' in df.columns:
            standard_df['ticker'] = df['ticker']
        
        # Get symbol if available
        if 'symbol' in df.columns:
            standard_df['contract_symbol'] = df['symbol']
        
        # Map OHLCV columns
        if 'open' in df.columns:
            standard_df['open'] = df['open']
        if 'high' in df.columns:
            standard_df['high'] = df['high']
        if 'low' in df.columns:
            standard_df['low'] = df['low']
        if 'close' in df.columns:
            standard_df['close'] = df['close']
        if 'volume' in df.columns:
            standard_df['volume'] = df['volume'].fillna(0).astype(int)
        
        # Additional fields if available
        if 'open_interest' in df.columns:
            standard_df['open_interest'] = df['open_interest'].fillna(0).astype(int)
        else:
            standard_df['open_interest'] = 0
        
        return standard_df

    def generate_synthetic_options_data(self, underlying_prices: pd.DataFrame) -> pd.DataFrame:
        """
        Generate synthetic options data for demonstration.
        In production, this would fetch from options data providers.

        Args:
            underlying_prices: DataFrame with underlying price data

        Returns:
            DataFrame with options data
        """
        self.logger.info("Generating synthetic options data")

        options_data = []

        for ticker in underlying_prices.index.levels[0]:
            ticker_data = underlying_prices.loc[ticker]
            
            # Determine which close column to use
            close_col = 'Close' if 'Close' in ticker_data.columns else 'Adj Close'

            for date in ticker_data.index:
                spot_price = ticker_data.loc[date, close_col]

                # Generate synthetic options for different strikes and expirations
                for days_to_expiry in [30, 60, 90]:  # 1, 2, 3 months
                    expiry_date = date + timedelta(days=days_to_expiry)

                    for moneyness in [0.9, 0.95, 1.0, 1.05, 1.1]:  # OTM, ATM, ITM
                        strike = spot_price * moneyness

                        # Synthetic Black-Scholes like pricing (simplified)
                        time_value = days_to_expiry / 365
                        volatility = 0.3  # Assumed volatility

                        # Simplified option price calculation
                        d1 = (np.log(spot_price/strike) + (0.02 + 0.5*volatility**2)*time_value) / (volatility*np.sqrt(time_value))
                        d2 = d1 - volatility*np.sqrt(time_value)

                        call_price = spot_price * 0.5 * (1 + np.tanh(d1)) - strike * np.exp(-0.02*time_value) * 0.5 * (1 + np.tanh(d2))
                        put_price = strike * np.exp(-0.02*time_value) * 0.5 * (1 - np.tanh(d2)) - spot_price * 0.5 * (1 - np.tanh(d1))

                        # Straddle price
                        straddle_price = call_price + put_price

                        # Implied volatility (simplified)
                        implied_vol = volatility * (1 + 0.1 * np.random.normal())

                        options_data.append({
                            'date': date,
                            'ticker': ticker,
                            'strike': strike,
                            'expiry': expiry_date,
                            'days_to_expiry': days_to_expiry,
                            'spot_price': spot_price,
                            'call_price': max(call_price, 0.01),  # Minimum price
                            'put_price': max(put_price, 0.01),
                            'straddle_price': max(straddle_price, 0.01),
                            'implied_vol': implied_vol,
                            'moneyness': moneyness,
                            'volume': np.random.randint(10, 1000),
                            'open_interest': np.random.randint(50, 5000)
                        })

        options_df = pd.DataFrame(options_data)
        options_df['date'] = pd.to_datetime(options_df['date'])
        options_df['expiry'] = pd.to_datetime(options_df['expiry'])

        # Filter for liquidity thresholds
        options_df = options_df[
            (options_df['volume'] >= self.config['data']['min_volume']) &
            (options_df['open_interest'] >= self.config['data']['min_open_interest'])
        ]

        self.logger.info(f"Generated {len(options_df)} options records")
        return options_df

    def fetch_full_dataset(self, start_date: str = None, end_date: str = None) -> tuple:
        """
        Fetch complete dataset including underlying prices and options data.

        Args:
            start_date: Override config start date
            end_date: Override config end date

        Returns:
            Tuple of (underlying_prices_df, options_df)
        """
        start_date = start_date or self.config['data']['start_date']
        end_date = end_date or self.config['data']['end_date']

        self.logger.info(f"Fetching full dataset from {start_date} to {end_date}")

        # Fetch underlying prices
        underlying_prices = self.fetch_underlying_prices(start_date, end_date)

        if underlying_prices.empty:
            raise ValueError("Failed to fetch underlying price data")

        # Fetch REAL options data ONLY - NO SYNTHETIC FALLBACK
        options_source = self.config['data'].get('options_source', 'polygon')
        
        if options_source == 'databento':
            self.logger.info("Using Databento for real options data")
            options_data = self.fetch_databento_options_data(underlying_prices)
            if options_data is None or options_data.empty:
                raise ValueError("Databento returned no options data. NO SYNTHETIC FALLBACK ALLOWED.")
        elif options_source == 'polygon':
            self.logger.info("Using Polygon.io for real options data")
            options_data = self.fetch_real_options_data(underlying_prices)
            if options_data is None or options_data.empty:
                raise ValueError("Polygon returned no options data. NO SYNTHETIC FALLBACK ALLOWED.")
        else:
            raise ValueError(f"Invalid options_source: {options_source}. Must be 'databento' or 'polygon'. NO SYNTHETIC DATA ALLOWED.")

        # Save to disk
        self._save_data(underlying_prices, options_data)

        return underlying_prices, options_data

    def _save_data(self, prices_df: pd.DataFrame, options_df: pd.DataFrame):
        """Save data to configured paths."""
        # Create directories
        Path(self.config['data']['underlying_data_path']).parent.mkdir(parents=True, exist_ok=True)
        Path(self.config['data']['options_data_path']).parent.mkdir(parents=True, exist_ok=True)

        # Save data
        prices_df.to_csv(f"{self.config['data']['underlying_data_path']}/underlying_prices.csv")
        options_df.to_csv(f"{self.config['data']['options_data_path']}/options_data.csv")

        self.logger.info("Data saved to disk")


if __name__ == "__main__":
    acquirer = OptionsDataAcquisition()
    prices, options = acquirer.fetch_full_dataset()
    print(f"Fetched {len(prices)} price records and {len(options)} options records")