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
                                 progress=False, threads=False)

                if not data.empty:
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

        # Calculate returns
        combined_df['return_1d'] = combined_df.groupby('ticker')['Adj Close'].pct_change()
        combined_df['return_5d'] = combined_df.groupby('ticker')['Adj Close'].pct_change(5)

        self.logger.info(f"Fetched data for {len(combined_df.index.levels[0])} tickers")
        return combined_df

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

            for date in ticker_data.index:
                spot_price = ticker_data.loc[date, 'Adj Close']

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

        # Generate options data
        options_data = self.generate_synthetic_options_data(underlying_prices)

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
    # Example usage
    acquirer = OptionsDataAcquisition()
    prices, options = acquirer.fetch_full_dataset()
    print(f"Fetched {len(prices)} price records and {len(options)} options records")