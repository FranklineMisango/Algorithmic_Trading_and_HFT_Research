"""
Data Fetcher for Put-Futures Arbitrage Backtesting
Fetches SPY options and ES futures data and converts to Lean format
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import json

class LeanDataFetcher:
    """Fetch data and convert to QuantConnect Lean format"""

    def __init__(self, data_root="../../../data"):
        self.data_root = os.path.abspath(data_root)
        print(f"Data root: {self.data_root}")
        self.ensure_directories()

    def ensure_directories(self):
        """Create necessary directories"""
        dirs = [
            f"{self.data_root}/equity/usa/daily",
            f"{self.data_root}/future/cme/daily/es",
            f"{self.data_root}/option/usa/daily/spy"
        ]
        for d in dirs:
            os.makedirs(d, exist_ok=True)

    def fetch_equity_data(self, symbol, start_date, end_date):
        """Fetch equity data (SPY as proxy for SPX)"""
        print(f"Fetching {symbol} data...")
        data = yf.download(symbol, start=start_date, end=end_date)

        if data.empty:
            print(f"No data for {symbol}")
            return

        # Convert to Lean format
        lean_data = self.convert_to_lean_equity(data, symbol)

        # Save daily files
        self.save_lean_equity(lean_data, symbol)

    def fetch_futures_data(self, symbol, start_date, end_date):
        """Fetch futures data (ES)"""
        print(f"Fetching {symbol} futures data...")
        # Use yfinance for futures
        ticker = yf.Ticker(f"{symbol}=F")
        data = ticker.history(start=start_date, end=end_date)

        if data.empty:
            print(f"No futures data for {symbol}")
            return

        # Convert to Lean format
        lean_data = self.convert_to_lean_futures(data, symbol)

        # Save daily files
        self.save_lean_futures(lean_data, symbol)

    def fetch_options_data(self, symbol, start_date, end_date):
        """Fetch options data for backtesting"""
        print(f"Fetching {symbol} options data...")

        # This is simplified - in practice need historical options data
        # For now, create synthetic data based on equity prices
        equity_data = yf.download(symbol, start=start_date, end=end_date)

        if equity_data.empty:
            return

        # Generate synthetic options data
        options_data = self.generate_synthetic_options(equity_data, symbol)

        # Save to Lean format
        self.save_lean_options(options_data, symbol)

    def generate_synthetic_options(self, equity_data, symbol):
        """Generate synthetic options data for testing"""
        options = []

        for date, row in equity_data.iterrows():
            price = row['Close'].item() if hasattr(row['Close'], 'item') else float(row['Close'])
            vol = row['Volume'].item() if hasattr(row['Volume'], 'item') else float(row['Volume'])

            # Generate strikes around current price
            strikes = np.linspace(price * 0.8, price * 1.2, 10)

            for strike in strikes:
                # Simple option price approximation
                T = 30/365  # 30 days to expiration
                r = 0.05
                sigma = 0.2  # Assumed volatility

                # Intrinsic value approximation
                call_intrinsic = max(price - strike, 0)
                put_intrinsic = max(strike - price, 0)

                # Time value (simplified)
                time_value = 0.05 * price * np.sqrt(T)

                call_price = call_intrinsic + time_value
                put_price = put_intrinsic + time_value

                # Create option records
                call_option = {
                    'date': date,
                    'symbol': f"{symbol}{int(strike)}C{date.strftime('%y%m%d')}",
                    'strike': strike,
                    'type': 'call',
                    'price': max(call_price, 0.01),
                    'expiration': date + timedelta(days=30)
                }

                put_option = {
                    'date': date,
                    'symbol': f"{symbol}{int(strike)}P{date.strftime('%y%m%d')}",
                    'strike': strike,
                    'type': 'put',
                    'price': max(put_price, 0.01),
                    'expiration': date + timedelta(days=30)
                }

                options.extend([call_option, put_option])

        return pd.DataFrame(options)

    def norm_cdf(self, x):
        """Approximate normal CDF"""
        return 0.5 * (1 + np.sign(x) * np.sqrt(1 - np.exp(-2*x**2/np.pi)))

    def convert_to_lean_equity(self, data, symbol):
        """Convert to Lean equity format"""
        lean_data = data.copy()
        lean_data['open'] = (lean_data['Open'] * 10000).round().astype(int)
        lean_data['high'] = (lean_data['High'] * 10000).round().astype(int)
        lean_data['low'] = (lean_data['Low'] * 10000).round().astype(int)
        lean_data['close'] = (lean_data['Close'] * 10000).round().astype(int)
        lean_data['volume'] = lean_data['Volume'].astype(int)

        return lean_data[['open', 'high', 'low', 'close', 'volume']]

    def convert_to_lean_futures(self, data, symbol):
        """Convert to Lean futures format"""
        lean_data = data.copy()
        # Futures use actual prices, not deci-cents
        lean_data['open'] = lean_data['Open']
        lean_data['high'] = lean_data['High']
        lean_data['low'] = lean_data['Low']
        lean_data['close'] = lean_data['Close']
        lean_data['volume'] = lean_data['Volume'].astype(int)

        return lean_data[['open', 'high', 'low', 'close', 'volume']]

    def save_lean_equity(self, data, symbol):
        """Save equity data in Lean format"""
        print(f"Saving {len(data)} equity records for {symbol}")
        for date, row in data.iterrows():
            date_str = date.strftime('%Y%m%d')
            filename = f"{date_str}_{symbol.lower()}_daily.csv"
            filepath = f"{self.data_root}/equity/usa/daily/{filename}"

            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            with open(filepath, 'w') as f:
                f.write(f"{float(row['open'])},{float(row['high'])},{float(row['low'])},{float(row['close'])},{int(row['volume'])}\n")

    def save_lean_futures(self, data, symbol):
        """Save futures data in Lean format"""
        print(f"Saving {len(data)} futures records for {symbol}")
        for date, row in data.iterrows():
            date_str = date.strftime('%Y%m%d')
            filename = f"{date_str}_{symbol.lower()}_daily.csv"
            filepath = f"{self.data_root}/future/cme/daily/es/{filename}"

            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            with open(filepath, 'w') as f:
                f.write(f"{float(row['open']):.2f},{float(row['high']):.2f},{float(row['low']):.2f},{float(row['close']):.2f},{int(row['volume'])}\n")

    def save_lean_options(self, data, symbol):
        """Save options data (simplified for Lean)"""
        # Options are more complex in Lean - this is a placeholder
        for _, row in data.iterrows():
            date_str = row['date'].strftime('%Y%m%d')
            filename = f"{date_str}_{symbol.lower()}_options.csv"
            filepath = f"{self.data_root}/option/usa/daily/spy/{filename}"

            with open(filepath, 'a') as f:
                f.write(f"{row['symbol']},{row['strike']},{row['type']},{row['price']:.4f},{row['expiration'].strftime('%Y%m%d')}\n")

    def fetch_all_data(self, start_date, end_date):
        """Fetch all required data"""
        print("Fetching data for backtesting...")

        # Fetch equity data (SPY)
        self.fetch_equity_data('SPY', start_date, end_date)

        # Fetch futures data (ES)
        self.fetch_futures_data('ES', start_date, end_date)

        # Fetch options data (synthetic)
        self.fetch_options_data('SPY', start_date, end_date)

        print("Data fetching complete!")

if __name__ == "__main__":
    # Example usage
    fetcher = LeanDataFetcher()

    start_date = datetime(2023, 1, 1)
    end_date = datetime(2024, 12, 31)

    fetcher.fetch_all_data(start_date, end_date)