"""
YFinance Options data downloader for Lean format
"""

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import pytz
import time
import os
from typing import List, Dict, Optional
from tqdm import tqdm

from config import (
    OPTION_DATA_PATH, LEAN_TIMEZONE_EQUITY, LEAN_TIME_FORMAT,
)
from utils import (
    setup_logging, ensure_directory_exists, format_lean_date,
    create_lean_tradebar_csv, write_lean_zip_file, DataValidator
)

logger = setup_logging()

class YFinanceOptionsDownloader:
    """Download options data from Yahoo Finance and convert to Lean format"""

    def __init__(self):
        self.rate_limit_delay = 1.0  # 1 second delay between requests

    def download_symbol_options(self, symbol: str):
        """Download options data for a single symbol"""
        logger.info(f"Downloading options data for {symbol}")

        try:
            # Create ticker object
            ticker = yf.Ticker(symbol)

            # Get options expiration dates
            expirations = ticker.options

            if not expirations:
                logger.warning(f"No options data available for {symbol}")
                return

            logger.info(f"Found {len(expirations)} expiration dates for {symbol}")

            # Download options for each expiration date
            for expiration in expirations[:5]:  # Limit to first 5 expirations to avoid too much data
                try:
                    # Get option chain
                    opt = ticker.option_chain(expiration)

                    # Process calls
                    if not opt.calls.empty:
                        self._process_options_data(opt.calls, symbol, expiration, 'call')

                    # Process puts
                    if not opt.puts.empty:
                        self._process_options_data(opt.puts, symbol, expiration, 'put')

                    # Rate limiting
                    time.sleep(self.rate_limit_delay)

                except Exception as e:
                    logger.error(f"Error downloading options for {symbol} expiration {expiration}: {str(e)}")
                    continue

        except Exception as e:
            logger.error(f"Error downloading options for {symbol}: {str(e)}")

    def _process_options_data(self, options_df: pd.DataFrame, symbol: str, expiration: str, option_type: str):
        """Process and save options data"""
        try:
            # Create directory structure
            option_dir = os.path.join(OPTION_DATA_PATH, symbol.lower(), option_type)
            ensure_directory_exists(option_dir)

            # Convert expiration to date
            exp_date = datetime.strptime(expiration, '%Y-%m-%d')
            date_str = format_lean_date(exp_date)

            # Create file paths
            output_path = os.path.join(option_dir, f"{date_str}_trade.zip")
            csv_filename = f"{date_str}_{symbol.lower()}_{option_type}_trade.csv"

            # Convert options data to Lean format
            csv_content = self._create_lean_options_csv(options_df, symbol, exp_date, option_type)

            if csv_content:
                write_lean_zip_file(csv_content, output_path, csv_filename)
                logger.debug(f"Saved {len(csv_content)} options for {symbol} {option_type} expiring {expiration}")

        except Exception as e:
            logger.error(f"Error processing options data for {symbol}: {str(e)}")

    def _create_lean_options_csv(self, options_df: pd.DataFrame, symbol: str, expiration_date: datetime, option_type: str) -> List[str]:
        """Convert options data to Lean CSV format"""
        csv_lines = []

        try:
            # Lean header for options
            csv_lines.append("Date,Open,High,Low,Close,Volume,OpenInterest")

            for _, option in options_df.iterrows():
                try:
                    # Use lastTradeDate as timestamp, or current time if not available
                    timestamp = option.get('lastTradeDate')
                    if timestamp is None or pd.isna(timestamp):
                        timestamp = datetime.now(pytz.timezone(LEAN_TIMEZONE_EQUITY))
                    elif isinstance(timestamp, (int, float)):
                        timestamp = datetime.fromtimestamp(timestamp, tz=pytz.timezone(LEAN_TIMEZONE_EQUITY))
                    else:
                        timestamp = pd.to_datetime(timestamp).tz_localize(LEAN_TIMEZONE_EQUITY)

                    # Format date
                    date_str = timestamp.strftime("%Y%m%d %H:%M:%S")

                    # Get OHLC data (use lastPrice for all if OHLC not available)
                    last_price = option.get('lastPrice', 0)
                    open_price = option.get('openInterest', last_price)  # Placeholder
                    high_price = last_price
                    low_price = last_price
                    close_price = last_price

                    # Volume and open interest
                    volume = option.get('volume', 0)
                    open_interest = option.get('openInterest', 0)

                    # Format prices (Lean uses deci-cents for equity options)
                    price_multiplier = 10000
                    open_price = int(open_price * price_multiplier)
                    high_price = int(high_price * price_multiplier)
                    low_price = int(low_price * price_multiplier)
                    close_price = int(close_price * price_multiplier)

                    # Create CSV line
                    csv_line = ",".join([
                        date_str,
                        str(open_price),
                        str(high_price),
                        str(low_price),
                        str(close_price),
                        str(volume),
                        str(open_interest)
                    ])

                    csv_lines.append(csv_line)

                except Exception as e:
                    logger.warning(f"Error processing option row: {str(e)}")
                    continue

        except Exception as e:
            logger.error(f"Error creating Lean CSV for options: {str(e)}")

        return csv_lines

    def download_symbols(self, symbols: List[str]):
        """Download options data for multiple symbols"""
        logger.info(f"Starting options download for {len(symbols)} symbols")

        for symbol in tqdm(symbols, desc="Downloading options"):
            try:
                self.download_symbol_options(symbol)
            except Exception as e:
                logger.error(f"Error downloading options for {symbol}: {str(e)}")
                continue

        logger.info("Options download completed")

def main():
    """Main function for testing"""
    from config import DEFAULT_OPTION_SYMBOLS

    downloader = YFinanceOptionsDownloader()

    # Test with a small set of symbols
    test_symbols = DEFAULT_OPTION_SYMBOLS[:2]  # Just SPY and QQQ for testing

    # Download options data
    downloader.download_symbols(test_symbols)

if __name__ == "__main__":
    main()