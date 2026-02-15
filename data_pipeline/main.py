"""
Main data pipeline script
"""

import argparse
import sys
from datetime import datetime, timedelta
from typing import List

from config import (
    DEFAULT_EQUITY_SYMBOLS, DEFAULT_CRYPTO_SYMBOLS, DEFAULT_OPTION_SYMBOLS, DEFAULT_FUTURES_SYMBOLS,
    DEFAULT_DATABENTO_FUTURES_SYMBOLS, DEFAULT_DATABENTO_OPTIONS_SYMBOLS, DEFAULT_START_DATE, DEFAULT_END_DATE,
    SUPPORTED_RESOLUTIONS
)
from alpaca_downloader import AlpacaDataDownloader
from binance_downloader import BinanceDataDownloader
# from yfinance_options_downloader import YFinanceOptionsDownloader
# from polygon_options_downloader import PolygonOptionsDownloader
from polygon_futures_downloader import PolygonFuturesDownloader
from databento_downloader import DatabentoFuturesDownloader
from databento_options_downloader import DatabentoOptionsDownloader
from utils import setup_logging

logger = setup_logging()

def parse_date(date_str: str) -> datetime:
    """Parse date string to datetime object"""
    try:
        return datetime.strptime(date_str, '%Y-%m-%d')
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid date format: {date_str}. Use YYYY-MM-DD")

def main():
    parser = argparse.ArgumentParser(description='Download financial data and convert to Lean format')
    
    # Data source arguments
    parser.add_argument('--source', choices=['alpaca', 'binance', 'options', 'futures', 'databento', 'databento-options', 'all'], default='all',
                       help='Data source to download from')
    parser.add_argument('--options-source', choices=['yfinance', 'polygon', 'databento'], default='polygon',
                       help='Options data source: yfinance (snapshot), polygon (historical), or databento (historical)')
    
    # Symbol arguments
    parser.add_argument('--equity-symbols', nargs='+', default=DEFAULT_EQUITY_SYMBOLS,
                       help='Equity symbols to download (for Alpaca)')
    parser.add_argument('--crypto-symbols', nargs='+', default=DEFAULT_CRYPTO_SYMBOLS,
                       help='Crypto symbols to download (for Binance)')
    parser.add_argument('--option-symbols', nargs='+', default=DEFAULT_OPTION_SYMBOLS,
                       help='Option symbols to download (for yfinance)')
    parser.add_argument('--futures-symbols', nargs='+', default=DEFAULT_FUTURES_SYMBOLS,
                       help='Futures symbols to download (for Polygon.io)')
    parser.add_argument('--databento-symbols', nargs='+', default=DEFAULT_DATABENTO_FUTURES_SYMBOLS,
                       help='Futures symbols to download (for Databento)')
    parser.add_argument('--databento-options-symbols', nargs='+', default=DEFAULT_DATABENTO_OPTIONS_SYMBOLS,
                       help='Options symbols to download (for Databento OPRA)')
    
    # Date range arguments
    parser.add_argument('--start-date', type=parse_date, default=DEFAULT_START_DATE,
                       help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=parse_date, default=DEFAULT_END_DATE,
                       help='End date (YYYY-MM-DD)')
    
    # Resolution arguments
    parser.add_argument('--resolution', choices=SUPPORTED_RESOLUTIONS, default='minute',
                       help='Data resolution')
    
    # Other arguments
    parser.add_argument('--test', action='store_true',
                       help='Run in test mode with limited symbols and date range')
    
    args = parser.parse_args()
    
    # Test mode adjustments
    if args.test:
        args.equity_symbols = ['AAPL', 'GOOGL', 'MSFT'][:2]
        args.crypto_symbols = ['BTCUSDT', 'ETHUSDT'][:2]
        args.option_symbols = ['SPY', 'AAPL'][:2]
        args.futures_symbols = ['ES', 'NQ'][:2]
        args.databento_symbols = ['ES.FUT', 'NQ.FUT'][:2]
        args.databento_options_symbols = ['SPY'][:1]
        args.start_date = datetime.now() - timedelta(days=7)
        args.end_date = datetime.now()
        logger.info("Running in test mode with limited symbols and date range")
    
    # Validate date range
    if args.start_date >= args.end_date:
        logger.error("Start date must be before end date")
        sys.exit(1)
    
    logger.info(f"Starting data download from {args.start_date.strftime('%Y-%m-%d')} to {args.end_date.strftime('%Y-%m-%d')}")
    logger.info(f"Resolution: {args.resolution}")
    
    # Download equity data from Alpaca
    if args.source in ['alpaca', 'all']:
        try:
            logger.info("Starting Alpaca data download...")
            alpaca_downloader = AlpacaDataDownloader()
            alpaca_downloader.download_multiple_symbols(
                args.equity_symbols, 
                args.resolution, 
                args.start_date, 
                args.end_date
            )
            logger.info("Alpaca download completed")
        except Exception as e:
            logger.error(f"Error with Alpaca download: {str(e)}")
            if args.source == 'alpaca':
                sys.exit(1)
    
    # Download crypto data from Binance
    if args.source in ['binance', 'all']:
        try:
            logger.info("Starting Binance data download...")
            binance_downloader = BinanceDataDownloader()
            binance_downloader.download_multiple_symbols(
                args.crypto_symbols, 
                args.resolution, 
                args.start_date, 
                args.end_date
            )
            logger.info("Binance download completed")
        except Exception as e:
            logger.error(f"Error with Binance download: {str(e)}")
            if args.source == 'binance':
                sys.exit(1)
    
    # Download options data
    if args.source in ['options', 'all']:
        try:
            if args.options_source == 'databento':
                logger.info("Starting HISTORICAL Options data download via Databento...")
                options_downloader = DatabentoOptionsDownloader()
                for symbol in args.option_symbols:
                    logger.info(f"Downloading Databento options for {symbol}")
                    options_downloader.download_options(
                        symbol,
                        args.start_date,
                        args.end_date,
                        args.resolution,
                        limit_contracts=20,
                        filter_near_money=True
                    )
                logger.info("Databento options download completed")
            # elif args.options_source == 'polygon':
            #     logger.info("Starting HISTORICAL Options data download via Polygon...")
            #     options_downloader = PolygonOptionsDownloader()
            #     for symbol in args.option_symbols:
            #         logger.info(f"Downloading historical options for {symbol}")
            #         options_downloader.download_historical_options(
            #             symbol,
            #             args.start_date,
            #             args.end_date,
            #             args.resolution,
            #             limit_contracts=20  # Limit to 20 contracts per symbol to stay within API limits
            #         )
            #     logger.info("Historical options download completed")
            # else:
            #     logger.info("Starting Options SNAPSHOT download via YFinance...")
            #     options_downloader = YFinanceOptionsDownloader()
            #     options_downloader.download_symbols(args.option_symbols)
            #     logger.info("Options snapshot download completed")
            else:
                logger.warning(f"Options source '{args.options_source}' not available. Only 'databento' is currently supported.")
        except Exception as e:
            logger.error(f"Error with Options download: {str(e)}")
            if args.source == 'options':
                sys.exit(1)

    # Download futures data from Polygon.io
    if args.source in ['futures', 'all']:
        try:
            logger.info("Starting Futures data download...")
            futures_downloader = PolygonFuturesDownloader()
            futures_downloader.download_symbols(
                args.futures_symbols,
                args.start_date,
                args.end_date,
                args.resolution
            )
            logger.info("Futures download completed")
        except Exception as e:
            logger.error(f"Error with Futures download: {str(e)}")
            if args.source == 'futures':
                sys.exit(1)
    
    # Download futures data from Databento
    if args.source in ['databento', 'all']:
        try:
            logger.info("Starting Databento futures data download...")
            databento_downloader = DatabentoFuturesDownloader()
            databento_downloader.download_symbols(
                args.databento_symbols,
                args.start_date,
                args.end_date,
                args.resolution
            )
            logger.info("Databento futures download completed")
        except Exception as e:
            logger.error(f"Error with Databento futures download: {str(e)}")
            if args.source == 'databento':
                sys.exit(1)
    
    # Download options data from Databento
    if args.source in ['databento-options', 'all']:
        try:
            logger.info("Starting Databento options data download...")
            databento_options_downloader = DatabentoOptionsDownloader()
            for symbol in args.databento_options_symbols:
                logger.info(f"Downloading Databento options for {symbol}")
                databento_options_downloader.download_options(
                    symbol,
                    args.start_date,
                    args.end_date,
                    args.resolution,
                    limit_contracts=20,
                    filter_near_money=True
                )
            logger.info("Databento options download completed")
        except Exception as e:
            logger.error(f"Error with Databento options download: {str(e)}")
            if args.source == 'databento-options':
                sys.exit(1)
    
    logger.info("Data pipeline completed successfully!")

if __name__ == "__main__":
    main()
