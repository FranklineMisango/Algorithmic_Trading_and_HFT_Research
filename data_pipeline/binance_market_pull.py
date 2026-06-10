#!/usr/bin/env python3
"""
Download Binance spot and perpetual futures candles plus funding and
reference-rate series for crypto arbitrage research.
"""

from __future__ import annotations

import argparse
from datetime import datetime

from binance_market_data_downloader import BinanceMarketDataDownloader


def parse_date(value: str) -> datetime:
    return datetime.strptime(value, "%Y-%m-%d")


def main() -> None:
    parser = argparse.ArgumentParser(description="Pull Binance spot/perp/funding data")
    parser.add_argument("--symbols", nargs="+", required=True, help="Symbols like BTCUSDT ETHUSDT SOLUSDT")
    parser.add_argument("--start-date", type=parse_date, required=True, help="Start date YYYY-MM-DD")
    parser.add_argument("--end-date", type=parse_date, required=True, help="End date YYYY-MM-DD")
    parser.add_argument("--interval", default="1h", choices=["1h", "4h", "1d"], help="Candle interval")
    parser.add_argument("--output-format", default="raw", choices=["lean", "raw", "csv", "parquet"], help="Output format")
    parser.add_argument("--borrow-asset", default="BTC", help="Asset label for borrow-rate export")
    parser.add_argument("--include-reference-rates", action="store_true", help="Write daily risk-free and borrow-rate series")

    args = parser.parse_args()

    downloader = BinanceMarketDataDownloader()
    downloader.download_crypto_market_data(
        args.symbols,
        args.start_date,
        args.end_date,
        interval=args.interval,
        output_format=args.output_format,
    )

    if args.include_reference_rates:
        downloader.download_reference_rates(args.borrow_asset, start_date=args.start_date, end_date=args.end_date, output_format=args.output_format)


if __name__ == "__main__":
    main()
