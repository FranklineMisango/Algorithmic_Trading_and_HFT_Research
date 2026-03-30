"""
Download daily OHLCV data for tickers found in congressional trade CSV files.

Primary source: Yahoo Finance (yfinance)
Fallback source: Alpaca Market Data API when Yahoo returns no rows

Outputs:
- Combined OHLCV CSV
- Missing symbols CSV
"""

from __future__ import annotations

import argparse
import io
import os
import time
import contextlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import pandas as pd
import requests
import yfinance as yf
import logging
from dotenv import load_dotenv
from tqdm import tqdm


# Keep yfinance/curl chatter out of tqdm output.
logging.getLogger("yfinance").setLevel(logging.CRITICAL)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download OHLCV with Yahoo + Alpaca fallback")
    parser.add_argument(
        "--inputs",
        nargs="*",
        default=[
            "data/quiver_congress_trades_roster.csv",
            "data/quiver_congress_trades.csv",
        ],
        help="Input trade CSV files used to extract ticker symbols",
    )
    parser.add_argument("--start-date", default="2014-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", default=None, help="End date (YYYY-MM-DD), default=today")
    parser.add_argument(
        "--output",
        default="data/ohlcv_yahoo_alpaca.csv",
        help="Output OHLCV CSV path",
    )
    parser.add_argument(
        "--missing-output",
        default="data/ohlcv_missing_symbols.csv",
        help="Output CSV path for unresolved symbols",
    )
    parser.add_argument("--max-symbols", type=int, default=None, help="Optional cap for testing")
    parser.add_argument("--sleep-seconds", type=float, default=0.05, help="Delay between requests")
    return parser.parse_args()


def normalize_symbol(symbol: str) -> str:
    return str(symbol or "").strip().upper()


def load_symbols(input_paths: Iterable[str]) -> List[str]:
    symbols = set()
    for path in input_paths:
        p = Path(path)
        if not p.exists():
            continue
        df = pd.read_csv(p, low_memory=False)
        if "ticker" not in df.columns:
            continue
        vals = (
            df["ticker"]
            .astype(str)
            .map(normalize_symbol)
            .replace({"": pd.NA, "NAN": pd.NA, "NONE": pd.NA})
            .dropna()
            .unique()
            .tolist()
        )
        symbols.update(vals)

    return sorted(symbols)


def to_yahoo_symbol(symbol: str) -> str:
    # Keep BRK.B style symbols compatible with Yahoo.
    return symbol


def fetch_yahoo(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    y_symbol = to_yahoo_symbol(symbol)
    # yfinance prints failures (delisted/no timezone) to stderr/stdout; suppress to keep tqdm readable.
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        df = yf.download(
            y_symbol,
            start=start_date,
            end=end_date,
            interval="1d",
            auto_adjust=False,
            progress=False,
            threads=False,
        )

    if df is None or df.empty:
        return pd.DataFrame()

    # Flatten multi-index columns if yfinance returns them.
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]

    required_cols = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        return pd.DataFrame()

    out = df.reset_index().rename(columns={"Date": "date"})
    out = out[["date", "Open", "High", "Low", "Close", "Adj Close", "Volume"]].copy()
    out.columns = ["date", "open", "high", "low", "close", "adj_close", "volume"]
    out["symbol"] = symbol
    out["source"] = "yahoo"
    return out


def alpaca_headers() -> Tuple[Optional[str], Optional[str], dict]:
    key = os.getenv("APCA_API_KEY_ID")
    secret = os.getenv("APCA_API_SECRET_KEY")
    headers = {}
    if key and secret:
        headers = {
            "APCA-API-KEY-ID": key,
            "APCA-API-SECRET-KEY": secret,
        }
    return key, secret, headers


def fetch_alpaca_stock(symbol: str, start_date: str, end_date: str, headers: dict) -> pd.DataFrame:
    url = "https://data.alpaca.markets/v2/stocks/bars"
    params = {
        "symbols": symbol,
        "timeframe": "1Day",
        "start": f"{start_date}T00:00:00Z",
        "end": f"{end_date}T23:59:59Z",
        "adjustment": "all",
        "feed": "iex",
        "limit": 10000,
    }
    resp = requests.get(url, headers=headers, params=params, timeout=30)
    if resp.status_code != 200:
        return pd.DataFrame()

    payload = resp.json()
    bars = payload.get("bars", {}).get(symbol, [])
    if not bars:
        return pd.DataFrame()

    out = pd.DataFrame(bars)
    out = out.rename(
        columns={
            "t": "date",
            "o": "open",
            "h": "high",
            "l": "low",
            "c": "close",
            "v": "volume",
            "vw": "vwap",
            "n": "trade_count",
        }
    )
    out["date"] = pd.to_datetime(out["date"], utc=True).dt.tz_convert(None)
    out["adj_close"] = out["close"]
    out["symbol"] = symbol
    out["source"] = "alpaca_stock"
    keep_cols = ["date", "open", "high", "low", "close", "adj_close", "volume", "symbol", "source"]
    return out[keep_cols]


def fetch_alpaca_crypto(symbol: str, start_date: str, end_date: str, headers: dict) -> pd.DataFrame:
    # Alpaca crypto symbols are usually BTC/USD style.
    c_symbol = symbol if "/" in symbol else symbol.replace("USD", "/USD")
    url = "https://data.alpaca.markets/v1beta3/crypto/us/bars"
    params = {
        "symbols": c_symbol,
        "timeframe": "1Day",
        "start": f"{start_date}T00:00:00Z",
        "end": f"{end_date}T23:59:59Z",
        "limit": 10000,
    }
    resp = requests.get(url, headers=headers, params=params, timeout=30)
    if resp.status_code != 200:
        return pd.DataFrame()

    payload = resp.json()
    bars = payload.get("bars", {}).get(c_symbol, [])
    if not bars:
        return pd.DataFrame()

    out = pd.DataFrame(bars)
    out = out.rename(
        columns={
            "t": "date",
            "o": "open",
            "h": "high",
            "l": "low",
            "c": "close",
            "v": "volume",
        }
    )
    out["date"] = pd.to_datetime(out["date"], utc=True).dt.tz_convert(None)
    out["adj_close"] = out["close"]
    out["symbol"] = symbol
    out["source"] = "alpaca_crypto"
    keep_cols = ["date", "open", "high", "low", "close", "adj_close", "volume", "symbol", "source"]
    return out[keep_cols]


def maybe_fetch_alpaca(symbol: str, start_date: str, end_date: str, headers: dict) -> pd.DataFrame:
    # Alpaca equities effectively start in 2016 for broad coverage.
    effective_start = max(pd.Timestamp(start_date), pd.Timestamp("2016-01-01")).strftime("%Y-%m-%d")

    stock_df = fetch_alpaca_stock(symbol, effective_start, end_date, headers)
    if not stock_df.empty:
        return stock_df

    # Try crypto endpoint as second fallback for symbols that may be crypto-like.
    crypto_like = symbol.endswith("USD") or symbol in {"BTC", "ETH", "SOL", "XRP", "DOGE"}
    if crypto_like:
        return fetch_alpaca_crypto(symbol, effective_start, end_date, headers)

    return pd.DataFrame()


def main() -> None:
    load_dotenv()
    args = parse_args()

    end_date = args.end_date or datetime.now(timezone.utc).strftime("%Y-%m-%d")

    symbols = load_symbols(args.inputs)
    if args.max_symbols:
        symbols = symbols[: args.max_symbols]

    if not symbols:
        raise RuntimeError("No symbols found in input CSV files.")

    key, secret, headers = alpaca_headers()
    alpaca_enabled = bool(key and secret)

    price_frames: List[pd.DataFrame] = []
    missing_symbols = []
    yahoo_hits = 0
    alpaca_hits = 0
    processed = 0
    interrupted = False

    try:
        for symbol in tqdm(symbols, desc="Downloading OHLCV", unit="symbol"):
            try:
                y_df = fetch_yahoo(symbol, args.start_date, end_date)
                if not y_df.empty:
                    price_frames.append(y_df)
                    yahoo_hits += 1
                    processed += 1
                    if args.sleep_seconds > 0:
                        time.sleep(args.sleep_seconds)
                    continue

                if alpaca_enabled:
                    a_df = maybe_fetch_alpaca(symbol, args.start_date, end_date, headers)
                    if not a_df.empty:
                        price_frames.append(a_df)
                        alpaca_hits += 1
                    else:
                        missing_symbols.append(symbol)
                else:
                    missing_symbols.append(symbol)
            except Exception:
                missing_symbols.append(symbol)

            processed += 1
            if args.sleep_seconds > 0:
                time.sleep(args.sleep_seconds)
    except KeyboardInterrupt:
        interrupted = True
        print("\nInterrupted by user. Saving partial results...", flush=True)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if price_frames:
        all_prices = pd.concat(price_frames, ignore_index=True)
        all_prices["date"] = pd.to_datetime(all_prices["date"], errors="coerce")
        all_prices = all_prices.dropna(subset=["date"]).sort_values(["symbol", "date"]).reset_index(drop=True)
        all_prices.to_csv(output_path, index=False)
    else:
        all_prices = pd.DataFrame(columns=["date", "open", "high", "low", "close", "adj_close", "volume", "symbol", "source"])
        all_prices.to_csv(output_path, index=False)

    missing_path = Path(args.missing_output)
    missing_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"symbol": sorted(set(missing_symbols))}).to_csv(missing_path, index=False)

    print("\nDownload complete")
    print(f"Symbols requested: {len(symbols)}")
    print(f"Symbols processed: {processed}")
    print(f"Yahoo successes: {yahoo_hits}")
    print(f"Alpaca fallback successes: {alpaca_hits}")
    print(f"Missing symbols: {len(set(missing_symbols))}")
    print(f"OHLCV CSV: {output_path}")
    print(f"Missing CSV: {missing_path}")
    if interrupted:
        print("Run ended early due to interrupt; files contain partial results.")
    if not alpaca_enabled:
        print("Alpaca keys not found in environment; fallback was skipped.")


if __name__ == "__main__":
    main()
