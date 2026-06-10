"""
Binance market data downloader for spot, perpetual futures, funding rates,
and daily reference rates used by crypto arbitrage strategies.
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import List, Optional

import pandas as pd
import requests
import time
import hmac
import hashlib
from urllib.parse import urlencode
import os

from config import DEFAULT_OUTPUT_FORMAT, RAW_MARKET_PATH, RAW_MACRO_PATH
from fred_downloader import FREDDownloader
from utils import ensure_directory_exists, setup_logging, write_parquet, write_raw_csv


class BinanceMarketDataDownloader:
    """Pull Binance spot/perp/funding data and store it in raw/parquet form."""

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"
        })
        self.logger = setup_logging()

    def _signed_get(self, base_url: str, endpoint: str, params: dict, api_key: str, api_secret: str):
        """Perform a signed GET request to Binance SAPI endpoints.

        Returns parsed JSON on success, or raises.
        """
        # add timestamp
        params = dict(params or {})
        params["timestamp"] = int(time.time() * 1000)
        query_string = urlencode(params)
        signature = hmac.new(api_secret.encode(), query_string.encode(), hashlib.sha256).hexdigest()
        signed_qs = f"{query_string}&signature={signature}"
        url = f"{base_url}{endpoint}?{signed_qs}"
        headers = {"X-MBX-APIKEY": api_key}
        resp = self.session.get(url, headers=headers, timeout=30)
        resp.raise_for_status()
        return resp.json()

    @staticmethod
    def _interval_to_milliseconds(interval: str) -> int:
        mapping = {
            "1h": 60 * 60 * 1000,
            "4h": 4 * 60 * 60 * 1000,
            "1d": 24 * 60 * 60 * 1000,
        }
        return mapping.get(interval, 60 * 60 * 1000)

    def _klines(self, base_url: str, endpoint: str, symbol: str, interval: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        url = f"{base_url}{endpoint}"
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": int(start_date.timestamp() * 1000),
            "endTime": int(end_date.timestamp() * 1000),
            "limit": 1000,
        }
        response = self.session.get(url, params=params, timeout=30)
        response.raise_for_status()
        rows = response.json()
        if not rows:
            return pd.DataFrame()

        frame = pd.DataFrame(rows, columns=[
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_asset_volume", "number_of_trades",
            "taker_buy_base_volume", "taker_buy_quote_volume", "ignore"
        ])
        frame["timestamp"] = pd.to_datetime(frame["open_time"], unit="ms", utc=True)
        for column in ["open", "high", "low", "close", "volume"]:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        return frame[["timestamp", "open", "high", "low", "close", "volume"]].dropna()

    def fetch_spot_prices(self, symbol: str, interval: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        return self._klines("https://api.binance.com", "/api/v3/klines", symbol, interval, start_date, end_date)

    def fetch_perp_prices(self, symbol: str, interval: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        perp_symbol = symbol if symbol.endswith("USDT") else f"{symbol}USDT"
        return self._klines("https://fapi.binance.com", "/fapi/v1/klines", perp_symbol, interval, start_date, end_date)

    def fetch_funding_rates(self, symbol: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        perp_symbol = symbol if symbol.endswith("USDT") else f"{symbol}USDT"
        url = "https://fapi.binance.com/fapi/v1/fundingRate"
        params = {
            "symbol": perp_symbol,
            "startTime": int(start_date.timestamp() * 1000),
            "endTime": int(end_date.timestamp() * 1000),
            "limit": 1000,
        }
        response = self.session.get(url, params=params, timeout=30)
        response.raise_for_status()
        rows = response.json()
        if not rows:
            return pd.DataFrame()
        frame = pd.DataFrame(rows)
        frame["timestamp"] = pd.to_datetime(frame["fundingTime"], unit="ms", utc=True)
        frame["funding_rate"] = pd.to_numeric(frame["fundingRate"], errors="coerce")
        return frame[["timestamp", "funding_rate"]].dropna()

    def fetch_crypto_borrow_rate(self, asset: str = "BTC", start_date: Optional[datetime] = None, end_date: Optional[datetime] = None) -> pd.DataFrame:
        """Attempt to derive a daily historical borrow-rate series.

        Strategy:
        - Try to derive borrow-rate from perpetual-futures funding rates (public endpoint).
        - If funding-rate history is available, compute daily mean funding * 3 (3 funding windows/day)
          and take absolute value as a conservative proxy for daily borrow costs.
        - If funding history is empty, fall back to a single-row placeholder (legacy behaviour).

        This is a proxy, not an authoritative borrow-rate. For production-grade borrow
        history use a dedicated exchange margin API (requires auth) or third-party data.
        """
        # default historical window if not provided
        if end_date is None:
            end_date = datetime.utcnow()
        if start_date is None:
            start_date = datetime(2020, 1, 1)

        perp_symbol = asset if asset.endswith("USDT") else f"{asset}USDT"

        # If user provided Binance API keys in env, try to pull official margin interest history
        api_key = os.getenv("BINANCE_API_KEY")
        api_secret = os.getenv("BINANCE_API_SECRET")
        margin_rows = None
        if api_key and api_secret:
            try:
                endpoint = "/sapi/v1/margin/interestHistory"
                params = {
                    "asset": asset.upper(),
                    "startTime": int(start_date.timestamp() * 1000) if start_date is not None else None,
                    "endTime": int(end_date.timestamp() * 1000) if end_date is not None else None,
                    "limit": 1000,
                }
                # remove None values
                params = {k: v for k, v in params.items() if v is not None}
                margin_rows = self._signed_get("https://api.binance.com", endpoint, params, api_key, api_secret)
            except Exception:
                margin_rows = None

        # If signed margin history returned useful data, parse it
        if margin_rows:
            try:
                frame = pd.DataFrame(margin_rows)
                # try to find rate-like columns
                if "interestRate" in frame.columns:
                    frame["timestamp"] = pd.to_datetime(frame.get("time", frame.get("timestamp", pd.Series())), unit="ms", utc=True)
                    frame["borrow_rate"] = pd.to_numeric(frame["interestRate"], errors="coerce")
                    df = frame[["timestamp", "borrow_rate"]].dropna()
                    df["asset"] = asset.upper()
                    return df
                # if interest and principal exist, compute rate = interest / principal
                if "interest" in frame.columns and "principal" in frame.columns:
                    frame["timestamp"] = pd.to_datetime(frame.get("time", frame.get("timestamp", pd.Series())), unit="ms", utc=True)
                    frame["borrow_rate"] = pd.to_numeric(frame["interest"], errors="coerce") / pd.to_numeric(frame["principal"], errors="coerce")
                    df = frame[["timestamp", "borrow_rate"]].dropna()
                    df["asset"] = asset.upper()
                    return df
            except Exception:
                pass

        # Fallback: derive from public funding rates
        try:
            funding = self.fetch_funding_rates(perp_symbol, start_date, end_date)
        except Exception:
            funding = pd.DataFrame()

        if funding.empty:
            timestamp = pd.Timestamp.utcnow().normalize()
            return pd.DataFrame([
                {"timestamp": timestamp, "asset": asset.upper(), "borrow_rate": 0.08}
            ])

        # resample to daily mean funding rate; funding rate entries are per funding event
        funding = funding.set_index("timestamp").sort_index()
        # funding_rate already numeric
        daily = funding["funding_rate"].resample("D").mean().dropna()
        # conservative proxy: daily borrow ~ abs(mean_funding_per_day) * 3 (three funding periods/day)
        borrow = (daily.abs() * 3).rename("borrow_rate").to_frame()
        borrow["asset"] = asset.upper()
        borrow = borrow.reset_index()
        return borrow

    def fetch_risk_free_rate(self) -> pd.DataFrame:
        """Best-effort daily USD risk-free proxy.

        If FRED is configured in the environment, use the FRED API to fetch a
        short-term Treasury series (TB3MS) over the requested window; otherwise
        return a single-row placeholder.
        """
        # try to use FRED if available
        try:
            fred_api_key = os.getenv("FRED_API_KEY")
            if fred_api_key:
                fred = FREDDownloader()
                # default window: last 10 years
                end = datetime.utcnow()
                start = datetime(end.year - 10, end.month, end.day)
                data = fred.get_economic_data("TB3MS", start, end)
                if data:
                    df = pd.DataFrame(data)
                    # Data from FREDDownloader is OHLCV-style with 'timestamp' and 'close'
                    df = df[["timestamp", "close"]].rename(columns={"close": "risk_free_rate"})
                    return df
        except Exception:
            pass

        timestamp = pd.Timestamp.utcnow().normalize()
        return pd.DataFrame([
            {"timestamp": timestamp, "series": "DGS3MO", "risk_free_rate": 0.05}
        ])

    def _write_frame(self, frame: pd.DataFrame, path: str, output_format: str):
        ensure_directory_exists(os.path.dirname(path))
        if output_format in {"parquet"}:
            write_parquet(frame.to_dict("records"), path)
        else:
            frame.to_csv(path, index=False)

    def download_crypto_market_data(
        self,
        symbols: List[str],
        start_date: datetime,
        end_date: datetime,
        interval: str = "1h",
        output_format: Optional[str] = None,
    ):
        """Download spot, perp and funding data for the provided symbols."""
        output_format = output_format or DEFAULT_OUTPUT_FORMAT
        for symbol in symbols:
            spot = self.fetch_spot_prices(symbol, interval, start_date, end_date)
            perp = self.fetch_perp_prices(symbol, interval, start_date, end_date)
            funding = self.fetch_funding_rates(symbol, start_date, end_date)

            symbol_slug = symbol.lower().replace("/", "").replace("-", "")
            suffix = "parquet" if output_format == "parquet" else "csv"

            if not spot.empty:
                self._write_frame(spot, os.path.join(RAW_MARKET_PATH, "spot", f"{symbol_slug}_{interval}.{suffix}"), output_format)
                self.logger.info(f"Saved spot data for {symbol}")

            if not perp.empty:
                self._write_frame(perp, os.path.join(RAW_MARKET_PATH, "perp", f"{symbol_slug}_{interval}.{suffix}"), output_format)
                self.logger.info(f"Saved perp data for {symbol}")

            if not funding.empty:
                self._write_frame(funding, os.path.join(RAW_MARKET_PATH, "funding", f"{symbol_slug}_funding.{suffix}"), output_format)
                self.logger.info(f"Saved funding rates for {symbol}")

    def download_reference_rates(self, asset: str = "BTC", start_date: Optional[datetime] = None, end_date: Optional[datetime] = None, output_format: Optional[str] = None):
        """Write daily borrow and risk-free rate series.

        `start_date`/`end_date` are passed to the borrow-rate derivation routine.
        """
        output_format = output_format or DEFAULT_OUTPUT_FORMAT
        suffix = "parquet" if output_format == "parquet" else "csv"

        risk_free = self.fetch_risk_free_rate()
        borrow = self.fetch_crypto_borrow_rate(asset, start_date=start_date, end_date=end_date)

        self._write_frame(risk_free, os.path.join(RAW_MACRO_PATH, f"risk_free_daily.{suffix}"), output_format)
        self._write_frame(borrow, os.path.join(RAW_MARKET_PATH, "borrow", f"{asset.lower()}_borrow_daily.{suffix}"), output_format)
        self.logger.info("Saved reference rate series")
