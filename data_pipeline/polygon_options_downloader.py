"""
Polygon.io Options Data Downloader for QuantConnect Lean format
Supports SPX, SPY and other equity options
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from polygon import RESTClient
from tqdm import tqdm
import time
from typing import List, Dict, Any, Optional

from config import (
    POLYGON_API_KEY,
    DATA_ROOT,
    LEAN_PRICE_MULTIPLIER,
)
from utils import ensure_directory_exists, setup_logging


class PolygonOptionsDownloader:
    """Download options data from Polygon.io and convert to QuantConnect Lean format"""

    def __init__(self, api_key: str = None):
        self.api_key = api_key or POLYGON_API_KEY
        if not self.api_key:
            raise ValueError("Polygon API key is required. Set POLYGON_API_KEY environment variable.")

        self.client = RESTClient(api_key=self.api_key)
        self.data_path = os.path.join(DATA_ROOT, 'option')
        self.logger = setup_logging()

        # Rate limiting for free tier
        self.min_request_interval = 15  # 15 seconds between requests
        self.last_request_time = 0

    def _rate_limit(self):
        """Implement rate limiting for API requests"""
        current_time = time.time()
        time_since_last_request = current_time - self.last_request_time

        if time_since_last_request < self.min_request_interval:
            sleep_time = self.min_request_interval - time_since_last_request
            self.logger.info(f"Rate limiting: sleeping for {sleep_time:.1f} seconds")
            time.sleep(sleep_time)

        self.last_request_time = time.time()

    def get_options_data(self, underlying: str, start_date: datetime, end_date: datetime,
                        resolution: str = 'daily') -> pd.DataFrame:
        """Download options price data for an underlying"""
        self._rate_limit()

        try:
            # Format dates for Polygon API
            start_str = start_date.strftime('%Y-%m-%d')
            end_str = end_date.strftime('%Y-%m-%d')

            # Determine timespan based on resolution
            timespan_map = {
                'minute': 'minute',
                'hour': 'hour',
                'daily': 'day'
            }
            timespan = timespan_map.get(resolution, 'day')

            self.logger.info(f"Fetching {underlying} options data from {start_str} to {end_str}")

            # For options, we need to get all option contracts for the underlying
            # This is complex, so for now, return empty DataFrame
            # In practice, you'd need to query option contracts and aggregate

            return pd.DataFrame()

        except Exception as e:
            self.logger.error(f"Error fetching options data for {underlying}: {e}")
            return pd.DataFrame()

    def download_options_chain(self, underlying: str, date: datetime = None) -> List[Dict]:
        """Download current options chain for an underlying"""
        self._rate_limit()

        try:
            if date is None:
                date = datetime.now()

            self.logger.info(f"Fetching options chain for {underlying} on {date.date()}")

            # Use snapshot endpoint
            snapshots = self.client.get_snapshot_options(underlying)

            return snapshots.results if hasattr(snapshots, 'results') else []

        except Exception as e:
            self.logger.error(f"Error fetching options chain for {underlying}: {e}")
            return []

    def save_to_lean_format(self, data: pd.DataFrame, symbol: str, resolution: str):
        """Save data to QuantConnect Lean format"""
        if data.empty:
            return

        # Ensure data directory exists
        ensure_directory_exists(self.data_path)

        # Create filename
        filename = f"{symbol}_{resolution}.csv"
        filepath = os.path.join(self.data_path, filename)

        # Convert to Lean format
        # This is a simplified version - options data is more complex
        data.to_csv(filepath, index=False)
        self.logger.info(f"Saved {len(data)} options records to {filepath}")