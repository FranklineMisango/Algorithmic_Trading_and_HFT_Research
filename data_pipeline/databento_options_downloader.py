"""
Databento Options Data Downloader for QuantConnect Lean format
Supports SPY, QQQ, AAPL and other equity options via OPRA.PILLAR dataset
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Any, Optional, Union
from tqdm import tqdm
import time

try:
    import databento as db
    from databento import Schema, Encoding, SType
except ImportError:
    raise ImportError("Please install databento: pip install databento")

from config import (
    DATA_BENTO_API_KEY,
    DATA_ROOT,
    LEAN_PRICE_MULTIPLIER,
)
from utils import ensure_directory_exists, setup_logging


class DatabentoOptionsDownloader:
    """Download options data from Databento and convert to QuantConnect Lean format"""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or DATA_BENTO_API_KEY
        if not self.api_key:
            raise ValueError("Databento API key is required. Set DATA_BENTO_API_KEY environment variable.")
        
        # Initialize Databento client
        self.client = db.Historical(key=self.api_key)
        self.data_path = os.path.join(DATA_ROOT, 'option', 'usa')
        self.logger = setup_logging()
        
        # Rate limiting - Databento has generous limits but we'll be conservative
        self.min_request_interval = 1  # 1 second between requests
        self.last_request_time = 0
        
        # OPRA dataset for options
        self.dataset = 'OPRA.PILLAR'
        
    def _rate_limit(self):
        """Implement rate limiting for API requests"""
        current_time = time.time()
        time_since_last_request = current_time - self.last_request_time
        
        if time_since_last_request < self.min_request_interval:
            sleep_time = self.min_request_interval - time_since_last_request
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()
    
    def download_options(self, underlying: str, start_date: datetime, end_date: datetime,
                        resolution: str = 'daily', schema: str = 'trades', 
                        limit_contracts: Optional[int] = None,
                        filter_near_money: bool = True) -> None:
        """
        Download options data for a given underlying symbol
        
        Args:
            underlying: Underlying ticker symbol (e.g., 'SPY', 'AAPL')
            start_date: Start date for data
            end_date: End date for data
            resolution: Data resolution ('tick', 'second', 'minute', 'hour', 'daily')
            schema: Databento schema ('trades', 'cmbp-1', 'tcbbo', 'ohlcv-1d', etc.)
            limit_contracts: Maximum number of contracts to download (None = all)
            filter_near_money: If True, only download contracts near the current price
        """
        self.logger.info(f"Downloading Databento options for {underlying} from {start_date.date()} to {end_date.date()}")
        
        try:
            # Step 1: Get all option contract definitions for the underlying
            option_contracts = self._get_option_definitions(underlying, start_date, end_date)
            
            if not option_contracts:
                self.logger.warning(f"No option contracts found for {underlying}")
                return
            
            self.logger.info(f"Found {len(option_contracts)} option contracts for {underlying}")
            
            # Step 2: Filter contracts if needed
            if filter_near_money:
                option_contracts = self._filter_near_money_options(option_contracts, limit=limit_contracts or 50)
                self.logger.info(f"Filtered to {len(option_contracts)} near-the-money contracts")
            elif limit_contracts:
                option_contracts = option_contracts[:limit_contracts]
                self.logger.info(f"Limited to {len(option_contracts)} contracts")
            
            # Step 3: Determine schema based on resolution
            schema_to_use = self._get_schema_for_resolution(resolution, schema)
            
            # Step 4: Download data for each contract (or in batches)
            self._download_option_data_batch(option_contracts, underlying, start_date, end_date, schema_to_use)
            
        except Exception as e:
            self.logger.error(f"Error downloading options for {underlying}: {e}")
            import traceback
            traceback.print_exc()
    
    def _get_option_definitions(self, underlying: str, start_date: datetime, end_date: datetime) -> List[Dict[str, Any]]:
        """Get option contract definitions from Databento"""
        self._rate_limit()
        
        try:
            self.logger.info(f"Fetching option definitions for {underlying}...")
            
            # Use parent symbology to get all options for the underlying
            # Format: TICKER.OPT (e.g., 'SPY.OPT')
            parent_symbol = f"{underlying}.OPT"
            
            data = self.client.timeseries.get_range(
                dataset=self.dataset,
                symbols=[parent_symbol],
                schema='definition',
                start=start_date.strftime('%Y-%m-%d'),
                end=end_date.strftime('%Y-%m-%d'),
                stype_in='parent'  # CRITICAL: Use 'parent' to get options for the underlying
            )
            
            # Convert to DataFrame
            df = data.to_df()
            
            if df.empty:
                self.logger.warning(f"No option definitions found for {underlying}")
                return []
            
            # Extract relevant contract information
            contracts = []
            for idx, row in df.iterrows():
                try:
                    contracts.append({
                        'symbol': row['symbol'],
                        'raw_symbol': row.get('raw_symbol', row['symbol']),
                        'strike': row.get('strike_price', 0) / 10000,  # Convert from Databento format
                        'expiration': row.get('expiration', ''),
                        'option_type': 'call' if 'C' in row['symbol'] else 'put',
                        'underlying': underlying,
                    })
                except Exception as e:
                    self.logger.warning(f"Error parsing contract {row.get('symbol', 'unknown')}: {e}")
                    continue
            
            self.logger.info(f"Parsed {len(contracts)} option contracts from definitions")
            return contracts
            
        except Exception as e:
            self.logger.error(f"Error fetching option definitions for {underlying}: {e}")
            return []
    
    def _filter_near_money_options(self, contracts: List[Dict], limit: int = 50) -> List[Dict]:
        """
        Filter options to only include near-the-money contracts
        Returns a subset of contracts sorted by strike price
        """
        if not contracts:
            return []
        
        # Group by expiration and option type
        expirations = {}
        for contract in contracts:
            exp = contract.get('expiration', '')
            opt_type = contract.get('option_type', 'call')
            key = (exp, opt_type)
            
            if key not in expirations:
                expirations[key] = []
            expirations[key].append(contract)
        
        # For each expiration/type, take contracts around middle strikes
        filtered = []
        for key, exp_contracts in expirations.items():
            # Sort by strike price
            sorted_contracts = sorted(exp_contracts, key=lambda x: x.get('strike', 0))
            
            # Take middle section (near ATM)
            n = len(sorted_contracts)
            if n <= 10:
                filtered.extend(sorted_contracts)
            else:
                # Take middle 40% of strikes
                start_idx = int(n * 0.3)
                end_idx = int(n * 0.7)
                filtered.extend(sorted_contracts[start_idx:end_idx])
        
        # Limit total contracts
        return filtered[:limit]
    
    def _get_schema_for_resolution(self, resolution: str, user_schema: str = None) -> str:
        """Determine the appropriate Databento schema based on resolution"""
        if user_schema:
            # User explicitly specified a schema
            return user_schema
        
        # Map resolution to schema
        schema_map = {
            'tick': 'trades',      # Trade ticks
            'second': 'ohlcv-1s',  # 1-second OHLCV
            'minute': 'ohlcv-1m',  # 1-minute OHLCV
            'hour': 'ohlcv-1h',    # 1-hour OHLCV
            'daily': 'ohlcv-1d'    # Daily OHLCV
        }
        
        return schema_map.get(resolution, 'ohlcv-1d')
    
    def _download_option_data_batch(self, contracts: List[Dict], underlying: str,
                                   start_date: datetime, end_date: datetime,
                                   schema: str) -> None:
        """Download option data in batches"""
        
        # Databento can handle multiple symbols in one request
        # But for large numbers, we'll batch them
        batch_size = 10  # Download 10 contracts at a time
        
        for i in tqdm(range(0, len(contracts), batch_size), desc=f"Downloading {underlying} options"):
            batch = contracts[i:i + batch_size]
            self._download_contract_batch(batch, underlying, start_date, end_date, schema)
    
    def _download_contract_batch(self, contracts: List[Dict], underlying: str,
                                start_date: datetime, end_date: datetime,
                                schema: str) -> None:
        """Download data for a batch of option contracts"""
        self._rate_limit()
        
        try:
            # Extract symbols from contracts
            symbols = [contract['raw_symbol'] for contract in contracts]
            
            if not symbols:
                return
            
            self.logger.debug(f"Downloading {len(symbols)} contracts with schema '{schema}'")
            
            # Download data from Databento
            data = self.client.timeseries.get_range(
                dataset=self.dataset,
                symbols=symbols,
                schema=schema,
                start=start_date.strftime('%Y-%m-%d'),
                end=end_date.strftime('%Y-%m-%d'),
                limit=100000  # Limit records per request
            )
            
            # Convert to DataFrame
            df = data.to_df()
            
            if df.empty:
                self.logger.warning(f"No data returned for batch of {len(symbols)} contracts")
                return
            
            self.logger.info(f"Downloaded {len(df)} records for {len(symbols)} contracts")
            
            # Process and save data for each contract
            for contract in contracts:
                symbol = contract['raw_symbol']
                contract_df = df[df['symbol'] == symbol] if 'symbol' in df.columns else pd.DataFrame()
                
                if not contract_df.empty:
                    self._save_option_data(contract_df, contract, underlying, schema)
            
        except Exception as e:
            self.logger.error(f"Error downloading batch: {e}")
    
    def _save_option_data(self, df: pd.DataFrame, contract: Dict, underlying: str, schema: str) -> None:
        """Save option data in QuantConnect Lean format"""
        try:
            if df.empty:
                return
            
            # Parse contract details
            symbol = contract['symbol']
            strike = contract.get('strike', 0)
            option_type = contract.get('option_type', 'call')
            expiration = contract.get('expiration', '')
            
            # Process based on schema type
            if 'ohlcv' in schema.lower():
                processed_df = self._process_ohlcv_data(df)
            elif schema == 'trades':
                processed_df = self._process_trade_data(df)
            elif 'cmbp' in schema.lower() or 'bbo' in schema.lower():
                processed_df = self._process_quote_data(df)
            else:
                self.logger.warning(f"Unknown schema type: {schema}, saving raw data")
                processed_df = df
            
            # Create directory structure for options
            # Format: data/option/usa/underlying/resolution/
            resolution_dir = 'daily' if 'ohlcv-1d' in schema else 'minute' if 'ohlcv-1m' in schema else 'tick'
            option_dir = os.path.join(self.data_path, underlying.lower(), resolution_dir)
            ensure_directory_exists(option_dir)
            
            # Create filename (simplified for this example)
            # Real implementation would follow Lean's naming convention
            date_str = datetime.now().strftime('%Y%m%d')
            filename = f"{symbol.replace(' ', '_')}_{date_str}.csv"
            filepath = os.path.join(option_dir, filename)
            
            # Save to CSV
            processed_df.to_csv(filepath, index=False)
            self.logger.debug(f"Saved {len(processed_df)} records to {filepath}")
            
        except Exception as e:
            self.logger.error(f"Error saving option data for {contract.get('symbol', 'unknown')}: {e}")
    
    def _process_ohlcv_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process OHLCV data to Lean format"""
        if df.empty:
            return df
        
        # Map Databento columns to Lean format
        result = pd.DataFrame()
        
        # Handle timestamp
        if 'ts_event' in df.columns:
            result['timestamp'] = pd.to_datetime(df['ts_event'])
        
        # OHLCV columns
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col in df.columns:
                if col != 'volume':
                    # Convert prices to Lean format (integer in deci-cents)
                    result[col] = (df[col] * LEAN_PRICE_MULTIPLIER).astype(int)
                else:
                    result[col] = df[col]
        
        return result
    
    def _process_trade_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process trade tick data to Lean format"""
        if df.empty:
            return df
        
        result = pd.DataFrame()
        
        # Timestamp
        if 'ts_event' in df.columns:
            result['timestamp'] = pd.to_datetime(df['ts_event'])
        
        # Trade price and size
        if 'price' in df.columns:
            result['price'] = (df['price'] * LEAN_PRICE_MULTIPLIER).astype(int)
        if 'size' in df.columns:
            result['size'] = df['size']
        
        return result
    
    def _process_quote_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process quote/BBO data to Lean format"""
        if df.empty:
            return df
        
        result = pd.DataFrame()
        
        # Timestamp
        if 'ts_event' in df.columns:
            result['timestamp'] = pd.to_datetime(df['ts_event'])
        
        # Bid/Ask
        for side in ['bid', 'ask']:
            price_col = f'{side}_px_00'  # Databento format for top of book
            size_col = f'{side}_sz_00'
            
            if price_col in df.columns:
                result[f'{side}_price'] = (df[price_col] * LEAN_PRICE_MULTIPLIER).astype(int)
            if size_col in df.columns:
                result[f'{side}_size'] = df[size_col]
        
        return result


def main():
    """Test the Databento options downloader"""
    from datetime import datetime, timedelta
    
    downloader = DatabentoOptionsDownloader()
    
    # Test with SPY options
    end_date = datetime.now() - timedelta(days=1)
    start_date = end_date - timedelta(days=7)
    
    downloader.download_options(
        underlying='SPY',
        start_date=start_date,
        end_date=end_date,
        resolution='daily',
        schema='trades',
        limit_contracts=5,
        filter_near_money=True
    )


if __name__ == "__main__":
    main()
