"""
Utility functions for data pipeline
"""

import os
import csv
import zipfile
import pandas as pd
from datetime import datetime, timedelta
import pytz
from typing import List, Dict, Optional
import logging

def setup_logging(log_level=logging.INFO):
    """Setup logging configuration"""
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('data_pipeline.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def ensure_directory_exists(path: str):
    """Ensure directory exists, create if not"""
    os.makedirs(path, exist_ok=True)

def milliseconds_since_midnight(dt: datetime) -> int:
    """Convert datetime to milliseconds since midnight"""
    midnight = dt.replace(hour=0, minute=0, second=0, microsecond=0)
    delta = dt - midnight
    return int(delta.total_seconds() * 1000)

def format_lean_date(date: datetime) -> str:
    """Format date for Lean file naming"""
    return date.strftime("%Y%m%d")

def create_lean_tradebar_csv(data: List[Dict], symbol: str, date: datetime, resolution: str) -> str:
    """Create Lean format CSV content for TradeBar data"""
    csv_content = []
    
    for bar in data:
        if resolution == 'daily':
            # For daily data, use full date format YYYYMMDD HH:MM
            time_str = bar['timestamp'].strftime("%Y%m%d %H:%M")
        else:
            # For intraday data, use milliseconds since midnight
            time_str = milliseconds_since_midnight(bar['timestamp'])
        
        # Format: Time, Open, High, Low, Close, Volume
        row = [
            time_str,
            int(bar['open'] * 10000),  # Convert to deci-cents for equity
            int(bar['high'] * 10000),
            int(bar['low'] * 10000),
            int(bar['close'] * 10000),
            int(bar['volume'])
        ]
        csv_content.append(row)
    
    return csv_content

def create_lean_crypto_csv(data: List[Dict], symbol: str, date: datetime, resolution: str) -> str:
    """Create Lean format CSV content for Crypto data"""
    csv_content = []
    
    for bar in data:
        time_ms = milliseconds_since_midnight(bar['timestamp'])
        
        # Format: Time, Open, High, Low, Close, Volume
        row = [
            time_ms,
            float(bar['open']),  # Keep actual prices for crypto
            float(bar['high']),
            float(bar['low']),
            float(bar['close']),
            float(bar['volume'])
        ]
        csv_content.append(row)
    
    return csv_content

def write_lean_zip_file(csv_content: List[List], output_path: str, csv_filename: str):
    """Write CSV content to a zip file in Lean format"""
    ensure_directory_exists(os.path.dirname(output_path))
    
    with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        # Create CSV content string
        csv_string = ""
        for row in csv_content:
            csv_string += ",".join(str(item) for item in row) + "\n"
        
        # Add CSV to zip file
        zip_file.writestr(csv_filename, csv_string)

def validate_symbol(symbol: str, asset_type: str) -> bool:
    """Validate symbol format"""
    if asset_type == 'equity':
        return symbol.isalpha() and len(symbol) <= 10
    elif asset_type == 'crypto':
        return len(symbol) > 6 and symbol.endswith('USDT')
    return False

def get_trading_days(start_date: datetime, end_date: datetime) -> List[datetime]:
    """Get list of trading days between start and end date"""
    trading_days = []
    current_date = start_date
    
    while current_date <= end_date:
        # Skip weekends (Saturday=5, Sunday=6)
        if current_date.weekday() < 5:
            trading_days.append(current_date)
        current_date += timedelta(days=1)
    
    return trading_days

def convert_timezone(dt: datetime, from_tz: str, to_tz: str) -> datetime:
    """Convert datetime from one timezone to another"""
    from_timezone = pytz.timezone(from_tz)
    to_timezone = pytz.timezone(to_tz)
    
    # Localize the datetime to the source timezone
    localized_dt = from_timezone.localize(dt) if dt.tzinfo is None else dt
    
    # Convert to target timezone
    converted_dt = localized_dt.astimezone(to_timezone)
    
    return converted_dt

class DataValidator:
    """Data validation utilities"""
    
    @staticmethod
    def validate_ohlcv_data(data: Dict) -> bool:
        """Validate OHLCV data structure"""
        required_fields = ['open', 'high', 'low', 'close', 'volume', 'timestamp']
        
        for field in required_fields:
            if field not in data:
                return False
        
        # Validate OHLC logic
        if not (data['low'] <= data['open'] <= data['high'] and 
                data['low'] <= data['close'] <= data['high']):
            return False
        
        # Validate volume is non-negative
        if data['volume'] < 0:
            return False
        
        return True
    
    @staticmethod
    def clean_ohlcv_data(data: List[Dict]) -> List[Dict]:
        """Clean and validate OHLCV data"""
        cleaned_data = []
        
        for bar in data:
            if DataValidator.validate_ohlcv_data(bar):
                cleaned_data.append(bar)
        
        return cleaned_data
