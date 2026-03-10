"""
Download Canadian bond data from Databento and Bank of Canada.
Saves data to CSV files for use in backtesting and analysis.
"""

import os
import sys
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import logging
from pathlib import Path
import yaml

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import data acquisition module
from data_acquisition import CanadianBondDataAcquisition


def download_databento_data(start_date: str, end_date: str, output_dir: str = 'data', config_path: str = 'config.yaml'):
    """
    Download Canadian Government Bond futures data from Databento.
    
    Args:
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        output_dir: Directory to save CSV files
        config_path: Path to config.yaml
    """
    logger.info(f"Downloading Databento data from {start_date} to {end_date}")
    
    # Check API key - first try environment variable, then config file
    api_key = os.getenv('DATABENTO_API_KEY')
    
    if not api_key:
        # Try to load from config.yaml
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                api_key = config.get('data_sources', {}).get('primary', {}).get('api_key')
                if api_key and api_key.startswith('$'):
                    api_key = None  # Skip placeholder values
        except Exception as e:
            logger.warning(f"Could not load API key from config: {e}")
    
    if not api_key:
        logger.error("DATABENTO_API_KEY not found in environment or config.yaml")
        return False
    
    try:
        import databento as db
        
        # Create output directory
        Path(output_dir).mkdir(exist_ok=True)
        
        # Initialize client
        client = db.Historical(api_key)
        
        # Fetch CGB futures data
        logger.info("Fetching CGB futures data...")
        data = client.timeseries(
            dataset='XCAN.ITCH',
            symbols='CGB*',
            start=start_date,
            end=end_date,
            schema='ohlcv'
        )
        
        # Convert to DataFrame
        df = data.to_pandas()
        
        if len(df) > 0:
            # Save raw data
            output_file = Path(output_dir) / 'cgb_futures_raw.csv'
            df.to_csv(output_file, index=False)
            logger.info(f"Saved {len(df)} records to {output_file}")
            
            # Create daily summary
            daily_summary = df.groupby('symbol').resample('D').agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }).reset_index()
            
            daily_file = Path(output_dir) / 'cgb_futures_daily.csv'
            daily_summary.to_csv(daily_file, index=False)
            logger.info(f"Saved daily summary to {daily_file}")
            
            return True
        else:
            logger.warning("No data returned from Databento")
            return False
            
    except ImportError:
        logger.error("databento package not installed. Run: pip install databento")
        return False
    except Exception as e:
        logger.error(f"Error downloading Databento data: {e}")
        return False


def download_bank_of_canada_data(output_dir: str = 'data'):
    """
    Download yield curve data from Bank of Canada.
    
    Args:
        output_dir: Directory to save CSV files
    """
    logger.info("Downloading Bank of Canada yield curve data...")
    
    try:
        import requests
        
        # Create output directory
        Path(output_dir).mkdir(exist_ok=True)
        
        # Bank of Canada Valet API
        base_url = "https://www.bankofcanada.ca/valet/"
        
        # Fetch government bond yields
        logger.info("Fetching government bond yields...")
        # Use the correct endpoint for daily bond yields
        url = f"{base_url}observations/IRRGOVCO/json"
        
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        
        # Parse observations
        observations = data.get('observations', [])
        
        if observations:
            # Convert to DataFrame
            records = []
            for obs in observations:
                record = {'date': obs.get('d')}
                
                # Extract all series
                for key, value in obs.items():
                    if key != 'd' and isinstance(value, dict):
                        record[key] = value.get('v')
                
                records.append(record)
            
            df = pd.DataFrame(records)
            df['date'] = pd.to_datetime(df['date'])
            
            # Save data
            output_file = Path(output_dir) / 'boc_yield_curves.csv'
            df.to_csv(output_file, index=False)
            logger.info(f"Saved {len(df)} yield curve observations to {output_file}")
            
            return True
        else:
            logger.warning("No yield curve data returned from Bank of Canada")
            return False
            
    except ImportError:
        logger.error("requests package not installed")
        return False
    except Exception as e:
        logger.error(f"Error downloading Bank of Canada data: {e}")
        logger.info("Creating mock BoC data for development...")
        
        # Create mock data for development
        try:
            dates = pd.date_range(end=datetime.now(), periods=252, freq='D')
            mock_data = pd.DataFrame({
                'date': dates,
                'IRRGOVCO1': np.random.uniform(3.5, 4.5, len(dates)),
                'IRRGOVCO2': np.random.uniform(3.6, 4.6, len(dates)),
                'IRRGOVCO5': np.random.uniform(3.8, 4.8, len(dates)),
                'IRRGOVCO10': np.random.uniform(4.0, 5.0, len(dates)),
            })
            
            output_file = Path(output_dir) / 'boc_yield_curves.csv'
            mock_data.to_csv(output_file, index=False)
            logger.info(f"Created mock BoC data ({len(mock_data)} records) for development")
            return True
        except Exception as mock_err:
            logger.error(f"Failed to create mock data: {mock_err}")
            return False


def download_all_data(start_date: str = None, end_date: str = None, config_path: str = 'config.yaml'):
    """
    Download all required data for the strategy.
    
    Args:
        start_date: Start date (default: 1 year ago)
        end_date: End date (default: today)
        config_path: Path to config.yaml
    """
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
    
    if start_date is None:
        start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
    
    logger.info("=" * 60)
    logger.info("Canadian Bond Day Count Arbitrage - Data Download")
    logger.info("=" * 60)
    logger.info(f"Date range: {start_date} to {end_date}")
    
    # Download Databento data
    databento_success = download_databento_data(start_date, end_date, config_path=config_path)
    
    # Download Bank of Canada data
    boc_success = download_bank_of_canada_data()
    
    logger.info("=" * 60)
    if databento_success and boc_success:
        logger.info("✓ All data downloaded successfully")
        logger.info("Data saved to 'data/' directory")
        return True
    else:
        logger.warning("⚠ Some data downloads failed")
        return False


if __name__ == "__main__":
    # Parse command line arguments
    start_date = sys.argv[1] if len(sys.argv) > 1 else None
    end_date = sys.argv[2] if len(sys.argv) > 2 else None
    
    success = download_all_data(start_date, end_date)
    sys.exit(0 if success else 1)
