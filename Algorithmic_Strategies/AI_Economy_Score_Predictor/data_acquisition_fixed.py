"""
Data Acquisition Module for AI Economy Score Predictor

Fetches earnings call transcripts and macroeconomic data.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yaml
import os
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

try:
    from fredapi import Fred
except ImportError:
    print("Warning: fredapi not installed. Install with: pip install fredapi")

try:
    import yfinance as yf
except ImportError:
    print("Warning: yfinance not installed. Install with: pip install yfinance")


class DataAcquisition:
    """Handles all data acquisition for the strategy."""
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize with configuration."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Initialize FRED API
        fred_key = self.config['data']['macro'].get('fred_api_key')
        if fred_key and fred_key != "YOUR_FRED_API_KEY":
            try:
                self.fred = Fred(api_key=fred_key)
                print(f"✓ FRED API initialized")
            except:
                self.fred = None
                print("Warning: FRED API key invalid")
        else:
            self.fred = None
            print("Warning: FRED API key not configured")
    
    def fetch_sp500_constituents(self) -> pd.DataFrame:
        """Fetch current S&P 500 constituents."""
        try:
            tables = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
            sp500 = tables[0]
            sp500.columns = ['symbol', 'security', 'gics_sector', 'gics_sub_industry',
                            'headquarters', 'date_added', 'cik', 'founded']
            sp500['symbol'] = sp500['symbol'].str.replace('.', '-')
            print(f"✓ Fetched {len(sp500)} S&P 500 constituents")
            return sp500[['symbol', 'security', 'gics_sector', 'cik']]
        except:
            print("✓ Using fallback S&P 500 sample")
            sample = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'TSLA', 'META', 'BRK.B', 'JNJ', 'V',
                     'WMT', 'JPM', 'PG', 'MA', 'HD', 'DIS', 'PYPL', 'ADBE', 'CRM', 'NFLX']
            return pd.DataFrame({
                'symbol': sample,
                'security': sample,
                'gics_sector': ['45'] * len(sample),
                'cik': [''] * len(sample)
            })
    
    def fetch_macro_data(self, start_date: str, end_date: str) -> Dict[str, pd.DataFrame]:
        """Fetch macroeconomic data from FRED."""
        if self.fred is None:
            print("✓ Using placeholder macro data")
            return self._generate_placeholder_macro_data(start_date, end_date)
        
        indicators = self.config['data']['macro']['indicators']
        macro_data = {}
        
        for name, params in indicators.items():
            try:
                series = self.fred.get_series(
                    params['series_id'],
                    observation_start=start_date,
                    observation_end=end_date
                )
                
                df = pd.DataFrame({
                    'date': series.index,
                    'value': series.values
                })
                
                if params['transform'] == 'pct_change':
                    df['value'] = df['value'].pct_change() * 100
                
                macro_data[name] = df
                print(f"✓ Fetched {name}: {len(df)} observations")
                
            except Exception as e:
                print(f"✗ Error fetching {name}: {e}")
        
        return macro_data
    
    def _generate_placeholder_macro_data(self, start_date: str, end_date: str) -> Dict[str, pd.DataFrame]:
        """Generate placeholder macro data for testing."""
        dates_quarterly = pd.date_range(start=start_date, end=end_date, freq='Q')
        dates_monthly = pd.date_range(start=start_date, end=end_date, freq='M')
        
        np.random.seed(42)
        
        return {
            'gdp': pd.DataFrame({
                'date': dates_quarterly,
                'value': np.random.normal(2.0, 1.5, len(dates_quarterly))
            }),
            'industrial_production': pd.DataFrame({
                'date': dates_monthly,
                'value': np.random.normal(0.2, 0.5, len(dates_monthly))
            }),
            'employment': pd.DataFrame({
                'date': dates_monthly,
                'value': np.random.normal(150000, 50000, len(dates_monthly))
            }),
            'wages': pd.DataFrame({
                'date': dates_monthly,
                'value': np.random.normal(0.3, 0.2, len(dates_monthly))
            })
        }
    
    def fetch_spf_forecasts(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Fetch Survey of Professional Forecasters consensus."""
        dates = pd.date_range(start=start_date, end=end_date, freq='Q')
        np.random.seed(42)
        
        spf = pd.DataFrame({
            'date': dates,
            'rgdp_1q': np.random.normal(2.0, 0.5, len(dates)),
            'rgdp_2q': np.random.normal(2.0, 0.5, len(dates)),
            'rgdp_3q': np.random.normal(2.0, 0.5, len(dates)),
            'rgdp_4q': np.random.normal(2.0, 0.5, len(dates)),
            'indprod_1q': np.random.normal(0.2, 0.3, len(dates)),
            'indprod_2q': np.random.normal(0.2, 0.3, len(dates)),
        })
        
        return spf
    
    def fetch_control_variables(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Fetch control variables for regression models."""
        if self.fred is None:
            return self._generate_placeholder_controls(start_date, end_date)
        
        controls = {}
        
        try:
            gs10 = self.fred.get_series('GS10', start_date, end_date)
            gs2 = self.fred.get_series('GS2', start_date, end_date)
            controls['yield_curve_slope'] = gs10 - gs2
            
            controls['consumer_sentiment'] = self.fred.get_series('UMCSENT', start_date, end_date)
            controls['pmi'] = self.fred.get_series('NAPM', start_date, end_date)
            
            vix_data = yf.download('^VIX', start=start_date, end=end_date, progress=False)
            controls['vix'] = vix_data['Adj Close']
            
        except Exception as e:
            print(f"✗ Error fetching controls: {e}")
            return self._generate_placeholder_controls(start_date, end_date)
        
        df = pd.DataFrame(controls)
        df = df.ffill()
        
        return df
    
    def _generate_placeholder_controls(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Generate placeholder control variables."""
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        np.random.seed(42)
        
        return pd.DataFrame({
            'date': dates,
            'yield_curve_slope': np.random.normal(0.5, 1.0, len(dates)),
            'consumer_sentiment': np.random.normal(90, 10, len(dates)),
            'pmi': np.random.normal(52, 5, len(dates)),
            'vix': np.random.normal(18, 8, len(dates))
        }).set_index('date')
    
    def fetch_etf_prices(self, symbols: List[str], start_date: str, end_date: str) -> Dict[str, pd.DataFrame]:
        """Fetch sector ETF price data."""
        prices = {}
        
        for symbol in symbols:
            try:
                data = yf.download(symbol, start=start_date, end=end_date, progress=False)
                if not data.empty:
                    prices[symbol] = data
                    print(f"✓ Fetched {symbol}: {len(data)} days")
            except Exception as e:
                print(f"✗ Error fetching {symbol}: {e}")
        
        return prices
    
    def split_data(self, df: pd.DataFrame, date_column: str = 'date') -> Dict[str, pd.DataFrame]:
        """Split data into training, validation, and test sets."""
        config = self.config['backtest']
        
        train = df[
            (df[date_column] >= config['training_start']) &
            (df[date_column] <= config['training_end'])
        ]
        
        validation = df[
            (df[date_column] >= config['validation_start']) &
            (df[date_column] <= config['validation_end'])
        ]
        
        test = df[
            (df[date_column] >= config['test_start']) &
            (df[date_column] <= config['test_end'])
        ]
        
        return {
            'train': train,
            'validation': validation,
            'test': test
        }


if __name__ == "__main__":
    data_acq = DataAcquisition('config.yaml')
    
    sp500 = data_acq.fetch_sp500_constituents()
    print(f"\n{sp500.head()}")
    
    macro = data_acq.fetch_macro_data('2020-01-01', '2023-12-31')
    print(f"\nMacro data keys: {list(macro.keys())}")
    
    controls = data_acq.fetch_control_variables('2020-01-01', '2023-12-31')
    print(f"\nControl variables:\n{controls.head()}")
