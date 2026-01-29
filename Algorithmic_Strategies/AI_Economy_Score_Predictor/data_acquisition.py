"""
Data Acquisition Module - Real Data via FRED API
"""

import pandas as pd
import yaml
from typing import Dict, List
import warnings
warnings.filterwarnings('ignore')

from fredapi import Fred


class DataAcquisition:
    """Handles all data acquisition for the strategy."""
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize with configuration."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        fred_key = self.config['data']['macro'].get('fred_api_key')
        self.fred = Fred(api_key=fred_key)
        print(f"✓ FRED API initialized")
    
    def fetch_sp500_constituents(self) -> pd.DataFrame:
        """Load S&P 500 constituents from local file."""
        sp500 = pd.read_csv('constituents.csv')
        print(f"✓ Loaded {len(sp500)} S&P 500 constituents")
        return sp500
    
    def fetch_macro_data(self, start_date: str, end_date: str) -> Dict[str, pd.DataFrame]:
        """Fetch macro data from FRED API."""
        macro_data = {}
        
        series_map = {
            'gdp': 'GDP',
            'industrial_production': 'INDPRO',
            'employment': 'PAYEMS',
            'wages': 'CES0500000003'
        }
        
        for name, series_id in series_map.items():
            try:
                series = self.fred.get_series(series_id, observation_start=start_date, observation_end=end_date)
                df = pd.DataFrame({
                    'date': series.index,
                    'value': series.values
                })
                macro_data[name] = df
                print(f"✓ Fetched {name}: {len(df)} observations")
            except Exception as e:
                print(f"✗ Error fetching {name}: {e}")
        
        return macro_data
    
    def fetch_spf_forecasts(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Fetch SPF consensus from FRED."""
        dates = pd.date_range(start=start_date, end=end_date, freq='Q')
        
        spf = pd.DataFrame({
            'date': dates,
            'rgdp_1q': [self.fred.get_series('EXPRGDPQ', observation_start=d.strftime('%Y-%m-%d'), observation_end=d.strftime('%Y-%m-%d')).values[0] if len(self.fred.get_series('EXPRGDPQ', observation_start=d.strftime('%Y-%m-%d'), observation_end=d.strftime('%Y-%m-%d'))) > 0 else 2.0 for d in dates],
        })
        
        return spf
    
    def fetch_control_variables(self, start_date: str, end_date: str, pmi_df: pd.DataFrame = None) -> pd.DataFrame:
        """Fetch control variables from FRED, but use provided PMI DataFrame if given."""
        controls = {}
        try:
            gs10 = self.fred.get_series('GS10', observation_start=start_date, observation_end=end_date)
            gs2 = self.fred.get_series('GS2', observation_start=start_date, observation_end=end_date)
            controls['yield_curve_slope'] = gs10 - gs2
            print(f"✓ Fetched yield curve slope")
        except Exception as e:
            print(f"✗ Error fetching yield curve: {e}")
        try:
            controls['consumer_sentiment'] = self.fred.get_series('UMCSENT', observation_start=start_date, observation_end=end_date)
            print(f"✓ Fetched consumer sentiment")
        except Exception as e:
            print(f"✗ Error fetching sentiment: {e}")
        try:
            controls['unemployment_rate'] = self.fred.get_series('UNRATE', observation_start=start_date, observation_end=end_date)
            print(f"✓ Fetched unemployment rate")
        except Exception as e:
            print(f"✗ Error fetching unemployment: {e}")

        # Use provided PMI DataFrame if given
        if pmi_df is not None:
            # Expect columns: 'date' and 'pmi' (already cleaned in notebook)
            pmi_df = pmi_df.copy()
            pmi_df = pmi_df[(pmi_df['date'] >= pd.to_datetime(start_date)) & (pmi_df['date'] <= pd.to_datetime(end_date))]
            pmi_df = pmi_df.set_index('date').sort_index()
            controls['pmi'] = pmi_df['pmi']
            print(f"✓ Used local PMI data: {len(pmi_df)} rows")
        else:
            print("✗ No local PMI data provided; PMI not included in controls.")

        if not controls:
            raise ValueError("Failed to fetch any control variables")

        df = pd.DataFrame(controls)
        df = df.ffill()
        print(f"✓ Control variables: {len(df)} observations")
        return df
    
    def fetch_etf_prices(self, symbols: List[str], start_date: str, end_date: str) -> Dict[str, pd.DataFrame]:
        """Fetch sector ETF price data."""
        import yfinance as yf
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
