"""
Rates Carry Strategy - Data Acquisition Module
Fetches government bond yields and calculates roll-down returns
"""

import pandas as pd
import numpy as np
import yfinance as yf
from fredapi import Fred
import yaml
from datetime import datetime
from typing import Dict, Tuple


class RatesDataAcquisition:
    """Fetch yield curves and bond price data"""
    
    def __init__(self, config_path: str = 'config.yaml'):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.countries = self.config['data']['countries']
        self.maturities = self.config['data']['maturities']
        self.start_date = self.config['data']['start_date']
        self.end_date = self.config['data']['end_date'] or datetime.today().strftime('%Y-%m-%d')
        
        import os
        fred_api_key = os.getenv('FRED_API_KEY', None)
        self.fred = Fred(api_key=fred_api_key) if fred_api_key else None
    
    def fetch_yield_curves(self) -> pd.DataFrame:
        """Download government bond yields from FRED"""
        print(f"Fetching yield data for {len(self.countries)} countries...")
        
        # FRED series codes for government yields
        yield_series = {
            ('US', 2): 'DGS2',
            ('US', 5): 'DGS5',
            ('US', 7): 'DGS7',
            ('US', 10): 'DGS10',
            ('US', 30): 'DGS30',
            ('Germany', 2): 'IRLTLT01DEM156N',
            ('Germany', 10): 'IRLTLT01DEM156N',
            ('UK', 10): 'IRLTLT01GBM156N',
            ('Japan', 10): 'IRLTLT01JPM156N',
            ('Australia', 10): 'IRLTLT01AUM156N',
            ('Canada', 10): 'IRLTLT01CAM156N',
        }
        
        yields_data = {}
        
        for (country, maturity), series_code in yield_series.items():
            key = f"{country}_{maturity}Y"
            try:
                if self.fred:
                    data = self.fred.get_series(series_code,
                                               observation_start=self.start_date,
                                               observation_end=self.end_date)
                    yields_data[key] = data
                    print(f"  ✓ {key}: {len(data)} observations")
            except Exception as e:
                print(f"  ✗ {key}: {e}")
        
        df_yields = pd.DataFrame(yields_data)
        df_yields = df_yields.resample('D').ffill()
        
        print(f"\nYield data shape: {df_yields.shape}")
        return df_yields
    
    def calculate_rolldown(self, yields_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate roll-down return for each bond"""
        print("\nCalculating roll-down yields...")
        
        roll_period = self.config['signals']['roll_period']  # days
        
        rolldown = pd.DataFrame(index=yields_df.index)
        
        # For each maturity, calculate expected return from rolling down curve
        for col in yields_df.columns:
            if '10Y' in col:
                # 10Y bond rolling to 9.75Y
                country = col.split('_')[0]
                current_yield = yields_df[col]
                
                # Approximate yield at 9.75Y (roll_period/252 years shorter)
                # Use linear interpolation from 7Y and 10Y
                if f"{country}_7Y" in yields_df.columns:
                    nearby_yield = yields_df[f"{country}_7Y"]
                    roll_yield = current_yield - (current_yield - nearby_yield) * (roll_period/252) / 3
                else:
                    roll_yield = current_yield
                
                # Roll-down return = (current_yield - roll_yield) * duration
                duration = 9.0  # Approx duration of 10Y bond
                rolldown[col] = (current_yield - roll_yield) * duration
        
        print(f"Roll-down data shape: {rolldown.shape}")
        return rolldown
    
    def save_data(self, yields_df: pd.DataFrame, rolldown_df: pd.DataFrame):
        """Save data to CSV"""
        import os
        os.makedirs('data', exist_ok=True)
        
        yields_df.to_csv('data/bond_yields.csv')
        rolldown_df.to_csv('data/rolldown.csv')
        print("\n✓ Data saved to data/ directory")
    
    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load saved data"""
        yields = pd.read_csv('data/bond_yields.csv', index_col=0, parse_dates=True)
        rolldown = pd.read_csv('data/rolldown.csv', index_col=0, parse_dates=True)
        return yields, rolldown


if __name__ == "__main__":
    rates_data = RatesDataAcquisition()
    yields = rates_data.fetch_yield_curves()
    rolldown = rates_data.calculate_rolldown(yields)
    rates_data.save_data(yields, rolldown)
