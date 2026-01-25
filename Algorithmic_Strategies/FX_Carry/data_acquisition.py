"""
FX Carry Strategy - Data Acquisition Module

Downloads and manages:
1. Spot FX rates from Yahoo Finance
2. Central bank policy rates from FRED
3. Safe-haven and commodity indices
"""

import pandas as pd
import numpy as np
import yfinance as yf
from pandas_datareader import data as pdr
from fredapi import Fred
import yaml
from datetime import datetime
from typing import Dict, List, Tuple


class FXDataAcquisition:
    """Fetches spot FX rates and interest rate data for carry strategy"""
    
    def __init__(self, config_path: str = 'config.yaml'):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.currency_pairs = self.config['data']['currency_pairs']
        self.start_date = self.config['data']['start_date']
        self.end_date = self.config['data']['end_date'] or datetime.today().strftime('%Y-%m-%d')
        
        # FRED API key (set via environment variable)
        import os
        fred_api_key = os.getenv('FRED_API_KEY', None)
        self.fred = Fred(api_key=fred_api_key) if fred_api_key else None
        
    def fetch_spot_rates(self) -> pd.DataFrame:
        """
        Download daily spot FX rates from Yahoo Finance
        
        Returns:
            DataFrame with dates as index, currency pairs as columns
        """
        print(f"Fetching spot FX data for {len(self.currency_pairs)} pairs...")
        
        spot_data = {}
        
        for pair in self.currency_pairs:
            try:
                # Yahoo Finance format: EURUSD=X
                ticker = f"{pair}=X"
                data = yf.download(ticker, start=self.start_date, end=self.end_date, 
                                   progress=False)
                
                if not data.empty:
                    spot_data[pair] = data['Adj Close']
                    print(f"  ✓ {pair}: {len(data)} observations")
                else:
                    print(f"  ✗ {pair}: No data returned")
                    
            except Exception as e:
                print(f"  ✗ {pair}: Error - {e}")
        
        df_spots = pd.DataFrame(spot_data)
        df_spots.index.name = 'date'
        
        # Fill missing values with forward fill then backward fill
        df_spots = df_spots.ffill().bfill()
        
        print(f"\nSpot data shape: {df_spots.shape}")
        print(f"Date range: {df_spots.index[0]} to {df_spots.index[-1]}")
        
        return df_spots
    
    def fetch_interest_rates(self) -> pd.DataFrame:
        """
        Download central bank policy rates from FRED
        
        Returns:
            DataFrame with dates as index, currency codes as columns
        """
        print(f"\nFetching interest rate data from FRED...")
        
        # FRED series codes for policy rates
        rate_series = {
            'USD': 'DFF',           # Fed Funds Rate
            'EUR': 'ECBDFR',        # ECB Deposit Facility Rate
            'JPY': 'IRSTCB01JPM156N',  # Japan Policy Rate
            'GBP': 'GBRONTD',       # Bank of England Bank Rate
            'AUD': 'RBATCTR',       # RBA Cash Rate Target
            'NZD': 'NZOCRS',        # RBNZ Official Cash Rate
            'CAD': 'IRSTCB01CAM156N',  # Bank of Canada Rate
            'CHF': 'IRSTCB01CHM156N',  # SNB Policy Rate
        }
        
        if not self.fred:
            print("WARNING: FRED API key not set. Using placeholder rates.")
            # Return placeholder data
            dates = pd.date_range(self.start_date, self.end_date, freq='D')
            placeholder = pd.DataFrame(
                {curr: np.full(len(dates), 1.0) for curr in rate_series.keys()},
                index=dates
            )
            return placeholder
        
        rate_data = {}
        
        for currency, series_code in rate_series.items():
            try:
                series = self.fred.get_series(series_code, 
                                             observation_start=self.start_date,
                                             observation_end=self.end_date)
                
                if series is not None and not series.empty:
                    rate_data[currency] = series
                    print(f"  ✓ {currency}: {len(series)} observations")
                else:
                    print(f"  ✗ {currency}: No data")
                    
            except Exception as e:
                print(f"  ✗ {currency}: Error - {e}")
        
        df_rates = pd.DataFrame(rate_data)
        
        # Convert to daily frequency and forward fill
        df_rates = df_rates.resample('D').ffill()
        df_rates = df_rates.loc[self.start_date:self.end_date]
        
        print(f"\nInterest rate data shape: {df_rates.shape}")
        
        return df_rates
    
    def calculate_carry(self, spot_df: pd.DataFrame, rate_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate interest rate differential (carry) for each currency pair
        
        Args:
            spot_df: Spot FX rates
            rate_df: Interest rates by currency
            
        Returns:
            DataFrame of carry (interest rate differential) for each pair
        """
        print("\nCalculating interest rate differentials...")
        
        carry = pd.DataFrame(index=spot_df.index)
        
        for pair in self.currency_pairs:
            base_curr = pair[:3]  # First 3 letters
            quote_curr = pair[3:]  # Last 3 letters
            
            if base_curr in rate_df.columns and quote_curr in rate_df.columns:
                # Carry = base currency rate - quote currency rate
                pair_carry = rate_df[base_curr] - rate_df[quote_curr]
                
                # Align with spot data dates
                carry[pair] = pair_carry.reindex(spot_df.index, method='ffill')
                
        print(f"Carry data shape: {carry.shape}")
        
        return carry
    
    def fetch_factor_data(self) -> Dict[str, pd.Series]:
        """
        Download FX risk factors for neutralization
        
        Returns:
            Dictionary of factor time series
        """
        print("\nFetching FX risk factors...")
        
        factors = {}
        
        # Dollar Index (DXY)
        try:
            dxy = yf.download('DX-Y.NYB', start=self.start_date, end=self.end_date, 
                             progress=False)['Adj Close']
            factors['dollar_index'] = dxy
            print(f"  ✓ Dollar Index: {len(dxy)} observations")
        except Exception as e:
            print(f"  ✗ Dollar Index: {e}")
        
        # Safe-haven proxy: VIX
        try:
            vix = yf.download('^VIX', start=self.start_date, end=self.end_date, 
                             progress=False)['Adj Close']
            factors['safe_haven'] = vix
            print(f"  ✓ VIX (safe-haven proxy): {len(vix)} observations")
        except Exception as e:
            print(f"  ✗ VIX: {e}")
        
        # Commodity FX proxy: Commodity Index
        try:
            gsci = yf.download('GCC', start=self.start_date, end=self.end_date, 
                              progress=False)['Adj Close']
            factors['commodity_fx'] = gsci
            print(f"  ✓ Commodity Index: {len(gsci)} observations")
        except Exception as e:
            print(f"  ✗ Commodity Index: {e}")
        
        return factors
    
    def save_data(self, spot_df: pd.DataFrame, rate_df: pd.DataFrame, 
                  carry_df: pd.DataFrame, factors: Dict):
        """Save data to CSV files"""
        import os
        os.makedirs('data', exist_ok=True)
        
        spot_df.to_csv('data/spot_rates.csv')
        rate_df.to_csv('data/interest_rates.csv')
        carry_df.to_csv('data/carry.csv')
        
        for name, series in factors.items():
            series.to_csv(f'data/factor_{name}.csv')
        
        print("\n✓ Data saved to data/ directory")
    
    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict]:
        """Load previously saved data"""
        spot_df = pd.read_csv('data/spot_rates.csv', index_col=0, parse_dates=True)
        rate_df = pd.read_csv('data/interest_rates.csv', index_col=0, parse_dates=True)
        carry_df = pd.read_csv('data/carry.csv', index_col=0, parse_dates=True)
        
        factors = {}
        import glob
        for filepath in glob.glob('data/factor_*.csv'):
            name = filepath.split('factor_')[1].replace('.csv', '')
            factors[name] = pd.read_csv(filepath, index_col=0, parse_dates=True, 
                                       squeeze=True)
        
        return spot_df, rate_df, carry_df, factors


if __name__ == "__main__":
    # Example usage
    fx_data = FXDataAcquisition()
    
    # Fetch all data
    spots = fx_data.fetch_spot_rates()
    rates = fx_data.fetch_interest_rates()
    carry = fx_data.calculate_carry(spots, rates)
    factors = fx_data.fetch_factor_data()
    
    # Save to disk
    fx_data.save_data(spots, rates, carry, factors)
    
    print("\n" + "="*60)
    print("Data acquisition complete!")
    print("="*60)
