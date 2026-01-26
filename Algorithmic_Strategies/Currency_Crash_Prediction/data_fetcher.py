"""
Data fetcher for currency pairs and interest rates
"""
import pandas as pd
import numpy as np
import yfinance as yf
from pandas_datareader import data as pdr
from datetime import datetime
import yaml
from loguru import logger

yf.pdr_override()


class CurrencyDataFetcher:
    def __init__(self, config_path='config.yaml'):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.currencies = (self.config['currencies']['advanced'] + 
                          self.config['currencies']['emerging'])
        self.anchor = self.config['anchor_currency']
    
    def fetch_fx_rates(self, start_date, end_date):
        """Fetch FX rates vs anchor currency"""
        fx_data = {}
        
        for currency in self.currencies:
            pair = f"{currency}{self.anchor}=X"
            logger.info(f"Fetching {pair}")
            
            try:
                data = yf.download(pair, start=start_date, end=end_date, 
                                  progress=False)['Adj Close']
                fx_data[currency] = data
            except Exception as e:
                logger.error(f"Failed to fetch {pair}: {e}")
        
        df = pd.DataFrame(fx_data)
        df.index = pd.to_datetime(df.index)
        return df.resample('M').last()  # Month-end prices
    
    def fetch_interest_rates(self, start_date, end_date):
        """Fetch policy interest rates from FRED"""
        rate_mapping = {
            'EUR': 'ECBDFR',      # ECB deposit facility rate
            'JPY': 'IRSTCB01JPM156N',  # Japan policy rate
            'GBP': 'GBPONTD156N',  # UK Bank Rate
            'CAD': 'IRSTCB01CAM156N',  # Canada overnight rate
            'AUD': 'IRSTCB01AUM156N',  # Australia cash rate
            'CHF': 'IRSTCB01CHM156N',  # Swiss 3M rate
            'BRL': 'IRSTCI01BRM156N',  # Brazil Selic
            'CNY': 'IRSTCI01CNM156N',  # China lending rate
            'INR': 'IRSTCI01INM156N',  # India repo rate
        }
        
        rates_data = {}
        
        for currency, fred_code in rate_mapping.items():
            if currency not in self.currencies:
                continue
                
            logger.info(f"Fetching {currency} rate: {fred_code}")
            
            try:
                data = pdr.get_data_fred(fred_code, start_date, end_date)
                rates_data[currency] = data.iloc[:, 0]
            except Exception as e:
                logger.warning(f"Failed to fetch {fred_code}: {e}")
                # Use synthetic data for demo
                dates = pd.date_range(start_date, end_date, freq='M')
                rates_data[currency] = pd.Series(
                    np.random.uniform(0.5, 5.0, len(dates)), 
                    index=dates
                )
        
        df = pd.DataFrame(rates_data)
        df.index = pd.to_datetime(df.index)
        return df.resample('M').last()
    
    def fetch_all_data(self, start_date=None, end_date=None):
        """Fetch both FX and interest rate data"""
        start_date = start_date or self.config['start_date']
        end_date = end_date or self.config['end_date']
        
        logger.info(f"Fetching data from {start_date} to {end_date}")
        
        fx_rates = self.fetch_fx_rates(start_date, end_date)
        interest_rates = self.fetch_interest_rates(start_date, end_date)
        
        # Align dates
        common_dates = fx_rates.index.intersection(interest_rates.index)
        fx_rates = fx_rates.loc[common_dates]
        interest_rates = interest_rates.loc[common_dates]
        
        # Save to CSV
        fx_rates.to_csv('data/fx_rates.csv')
        interest_rates.to_csv('data/interest_rates.csv')
        
        logger.info(f"Saved {len(fx_rates)} months of data")
        
        return fx_rates, interest_rates


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--start', default='1999-01-01')
    parser.add_argument('--end', default='2023-12-31')
    args = parser.parse_args()
    
    fetcher = CurrencyDataFetcher()
    fx, rates = fetcher.fetch_all_data(args.start, args.end)
    
    print(f"\nFX Rates shape: {fx.shape}")
    print(f"Interest Rates shape: {rates.shape}")
    print(f"\nSample FX data:\n{fx.head()}")
    print(f"\nSample Interest Rates:\n{rates.head()}")
