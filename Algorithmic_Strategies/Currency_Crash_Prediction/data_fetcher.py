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
import databento as db
from dotenv import load_dotenv
import os

# yf.pdr_override()  # Commented out - not needed for current implementation
load_dotenv()


class CurrencyDataFetcher:
    def __init__(self, config_path=None, use_databento=False):
        if config_path is None:
            # Default to config.yaml in the same directory as this script
            config_path = os.path.join(os.path.dirname(__file__), 'config.yaml')
        
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.currencies = (self.config['currencies']['advanced'] + 
                          self.config['currencies']['emerging'])
        self.anchor = self.config['anchor_currency']
        self.use_databento = use_databento
        
        # Databento client setup
        if self.use_databento:
            self.databento_api_key = os.getenv('DATABENTO_API_KEY')
            if self.databento_api_key:
                self.client = db.Historical(self.databento_api_key)
            # If no API key, we'll handle this in the fetch method
    
    def fetch_fx_rates(self, start_date, end_date):
        """Fetch FX rates vs anchor currency"""
        if self.use_databento:
            try:
                return self.fetch_fx_rates_databento(start_date, end_date)
            except Exception as e:
                logger.warning(f"Databento failed, falling back to Yahoo Finance: {e}")
                return self.fetch_fx_rates_yfinance(start_date, end_date)
        else:
            return self.fetch_fx_rates_yfinance(start_date, end_date)
    
    def fetch_fx_rates_databento(self, start_date, end_date):
        """Fetch FX rates from Databento"""
        if not hasattr(self, 'client') or not self.client:
            logger.warning("Databento client not initialized - no API key provided")
            raise ValueError("Databento API key not configured")
        
        logger.warning("Databento datasets available do not include FX data. Falling back to Yahoo Finance.")
        raise ValueError("FX data not available in current Databento datasets - use Yahoo Finance instead")
    
    def fetch_fx_rates_yfinance(self, start_date, end_date):
        """Fetch FX rates from Yahoo Finance (fallback)"""
        fx_data = {}
        failed_currencies = []
        
        for currency in self.currencies:
            pair = f"{currency}{self.anchor}=X"
            logger.info(f"Fetching {pair} from Yahoo Finance")
            
            try:
                data = yf.download(pair, start=start_date, end=end_date, 
                                  progress=False)
                if data.empty:
                    raise ValueError(f"No data returned for {pair}")
                
                # Handle new yfinance MultiIndex structure
                if isinstance(data.columns, pd.MultiIndex):
                    # Extract Close price from MultiIndex
                    close_data = data[('Close', pair)]
                else:
                    # Fallback for older yfinance versions
                    close_data = data['Close'] if 'Close' in data.columns else data['Adj Close']
                
                # Check if we have any valid data
                if close_data.dropna().empty:
                    raise ValueError(f"No valid price data for {pair}")
                
                fx_data[currency] = close_data
                logger.info(f"Successfully fetched {len(close_data.dropna())} data points for {pair}")
            except Exception as e:
                logger.warning(f"Failed to fetch {pair}: {e}")
                failed_currencies.append(currency)
        
        if not fx_data:
            raise ValueError("No FX data could be fetched from Yahoo Finance - check your internet connection and try again later")
        
        if failed_currencies:
            logger.warning(f"Skipped currencies with no data: {', '.join(failed_currencies)}")
            
        df = pd.DataFrame(fx_data)
        df.index = pd.to_datetime(df.index)
        return df.resample('ME').last()  # Month-end prices
    
    
    def fetch_interest_rates(self, start_date, end_date):
        """Fetch policy interest rates from FRED with 2026 active series"""
        # Updated mapping with active series IDs that track policy rates in 2026
        rate_mapping = {
            'EUR': 'ECBDFR',           # ECB Deposit Facility (Internal FRED)
            'JPY': 'IRSTCI01JPM156N',  # Japan Immediate Rate (Internal FRED - fixes 0.3% issue)
            'GBP': 'IUDSOIA',          # UK SONIA (Internal FRED - fixes 404 issue)
            'CAD': 'IRSTCB01CAM156N',  # Canada Immediate Rate (Internal FRED)
            'AUD': 'IRSTCI01AUM156N',  # Australia Immediate Rate (Internal FRED - fixes RBA issue)
            'CHF': 'IRSTCI01CHM156N',  # Swiss Immediate Rate (Internal FRED - fixes SNB issue)
            'BRL': 'IRSTCI01BRM156N',  # Brazil Selic (Internal FRED)
            'CNY': 'IRSTCI01CNM156N',  # China 1Y Loan Prime (Internal FRED)
            'INR': 'IRSTCI01INM156N',  # India Repo Rate (Internal FRED)
            'MXN': 'IRSTCI01MXM156N',  # Mexico Reference Rate (Internal FRED)
            'ZAR': 'IRSTCI01ZAM156N',  # South Africa Repo Rate (Internal FRED)
            'RUB': 'IRSTCB01RUM156N',  # Russia Key Rate
            'SEK': 'IRSTCI01SEM156N',  # Sweden Immediate Rate
            'NOK': 'IRSTCI01NOM156N',  # Norway Immediate Rate
            'NZD': 'IRSTCI01NZM156N',  # New Zealand Immediate Rate
            'KRW': 'IRSTCI01KRM156N',  # South Korea Base Rate
            'TRY': 'INTDSRTRM193N'     # Turkey Discount Rate
        }



        
        
        rates_data = {}
        for currency in self.currencies:
            fred_code = rate_mapping.get(currency)
            if not fred_code:
                continue
                
            logger.info(f"Fetching {currency} rate: {fred_code}")
            try:
                # Using pandas_datareader to pull from FRED
                data = pdr.get_data_fred(fred_code, start_date, end_date)
                if not data.empty:
                    rates_data[currency] = data.iloc[:, 0]
            except Exception as e:
                logger.warning(f"Failed to fetch {fred_code} for {currency}: {e}")
        
        if not rates_data:
            raise ValueError("No interest rate data could be fetched.")
            
        df = pd.DataFrame(rates_data)
        # Use 'ffill' to handle days where FRED doesn't publish (weekends/holidays)
        return df.resample('ME').last().ffill()

    
    def fetch_all_data(self, start_date=None, end_date=None):
        """Fetch both FX and interest rate data"""
        start_date = start_date or self.config['start_date']
        end_date = end_date or self.config['end_date']
        
        logger.info(f"Fetching data from {start_date} to {end_date}")
        
        try:
            fx_rates = self.fetch_fx_rates(start_date, end_date)
        except Exception as e:
            logger.error(f"Failed to fetch FX rates: {e}")
            raise
        
        try:
            interest_rates = self.fetch_interest_rates(start_date, end_date)
        except Exception as e:
            logger.error(f"Failed to fetch interest rates: {e}")
            raise
        
        # Align dates
        common_dates = fx_rates.index.intersection(interest_rates.index)
        if len(common_dates) == 0:
            raise ValueError("No overlapping dates between FX and interest rate data")
            
        fx_rates = fx_rates.loc[common_dates]
        interest_rates = interest_rates.loc[common_dates]
        
        # Save to CSV
        data_dir = os.path.join(os.path.dirname(__file__), 'data')
        os.makedirs(data_dir, exist_ok=True)
        fx_rates.to_csv(os.path.join(data_dir, 'fx_rates.csv'))
        interest_rates.to_csv(os.path.join(data_dir, 'interest_rates.csv'))
        
        logger.info(f"Saved {len(fx_rates)} months of data")
        
        return fx_rates, interest_rates


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--start', default='2007-01-01')
    parser.add_argument('--end', default='2025-12-31')
    parser.add_argument('--databento', action='store_true', 
                       help='Use Databento for FX data instead of Yahoo Finance')
    args = parser.parse_args()
    
    fetcher = CurrencyDataFetcher(use_databento=args.databento)
    fx, rates = fetcher.fetch_all_data(args.start, args.end)
    
    print(f"\nFX Rates shape: {fx.shape}")
    print(f"Interest Rates shape: {rates.shape}")
    print(f"Data source: {'Databento' if args.databento else 'Yahoo Finance'}")
    print(f"\nSample FX data:\n{fx.head()}")
    print(f"\nSample Interest Rates:\n{rates.head()}")
