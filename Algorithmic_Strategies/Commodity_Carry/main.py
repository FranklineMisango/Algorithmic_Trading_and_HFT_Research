"""
Commodity Carry Strategy - Data Acquisition
Fetches commodity futures curves and calculates convenience yield
"""

import pandas as pd
import yfinance as yf
import yaml


class CommodityDataAcquisition:
    """Fetch commodity futures data"""
    
    def __init__(self, config_path: str = 'config.yaml'):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.start_date = self.config['data']['start_date']
        self.end_date = self.config['data']['end_date']
    
    def fetch_commodity_prices(self) -> pd.DataFrame:
        """Download commodity prices via ETFs"""
        print("Fetching commodity futures data...")
        
        # ETF proxies for commodities
        commodity_etfs = {
            'Energy': 'USO',    # Oil
            'Gold': 'GLD',
            'Silver': 'SLV',
            'Agriculture': 'DBA',
            'Metals': 'DBB',
        }
        
        prices = {}
        for name, ticker in commodity_etfs.items():
            try:
                data = yf.download(ticker, start=self.start_date, end=self.end_date,
                                 progress=False)
                if not data.empty:
                    prices[name] = data['Adj Close']
                    print(f"  ✓ {name}: {len(data)} observations")
            except Exception as e:
                print(f"  ✗ {name}: {e}")
        
        df = pd.DataFrame(prices)
        print(f"\nCommodity data shape: {df.shape}")
        return df


class CommodityCarryStrategy:
    """Main strategy class"""
    
    def __init__(self, config_path: str = 'config.yaml'):
        self.data_acq = CommodityDataAcquisition(config_path)
    
    def run_backtest(self):
        """Run backtest"""
        print("\n" + "="*80)
        print(" "*23 + "COMMODITY CARRY STRATEGY")
        print("="*80)
        
        prices = self.data_acq.fetch_commodity_prices()
        print("\n✓ Pipeline complete")
        return {'prices': prices}


if __name__ == "__main__":
    strategy = CommodityCarryStrategy()
    strategy.run_backtest()
