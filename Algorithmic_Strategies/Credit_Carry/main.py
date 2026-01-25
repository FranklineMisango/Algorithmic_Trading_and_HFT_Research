"""
Credit Carry Strategy - Data Acquisition
Fetches CDS spreads and credit fundamentals
"""

import pandas as pd
import yfinance as yf
import yaml


class CreditDataAcquisition:
    """Fetch CDS spread data"""
    
    def __init__(self, config_path: str = 'config.yaml'):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.start_date = self.config['data']['start_date']
        self.end_date = self.config['data']['end_date']
    
    def fetch_cds_spreads(self) -> pd.DataFrame:
        """Download CDS spread proxies via ETFs"""
        print("Fetching credit spread data...")
        
        # ETF proxies for credit spreads
        credit_etfs = {
            'IG': 'LQD',  # Investment Grade
            'HY': 'HYG',  # High Yield
            'EM': 'EMB',  # Emerging Markets
        }
        
        spreads = {}
        for name, ticker in credit_etfs.items():
            try:
                data = yf.download(ticker, start=self.start_date, end=self.end_date,
                                 progress=False)
                if not data.empty:
                    spreads[name] = data['Adj Close']
                    print(f"  ✓ {name}: {len(data)} observations")
            except Exception as e:
                print(f"  ✗ {name}: {e}")
        
        df = pd.DataFrame(spreads)
        print(f"\nCredit data shape: {df.shape}")
        return df


class CreditCarryStrategy:
    """Main strategy class"""
    
    def __init__(self, config_path: str = 'config.yaml'):
        self.data_acq = CreditDataAcquisition(config_path)
    
    def run_backtest(self):
        """Run backtest"""
        print("\n" + "="*80)
        print(" "*25 + "CREDIT CARRY STRATEGY")
        print("="*80)
        
        spreads = self.data_acq.fetch_cds_spreads()
        print("\n✓ Pipeline complete")
        return {'spreads': spreads}


if __name__ == "__main__":
    strategy = CreditCarryStrategy()
    strategy.run_backtest()
