"""Rates Carry - Main Script"""

import yaml
from data_acquisition import RatesDataAcquisition


class RatesCarryStrategy:
    """Main class for Rates carry strategy"""
    
    def __init__(self, config_path: str = 'config.yaml'):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.data_acq = RatesDataAcquisition(config_path)
    
    def run_backtest(self):
        """Run backtest pipeline"""
        print("\n" + "="*80)
        print(" "*25 + "RATES CARRY STRATEGY")
        print("="*80)
        
        # Fetch data
        yields = self.data_acq.fetch_yield_curves()
        rolldown = self.data_acq.calculate_rolldown(yields)
        
        # Save
        self.data_acq.save_data(yields, rolldown)
        
        print("\n✓ Pipeline complete")
        return {'yields': yields, 'rolldown': rolldown}


if __name__ == "__main__":
    strategy = RatesCarryStrategy()
    strategy.run_backtest()
