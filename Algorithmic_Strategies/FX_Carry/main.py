"""
FX Carry Strategy - Main Orchestration Script

Runs complete FX carry strategy pipeline:
1. Data acquisition
2. Signal generation
3. Factor neutralization
4. Portfolio construction
5. Backtesting
"""

import yaml
from data_acquisition import FXDataAcquisition
from signal_generator import CarrySignalGenerator
from factor_models import FXFactorModel
from portfolio_constructor import PortfolioConstructor
from backtester import FXCarryBacktester


class FXCarryStrategy:
    """Main class for FX carry strategy"""
    
    def __init__(self, config_path: str = 'config.yaml'):
        self.config_path = config_path
        
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Initialize components
        self.data_acq = FXDataAcquisition(config_path)
        self.signal_gen = CarrySignalGenerator(config_path)
        self.factor_model = FXFactorModel(config_path)
        self.portfolio = PortfolioConstructor(config_path)
        self.backtester = FXCarryBacktester(config_path)
        
        # Storage
        self.data = {}
        self.results = {}
    
    def fetch_data(self, use_cached: bool = False):
        """Fetch or load data"""
        print("\n" + "="*80)
        print(" "*25 + "FX CARRY STRATEGY")
        print("="*80)
        print(f"\nStrategy: {self.config['strategy']['name']}")
        print(f"Description: {self.config['strategy']['description']}")
        print(f"Target Volatility: {self.config['strategy']['target_volatility']:.0%}")
        print(f"Rebalance Frequency: {self.config['strategy']['rebalance_frequency']}")
        
        if use_cached:
            print("\n" + "="*60)
            print("LOADING CACHED DATA")
            print("="*60)
            try:
                spots, rates, carry, factors = self.data_acq.load_data()
                print("✓ Data loaded from cache")
            except Exception as e:
                print(f"✗ Could not load cache: {e}")
                print("Fetching new data...")
                use_cached = False
        
        if not use_cached:
            print("\n" + "="*60)
            print("DATA ACQUISITION")
            print("="*60)
            spots = self.data_acq.fetch_spot_rates()
            rates = self.data_acq.fetch_interest_rates()
            carry = self.data_acq.calculate_carry(spots, rates)
            factors = self.data_acq.fetch_factor_data()
            self.data_acq.save_data(spots, rates, carry, factors)
        
        self.data = {
            'spots': spots,
            'rates': rates,
            'carry': carry,
            'factors': factors
        }
    
    def generate_signals(self):
        """Generate carry signals"""
        zscores, signals, returns = self.signal_gen.run_signal_generation(
            self.data['carry'],
            self.data['spots']
        )
        
        self.results['zscores'] = zscores
        self.results['signals'] = signals
        self.results['raw_returns'] = returns
    
    def neutralize_factors(self):
        """Apply factor neutralization"""
        factor_returns = self.factor_model.calculate_factor_returns(self.data['factors'])
        
        neutral_returns = self.factor_model.neutralize_returns(
            self.results['raw_returns'],
            factor_returns
        )
        
        self.results['factor_returns'] = factor_returns
        self.results['neutral_returns'] = neutral_returns
    
    def construct_portfolio(self):
        """Build portfolio with risk management"""
        weights, portfolio_returns = self.portfolio.construct_portfolio(
            self.results['neutral_returns'],
            self.results['signals']
        )
        
        self.results['weights'] = weights
        self.results['portfolio_returns'] = portfolio_returns
    
    def backtest(self):
        """Run backtest with transaction costs"""
        backtest_results = self.backtester.run_backtest(
            self.results['portfolio_returns'],
            self.results['weights']
        )
        
        self.results['backtest'] = backtest_results
    
    def run_backtest(self, use_cached_data: bool = False):
        """Run complete strategy pipeline"""
        # Step 1: Fetch data
        self.fetch_data(use_cached=use_cached_data)
        
        # Step 2: Generate signals
        self.generate_signals()
        
        # Step 3: Neutralize factors
        self.neutralize_factors()
        
        # Step 4: Construct portfolio
        self.construct_portfolio()
        
        # Step 5: Backtest
        self.backtest()
        
        print("\n" + "="*80)
        print(" "*25 + "PIPELINE COMPLETE")
        print("="*80)
        
        return self.results
    
    def generate_report(self, output_path: str = 'strategy_report.txt'):
        """Generate text report of results"""
        if 'backtest' not in self.results:
            print("No backtest results available. Run run_backtest() first.")
            return
        
        with open(output_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write(" "*25 + "FX CARRY STRATEGY REPORT\n")
            f.write("="*80 + "\n\n")
            
            # Configuration
            f.write("STRATEGY CONFIGURATION\n")
            f.write("-" * 80 + "\n")
            f.write(f"Name: {self.config['strategy']['name']}\n")
            f.write(f"Description: {self.config['strategy']['description']}\n")
            f.write(f"Target Volatility: {self.config['strategy']['target_volatility']:.0%}\n")
            f.write(f"Rebalance Frequency: {self.config['strategy']['rebalance_frequency']}\n\n")
            
            # Performance metrics
            f.write("PERFORMANCE METRICS\n")
            f.write("-" * 80 + "\n")
            metrics = self.results['backtest']['metrics']
            for key, value in metrics.items():
                if isinstance(value, float):
                    if abs(value) < 0.1:
                        f.write(f"{key:.<40} {value:>10.4%}\n")
                    else:
                        f.write(f"{key:.<40} {value:>10.2f}\n")
            
            f.write("\n" + "="*80 + "\n")
        
        print(f"\n✓ Report saved to {output_path}")


if __name__ == "__main__":
    # Run FX carry strategy
    strategy = FXCarryStrategy(config_path='config.yaml')
    
    # Run backtest (set use_cached_data=True to skip data download)
    results = strategy.run_backtest(use_cached_data=False)
    
    # Generate report
    strategy.generate_report()
