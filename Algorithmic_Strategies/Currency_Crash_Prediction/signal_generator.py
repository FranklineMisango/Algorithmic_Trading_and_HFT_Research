"""
Currency Crash Prediction Signal Generator
"""
import pandas as pd
import numpy as np
import yaml
from loguru import logger


class CurrencyCrashPredictor:
    def __init__(self, config_path='config.yaml'):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.lookback = self.config['lookback_months']
        self.rate_threshold = self.config['rate_threshold_percentile']
        self.fx_threshold = self.config['fx_threshold_percentile']
        self.crash_threshold = self.config['crash_threshold_percentile']
        
        self.fx_rates = None
        self.interest_rates = None
        self.signals = None
    
    def load_data(self, fx_path='data/fx_rates.csv', 
                  rates_path='data/interest_rates.csv'):
        """Load FX and interest rate data"""
        self.fx_rates = pd.read_csv(fx_path, index_col=0, parse_dates=True)
        self.interest_rates = pd.read_csv(rates_path, index_col=0, parse_dates=True)
        logger.info(f"Loaded {len(self.fx_rates)} months of data")
    
    def calculate_features(self):
        """Calculate Δi and ΔFX over lookback period"""
        # Interest rate change (Δi)
        delta_i = self.interest_rates.diff(self.lookback)
        
        # Currency depreciation (ΔFX = log change)
        log_fx = np.log(self.fx_rates)
        delta_fx = log_fx.diff(self.lookback)
        
        # Monthly returns for crash detection
        monthly_returns = self.fx_rates.pct_change()
        
        return delta_i, delta_fx, monthly_returns
    
    def calculate_thresholds(self, delta_i, delta_fx, monthly_returns, 
                            in_sample_end=None):
        """Calculate historical distribution thresholds"""
        in_sample_end = in_sample_end or self.config['in_sample_end']
        
        # Use only in-sample data for threshold calibration
        delta_i_train = delta_i.loc[:in_sample_end]
        delta_fx_train = delta_fx.loc[:in_sample_end]
        returns_train = monthly_returns.loc[:in_sample_end]
        
        thresholds = {}
        
        for currency in delta_i.columns:
            # Top 20% for rate increases (aggressive tightening)
            rate_threshold = delta_i_train[currency].quantile(
                self.rate_threshold / 100
            )
            
            # Bottom 33% for FX (currency weakness)
            fx_threshold = delta_fx_train[currency].quantile(
                self.fx_threshold / 100
            )
            
            # Bottom 4% for crash definition
            crash_threshold = returns_train[currency].quantile(
                self.crash_threshold / 100
            )
            
            thresholds[currency] = {
                'rate': rate_threshold,
                'fx': fx_threshold,
                'crash': crash_threshold
            }
        
        return thresholds
    
    def generate_r_zone_signals(self, delta_i, delta_fx, thresholds):
        """Generate R-Zone binary signals"""
        r_zone = pd.DataFrame(0, index=delta_i.index, columns=delta_i.columns)
        
        for currency in delta_i.columns:
            if currency not in thresholds:
                continue
            
            # Condition A: Δi in top 20%
            condition_a = delta_i[currency] >= thresholds[currency]['rate']
            
            # Condition B: ΔFX in bottom 33%
            condition_b = delta_fx[currency] <= thresholds[currency]['fx']
            
            # R-Zone = both conditions met
            r_zone[currency] = (condition_a & condition_b).astype(int)
        
        return r_zone
    
    def identify_crashes(self, monthly_returns, thresholds):
        """Identify actual crash events"""
        crashes = pd.DataFrame(0, index=monthly_returns.index, 
                              columns=monthly_returns.columns)
        
        for currency in monthly_returns.columns:
            if currency not in thresholds:
                continue
            
            crashes[currency] = (
                monthly_returns[currency] <= thresholds[currency]['crash']
            ).astype(int)
        
        return crashes
    
    def generate_signals(self):
        """Main signal generation pipeline"""
        logger.info("Calculating features...")
        delta_i, delta_fx, monthly_returns = self.calculate_features()
        
        logger.info("Calculating thresholds...")
        thresholds = self.calculate_thresholds(delta_i, delta_fx, monthly_returns)
        
        logger.info("Generating R-Zone signals...")
        r_zone = self.generate_r_zone_signals(delta_i, delta_fx, thresholds)
        
        logger.info("Identifying crashes...")
        crashes = self.identify_crashes(monthly_returns, thresholds)
        
        # Store results
        self.signals = {
            'delta_i': delta_i,
            'delta_fx': delta_fx,
            'monthly_returns': monthly_returns,
            'r_zone': r_zone,
            'crashes': crashes,
            'thresholds': thresholds
        }
        
        # Calculate statistics
        self.calculate_statistics()
        
        return self.signals
    
    def calculate_statistics(self):
        """Calculate crash probabilities and performance metrics"""
        r_zone = self.signals['r_zone']
        crashes = self.signals['crashes']
        
        # Look forward 6 months for crash after R-Zone entry
        stats = []
        
        for currency in r_zone.columns:
            r_zone_dates = r_zone[r_zone[currency] == 1].index
            
            if len(r_zone_dates) == 0:
                continue
            
            crash_count = 0
            total_signals = len(r_zone_dates)
            
            for date in r_zone_dates:
                # Check next 6 months for crash
                future_dates = pd.date_range(
                    date, periods=7, freq='M'
                )[1:]  # Exclude current month
                
                future_crashes = crashes.loc[
                    crashes.index.isin(future_dates), currency
                ]
                
                if future_crashes.sum() > 0:
                    crash_count += 1
            
            crash_prob = crash_count / total_signals if total_signals > 0 else 0
            
            # Baseline crash probability
            baseline_prob = crashes[currency].mean()
            
            stats.append({
                'currency': currency,
                'r_zone_signals': total_signals,
                'crashes_after_signal': crash_count,
                'crash_prob_in_rzone': crash_prob,
                'baseline_crash_prob': baseline_prob,
                'probability_ratio': crash_prob / baseline_prob if baseline_prob > 0 else np.nan
            })
        
        self.statistics = pd.DataFrame(stats)
        logger.info(f"\nStatistics:\n{self.statistics}")
        
        return self.statistics


if __name__ == '__main__':
    predictor = CurrencyCrashPredictor()
    predictor.load_data()
    signals = predictor.generate_signals()
    
    print("\nR-Zone Signals (last 10 months):")
    print(signals['r_zone'].tail(10))
    
    print("\nCrash Events (last 10 months):")
    print(signals['crashes'].tail(10))
    
    print("\nStatistics:")
    print(predictor.statistics)
