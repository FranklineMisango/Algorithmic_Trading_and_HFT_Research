import numpy as np
import pandas as pd
from loguru import logger

class RiskManager:
    def __init__(self, config):
        self.config = config
        self.stop_loss_bps = config['parameters']['stop_loss_bps'] / 10000
        self.max_spread_mult = config['parameters']['max_spread_multiplier']
        self.min_volume = config['parameters']['min_volume_usd']
        
    def check_liquidity(self, perp_volume, spot_volume, perp_price, spot_price):
        """Check if liquidity conditions are met"""
        perp_volume_usd = perp_volume * perp_price
        spot_volume_usd = spot_volume * spot_price
        
        return (perp_volume_usd > self.min_volume and 
                spot_volume_usd > self.min_volume)
    
    def check_spread(self, current_spread, normal_spread):
        """Check if spread is within acceptable range"""
        return current_spread < normal_spread * self.max_spread_mult
    
    def stress_test_funding_shock(self, df, signals, backtester):
        """Simulate funding rate clamping disabled (δ = 0)"""
        logger.info("Running stress test: Funding Rate Shock")
        
        config_stress = self.config.copy()
        config_stress['parameters']['delta'] = 0.0
        
        from signal_generator import SignalGenerator
        signal_gen_stress = SignalGenerator(config_stress)
        signals_stress = signal_gen_stress.generate_signals(df)
        
        results, trades = backtester.run_backtest(df, signals_stress)
        metrics = backtester.calculate_metrics(results, trades)
        
        return metrics
    
    def stress_test_liquidity_crisis(self, df, signals, backtester):
        """Simulate 10x wider bid-ask spreads"""
        logger.info("Running stress test: Liquidity Crisis")
        
        config_stress = self.config.copy()
        config_stress['parameters']['spot_fee'] *= 10
        config_stress['backtest']['slippage_factor'] *= 10
        
        from backtester import Backtester
        backtester_stress = Backtester(config_stress)
        
        results, trades = backtester_stress.run_backtest(df, signals)
        metrics = backtester_stress.calculate_metrics(results, trades)
        
        return metrics
    
    def stress_test_rate_spike(self, df, signals, backtester):
        """Simulate crypto borrowing rate spike by 500 bps"""
        logger.info("Running stress test: Borrowing Rate Spike")
        
        df_stress = df.copy()
        df_stress['borrow_rate'] += 0.05  # +500 bps
        
        from signal_generator import SignalGenerator
        signal_gen = SignalGenerator(self.config)
        signals_stress = signal_gen.generate_signals(df_stress)
        
        results, trades = backtester.run_backtest(df_stress, signals_stress)
        metrics = backtester.calculate_metrics(results, trades)
        
        return metrics
    
    def run_all_stress_tests(self, df, signals, backtester):
        """Run all stress tests and return results"""
        logger.info("Running comprehensive stress tests")
        
        baseline_results, baseline_trades = backtester.run_backtest(df, signals)
        baseline_metrics = backtester.calculate_metrics(baseline_results, baseline_trades)
        
        stress_results = {
            'Baseline': baseline_metrics,
            'Funding Shock': self.stress_test_funding_shock(df, signals, backtester),
            'Liquidity Crisis': self.stress_test_liquidity_crisis(df, signals, backtester),
            'Rate Spike': self.stress_test_rate_spike(df, signals, backtester)
        }
        
        return stress_results
    
    def calculate_var(self, returns, confidence=0.95):
        """Calculate Value at Risk"""
        return np.percentile(returns, (1 - confidence) * 100)
    
    def calculate_cvar(self, returns, confidence=0.95):
        """Calculate Conditional Value at Risk (Expected Shortfall)"""
        var = self.calculate_var(returns, confidence)
        return returns[returns <= var].mean()
