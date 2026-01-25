"""
Main Execution Script for Holiday Effect Strategy

Orchestrates complete pipeline: data, signals, backtesting, options.
"""

import argparse
import yaml
from pathlib import Path
import pandas as pd
import numpy as np

from data_acquisition import DataAcquisition
from signal_generator import SignalGenerator
from backtester import Backtester
from options_strategy import OptionsStrategy


class HolidayEffectPipeline:
    """Complete pipeline for Holiday Effect trading strategy."""
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize pipeline with configuration."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.data_acq = DataAcquisition(config_path)
        self.signal_gen = SignalGenerator(config_path)
        self.backtester = Backtester(config_path)
        self.options_strat = OptionsStrategy(config_path)
        
    def run_data_acquisition(self):
        """Step 1: Fetch and prepare data."""
        print("\n" + "="*60)
        print("STEP 1: DATA ACQUISITION")
        print("="*60)
        
        dataset = self.data_acq.fetch_full_dataset()
        
        print(f"\nData loaded:")
        print(f"  Date range: {dataset['start_date']} to {dataset['end_date']}")
        print(f"  Trading days: {dataset['trading_days']}")
        print(f"  AMZN price range: ${dataset['amzn_prices']['Adj Close'].min():.2f} - ${dataset['amzn_prices']['Adj Close'].max():.2f}")
        
        return dataset
    
    def run_signal_generation(self, dataset):
        """Step 2: Generate calendar-based signals."""
        print("\n" + "="*60)
        print("STEP 2: SIGNAL GENERATION")
        print("="*60)
        
        # Generate base signals
        signals, windows = self.signal_gen.generate_signal_series(
            dataset['amzn_prices'].index
        )
        
        print(f"\nEvent windows identified:")
        print(f"  Black Friday events: {len(windows[windows['event_type'] == 'black_friday'])}")
        print(f"  Prime Day events: {len(windows[windows['event_type'] == 'prime_day'])}")
        print(f"  Total event days: {signals['in_window'].sum()}")
        
        # Apply market filters
        filtered_signals = self.signal_gen.apply_market_filters(
            signals,
            dataset['spy_prices']['Adj Close'],
            dataset['vix']
        )
        
        filtered_days = filtered_signals['in_window'].sum()
        filter_rate = (signals['in_window'].sum() - filtered_days) / signals['in_window'].sum() * 100
        
        print(f"\nAfter market filters:")
        print(f"  Event days remaining: {filtered_days}")
        print(f"  Filtered out: {filter_rate:.1f}%")
        
        return filtered_signals, windows
    
    def run_equity_backtest(self, dataset, signals):
        """Step 3: Backtest equity long strategy."""
        print("\n" + "="*60)
        print("STEP 3: EQUITY STRATEGY BACKTEST")
        print("="*60)
        
        results = self.backtester.run_backtest(
            dataset['amzn_prices'],
            signals
        )
        
        print("\n=== Performance Metrics ===")
        print(f"Initial Capital:       ${results['initial_capital']:,.2f}")
        print(f"Final Value:           ${results['final_value']:,.2f}")
        print(f"Total Return:          {results['metrics']['total_return_pct']:.2f}%")
        print(f"Annualized Return:     {results['metrics']['annualized_return_pct']:.2f}%")
        print(f"Sharpe Ratio:          {results['metrics']['sharpe_ratio']:.2f}")
        print(f"Max Drawdown:          {results['metrics']['max_drawdown_pct']:.2f}%")
        print(f"Number of Trades:      {results['metrics']['num_trades']}")
        print(f"Win Rate:              {results['metrics']['win_rate']*100:.1f}%")
        
        # Compare to benchmark
        comparison = self.backtester.compare_to_benchmark(
            results,
            dataset['spy_prices']
        )
        
        print("\n=== vs Benchmark (SPY) ===")
        print(f"Strategy Return:       {comparison['strategy_total_return']:.2f}%")
        print(f"Benchmark Return:      {comparison['benchmark_total_return']:.2f}%")
        print(f"Excess Return:         {comparison['excess_return']:.2f}%")
        print(f"Strategy Sharpe:       {comparison['strategy_sharpe']:.2f}")
        print(f"Benchmark Sharpe:      {comparison['benchmark_sharpe']:.2f}")
        print(f"Sharpe Improvement:    {comparison['sharpe_improvement']:.2f}")
        
        # Save results
        results['portfolio'].to_csv('equity_strategy_portfolio.csv')
        results['trades'].to_csv('equity_strategy_trades.csv')
        
        return results
    
    def run_options_backtest(self, dataset, windows):
        """Step 4: Backtest options strategy."""
        print("\n" + "="*60)
        print("STEP 4: OPTIONS STRATEGY BACKTEST")
        print("="*60)
        
        # Use only 2012+ per research
        recent_windows = windows[windows['year'] >= 2012]
        
        print(f"\nSimulating put selling (2012+)")
        print(f"  Event windows: {len(recent_windows)}")
        
        trades = self.options_strat.simulate_options_trades(
            recent_windows,
            dataset['amzn_prices'],
            initial_capital=self.config['portfolio']['initial_capital']
        )
        
        metrics = self.options_strat.calculate_metrics(trades)
        
        print("\n=== Options Performance ===")
        print(f"Total Trades:          {metrics['total_trades']}")
        print(f"Winning Trades:        {metrics['winning_trades']}")
        print(f"Win Rate:              {metrics['win_rate']*100:.1f}%")
        print(f"Total Premium:         ${metrics['total_premium_collected']:,.2f}")
        print(f"Total PnL:             ${metrics['total_pnl']:,.2f}")
        print(f"Avg Premium/Trade:     ${metrics['avg_premium_per_trade']:,.2f}")
        print(f"Final Value:           ${metrics['final_portfolio_value']:,.2f}")
        
        # Save results
        trades.to_csv('options_strategy_trades.csv', index=False)
        
        return trades, metrics
    
    def run_full_pipeline(self):
        """Execute complete pipeline."""
        print("\n" + "="*70)
        print("HOLIDAY EFFECT TRADING STRATEGY - FULL PIPELINE")
        print("="*70)
        
        # Step 1: Data
        dataset = self.run_data_acquisition()
        
        # Step 2: Signals
        signals, windows = self.run_signal_generation(dataset)
        
        # Step 3: Equity backtest
        equity_results = self.run_equity_backtest(dataset, signals)
        
        # Step 4: Options backtest
        options_trades, options_metrics = self.run_options_backtest(dataset, windows)
        
        print("\n" + "="*70)
        print("PIPELINE COMPLETE")
        print("="*70)
        
        print("\nKey Findings:")
        print(f"  Equity Sharpe: {equity_results['metrics']['sharpe_ratio']:.2f} (Target: >0.5)")
        print(f"  Options Win Rate: {options_metrics['win_rate']*100:.1f}% (Research: 100%)")
        print(f"  Total Event Windows: {len(windows)}")
        
        return {
            'dataset': dataset,
            'signals': signals,
            'windows': windows,
            'equity_results': equity_results,
            'options_trades': options_trades,
            'options_metrics': options_metrics
        }


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Holiday Effect Trading Strategy')
    parser.add_argument('--mode', type=str, default='full',
                       choices=['full', 'equity', 'options', 'data'],
                       help='Execution mode')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to config file')
    
    args = parser.parse_args()
    
    pipeline = HolidayEffectPipeline(args.config)
    
    if args.mode == 'full':
        pipeline.run_full_pipeline()
        
    elif args.mode == 'data':
        dataset = pipeline.run_data_acquisition()
        
    elif args.mode == 'equity':
        dataset = pipeline.run_data_acquisition()
        signals, windows = pipeline.run_signal_generation(dataset)
        pipeline.run_equity_backtest(dataset, signals)
        
    elif args.mode == 'options':
        dataset = pipeline.run_data_acquisition()
        _, windows = pipeline.run_signal_generation(dataset)
        pipeline.run_options_backtest(dataset, windows)


if __name__ == "__main__":
    main()
