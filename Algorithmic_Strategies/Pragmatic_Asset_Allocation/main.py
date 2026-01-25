#!/usr/bin/env python3
"""
Pragmatic Asset Allocation - Main Pipeline
Complete implementation of the Quant Radio strategy with GTAA signals.
"""

import sys
import os
import argparse
import logging
from pathlib import Path
from datetime import datetime
import yaml
import pandas as pd

# Add current directory to path for imports
sys.path.append('.')

from data_acquisition import PragmaticAssetAllocationData
from signal_generation import PragmaticAssetAllocationSignals
from portfolio_construction import PragmaticAssetAllocationPortfolio
from backtester import PragmaticAssetAllocationBacktester


class PragmaticAssetAllocationPipeline:
    """Main pipeline orchestrator for the Pragmatic Asset Allocation strategy."""

    def __init__(self, config_path='config.yaml'):
        """Initialize the pipeline with configuration."""
        self.config_path = config_path
        self.config = self.load_config()
        self.setup_logging()

        # Initialize components
        self.data_acq = PragmaticAssetAllocationData()
        self.signal_gen = PragmaticAssetAllocationSignals()
        self.portfolio = PragmaticAssetAllocationPortfolio()
        self.backtester = PragmaticAssetAllocationBacktester()

        self.logger.info("Pragmatic Asset Allocation Pipeline initialized")

    def load_config(self):
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            return config
        except Exception as e:
            print(f"Error loading config: {e}")
            return {}

    def setup_logging(self):
        """Setup logging configuration."""
        log_level = getattr(logging, self.config.get('logging', {}).get('level', 'INFO').upper())
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('logs/pipeline.log'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger('PragmaticAssetAllocation')

    def run_data_acquisition(self, start_date=None, end_date=None):
        """Run data acquisition pipeline."""
        self.logger.info("=== DATA ACQUISITION ===")

        # Use provided dates or config defaults
        start_date = start_date or self.config['backtest']['start_date']
        end_date = end_date or self.config['backtest']['end_date']

        self.logger.info(f"Acquiring data from {start_date} to {end_date}")

        # Fetch all required data
        all_data = self.data_acq.fetch_all_data(start_date, end_date)

        if all_data:
            self.logger.info("Data acquisition completed successfully")
            return all_data
        else:
            self.logger.error("Data acquisition failed")
            return None

    def run_signal_generation(self, all_data):
        """Run signal generation pipeline."""
        self.logger.info("=== SIGNAL GENERATION ===")

        if not all_data:
            self.logger.error("No data available for signal generation")
            return None

        # Generate all signals
        signals_dict = self.signal_gen.generate_all_signals(all_data)

        if signals_dict:
            self.logger.info("Signal generation completed successfully")
            return signals_dict
        else:
            self.logger.error("Signal generation failed")
            return None

    def run_portfolio_construction(self, signals_dict, all_data):
        """Run portfolio construction pipeline."""
        self.logger.info("=== PORTFOLIO CONSTRUCTION ===")

        if not signals_dict or not all_data:
            self.logger.error("Missing signals or data for portfolio construction")
            return None

        # Run portfolio construction
        portfolio_data = self.portfolio.run_portfolio_construction(signals_dict, all_data)

        if portfolio_data:
            self.logger.info("Portfolio construction completed successfully")
            return portfolio_data
        else:
            self.logger.error("Portfolio construction failed")
            return None

    def run_backtest(self, signals_dict, portfolio_data, all_data):
        """Run backtesting pipeline."""
        self.logger.info("=== BACKTESTING ===")

        if not portfolio_data or not all_data or not signals_dict:
            self.logger.error("Missing signals, portfolio data or market data for backtesting")
            return None

        # Combine all price data into single DataFrame for backtester
        combined_price_data = pd.concat(list(all_data.values()), axis=1, keys=all_data.keys())

        # Run backtest
        backtest_results = self.backtester.run_backtest(signals_dict, combined_price_data, portfolio_data)

        if backtest_results:
            self.logger.info("Backtesting completed successfully")
            return backtest_results
        else:
            self.logger.error("Backtesting failed")
            return None

    def run_full_pipeline(self, start_date=None, end_date=None):
        """Run the complete pipeline from data acquisition to backtesting."""
        self.logger.info("=== STARTING FULL PIPELINE ===")

        # Step 1: Data Acquisition
        all_data = self.run_data_acquisition(start_date, end_date)
        if not all_data:
            return None

        # Step 2: Signal Generation
        signals_dict = self.run_signal_generation(all_data)
        if not signals_dict:
            return None

        # Step 3: Portfolio Construction
        portfolio_data = self.run_portfolio_construction(signals_dict, all_data)
        if not portfolio_data:
            return None

        # Step 4: Backtesting
        backtest_results = self.run_backtest(signals_dict, portfolio_data, all_data)
        if not backtest_results:
            return None

        self.logger.info("=== FULL PIPELINE COMPLETED SUCCESSFULLY ===")
        return backtest_results

    def run_quick_analysis(self, backtest_results):
        """Run quick analysis of backtest results."""
        self.logger.info("=== QUICK ANALYSIS ===")

        if not backtest_results:
            self.logger.error("No backtest results available for analysis")
            return

        # Extract key metrics
        metrics = backtest_results.get('performance_metrics', {})
        benchmarks = backtest_results.get('benchmark_results', {})

        print("\nStrategy Performance:")
        print(f"  Annual Return: {metrics.get('annual_return', 0):.2%}")
        print(f"  Annual Volatility: {metrics.get('annual_volatility', 0):.2%}")
        print(f"  Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}")
        print(f"  Max Drawdown: {metrics.get('max_drawdown', 0):.2%}")

        # Benchmark comparison
        if benchmarks:
            print("\nBenchmark Comparison:")
            for bench_name, bench_data in benchmarks.items():
                bench_sharpe = bench_data.get('sharpe_ratio', 0)
                strategy_sharpe = metrics.get('sharpe_ratio', 0)
                outperformance = strategy_sharpe - bench_sharpe
                print(f"  {bench_name}: Sharpe {bench_sharpe:.2f} (Strategy outperforms by {outperformance:+.2f})")

        # Risk assessment
        targets = self.config['backtest']['targets']
        if metrics.get('sharpe_ratio', 0) >= targets['sharpe_ratio']:
            print("\n✅ SUCCESS: Strategy meets Sharpe ratio target")
        else:
            print("\n⚠️ CAUTION: Strategy below Sharpe ratio target")

        print("\n=== ANALYSIS COMPLETE ===")


def main():
    """Main entry point for command line execution."""
    parser = argparse.ArgumentParser(description='Pragmatic Asset Allocation Pipeline')
    parser.add_argument('--mode', choices=['data', 'signals', 'portfolio', 'backtest', 'full'],
                       default='full', help='Pipeline execution mode')
    parser.add_argument('--start-date', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', help='End date (YYYY-MM-DD)')
    parser.add_argument('--config', default='config.yaml', help='Configuration file path')

    args = parser.parse_args()

    # Initialize pipeline
    pipeline = PragmaticAssetAllocationPipeline(args.config)

    # Execute based on mode
    if args.mode == 'data':
        # Data acquisition only
        all_data = pipeline.run_data_acquisition(args.start_date, args.end_date)
        if all_data:
            print("✅ Data acquisition completed successfully")

    elif args.mode == 'signals':
        # Data + signals
        all_data = pipeline.run_data_acquisition(args.start_date, args.end_date)
        if all_data:
            signals = pipeline.run_signal_generation(all_data)
            if signals:
                print("✅ Signal generation completed successfully")

    elif args.mode == 'portfolio':
        # Data + signals + portfolio
        all_data = pipeline.run_data_acquisition(args.start_date, args.end_date)
        if all_data:
            signals = pipeline.run_signal_generation(all_data)
            if signals:
                portfolio = pipeline.run_portfolio_construction(signals, all_data)
                if portfolio:
                    print("✅ Portfolio construction completed successfully")

    elif args.mode == 'backtest':
        # Data + signals + portfolio + backtest
        all_data = pipeline.run_data_acquisition(args.start_date, args.end_date)
        if all_data:
            signals = pipeline.run_signal_generation(all_data)
            if signals:
                portfolio = pipeline.run_portfolio_construction(signals, all_data)
                if portfolio:
                    backtest = pipeline.run_backtest(portfolio, all_data)
                    if backtest:
                        print("✅ Backtesting completed successfully")

    elif args.mode == 'full':
        # Complete pipeline
        results = pipeline.run_full_pipeline(args.start_date, args.end_date)
        if results:
            pipeline.run_quick_analysis(results)
            print("✅ Full pipeline completed successfully")
        else:
            print("❌ Pipeline execution failed")
            sys.exit(1)


if __name__ == '__main__':
    main()