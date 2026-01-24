"""
Main Pipeline

End-to-end execution of the Intraday Momentum Breakout Strategy.

Workflow:
1. Load configuration
2. Download futures data (ES & NQ)
3. Calculate noise area boundaries
4. Generate breakout signals
5. Size positions (volatility targeting)
6. Run backtest with transaction costs
7. Evaluate performance
8. Generate reports

Usage:
    python main.py --start_date 2023-01-01 --end_date 2023-12-31
"""

import yaml
import argparse
import pandas as pd
import numpy as np
from datetime import datetime
import os
import warnings
warnings.filterwarnings('ignore')

# Import modules
from data_acquisition import FuturesDataDownloader
from noise_area import NoiseAreaCalculator
from signal_generator import SignalGenerator
from position_sizer import PositionSizer
from backtester import Backtester
from performance_evaluator import PerformanceEvaluator, visualize_performance


def load_config(config_path: str = 'config.yaml') -> dict:
    """
    Load configuration from YAML file.
    
    Parameters
    ----------
    config_path : str
        Path to config file
        
    Returns
    -------
    dict
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_results_directory():
    """Create results directory if it doesn't exist."""
    os.makedirs('results', exist_ok=True)
    os.makedirs('data', exist_ok=True)


def run_pipeline(
    config: dict,
    start_date: str = None,
    end_date: str = None,
    use_cached_data: bool = False
):
    """
    Run complete strategy pipeline.
    
    Parameters
    ----------
    config : dict
        Configuration dictionary
    start_date : str, optional
        Override start date
    end_date : str, optional
        Override end date
    use_cached_data : bool
        Use cached data if available
    """
    print("\n" + "="*60)
    print("INTRADAY MOMENTUM BREAKOUT STRATEGY")
    print("ES & NQ Futures")
    print("="*60)
    
    # Override dates if provided
    if start_date:
        config['data']['start_date'] = start_date
    if end_date:
        config['data']['end_date'] = end_date
    
    print(f"\nConfiguration:")
    print(f"  Lookback: {config['strategy']['noise_area']['lookback_days']} days")
    print(f"  Target Volatility: {config['strategy']['position_sizing']['target_daily_volatility']}%")
    print(f"  Max Leverage: {config['strategy']['position_sizing']['max_leverage']}x")
    print(f"  Initial Capital: ${config['strategy']['portfolio']['initial_capital']:,}")
    print(f"  Date Range: {config['data']['start_date']} to {config['data']['end_date']}")
    
    # Step 1: Data Acquisition
    print("\n" + "-"*60)
    print("STEP 1: DATA ACQUISITION")
    print("-"*60)
    
    downloader = FuturesDataDownloader(config)
    
    if use_cached_data:
        try:
            data = downloader.load_data('data')
            if len(data) == 2:
                print("Loaded cached data")
            else:
                raise FileNotFoundError("Cached data incomplete")
        except:
            print("Cached data not found, downloading...")
            data = downloader.download_all_data()
            downloader.save_data(data, 'data')
    else:
        data = downloader.download_all_data()
        downloader.save_data(data, 'data')
    
    es_data = data['ES'].copy()
    nq_data = data['NQ'].copy()
    
    # Step 2: Noise Area Calculation
    print("\n" + "-"*60)
    print("STEP 2: NOISE AREA CALCULATION")
    print("-"*60)
    
    calculator = NoiseAreaCalculator(config)
    
    es_data = calculator.calculate_noise_area(es_data)
    es_data = calculator.identify_breakouts(es_data)
    
    nq_data = calculator.calculate_noise_area(nq_data)
    nq_data = calculator.identify_breakouts(nq_data)
    
    # Step 3: Signal Generation
    print("\n" + "-"*60)
    print("STEP 3: SIGNAL GENERATION")
    print("-"*60)
    
    signal_gen = SignalGenerator(config)
    
    es_data = signal_gen.generate_signals(es_data)
    nq_data = signal_gen.generate_signals(nq_data)
    
    # Step 4: Position Sizing
    print("\n" + "-"*60)
    print("STEP 4: POSITION SIZING")
    print("-"*60)
    
    sizer = PositionSizer(config)
    portfolio = sizer.calculate_portfolio_positions(es_data, nq_data)
    
    # Step 5: Backtesting
    print("\n" + "-"*60)
    print("STEP 5: BACKTESTING")
    print("-"*60)
    
    backtester = Backtester(config)
    equity_curve = backtester.run_backtest(portfolio)
    trades_df = backtester.get_trades_dataframe()
    
    # Step 6: Performance Evaluation
    print("\n" + "-"*60)
    print("STEP 6: PERFORMANCE EVALUATION")
    print("-"*60)
    
    evaluator = PerformanceEvaluator(config)
    metrics = evaluator.evaluate_strategy(equity_curve, trades_df)
    
    # Step 7: Save Results
    print("\n" + "-"*60)
    print("STEP 7: SAVING RESULTS")
    print("-"*60)
    
    # Save data
    portfolio['ES_momentum'].to_csv('results/es_momentum_data.csv')
    portfolio['NQ_momentum'].to_csv('results/nq_momentum_data.csv')
    portfolio['NQ_long_only'].to_csv('results/nq_long_only_data.csv')
    
    # Save backtest results
    equity_curve.to_csv('results/equity_curve.csv')
    trades_df.to_csv('results/trades.csv')
    
    # Save metrics
    metrics_df = pd.DataFrame([metrics]).T
    metrics_df.columns = ['Value']
    metrics_df.to_csv('results/performance_metrics.csv')
    
    print("\nResults saved to results/ directory:")
    print("  - equity_curve.csv")
    print("  - trades.csv")
    print("  - performance_metrics.csv")
    print("  - es_momentum_data.csv")
    print("  - nq_momentum_data.csv")
    print("  - nq_long_only_data.csv")
    
    # Step 8: Visualization
    print("\n" + "-"*60)
    print("STEP 8: VISUALIZATION")
    print("-"*60)
    
    visualize_performance(equity_curve, trades_df, metrics)
    
    # Final Summary
    print("\n" + "="*60)
    print("PIPELINE COMPLETE")
    print("="*60)
    print(f"\nKey Results:")
    print(f"  Total Return: {metrics['total_return']*100:.2f}%")
    print(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
    print(f"  Max Drawdown: {metrics['max_drawdown']*100:.2f}%")
    print(f"  Win Rate: {metrics.get('win_rate', 0)*100:.1f}%")
    print(f"  Total Trades: {metrics.get('total_trades', 0)}")
    
    # Check if strategy meets minimum Sharpe threshold
    min_sharpe = config['strategy']['backtesting']['min_sharpe_ratio']
    if metrics['sharpe_ratio'] >= min_sharpe:
        print(f"\n✓ Strategy PASSED minimum Sharpe threshold ({min_sharpe})")
    else:
        print(f"\n✗ Strategy FAILED minimum Sharpe threshold ({min_sharpe})")
    
    print("="*60)
    
    return metrics


def main():
    """
    Main entry point.
    """
    # Parse arguments
    parser = argparse.ArgumentParser(description='Intraday Momentum Breakout Strategy')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to config file')
    parser.add_argument('--start_date', type=str, default=None,
                       help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end_date', type=str, default=None,
                       help='End date (YYYY-MM-DD)')
    parser.add_argument('--use_cached_data', action='store_true',
                       help='Use cached data if available')
    
    args = parser.parse_args()
    
    # Create directories
    create_results_directory()
    
    # Load config
    config = load_config(args.config)
    
    # Run pipeline
    metrics = run_pipeline(
        config,
        start_date=args.start_date,
        end_date=args.end_date,
        use_cached_data=args.use_cached_data
    )
    
    print("\nStrategy execution complete. Check results/ directory for outputs.")


if __name__ == "__main__":
    main()
