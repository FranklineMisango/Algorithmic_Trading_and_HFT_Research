"""
Main Pipeline for Dividend-Price Ratio Market Timing Strategy

This script runs the complete end-to-end workflow:
1. Data acquisition
2. Feature engineering
3. Model training
4. Out-of-sample validation
5. Trading strategy backtest
6. Performance analysis
7. Visualization
"""

import yaml
import os
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import modules
from data_acquisition import DividendPriceDataFetcher
from feature_engineering import DPRatioFeatureEngineer
from ols_model import DPRatioOLSModel
from trading_strategy import DPRatioTradingStrategy


def create_output_directories():
    """Create necessary output directories."""
    os.makedirs('results', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    print("Output directories created.\n")


def load_configuration(config_file='config.yaml'):
    """Load configuration from YAML file."""
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    print(f"Configuration loaded from {config_file}\n")
    return config


def run_pipeline(config):
    """
    Execute complete strategy pipeline.
    
    Parameters
    ----------
    config : dict
        Configuration dictionary
    """
    print("="*80)
    print(" DIVIDEND-PRICE RATIO MARKET TIMING STRATEGY")
    print(" Complete Pipeline Execution")
    print("="*80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Step 1: Data Acquisition
    print("\n" + "="*80)
    print("STEP 1: DATA ACQUISITION")
    print("="*80)
    fetcher = DividendPriceDataFetcher(config)
    data = fetcher.fetch_and_prepare_data()
    data.to_csv('results/sp500_monthly_data.csv')
    print("\nData saved to results/sp500_monthly_data.csv")
    
    # Step 2: Feature Engineering
    print("\n" + "="*80)
    print("STEP 2: FEATURE ENGINEERING")
    print("="*80)
    engineer = DPRatioFeatureEngineer(config)
    data = engineer.engineer_all_features(data)
    X, y = engineer.prepare_model_dataset(data)
    data.to_csv('results/engineered_features.csv')
    print("\nFeatures saved to results/engineered_features.csv")
    
    # Step 3: Model Training (In-Sample)
    print("\n" + "="*80)
    print("STEP 3: MODEL TRAINING")
    print("="*80)
    model = DPRatioOLSModel(config)
    X_in, X_out, y_in, y_out = model.split_data(X, y)
    model.train_ols_model(X_in, y_in)
    model.print_model_summary()
    
    # Step 4: Model Diagnostics
    print("\n" + "="*80)
    print("STEP 4: MODEL DIAGNOSTICS")
    print("="*80)
    diagnostics = model.diagnose_model(X_in, y_in)
    
    # Step 5: Out-of-Sample Validation
    print("\n" + "="*80)
    print("STEP 5: OUT-OF-SAMPLE VALIDATION")
    print("="*80)
    model.validate_out_of_sample(X_out, y_out)
    model.save_model('models/ols_model.pkl')
    
    # Step 6: Trading Strategy Backtest
    print("\n" + "="*80)
    print("STEP 6: TRADING STRATEGY BACKTEST")
    print("="*80)
    strategy = DPRatioTradingStrategy(config, model)
    
    # Generate signals for out-of-sample period
    signals_out = strategy.generate_signals(X_out)
    backtest_results = strategy.backtest_strategy(signals_out, y_out)
    backtest_results.to_csv('results/backtest_results.csv')
    print("\nBacktest results saved to results/backtest_results.csv")
    
    # Step 7: Performance Analysis
    print("\n" + "="*80)
    print("STEP 7: PERFORMANCE ANALYSIS")
    print("="*80)
    performance = strategy.calculate_performance_metrics(backtest_results)
    signal_analysis = strategy.analyze_signal_effectiveness(backtest_results)
    
    # Save performance metrics
    all_metrics = {
        'performance': performance,
        'signal_analysis': signal_analysis,
        'in_sample': model.in_sample_results,
        'out_sample': model.out_sample_results
    }
    
    # Convert numpy types to native Python types for JSON
    def convert_to_json_serializable(obj):
        """Convert numpy types to native Python types."""
        import numpy as np
        if isinstance(obj, dict):
            return {k: convert_to_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj
    
    all_metrics_serializable = convert_to_json_serializable(all_metrics)
    
    with open('results/performance_metrics.json', 'w') as f:
        json.dump(all_metrics_serializable, f, indent=4)
    print("\nPerformance metrics saved to results/performance_metrics.json")
    
    # Step 8: Summary
    print("\n" + "="*80)
    print("PIPELINE EXECUTION SUMMARY")
    print("="*80)
    
    print(f"\nModel Quality (In-Sample):")
    print(f"  R²:            {model.in_sample_results['r_squared']:.4f}")
    print(f"  Beta:          {model.in_sample_results['beta']:.6f}")
    print(f"  p-value:       {model.in_sample_results['p_value']:.6f}")
    print(f"  RMSE:          {model.in_sample_results['rmse']:.4f}")
    
    print(f"\nModel Quality (Out-of-Sample):")
    print(f"  RMSE:          {model.out_sample_results['rmse']:.4f}")
    print(f"  vs Naive:      {model.out_sample_results['rmse_naive']:.4f}")
    print(f"  Directional:   {model.out_sample_results['directional_accuracy']:.2%}")
    
    print(f"\nStrategy Performance (Out-of-Sample):")
    print(f"  Annual Return: {performance['annual_return_strategy']:+.2%}")
    print(f"  Benchmark:     {performance['annual_return_benchmark']:+.2%}")
    print(f"  Alpha:         {performance['alpha']:+.2%}")
    print(f"  Sharpe Ratio:  {performance['sharpe_strategy']:.3f}")
    print(f"  Max Drawdown:  {performance['max_drawdown_strategy']:.2%}")
    print(f"  Win Rate:      {performance['win_rate']:.2%}")
    
    print(f"\nFiles Generated:")
    print(f"  - results/sp500_monthly_data.csv")
    print(f"  - results/engineered_features.csv")
    print(f"  - results/backtest_results.csv")
    print(f"  - results/performance_metrics.json")
    print(f"  - models/ols_model.pkl")
    
    print(f"\nEnd time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    return {
        'model': model,
        'strategy': strategy,
        'backtest_results': backtest_results,
        'performance': performance,
        'data': data
    }


def main():
    """Main entry point."""
    # Create directories
    create_output_directories()
    
    # Load config
    config = load_configuration()
    
    # Run pipeline
    results = run_pipeline(config)
    
    print("\n" + "="*80)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("="*80)
    print("\nNext steps:")
    print("  1. Review results/performance_metrics.json")
    print("  2. Examine results/backtest_results.csv")
    print("  3. Run visualization.py for charts")
    print("  4. Open notebooks/complete_analysis.ipynb for interactive analysis")
    print("\n" + "="*80)


if __name__ == "__main__":
    main()
