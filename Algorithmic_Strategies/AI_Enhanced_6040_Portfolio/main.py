"""
Enhanced Main Script for AI-Enhanced 60/40 Portfolio
Incorporates all critical improvements including walk-forward optimization.
"""

import yaml
import pandas as pd
import numpy as np
import os
from datetime import datetime
from typing import Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

from data_acquisition import DataAcquisition
from feature_engineering import FeatureEngineer
from ml_model import PortfolioMLModel
from backtester import PortfolioBacktester
from regime_detector import RegimeDetector


def walk_forward_optimization(config: Dict,
                              train_window: int = 60,
                              test_window: int = 12) -> Tuple:
    """
    Perform walk-forward optimization.

    Args:
        config: Configuration dictionary
        train_window: Training window in months
        test_window: Test window in months

    Returns:
        Tuple of (all_allocations, all_predictions, performance_metrics)
    """
    print("\n" + "="*70)
    print("WALK-FORWARD OPTIMIZATION")
    print("="*70)

    # Fetch data
    print("\n1. Fetching data...")
    data_acq = DataAcquisition(config)
    prices, returns, indicators = data_acq.get_full_dataset()

    # Initialize
    feature_eng = FeatureEngineer(config)
    all_allocations = []
    all_predictions = []

    # Calculate number of windows
    total_periods = len(returns)
    n_windows = (total_periods - train_window) // test_window

    print(f"\nTotal periods: {total_periods}")
    print(f"Training window: {train_window} months")
    print(f"Test window: {test_window} months")
    print(f"Number of windows: {n_windows}")

    for i in range(n_windows):
        start_train = i * test_window
        end_train = start_train + train_window
        end_test = min(end_train + test_window, total_periods)

        print(f"\n{'='*70}")
        print(f"Window {i+1}/{n_windows}")
        print(f"Train: {returns.index[start_train]} to {returns.index[end_train-1]}")
        print(f"Test: {returns.index[end_train]} to {returns.index[end_test-1]}")
        print(f"{'='*70}")

        # Split data
        train_prices = prices.iloc[start_train:end_train]
        train_returns = returns.iloc[start_train:end_train]
        train_indicators = indicators.iloc[start_train:end_train]

        test_prices = prices.iloc[end_train:end_test]
        test_returns = returns.iloc[end_train:end_test]
        test_indicators = indicators.iloc[end_train:end_test]

        # Engineer features with momentum and cross-asset
        train_features = feature_eng.engineer_all_features(
            train_indicators,
            prices=train_prices,
            returns=train_returns
        )
        train_features = feature_eng.prepare_features_for_training(train_features)

        # Engineer test features (use data up to end_test for feature calculation)
        test_features = feature_eng.engineer_all_features(
            indicators.iloc[:end_test],
            prices=prices.iloc[:end_test],
            returns=returns.iloc[:end_test]
        )
        test_features = feature_eng.prepare_features_for_training(test_features)
        # Select only test period features
        test_features = test_features.loc[test_returns.index.intersection(test_features.index)]

        # Create targets
        ml_model = PortfolioMLModel(config)
        train_targets = ml_model.create_target_variables(train_returns, lookback=1)

        # Align features and targets
        common_train = train_features.index.intersection(train_targets.index)
        X_train = train_features.loc[common_train]
        y_train = train_targets.loc[common_train]

        # FIX: Reset feature selector for each window to avoid using future information
        ml_model.feature_selector = None

        # Train models
        print(f"\nTraining models on {len(X_train)} samples...")
        ml_model.train_all_models(X_train, y_train)

        # Predict on test set
        common_test = test_features.index.intersection(test_returns.index)
        X_test = test_features.loc[common_test]

        print(f"Predicting on {len(X_test)} samples...")
        predictions = ml_model.predict_returns(X_test)

        # Calculate allocations with risk parity and regime adjustment
        # Detect regimes (FIXED: no lookahead bias)
        detector = RegimeDetector(config)
        if 'VIX' in indicators.columns and 'Yield_Spread' in indicators.columns:
            # Only use historical data available at prediction time
            historical_indicators = indicators.iloc[:end_train]
            historical_prices = prices.iloc[:end_train]
            regimes = detector.detect_combined_regime(
                historical_indicators['VIX'],
                historical_prices['SPY'],
                historical_indicators['Yield_Spread']
            )
            # Extend regimes to test period (assume regime persists)
            regimes = pd.Series([regimes.iloc[-1]] * len(test_returns), index=test_returns.index)
        else:
            regimes = None

        allocations = ml_model.calculate_optimal_allocations(
            predictions,
            historical_returns=returns.iloc[start_train:end_train],  # FIX: Only use historical data available at prediction time
            use_risk_parity=config.get('risk', {}).get('use_risk_parity', True),
            regimes=regimes
        )

        all_allocations.append(allocations)
        all_predictions.append(predictions)

    # Combine all windows
    final_allocations = pd.concat(all_allocations)
    final_predictions = pd.concat(all_predictions)

    print(f"\n{'='*70}")
    print("Walk-Forward Optimization Complete!")
    print(f"Total allocation periods: {len(final_allocations)}")
    print(f"{'='*70}")

    return final_allocations, final_predictions, prices, returns


def main():
    """Main execution function with all enhancements."""

    print("\n" + "="*70)
    print("AI-ENHANCED 60/40 PORTFOLIO - ENHANCED VERSION")
    print("="*70)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Load configuration
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # Create output directories
    os.makedirs('results', exist_ok=True)
    os.makedirs('figures', exist_ok=True)
    os.makedirs('models', exist_ok=True)

    # Run walk-forward optimization
    allocations, predictions, prices, returns = walk_forward_optimization(
        config,
        train_window=60,
        test_window=12
    )

    # Backtest with enhanced features
    print("\n" + "="*70)
    print("BACKTESTING")
    print("="*70)

    backtester = PortfolioBacktester(config)

    # AI Strategy with dynamic rebalancing and stop-loss
    print("\n1. AI-Enhanced Strategy (with dynamic rebalancing & stop-loss)...")
    ai_results = backtester.backtest_strategy(
        allocations,
        returns,
        prices,
        use_dynamic_rebalancing=True,
        use_stop_loss=True
    )

    # Benchmarks
    print("\n2. Buy & Hold SPY...")
    spy_results = backtester.create_benchmark_strategy(returns, prices, 'SPY')

    print("\n3. Traditional 60/40...")
    trad_results = backtester.create_traditional_6040(returns, prices)

    # Compare strategies
    strategies = {
        'AI-Enhanced (Walk-Forward)': ai_results,
        'Buy & Hold SPY': spy_results,
        'Traditional 60/40': trad_results
    }

    comparison = backtester.compare_strategies(strategies)

    print("\n" + "="*70)
    print("PERFORMANCE COMPARISON")
    print("="*70)
    print(comparison.round(4))

    # Statistical Testing
    if config.get('backtest', {}).get('statistical_testing', {}).get('enabled', False):
        print("\n" + "="*70)
        print("STATISTICAL TESTING")
        print("="*70)

        stat_tests = backtester.perform_statistical_tests(
            ai_results,
            spy_results,
            significance_level=config['backtest']['statistical_testing'].get('significance_level', 0.05)
        )

        print("\nReturns T-Test vs SPY:")
        print(f"  T-statistic: {stat_tests['returns_t_test']['t_statistic']:.4f}")
        print(f"  P-value: {stat_tests['returns_t_test']['p_value']:.4f}")
        print(f"  Significant: {stat_tests['returns_t_test']['significant']}")
        print(f"  AI Mean Return: {stat_tests['returns_t_test']['strategy_mean']:.6f}")
        print(f"  SPY Mean Return: {stat_tests['returns_t_test']['benchmark_mean']:.6f}")

        print(f"\nSharpe Ratio Comparison:")
        print(f"  AI Sharpe: {stat_tests['sharpe_ratio_comparison']['strategy_sharpe']:.4f}")
        print(f"  SPY Sharpe: {stat_tests['sharpe_ratio_comparison']['benchmark_sharpe']:.4f}")

    # Stress Testing
    if config.get('backtest', {}).get('stress_testing', {}).get('enabled', False):
        print("\n" + "="*70)
        print("STRESS TESTING")
        print("="*70)

        stress_scenarios = config['backtest']['stress_testing'].get('scenarios', [])
        stress_results = backtester.run_stress_tests(allocations, returns, prices, stress_scenarios)

        print("\nStress Test Results:")
        for scenario_name, results in stress_results.items():
            metrics = results['metrics']
            print(f"\n{scenario_name}:")
            print(f"  Total Return: {metrics['Total Return']:.2%}")
            print(f"  Max Drawdown: {metrics['Max Drawdown']:.2%}")
            print(f"  Sharpe Ratio: {metrics['Sharpe Ratio']:.4f}")

    # Regime analysis
    print("\n" + "="*70)
    print("REGIME ANALYSIS")
    print("="*70)

    data_acq = DataAcquisition(config)
    _, _, indicators = data_acq.get_full_dataset()

    detector = RegimeDetector(config)
    if 'VIX' in indicators.columns and 'Yield_Spread' in indicators.columns:
        regime = detector.detect_combined_regime(
            indicators['VIX'],
            prices['SPY'],
            indicators['Yield_Spread']
        )
        print(f"\nRegime distribution:")
        print(regime.value_counts())
        print(f"\nCurrent regime: {['Defensive', 'Neutral', 'Aggressive'][int(regime.iloc[-1])]}")

    # Save results
    print("\n" + "="*70)
    print("SAVING RESULTS")
    print("="*70)

    allocations.to_csv('results/allocations_enhanced.csv')
    predictions.to_csv('results/predictions_enhanced.csv')
    comparison.to_csv('results/performance_comparison_enhanced.csv')

    print("\nSaved:")
    print("  - results/allocations_enhanced.csv")
    print("  - results/predictions_enhanced.csv")
    print("  - results/performance_comparison_enhanced.csv")

    # Generate plots
    print("\nGenerating visualizations...")

    fig1 = backtester.plot_portfolio_value(strategies)
    fig1.savefig('figures/portfolio_value_enhanced.png', dpi=300, bbox_inches='tight')

    fig2 = backtester.plot_drawdown(strategies)
    fig2.savefig('figures/drawdown_enhanced.png', dpi=300, bbox_inches='tight')

    fig3 = backtester.plot_monthly_returns(ai_results)
    fig3.savefig('figures/monthly_returns_enhanced.png', dpi=300, bbox_inches='tight')

    fig4 = backtester.plot_allocations(allocations)
    fig4.savefig('figures/allocations_enhanced.png', dpi=300, bbox_inches='tight')

    print("\nSaved figures:")
    print("  - figures/portfolio_value_enhanced.png")
    print("  - figures/drawdown_enhanced.png")
    print("  - figures/monthly_returns_enhanced.png")
    print("  - figures/allocations_enhanced.png")

    print("\n" + "="*70)
    print("COMPLETE!")
    print("="*70)
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


    return comparison


if __name__ == "__main__":
    results = main()


def walk_forward_optimization(config: Dict, 
                              train_window: int = 60,
                              test_window: int = 12) -> Tuple:
    """
    Perform walk-forward optimization.
    
    Args:
        config: Configuration dictionary
        train_window: Training window in months
        test_window: Test window in months
        
    Returns:
        Tuple of (all_allocations, all_predictions, performance_metrics)
    """
    print("\n" + "="*70)
    print("WALK-FORWARD OPTIMIZATION")
    print("="*70)
    
    # Fetch data
    print("\n1. Fetching data...")
    data_acq = DataAcquisition(config)
    prices, returns, indicators = data_acq.get_full_dataset()
    
    # Initialize
    feature_eng = FeatureEngineer(config)
    all_allocations = []
    all_predictions = []
    
    # Calculate number of windows
    total_periods = len(returns)
    n_windows = (total_periods - train_window) // test_window
    
    print(f"\nTotal periods: {total_periods}")
    print(f"Training window: {train_window} months")
    print(f"Test window: {test_window} months")
    print(f"Number of windows: {n_windows}")
    
    for i in range(n_windows):
        start_train = i * test_window
        end_train = start_train + train_window
        end_test = min(end_train + test_window, total_periods)
        
        print(f"\n{'='*70}")
        print(f"Window {i+1}/{n_windows}")
        print(f"Train: {returns.index[start_train]} to {returns.index[end_train-1]}")
        print(f"Test: {returns.index[end_train]} to {returns.index[end_test-1]}")
        print(f"{'='*70}")
        
        # Split data
        train_prices = prices.iloc[start_train:end_train]
        train_returns = returns.iloc[start_train:end_train]
        train_indicators = indicators.iloc[start_train:end_train]
        
        test_prices = prices.iloc[end_train:end_test]
        test_returns = returns.iloc[end_train:end_test]
        test_indicators = indicators.iloc[end_train:end_test]
        
        # Engineer features with momentum and cross-asset (FIXED: no lookahead bias)
        train_features = feature_eng.engineer_all_features(
            train_indicators,
            prices=train_prices,
            returns=train_returns
        )
        train_features = feature_eng.prepare_features_for_training(train_features)

        # CRITICAL FIX: Engineer test features properly without lookahead bias
        # We need features for the test period, but calculated using data available at prediction time
        # Use a rolling approach: for each test date, use data up to that date
        test_features_list = []
        for test_date in test_returns.index:
            # Use data up to the test date for feature calculation
            historical_data_up_to_test = indicators.loc[:test_date].iloc[:-1]  # Exclude current date
            historical_prices_up_to_test = prices.loc[:test_date].iloc[:-1]
            historical_returns_up_to_test = returns.loc[:test_date].iloc[:-1]

            if len(historical_data_up_to_test) >= 12:  # Need minimum history for features
                features_at_test_date = feature_eng.engineer_all_features(
                    historical_data_up_to_test,
                    prices=historical_prices_up_to_test,
                    returns=historical_returns_up_to_test
                )
                features_at_test_date = feature_eng.prepare_features_for_training(features_at_test_date)

                if len(features_at_test_date) > 0:
                    # Use the most recent available features
                    latest_features = features_at_test_date.iloc[-1:]
                    latest_features.index = [test_date]
                    test_features_list.append(latest_features)

        if test_features_list:
            test_features = pd.concat(test_features_list)
        else:
            # Fallback: use last training features repeated
            if len(train_features) > 0:
                last_train_features = train_features.iloc[-1:]
                test_features = pd.DataFrame(
                    np.repeat(last_train_features.values, len(test_returns), axis=0),
                    index=test_returns.index,
                    columns=last_train_features.columns
                )
            else:
                test_features = pd.DataFrame(index=test_returns.index)
        ml_model = PortfolioMLModel(config)
        train_targets = ml_model.create_target_variables(train_returns, lookback=1)
        
        # Align features and targets
        common_train = train_features.index.intersection(train_targets.index)
        X_train = train_features.loc[common_train]
        y_train = train_targets.loc[common_train]
        
        # FIX: Reset feature selector for each window to avoid using future information
        ml_model.feature_selector = None
        
        # Train models
        print(f"\nTraining models on {len(X_train)} samples...")
        ml_model.train_all_models(X_train, y_train)
        
        # Predict on test set
        common_test = test_features.index.intersection(test_returns.index)
        X_test = test_features.loc[common_test]
        
        print(f"Predicting on {len(X_test)} samples...")
        predictions = ml_model.predict_returns(X_test)
        
        # Calculate allocations with risk parity and regime adjustment
        # Detect regimes (FIXED: no lookahead bias)
        detector = RegimeDetector(config)
        if 'VIX' in indicators.columns and 'Yield_Spread' in indicators.columns:
            # Only use historical data available at prediction time
            historical_indicators = indicators.iloc[:end_train]
            historical_prices = prices.iloc[:end_train]
            regimes = detector.detect_combined_regime(
                historical_indicators['VIX'],
                historical_prices['SPY'],
                historical_indicators['Yield_Spread']
            )
            # Extend regimes to test period (assume regime persists)
            regimes = pd.Series([regimes.iloc[-1]] * len(test_returns), index=test_returns.index)
        else:
            regimes = None
        
        allocations = ml_model.calculate_optimal_allocations(
            predictions,
            historical_returns=returns.iloc[start_train:end_train],  # FIX: Only use historical data available at prediction time
            use_risk_parity=config.get('risk', {}).get('use_risk_parity', True),
            regimes=regimes
        )
        
        all_allocations.append(allocations)
        all_predictions.append(predictions)
    
    # Combine all windows
    final_allocations = pd.concat(all_allocations)
    final_predictions = pd.concat(all_predictions)
    
    print(f"\n{'='*70}")
    print("Walk-Forward Optimization Complete!")
    print(f"Total allocation periods: {len(final_allocations)}")
    print(f"{'='*70}")
    
    return final_allocations, final_predictions, prices, returns


def main():
    """Main execution function with all enhancements."""
    
    print("\n" + "="*70)
    print("AI-ENHANCED 60/40 PORTFOLIO - ENHANCED VERSION")
    print("="*70)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Load configuration
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Create output directories
    os.makedirs('results', exist_ok=True)
    os.makedirs('figures', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    # Run walk-forward optimization
    allocations, predictions, prices, returns = walk_forward_optimization(
        config,
        train_window=60,
        test_window=12
    )
    
    # Backtest with enhanced features
    print("\n" + "="*70)
    print("BACKTESTING")
    print("="*70)
    
    backtester = PortfolioBacktester(config)
    
    # AI Strategy with dynamic rebalancing and stop-loss
    print("\n1. AI-Enhanced Strategy (with dynamic rebalancing & stop-loss)...")
    ai_results = backtester.backtest_strategy(
        allocations, 
        returns, 
        prices,
        use_dynamic_rebalancing=True,
        use_stop_loss=True
    )
    
    # Benchmarks
    print("\n2. Buy & Hold SPY...")
    spy_results = backtester.create_benchmark_strategy(returns, prices, 'SPY')
    
    print("\n3. Traditional 60/40...")
    trad_results = backtester.create_traditional_6040(returns, prices)
    
    # Compare strategies
    strategies = {
        'AI-Enhanced (Walk-Forward)': ai_results,
        'Buy & Hold SPY': spy_results,
        'Traditional 60/40': trad_results
    }
    
    comparison = backtester.compare_strategies(strategies)
    
    print("\n" + "="*70)
    print("PERFORMANCE COMPARISON")
    print("="*70)
    print(comparison.round(4))
    
    # Statistical Testing
    if config.get('backtest', {}).get('statistical_testing', {}).get('enabled', False):
        print("\n" + "="*70)
        print("STATISTICAL TESTING")
        print("="*70)
        
        stat_tests = backtester.perform_statistical_tests(
            ai_results, 
            spy_results,
            significance_level=config['backtest']['statistical_testing'].get('significance_level', 0.05)
        )
        
        print("\nReturns T-Test vs SPY:")
        print(f"  T-statistic: {stat_tests['returns_t_test']['t_statistic']:.4f}")
        print(f"  P-value: {stat_tests['returns_t_test']['p_value']:.4f}")
        print(f"  Significant: {stat_tests['returns_t_test']['significant']}")
        print(f"  AI Mean Return: {stat_tests['returns_t_test']['strategy_mean']:.6f}")
        print(f"  SPY Mean Return: {stat_tests['returns_t_test']['benchmark_mean']:.6f}")
        
        print(f"\nSharpe Ratio Comparison:")
        print(f"  AI Sharpe: {stat_tests['sharpe_ratio_comparison']['strategy_sharpe']:.4f}")
        print(f"  SPY Sharpe: {stat_tests['sharpe_ratio_comparison']['benchmark_sharpe']:.4f}")
    
    # Stress Testing
    if config.get('backtest', {}).get('stress_testing', {}).get('enabled', False):
        print("\n" + "="*70)
        print("STRESS TESTING")
        print("="*70)
        
        stress_scenarios = config['backtest']['stress_testing'].get('scenarios', [])
        stress_results = backtester.run_stress_tests(allocations, returns, prices, stress_scenarios)
        
        print("\nStress Test Results:")
        for scenario_name, results in stress_results.items():
            metrics = results['metrics']
            print(f"\n{scenario_name}:")
            print(f"  Total Return: {metrics['Total Return']:.2%}")
            print(f"  Max Drawdown: {metrics['Max Drawdown']:.2%}")
            print(f"  Sharpe Ratio: {metrics['Sharpe Ratio']:.4f}")
    
    # Regime analysis
    print("\n" + "="*70)
    print("REGIME ANALYSIS")
    print("="*70)
    
    data_acq = DataAcquisition(config)
    _, _, indicators = data_acq.get_full_dataset()
    
    detector = RegimeDetector(config)
    if 'VIX' in indicators.columns and 'Yield_Spread' in indicators.columns:
        regime = detector.detect_combined_regime(
            indicators['VIX'],
            prices['SPY'],
            indicators['Yield_Spread']
        )
        print(f"\nRegime distribution:")
        print(regime.value_counts())
        print(f"\nCurrent regime: {['Defensive', 'Neutral', 'Aggressive'][int(regime.iloc[-1])]}")
    
    # Save results
    print("\n" + "="*70)
    print("SAVING RESULTS")
    print("="*70)
    
    allocations.to_csv('results/allocations_enhanced.csv')
    predictions.to_csv('results/predictions_enhanced.csv')
    comparison.to_csv('results/performance_comparison_enhanced.csv')
    
    print("\nSaved:")
    print("  - results/allocations_enhanced.csv")
    print("  - results/predictions_enhanced.csv")
    print("  - results/performance_comparison_enhanced.csv")
    
    # Generate plots
    print("\nGenerating visualizations...")
    
    fig1 = backtester.plot_portfolio_value(strategies)
    fig1.savefig('figures/portfolio_value_enhanced.png', dpi=300, bbox_inches='tight')
    
    fig2 = backtester.plot_drawdown(strategies)
    fig2.savefig('figures/drawdown_enhanced.png', dpi=300, bbox_inches='tight')
    
    fig3 = backtester.plot_monthly_returns(ai_results)
    fig3.savefig('figures/monthly_returns_enhanced.png', dpi=300, bbox_inches='tight')
    
    fig4 = backtester.plot_allocations(allocations)
    fig4.savefig('figures/allocations_enhanced.png', dpi=300, bbox_inches='tight')
    
    print("\nSaved figures:")
    print("  - figures/portfolio_value_enhanced.png")
    print("  - figures/drawdown_enhanced.png")
    print("  - figures/monthly_returns_enhanced.png")
    print("  - figures/allocations_enhanced.png")
    
    print("\n" + "="*70)
    print("COMPLETE!")
    print("="*70)
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    
    return comparison


if __name__ == "__main__":
    results = main()
