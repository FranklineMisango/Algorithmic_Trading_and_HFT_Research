"""
Main script for Regime-Based ML Strategy

This approach focuses on regime detection rather than return prediction.
"""

import yaml
import pandas as pd
import numpy as np
from typing import Dict, Tuple
from data_acquisition import DataAcquisition
from regime_ml_strategy import RegimeMLStrategy
from backtester import PortfolioBacktester
import matplotlib.pyplot as plt
import seaborn as sns


def walk_forward_regime_strategy(config: Dict,
                                 train_window: int = 60,
                                 test_window: int = 12,
                                 model_name: str = 'ensemble') -> Tuple:
    """
    Walk-forward optimization using regime-based strategy.
    
    Args:
        config: Configuration dictionary
        train_window: Training window in months
        test_window: Test window in months
        model_name: Model to use ('gradient_boosting', 'random_forest', 'neural_network', 'ensemble')
    
    Returns:
        Tuple of (allocations, regime_predictions, prices, returns)
    """
    print("\n" + "="*70)
    print("REGIME-BASED ML STRATEGY - WALK-FORWARD OPTIMIZATION")
    print("="*70)
    
    # Fetch data
    print("\n1. Fetching data...")
    data_acq = DataAcquisition(config)
    prices, returns, indicators = data_acq.get_full_dataset()
    
    # Initialize strategy
    strategy = RegimeMLStrategy(config)
    
    # Calculate number of windows
    total_periods = len(returns)
    n_windows = (total_periods - train_window) // test_window
    
    print(f"\nTotal periods: {total_periods}")
    print(f"Training window: {train_window} months")
    print(f"Test window: {test_window} months")
    print(f"Number of windows: {n_windows}")
    print(f"Model: {model_name}")
    
    all_allocations = []
    all_regimes = []
    all_accuracies = []
    
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
        
        # Create regime labels for training (uses forward-looking data)
        print("\nCreating regime labels...")
        train_regimes = strategy.create_regime_labels(
            train_indicators['VIX'],
            train_prices['SPY'],
            train_indicators['Yield_Spread'],
            train_returns['SPY']
        )
        
        # Create features
        print("Engineering features...")
        train_features = strategy.create_features(train_indicators, train_prices, train_returns)
        
        # For test, use all data up to test period (no lookahead)
        test_features = strategy.create_features(
            indicators.iloc[:end_test],
            prices.iloc[:end_test],
            returns.iloc[:end_test]
        )
        test_features = test_features.loc[test_returns.index.intersection(test_features.index)]
        
        # Align training data
        common_train = train_features.index.intersection(train_regimes.index)
        X_train = train_features.loc[common_train]
        y_train = train_regimes.loc[common_train]
        
        # Remove samples with missing regime labels
        valid_idx = ~y_train.isna()
        X_train = X_train[valid_idx]
        y_train = y_train[valid_idx]
        
        print(f"\nTraining samples: {len(X_train)}")
        print(f"Regime distribution: {y_train.value_counts().to_dict()}")
        
        # Train models
        strategy.train_models(X_train, y_train)
        
        # Predict regimes on test set
        print(f"\nPredicting regimes on {len(test_features)} samples...")
        predicted_regimes = strategy.predict_regime(test_features, model_name=model_name)
        
        print(f"Predicted regime distribution: {predicted_regimes.value_counts().to_dict()}")
        
        # Convert regimes to allocations
        allocations = strategy.get_allocations_for_regime(predicted_regimes)
        
        all_allocations.append(allocations)
        all_regimes.append(predicted_regimes)
    
    # Combine all windows
    final_allocations = pd.concat(all_allocations)
    final_regimes = pd.concat(all_regimes)
    
    print(f"\n{'='*70}")
    print("Walk-Forward Optimization Complete!")
    print(f"Total allocation periods: {len(final_allocations)}")
    print(f"Overall regime distribution: {final_regimes.value_counts().to_dict()}")
    print(f"{'='*70}")
    
    return final_allocations, final_regimes, prices, returns


def compare_all_models(config: Dict) -> pd.DataFrame:
    """
    Compare all models: Gradient Boosting, Random Forest, Neural Network, Ensemble.
    
    Returns:
        DataFrame with performance comparison
    """
    print("\n" + "="*70)
    print("COMPARING ALL MODELS")
    print("="*70)
    
    models = ['gradient_boosting', 'random_forest', 'neural_network', 'ensemble']
    results = []
    
    for model_name in models:
        print(f"\n\nTesting {model_name.upper()}...")
        print("-" * 70)
        
        # Run strategy
        allocations, regimes, prices, returns = walk_forward_regime_strategy(
            config,
            train_window=60,
            test_window=12,
            model_name=model_name
        )
        
        # Backtest
        backtester = PortfolioBacktester(config)
        
        # Align data
        common_idx = allocations.index.intersection(returns.index)
        allocations_aligned = allocations.loc[common_idx]
        returns_aligned = returns.loc[common_idx]
        prices_aligned = prices.loc[common_idx]
        
        # Run backtest
        backtest_results = backtester.backtest_strategy(
            allocations_aligned,
            returns_aligned,
            prices_aligned
        )
        
        # Calculate metrics
        metrics = backtester.calculate_all_metrics(backtest_results)
        metrics['Model'] = model_name
        metrics['Regime_Changes'] = (regimes.diff() != 0).sum()
        
        results.append(metrics)
    
    comparison_df = pd.DataFrame(results)
    comparison_df = comparison_df.set_index('Model')
    
    print("\n" + "="*70)
    print("MODEL COMPARISON RESULTS")
    print("="*70)
    print(comparison_df[['Sharpe Ratio', 'CAGR', 'Max Drawdown', 'Volatility', 'Regime_Changes']])
    
    return comparison_df


def main():
    """Main execution function."""
    print("\n" + "="*70)
    print("AI-ENHANCED PORTFOLIO - REGIME-BASED STRATEGY")
    print("="*70)
    
    # Load configuration
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Option 1: Run with best model (ensemble)
    print("\n\nOPTION 1: Running with Ensemble Model...")
    allocations, regimes, prices, returns = walk_forward_regime_strategy(
        config,
        train_window=60,
        test_window=12,
        model_name='ensemble'
    )
    
    # Backtest
    print("\n\n2. Running Backtest...")
    backtester = PortfolioBacktester(config)
    
    # Align data
    common_idx = allocations.index.intersection(returns.index)
    allocations_aligned = allocations.loc[common_idx]
    returns_aligned = returns.loc[common_idx]
    prices_aligned = prices.loc[common_idx]
    
    # Backtest AI strategy
    ai_results = backtester.backtest_strategy(
        allocations_aligned,
        returns_aligned,
        prices_aligned
    )
    
    # Backtest benchmarks
    spy_results = backtester.create_benchmark_strategy(returns_aligned, prices_aligned)
    traditional_results = backtester.create_traditional_6040(returns_aligned, prices_aligned)
    
    # Compare strategies
    print("\n\n3. Comparing Strategies...")
    strategies = {
        'Regime-Based AI': ai_results,
        'Buy & Hold SPY': spy_results,
        'Traditional 60/40': traditional_results
    }
    
    comparison = backtester.compare_strategies(strategies)
    
    print("\n" + "="*70)
    print("PERFORMANCE COMPARISON")
    print("="*70)
    print(comparison)
    
    # Save results
    print("\n\n4. Saving Results...")
    allocations_aligned.to_csv('results/allocations_regime.csv')
    regimes.to_csv('results/regimes_predicted.csv')
    comparison.to_csv('results/performance_comparison_regime.csv')
    
    # Create visualizations
    print("\n5. Creating Visualizations...")
    
    # Plot portfolio value
    fig = backtester.plot_portfolio_value(strategies)
    fig.savefig('figures/portfolio_value_regime.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot drawdown
    fig = backtester.plot_drawdown(strategies)
    fig.savefig('figures/drawdown_regime.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot allocations
    fig = backtester.plot_allocations(allocations_aligned)
    fig.savefig('figures/allocations_regime.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot regime distribution
    plt.figure(figsize=(12, 6))
    regimes.value_counts().sort_index().plot(kind='bar')
    plt.title('Regime Distribution', fontsize=14, fontweight='bold')
    plt.xlabel('Regime (0=Defensive, 1=Neutral, 2=Aggressive)')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig('figures/regime_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot regime over time
    plt.figure(figsize=(14, 6))
    plt.plot(regimes.index, regimes.values, marker='o', linestyle='-', markersize=3)
    plt.title('Predicted Regimes Over Time', fontsize=14, fontweight='bold')
    plt.xlabel('Date')
    plt.ylabel('Regime (0=Defensive, 1=Neutral, 2=Aggressive)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('figures/regime_timeline.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("\n" + "="*70)
    print("REGIME-BASED STRATEGY COMPLETE!")
    print("="*70)
    print("\nResults saved to:")
    print("  - results/allocations_regime.csv")
    print("  - results/regimes_predicted.csv")
    print("  - results/performance_comparison_regime.csv")
    print("\nFigures saved to:")
    print("  - figures/portfolio_value_regime.png")
    print("  - figures/drawdown_regime.png")
    print("  - figures/allocations_regime.png")
    print("  - figures/regime_distribution.png")
    print("  - figures/regime_timeline.png")
    
    # Option 2: Compare all models (optional, takes longer)
    print("\n\nWould you like to compare all models? (This will take longer)")
    print("To run comparison, execute: compare_all_models(config)")
    
    return allocations, regimes, comparison


if __name__ == "__main__":
    main()
