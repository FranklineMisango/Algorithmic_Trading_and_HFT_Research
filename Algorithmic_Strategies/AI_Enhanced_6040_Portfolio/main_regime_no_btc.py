"""
Main script for Regime-Based ML Strategy - NO BITCOIN VERSION

This version:
1. Removes Bitcoin to test back to 2000 (25 years)
2. Eliminates survivorship bias from crypto
3. Tests on complete market cycles (dot-com, 2008, COVID)
"""

import yaml
import pandas as pd
import numpy as np
from typing import Dict, Tuple
from data_acquisition import DataAcquisition
from regime_ml_strategy_no_btc import RegimeMLStrategyNoBTC
from backtester import PortfolioBacktester
import matplotlib.pyplot as plt
import seaborn as sns


def walk_forward_no_btc(config: Dict,
                        train_window: int = 60,
                        test_window: int = 12,
                        model_name: str = 'ensemble',
                        use_dynamic_optimization: bool = True) -> Tuple:
    """
    Walk-forward optimization - NO BITCOIN VERSION.
    """
    print("\n" + "="*70)
    print("REGIME-BASED ML STRATEGY - NO BITCOIN VERSION")
    print("="*70)
    print("\nKey Changes:")
    print("  1. NO BITCOIN - eliminates survivorship bias")
    print("  2. Extended data back to 2000 (25 years)")
    print("  3. Includes dot-com crash, 2008 crisis, COVID")
    print("  4. Traditional assets only (SPY, TLT, GLD, SHY, IEF)")
    
    # Fetch data
    print("\n1. Fetching data...")
    data_acq = DataAcquisition(config)
    prices, returns, indicators = data_acq.get_full_dataset()
    
    print(f"\nData period: {prices.index[0]} to {prices.index[-1]}")
    print(f"Total months: {len(prices)}")
    
    # Initialize strategy
    strategy = RegimeMLStrategyNoBTC(config)
    
    # Calculate number of windows
    total_periods = len(returns)
    n_windows = (total_periods - train_window) // test_window
    
    print(f"\nTraining window: {train_window} months")
    print(f"Test window: {test_window} months")
    print(f"Number of windows: {n_windows}")
    print(f"Model: {model_name}")
    print(f"Dynamic optimization: {use_dynamic_optimization}")
    
    all_allocations = []
    all_regimes = []
    
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
        
        # Create regime labels
        print("\nCreating regime labels...")
        train_regimes = strategy.create_regime_labels(
            train_indicators['VIX'],
            train_prices['SPY'],
            train_indicators['Yield_Spread'],
            train_returns['SPY'],
            use_forward_looking=True
        )
        
        print(f"Training regime distribution: {train_regimes.value_counts().to_dict()}")
        
        # Create features
        print("Engineering features...")
        train_features = strategy.create_features(train_indicators, train_prices, train_returns)
        
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
        
        valid_idx = ~y_train.isna()
        X_train = X_train[valid_idx]
        y_train = y_train[valid_idx]
        
        print(f"\nTraining samples: {len(X_train)}")
        
        # Train models
        strategy.train_models(X_train, y_train)
        
        # Predict regimes
        print(f"\nPredicting regimes on {len(test_features)} samples...")
        predicted_regimes = strategy.predict_regime(test_features, model_name=model_name)
        
        print(f"Predicted regime distribution: {predicted_regimes.value_counts().to_dict()}")
        
        # Convert to allocations
        if use_dynamic_optimization:
            print("Optimizing allocations within regimes...")
            allocations = strategy.get_allocations_for_regime(
                predicted_regimes,
                returns=returns.iloc[:end_test],
                use_dynamic_optimization=True
            )
        else:
            allocations = strategy.get_allocations_for_regime(
                predicted_regimes,
                use_dynamic_optimization=False
            )
        
        all_allocations.append(allocations)
        all_regimes.append(predicted_regimes)
    
    # Combine results
    final_allocations = pd.concat(all_allocations)
    final_regimes = pd.concat(all_regimes)
    
    print(f"\n{'='*70}")
    print("Walk-Forward Optimization Complete!")
    print(f"Total allocation periods: {len(final_allocations)}")
    print(f"Overall regime distribution:")
    for regime, count in final_regimes.value_counts().sort_index().items():
        regime_name = ['Defensive', 'Neutral', 'Aggressive'][regime]
        pct = count / len(final_regimes) * 100
        print(f"  {regime} ({regime_name}): {count} months ({pct:.1f}%)")
    print(f"{'='*70}")
    
    return final_allocations, final_regimes, prices, returns


def main():
    """Main execution function."""
    print("\n" + "="*70)
    print("AI-ENHANCED PORTFOLIO - NO BITCOIN VERSION")
    print("="*70)
    
    # Load configuration
    with open('config_no_btc.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Run strategy
    print("\nRunning NO BITCOIN Strategy...")
    allocations, regimes, prices, returns = walk_forward_no_btc(
        config,
        train_window=60,
        test_window=12,
        model_name='ensemble',
        use_dynamic_optimization=True
    )
    
    # Backtest
    print("\n\n2. Running Backtest...")
    backtester = PortfolioBacktester(config)
    
    # Align data
    common_idx = allocations.index.intersection(returns.index)
    allocations_aligned = allocations.loc[common_idx]
    returns_aligned = returns.loc[common_idx]
    prices_aligned = prices.loc[common_idx]
    
    # Backtest strategies
    ai_results = backtester.backtest_strategy(
        allocations_aligned,
        returns_aligned,
        prices_aligned,
        use_dynamic_rebalancing=True,
        use_stop_loss=True
    )
    
    spy_results = backtester.create_benchmark_strategy(returns_aligned, prices_aligned)
    traditional_results = backtester.create_traditional_6040(returns_aligned, prices_aligned)
    
    # Compare
    print("\n\n3. Comparing Strategies...")
    strategies = {
        'Regime AI (No BTC)': ai_results,
        'Buy & Hold SPY': spy_results,
        'Traditional 60/40': traditional_results
    }
    
    comparison = backtester.compare_strategies(strategies)
    
    print("\n" + "="*70)
    print("PERFORMANCE COMPARISON - NO BITCOIN")
    print("="*70)
    print(comparison[['Sharpe Ratio', 'CAGR', 'Max Drawdown', 'Volatility', 'Total Return']])
    
    # Statistical tests
    print("\n\n4. Statistical Tests...")
    tests = backtester.perform_statistical_tests(ai_results, spy_results)
    print("\nAI Strategy (No BTC) vs SPY:")
    print(f"  Mean return difference: {tests['returns_t_test']['difference']:.4f}")
    print(f"  P-value: {tests['returns_t_test']['p_value']:.4f}")
    print(f"  Statistically significant: {tests['returns_t_test']['significant']}")
    print(f"  Sharpe ratio difference: {tests['sharpe_ratio_comparison']['difference']:.4f}")
    
    # Save results
    print("\n\n5. Saving Results...")
    allocations_aligned.to_csv('results/allocations_no_btc.csv')
    regimes.to_csv('results/regimes_no_btc.csv')
    comparison.to_csv('results/performance_no_btc.csv')
    
    # Create visualizations
    print("\n6. Creating Visualizations...")
    
    # Portfolio value
    fig = backtester.plot_portfolio_value(strategies)
    fig.savefig('figures/portfolio_value_no_btc.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Drawdown
    fig = backtester.plot_drawdown(strategies)
    fig.savefig('figures/drawdown_no_btc.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Allocations
    fig = backtester.plot_allocations(allocations_aligned)
    fig.savefig('figures/allocations_no_btc.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Regime distribution
    plt.figure(figsize=(12, 6))
    regime_counts = regimes.value_counts().sort_index()
    regime_labels = ['Defensive', 'Neutral', 'Aggressive']
    plt.bar(range(len(regime_counts)), regime_counts.values, color=['red', 'yellow', 'green'])
    plt.xticks(range(len(regime_counts)), [regime_labels[i] for i in regime_counts.index])
    plt.title('Regime Distribution (No Bitcoin)', fontsize=14, fontweight='bold')
    plt.xlabel('Regime')
    plt.ylabel('Count (months)')
    plt.tight_layout()
    plt.savefig('figures/regime_distribution_no_btc.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Regime timeline
    plt.figure(figsize=(16, 6))
    colors = ['red', 'yellow', 'green']
    for i in range(len(regimes)):
        plt.axvspan(regimes.index[i], regimes.index[min(i+1, len(regimes)-1)], 
                   alpha=0.3, color=colors[regimes.iloc[i]])
    plt.plot(regimes.index, regimes.values, marker='o', linestyle='-', markersize=2, color='black')
    plt.title('Predicted Regimes Over Time (No Bitcoin)', fontsize=14, fontweight='bold')
    plt.xlabel('Date')
    plt.ylabel('Regime')
    plt.yticks([0, 1, 2], ['Defensive', 'Neutral', 'Aggressive'])
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('figures/regime_timeline_no_btc.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Comparison chart
    fig, ax = plt.subplots(figsize=(12, 6))
    comparison_metrics = comparison[['Sharpe Ratio', 'CAGR', 'Sortino Ratio', 'Calmar Ratio']].T
    comparison_metrics.plot(kind='bar', ax=ax)
    plt.title('Strategy Comparison (No Bitcoin)', fontsize=14, fontweight='bold')
    plt.xlabel('Metric')
    plt.ylabel('Value')
    plt.legend(title='Strategy')
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig('figures/strategy_comparison_no_btc.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("\n" + "="*70)
    print("NO BITCOIN STRATEGY COMPLETE!")
    print("="*70)
    print("\nResults saved to:")
    print("  - results/allocations_no_btc.csv")
    print("  - results/regimes_no_btc.csv")
    print("  - results/performance_no_btc.csv")
    print("\nFigures saved to:")
    print("  - figures/portfolio_value_no_btc.png")
    print("  - figures/drawdown_no_btc.png")
    print("  - figures/allocations_no_btc.png")
    print("  - figures/regime_distribution_no_btc.png")
    print("  - figures/regime_timeline_no_btc.png")
    print("  - figures/strategy_comparison_no_btc.png")
    
    # Analysis
    print("\n" + "="*70)
    print("KEY BENEFITS OF NO BITCOIN VERSION")
    print("="*70)
    print("\n1. Eliminates Survivorship Bias:")
    print("   - No crypto selection bias")
    print("   - Traditional assets only")
    print("   - More realistic for institutional investors")
    
    print("\n2. Extended Historical Data:")
    print(f"   - Test period: {prices.index[0]} to {prices.index[-1]}")
    print(f"   - Total years: {(prices.index[-1] - prices.index[0]).days / 365:.1f}")
    print("   - Includes: Dot-com crash, 2008 crisis, COVID")
    
    print("\n3. More Stable Allocations:")
    print("   - No volatile crypto swings")
    print("   - Lower turnover expected")
    print("   - More suitable for conservative investors")
    
    return allocations, regimes, comparison


if __name__ == "__main__":
    main()
