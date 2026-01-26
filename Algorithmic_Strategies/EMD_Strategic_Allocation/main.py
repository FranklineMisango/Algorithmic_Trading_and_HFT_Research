#!/usr/bin/env python3
"""
Emerging Markets Debt Strategic Allocation Strategy
Main execution script for backtesting and analysis
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import yaml
from pathlib import Path
from signals import EMDSignals
from risk_manager import RiskManager

def load_config(config_path='config.yaml'):
    """Load strategy configuration"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def download_data(tickers, start_date, end_date):
    """Download historical price data"""
    print(f"Downloading data from {start_date} to {end_date}...")
    data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']
    return data.dropna()

def backtest_strategy(returns_df, config):
    """Backtest EMD strategic allocation"""
    emd_alloc = config['strategy']['allocation_pct']
    local_split, hard_split = config['strategy']['local_hard_split']
    rebalance_freq = 63  # Quarterly
    
    portfolio_value = [config['backtest']['initial_capital']]
    dates = returns_df.index
    
    weights = {
        'EMLC': emd_alloc * local_split,
        'EMB': emd_alloc * hard_split,
        'SPY': (1 - emd_alloc) * 0.6,
        'AGG': (1 - emd_alloc) * 0.4
    }
    
    last_rebalance = 0
    
    for i, date in enumerate(dates):
        # VIX trigger check
        vix_level = returns_df.loc[date, '^VIX'] if '^VIX' in returns_df.columns else 15
        
        current_weights = weights.copy()
        if vix_level > config['risk']['vix_reduction_trigger']:
            emd_alloc_adj = emd_alloc * 0.5
            current_weights = {
                'EMLC': emd_alloc_adj * local_split,
                'EMB': emd_alloc_adj * hard_split,
                'SPY': (1 - emd_alloc_adj) * 0.6,
                'AGG': (1 - emd_alloc_adj) * 0.4
            }
        
        # Calculate daily return
        daily_return = sum(current_weights.get(ticker, 0) * returns_df.loc[date, ticker] 
                          for ticker in current_weights.keys() if ticker in returns_df.columns)
        
        # Transaction costs on rebalance
        if i - last_rebalance >= rebalance_freq:
            tc = (current_weights['EMLC'] * config['costs']['local_currency_bps'] / 10000 +
                  current_weights['EMB'] * config['costs']['hard_currency_bps'] / 10000 +
                  (current_weights['SPY'] + current_weights['AGG']) * 5 / 10000)
            daily_return -= tc
            last_rebalance = i
        
        portfolio_value.append(portfolio_value[-1] * (1 + daily_return))
    
    portfolio_value = portfolio_value[1:]
    portfolio_returns = pd.Series(portfolio_value, index=dates).pct_change().dropna()
    
    return portfolio_returns, pd.Series(portfolio_value, index=dates)

def calculate_metrics(returns_series):
    """Calculate performance metrics"""
    total_return = (1 + returns_series).prod() - 1
    ann_return = returns_series.mean() * 252
    ann_vol = returns_series.std() * np.sqrt(252)
    sharpe = ann_return / ann_vol if ann_vol > 0 else 0
    
    cum_returns = (1 + returns_series).cumprod()
    running_max = cum_returns.expanding().max()
    drawdown = (cum_returns - running_max) / running_max
    max_dd = drawdown.min()
    
    downside_returns = returns_series[returns_series < 0]
    downside_vol = downside_returns.std() * np.sqrt(252)
    sortino = ann_return / downside_vol if downside_vol > 0 else 0
    
    return {
        'Total Return': total_return,
        'Annual Return': ann_return,
        'Annual Volatility': ann_vol,
        'Sharpe Ratio': sharpe,
        'Sortino Ratio': sortino,
        'Max Drawdown': max_dd
    }

def main():
    """Main execution function"""
    print("="*70)
    print("EMD STRATEGIC ALLOCATION STRATEGY")
    print("="*70)
    
    # Load configuration
    config = load_config()
    
    # Download data
    tickers = ['EMLC', 'EMB', 'SPY', 'AGG', '^VIX']
    data = download_data(tickers, config['backtest']['start_date'], config['backtest']['end_date'])
    
    print(f"\nData shape: {data.shape}")
    print(f"Date range: {data.index[0].date()} to {data.index[-1].date()}")
    
    # Calculate returns
    returns = data.pct_change().dropna()
    
    # Run backtest
    print("\nRunning backtest...")
    strategy_returns, strategy_value = backtest_strategy(returns, config)
    
    # Benchmark
    benchmark_returns = 0.6 * returns['SPY'] + 0.4 * returns['AGG']
    
    # Calculate metrics
    print("\n" + "="*70)
    print("PERFORMANCE METRICS")
    print("="*70)
    
    strategy_metrics = calculate_metrics(strategy_returns)
    benchmark_metrics = calculate_metrics(benchmark_returns)
    
    print("\nEMD Strategy:")
    for key, value in strategy_metrics.items():
        print(f"  {key:20s}: {value:8.2%}" if 'Return' in key or 'Volatility' in key or 'Drawdown' in key 
              else f"  {key:20s}: {value:8.3f}")
    
    print("\n60/40 Benchmark:")
    for key, value in benchmark_metrics.items():
        print(f"  {key:20s}: {value:8.2%}" if 'Return' in key or 'Volatility' in key or 'Drawdown' in key 
              else f"  {key:20s}: {value:8.3f}")
    
    # Check targets
    sharpe_improvement = strategy_metrics['Sharpe Ratio'] - benchmark_metrics['Sharpe Ratio']
    dd_increase = strategy_metrics['Max Drawdown'] - benchmark_metrics['Max Drawdown']
    
    print("\n" + "="*70)
    print("TARGET VALIDATION")
    print("="*70)
    print(f"Sharpe Improvement: {sharpe_improvement:6.3f} (Target: >0.15)")
    print(f"Max DD Increase:    {dd_increase:6.2%} (Target: <2%)")
    
    if sharpe_improvement > 0.15 and dd_increase < 0.02:
        print("\n✓ Strategy MEETS performance targets!")
    else:
        print("\n✗ Strategy does NOT meet performance targets.")
    
    print("="*70)
    
    # Save results
    results_df = pd.DataFrame({
        'Strategy_Value': strategy_value,
        'Strategy_Returns': strategy_returns,
        'Benchmark_Returns': benchmark_returns
    })
    results_df.to_csv('backtest_results.csv')
    print("\nResults saved to backtest_results.csv")

if __name__ == "__main__":
    main()
