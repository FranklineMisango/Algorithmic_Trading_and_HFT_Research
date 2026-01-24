"""
Main Execution Script for Volts Volatility Strategy

This script orchestrates the complete pipeline:
1. Data acquisition
2. Volatility estimation
3. Clustering
4. Granger causality testing
5. Signal generation
6. Backtesting

Usage:
    python main.py --config config.yaml
"""

import argparse
import yaml
import pandas as pd
import yfinance as yf
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from volatility_estimators import calculate_volatility_for_assets
from volatility_clustering import cluster_assets_by_volatility
from granger_causality import identify_trading_pairs
from signal_generator import SignalGenerator, SignalAnalyzer
from backtester import VoltBacktester


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def download_data(tickers: list, start_date: str, end_date: str) -> dict:
    """
    Download price data for all tickers.
    
    Parameters:
    -----------
    tickers : list
        List of ticker symbols
    start_date : str
        Start date (YYYY-MM-DD)
    end_date : str
        End date (YYYY-MM-DD)
        
    Returns:
    --------
    dict : Dictionary of {ticker: DataFrame}
    """
    print("\n" + "="*80)
    print("STEP 1: DATA ACQUISITION")
    print("="*80)
    print(f"Downloading data for {len(tickers)} tickers...")
    print(f"Period: {start_date} to {end_date}")
    
    data_dict = {}
    for ticker in tickers:
        try:
            df = yf.download(ticker, start=start_date, end=end_date, progress=False)
            if len(df) > 0:
                data_dict[ticker] = df
                print(f"  ✓ {ticker}: {len(df)} days")
            else:
                print(f"  ✗ {ticker}: No data available")
        except Exception as e:
            print(f"  ✗ {ticker}: Error - {e}")
    
    print(f"\nSuccessfully downloaded data for {len(data_dict)} assets")
    return data_dict


def estimate_volatility(data_dict: dict, config: dict) -> pd.DataFrame:
    """
    Calculate historical volatility for all assets.
    
    Parameters:
    -----------
    data_dict : dict
        Price data dictionary
    config : dict
        Configuration dictionary
        
    Returns:
    --------
    pd.DataFrame : Volatility time series
    """
    print("\n" + "="*80)
    print("STEP 2: VOLATILITY ESTIMATION")
    print("="*80)
    
    vol_config = config['volatility']
    print(f"Using {vol_config['primary_estimator']} estimator")
    print(f"Rolling window: {vol_config['rolling_window']} days")
    
    volatility_df = calculate_volatility_for_assets(
        data_dict,
        estimator=vol_config['primary_estimator'],
        rolling_window=vol_config['rolling_window'],
        annualization_factor=vol_config['annualization_factor']
    )
    
    # Remove NaN values
    volatility_df = volatility_df.dropna()
    
    print(f"\nVolatility calculated for period: {volatility_df.index[0]} to {volatility_df.index[-1]}")
    print(f"Data points: {len(volatility_df)}")
    print("\nMean volatility by asset:")
    print(volatility_df.mean().sort_values(ascending=False).to_string())
    
    return volatility_df


def cluster_volatility(volatility_df: pd.DataFrame, config: dict) -> tuple:
    """
    Cluster assets by volatility.
    
    Parameters:
    -----------
    volatility_df : pd.DataFrame
        Volatility time series
    config : dict
        Configuration dictionary
        
    Returns:
    --------
    tuple : (clustering object, mid-cluster members)
    """
    print("\n" + "="*80)
    print("STEP 3: VOLATILITY CLUSTERING")
    print("="*80)
    
    cluster_config = config['clustering']
    print(f"Number of clusters: {cluster_config['n_clusters']}")
    print(f"Target cluster: Mid-volatility (cluster {cluster_config['target_cluster']})")
    
    clustering, mid_cluster_members = cluster_assets_by_volatility(
        volatility_df,
        n_clusters=cluster_config['n_clusters'],
        random_state=cluster_config['random_state'],
        target_cluster='mid'
    )
    
    # Save plot
    output_dir = Path(config['output']['results_dir'])
    output_dir.mkdir(exist_ok=True)
    
    if config['output']['save_plots']:
        clustering.plot_clusters(
            volatility_df,
            save_path=str(output_dir / 'volatility_clusters.png')
        )
        clustering.plot_time_series_by_cluster(
            volatility_df,
            save_path=str(output_dir / 'volatility_timeseries.png')
        )
    
    return clustering, mid_cluster_members


def identify_pairs(volatility_df: pd.DataFrame, mid_cluster: list, config: dict) -> tuple:
    """
    Identify trading pairs using Granger causality.
    
    Parameters:
    -----------
    volatility_df : pd.DataFrame
        Volatility time series
    mid_cluster : list
        Mid-cluster member tickers
    config : dict
        Configuration dictionary
        
    Returns:
    --------
    tuple : (trading_pairs DataFrame, analyzer object)
    """
    print("\n" + "="*80)
    print("STEP 4: GRANGER CAUSALITY ANALYSIS")
    print("="*80)
    
    granger_config = config['granger']
    print(f"Testing lags: {granger_config['min_lag']} to {granger_config['max_lag']}")
    print(f"Optimal lag: {granger_config['optimal_lag']}")
    print(f"Significance level: {granger_config['alpha']}")
    
    trading_pairs, analyzer = identify_trading_pairs(
        volatility_df,
        mid_cluster,
        target_lag=granger_config['optimal_lag'],
        significance_level=granger_config['alpha'],
        max_lag=granger_config['max_lag'],
        remove_circular=True
    )
    
    if len(trading_pairs) == 0:
        print("\n⚠️  WARNING: No significant Granger causality relationships found!")
        print("Consider:")
        print("  - Adjusting the significance level")
        print("  - Testing different lag ranges")
        print("  - Using a longer historical period")
        return None, None
    
    # Save plot
    output_dir = Path(config['output']['results_dir'])
    if config['output']['save_plots']:
        analyzer.plot_causality_network(
            trading_pairs,
            save_path=str(output_dir / 'granger_network.png')
        )
    
    return trading_pairs, analyzer


def generate_signals(volatility_df: pd.DataFrame, trading_pairs: pd.DataFrame, 
                     data_dict: dict, config: dict) -> dict:
    """
    Generate trading signals.
    
    Parameters:
    -----------
    volatility_df : pd.DataFrame
        Volatility time series
    trading_pairs : pd.DataFrame
        Identified trading pairs
    data_dict : dict
        Price data
    config : dict
        Configuration dictionary
        
    Returns:
    --------
    dict : Trading signals for all pairs
    """
    print("\n" + "="*80)
    print("STEP 5: SIGNAL GENERATION")
    print("="*80)
    
    strategy_config = config['strategy']
    print(f"Strategy type: {strategy_config['type']}")
    print(f"Trend method: {strategy_config['trend_method']}")
    print(f"Trend parameters: {strategy_config['trend_params']}")
    
    signal_gen = SignalGenerator(
        trend_method=strategy_config['trend_method'],
        trend_params=strategy_config['trend_params']
    )
    
    signals = signal_gen.generate_signals_for_all_pairs(volatility_df, trading_pairs)
    
    # Analyze signals
    stats = SignalAnalyzer.get_signal_statistics(signals)
    print("\nSignal Statistics:")
    print(stats.to_string(index=False))
    
    # Plot signals for each pair
    output_dir = Path(config['output']['results_dir'])
    if config['output']['save_plots']:
        for pair_name, signals_df in signals.items():
            target_ticker = pair_name.split('->')[1]
            target_price = data_dict[target_ticker]['Close']
            
            SignalAnalyzer.plot_signals(
                signals_df,
                price_data=target_price,
                title=pair_name,
                save_path=str(output_dir / f'signals_{pair_name.replace("->", "_")}.png')
            )
    
    return signals


def run_backtest(price_data: dict, signals: dict, config: dict) -> dict:
    """
    Run backtest on the strategy.
    
    Parameters:
    -----------
    price_data : dict
        Price data dictionary
    signals : dict
        Trading signals
    config : dict
        Configuration dictionary
        
    Returns:
    --------
    dict : Backtest results
    """
    print("\n" + "="*80)
    print("STEP 6: BACKTESTING")
    print("="*80)
    
    backtest_config = config['backtest']
    strategy_config = config['strategy']
    
    print(f"Backtest period: {backtest_config['start_date']} to {backtest_config['end_date']}")
    print(f"Initial capital per pair: ${strategy_config['initial_capital']}")
    print(f"Commission: ${strategy_config['commission']} per trade")
    
    backtester = VoltBacktester(
        initial_capital_per_pair=strategy_config['initial_capital'],
        commission=strategy_config['commission'],
        slippage=strategy_config['slippage'],
        position_size_pct=strategy_config['position_size_pct']
    )
    
    results = backtester.run_backtest(
        price_data,
        signals,
        start_date=pd.Timestamp(backtest_config['start_date']),
        end_date=pd.Timestamp(backtest_config['end_date'])
    )
    
    # Print results
    backtester.print_results()
    
    # Plot results
    output_dir = Path(config['output']['results_dir'])
    if config['output']['save_plots']:
        backtester.plot_results(
            save_path=str(output_dir / 'backtest_results.png')
        )
    
    return results


def save_results(results: dict, trading_pairs: pd.DataFrame, config: dict) -> None:
    """
    Save results to files.
    
    Parameters:
    -----------
    results : dict
        Backtest results
    trading_pairs : pd.DataFrame
        Trading pairs
    config : dict
        Configuration dictionary
    """
    output_dir = Path(config['output']['results_dir'])
    output_dir.mkdir(exist_ok=True)
    
    # Save trading pairs
    trading_pairs.to_csv(output_dir / 'trading_pairs.csv', index=False)
    
    # Save aggregated metrics
    agg_metrics = pd.DataFrame([results['aggregated']['metrics']])
    agg_metrics.to_csv(output_dir / 'aggregated_metrics.csv', index=False)
    
    # Save per-pair metrics
    pair_metrics = []
    for pair_name, result in results['pair_results'].items():
        metrics = result['metrics'].copy()
        metrics['pair'] = pair_name
        pair_metrics.append(metrics)
    
    pair_metrics_df = pd.DataFrame(pair_metrics)
    pair_metrics_df.to_csv(output_dir / 'pair_metrics.csv', index=False)
    
    # Save equity curve
    equity_df = pd.DataFrame({
        'date': results['aggregated']['dates'],
        'equity': results['aggregated']['equity_curve']
    })
    equity_df.to_csv(output_dir / 'equity_curve.csv', index=False)
    
    # Save all trades
    all_trades = []
    for trade in results['aggregated']['all_trades']:
        all_trades.append({
            'pair': trade.pair_name,
            'entry_date': trade.entry_date,
            'exit_date': trade.exit_date,
            'entry_price': trade.entry_price,
            'exit_price': trade.exit_price,
            'shares': trade.shares,
            'direction': trade.direction,
            'pnl': trade.pnl,
            'return_pct': trade.return_pct
        })
    
    trades_df = pd.DataFrame(all_trades)
    trades_df.to_csv(output_dir / 'all_trades.csv', index=False)
    
    print(f"\n✓ Results saved to {output_dir}")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description='Volts Volatility Strategy')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to configuration file')
    args = parser.parse_args()
    
    # Load configuration
    print("Loading configuration...")
    config = load_config(args.config)
    
    # Create output directory
    output_dir = Path(config['output']['results_dir'])
    output_dir.mkdir(exist_ok=True)
    
    print("\n" + "="*80)
    print("VOLTS VOLATILITY-BASED PREDICTIVE TRADING STRATEGY")
    print("="*80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # Step 1: Download data
        data_dict = download_data(
            config['data']['assets'],
            config['data']['start_date'],
            config['data']['end_date']
        )
        
        if len(data_dict) < 3:
            raise ValueError("Insufficient data downloaded. Need at least 3 assets.")
        
        # Step 2: Estimate volatility
        volatility_df = estimate_volatility(data_dict, config)
        
        # Step 3: Cluster by volatility
        clustering, mid_cluster = cluster_volatility(volatility_df, config)
        
        if len(mid_cluster) < 2:
            raise ValueError("Insufficient assets in mid-volatility cluster.")
        
        # Step 4: Identify trading pairs
        trading_pairs, analyzer = identify_pairs(volatility_df, mid_cluster, config)
        
        if trading_pairs is None or len(trading_pairs) == 0:
            print("\n⚠️  Cannot proceed without trading pairs.")
            return
        
        # Step 5: Generate signals
        signals = generate_signals(volatility_df, trading_pairs, data_dict, config)
        
        # Step 6: Run backtest
        results = run_backtest(data_dict, signals, config)
        
        # Save results
        save_results(results, trading_pairs, config)
        
        print("\n" + "="*80)
        print("EXECUTION COMPLETED SUCCESSFULLY")
        print("="*80)
        print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return


if __name__ == "__main__":
    main()
