#!/usr/bin/env python3
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from sentinelsat import SentinelAPI
from container_detector import ContainerDetector, analyze_container_trends
from signal_generator import TradingSignalGenerator, generate_trading_report
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

def main():
    print("=" * 60)
    print("SATELLITE CONTAINER ANALYSIS - TRADING SIGNAL GENERATOR")
    print("=" * 60)
    
    # Load config
    with open('config.json', 'r') as f:
        config = json.load(f)
    
    # Initialize Copernicus API
    print("\n[1/6] Connecting to ESA Copernicus API...")
    api = SentinelAPI(
        config['ESACopernicus']['Username'],
        config['ESACopernicus']['Password'],
        'https://apihub.copernicus.eu/apihub'
    )
    print(f"✓ Connected as {config['ESACopernicus']['Username']}")
    
    ports = config['ports']
    print(f"✓ Loaded {len(ports)} major ports: {', '.join([p['name'] for p in ports])}")
    
    # Search for satellite imagery
    print("\n[2/6] Searching for satellite imagery...")
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    
    all_products = {}
    for port in ports:
        print(f"  Searching {port['name']}...", end=" ")
        bbox = f"POLYGON(({port['lon']-0.1} {port['lat']-0.1},{port['lon']+0.1} {port['lat']-0.1},{port['lon']+0.1} {port['lat']+0.1},{port['lon']-0.1} {port['lat']+0.1},{port['lon']-0.1} {port['lat']-0.1}))"
        
        try:
            products = api.query(
                bbox,
                date=(start_date.strftime('%Y%m%d'), end_date.strftime('%Y%m%d')),
                platformname='Sentinel-2',
                cloudcoverpercentage=(0, 20)
            )
            all_products[port['name']] = products
            print(f"✓ Found {len(products)} images")
        except Exception as e:
            print(f"✗ Error: {str(e)[:50]}")
            all_products[port['name']] = {}
    
    # Generate synthetic container data (replace with actual detection)
    print("\n[3/6] Generating container count data...")
    print("  (Using synthetic data - replace with actual image analysis)")
    
    np.random.seed(42)
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    container_data = []
    
    for port in ports:
        base_count = np.random.randint(1000, 5000)
        for i, date in enumerate(dates):
            trend = np.sin(i * 0.1) * 500
            noise = np.random.normal(0, 200)
            count = max(0, int(base_count + trend + noise))
            
            container_data.append({
                'port': port['name'],
                'datetime': date,
                'container_count': count,
                'pct_change': 0,
                'avg_confidence': 0.85
            })
    
    df_containers = pd.DataFrame(container_data)
    df_containers = df_containers.sort_values(['port', 'datetime'])
    df_containers['pct_change'] = df_containers.groupby('port')['container_count'].pct_change()
    
    print(f"✓ Generated {len(df_containers)} observations across {len(ports)} ports")
    
    # Analyze trends
    print("\n[4/6] Analyzing container trends...")
    df_analyzed = analyze_container_trends(df_containers)
    print(f"✓ Calculated moving averages and trend indicators")
    
    # Generate trading signals
    print("\n[5/6] Generating trading signals...")
    signal_gen = TradingSignalGenerator()
    port_signals = signal_gen.calculate_port_signals(df_analyzed)
    global_signals = signal_gen.generate_global_signal(port_signals)
    
    print(f"✓ Generated signals for {len(port_signals)} port-days")
    print(f"✓ Generated {len(global_signals)} global signals")
    
    # Backtest
    print("\n[6/6] Running backtest...")
    backtest_results = signal_gen.backtest_signals(global_signals)
    report = generate_trading_report(global_signals, backtest_results)
    
    # Display results
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"\nSignal Statistics:")
    print(f"  Total signals: {report['summary']['total_signals']}")
    print(f"  Bullish signals: {report['summary']['bullish_signals']}")
    print(f"  Bearish signals: {report['summary']['bearish_signals']}")
    print(f"  Warning days: {report['summary']['warning_days']}")
    print(f"  Date range: {report['summary']['date_range']}")
    
    print(f"\nBacktest Performance:")
    print(f"  Strategy return: {report['performance']['total_return']:.2%}")
    print(f"  Market return: {report['performance']['market_return']:.2%}")
    print(f"  Excess return: {report['performance']['excess_return']:.2%}")
    print(f"  Sharpe ratio: {report['performance']['sharpe_ratio']:.2f}")
    
    # Save results
    output_dir = Path('data/results')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    df_containers.to_csv(output_dir / 'container_counts.csv', index=False)
    port_signals.to_csv(output_dir / 'port_signals.csv', index=False)
    global_signals.to_csv(output_dir / 'global_signals.csv', index=False)
    backtest_results['backtest_data'].to_csv(output_dir / 'backtest_results.csv', index=False)
    
    print(f"\n✓ Results saved to {output_dir}/")
    
    # Visualization
    print("\n[Generating visualizations...]")
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Container counts by port
    for port in ports[:3]:
        port_data = df_containers[df_containers['port'] == port['name']]
        axes[0,0].plot(port_data['datetime'], port_data['container_count'], label=port['name'])
    axes[0,0].set_title('Container Counts by Port')
    axes[0,0].legend()
    axes[0,0].tick_params(axis='x', rotation=45)
    
    # Global signals
    axes[0,1].plot(global_signals['datetime'], global_signals['global_signal'], marker='o')
    axes[0,1].set_title('Global Trading Signal')
    axes[0,1].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    axes[0,1].tick_params(axis='x', rotation=45)
    
    # Cumulative returns
    axes[1,0].plot(backtest_results['backtest_data']['datetime'], 
                   backtest_results['backtest_data']['cumulative_return'], 
                   label='Strategy')
    axes[1,0].plot(backtest_results['backtest_data']['datetime'], 
                   backtest_results['backtest_data']['market_cumulative'], 
                   label='Market', alpha=0.7)
    axes[1,0].set_title('Cumulative Returns')
    axes[1,0].legend()
    axes[1,0].tick_params(axis='x', rotation=45)
    
    # Signal distribution
    signal_counts = global_signals['global_signal'].value_counts().sort_index()
    axes[1,1].bar(signal_counts.index, signal_counts.values)
    axes[1,1].set_title('Signal Distribution')
    axes[1,1].set_xlabel('Signal Type')
    axes[1,1].set_ylabel('Count')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'analysis_results.png', dpi=150)
    print(f"✓ Visualization saved to {output_dir}/analysis_results.png")
    
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    main()
