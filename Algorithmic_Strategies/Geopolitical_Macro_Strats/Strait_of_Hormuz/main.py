"""
Main execution script for Strait of Hormuz geopolitical strategy.
"""

import pandas as pd
import numpy as np
import yaml
from datetime import datetime
from data_acquisition import DataAcquisition
from shipping_monitor import ShippingMonitor
from geopolitical_scorer import GeopoliticalScorer
from signal_generator import SignalGenerator
from portfolio_constructor import PortfolioConstructor


def main():
    """Execute full strategy pipeline."""
    
    print("\n" + "="*70)
    print("STRAIT OF HORMUZ GEOPOLITICAL RISK STRATEGY")
    print("Multi-Asset, Cross-Regional Trading System")
    print("="*70 + "\n")
    
    # Load configuration
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    print(f"Strategy: {config['strategy']['name']}")
    print(f"Version: {config['strategy']['version']}")
    print(f"Period: {config['data']['start_date']} to {config['data']['end_date']}\n")
    
    # Step 1: Data Acquisition
    print("STEP 1: Data Acquisition")
    print("-" * 70)
    acq = DataAcquisition()
    data = acq.fetch_all_data()
    
    # Step 2: Shipping Traffic Analysis
    print("\nSTEP 2: Shipping Traffic Analysis")
    print("-" * 70)
    monitor = ShippingMonitor()
    shipping_analyzed = monitor.detect_anomalies(data['shipping'])
    status = monitor.get_current_status(data['shipping'])
    
    print(f"Current Status:")
    print(f"  Alert Level: {status['alert_level'].upper()}")
    print(f"  Current Transits: {status['current_transits']:.1f}")
    print(f"  Baseline: {status['baseline_transits']:.1f}")
    print(f"  Reduction: {status['reduction_pct']:.1f}%")
    
    crisis_periods = monitor.analyze_crisis_periods(data['shipping'])
    if len(crisis_periods) > 0:
        print(f"\nIdentified {len(crisis_periods)} crisis periods")
        print(f"  Total crisis days: {crisis_periods['duration_days'].sum()}")
        print(f"  Average duration: {crisis_periods['duration_days'].mean():.1f} days")
    
    # Step 3: Geopolitical Risk Scoring
    print("\nSTEP 3: Geopolitical Risk Analysis")
    print("-" * 70)
    scorer = GeopoliticalScorer()
    geo_analyzed = scorer.classify_risk_level(data['geopolitical'])
    assessment = scorer.get_current_assessment(data['geopolitical'])
    
    print(f"Current Assessment:")
    print(f"  Risk Score: {assessment['composite_risk_score']:.1f}/100")
    print(f"  Risk Level: {assessment['risk_level'].upper()}")
    print(f"  News Sentiment: {assessment['news_sentiment']:.2f}")
    print(f"  Conflict Events: {assessment['conflict_events']}")
    
    # Step 4: Signal Generation
    print("\nSTEP 4: Signal Generation")
    print("-" * 70)
    generator = SignalGenerator()
    master_signal = generator.generate_master_signal(data)
    current_signals = generator.get_current_signals(data)
    
    print(f"Master Signal:")
    print(f"  Composite Score: {current_signals['composite_signal']:.1f}/100")
    print(f"  Risk Level: {current_signals['risk_level'].upper()}")
    print(f"  Position Multiplier: {current_signals['position_multiplier']:.2f}")
    
    print(f"\nAsset Signals:")
    for asset, signal in current_signals['asset_signals'].items():
        direction = "LONG" if signal > 0 else "SHORT" if signal < 0 else "NEUTRAL"
        print(f"  {asset:20s}: {signal:+.3f} ({direction})")
    
    # Step 5: Portfolio Construction
    print("\nSTEP 5: Portfolio Construction")
    print("-" * 70)
    constructor = PortfolioConstructor()
    weights = constructor.construct_portfolio(master_signal, data['market'])
    
    current_weights = weights.iloc[-1]
    print(f"Current Portfolio Allocation:")
    for position, weight in current_weights.items():
        if abs(weight) > 0.001:
            print(f"  {position:20s}: {weight:+.1%}")
    
    # Step 6: Performance Summary
    print("\nSTEP 6: Historical Performance Analysis")
    print("-" * 70)
    
    # Calculate signal statistics
    signal_stats = master_signal['risk_level'].value_counts()
    total_days = len(master_signal)
    
    print(f"Signal Distribution ({total_days} days):")
    for level in ['low', 'medium', 'high', 'critical']:
        count = signal_stats.get(level, 0)
        pct = count / total_days * 100
        print(f"  {level.capitalize():10s}: {count:4d} days ({pct:5.1f}%)")
    
    # Risk transitions
    risk_changes = (master_signal['risk_level'] != master_signal['risk_level'].shift()).sum()
    print(f"\nRisk Level Transitions: {risk_changes}")
    print(f"Average holding period: {total_days / risk_changes:.1f} days")
    
    # Crisis event summary
    print("\nHistorical Crisis Events:")
    for event in config['backtest']['crisis_events']:
        print(f"  • {event['name']}")
        print(f"    {event['start']} to {event['end']}")
    
    # Step 7: Export Results
    print("\nSTEP 7: Export Results")
    print("-" * 70)
    
    # Save signals
    output_signals = master_signal.copy()
    output_signals['shipping_alert'] = shipping_analyzed['alert_level']
    output_signals['geo_risk_level'] = geo_analyzed['risk_level']
    
    output_file = f"signals_{datetime.now().strftime('%Y%m%d')}.csv"
    output_signals.to_csv(output_file)
    print(f"✓ Signals saved to: {output_file}")
    
    # Save weights
    weights_file = f"portfolio_weights_{datetime.now().strftime('%Y%m%d')}.csv"
    weights.to_csv(weights_file)
    print(f"✓ Portfolio weights saved to: {weights_file}")
    
    # Summary report
    print("\n" + "="*70)
    print("EXECUTION COMPLETE")
    print("="*70)
    print(f"\nStrategy is currently: {current_signals['risk_level'].upper()}")
    print(f"Position sizing: {current_signals['position_multiplier']:.0%} of maximum")
    print(f"\nRecommended Actions:")
    
    if current_signals['risk_level'] in ['high', 'critical']:
        print("  ⚠ HIGH RISK DETECTED")
        print("  → Increase long energy and defense positions")
        print("  → Add Treasury hedges")
        print("  → Short Asian equities and transportation")
    elif current_signals['risk_level'] == 'medium':
        print("  ⚡ MODERATE RISK")
        print("  → Maintain partial positions")
        print("  → Monitor shipping traffic closely")
    else:
        print("  ✓ LOW RISK")
        print("  → Minimal positions")
        print("  → Stay alert for risk escalation")
    
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    main()
