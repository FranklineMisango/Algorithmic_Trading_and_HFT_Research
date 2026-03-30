"""
Merge congressional trades with OHLCV market data.
Creates aligned dataset for signal generation and backtesting.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm

def load_data(trades_csv, ohlcv_csv):
    """Load and parse datasets."""
    print(f"Loading trades from {trades_csv}...")
    trades = pd.read_csv(trades_csv)
    trades['filing_date'] = pd.to_datetime(trades['filing_date'])
    trades['transaction_date'] = pd.to_datetime(trades['transaction_date'])
    
    print(f"Loading OHLCV from {ohlcv_csv}...")
    ohlcv = pd.read_csv(ohlcv_csv)
    ohlcv['date'] = pd.to_datetime(ohlcv['date'], format='mixed')
    
    return trades, ohlcv

def merge_trades_to_ohlcv(trades, ohlcv, use_date_column='filing_date'):
    """
    Left-join trades to OHLCV on exact date match.
    
    Args:
        trades: Congressional trades DataFrame
        ohlcv: OHLCV data DataFrame
        use_date_column: 'filing_date' or 'transaction_date'
    
    Returns:
        Merged DataFrame with OHLCV columns appended
    """
    print(f"\n{'='*60}")
    print(f"MERGING TRADES TO OHLCV")
    print(f"{'='*60}")
    print(f"Using date column: {use_date_column}")
    print(f"Trades: {len(trades):,}")
    print(f"OHLCV unique symbols: {ohlcv['symbol'].nunique():,}")
    print(f"OHLCV date range: {ohlcv['date'].min().date()} to {ohlcv['date'].max().date()}")
    
    # Rename for merge
    merge_trades = trades.copy()
    merge_trades['trade_date'] = merge_trades[use_date_column]
    
    # Normalize dates to remove time component for consistent matching
    merge_trades['trade_date'] = merge_trades['trade_date'].dt.normalize()
    ohlcv_merge = ohlcv.copy()
    ohlcv_merge['date'] = ohlcv_merge['date'].dt.normalize()
    
    # Left join: each trade tries to match to market price on same day
    print(f"\nPerforming left join on (ticker, date)...")
    merged = merge_trades.merge(
            ohlcv_merge,
           left_on=['ticker', 'trade_date'],
           right_on=['symbol', 'date'],
        how='left'
    )
    
    # Track coverage
    has_ohlcv = merged['close'].notna()
    print(f"\nMerge results:")
    print(f"  Rows with OHLCV: {has_ohlcv.sum():,} ({100*has_ohlcv.sum()/len(merged):.1f}%)")
    print(f"  Rows without OHLCV: {(~has_ohlcv).sum():,} ({100*(~has_ohlcv).sum()/len(merged):.1f}%)")
    
    # Add coverage flag
    merged['has_ohlcv'] = has_ohlcv
    
    # Add basic derived columns for backtest-ready dataset
    merged['days_to_filing'] = (merged['trade_date'] - merged['filing_date']).dt.days
    
    return merged

def analyze_coverage(merged, ohlcv):
    """Analyze which tickers have/don't have OHLCV coverage."""
    print(f"\n{'='*60}")
    print(f"COVERAGE ANALYSIS")
    print(f"{'='*60}")
    
    trade_symbols = set(merged['ticker'].unique())
    ohlcv_symbols = set(ohlcv['symbol'].unique())
    
    covered = trade_symbols & ohlcv_symbols
    missing = trade_symbols - ohlcv_symbols
    
    print(f"Unique traded symbols: {len(trade_symbols):,}")
    print(f"Symbols with OHLCV: {len(covered):,}")
    print(f"Symbols missing OHLCV: {len(missing):,}")
    
    trades_covered = merged[merged['ticker'].isin(covered)]
    trades_missing = merged[merged['ticker'].isin(missing)]
    
    print(f"\nTrade impact:")
    print(f"  Trades on covered symbols: {len(trades_covered):,} ({100*len(trades_covered)/len(merged):.1f}%)")
    print(f"  Trades on missing symbols: {len(trades_missing):,} ({100*len(trades_missing)/len(merged):.1f}%)")
    
    if len(missing) > 0:
        print(f"\nSample missing symbols (no OHLCV available):")
        print(sorted(list(missing))[:20])
    
    return merged

def save_outputs(merged, output_csv, backtest_csv=None):
    """Save merged and filtered datasets."""
    print(f"\n{'='*60}")
    print(f"SAVING OUTPUTS")
    print(f"{'='*60}")
    
    # Full merged dataset (with missing values)
    print(f"Saving full merged dataset ({len(merged):,} rows) to {output_csv}...")
    merged.to_csv(output_csv, index=False)
    
    # Backtest-ready dataset (only rows with OHLCV)
    if backtest_csv:
        backtest_ready = merged[merged['has_ohlcv']].copy()
        print(f"Saving backtest-ready dataset ({len(backtest_ready):,} rows) to {backtest_csv}...")
        backtest_ready.to_csv(backtest_csv, index=False)
        
        print(f"\nBacktest dataset statistics:")
        print(f"  Date range: {backtest_ready['trade_date'].min().date()} to {backtest_ready['trade_date'].max().date()}")
        print(f"  Unique symbols: {backtest_ready['symbol'].nunique():,}")
        print(f"  Unique filing dates: {backtest_ready['filing_date'].nunique():,}")
        print(f"  Buy trades: {(backtest_ready['transaction_type'] == 'buy').sum():,}")
        print(f"  Sell trades: {(backtest_ready['transaction_type'] == 'sell').sum():,}")

def main():
    parser = argparse.ArgumentParser(description='Merge congressional trades with OHLCV data')
    parser.add_argument('--trades', default='data/quiver_congress_trades_roster.csv',
                        help='Path to congressional trades CSV')
    parser.add_argument('--ohlcv', default='data/ohlcv_yahoo_alpaca.csv',
                        help='Path to OHLCV data CSV')
    parser.add_argument('--output', default='data/trades_with_ohlcv.csv',
                        help='Output file for merged dataset (all rows)')
    parser.add_argument('--backtest-output', default='data/trades_with_ohlcv_backtest.csv',
                        help='Output file for backtest-ready dataset (only rows with OHLCV)')
    parser.add_argument('--date-column', default='filing_date',
                        choices=['filing_date', 'transaction_date'],
                        help='Which date column to use for OHLCV match')
    
    args = parser.parse_args()
    
    # Validate inputs
    for path in [args.trades, args.ohlcv]:
        if not Path(path).exists():
            print(f"ERROR: {path} not found")
            return
    
    # Load data
    trades, ohlcv = load_data(args.trades, args.ohlcv)
    
    # Merge
    merged = merge_trades_to_ohlcv(trades, ohlcv, use_date_column=args.date_column)
    
    # Analyze
    merged = analyze_coverage(merged, ohlcv)
    
    # Save
    save_outputs(merged, args.output, args.backtest_output)
    
    print(f"\n{'='*60}")
    print("✅ MERGE COMPLETE")
    print(f"{'='*60}")

if __name__ == '__main__':
    main()
