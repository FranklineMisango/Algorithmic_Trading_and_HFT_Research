# LEAN Backtesting - Holiday Effect Strategy

## Setup

### Install LEAN CLI

```bash
pip install lean
```

### Initialize LEAN Project

```bash
cd Algorithmic_Strategies/Holiday_Effect/lean
lean init
```

## Running Backtest

### Local Backtest

```bash
lean backtest "Holiday Effect"
```

### Cloud Backtest (QuantConnect)

```bash
lean cloud push --project "Holiday Effect"
lean cloud backtest "Holiday Effect" --open
```

## Algorithm Overview

The algorithm trades AMZN stock around two major shopping events:
- **Black Friday**: 10 trading days before (Friday after Thanksgiving)
- **Prime Day**: 10 trading days before (mid-July)

### Entry Conditions
- 10 trading days before event
- SPY above 200-day moving average (market filter)

### Exit Conditions
- Day before event
- Or 8% stop-loss

### Parameters
- Initial Capital: $1,000,000
- Position Size: 100% (full allocation)
- Holding Period: ~10 trading days per event
- Transaction Costs: Included by LEAN

## Expected Performance

Based on research (1998-2024):
- **Sharpe Ratio**: 0.51-0.55
- **Annual Return**: 3.9%
- **Win Rate**: 76.67%
- **Max Drawdown**: -14.26%

## Files

- `main.py` - Main algorithm implementation
- `config.json` - LEAN configuration
- `README.md` - This file

## Notes

- Algorithm automatically calculates Black Friday dates
- Prime Day dates are hardcoded (2015-2024) with fallback to July 15
- Market filter prevents trading during bear markets
- Benchmark: SPY (S&P 500)
