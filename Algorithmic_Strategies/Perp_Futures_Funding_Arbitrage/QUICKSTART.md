# Quick Start Guide

## Installation

```bash
cd Perp_Futures_Funding_Arbitrage
pip install -r requirements.txt
```

## Configuration

1. Copy environment template:
```bash
cp .env.example .env
```

2. Edit `.env` with your API keys:
```bash
BINANCE_API_KEY=your_key
BINANCE_SECRET_KEY=your_secret
```

3. Review `config.yaml` and adjust parameters if needed

## Running the Strategy

### Option 1: Full Backtest
```bash
python main.py
```

### Option 2: Jupyter Notebooks (Recommended for Research)

```bash
jupyter notebook
```

Then open notebooks in order:
1. `01_data_exploration.ipynb` - Explore perpetual and spot data
2. `02_bound_validation.ipynb` - Validate no-arbitrage bounds (replicate Figure 5)
3. `03_backtest_analysis.ipynb` - Run in-sample/out-of-sample backtests
4. `04_capacity_analysis.ipynb` - Determine strategy capacity

### Option 3: LEAN Backtesting

```bash
lean backtest lean_algorithm.py
```

## Expected Output

The strategy will:
1. Fetch 2 years of hourly data from Binance
2. Calculate clamp-adjusted no-arbitrage bounds
3. Validate that >95% of price ratios stay within bounds
4. Generate trading signals when bounds are breached
5. Execute delta-neutral arbitrage trades
6. Report performance metrics

## Key Metrics to Monitor

- **Sharpe Ratio**: Target >1.5
- **Max Drawdown**: Target <10%
- **Win Rate**: Target >60%
- **Bound Coverage**: Should be >95%

## Troubleshooting

### API Rate Limits
If you hit Binance rate limits, increase the delay in `data_acquisition.py`:
```python
time.sleep(0.5)  # Increase this value
```

### Missing Data
Ensure you have stable internet connection. The script will automatically retry failed requests.

### Low Performance
Check:
1. Slippage assumptions in `config.yaml`
2. Transaction fee settings
3. Funding rate data quality

## Next Steps

1. Run stress tests using `risk_manager.py`
2. Optimize parameters using grid search
3. Test on multiple assets (ETH, other perpetuals)
4. Implement live trading with paper account

## Support

For issues, refer to the main README or open an issue on GitHub.
