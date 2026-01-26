# ML Mean Reversion Strategy - Russell 3000

Implementation of the Machine Learning-based Mean Reversion strategy from Quant Radio, targeting short-term price reversals in Russell 3000 stocks.

## Strategy Overview

**Objective**: Capture 6-day mean reversion in oversold/overbought stocks using ML probability forecasts and technical signals.

**Key Components**:
1. **3-Day QPI (Quantitative Pressure Index)**: Proprietary oversold/overbought indicator (0-100 scale)
2. **ML Probability Model**: Gradient Boosting classifier predicting 6-day forward returns
3. **VIX Regime Filter**: Dynamic allocation based on market volatility
4. **Risk Management**: 5% stop loss + 6-day time stop

## Entry Signals

### Long Entry
- QPI_3day < 15 (oversold)
- ML_Probability > 0.60 (>60% chance of bounce)

### Short Entry
- QPI_3day > 85 (overbought)
- ML_Probability > 0.60 (>60% chance of decline)

## Position Management

### VIX Regime Filter
```
VIX_SMA15 = 15-day simple moving average of VIX
Threshold = VIX_SMA15 * 1.15

Bear Market: VIX_Close > Threshold
Bull Market: VIX_Close <= Threshold
```

### Allocation
| Regime | Long Allocation | Short Allocation | Net Exposure |
|--------|----------------|------------------|--------------|
| Bull   | 1.1x           | 0.2x             | +0.9x        |
| Bear   | 0.1x           | 0.2x             | -0.1x        |

### Position Limits
- Max 20 long positions
- Max 20 short positions
- Equal weight within each book

## Risk Management

1. **Stop Loss**: 5% below entry (long) / above entry (short)
2. **Time Stop**: Automatic exit after 6 trading days
3. **Liquidity Filter**: 
   - Min price: $1.00
   - Max position: 5% of 3-month average daily volume
4. **Universe Filter**: Russell 3000 constituents only (point-in-time)

## Files

### Jupyter Notebook
- `notebooks/ml_mean_reversion_training.ipynb`: Model training and feature engineering

### LEAN Algorithm
- `lean/ml_mean_reversion_main.py`: Production backtesting algorithm

### Utilities
- `src/ml_mean_reversion_utils.py`: QPI calculation, feature engineering, signal validation

## Quick Start

### 1. Train Models

```bash
cd notebooks
jupyter notebook ml_mean_reversion_training.ipynb
```

Run all cells to:
- Download sample Russell 3000 data
- Calculate QPI and features
- Train long/short ML models
- Save models to `models/` directory

### 2. Run LEAN Backtest

```bash
cd lean
lean backtest ml_mean_reversion_main.py
```

### 3. Analyze Results

Expected performance (from Quant Radio):
- **Annual Return**: ~27.4%
- **Sharpe Ratio**: 1.17
- **Max Drawdown**: ~46% (COVID crash)
- **Win Rate**: ~62%
- **Avg Return per Trade**: 0.5%
- **Payoff Ratio**: 0.79

## QPI Formula

```python
# Price momentum
ret_3d = (Close[t] - Close[t-3]) / Close[t-3]

# Volume pressure
vol_ratio = Volume[t] / SMA(Volume, 20)

# Volatility
volatility = StdDev(returns, 20)

# QPI calculation
raw_qpi = 50 + (ret_3d / volatility) * 10 - (vol_ratio - 1) * 5
qpi_3day = clip(raw_qpi, 0, 100)
```

**Interpretation**:
- QPI < 15: Oversold (long signal)
- QPI > 85: Overbought (short signal)
- QPI ≈ 50: Neutral

## ML Features

1. **qpi_3day**: Quantitative Pressure Index
2. **rsi_14**: 14-day Relative Strength Index
3. **bb_position**: Bollinger Band position (z-score)
4. **volume_surge**: Volume / SMA(Volume, 5)
5. **mom_5**: 5-day momentum
6. **mom_10**: 10-day momentum
7. **mom_20**: 20-day momentum
8. **vol_ratio**: Volume / SMA(Volume, 20)

## Target Definition

```python
forward_return = (Close[t+6] - Close[t]) / Close[t]
target_long = 1 if forward_return > 0 else 0
target_short = 1 if forward_return < 0 else 0
```

## Critical Implementation Notes

### 1. Survivorship Bias
**CRITICAL**: Use point-in-time Russell 3000 constituent lists. Do NOT include stocks that were added to the index in the future.

From Quant Radio:
> "For any point in the backtest, the trading universe only included stocks that were actually in the Russell 3000 at that specific historical moment."

### 2. Training Data Scope
Train models only on price history from periods when a stock was a Russell 3000 constituent.

### 3. Transaction Costs
Backtest assumes ~2 bps per trade. Real-world costs may be 10-20 bps for illiquid small-caps.

**Mitigation**:
- Tighten liquidity filter (2% of ADV instead of 5%)
- Model slippage as 10 bps penalty
- Paper trade before live deployment

### 4. VIX Threshold Calibration
The 15% threshold was calibrated to historical data (~90th percentile). May need recalibration in new volatility regimes.

## Performance Attribution

### Key Drivers
1. **Mean Reversion Alpha**: Short-term overreactions correct systematically
2. **ML Edge**: Probability filter improves signal quality (62% win rate vs ~50% baseline)
3. **Regime Adaptation**: VIX filter reduces drawdowns in bear markets

### Failure Modes
1. **Overfitting**: Model trained on historical patterns may not generalize
2. **Regime Shift**: VIX threshold may become ineffective
3. **Liquidity Crunch**: Real-world slippage higher than modeled
4. **Short Squeeze**: Small short book vulnerable to coordinated squeezes

## Backtesting Checklist

- [ ] Point-in-time Russell 3000 universe
- [ ] No look-ahead bias in features
- [ ] Transaction costs modeled (5-10 bps)
- [ ] Slippage modeled (VWAP or 10 bps penalty)
- [ ] Liquidity filters applied ($1 min, 5% ADV cap)
- [ ] Walk-forward validation for ML models
- [ ] Out-of-sample testing period
- [ ] Stress test through COVID crash (Q1 2020)

## Extensions

1. **Enhanced ML Models**: Try XGBoost, LightGBM, or LSTM
2. **Alternative QPI**: Experiment with different volatility estimators
3. **Dynamic Thresholds**: Adaptive QPI and probability thresholds
4. **Factor Neutralization**: Hedge out market beta, sector exposure
5. **Options Overlay**: Use puts/calls for tail risk protection

## References

- Quant Radio Podcast: "Machine Learning based Mean Reversion Model"
- Russell 3000 Index Methodology: FTSE Russell
- VIX Methodology: CBOE

## Disclaimer

This implementation is for educational and research purposes only. Past performance does not guarantee future results. The strategy involves substantial risk of loss.

**Key Risk**: The reported 27.4% annual return with 1.17 Sharpe ratio is based on historical simulation. Real-world performance will differ due to:
- Transaction costs and slippage
- Model decay over time
- Regime changes
- Implementation challenges

## License

MIT License - See repository root for details.
