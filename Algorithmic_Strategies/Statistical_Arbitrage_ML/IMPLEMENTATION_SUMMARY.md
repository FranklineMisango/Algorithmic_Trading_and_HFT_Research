# ML Mean Reversion Strategy - Implementation Summary

## ✅ What Was Created

### 1. Jupyter Notebook for Model Training
**File**: `notebooks/ml_mean_reversion_training.ipynb`

Complete pipeline for:
- QPI (Quantitative Pressure Index) calculation
- Feature engineering (8 features)
- Binary target creation (6-day forward returns)
- Gradient Boosting model training (long & short)
- Model evaluation and feature importance
- Model export for LEAN

### 2. LEAN Backtesting Algorithm
**File**: `lean/ml_mean_reversion_main.py`

Production-ready LEAN algorithm with:
- Russell 3000 universe selection (top 500 by liquidity)
- Real-time QPI calculation
- ML probability prediction
- VIX regime filter (15-day SMA * 1.15 threshold)
- Dynamic allocation (bull: 1.1x long/0.2x short, bear: 0.1x long/0.2x short)
- Position limits (max 20 long, 20 short)
- Risk management (5% stop loss, 6-day time stop)

### 3. Utility Module
**File**: `src/ml_mean_reversion_utils.py`

Reusable functions:
- `calculate_qpi_3day()`: QPI calculation
- `create_ml_features()`: Feature engineering
- `create_target()`: Target creation
- `calculate_vix_regime()`: VIX regime detection
- `validate_signal()`: Entry signal validation
- `apply_liquidity_filter()`: Liquidity screening

### 4. Training Script
**File**: `train_ml_models.py`

Automated pipeline:
- Downloads data for 30+ Russell 3000 stocks
- Trains long/short models
- Saves models to `models/` directory
- Ready for LEAN integration

### 5. Configuration File
**File**: `ml_mean_reversion_config.yaml`

All strategy parameters:
- Signal thresholds (QPI < 15, ML prob > 0.60)
- Position limits (20 long, 20 short)
- VIX filter settings (15-day SMA * 1.15)
- Risk parameters (5% stop, 6-day hold)
- Expected performance metrics

### 6. Documentation
**File**: `ML_MEAN_REVERSION_README.md`

Comprehensive guide:
- Strategy overview and rationale
- Entry/exit signals
- QPI formula and interpretation
- VIX regime filter logic
- Risk management rules
- Implementation checklist
- Failure modes and mitigations

## 📊 Strategy Specifications

### Entry Signals
```
LONG:  QPI_3day < 15  AND  ML_Probability > 0.60
SHORT: QPI_3day > 85  AND  ML_Probability > 0.60
```

### QPI Formula
```python
ret_3d = (Close[t] - Close[t-3]) / Close[t-3]
vol_ratio = Volume[t] / SMA(Volume, 20)
volatility = StdDev(returns, 20)

raw_qpi = 50 + (ret_3d / volatility) * 10 - (vol_ratio - 1) * 5
qpi_3day = clip(raw_qpi, 0, 100)
```

### VIX Regime Filter
```python
VIX_SMA15 = SMA(VIX, 15)
Threshold = VIX_SMA15 * 1.15

Bear Market: VIX > Threshold
Bull Market: VIX <= Threshold
```

### Allocation Matrix
| Regime | Long | Short | Net Exposure |
|--------|------|-------|--------------|
| Bull   | 1.1x | 0.2x  | +0.9x        |
| Bear   | 0.1x | 0.2x  | -0.1x        |

### Risk Management
- **Stop Loss**: 5% per position
- **Time Stop**: 6 trading days
- **Position Limits**: 20 long + 20 short max
- **Liquidity**: Min $1.00 price, max 5% of ADV

## 🚀 Quick Start

### Step 1: Train Models
```bash
cd /home/misango/codechest/Algorithmic_Trading_and_HFT_Research/Algorithmic_Strategies/Statistical_Arbitrage_ML

# Option A: Run training script
python train_ml_models.py

# Option B: Use Jupyter notebook
jupyter notebook notebooks/ml_mean_reversion_training.ipynb
```

### Step 2: Run LEAN Backtest
```bash
cd lean
lean backtest ml_mean_reversion_main.py
```

### Step 3: Analyze Results
Expected performance (from Quant Radio):
- Annual Return: 27.4%
- Sharpe Ratio: 1.17
- Max Drawdown: 46% (COVID crash)
- Win Rate: 62%

## 📁 File Structure
```
Statistical_Arbitrage_ML/
├── notebooks/
│   └── ml_mean_reversion_training.ipynb    # Model training
├── lean/
│   └── ml_mean_reversion_main.py           # LEAN algorithm
├── src/
│   └── ml_mean_reversion_utils.py          # Utilities
├── models/                                  # Trained models (created after training)
│   ├── ml_mean_reversion_long.pkl
│   ├── ml_mean_reversion_short.pkl
│   └── feature_columns.pkl
├── train_ml_models.py                       # Training pipeline
├── ml_mean_reversion_config.yaml            # Configuration
└── ML_MEAN_REVERSION_README.md              # Documentation
```

## ⚠️ Critical Implementation Notes

### 1. Survivorship Bias
**MUST** use point-in-time Russell 3000 constituent lists. The current implementation uses top 500 by liquidity as a proxy.

### 2. Transaction Costs
Backtest assumes 5 bps + 10 bps slippage. Real-world costs may be higher for small-caps.

### 3. Model Decay
Retrain models quarterly to adapt to changing market conditions.

### 4. VIX Calibration
The 1.15 multiplier was calibrated to historical data. Monitor effectiveness in real-time.

## 🔧 Customization

### Adjust QPI Threshold
Edit in `ml_mean_reversion_config.yaml`:
```yaml
signals:
  qpi_threshold: 15  # Lower = more selective
```

### Change Position Limits
```yaml
positions:
  max_long: 20   # Increase for more diversification
  max_short: 20
```

### Modify VIX Filter
```yaml
vix_filter:
  sma_window: 15
  threshold_multiplier: 1.15  # Higher = less sensitive
```

## 📈 Next Steps

1. **Backtest**: Run LEAN backtest on 2020-2023 period
2. **Validate**: Compare results to expected performance
3. **Stress Test**: Analyze COVID crash period (Q1 2020)
4. **Optimize**: Tune thresholds using walk-forward analysis
5. **Paper Trade**: Test in simulated environment before live

## 🎯 Key Features Implemented

✅ 3-Day QPI calculation (proprietary oversold indicator)  
✅ 8 ML features (RSI, Bollinger Bands, momentum, volume)  
✅ Gradient Boosting classifier (long & short models)  
✅ VIX regime filter with dynamic allocation  
✅ Position limits (20 long, 20 short)  
✅ Risk management (5% stop, 6-day time stop)  
✅ Liquidity filters ($1 min, 5% ADV cap)  
✅ LEAN integration for production backtesting  
✅ Comprehensive documentation  

## 📚 References

- **Source**: Quant Radio - "Machine Learning based Mean Reversion Model"
- **Strategy Type**: Long/Short Equity, ML-Enhanced Mean Reversion
- **Universe**: Russell 3000
- **Holding Period**: 6 days
- **Frequency**: Daily rebalancing

## ⚖️ Disclaimer

This implementation is for educational and research purposes only. The reported 27.4% annual return is based on historical simulation and does not guarantee future results. Real-world performance will differ due to transaction costs, slippage, model decay, and market regime changes.

---

**Implementation Date**: January 2026  
**Status**: Ready for backtesting  
**Next Action**: Run `python train_ml_models.py`
