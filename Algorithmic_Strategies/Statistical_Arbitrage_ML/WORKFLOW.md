# ML Mean Reversion Strategy - Complete Workflow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         IMPLEMENTATION WORKFLOW                             │
└─────────────────────────────────────────────────────────────────────────────┘

PHASE 1: DATA COLLECTION & FEATURE ENGINEERING
═══════════════════════════════════════════════
┌──────────────┐
│ Russell 3000 │
│   Stocks     │──────┐
└──────────────┘      │
                      ▼
┌──────────────────────────────────────┐
│  Download Historical Data            │
│  - OHLCV (Open, High, Low, Close, V) │
│  - 2018-2023 (5+ years)              │
│  - Daily resolution                  │
└──────────────────────────────────────┘
                      │
                      ▼
┌──────────────────────────────────────┐
│  Calculate QPI (3-Day)               │
│  ┌────────────────────────────────┐  │
│  │ ret_3d = price momentum        │  │
│  │ vol_ratio = volume pressure    │  │
│  │ volatility = price volatility  │  │
│  │ qpi = f(ret_3d, vol, vol_ratio)│  │
│  └────────────────────────────────┘  │
└──────────────────────────────────────┘
                      │
                      ▼
┌──────────────────────────────────────┐
│  Engineer 8 ML Features              │
│  1. qpi_3day                         │
│  2. rsi_14                           │
│  3. bb_position                      │
│  4. volume_surge                     │
│  5-7. mom_5, mom_10, mom_20          │
│  8. vol_ratio                        │
└──────────────────────────────────────┘
                      │
                      ▼
┌──────────────────────────────────────┐
│  Create Targets                      │
│  - forward_return = Close[t+6]/Close │
│  - target_long = (fwd_ret > 0)       │
│  - target_short = (fwd_ret < 0)      │
└──────────────────────────────────────┘


PHASE 2: MODEL TRAINING
════════════════════════
                      │
                      ▼
┌──────────────────────────────────────┐
│  Split Data: 80% Train / 20% Test   │
└──────────────────────────────────────┘
                      │
          ┌───────────┴───────────┐
          ▼                       ▼
┌──────────────────┐    ┌──────────────────┐
│  Train Long Model│    │ Train Short Model│
│  ──────────────  │    │  ──────────────  │
│  GradientBoosting│    │  GradientBoosting│
│  - 100 trees     │    │  - 100 trees     │
│  - depth 4       │    │  - depth 4       │
│  - lr 0.1        │    │  - lr 0.1        │
└──────────────────┘    └──────────────────┘
          │                       │
          └───────────┬───────────┘
                      ▼
┌──────────────────────────────────────┐
│  Evaluate Models                     │
│  - Accuracy (>0.6 threshold)         │
│  - AUC-ROC                           │
│  - Feature importance                │
└──────────────────────────────────────┘
                      │
                      ▼
┌──────────────────────────────────────┐
│  Save Models                         │
│  - ml_mean_reversion_long.pkl        │
│  - ml_mean_reversion_short.pkl       │
│  - feature_columns.pkl               │
└──────────────────────────────────────┘


PHASE 3: BACKTESTING (LEAN)
════════════════════════════
                      │
                      ▼
┌──────────────────────────────────────┐
│  Initialize LEAN Algorithm           │
│  - Start: 2020-01-01                 │
│  - End: 2023-12-31                   │
│  - Capital: $1,000,000               │
└──────────────────────────────────────┘
                      │
                      ▼
┌──────────────────────────────────────┐
│  Daily Rebalancing Loop              │
│  ┌────────────────────────────────┐  │
│  │ 1. Update VIX data             │  │
│  │ 2. Calculate VIX regime        │  │
│  │ 3. Exit positions (stop/time)  │  │
│  │ 4. Generate new signals        │  │
│  │ 5. Enter new positions         │  │
│  └────────────────────────────────┘  │
└──────────────────────────────────────┘
                      │
          ┌───────────┴───────────┐
          ▼                       ▼
┌──────────────────┐    ┌──────────────────┐
│  VIX Regime      │    │  Signal Generation│
│  ──────────────  │    │  ──────────────  │
│  IF VIX > SMA*1.15│   │  Calculate QPI   │
│    Bear: 0.1L/0.2S│   │  Get ML prob     │
│  ELSE              │   │  IF QPI<15 & P>60│
│    Bull: 1.1L/0.2S│   │    → LONG        │
│                   │    │  IF QPI>85 & P>60│
│                   │    │    → SHORT       │
└──────────────────┘    └──────────────────┘
          │                       │
          └───────────┬───────────┘
                      ▼
┌──────────────────────────────────────┐
│  Position Management                 │
│  - Max 20 long + 20 short            │
│  - Equal weight per position         │
│  - 5% stop loss                      │
│  - 6-day time stop                   │
└──────────────────────────────────────┘
                      │
                      ▼
┌──────────────────────────────────────┐
│  Performance Analytics               │
│  - Annual Return: 27.4% (expected)   │
│  - Sharpe Ratio: 1.17                │
│  - Max Drawdown: 46%                 │
│  - Win Rate: 62%                     │
└──────────────────────────────────────┘


PHASE 4: MONITORING & OPTIMIZATION
═══════════════════════════════════
                      │
                      ▼
┌──────────────────────────────────────┐
│  Monitor Performance                 │
│  - Track daily P&L                   │
│  - Monitor model accuracy            │
│  - Check VIX regime effectiveness    │
└──────────────────────────────────────┘
                      │
                      ▼
┌──────────────────────────────────────┐
│  Quarterly Retraining                │
│  - Retrain models on recent data     │
│  - Validate out-of-sample            │
│  - Update if performance improves    │
└──────────────────────────────────────┘
                      │
                      ▼
┌──────────────────────────────────────┐
│  Risk Management Review              │
│  - Adjust position limits            │
│  - Recalibrate VIX threshold         │
│  - Update transaction cost estimates │
└──────────────────────────────────────┘
```

## Key Decision Points

### Entry Decision Tree
```
For each stock in Russell 3000:
│
├─ Calculate QPI_3day
│  │
│  ├─ QPI < 15? (Oversold)
│  │  │
│  │  └─ Calculate ML Features
│  │     │
│  │     └─ ML_Prob_Long > 0.60?
│  │        │
│  │        └─ YES → LONG SIGNAL ✓
│  │
│  └─ QPI > 85? (Overbought)
│     │
│     └─ Calculate ML Features
│        │
│        └─ ML_Prob_Short > 0.60?
│           │
│           └─ YES → SHORT SIGNAL ✓
```

### Allocation Decision Tree
```
Check VIX Regime:
│
├─ VIX > SMA(15) * 1.15? (Bear Market)
│  │
│  └─ YES → Allocate: 0.1x Long, 0.2x Short
│
└─ NO (Bull Market)
   │
   └─ Allocate: 1.1x Long, 0.2x Short
```

### Exit Decision Tree
```
For each open position:
│
├─ Current Price < Entry * 0.95? (Long)
│  │
│  └─ YES → EXIT (Stop Loss) ✗
│
├─ Current Price > Entry * 1.05? (Short)
│  │
│  └─ YES → EXIT (Stop Loss) ✗
│
└─ Days Held >= 6?
   │
   └─ YES → EXIT (Time Stop) ✓
```

## File Execution Order

```
1. train_ml_models.py
   └─ Downloads data
   └─ Trains models
   └─ Saves to models/

2. notebooks/ml_mean_reversion_training.ipynb (optional)
   └─ Interactive exploration
   └─ Feature importance analysis
   └─ Model validation

3. lean/ml_mean_reversion_main.py
   └─ Runs backtest
   └─ Generates performance report
   └─ Outputs equity curve
```

## Success Metrics

✓ Models trained with AUC > 0.60
✓ Backtest Sharpe Ratio > 1.0
✓ Win Rate > 55%
✓ Max Drawdown < 50%
✓ Annual Return > 15%

## Next Steps After Implementation

1. ✅ Train models: `python train_ml_models.py`
2. ✅ Review training results in notebook
3. ✅ Run LEAN backtest: `cd lean && lean backtest ml_mean_reversion_main.py`
4. ⏳ Analyze backtest results
5. ⏳ Stress test through COVID period
6. ⏳ Optimize parameters if needed
7. ⏳ Paper trade for 1-3 months
8. ⏳ Deploy to live trading (if validated)
