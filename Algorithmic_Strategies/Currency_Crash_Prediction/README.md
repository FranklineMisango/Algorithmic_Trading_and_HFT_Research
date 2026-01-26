# Currency Crash Prediction Model

## Overview

Early warning system for short-term currency crashes (1-6 month horizon) using domestic economic stress indicators. Based on the finding that aggressive interest rate tightening into existing currency weakness creates a "Red Zone" (R-Zone) that dramatically increases crash probability.

## Strategy Logic

**Crash Definition**: Monthly currency return in bottom 4% of historical distribution

**R-Zone Signal** (High Risk):
- Condition A: Δi (6-month interest rate change) in top 20% (highest quintile)
- Condition B: ΔFX (6-month currency depreciation) in bottom 33% (lowest tertile)

**Economic Rationale**: Rate hike into weak currency signals desperation, triggering capital flight

## Performance Metrics

- **Crash Probability in R-Zone**: ~43%
- **Baseline Crash Probability**: ~7.8%
- **Average Lead Time**: 4-5 months
- **Sample**: 17 economies (9 advanced, 8 emerging), 1999-2023

## Data Requirements

- **Frequency**: Monthly
- **Currency Pairs**: 17+ economies vs USD/EUR
- **Policy Rates**: Local benchmark rates (3M T-bill, policy rate)
- **Vendors**: Bloomberg (FXFX, C016), Refinitiv, Haver Analytics

## Project Structure

```
Currency_Crash_Prediction/
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_signal_generation.ipynb
│   └── 03_backtest_analysis.ipynb
├── lean/
│   └── main.py                    # QuantConnect LEAN algorithm
├── data/
│   └── currency_data.csv
├── tests/
│   └── test_signals.py
├── config.yaml
├── data_fetcher.py
├── signal_generator.py
├── requirements.txt
└── README.md
```

## Installation

```bash
cd Algorithmic_Strategies/Currency_Crash_Prediction
pip install -r requirements.txt
```

## Usage

### 1. Data Collection
```bash
python data_fetcher.py --start 1999-01-01 --end 2023-12-31
```

### 2. Signal Generation
```python
from signal_generator import CurrencyCrashPredictor

predictor = CurrencyCrashPredictor()
predictor.load_data('data/currency_data.csv')
signals = predictor.generate_signals()
```

### 3. Jupyter Analysis
```bash
jupyter notebook notebooks/01_data_exploration.ipynb
```

### 4. LEAN Backtest
```bash
cd lean
lean backtest main.py
```

## Configuration

Edit `config.yaml`:

```yaml
currencies:
  - EUR
  - JPY
  - GBP
  - CAD
  - AUD
  - BRL
  - CNY
  - INR

lookback_months: 6
rate_threshold_percentile: 80  # Top 20%
fx_threshold_percentile: 33    # Bottom 33%
crash_threshold_percentile: 4  # Bottom 4%
```

## Risk Management

- **Max Country Exposure**: 10% of capital
- **Leverage**: 1x-2x
- **Position Duration**: 6 months or until crash
- **Transaction Costs**: 5-25 bps depending on currency

## Failure Modes

1. **Structural Break**: Central bank communication regime change
2. **Global Risk-Off**: Extreme synchronized crisis (VIX > 40 filter)
3. **Capital Controls**: Government intervention (exclude high-risk countries)
4. **False Positives**: 57% of signals don't crash in 6 months
5. **Data Snooping**: Validate on out-of-sample period

## References

- Quant Radio: "Interest Rates and Short Term Currency Crash Risk"
- Sample Period: 1999-2023, 17 economies

## License

MIT License
