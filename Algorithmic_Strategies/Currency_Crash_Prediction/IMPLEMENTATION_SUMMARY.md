# Currency Crash Prediction Model - Implementation Summary

## Overview
Successfully implemented a complete quantitative research strategy for predicting short-term currency crashes based on the "R-Zone" methodology from Quant Radio research.

## Files Created

### Core Implementation
1. **data_fetcher.py** - Fetches FX rates and interest rates from yfinance/FRED
2. **signal_generator.py** - Generates R-Zone signals and identifies crash events
3. **config.yaml** - Configuration for currencies, thresholds, and parameters

### LEAN Backtesting
4. **lean/main.py** - QuantConnect LEAN algorithm for institutional-grade backtesting

### Jupyter Notebooks
5. **notebooks/01_data_exploration.ipynb** - Data visualization and EDA
6. **notebooks/02_signal_generation.ipynb** - Signal generation and statistics
7. **notebooks/03_backtest_analysis.ipynb** - Backtest with transaction costs

### Testing & Documentation
8. **tests/test_signals.py** - Unit tests for signal generation
9. **README.md** - Comprehensive documentation
10. **requirements.txt** - Python dependencies
11. **setup.sh** - Quick start script
12. **.gitignore** - Git ignore rules

## Strategy Logic

### R-Zone Signal (High Crash Risk)
- **Condition A**: Δi (6-month interest rate change) in top 20% (highest quintile)
- **Condition B**: ΔFX (6-month currency depreciation) in bottom 33% (lowest tertile)
- **Signal**: Both conditions met → Short currency for 6 months

### Crash Definition
- Monthly currency return in bottom 4% of historical distribution

### Expected Performance
- **Crash Probability in R-Zone**: ~43%
- **Baseline Crash Probability**: ~7.8%
- **Probability Ratio**: 5.5x
- **Average Lead Time**: 4-5 months

## Key Features

### Data Coverage
- 17 economies (9 advanced, 8 emerging)
- Sample period: 1999-2023 (25 years)
- Monthly frequency
- FX rates vs USD
- Policy interest rates

### Risk Management
- Max country exposure: 10% of capital
- Position duration: 6 months
- Leverage: 1.5x
- Transaction costs: 5-25 bps
- VIX filter: Pause if VIX > 40

### Backtesting
- In-sample calibration: 1999-2018
- Out-of-sample validation: 2019-2023
- Transaction costs and slippage included
- LEAN integration for institutional backtesting

## Usage

### Quick Start
```bash
cd Algorithmic_Strategies/Currency_Crash_Prediction
./setup.sh
```

### Manual Steps
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Fetch data
python data_fetcher.py --start 1999-01-01 --end 2023-12-31

# 3. Generate signals
python signal_generator.py

# 4. Run Jupyter notebooks
jupyter notebook notebooks/01_data_exploration.ipynb
```

### LEAN Backtest
```bash
cd lean
lean backtest main.py
```

## Technical Stack
- **Data**: yfinance, pandas_datareader (FRED)
- **Analysis**: pandas, numpy, scipy, statsmodels
- **Visualization**: matplotlib, seaborn, plotly
- **Backtesting**: QuantConnect LEAN
- **Testing**: pytest

## Economic Rationale
Aggressive central bank rate hike into an already weak currency is interpreted by markets as a signal of desperation and severe underlying economic stress (e.g., inflation, capital flight). This triggers loss of confidence and accelerates capital outflows, leading to a nonlinear crash event.

## Failure Modes & Mitigations

1. **Structural Break**: Central bank communication regime change
   - Mitigation: Rolling out-of-sample validation

2. **Global Risk-Off**: Extreme synchronized crisis
   - Mitigation: VIX > 40 filter to pause strategy

3. **Capital Controls**: Government intervention
   - Mitigation: Exclude high-risk countries

4. **False Positives**: 57% of signals don't crash
   - Mitigation: Position sizing, stop-losses

5. **Data Snooping**: Overfitting to specific thresholds
   - Mitigation: Out-of-sample validation, robustness checks

## Research Reference
Based on "Interest Rates and Short Term Currency Crash Risk" - Quant Radio

## Next Steps
1. Validate on out-of-sample data (2019-2023)
2. Test alternative thresholds (robustness checks)
3. Add sentiment indicators for confirmation
4. Implement real-time monitoring dashboard
5. Integrate with live trading API

## License
MIT License - Educational and research purposes only

---
**Created**: January 2026
**Status**: Complete and ready for backtesting
