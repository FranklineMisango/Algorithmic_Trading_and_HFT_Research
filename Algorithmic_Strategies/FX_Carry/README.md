# FX Carry Strategy

## Overview
Currency carry trade strategy that exploits interest rate differentials between currency pairs. Goes long high-yielding currencies and short low-yielding currencies, neutralized against systematic FX risk factors.

## Strategy Logic

### Signal Generation
- **Primary Signal**: Interest rate differential between currency pairs
- **Entry**: Z-score of carry > 1.0
- **Exit**: Z-score of carry < 0.5
- **Normalization**: Rolling 252-day z-score

### Risk Factors
The strategy neutralizes exposure to:
- **Dollar Index**: Overall USD strength/weakness
- **Safe Haven**: Flight-to-quality flows (CHF, JPY, USD)
- **Commodity FX**: Commodity-linked currencies (AUD, NZD, CAD)

### Portfolio Construction
- **Weighting**: Inverse volatility (63-day lookback)
- **Rebalancing**: Weekly
- **Target Volatility**: 10% annualized
- **Gross Leverage**: 2x
- **Net Leverage**: 1x (market neutral)

## Data Requirements

### Spot FX Rates
- Source: Yahoo Finance
- Pairs: AUDJPY, NZDJPY, EURUSD, GBPUSD, USDJPY, etc.
- Frequency: Daily

### Interest Rates
- Source: FRED
- Data: Central bank policy rates for each currency
- Update: Monthly or as announced

## File Structure

```
FX_Carry/
├── config.yaml                      # Strategy configuration
├── requirements.txt                 # Python dependencies
├── README.md                        # This file
├── data_acquisition.py             # FX spot and interest rate data
├── signal_generator.py             # Carry signal calculation
├── factor_models.py                # FX factor neutralization
├── portfolio_constructor.py        # Position sizing and risk management
├── backtester.py                   # Performance evaluation
├── main.py                         # Orchestration script
├── 01_data_exploration.ipynb       # EDA on FX data
├── 02_signal_analysis.ipynb        # Carry signal analysis
├── 03_backtest_analysis.ipynb      # Performance results
└── 04_stress_analysis.ipynb        # Crisis period analysis
```

## Usage

```python
from main import FXCarryStrategy

# Initialize strategy
strategy = FXCarryStrategy(config_path='config.yaml')

# Run backtest
results = strategy.run_backtest()

# Generate report
strategy.generate_report()
```

## Key Metrics
- **Target Sharpe**: > 1.0
- **Max Drawdown**: < 20%
- **Win Rate**: > 55%
- **Factor Neutrality**: |beta| < 0.3 for all factors

## References
- Lustig, H., Roussanov, N., & Verdelhan, A. (2011). Common risk factors in currency markets. *Review of Financial Studies*
- Menkhoff, L., Sarno, L., Schmeling, M., & Schrimpf, A. (2012). Carry trades and global FX volatility. *Journal of Finance*
