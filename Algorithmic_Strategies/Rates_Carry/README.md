# Rates Carry Strategy

## Overview
Government bond carry strategy exploiting roll-down yield. Goes long bonds with positive carry (yield > expected roll-down return), neutralized against duration and curve risk.

## Strategy Logic

### Signal Generation
- **Primary Signal**: Roll-down yield (return from bond aging down the curve)
- **Entry**: Z-score of roll-down > 1.0
- **Exit**: Z-score of roll-down < 0.5
- **Normalization**: Rolling 252-day z-score

### Risk Factors
- **Duration**: Interest rate sensitivity
- **Curve Slope**: Yield curve steepness (2s10s spread)
- **Flight-to-Quality**: Safe-haven demand during crises

### Portfolio Construction
- **Weighting**: Inverse volatility (63-day lookback)
- **Rebalancing**: Weekly
- **Target Volatility**: 10% annualized
- **Duration Limit**: Max 7 years

## Data Requirements

### Yield Curves
- Source: FRED
- Countries: US, Germany, UK, Japan, Australia, Canada
- Maturities: 2Y, 5Y, 7Y, 10Y, 30Y
- Frequency: Daily

### Bond Prices
- Source: Yahoo Finance (ETF proxies)
- Instruments: TLT, IEF, SHY (US), BUND, GILT, etc.

## File Structure

```
Rates_Carry/
├── config.yaml
├── requirements.txt
├── README.md
├── data_acquisition.py
├── signal_generator.py
├── factor_models.py
├── portfolio_constructor.py
├── backtester.py
├── main.py
├── 01_data_exploration.ipynb
├── 02_signal_analysis.ipynb
├── 03_backtest_analysis.ipynb
└── 04_stress_analysis.ipynb
```

## Usage

```python
from main import RatesCarryStrategy

strategy = RatesCarryStrategy(config_path='config.yaml')
results = strategy.run_backtest()
strategy.generate_report()
```

## Key Metrics
- **Target Sharpe**: > 1.2
- **Max Drawdown**: < 15%
- **Duration Neutrality**: |duration| < 2 years

## References
- Ilmanen, A. (1996). Does duration extension enhance long-term expected returns? *Journal of Fixed Income*
- Koijen, R., Moskowitz, T., Pedersen, L., & Vrugt, E. (2018). Carry. *Journal of Financial Economics*
