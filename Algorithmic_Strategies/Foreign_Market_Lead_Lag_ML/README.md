# Foreign Market Lead-Lag ML Strategy

## Overview
Cross-asset international equity momentum strategy that predicts next-day S&P 500 stock returns using lagged weekly returns from 47 foreign equity markets.

## Strategy Hypothesis
Lagged returns from foreign equity markets contain predictive information for US stock returns due to:
- Global market interconnectedness through supply chains and competition
- Delayed information diffusion from foreign markets to US prices
- Lower media attention to foreign events affecting US firms

## Key Parameters
- **Target Universe**: S&P 500 constituents
- **Predictors**: 47 foreign market indices + individual foreign stocks
- **Lags**: 1, 2, 3, 4 weeks
- **Signals**: 188 market-level + ~13,000 stock-level (market-level preferred)
- **Rebalancing**: Daily
- **Portfolio**: Long top 5%, short bottom 5% (dollar-neutral)

## Expected Performance
- **Gross Annual Return**: ~14.2% (Lasso model)
- **Predictive Coverage**: ~24% of S&P 500 stocks show significant R²_OOS
- **Predictive Horizon**: 5-8 weeks
- **Transaction Costs**: 10-20 bps per trade significantly impacts net returns

## Implementation Structure
```
Foreign_Market_Lead_Lag_ML/
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_model_training.ipynb
│   └── 04_backtest_analysis.ipynb
├── data_acquisition.py
├── feature_engineering.py
├── ml_models.py
├── portfolio_constructor.py
├── backtester.py
├── lean_algorithm.py
├── main.py
├── config.yaml
└── requirements.txt
```

## Usage
```bash
# Install dependencies
pip install -r requirements.txt

# Run full pipeline
python main.py

# Or use notebooks for step-by-step analysis
jupyter notebook notebooks/01_data_exploration.ipynb
```

## Risk Factors
1. **Transaction Costs**: High daily turnover erodes alpha significantly
2. **Model Decay**: Alpha may diminish as markets become more efficient
3. **Liquidity Risk**: Short leg execution challenges
4. **Data Snooping**: Strict OOS validation required

## References
- Quant Radio: "How Foreign Market Data Predicts US Stock Movements"
- Research shows ~24% of S&P 500 stocks have significant predictability
