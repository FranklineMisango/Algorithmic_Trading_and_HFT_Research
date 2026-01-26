# Market-Neutral News Sentiment Strategy

## Overview
Market-neutral daily alpha strategy exploiting short-term price reactions to news sentiment. Based on Quant Radio research: "Market Signals from Social Media".

## Strategy Summary
- **Type**: Market-Neutral, Long/Short Equity
- **Holding Period**: 24 hours (daily rebalance)
- **Signal**: News sentiment scores (log-odds ratio aggregation)
- **Universe**: Russell 3000 (shorts limited to S&P 1500)
- **Target Sharpe**: >1.5 (backtest achieved 1.88-2.14)
- **Max Drawdown**: <15% (backtest: 11-13%)

## Key Features
- Log-odds sentiment aggregation from news probabilities
- Sector-relative return prediction (GICS standardization)
- Linear regression with 10-year rolling walk-forward validation
- Topic one-hot encoding
- Conservative transaction cost modeling (10-15 bps)

## Installation

```bash
cd Algorithmic_Strategies/Market_Neutral_News_Sentiment
pip install -r requirements.txt
```

## Usage

### Jupyter Notebooks
```bash
jupyter notebook notebooks/01_data_preparation.ipynb
```

### LEAN Backtest
```bash
cd lean
lean backtest main.py
```

## Data Requirements
- News sentiment feed (Dow Jones + NLP provider)
- Fields: ticker, timestamp, P_positive, P_negative, P_neutral, relevance_score, topics
- Daily pricing and GICS sector classifications
- S&P 1500 constituent history

## Performance Metrics
- Annualized Return: 24.4% alpha (Fama-French 3-factor)
- Sharpe Ratio: 1.88 (base), 2.14 (10% net long)
- Max Drawdown: 11% (base), 13% (10% net long)
- Win Rate: Monitor daily

## Risk Management
- Per-stock limit: 2% max weight
- Sector exposure: <20% net per GICS sector
- Stop-loss: Pause if 30-day MDD >15%
- Short universe: S&P 1500 only

## References
- Quant Radio: "Market Signals from Social Media"
- Fama-French 3-Factor Model for alpha attribution
