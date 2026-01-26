# Foreign Market Lead-Lag ML Strategy - Implementation Summary

## Overview
Complete implementation of a cross-asset international equity momentum strategy that predicts S&P 500 stock returns using lagged weekly returns from 47 foreign equity markets via machine learning.

## Strategy Specification

### Core Hypothesis
Lagged returns from foreign equity markets contain predictive information for US stock returns due to:
- Global market interconnectedness through supply chains and competition
- Delayed information diffusion from foreign markets to US prices
- Lower media attention to foreign events affecting US firms

### Signal Definition
- **Predictors**: Weekly returns from 47 foreign market ETFs
- **Lags**: 1, 2, 3, 4 weeks
- **Features**: 188 market-level signals (47 markets × 4 lags)
- **Target**: Next-day returns for S&P 500 constituents

### Portfolio Construction
- **Universe**: S&P 500 stocks
- **Long**: Top 5% by predicted return
- **Short**: Bottom 5% by predicted return
- **Weighting**: Equal weight within each leg
- **Rebalancing**: Daily
- **Dollar Neutral**: Yes

### Expected Performance
- **Gross Annual Return**: ~14.2% (Lasso model)
- **Predictive Coverage**: ~24% of stocks show significant R²_OOS
- **Predictive Horizon**: 5-8 weeks
- **Transaction Costs Impact**: Significant (10-20 bps per trade)

## File Structure

```
Foreign_Market_Lead_Lag_ML/
├── README.md                    # Strategy overview
├── config.yaml                  # Configuration parameters
├── requirements.txt             # Python dependencies
├── main.py                      # Main execution script
├── data_acquisition.py          # Data download module
├── feature_engineering.py       # Feature creation module
├── ml_models.py                 # ML model training
├── portfolio_constructor.py     # Portfolio construction
├── backtester.py               # Performance analysis
├── lean_algorithm.py           # LEAN/QuantConnect integration
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_model_training.ipynb
│   └── 04_backtest_analysis.ipynb
├── data/                       # Downloaded data
├── models/                     # Trained models
└── results/                    # Backtest results
```

## Quick Start

### 1. Installation

```bash
# Navigate to strategy directory
cd Algorithmic_Strategies/Foreign_Market_Lead_Lag_ML

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Complete Pipeline

```bash
# Run full strategy pipeline
python main.py
```

This will:
1. Download S&P 500 and foreign market data
2. Create lagged features with standardization
3. Train Lasso models for all stocks
4. Generate daily predictions
5. Simulate portfolio performance
6. Calculate performance metrics
7. Generate visualizations

### 3. Step-by-Step Analysis (Jupyter Notebooks)

```bash
# Start Jupyter
jupyter notebook

# Open notebooks in order:
# 1. notebooks/01_data_exploration.ipynb
# 2. notebooks/02_feature_engineering.ipynb
# 3. notebooks/03_model_training.ipynb
# 4. notebooks/04_backtest_analysis.ipynb
```

### 4. LEAN/QuantConnect Integration

```bash
# Copy LEAN algorithm
cp lean_algorithm.py /path/to/lean/Algorithm.Python/

# Run backtest
lean backtest lean_algorithm.py
```

## Configuration

Edit `config.yaml` to customize:

### Data Parameters
```yaml
data:
  start_date: "2010-01-01"
  end_date: "2024-12-31"
  foreign_markets: [...]  # 47 market ETFs
```

### Feature Engineering
```yaml
features:
  lags: [1, 2, 3, 4]
  standardize: true
  winsorize: true
```

### Model Selection
```yaml
models:
  primary_model: "lasso"  # Options: lasso, random_forest, gradient_boosting
  lasso:
    alpha: 0.01
```

### Portfolio Construction
```yaml
portfolio:
  long_percentile: 95   # Top 5%
  short_percentile: 5   # Bottom 5%
  rebalance_frequency: "daily"
```

### Transaction Costs
```yaml
costs:
  commission: 0.001  # 10 bps
  slippage: 0.0005   # 5 bps
```

## Module Documentation

### data_acquisition.py
Downloads and processes market data.

**Key Functions**:
- `get_sp500_constituents()`: Fetch S&P 500 tickers
- `download_daily_prices()`: Download price data
- `download_foreign_markets()`: Download foreign ETF data
- `resample_to_weekly()`: Convert to weekly frequency

### feature_engineering.py
Creates predictive features from foreign market returns.

**Key Functions**:
- `create_lagged_features()`: Generate lagged return features
- `standardize_features()`: Cross-sectional standardization
- `winsorize_features()`: Handle extreme values
- `prepare_training_data()`: Align features with targets

### ml_models.py
Trains and validates machine learning models.

**Key Classes**:
- `MLModels`: Single model training and validation
- `MultiStockPredictor`: Manages models for multiple stocks

**Key Methods**:
- `walk_forward_validation()`: Out-of-sample testing
- `train_final_model()`: Train on full dataset
- `predict()`: Generate predictions

### portfolio_constructor.py
Constructs long/short portfolio from predictions.

**Key Classes**:
- `PortfolioConstructor`: Portfolio weight calculation
- `PortfolioSimulator`: Full simulation with costs

**Key Methods**:
- `select_long_short()`: Select top/bottom percentiles
- `calculate_position_weights()`: Equal weight allocation
- `apply_transaction_costs()`: Apply costs to returns

### backtester.py
Comprehensive performance analysis.

**Key Methods**:
- `calculate_metrics()`: Compute performance metrics
- `plot_performance()`: Visualization suite
- `run_backtest()`: Complete backtest analysis

## Performance Metrics

The strategy tracks:

### Returns
- Total Return
- Annualized Return
- Monthly/Yearly Returns

### Risk
- Volatility (annualized)
- Maximum Drawdown
- Downside Deviation

### Risk-Adjusted
- Sharpe Ratio
- Sortino Ratio
- Information Ratio

### Trading
- Win Rate
- Average Turnover
- Number of Positions

### Benchmark Comparison
- Alpha
- Beta
- Correlation

## Research Validation

### Expected Results (from research)
- **R²_OOS > 0**: ~24% of S&P 500 stocks
- **Gross Annual Return**: ~14.2% (Lasso)
- **Predictive Horizon**: 5-8 weeks
- **Information Diffusion**: Slower for less-covered stocks

### Validation Checks
1. **Out-of-Sample R²**: Should be positive for ~24% of stocks
2. **Information Coefficient**: Should show significant correlation
3. **Transaction Cost Impact**: Should significantly reduce net returns
4. **Turnover**: High due to daily rebalancing

## Risk Factors & Mitigations

### 1. Transaction Costs (HIGH SEVERITY)
**Risk**: High daily turnover erodes alpha significantly

**Mitigations**:
- Reduce rebalancing frequency (weekly instead of daily)
- Implement position buffers (e.g., only rebalance if prediction changes by >X%)
- Use futures/ETFs for short exposure
- Negotiate institutional trading costs

### 2. Model Decay (HIGH SEVERITY)
**Risk**: Predictive power diminishes as markets become more efficient

**Mitigations**:
- Monitor R²_OOS decay over time
- Retrain models regularly (monthly/quarterly)
- Expand signal set (add text data, satellite imagery)
- Implement ensemble models

### 3. Liquidity Risk (MEDIUM SEVERITY)
**Risk**: Short leg execution challenges, especially for smaller stocks

**Mitigations**:
- Apply minimum ADV filters
- Use ETFs/futures for short exposure
- Limit position sizes
- Monitor borrow costs

### 4. Data Snooping (MEDIUM SEVERITY)
**Risk**: Overfitting to historical patterns

**Mitigations**:
- Strict walk-forward validation
- Never test on training data
- Use market-level signals (avoid stock-level noise)
- Regular out-of-sample testing

### 5. Black Box Risk (LOW-MEDIUM SEVERITY)
**Risk**: Unstable relationships, unclear drivers

**Mitigations**:
- Use interpretable models (Lasso preferred)
- Monitor feature importance with SHAP
- Stress test by removing top contributors
- Document economic rationale

## Optimization Opportunities

### Reduce Transaction Costs
1. **Weekly Rebalancing**: Reduce from daily to weekly
2. **Position Buffers**: Only rebalance if prediction changes significantly
3. **Threshold Filters**: Require minimum prediction confidence

### Improve Predictive Power
1. **Sector-Specific Models**: Train separate models per sector
2. **Regime Detection**: Adjust strategy based on market regime
3. **Alternative Data**: Add text, sentiment, satellite data
4. **Ensemble Methods**: Combine multiple model types

### Risk Management
1. **Dynamic Position Sizing**: Scale based on prediction confidence
2. **Volatility Targeting**: Adjust exposure based on realized volatility
3. **Stop-Loss Rules**: Exit positions with large adverse moves
4. **Correlation Monitoring**: Reduce exposure during high correlation periods

## Troubleshooting

### Data Download Issues
```python
# If yfinance fails, try:
import yfinance as yf
yf.pdr_override()

# Or use alternative data sources
from pandas_datareader import data as pdr
```

### Memory Issues
```python
# Process stocks in batches
batch_size = 50
for i in range(0, len(stocks), batch_size):
    batch = stocks[i:i+batch_size]
    # Process batch
```

### Slow Training
```python
# Use parallel processing
from joblib import Parallel, delayed

results = Parallel(n_jobs=-1)(
    delayed(train_model)(stock) for stock in stocks
)
```

## References

1. **Research Paper**: "How Foreign Market Data Predicts US Stock Movements" (Quant Radio)
2. **Lasso Regression**: Tibshirani (1996) - "Regression Shrinkage and Selection via the Lasso"
3. **Walk-Forward Validation**: Pardo (2008) - "The Evaluation and Optimization of Trading Strategies"
4. **Transaction Cost Analysis**: Frazzini et al. (2018) - "Buffett's Alpha"

## Support

For issues or questions:
1. Check the README.md
2. Review notebook examples
3. Examine log files (strategy.log)
4. Open GitHub issue

## License

MIT License - See LICENSE file for details

## Disclaimer

This implementation is for educational and research purposes only. Not financial advice. Past performance does not guarantee future results. Trading involves substantial risk of loss.

---

**Last Updated**: January 2026
**Version**: 1.0.0
