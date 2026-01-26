# Quick Start Guide - Foreign Market Lead-Lag ML Strategy

## 🚀 Get Started in 5 Minutes

### Step 1: Install Dependencies (1 minute)
```bash
cd Algorithmic_Strategies/Foreign_Market_Lead_Lag_ML
pip install -r requirements.txt
```

### Step 2: Run the Strategy (3 minutes)
```bash
python main.py
```

That's it! The strategy will:
- ✅ Download S&P 500 and 47 foreign market data
- ✅ Create 188 lagged features
- ✅ Train Lasso models for all stocks
- ✅ Generate predictions and backtest
- ✅ Save results to `results/`

### Step 3: View Results (1 minute)
```bash
# Check performance metrics
cat results/performance_metrics.csv

# View performance chart
open results/performance.png
```

## 📊 Expected Output

```
PERFORMANCE METRICS
============================================================

Returns:
  Total Return:        XX.XX%
  Annual Return:       XX.XX%

Risk:
  Volatility:          XX.XX%
  Max Drawdown:        XX.XX%

Risk-Adjusted:
  Sharpe Ratio:        X.XX
  Sortino Ratio:       X.XX

Trading:
  Win Rate:            XX.XX%
  Avg Turnover:        X.XX
  Avg Long Positions:  XX.X
  Avg Short Positions: XX.X

Benchmark Comparison:
  Benchmark Return:    XX.XX%
  Alpha:               XX.XX%
  Beta:                X.XX
  Information Ratio:   X.XX
============================================================
```

## 🎯 Key Strategy Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| Universe | S&P 500 | Target stocks |
| Predictors | 47 foreign markets | ETFs representing global markets |
| Lags | 1, 2, 3, 4 weeks | Time delays for predictive signals |
| Features | 188 | 47 markets × 4 lags |
| Model | Lasso | Primary ML algorithm |
| Long | Top 5% | Best predicted returns |
| Short | Bottom 5% | Worst predicted returns |
| Rebalance | Daily | Portfolio update frequency |

## 📈 Research Expectations

Based on academic research:
- **Predictive Coverage**: ~24% of stocks show positive R²_OOS
- **Gross Annual Return**: ~14.2% (before costs)
- **Transaction Costs**: Significantly impact net returns (10-20 bps per trade)
- **Predictive Horizon**: 5-8 weeks

## 🔧 Customization

### Change Model Type
Edit `config.yaml`:
```yaml
models:
  primary_model: "random_forest"  # Options: lasso, random_forest, gradient_boosting
```

### Adjust Portfolio
```yaml
portfolio:
  long_percentile: 90   # Top 10% instead of 5%
  short_percentile: 10  # Bottom 10% instead of 5%
```

### Reduce Transaction Costs
```yaml
portfolio:
  rebalance_frequency: "weekly"  # Instead of daily
```

## 📓 Interactive Analysis

For step-by-step exploration:
```bash
jupyter notebook

# Open in order:
# 1. notebooks/01_data_exploration.ipynb
# 2. notebooks/02_feature_engineering.ipynb
# 3. notebooks/03_model_training.ipynb
# 4. notebooks/04_backtest_analysis.ipynb
```

## 🎓 Understanding the Strategy

### How It Works
1. **Foreign markets** (e.g., Japan, Germany, UK) move first
2. **Information diffuses** slowly to US markets
3. **ML models** detect these lead-lag relationships
4. **Predict** which US stocks will move next
5. **Trade** long the winners, short the losers

### Why It Works
- Global supply chains create dependencies
- Media attention to foreign events is lower
- Information incorporation is delayed
- Market inefficiency creates opportunity

### Key Innovation
- **Cross-sectional standardization** prevents volatility bias
- **Walk-forward validation** ensures no look-ahead bias
- **Market-level signals** avoid stock-level noise
- **Lasso regularization** handles high-dimensional features

## ⚠️ Important Warnings

### Transaction Costs
High daily turnover means costs matter A LOT:
- Gross return: ~14.2%
- Net return: Significantly lower after 10-20 bps per trade
- **Mitigation**: Reduce rebalancing frequency

### Model Decay
Predictive power may diminish over time:
- Markets become more efficient
- Information diffusion speeds up
- **Mitigation**: Regular retraining, monitor R²_OOS

### Capacity Constraints
Strategy is capacity-limited:
- High turnover requires low trading costs
- Best suited for institutional investors
- **Mitigation**: Negotiate bulk trading rates

## 🐛 Troubleshooting

### "No module named 'yfinance'"
```bash
pip install yfinance
```

### "Data download failed"
```bash
# Check internet connection
# Try alternative data source in config.yaml
```

### "Out of memory"
```python
# In config.yaml, reduce date range:
data:
  start_date: "2020-01-01"  # Shorter period
```

### "Models training too slow"
```python
# Use fewer stocks for testing:
# Edit main.py to limit stock universe
```

## 📚 Next Steps

1. **Review Results**: Check `results/` directory
2. **Analyze Notebooks**: Understand each component
3. **Optimize Parameters**: Tune for better performance
4. **Deploy to LEAN**: Use `lean_algorithm.py` for QuantConnect
5. **Monitor Live**: Track R²_OOS decay, retrain regularly

## 🤝 Getting Help

- **Documentation**: See `IMPLEMENTATION_SUMMARY.md`
- **Examples**: Check Jupyter notebooks
- **Logs**: Review `strategy.log`
- **Issues**: Open GitHub issue

## 📄 Files Generated

After running `main.py`:
```
results/
├── performance_metrics.csv      # All metrics
├── portfolio_results.csv        # Daily portfolio values
├── predictions.csv              # Daily stock predictions
├── validation_r2_scores.csv     # Model validation scores
└── performance.png              # Performance charts

data/
├── sp500_daily_prices.csv       # S&P 500 price data
├── sp500_daily_returns.csv      # S&P 500 returns
└── foreign_weekly_returns.csv   # Foreign market returns

models/
└── [STOCK]_model.pkl            # Trained models per stock
```

## 🎯 Success Criteria

Your implementation is working correctly if:
- ✅ ~24% of stocks have positive R²_OOS
- ✅ Sharpe ratio > 0.5 (after costs)
- ✅ Long/short legs have ~25 positions each
- ✅ Daily turnover is 0.5-2.0
- ✅ Performance varies across market regimes

## 💡 Pro Tips

1. **Start Small**: Test with 2-3 years of data first
2. **Monitor Costs**: Transaction costs are the #1 killer
3. **Validate Strictly**: Never test on training data
4. **Retrain Often**: Models decay, update monthly
5. **Use Lasso**: Performed best in research

---

**Ready to go?** Run `python main.py` and watch the magic happen! 🚀
