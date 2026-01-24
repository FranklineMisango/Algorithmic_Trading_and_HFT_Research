# Dividend-Price Ratio Market Timing Strategy

Implementation of a research-based market timing strategy using the change in the logarithm of the dividend-price ratio to predict S&P 500 monthly returns.

## Overview

This project implements the academic research that found **Δlog(D/P)** (the change in log dividend-price ratio) to be a statistically significant predictor of next-month S&P 500 returns.

### Key Finding from Research

- **R² (In-Sample)**: ~7.8%
- **RMSE (Out-of-Sample)**: ~3.42% monthly
- **Statistical Significance**: p-value ≈ 0
- **Time Horizon**: Monthly predictions (short-term)

### Critical Limitations

This strategy has **significant practical limitations**:

1. **Single-Factor Model**: Uses only one predictor (ignores rates, valuations, macro)
2. **Weak Predictive Power**: R² of 7.8% means 92% of variation is unexplained
3. **High Noise**: 3.4% monthly RMSE is large relative to typical returns
4. **Transaction Costs**: Monthly rebalancing incurs significant costs
5. **Short-Term Only**: Does not validate long-term (5-10 year) predictability

**Educational Purpose**: This implementation serves as a template for rigorous quantitative research methodology, not as a recommended live trading strategy.

## Strategy Mechanics

### Signal Generation

```
Signal_t = log(D/P_t) - log(D/P_t-1) = Δlog(D/P)
```

Where:
- **D/P**: Dividend-Price ratio (trailing 12-month dividends / current price)
- **Δlog(D/P)**: Monthly change in the natural logarithm of D/P

### Model

**Ordinary Least Squares (OLS) Regression**:

```
Return_month_t = α + β * Δlog(D/P)_month_t-1 + ε
```

### Trading Rules

- **Positive Signal** (Δlog(D/P) > 0): D/P increased → Market "cheaper" → Predict higher return → **LONG**
- **Negative Signal** (Δlog(D/P) < 0): D/P decreased → Market "expensive" → Predict lower return → **CASH/REDUCE**

### Implementation Phases

1. **Data Acquisition**: S&P 500 prices + dividends (30+ years recommended)
2. **Feature Engineering**: Calculate Δlog(D/P) signal
3. **Model Training**: OLS regression on in-sample period (e.g., 1990-2002)
4. **Out-of-Sample Validation**: Test on hold-out period (e.g., 2003-2024)
5. **Backtesting**: Simulate trading with transaction costs
6. **Performance Analysis**: Compare to buy-and-hold benchmark

## Project Structure

```
DP_Ratio_Market_Timing/
├── config.yaml                    # Configuration parameters
├── requirements.txt               # Python dependencies
├── data_acquisition.py            # Download S&P 500 + dividend data
├── feature_engineering.py         # Calculate Δlog(D/P) signal
├── ols_model.py                   # OLS regression with stats
├── trading_strategy.py            # Backtesting engine
├── visualization.py               # Performance charts
├── main.py                        # End-to-end pipeline
├── notebooks/
│   └── complete_analysis.ipynb    # Interactive walkthrough
├── results/                       # Output directory
│   ├── backtest_results.csv
│   ├── performance_metrics.json
│   └── *.png (charts)
└── README.md                      # This file
```

## Installation

### Prerequisites

- Python 3.8+
- pip

### Setup

```bash
# Navigate to directory
cd DP_Ratio_Market_Timing

# Create virtual environment (optional)
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Quick Start - Run Complete Pipeline

```bash
python main.py
```

This will:
1. Download S&P 500 data (1990-present)
2. Calculate D/P ratios and Δlog(D/P) signal
3. Train OLS model on in-sample data
4. Validate on out-of-sample data
5. Backtest trading strategy
6. Generate performance report + visualizations

### Interactive Analysis

```bash
jupyter notebook notebooks/complete_analysis.ipynb
```

Walkthrough includes:
- Data exploration and validation
- Signal analysis and correlation
- Model training with diagnostics
- Out-of-sample testing
- Trading simulation
- Performance visualization

### Run Individual Modules

```bash
# Test data acquisition
python data_acquisition.py

# Test feature engineering
python feature_engineering.py

# Test OLS model
python ols_model.py

# Test trading strategy
python trading_strategy.py
```

## Configuration

Edit `config.yaml` to customize:

### Data Parameters
- **Start/End Dates**: Historical data range
- **Tickers**: S&P 500 symbols (fallback options)

### Model Parameters
- **In-Sample Period**: Training data (default: up to 2002)
- **Out-Sample Period**: Test data (default: 2003+)

### Strategy Parameters
- **Signal Threshold**: Minimum signal strength to trade
- **Position Sizing**: Long allocation (0-1)

### Transaction Costs
- **Commission**: Per-trade cost (default: 0.1%)
- **Slippage**: Execution slippage (default: 0.05%)

### Backtest Settings
- **Initial Capital**: Starting portfolio value
- **Risk-Free Rate**: For Sharpe ratio calculation

## Expected Results

### Research Paper Results (1928-2017)

| Metric | In-Sample | Out-Sample |
|--------|-----------|------------|
| R² | 7.8% | N/A |
| RMSE | N/A | 3.42% |
| p-value | ~0 (significant) | N/A |

### Typical Implementation Results (1990-2024)

**Warning**: Results highly dependent on time period and implementation details.

| Metric | Strategy | Buy-Hold |
|--------|----------|----------|
| Annual Return | Variable | ~10% |
| Sharpe Ratio | 0.3 - 0.8 | 0.5 - 0.7 |
| Max Drawdown | 30-50% | 40-60% |
| Transaction Costs | High | Low |

**Key Insight**: The strategy's weak predictive power (R² ~7.8%) makes it difficult to overcome transaction costs in practice.

## Technical Details

### Data Requirements

**Price Data**:
- S&P 500 daily closes
- At least 30 years of history
- Source: yfinance, Alpaca, or similar

**Dividend Data**:
- S&P 500 dividends (monthly or daily)
- Used to calculate trailing 12-month total
- Often embedded in price data feeds

### Statistical Tests Performed

1. **Coefficient Significance**: t-test, p-value
2. **Model Fit**: R², Adjusted R², F-statistic
3. **Residual Diagnostics**:
   - Normality (Jarque-Bera test)
   - Heteroskedasticity (Breusch-Pagan test)
   - Autocorrelation (Durbin-Watson statistic)
4. **Out-of-Sample Validation**: RMSE, MAE, directional accuracy

### Performance Metrics

**Returns**:
- Total return, annualized return, monthly average
- Alpha (excess vs benchmark)

**Risk**:
- Volatility (annualized std)
- Sharpe ratio, Sortino ratio
- Maximum drawdown, Calmar ratio

**Trading**:
- Win rate, profit factor
- Number of trades, trade frequency

## Critique & Research Limitations

### From the Research Paper

1. **Single-Factor Simplicity**: Deliberately ignores other known predictors
2. **Short-Term Focus**: Only tests monthly predictions, not long-term
3. **Data Mining Risk**: Similar studies found R² from 0.5% to 12% (unstable)
4. **No Transaction Costs**: Academic model assumes frictionless trading

### Practical Implementation Issues

1. **Data Lag**: Dividend data not available in real-time
2. **Noise Dominance**: 92% of returns unexplained by model
3. **Cost Sensitivity**: Monthly rebalancing is expensive
4. **Bubble Periods**: Model fails during prolonged speculative phases
5. **Sample Dependency**: Performance varies greatly by period

## Potential Enhancements

If pursuing this research further:

### Strategic Improvements
- **Long-Term Signal**: Test 5-10 year predictability (more aligned with theory)
- **Multi-Factor Model**: Add momentum, volatility, macro indicators
- **Conditional Trading**: Only trade when signal is very strong
- **Lower Frequency**: Quarterly rebalancing to reduce costs

### Implementation Improvements
- **Real-Time Data**: Use live APIs for current D/P ratios
- **Machine Learning**: Compare OLS to ML models (Random Forest, XGBoost)
- **Regime Detection**: Separate bull/bear market logic
- **Risk Overlay**: Add position sizing based on volatility

### Research Extensions
- **International Markets**: Test on other country indices
- **Factor Analysis**: Decompose returns by Fama-French factors
- **Tail Risk**: Analyze performance during crashes
- **Out-of-Sample Walk-Forward**: Rolling window retraining

## References

### Academic Papers
1. Original Research: "Can Dividend-Price Ratio Predict Stock Return?" (1927-2017 analysis)
2. Campbell, J. Y., & Shiller, R. J. (1988). "The Dividend-Price Ratio and Expectations of Future Dividends and Discount Factors"
3. Fama, E. F., & French, K. R. (1988). "Dividend Yields and Expected Stock Returns"

### Implementation References
- **statsmodels**: OLS regression and diagnostics
- **yfinance**: Financial data acquisition
- **scikit-learn**: Model evaluation metrics

## Development

### Running Tests

```bash
pytest tests/
```

### Code Quality

```bash
# Format
black .

# Lint
flake8 .
```

### Contributing

1. Fork the repository
2. Create feature branch
3. Add tests for new functionality
4. Submit pull request

## License

MIT License - See LICENSE file

## Disclaimer

**FOR EDUCATIONAL AND RESEARCH PURPOSES ONLY**

This code is provided for academic study of quantitative research methodology. It is **NOT** investment advice and **NOT** recommended for live trading.

**Key Risks**:
- Weak predictive power (R² ~7.8%)
- High transaction costs erode returns
- Strategy may underperform buy-and-hold
- Past performance does not predict future results

Always conduct your own research and consult financial advisors before making investment decisions.

## Contact

For questions about the implementation or research methodology, please open an issue on GitHub.

---

**Implementation Date**: January 2026
**Research Period Analyzed**: 1990-2024
**Status**: Educational/Research Only
