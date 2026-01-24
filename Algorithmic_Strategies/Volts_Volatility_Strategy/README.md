# Volts Volatility-Based Predictive Trading Strategy

A comprehensive implementation of the "Volts" strategy that uses volatility-based Granger causality to identify predictive relationships between stocks and generate trend-following trading signals.

## Strategy Overview

The Volts strategy operates by:

1. **Estimating Historical Volatility** using multiple robust estimators (Parkinson, Garman-Klass, Rogers-Satchell, Yang-Zhang)
2. **Clustering Assets** by volatility using K-means++ to identify low, mid, and high volatility groups
3. **Testing Granger Causality** on mid-volatility stocks to find predictive volatility relationships
4. **Generating Signals** via trend-following on the predictor stock's volatility to trade the target stock
5. **Backtesting** to evaluate performance with realistic transaction costs

### Core Concept

If Stock A's volatility "Granger-causes" Stock B's volatility (A→B):
- **Positive trend** in A's volatility → **BUY** B
- **Negative trend** in A's volatility → **SELL** B

## Project Structure

```
Volts_Volatility_Strategy/
├── config.yaml                    # Strategy configuration
├── requirements.txt               # Python dependencies
├── main.py                        # Main execution script
├── volatility_estimators.py       # Historical volatility calculators
├── volatility_clustering.py       # K-means++ clustering
├── granger_causality.py           # Granger causality tests
├── signal_generator.py            # Trading signal generation
├── backtester.py                  # Backtesting engine
├── README.md                      # This file
└── results/                       # Output directory (created at runtime)
    ├── volatility_clusters.png
    ├── granger_network.png
    ├── signals_*.png
    ├── backtest_results.png
    ├── trading_pairs.csv
    ├── aggregated_metrics.csv
    ├── pair_metrics.csv
    ├── equity_curve.csv
    └── all_trades.csv
```

## Quick Start

### Installation

```bash
# Navigate to strategy directory
cd Volts_Volatility_Strategy

# Install dependencies
pip install -r requirements.txt

# Note: ta-lib may require system-level installation
# Ubuntu/Debian: sudo apt-get install ta-lib
# macOS: brew install ta-lib
```

### Basic Usage

```bash
# Run with default configuration
python main.py

# Run with custom configuration
python main.py --config my_config.yaml
```

### Configuration

Edit `config.yaml` to customize:

- **Asset Universe**: List of tickers to analyze
- **Data Period**: Historical data range
- **Volatility Settings**: Estimator type, rolling window
- **Clustering**: Number of clusters, target cluster
- **Granger Causality**: Lag range, significance level
- **Trading Strategy**: Trend method, position sizing
- **Backtesting**: Out-of-sample period, transaction costs
- **Risk Management**: Stop loss, take profit, max drawdown

## 📈 Strategy Components

### 1. Historical Volatility Estimators

Four robust estimators are implemented:

**Parkinson** (1980):
```
σ² = (1/(4*ln(2))) * ln(High/Low)²
```
- Uses high-low range
- Simple but misses overnight gaps

**Garman-Klass** (1980):
```
σ² = 0.5*ln(High/Low)² - (2*ln(2)-1)*ln(Close/Open)²
```
- Incorporates OHLC
- Assumes log-normal distribution

**Rogers-Satchell** (1991):
```
σ² = ln(High/Close)*ln(High/Open) + ln(Low/Close)*ln(Low/Open)
```
- Accounts for drift
- Good for trending markets

**Yang-Zhang** (2000):
```
σ² = σ²_open + k*σ²_close + (1-k)*σ²_RS
```
- Most robust estimator (recommended)
- Accounts for opening jumps and drift

### 2. Volatility Clustering

Uses K-means++ algorithm to partition assets into three clusters:

- **Low Volatility**: Too stable, insufficient profit potential
- **Mid Volatility**: **Target cluster** - optimal risk/reward balance
- **High Volatility**: Too unpredictable, high risk

### 3. Granger Causality Testing

Tests the null hypothesis:
```
H₀: X's volatility does NOT Granger-cause Y's volatility
```

Using F-test on autoregressive models:
- **Restricted**: Y(t) = alpha + sum(beta_i * Y(t-i)) + epsilon(t)
- **Unrestricted**: Y(t) = alpha + sum(beta_i * Y(t-i)) + sum(gamma_j * X(t-j)) + epsilon(t)

Significant result (p < 0.05) indicates predictive relationship.

### 4. Signal Generation

Multiple trend detection methods supported:

- **SMA Crossover**: Fast SMA > Slow SMA → Uptrend
- **Linear Regression**: Positive slope → Uptrend
- **MACD**: MACD line > Signal line → Uptrend
- **Rate of Change**: Positive ROC → Uptrend

### 5. Backtesting

Comprehensive backtesting with:
- Realistic transaction costs (commission + slippage)
- Position sizing
- Risk management (stop loss, take profit)
- Performance metrics (Sharpe, Sortino, Calmar ratios)

## Performance Metrics

The strategy reports:

### Aggregated Performance
- Total Return ($, %)
- Number of Trades
- Win Rate
- Profit Factor
- Maximum Drawdown
- Sharpe Ratio (risk-adjusted return)
- Sortino Ratio (downside risk)
- Calmar Ratio (return/max drawdown)

### Per-Pair Performance
- Individual returns
- Win/loss statistics
- Risk metrics

## Example Usage

### Python API

```python
from volatility_estimators import calculate_volatility_for_assets
from volatility_clustering import cluster_assets_by_volatility
from granger_causality import identify_trading_pairs
from signal_generator import SignalGenerator
from backtester import VoltBacktester

# 1. Download and prepare data
data_dict = {ticker: yf.download(ticker, ...) for ticker in tickers}

# 2. Calculate volatility
volatility_df = calculate_volatility_for_assets(
    data_dict, 
    estimator='yang_zhang',
    rolling_window=20
)

# 3. Cluster by volatility
clustering, mid_cluster = cluster_assets_by_volatility(
    volatility_df,
    n_clusters=3,
    target_cluster='mid'
)

# 4. Identify trading pairs via Granger causality
trading_pairs, analyzer = identify_trading_pairs(
    volatility_df,
    mid_cluster,
    target_lag=5
)

# 5. Generate signals
signal_gen = SignalGenerator(trend_method='sma_crossover')
signals = signal_gen.generate_signals_for_all_pairs(volatility_df, trading_pairs)

# 6. Backtest
backtester = VoltBacktester(initial_capital_per_pair=1000)
results = backtester.run_backtest(data_dict, signals)
backtester.print_results()
backtester.plot_results()
```

## Research Notes

### Original Paper Results

The strategy was tested on 9 large-cap US tech stocks (MSFT, GOOGL, NVDA, AMZN, META, QCOM, IBM, INTC, MU) over ~40 trading days (April-June 2023):

| Pair (X→Y) | Trades | Win Rate | Return | Max DD |
|------------|--------|----------|--------|--------|
| AMZN→META  | 13     | 61.5%    | ~$90   | 3.8%   |
| META→QCOM  | 6      | 66.7%    | ~$74   | 2.1%   |
| MU→QCOM    | 7      | ~57%     | ~$67   | 1.9%   |

**Total Return**: $231.77 (7.725% on $3,000)

### Important Caveats

**Critical Limitations**:

1. **Short Backtest**: 40 days is insufficient for statistical validation
2. **Non-Stationarity**: Granger causality relationships change over time
3. **Limited Assets**: Tested only on 9 tech stocks
4. **Overfitting Risk**: Specific lag and parameters may be sample-specific
5. **Regime Dependency**: Performance may not generalize across market conditions

### Recommended Enhancements

- **Longer Backtests**: Test across multiple years and market regimes
- **Rolling Re-estimation**: Update Granger causality relationships periodically
- **Expanded Universe**: Test on different sectors, asset classes (crypto, FX)
- **Alternative Strategies**: Combine with mean reversion for mid-vol assets
- **Ensemble Methods**: Use multiple volatility estimators in consensus
- **Data Fusion**: Incorporate news sentiment, fundamental data
- **Higher Frequency**: Adapt for intraday trading

## Advanced Configuration

### Custom Volatility Estimator

```python
from volatility_estimators import VolatilityEstimator

estimator = VolatilityEstimator(annualization_factor=252)
all_vols = estimator.calculate_all(df, rolling_window=20)
```

### Custom Trend Detection

```python
from signal_generator import SignalGenerator

signal_gen = SignalGenerator(
    trend_method='macd',
    trend_params={'fast': 12, 'slow': 26, 'signal': 9}
)
```

### Custom Risk Management

```yaml
risk:
  max_drawdown: 10.0
  stop_loss: 5.0
  take_profit: 15.0
  max_hold_time: 30
```

## References

### Volatility Estimators
- Parkinson, M. (1980). "The Extreme Value Method for Estimating the Variance of the Rate of Return"
- Garman, M. B., & Klass, M. J. (1980). "On the Estimation of Security Price Volatilities from Historical Data"
- Rogers, L. C. G., & Satchell, S. E. (1991). "Estimating Variance from High, Low and Closing Prices"
- Yang, D., & Zhang, Q. (2000). "Drift-Independent Volatility Estimation Based on High, Low, Open, and Close Prices"

### Granger Causality
- Granger, C. W. J. (1969). "Investigating Causal Relations by Econometric Models and Cross-spectral Methods"
- Hamilton, J. D. (1994). "Time Series Analysis"

## License

This implementation is for research and educational purposes. See LICENSE file for details.

## Contributing

Contributions are welcome! Areas for improvement:

- Additional volatility estimators
- Alternative clustering methods
- Machine learning-based trend detection
- Portfolio optimization
- Real-time trading integration
- Multi-asset pair strategies

## Disclaimer

**This software is for educational and research purposes only. It is not financial advice. Trading involves substantial risk of loss. Past performance does not guarantee future results. Always conduct thorough research and consult with financial professionals before making investment decisions.**

## Contact

For questions, issues, or contributions, please open an issue on the repository.

---

**Built for quantitative finance research**
