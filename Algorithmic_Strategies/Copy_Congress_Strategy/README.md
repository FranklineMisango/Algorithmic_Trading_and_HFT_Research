# Copy Congress Trading Strategy

A systematic equity strategy that replicates U.S. Congressional stock trades using publicly disclosed financial information. The strategy applies inverse volatility weighting and portfolio optimization to Congressional trade signals.

## Overview

This strategy systematically tracks and replicates stock trades made by members of the U.S. Congress based on publicly available financial disclosure forms. Congressional members are required to disclose stock transactions within 30-45 days, creating a delayed but publicly accessible signal for equity investing.

### Key Features

- Aggregates Congressional buy/sell flows over 45-day lookback window
- Inverse volatility weighting for risk-adjusted position sizing
- Weekly rebalancing with 10% position limits
- Committee weighting and bipartisan filtering options
- Comprehensive risk management and capacity analysis

### Performance Targets

- Target Sharpe Ratio: 0.934
- Target Annualized Alpha: 3%
- Maximum Drawdown Limit: 20%
- Estimated Capacity: $50-100M

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

1. Clone or download this directory

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Configure data sources in `config.yaml`:
```yaml
data_sources:
  congressional_trades:
    provider: "quiver_quantitative"  # Or your preferred provider
```

## Quick Start

### Running the Complete Strategy

Execute the full strategy pipeline:

```bash
python main.py
```

This will:
1. Fetch Congressional trade data
2. Generate trading signals
3. Construct portfolios
4. Run backtest simulation
5. Generate performance visualizations

### Using Jupyter Notebooks

For interactive analysis:

1. Data Exploration:
```bash
jupyter notebook 01_data_exploration.ipynb
```

2. Signal Analysis:
```bash
jupyter notebook 02_signal_analysis.ipynb
```

3. Backtest Analysis:
```bash
jupyter notebook 03_backtest_analysis.ipynb
```

4. Risk and Capacity Analysis:
```bash
jupyter notebook 04_risk_capacity_analysis.ipynb
```

## Project Structure

```
Copy_Congress_Strategy/
├── config.yaml                       # Strategy configuration
├── requirements.txt                  # Python dependencies
│
├── data_acquisition.py              # Congressional trade & market data fetching
├── signal_generator.py              # Trading signal generation
├── portfolio_constructor.py         # Portfolio optimization
├── backtester.py                    # Performance simulation
├── main.py                          # Strategy orchestration
│
├── 01_data_exploration.ipynb        # Data analysis notebook
├── 02_signal_analysis.ipynb         # Signal quality analysis
├── 03_backtest_analysis.ipynb       # Performance analysis
├── 04_risk_capacity_analysis.ipynb  # Risk and capacity analysis
│
└── README.md                        # This file
```

## Strategy Methodology

### 1. Data Acquisition

Congressional trade data sources:
- Quiver Quantitative API
- Capitol Trades
- Senate/House disclosure databases

Data fields:
- Politician name and committee
- Transaction date vs filing date
- Ticker symbol
- Transaction type (buy/sell)
- Transaction amount

Market data via yfinance:
- Daily OHLCV data
- 30-day historical volatility
- Market capitalization estimates

### 2. Signal Generation

**Signal Formula:**

```
Signal_i = (Total_Buy_$ - Total_Sell_$) / Total_Volume_$
```

**Enhancements:**
- Committee weighting (Finance/Banking: 1.5x, Technology/Energy: 1.3x)
- Bipartisan agreement bonus (1.5x when both parties agree)
- 45-day lookback aggregation window

### 3. Portfolio Construction

**Inverse Volatility Weighting:**

```
Weight_i = (1/σ_i) / Σ(1/σ_j)
```

Where σ_i is 30-day annualized volatility.

**Constraints:**
- 20-50 holdings per portfolio
- 10% maximum position size
- Weekly rebalancing (Fridays)
- Minimum $1,000 transaction filter

### 4. Risk Management

- 20% portfolio stop-loss
- 15% individual position stops
- Maximum 200% annual turnover
- Survivorship bias control

### 5. Transaction Costs

- Commission: 5 bps
- Slippage: 5-10 bps (size-dependent)
- Impact cost modeling

## Configuration

Key parameters in `config.yaml`:

```yaml
signal:
  lookback_days: 45              # Signal aggregation window
  min_transaction_size: 1000     # Filter small trades

portfolio:
  rebalance_frequency: "W"       # Weekly rebalancing
  min_holdings: 20               # Minimum positions
  max_holdings: 50               # Maximum positions
  max_position_size: 0.10        # 10% position limit

weighting:
  method: "inverse_volatility"   # Weighting scheme
  volatility_lookback: 30        # Volatility calculation window

risk:
  max_drawdown: 0.20            # 20% max drawdown limit
  position_stop_loss: 0.15       # 15% position stops
```

## Data Sources

### Congressional Trade Data

The strategy requires Congressional financial disclosure data. Options:

1. **Quiver Quantitative** (Recommended)
   - API access to historical disclosures
   - Clean, structured data
   - Subscription required

2. **Capitol Trades**
   - Free alternative
   - Manual data collection required

3. **Official Sources**
   - Senate/House disclosure databases
   - Requires web scraping

### Market Data

Market data via yfinance (free):
- Historical prices
- Trading volumes
- Split/dividend adjustments

## Implementation Notes

### Sample Data Mode

The current implementation uses **synthetic sample data** for demonstration. To use real Congressional trade data:

1. Obtain API key from preferred provider
2. Set environment variable:
```bash
export CONGRESS_TRADES_API_KEY="your_key_here"
```

3. Modify `data_acquisition.py`:
   - Replace `fetch_congressional_trades_sample()` with real API integration
   - Implement provider-specific parsing

### Real-Time vs Backtest

Current mode: **Backtest only**

For live trading:
1. Implement daily data refresh
2. Add order execution integration
3. Monitor regulatory compliance
4. Track actual vs expected execution

## Performance Metrics

The backtester calculates:

- **Return Metrics:**
  - Total return
  - Annualized return
  - Monthly/annual returns

- **Risk-Adjusted Metrics:**
  - Sharpe ratio
  - Sortino ratio
  - Calmar ratio
  - Information ratio

- **Risk Metrics:**
  - Volatility
  - Maximum drawdown
  - Value at Risk (VaR)

- **Portfolio Metrics:**
  - Turnover
  - Average holdings
  - Concentration

## Ethical and Regulatory Considerations

### Information Lag

- Congressional trades disclosed 30-45 days after execution
- Strategy operates on **public information only**
- No material non-public information (MNPI) concerns

### Legal Compliance

- Uses publicly available financial disclosures
- No insider trading issues (public data)
- Complies with securities regulations

### Reputational Risk

This strategy may be viewed as controversial:
- Exploits information asymmetry from political insiders
- May be perceived as unfair advantage
- Consider ESG and ethical investment guidelines

**Recommendation:**
- Implement as satellite strategy (<10% of portfolio)
- Maintain transparency with investors
- Regular compliance reviews

### Capacity Constraints

- Estimated capacity: $50-100M
- Limited by Congressional trade volumes
- Turnover constraints for larger portfolios
- Signal deterioration risk if strategy becomes crowded

## Outputs

### Visualizations

Generated in `output/` directory:
- `cumulative_returns.png` - Strategy vs benchmark
- `drawdown.png` - Drawdown analysis
- `rolling_sharpe.png` - Time-varying Sharpe ratio
- `holdings_timeseries.png` - Portfolio composition
- `monthly_returns_heatmap.png` - Calendar returns

### Reports

Console output includes:
- Performance summary
- Benchmark comparison
- Risk metrics
- Portfolio characteristics

## Troubleshooting

### Common Issues

1. **Missing Congressional trade data:**
   - Strategy uses sample data by default
   - Integrate real API for production use

2. **Market data gaps:**
   - yfinance occasionally has missing data
   - Implement fallback data sources
   - Use forward-fill for minor gaps

3. **Memory issues with large datasets:**
   - Reduce date range in config
   - Sample rebalance dates
   - Use incremental processing

4. **Slow execution:**
   - Reduce number of rebalance dates
   - Use parallel processing for signal generation
   - Cache intermediate results

## Extending the Strategy

### Adding Features

1. **Sentiment Analysis:**
   - Incorporate news sentiment
   - Social media signals
   - Committee hearing transcripts

2. **Machine Learning:**
   - Predict which Congressional trades to follow
   - Learn optimal lookback windows
   - Dynamic position sizing

3. **Multi-Asset:**
   - Extend to options trades
   - Corporate bond positions
   - Alternative data sources

4. **Enhanced Weighting:**
   - Factor-based tilts
   - Momentum overlays
   - Quality filters

## References

### Academic Research

- "Political Information Flow and Management Guidance" (Cohen et al., 2020)
- "Do Politicians Trade on Inside Information?" (Ziobrowski et al., 2004)
- "Abnormal Returns from the Common Stock Investments of the US Senate" (Ziobrowski et al., 2011)

### Data Sources

- U.S. Senate Financial Disclosures: https://efdsearch.senate.gov
- U.S. House Financial Disclosures: https://disclosures-clerk.house.gov
- Quiver Quantitative: https://www.quiverquant.com

### Technical Documentation

- yfinance: https://github.com/ranaroussi/yfinance
- pandas: https://pandas.pydata.org
- quantstats: https://github.com/ranaroussi/quantstats

## License

This strategy is for educational and research purposes only. Not financial advice.

## Disclaimer

**IMPORTANT DISCLAIMERS:**

1. **Not Financial Advice:** This strategy is for educational purposes only and does not constitute investment advice.

2. **No Guarantee of Returns:** Past performance does not guarantee future results. The strategy may lose money.

3. **Regulatory Risk:** Congressional disclosure requirements may change, affecting strategy viability.

4. **Ethical Considerations:** Users should evaluate whether this strategy aligns with their ethical investment principles.

5. **Sample Data:** The default implementation uses synthetic data. Real Congressional trade data requires separate data subscription.

6. **Compliance:** Users are responsible for ensuring their use complies with all applicable laws and regulations.

## Contact and Support

For questions or issues:
1. Review documentation and notebooks
2. Check troubleshooting section
3. Verify configuration settings
4. Ensure data sources are properly configured

## Version History

- **v1.0** - Initial implementation with core functionality
  - Congressional trade data acquisition
  - Signal generation with committee/bipartisan weighting
  - Inverse volatility portfolio construction
  - Comprehensive backtesting framework
  - Risk and capacity analysis
  - Full documentation and notebooks
