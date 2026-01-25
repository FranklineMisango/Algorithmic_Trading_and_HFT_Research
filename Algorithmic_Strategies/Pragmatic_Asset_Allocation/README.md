# Pragmatic Asset Allocation Model

A systematic, rules-based Global Tactical Asset Allocation (GTAA) strategy based on the Quant Radio transcript. The model combines momentum, trend-following, and macroeconomic hedging signals with quarterly tranched rebalancing for practical implementation.

## 🎯 Strategy Overview

**Objective**: Achieve solid risk-adjusted returns with reduced management overhead by selecting top-performing risky assets based on 12-month momentum and upward trends, while employing dynamic hedging and cash allocation during unfavorable economic conditions.

**Key Features**:
- **Momentum Ranking**: Select top 2 risky assets from 12-month total returns
- **Trend Filtering**: Only invest in assets trading above their 12-month SMA
- **Market Health Signals**: Cash allocation when 2+ risky assets are in downtrend
- **Yield Curve Hedging**: Full portfolio cash when yield curve inverts
- **Tranched Rebalancing**: 4-tranche system with 12-month holding periods
- **Quarterly Execution**: Practical for individual investors with tax optimization

## 📊 Assets & Signals

### Risky Assets
- **NASDAQ 100** (QQQ): Large-cap US technology focus
- **MSCI World** (URTH): Global developed markets
- **MSCI Emerging Markets** (EEM): Emerging market exposure

### Hedging Assets
- **10-Year US Treasury** (IEF): Duration hedging
- **Gold** (GLD): Inflation and crisis hedge

### Signal Hierarchy

1. **Momentum Selection** (Step 1)
   - Rank risky assets by 12-month total return
   - Select top 2 performers

2. **Trend Filter** (Step 2)
   - Price > 12-month Simple Moving Average
   - Applied to all selected assets

3. **Market Health** (Step 3)
   - If 2+ risky assets in downtrend → Allocate portion to cash
   - Threshold: 50% cash allocation (configurable)

4. **Yield Curve Crisis Signal** (Step 4)
   - 10Y Treasury yield < 3M T-bill yield → 100% cash
   - Requires persistence for 1 month (robustness)

5. **Hedging Portfolio** (Step 5)
   - 50/50 split: 10Y Treasuries + Gold
   - Only when not in full cash mode

6. **Stop-Loss Protection** (Step 6)
   - 15% stop-loss per position
   - Evaluated at quarterly rebalance

## 🏗️ Implementation Architecture

```
Pragmatic_Asset_Allocation/
├── config.yaml              # Strategy parameters & configuration
├── data_acquisition.py      # Historical data fetching & validation
├── signal_generation.py     # All 6 signal types implementation
├── portfolio_construction.py # Tranche system & position sizing
├── backtester.py           # Performance simulation & benchmarking
├── main.py                 # Pipeline orchestration
├── requirements.txt        # Python dependencies
├── notebooks/              # Analytical notebooks
│   ├── 01_data_exploration.ipynb
│   ├── 02_signal_analysis.ipynb
│   ├── 03_portfolio_construction.ipynb
│   └── 04_backtest_evaluation.ipynb
└── results/                # Output directory
```

## 🚀 Quick Start

### 1. Environment Setup
```bash
cd Pragmatic_Asset_Allocation
pip install -r requirements.txt
```

### 2. Run Full Pipeline
```bash
# Complete strategy execution (1927-2023)
python main.py --mode full

# Custom date range
python main.py --mode full --start-date 2000-01-01 --end-date 2023-12-31
```

### 3. Modular Execution
```bash
# Data acquisition only
python main.py --mode data

# Add signal generation
python main.py --mode signals

# Add portfolio construction
python main.py --mode portfolio

# Full backtest
python main.py --mode backtest

# Analyze existing results
python main.py --mode analyze
```

## 📈 Expected Performance

Based on the Quant Radio transcript backtest (1927-2023):

| Metric | Target | Description |
|--------|--------|-------------|
| **Annual Return** | 10.73% | Long-term capital appreciation |
| **Sharpe Ratio** | 0.93 | Risk-adjusted return measure |
| **Max Drawdown** | 24% | Worst peak-to-trough decline |
| **Win Rate** | ~55% | Percentage of positive quarters |

### Benchmark Comparisons
- **60/40 Portfolio**: Traditional balanced allocation
- **Risky Assets Only**: Equal-weighted risky basket
- **Hedging Assets Only**: Equal-weighted defensive basket

## 🔧 Configuration

Key parameters in `config.yaml`:

```yaml
# Signal Parameters
signals:
  momentum:
    lookback_months: 12
    selection_count: 2

  trend_filter:
    lookback_months: 12
    ma_type: "SMA"

  market_health:
    downtrend_threshold: 2  # 2 out of 3 assets
    cash_allocation_pct: 0.5

  yield_curve:
    persistence_months: 1  # Require inversion persistence

# Portfolio Construction
portfolio:
  tranches: 4  # Quarterly staggered rebalancing
  rebalance_frequency: "quarterly"
  min_holding_period: 12  # months

# Transaction Costs
costs:
  etf_trading_cost_bps: 5  # 5 bps per trade
  slippage_bps: 5          # 5 bps slippage
```

## 📊 Analytical Notebooks

### 1. Data Exploration (`01_data_exploration.ipynb`)
- Asset data quality assessment
- Statistical distributions and correlations
- Liquidity and survivorship bias analysis

### 2. Signal Analysis (`02_signal_analysis.ipynb`)
- Signal generation validation
- Historical signal distributions
- Signal effectiveness testing

### 3. Portfolio Construction (`03_portfolio_construction.ipynb`)
- Tranche system mechanics
- Position sizing algorithms
- Transaction cost impact analysis

### 4. Backtest Evaluation (`04_backtest_evaluation.ipynb`)
- Performance metrics deep-dive
- Benchmark comparison analysis
- Risk decomposition and attribution

## 🎲 Risk Management

### Built-in Safeguards
1. **Trend Filter**: Avoids buying falling knives
2. **Market Health Check**: Reduces exposure in broad downturns
3. **Yield Curve Signal**: Crisis beta protection
4. **Stop-Loss Orders**: Position-level risk control
5. **Tranched Rebalancing**: Smooths market timing risk

### Stress Testing
- **COVID-19 Period**: Feb-Apr 2020 validation
- **Financial Crisis**: Sep 2008-Mar 2009 analysis
- **Tech Bubble**: Mar-Oct 2000 stress testing

### Capacity Considerations
- **Very High Capacity**: Uses liquid global indices
- **No Crowding Risk**: Systematic rule-based approach
- **ETF Implementation**: Institutional-grade liquidity

## 🔬 Research Validation

### Statistical Rigor
- **Walk-Forward Testing**: 5-year rolling windows
- **Out-of-Sample**: Post-2020 holdout period
- **Diebold-Mariano Test**: Benchmark comparison significance
- **Regime Stability**: Sub-period analysis (1970-1990, 1990-2010, 2010-2023)

### Failure Mode Analysis

| Risk | Severity | Mitigation |
|------|----------|------------|
| **Parameter Instability** | High | Volatility-adjusted lookbacks |
| **Hedging Asset Correlation** | High | Dynamic hedge ratios, alternative hedges |
| **Yield Curve False Signals** | Medium | Persistence requirements, partial triggers |
| **Stop-Loss Whipsaw** | Medium | Wider stops, quarterly evaluation only |

## 💻 API Reference

### Core Classes

#### `PragmaticAssetAllocationData`
```python
data_acq = PragmaticAssetAllocationData()
all_data = data_acq.fetch_all_data(start_date, end_date)
```

#### `PragmaticAssetAllocationSignals`
```python
signal_gen = PragmaticAssetAllocationSignals()
signals = signal_gen.generate_all_signals(data_dict)
```

#### `PragmaticAssetAllocationPortfolio`
```python
portfolio = PragmaticAssetAllocationPortfolio()
results = portfolio.run_portfolio_construction(signals, price_data)
```

#### `PragmaticAssetAllocationBacktester`
```python
backtester = PragmaticAssetAllocationBacktester()
results = backtester.run_backtest(signals, price_data, portfolio_results)
```

## 📋 Dependencies

Core requirements:
- `pandas` - Data manipulation
- `numpy` - Numerical computations
- `yfinance` - Financial data acquisition
- `matplotlib`/`seaborn` - Visualization
- `scikit-learn` - Statistical analysis
- `pyyaml` - Configuration management

## 🎯 Usage Examples

### Basic Strategy Run
```python
from main import PragmaticAssetAllocationPipeline

# Initialize and run
pipeline = PragmaticAssetAllocationPipeline()
results = pipeline.run_full_pipeline()

# Quick analysis
pipeline.run_quick_analysis(results['backtest'])
```

### Custom Configuration
```python
# Modify signal parameters
config = yaml.safe_load(open('config.yaml'))
config['signals']['momentum']['lookback_months'] = 6  # Shorter momentum

# Save and run
with open('custom_config.yaml', 'w') as f:
    yaml.dump(config, f)

pipeline = PragmaticAssetAllocationPipeline('custom_config.yaml')
results = pipeline.run_full_pipeline()
```

### Signal Analysis
```python
from signal_generation import PragmaticAssetAllocationSignals

signal_gen = PragmaticAssetAllocationSignals()
signals = signal_gen.generate_all_signals(data_dict)

# Get current signal summary
summary = signal_gen.get_signal_summary(signals)
print(f"Selected assets: {summary['selected_risky_assets']}")
print(f"Cash allocation: {summary['total_cash_pct']:.1%}")
```

## 📈 Performance Monitoring

### Key Metrics to Track
- **Rolling Sharpe Ratio**: 1-year trailing performance
- **Signal Effectiveness**: Win rate by signal type
- **Portfolio Turnover**: Trading activity levels
- **Cash Allocation %**: Defensive positioning

### Alert Triggers
- Sharpe ratio < 0.5 for 3+ months
- Cash allocation > 80% for 2+ quarters
- Single asset > 40% of portfolio
- Yield curve inversion without cash trigger

## 🔗 References

- **Source**: Quant Radio Podcast - "The Pragmatic Asset Allocation Model"
- **Authors**: Wouter Keller and Adam Butler
- **Backtest Period**: 1927-2023
- **Publication**: ReSolve Asset Management research

## 📄 License

This implementation is for educational and research purposes. Commercial use requires appropriate licensing from the original researchers and data providers.

---

**Disclaimer**: This is a complete implementation of the published strategy for research purposes. Past performance does not guarantee future results. Always conduct thorough due diligence and risk assessment before implementing any investment strategy.