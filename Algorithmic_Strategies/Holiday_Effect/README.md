# Holiday Effect Trading Strategy

## Overview

Calendar-based seasonal strategy capturing pre-event price momentum in Amazon (AMZN) stock around major shopping holidays.

## Strategy Logic

### Core Thesis
Amazon's stock exhibits statistically significant positive drift in the two weeks preceding Black Friday and Prime Day, driven by anticipated revenue surges and positive investor sentiment.

### Economic Rationale
- Market anticipation of massive consumer spending creates buying pressure
- "Holiday effect" or boost from anticipated holiday spending
- Pre-event run-up before actual sales materialize

### Signal Generation
Pure calendar-based signals:
1. **Black Friday**: Enter 10 trading days before (Friday after 4th Thursday in November)
2. **Prime Day**: Enter 10 trading days before mid-July event
3. **Exit**: Close of last trading day before event

## Implementation

### Files

**Python Modules**
- `data_acquisition.py`: Fetch AMZN and SPY historical data
- `signal_generator.py`: Calendar-based event detection
- `backtester.py`: Event-driven backtest with realistic costs
- `options_strategy.py`: Put-selling overlay strategy
- `main.py`: Complete pipeline orchestration

**Jupyter Notebooks**
- `01_data_exploration.ipynb`: AMZN price history and event analysis
- `02_signal_analysis.ipynb`: Event windows and statistical testing
- `03_backtest_analysis.ipynb`: Equity long strategy performance
- `04_options_strategy.ipynb`: Put selling and combined portfolio

**Configuration**
- `config.yaml`: All strategy parameters
- `requirements.txt`: Python dependencies

**QuantConnect**
- `lean_algorithm.py`: Event-driven production implementation

## Research Results (Paper)

### Equity Long Strategy (1998-2024)
- **Sharpe Ratio**: 0.51 (vs SPY: 0.303)
- **Statistical Significance**: Confirmed via t-tests
- **Pattern**: Consistent pre-event positive drift

### Put Selling Strategy (2012-2024)
- **Win Rate**: 100%
- **Risk**: Small premiums, potential large loss
- **Allocation**: Max 5% of capital

### Combined Strategy
- **Sharpe Ratio**: 0.722
- **Approach**: Blend SPY base + AMZN put selling overlay

## Usage

### Full Pipeline
```bash
python main.py --mode full
```

### Equity Strategy Only
```bash
python main.py --mode equity
```

### Options Strategy
```bash
python main.py --mode options
```

### Lean Backtest
```bash
lean backtest --algorithm-location Holiday_Effect/lean_algorithm.py
```

## Risk Management

### Critical Failure Modes

1. **Effect Decay** (High Severity, Medium Likelihood)
   - Market learns and arbitrages away anomaly
   - Monitor rolling 3-year Sharpe, pause if < 0.2

2. **Company-Specific Shock** (High Severity, Low Likelihood)
   - Negative news during holding period
   - Implement 8% stop-loss, consider long/short hedge

3. **Macro Override** (Medium Severity, Medium Likelihood)
   - Bear market negates seasonal tailwind
   - Only trade when SPY > 200-day MA, VIX < 25

4. **Data Snooping Bias** (Medium Severity, High Likelihood)
   - Strategy discovered on same data used to test
   - Use 1998-2010 discovery, 2011-2018 validation, 2019+ holdout

5. **Options Execution Risk** (Medium Severity, Medium Likelihood)
   - 100% win rate from small premiums; one large move wipes gains
   - Limit options to 5% capital, run scenario analysis

6. **Changing Consumer Behavior** (Low Severity, Medium Likelihood)
   - Shift away from Black Friday shopping
   - Extend to other critical dates (Singles' Day, product launches)

## Parameters

### Event Windows
- **Holding Period**: 10 trading days (2 weeks)
- **Entry**: 10 days before event
- **Exit**: Day before event

### Transaction Costs
- **Slippage**: 5 bps
- **Commission**: 15 bps per trade

### Risk Controls
- **Stop Loss**: 8% trailing
- **Market Filter**: SPY above 200-day MA
- **VIX Filter**: VIX below 25

## Extensions

Potential enhancements mentioned in research:
- Extend to other retailers (WMT, TGT)
- Product launch events (iPhone, gaming consoles)
- International shopping holidays (Singles' Day)
- Earnings announcement windows

## Dependencies

- Python 3.9+
- pandas, numpy
- yfinance (data)
- scipy (statistical tests)
- matplotlib, seaborn (visualization)

## References

- Quant Radio: "Holiday Effect on Amazon Stock"
- Backtest period: 1998-2024
- Statistical testing: t-tests, bootstrapping for significance
