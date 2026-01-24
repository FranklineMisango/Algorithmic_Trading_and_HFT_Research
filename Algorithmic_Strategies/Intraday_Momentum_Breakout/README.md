# Intraday Momentum Breakout Strategy

**ES & NQ E-mini Futures Momentum Trading System**

A sophisticated intraday momentum strategy that trades breakouts from volatility-based "noise areas" on ES and NQ futures contracts. The strategy employs volatility-targeted position sizing, strict intraday-only execution, and conservative transaction cost modeling.

---

## 📊 Strategy Overview

### Core Concept
The strategy identifies when price breaks out of its recent "noise area" - a volatility-based band representing normal market fluctuation. Breakouts above/below this area signal potential momentum opportunities that are traded intraday only.

### Key Features
- **Noise Area Detection**: 90-day lookback for volatility-based boundaries
- **Volatility Targeting**: 3% daily portfolio volatility with 8x max leverage
- **Intraday Only**: No overnight exposure; all positions closed by 4:00 PM ET
- **Conservative Costs**: 1 tick slippage per side + $4.20 commission per round-trip
- **Portfolio Allocation**: 50% NQ momentum, 25% ES momentum, 25% NQ long-only

### Research Foundation
Based on analysis from "Quant Radio" podcast episode discussing intraday futures momentum. Key insights:
- **90-day lookback** (not 14-day from original paper) provides optimal noise area
- **Slippage is critical**: 0.5-1 tick per side dramatically impacts results
- **ES had 7-year flat period** (2010-2017) requiring walk-forward validation
- **Portfolio allocation** (50/25/25) outperformed equal-weight alternatives

---

## 🚀 Quick Start

### Installation

```bash
# Clone repository
cd Algorithmic_Strategies/Intraday_Momentum_Breakout

# Install dependencies
pip install -r requirements.txt
```

### Run Strategy

```bash
# Run with default config
python main.py

# Custom date range
python main.py --start_date 2023-01-01 --end_date 2023-12-31

# Use cached data
python main.py --use_cached_data
```

### Output
Results saved to `results/` directory:
- `equity_curve.csv` - Portfolio value over time
- `trades.csv` - All executed trades
- `performance_metrics.csv` - Comprehensive metrics
- `performance_visualization.png` - Charts

---

## 📁 Project Structure

```
Intraday_Momentum_Breakout/
│
├── config.yaml                 # Strategy configuration
├── requirements.txt            # Python dependencies
├── main.py                    # End-to-end pipeline
│
├── data_acquisition.py        # Download ES/NQ futures data
├── noise_area.py              # Calculate volatility boundaries
├── signal_generator.py        # Detect breakouts & generate signals
├── position_sizer.py          # Volatility-targeted sizing
├── backtester.py              # Event-driven backtest with costs
├── performance_evaluator.py   # Calculate metrics & visualizations
│
├── data/                      # Downloaded futures data
├── results/                   # Backtest outputs
└── README.md                  # This file
```

---

## ⚙️ Configuration

### Key Parameters

**Noise Area Calculation**
```yaml
noise_area:
  lookback_days: 90              # Historical period for boundary calculation
  method: percentile             # percentile | std_dev | atr
  upper_percentile: 75           # Upper boundary (75th percentile)
  lower_percentile: 25           # Lower boundary (25th percentile)
```

**Position Sizing**
```yaml
position_sizing:
  method: volatility_target
  target_daily_volatility: 3.0   # Target 3% daily portfolio volatility
  max_leverage: 8.0              # Maximum 8x leverage
  min_leverage: 1.0              # Minimum 1x leverage
  volatility_estimation: ewma    # EWMA with 20-day span
```

**Portfolio Allocation**
```yaml
portfolio:
  allocation:
    NQ_momentum: 50              # 50% to NQ momentum
    ES_momentum: 25              # 25% to ES momentum  
    NQ_long_only: 25             # 25% to NQ long-only
  initial_capital: 100000        # $100,000 starting capital
```

**Transaction Costs**
```yaml
transaction_costs:
  commission_per_contract: 4.20  # $4.20 round-trip commission
  slippage_ticks: 1.0            # 1 tick per side (conservative)
  ES:
    tick_value: 12.50            # $12.50 per tick
    tick_size: 0.25              # 0.25 point tick
  NQ:
    tick_value: 5.00             # $5.00 per tick
    tick_size: 0.25              # 0.25 point tick
```

---

## 🔬 Strategy Logic

### 1. Noise Area Calculation

The noise area defines the range of "normal" price fluctuation:

**Percentile Method** (default):
```
Upper Boundary = Price + (75th percentile of 90-day intraday range)
Lower Boundary = Price - (25th percentile of 90-day intraday range)
```

**Alternative Methods**:
- **Standard Deviation**: Mean ± (Std Dev × Multiplier)
- **ATR**: Price ± (ATR × 2)

### 2. Signal Generation

**Entry Conditions** (all must be met):
1. Price breaks above upper boundary (long) or below lower boundary (short)
2. Breakout sustained for confirmation bars (default: 2 bars)
3. Volume exceeds threshold percentile (default: 50th percentile)
4. Within entry window (9:30 AM - 3:00 PM ET)

**Exit Conditions** (any triggers exit):
1. **Momentum Failure**: Price re-enters noise area
2. **Session Close**: 4:00 PM ET automatic exit
3. **Trailing Stop**: Price crosses opposite boundary (optional)

### 3. Position Sizing

**Volatility Targeting Formula**:
```python
contracts = (target_vol × allocation_weight × portfolio_value) / 
            (instrument_vol × contract_value)
```

- **Target Vol**: 3% daily (portfolio level)
- **Instrument Vol**: EWMA with 20-day span
- **Leverage Bounds**: 1x - 8x
- **Max Contracts**: 50 per instrument

### 4. Transaction Costs

**Total Cost per Round-Trip**:
```
Total Cost = Commission + (Entry Slippage + Exit Slippage)
```

**ES Example** (1 contract):
- Commission: $4.20
- Entry slippage: 1 tick × $12.50 = $12.50
- Exit slippage: 1 tick × $12.50 = $12.50
- **Total: $29.20 per round-trip**

**NQ Example** (1 contract):
- Commission: $4.20
- Entry slippage: 1 tick × $5.00 = $5.00
- Exit slippage: 1 tick × $5.00 = $5.00
- **Total: $14.20 per round-trip**

---

## 📈 Performance Metrics

### Primary Metrics

| Metric | Target | Description |
|--------|--------|-------------|
| **Sharpe Ratio** | ≥ 1.0 | Risk-adjusted return (primary optimization target) |
| **Max Drawdown** | < 20% | Largest peak-to-trough decline |
| **Win Rate** | 50-60% | Percentage of profitable trades |
| **Profit Factor** | ≥ 1.5 | Gross profit / gross loss |

### Secondary Metrics
- **Sortino Ratio**: Downside risk-adjusted return
- **Calmar Ratio**: Annual return / max drawdown
- **Expectancy**: Average profit per trade
- **Transaction Costs**: As % of initial capital

---

## ⚠️ Known Limitations & Failure Modes

### 1. Extended Flat Markets
**Issue**: ES had a 7-year flat period (2010-2017) with minimal momentum opportunities.

**Mitigation**:
- Walk-forward optimization to detect regime changes
- VIX-based regime filter (optional)
- Portfolio diversification (NQ long-only provides stability)

### 2. Transaction Cost Sensitivity
**Issue**: Slippage dominates performance. 0.5 tick vs 1.0 tick dramatically changes results.

**Mitigation**:
- Conservative 1-tick slippage assumption
- Volume filter to ensure liquidity
- Limit orders with 1-tick offset (optional)
- Monitor actual fill quality in live trading

### 3. Flash Crashes & Market Dislocations
**Issue**: Extreme volatility causes wider spreads and greater slippage.

**Mitigation**:
- Stress test with 5x slippage multiplier
- Circuit breaker: halt on 5% daily loss
- Maximum drawdown limit: 20%
- Avoid trading during known events (FOMC, NFP)

### 4. Overfitting Risk
**Issue**: Optimized parameters may not generalize to future data.

**Mitigation**:
- Walk-forward analysis (365-day training, 90-day test, 90-day step)
- Limited parameter optimization (only lookback and target vol)
- Out-of-sample testing on multiple regimes
- Conservative parameter choices based on research

### 5. Capacity Constraints
**Issue**: Strategy may not scale to large capital.

**Mitigation**:
- ES/NQ are highly liquid ($100B+ daily volume)
- Max 50 contracts per instrument (~$10M notional)
- Capacity analysis required before scaling

---

## 🧪 Stress Testing Scenarios

### 1. Flash Crash (2010-05-06)
- **Slippage Multiplier**: 5x
- **Expected Impact**: 5-10% portfolio loss
- **Circuit Breaker**: Halt if daily loss > 5%

### 2. COVID-19 Crash (March 2020)
- **Volatility Spike**: 3x normal levels
- **Position Sizing**: Automatically reduces due to vol targeting
- **Expected Behavior**: Smaller positions, more frequent stops

### 3. 7-Year Flat Period (ES 2010-2017)
- **Expected Result**: Minimal momentum signals, poor performance
- **Detection**: Rolling 6-month Sharpe < 0.5
- **Response**: Reduce allocation or pause strategy

---

## 🔄 Walk-Forward Optimization

### Framework
```yaml
backtesting:
  walk_forward:
    enabled: true
    training_days: 365           # 1 year training window
    testing_days: 90             # 3 month test window
    step_days: 90                # Step forward 3 months
```

### Optimized Parameters
1. **lookback_days**: [60, 90, 120] days
2. **target_daily_volatility**: [2.0, 3.0, 4.0] %

### Optimization Metric
**Sharpe Ratio** (primary)
- Minimum threshold: 1.0
- Optimization method: Grid search
- Overfitting protection: Max 2 parameters

---

## 📊 Expected Results

### Historical Performance (2018-2023 Backtest)

| Metric | Value |
|--------|-------|
| Total Return | 45-60% |
| Annualized Return | 8-12% |
| Sharpe Ratio | 1.2-1.8 |
| Max Drawdown | -15% to -25% |
| Win Rate | 52-58% |
| Profit Factor | 1.3-1.7 |
| Total Trades | 150-250/year |
| Transaction Costs | 5-8% of capital |

**Note**: These are estimates based on research parameters. Actual results depend on data quality, execution, and market regime.

---

## 🛠️ Development & Testing

### Run Individual Modules

```bash
# Test noise area calculation
python noise_area.py

# Test signal generation
python signal_generator.py

# Test position sizing
python position_sizer.py

# Test backtesting engine
python backtester.py

# Test performance evaluation
python performance_evaluator.py
```

### Unit Tests
```bash
# Run all tests
pytest tests/

# Run specific test
pytest tests/test_noise_area.py -v
```

---

## 📚 Research References

1. **"Quant Radio" Podcast**: Intraday Futures Momentum Strategy Analysis
   - Key insight: 90-day lookback optimal (not 14-day from original paper)
   - Slippage modeling is critical for realistic results
   - ES 7-year flat period (2010-2017) requires walk-forward validation

2. **Volatility Targeting**: 
   - Kelly Criterion for position sizing
   - Constant volatility exposure reduces leverage during high-vol periods

3. **Transaction Cost Research**:
   - Slippage dominates futures trading costs
   - 0.5-1 tick per side is realistic for ES/NQ
   - Volume filters improve execution quality

---

## ⚡ Live Trading Considerations

### Implementation Checklist
- [ ] Real-time data feed (5-minute bars)
- [ ] Order management system (OMS) integration
- [ ] Actual fill quality monitoring
- [ ] Slippage tracking vs assumptions
- [ ] Latency testing (< 100ms desirable)
- [ ] Circuit breakers operational
- [ ] Position limits enforced
- [ ] Risk dashboard monitoring

### Data Requirements
- **Primary**: Interactive Brokers, CQG, or similar
- **Backup**: CME DataMine, Databento
- **Latency**: < 100ms for signal generation
- **Reliability**: 99.9% uptime during RTH

### Execution
- **Order Type**: Market (or limit with 1-tick offset)
- **Slippage Monitoring**: Track actual vs assumed (1 tick)
- **Partial Fills**: Acceptable, adjust position size
- **Connection Loss**: Flatten all positions immediately

---

## 🔐 Risk Management

### Position Limits
- **Max Contracts**: 50 per instrument
- **Max Notional**: $10M per instrument
- **Max Leverage**: 8x portfolio
- **Max Drawdown**: 20% (halt strategy)

### Daily Limits
- **Max Daily Loss**: 5% or $5,000
- **Max Daily Trades**: 20 per instrument
- **Trading Hours**: 9:30 AM - 4:00 PM ET only
- **No Overnight**: All positions closed by 4:00 PM

### Circuit Breakers
1. **5% Daily Loss**: Flatten all positions, halt new entries
2. **20% Drawdown**: Halt strategy, manual review required
3. **Flash Crash Detection**: Halt if 3% move in 5 minutes
4. **VIX Spike**: Optional halt if VIX > 40

---

## 🐛 Troubleshooting

### Common Issues

**1. "No data downloaded" error**
- yfinance has limited futures data
- Use synthetic data for testing: `python data_acquisition.py`
- For production, use Interactive Brokers API or Databento

**2. "Not enough data for noise area calculation"**
- Ensure minimum 90 days of data
- Check data quality (missing bars, gaps)
- Reduce lookback_days temporarily for testing

**3. "No signals generated"**
- Markets may be in low-volatility regime
- Check noise area boundaries (plot with `noise_area.py`)
- Verify volume filter settings
- Review confirmation bars parameter

**4. "Negative Sharpe ratio"**
- Transaction costs may be too high
- Adjust slippage assumptions
- Check for overfitting (walk-forward test)
- Consider different market regime

---

## 📝 License

This strategy is for educational and research purposes only. Not financial advice.

**Disclaimer**: Past performance does not guarantee future results. Futures trading involves substantial risk of loss.

---

## 👥 Contributing

Contributions welcome! Areas for improvement:
- Alternative noise area methods (Kalman filter, machine learning)
- Regime detection (HMM, change-point detection)
- Execution quality optimization
- Additional instruments (RTY, YM futures)
- Options overlay for downside protection

---

## 📧 Contact

For questions or collaboration:
- Open an issue on GitHub
- Email: [your-email@example.com]

---

**Last Updated**: January 2024  
**Version**: 1.0.0  
**Status**: Research/Development (Not Production Ready)
