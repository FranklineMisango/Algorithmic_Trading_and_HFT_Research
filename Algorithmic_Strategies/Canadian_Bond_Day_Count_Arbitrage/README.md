# Canadian Bond Day Count Arbitrage Strategy

## Strategy Overview

**Strategy Family**: Fixed-Income Calendar Arbitrage / Micro-Structure Exploit

**Objective**: Exploit the pricing discrepancy in Canadian Government bonds caused by the mismatch between the "Actual/Actual" convention used for quoted price calculation and the "Actual/365" convention used for interest settlement.

## Core Hypothesis

A risk-free arbitrage profit exists because the market's quoted bond price (using Actual/Actual day count) does not perfectly align with the accrued interest paid upon settlement (using Actual/365), creating a mispricing window specifically when coupon periods are 181 or 182 days.

## Economic Rationale

This is a structural inefficiency embedded in Canadian bond market conventions. It is not widely exploited because it is a niche, mechanical detail overlooked by most investors. The opportunity arises from a "deeply buried" quirk in how Canadian bonds calculate accrued interest versus settlement cash flows.

## Signal Definition

### Entry Criteria
1. **Universe**: Canadian Government bonds only
2. **Coupon Period Filter**: Identify bonds where the current coupon period is exactly 181 or 182 days
3. **Timing**: Enter position 1-5 days before the coupon payment date
4. **Signal Strength**: Calculate the theoretical mispricing between the two day count conventions

### Mathematical Framework

- **Accrued Interest (Quoted)** = (Coupon Amount) × (Days since last coupon / CPL)
- **Accrued Interest (Paid)** = Based on Actual/365 convention
- **Arbitrage Profit** = Difference between conventions at settlement

### Exit Criteria
- Exit immediately after receiving the coupon payment
- Duration hedge is unwound simultaneously

## Implementation Requirements

### Data Requirements
- **Bond Identifiers**: ISIN, CUSIP
- **Coupon Data**: Rate, frequency, payment dates
- **Pricing**: Daily dirty price, clean price, yield
- **Duration Metrics**: Modified duration, PV01
- **Frequency**: Daily EOD minimum, intraday preferred for execution

### Data Sources
- Bloomberg (YCNS for Canadian Government bond curves)
- Refinitiv Eikon
- Bank of Canada direct data feeds

### Key Features
1. `DaysToNextCoupon`: Calendar days until next payment
2. `CouponPeriodLength`: Days between last and next coupon
3. `AccruedInterestDifference`: Theoretical arbitrage profit
4. `DurationMatchedHedge`: PV01-matched offsetting position

## Risk Management

### Primary Risks
1. **Execution/Timing Risk** (HIGH): Missing the precise entry/exit window
2. **Transaction Cost Erosion** (HIGH): 0.05%+ fees can eliminate profits
3. **Hedge Slippage** (MEDIUM): Imperfect duration matching
4. **Strategy Crowding** (MEDIUM): Published research may reduce alpha

### Risk Controls
- Maximum single-trade allocation: 1-2% of capital
- Kill switch if hedge slippage > 10% of target profit
- Strict limit orders with no market orders
- Pre-arranged financing and settlement infrastructure

## Backtest Specifications

### Test Period
- Minimum: 2023 calendar year (28 Canadian Government bonds)
- Recommended: 2020-2024 for robustness

### Assumptions
- **Transaction Costs**: 0.05% per trade
- **Slippage**: 2-5 basis points
- **Execution Delay**: Same-day settlement (T+1 awareness critical)
- **Survivorship Bias**: Point-in-time universe only

### Validation Methodology
1. **Duration Matching**: Hedge must neutralize all interest rate risk
2. **Control Test**: Run strategy on non-181/182 day periods (profits should disappear)
3. **Statistical Test**: t-test on returns between target periods vs control periods

## Expected Performance

### Target Metrics
- **Win Rate**: >90% (excluding failed executions)
- **Average Profit/Trade**: 2-5 basis points (after costs)
- **Sharpe Ratio**: High (if truly risk-free), degraded by execution friction
- **Capacity**: Extremely low (specific bonds, specific days only)

### Failure Modes (Ranked by Risk)

| Rank | Failure Mode | Severity | Mitigation |
|------|-------------|----------|------------|
| 1 | Execution Slippage & Timing Miss | High | Automate trading, use DMA, pre-arrange financing |
| 2 | Transaction Cost Erosion | High | Institutional rates, conservative cost modeling |
| 3 | Incorrect Duration Hedge | Medium | PV01 matching, daily rebalance |
| 4 | Regulatory/Convention Change | Low | Monitor BoC and IIROC announcements |
| 5 | Strategy Crowding | Medium | Track profit margin decay over time |

## Files in This Strategy

- `config.yaml`: Strategy parameters and bond universe configuration
- `data_acquisition.py`: Canadian bond data fetching from Bloomberg/BoC
- `feature_engineering.py`: Day count calculations and signal generation
- `backtester.py`: Full strategy implementation with duration hedging
- `signal_generator.py`: Real-time opportunity scanner
- `notebooks/01_data_exploration.ipynb`: Bond universe analysis
- `notebooks/02_day_count_analysis.ipynb`: Convention mismatch quantification
- `notebooks/03_backtest_results.ipynb`: Performance validation
- `notebooks/04_live_monitoring.ipynb`: Real-time signal tracking

## Usage

```python
from backtester import CanadianBondArbitrageBacktester
from config import load_config

# Load configuration
config = load_config('config.yaml')

# Initialize backtester
bt = CanadianBondArbitrageBacktester(config)

# Run backtest
results = bt.run(start_date='2023-01-01', end_date='2023-12-31')

# Analyze results
print(f"Total Trades: {results['num_trades']}")
print(f"Win Rate: {results['win_rate']:.2%}")
print(f"Avg Profit (bps): {results['avg_profit_bps']:.2f}")
print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
```

## Warnings

⚠️ **NOT A GET-RICH-QUICK SCHEME**: This strategy requires:
- Sophisticated execution infrastructure
- Institutional-level transaction costs
- Precise timing and automation
- Very specific skill set in fixed income markets

⚠️ **CAPACITY CONSTRAINTS**: Extremely limited opportunity set (specific bonds on specific days only)

⚠️ **PUBLICATION RISK**: Strategy is now public knowledge, crowding may reduce/eliminate alpha

## References

Based on "Quant Radio: Arbitrage Opportunities in the Canadian Bond Markets"

Study: Real market data from 28 Canadian government bonds throughout 2023

## License

Same as parent repository
