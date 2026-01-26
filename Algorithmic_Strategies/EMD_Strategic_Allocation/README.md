# Emerging Markets Debt (EMD) Strategic Allocation Strategy

## Overview
Systematic multi-asset portfolio strategy allocating 5-15% to Emerging Markets Debt (local + hard currency) to capture diversification benefits and yield premium while managing currency and credit risk.

**Source**: Quant Radio - "Is Emerging Markets Debt Right for Your Portfolio?"

## Strategy Hypothesis
Adding EMD (particularly local currency government bonds) improves risk-adjusted returns due to:
1. **Low correlation** with developed market bonds (doesn't move in lockstep with US Treasuries)
2. **Valuation opportunity**: EM currencies at 2003-like PPP levels (historically preceded strong returns)
3. **Yield premium**: Higher yields compensate for perceived risks
4. **"Original Sin" reversal**: Local currency market grew from $100B (2002) to $5T (2023)

## Key Signals

### 1. PPP Valuation Z-Score
- **Formula**: `Z = (PPP_current - PPP_mean_10y) / PPP_std_10y`
- **Rule**: Long currencies where Z < -0.5 (undervalued)
- **Rationale**: "Currency valuations can have a big impact on returns" using PPP deflator

### 2. Yield Spread vs Developed Markets
- **Formula**: `Spread = EM_Yield - US_Treasury_Yield`
- **Rule**: Allocate to top 60th percentile spreads
- **Rationale**: "Higher yields compensate investors for perceived risks"

### 3. Diversified Index Construction
- **Rule**: 5% maximum country weight cap
- **Rationale**: "Diversified indices... put limits... difference vs broad index is huge over 50%"

## Data Requirements

| Data Type | Source | Frequency |
|-----------|--------|-----------|
| Bond Yields (Local/Hard) | Bloomberg, JP Morgan Indices | Daily/Weekly |
| FX Rates | Bloomberg, Refinitiv | Daily |
| PPP Factors | IMF, World Bank | Monthly |
| Credit Ratings | S&P, Moody's | Event-driven |
| Commodity Prices | Bloomberg BCOM | Daily |

## Performance Targets

| Metric | Target | Rationale |
|--------|--------|-----------|
| Sharpe Improvement | >0.15 | vs 60/40 benchmark (OOS) |
| Max Drawdown Increase | <2% | Risk-adjusted diversification |
| Correlation with 60/40 | <0.4 | True diversification benefit |

## Implementation

### Portfolio Construction
- **EMD Allocation**: 10% of total portfolio
- **Split**: 60% local currency / 40% hard currency
- **Rebalancing**: Quarterly (63 trading days)
- **Leverage**: None (strategic allocation)

### Transaction Costs
- Local currency bonds: 30 bps
- Hard currency bonds: 25 bps
- Slippage: 10 bps for large orders

### ETF Proxies (for backtesting)
- **EMLC**: VanEck EM Local Currency Bond ETF
- **EMB**: iShares JP Morgan USD EM Bond ETF
- **SPY**: S&P 500 (60/40 benchmark)
- **AGG**: US Aggregate Bonds (60/40 benchmark)

## Risk Management

### Risk Rules
1. **Country Limit**: Max 5% per country
2. **Credit Floor**: Minimum B- rating (S&P scale)
3. **FX Hedging**: Hedge 50% if 1M realized vol >20% annualized
4. **VIX Trigger**: Reduce EMD allocation 50% if VIX >35

### Monitoring Signals
- **Correlation Alert**: 90-day rolling correlation >0.6 with equities
- **Spread Compression**: Average spread <300 bps
- **Commodity Beta**: Track exposure to commodity-driven economies

### Stress Tests
1. **Taper Tantrum (2013)**: +100 bps US 10Y yield + 10% USD strength
2. **COVID Crash (2020)**: -30% equities + -50% oil

## Failure Modes & Mitigations

| Rank | Risk | Severity | Mitigation |
|------|------|----------|------------|
| 1 | USD Strengthening Shock | High | Dynamic hedging rule + maintain hard currency allocation |
| 2 | Synchronized Risk-Off | High | VIX trigger reduces allocation; flight-to-quality rule |
| 3 | Idiosyncratic Default | Medium | Credit rating floor (B-); diversified/capped index |
| 4 | Commodity Price Collapse | Medium | Underweight high commodity-beta countries |
| 5 | Valuation Signal Degradation | Low | Conditional signals; reduce allocation if spreads compress |

## Files Structure

```
EMD_Strategic_Allocation/
├── config.yaml              # Strategy parameters
├── signals.py               # Signal calculation logic
├── risk_manager.py          # Risk rules and monitoring
├── requirements.txt         # Python dependencies
├── README.md               # This file
├── lean/
│   └── main.py             # LEAN algorithm
└── notebooks/
    └── emd_backtest.ipynb  # Analysis notebook
```

## Quick Start

### 1. Install Dependencies
```bash
cd EMD_Strategic_Allocation
pip install -r requirements.txt
```

### 2. Run Jupyter Notebook
```bash
jupyter notebook notebooks/emd_backtest.ipynb
```

### 3. Run LEAN Backtest
```bash
cd lean
lean backtest main.py
```

## Validation Procedure

### In-Sample (2003-2012)
- Period where "benefit of including emerging debt was much bigger"
- Used for signal calibration

### Out-of-Sample (2013-2023)
- "Tougher environment" post-taper tantrum
- True test of strategy robustness
- **Selection Criteria**: Strategy selected if OOS Sharpe improvement >0.15 without >2% DD increase

## Key Insights from Research

1. **Diversification Value**: Local currency EMD "historically had lower correlation with traditional fixed income assets"

2. **Market Evolution**: Growth from $100B to $5T shows "original sin" reversal - countries can now borrow in local currency

3. **Valuation Opportunity**: "EM currency valuations look similar to 2003" - a period of subsequent strong returns

4. **Index Selection Matters**: "Difference between JPM GBI EM broad and diversified version is huge over 50%"

5. **Risk Awareness**: "If dollar suddenly strengthens, could really hurt unhedged local currency debt"

6. **Hard Currency Trade-off**: "Higher correlation with corporate credit" limits diversification in crises

## Expected Performance (Historical Context)

Based on transcript discussion:
- **2003-2012**: Strong performance period (currencies undervalued)
- **2013+**: "Tougher environment" post-taper tantrum
- **Current**: Valuations "similar to 2003" suggest potential opportunity

## Capacity Analysis
- Strategy capacity limited by EMD ETF liquidity
- Recommend not exceeding 20% of average daily volume per position
- Suitable for portfolios up to ~$100M with ETF implementation

## References
1. Quant Radio: "Is Emerging Markets Debt Right for Your Portfolio?"
2. JP Morgan GBI-EM Global Diversified Index
3. JP Morgan EMBI Global Diversified Index
4. IMF/World Bank PPP data

## Disclaimer
Educational and research purposes only. Not financial advice. Past performance does not guarantee future results. EMD involves substantial risks including currency, credit, and political risks.

## License
MIT License - See repository root LICENSE file
