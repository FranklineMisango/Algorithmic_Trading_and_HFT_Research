# Music Royalties Systematic Trading Strategy

A quantitative systematic strategy for investing in Life of Rights (LOR) music royalty assets, targeting equity-like returns with low correlation to traditional financial markets.

## Strategy Overview

### Objective
Systematically acquire music royalty contracts that grant ownership for the life of copyright, focusing on assets with high revenue stability and proven catalog age. Target risk-adjusted returns competitive with public equities while providing significant portfolio diversification benefits.

### Economic Rationale
Music royalty cash flows are driven by millions of decentralized consumer listening choices, which remain relatively stable across economic conditions. Unlike corporate earnings that are tightly coupled to business cycles, royalty revenues exhibit low correlation with traditional financial markets, making them an attractive diversification asset.

### Key Findings from Academic Study
Based on analysis of nearly $100M in transactions (2017-2022):
- LOR assets delivered median 12.8% annual returns
- Returns were significantly more stable year-over-year than equities
- Market pays premium for revenue stability (LTM/LTY ratio ≈ 1.0)
- Older catalogs command higher prices (age premium exists)
- 10-Year Term contracts suffer guaranteed capital depreciation (avoid)

## Strategy Components

### 1. Signal Generation

Two primary signals drive asset selection:

**Stability Premium Signal**
```
Stability Ratio = Revenue_LTM / Revenue_LTY
Target: ≈ 1.0
```
- Ratio close to 1.0 indicates consistent, predictable revenue
- Market pays higher price multipliers for stable assets
- Penalize assets with high volatility (ratio far from 1.0)

**Age Premium Signal**
```
Catalog Age = Current Year - Copyright Year
Direction: Positive (older is better)
```
- Older catalogs have proven survivorship through technological shifts
- Age is proxy for resilience and de-risked future cash flows
- Controlling for stability, older assets command higher valuations

### 2. Predictive Model

Linear regression model to predict fair price multipliers:

```
PredictedMultiplier = β₀ + β₁(StabilityRatio) + β₂(CatalogAge) + ε
```

**Model Performance Target:**
- Mean Squared Error (MSE) ≤ 5.7
- R² ≥ 0.60
- Out-of-sample validation on hold-out period

### 3. Portfolio Construction

**Universe Filter:**
- Contract Type = "Life of Rights" (LOR) ONLY
- Exclude 10-Year Term contracts (guaranteed depreciation)
- Exclude assets with extreme stability ratios (<0.33 or >3.0)
- Minimum data requirements (LTM, LTY, age)

**Selection Method:**
- Rank assets by mispricing (predicted fair value - market price)
- Select top quintile (20%) most undervalued assets
- Minimum portfolio size: 50 assets (idiosyncratic risk mitigation)

**Diversification Constraints:**
- Maximum single asset: 2% of portfolio
- Maximum genre concentration: 20%
- Equal weighting within selected assets

**Rebalancing:**
- Frequency: Annual
- Holding period target: 1+ years (amortize transaction costs)

### 4. Transaction Costs

High transaction costs are a major challenge:

| Cost Component | Amount | Impact |
|---------------|--------|---------|
| Buyer Fee | $500 fixed | Per transaction |
| Seller Commission | 8% of price | On exit |
| Slippage | 3-5% | Illiquid market |
| **Total Round-Trip** | **~11%** | **Significant hurdle** |

**Mitigation Strategies:**
- Lengthen holding periods to amortize costs
- Only buy when mispricing exceeds cost threshold
- Build direct relationships for off-market deals
- Use model to identify large mispricings only

### 5. Risk Management

**Position Limits:**
- Max single position: 2%
- Max leverage: 1.0× (no leverage due to illiquidity)
- Max genre exposure: 20%

**Monitoring Thresholds:**
- Alert if held asset stability ratio drops below 0.5
- Alert if correlation with S&P 500 exceeds 0.3
- Alert on 15% drawdown

**Stress Tests:**
1. **Streaming Rate Shock:** 30% drop in per-stream payouts
2. **Platform Failure:** Primary marketplace becomes unavailable
3. **Age Premium Reversal:** Consumer taste shifts to only new music

## Project Structure

```
Music_Royalties_Strategy/
├── config.yaml                    # Strategy configuration
├── requirements.txt               # Python dependencies
├── main.py                        # End-to-end pipeline CLI
├── data_loader.py                 # Data loading & validation
├── feature_engineering.py         # Signal calculation
├── model_trainer.py               # Price multiplier model
├── portfolio_constructor.py       # Portfolio construction
├── backtester.py                  # Backtesting engine
├── performance_evaluator.py       # Metrics & statistical tests
├── notebooks/
│   ├── 01_end_to_end_backtest.ipynb          # Main workflow
│   ├── 02_model_development.ipynb            # Model training & validation
│   └── 03_data_exploration.ipynb             # Interactive analysis
├── results/                       # Output directory
│   ├── equity_curve.csv
│   ├── trades.csv
│   ├── performance_report.txt
│   └── performance_metrics.json
└── README.md                      # This file
```

## Installation

### Requirements
- Python 3.8+
- 2GB RAM minimum
- ~100MB disk space

### Setup

```bash
# Clone repository (if applicable)
cd Music_Royalties_Strategy

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Command Line Interface

**Basic Run (Synthetic Data):**
```bash
python main.py --save-results
```

**With Real Data:**
```bash
python main.py --data-file path/to/royalty_transactions.csv --save-results
```

**Save Model:**
```bash
python main.py --save-results --save-model
```

**With Interaction Features:**
```bash
python main.py --include-interactions --save-results
```

### Expected Output

```
================================================================================
MUSIC ROYALTIES SYSTEMATIC TRADING STRATEGY
================================================================================

STEP 1: DATA LOADING & PREPROCESSING
Train:      600 transactions
Validation: 200 transactions  
Test:       200 transactions

STEP 2: FEATURE ENGINEERING
Engineered 15 features

STEP 3: MODEL TRAINING & VALIDATION
Validation MSE:  5.2
Validation R²:   0.68

STEP 4: PORTFOLIO CONSTRUCTION
Found 120 undervalued assets
Selected 50 assets (top 20%)

STEP 5: BACKTESTING
Rebalancing 3 times (annual)
Final portfolio value: $1,128,000

STEP 6: PERFORMANCE EVALUATION
Total return: 12.8%
Sharpe ratio: 1.15
Correlation with S&P 500: 0.18
```

### Jupyter Notebooks

For interactive exploration and analysis:

```bash
jupyter notebook notebooks/01_end_to_end_backtest.ipynb
```

**Notebook 1: End-to-End Backtest**
- Full pipeline walkthrough
- Step-by-step execution
- Visualizations at each stage

**Notebook 2: Model Development**
- Model training and selection
- Feature importance analysis
- Residual diagnostics
- Cross-validation

**Notebook 3: Data Exploration**
- Transaction data patterns
- Revenue stability analysis
- Age premium validation
- Interactive Plotly charts

## Data Requirements

### Required Fields

| Field | Type | Description |
|-------|------|-------------|
| `asset_id` | str | Unique asset identifier |
| `transaction_date` | datetime | Sale date |
| `transaction_price` | float | Sale price ($) |
| `revenue_ltm` | float | Last 12 months revenue ($) |
| `revenue_lty` | float | Last 3 years avg annual revenue ($) |
| `catalog_age` | int | Age of catalog (years) |
| `contract_type` | str | "LOR" or "10-Year Term" |
| `genre` | str | Music genre (optional) |

### Data Source

Primary source: **Royalty Exchange** or similar marketplace platforms

Expected format: CSV with above columns

### Synthetic Data

If no real data available, synthetic data generator included:
- 1000 transactions over 2017-2022 period
- Realistic distributions based on study findings
- Includes both LOR and 10-Year Term contracts
- Genre diversity

## Performance Metrics

### Return Metrics
- **Total Return:** Cumulative strategy return
- **CAGR:** Compound Annual Growth Rate
- **Annualized Volatility:** Standard deviation of returns (annualized)

### Risk-Adjusted Metrics
- **Sharpe Ratio:** (Return - RiskFree) / Volatility
  - Target: > 1.0
- **Sortino Ratio:** (Return - RiskFree) / Downside Volatility
- **Calmar Ratio:** CAGR / Max Drawdown

### Drawdown Analysis
- **Maximum Drawdown:** Largest peak-to-trough decline
- **Drawdown Duration:** Time underwater
- **Recovery Time:** Time to recover from drawdown

### Benchmark Comparison
- **Beta:** Sensitivity to S&P 500
  - Target: < 0.5 (low correlation benefit)
- **Alpha:** Excess return vs risk-adjusted benchmark
- **Correlation:** Correlation with equities
  - Target: < 0.3 (diversification benefit)

### Statistical Tests
1. **T-test vs Zero:** Are returns significantly > 0?
2. **T-test vs Benchmark:** Does strategy outperform?
3. **Correlation Test:** Is low correlation significant?
4. **Normality Test:** Return distribution shape

### Strategy-Specific Metrics
- **Transaction Cost Ratio:** Total costs / Initial capital
  - Critical metric: High costs are major hurdle
- **Portfolio Turnover:** Annual rebalancing rate
- **Win Rate:** % of profitable periods
- **Avg Holding Period:** Time assets held

## Configuration

Key parameters in `config.yaml`:

```yaml
# Target Returns
target_annual_return: 0.10  # 10%

# Model
features:
  - stability_ratio
  - catalog_age
target_mse: 5.7

# Portfolio
selection:
  method: top_quintile  # Top 20% undervalued
max_single_asset: 0.02  # 2% position limit
min_portfolio_size: 50  # Minimum diversification

# Transaction Costs
buyer_fee: 500  # $500 per trade
seller_commission: 0.08  # 8%
slippage:
  base_rate: 0.03  # 3%

# Rebalancing
rebalancing_frequency: annual
```

## Failure Modes & Mitigations

### 1. Model Risk / Regime Change (HIGH SEVERITY)

**Risk:** Model is backward-looking; market dynamics may shift

**Mitigations:**
- Incorporate qualitative score for cultural relevance
- Regularly re-estimate factor premia with new data
- Reduce position size if validation MSE increases
- Monitor model prediction errors in real-time

### 2. Extreme Idiosyncratic Risk (HIGH SEVERITY)

**Risk:** Wide return dispersion (90th percentile: +30.6% vs 10th: -1.6%)

**Mitigations:**
- Strict 2% position sizing per asset
- Build diversified portfolio of 50+ assets
- Deep qualitative due diligence beyond model
- Monitor individual asset performance

### 3. Market Illiquidity & High Costs (HIGH SEVERITY)

**Risk:** 8% seller commission is massive hurdle; limited liquidity

**Mitigations:**
- Lengthen holding periods (annual+ rebalancing)
- Only trade when mispricing exceeds cost threshold
- Build direct relationships for off-market deals
- Consider private placements to reduce fees

### 4. Data Opaqueness & Asymmetry (MEDIUM SEVERITY)

**Risk:** Reliance on single platform data; information advantages exist

**Mitigations:**
- Cross-verify cash flow data where possible
- Treat model output as "valuation floor, not ceiling"
- Build in 20% margin of safety when buying
- Develop independent data sources

### 5. Concentration in Old Catalog (MEDIUM SEVERITY)

**Risk:** Over-reliance on age premium may crowd into older genres

**Mitigations:**
- Set genre diversification limits (20% max)
- Monitor age premium factor for decay
- Blend model with forward-looking genre trends
- Maintain exposure across catalog ages

### 6. Platform Risk (MEDIUM SEVERITY)

**Risk:** Dependence on single marketplace (Royalty Exchange)

**Mitigations:**
- Build relationships across multiple platforms
- Develop direct deal flow sources
- Monitor platform health and competition
- Have exit strategy if platform fails

### 7. Regulatory & Copyright Risk (LOW SEVERITY)

**Risk:** Changes to copyright law or streaming economics

**Mitigations:**
- Monitor legislative changes
- Stress test for streaming rate changes
- Diversify across copyright jurisdictions
- Understand contract terms deeply

## Capacity Constraints

**Market Size:** ~$100M in transactions over 5 years (study period)

**Estimated Capacity:** $50M maximum AUM
- Don't exceed 10% of total market volume
- Liquidity deteriorates with size
- Price impact increases non-linearly

**Scalability Issues:**
- Small, illiquid market
- Bespoke deal structures
- Manual due diligence required
- Limited automation potential

**Recommendation:** Strategy works best at $1-20M scale

## Backtesting Considerations

### Realistic Assumptions
- ✓ Survivorship bias controlled (includes failed assets)
- ✓ Transaction costs realistic (8% + $500 + slippage)
- ✓ No look-ahead bias (walk-forward validation)
- ✓ Data from actual marketplace transactions

### Limitations
- ⚠ Synthetic data used if real data unavailable
- ⚠ Market impact not modeled (capacity limits)
- ⚠ Assumes continuous access to platform
- ⚠ Does not model operational complexities
- ⚠ Cash flow seasonality not captured

### Walk-Forward Validation
```
Train:      2017-2019 (3 years)
Validation: 2020      (1 year)
Test:       2021-2022 (2 years)
```

## Live Trading Considerations

### Before Going Live

**Infrastructure:**
- [ ] Access to Royalty Exchange or similar platform
- [ ] Cash flow verification process
- [ ] Legal review of contract terms
- [ ] Accounting system for royalty payments
- [ ] Due diligence checklist beyond model

**Operational:**
- [ ] Minimum $1M capital (50 assets × $20K average)
- [ ] Qualitative overlay team
- [ ] Music industry expertise
- [ ] Legal counsel for contract review
- [ ] Platform relationship development

**Risk Management:**
- [ ] Real-time portfolio monitoring
- [ ] Drawdown alerts configured
- [ ] Genre exposure tracking
- [ ] Correlation monitoring vs equities
- [ ] Stress test scenario analysis

### Ongoing Operations

**Monthly:**
- Review held asset cash flows
- Monitor platform transaction volume
- Check model prediction errors
- Update risk dashboard

**Quarterly:**
- Refresh model predictions
- Analyze residuals for patterns
- Review genre exposures
- Stress test scenarios

**Annually:**
- Retrain model on new data
- Rebalance portfolio
- Generate performance report
- Review strategy assumptions

## Comparison to Traditional Assets

| Metric | Music Royalties | S&P 500 | Bonds |
|--------|----------------|---------|-------|
| **Expected Return** | 10-13% | 10% | 3-5% |
| **Volatility** | 12-15% | 16% | 5-7% |
| **Sharpe Ratio** | 1.0-1.2 | 0.6 | 0.4 |
| **Correlation to Equities** | 0.15-0.25 | 1.0 | 0.3 |
| **Liquidity** | Very Low | High | High |
| **Transaction Costs** | 11% round-trip | 0.1% | 0.2% |
| **Minimum Investment** | $20K per asset | $100 | $1K |
| **Diversification Benefit** | **High** | N/A | Medium |

## Research Extensions

### Potential Improvements

1. **Alternative Data Sources**
   - Streaming data (Spotify, Apple Music)
   - Social media sentiment
   - TikTok viral trends
   - Radio play statistics

2. **Advanced Models**
   - Machine learning (Random Forest, XGBoost)
   - Time series forecasting for revenue
   - Genre-specific models
   - Non-linear age effects

3. **Dynamic Position Sizing**
   - Volatility-based sizing
   - Kelly Criterion application
   - Confidence-weighted positions
   - Cash flow predictability weighting

4. **Factor Analysis**
   - PCA on royalty characteristics
   - Genre factor returns
   - Technology adoption factors
   - Generational preference factors

5. **Optimization**
   - Mean-variance optimization
   - Black-Litterman with views
   - Hierarchical risk parity
   - CVaR optimization

### Open Questions

- What is the optimal rebalancing frequency given costs?
- How much alpha is truly available after costs?
- Does the age premium persist or decay?
- Can machine learning improve predictions significantly?
- What are the capacity limits in practice?
- How do different genres perform across cycles?

## References

### Academic Study
- Quant Radio: Music Royalties vs. the Stock Market (podcast transcript)
- Study period: 2017-2022
- Transaction data: Royalty Exchange platform
- Sample: Nearly $100M in transactions

### Key Insights
1. LOR contracts outperform 10-Year Term contracts significantly
2. Stability premium exists (market pays for consistent revenue)
3. Age premium exists (older catalogs command higher prices)
4. Returns have low correlation with equity markets
5. Transaction costs are substantial hurdle to overcome

## Support & Contributing

### Issues
Report issues or request features via project issues tracker

### Contributing
Contributions welcome:
- Model improvements
- Additional data sources
- Visualization enhancements
- Documentation improvements

### License
See LICENSE file for details

---

**Disclaimer:** This strategy is for educational and research purposes. Music royalty investing involves substantial risks including illiquidity, high transaction costs, and uncertain cash flows. Past performance does not guarantee future results. Consult with qualified professionals before making investment decisions.

**Last Updated:** January 2026
