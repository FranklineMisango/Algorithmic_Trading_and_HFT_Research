# Crypto Macro-Fundamental Trading Strategy

## Research Overview

**Strategy Family:** Macro-Fundamental Cross-Asset Strategy for Cryptocurrencies

**One-Sentence Objective:** Systematically trade Bitcoin by modeling its price based on the interaction between external traditional financial factors (Fed monetary policy, aggregate market risk) and internal crypto-specific factors (adoption rate, crypto risk premium).

**Based On:** Research by Adams, Ibert, and Lea - "What Really Drives Crypto Prices?" (Quant Radio)

---

## Hypothesis & Economic Rationale

### Core Hypothesis

Bitcoin's long-term price trajectory is significantly driven by external macroeconomic forces, while its short-term volatility is dominated by internal crypto-market dynamics. The crypto risk premium, observable through stablecoin flows, is a key transmission mechanism.

### Economic Foundation

The research finds that crypto is **not an isolated system**:

- **2022 Bitcoin Crash:** Would have been only **14% instead of 64%** without Fed tightening
- **Proof of Connection:** Strong empirical link to traditional finance
- **Price Dynamics:** Result from a "complicated dance" between:
  - **External Forces:** Fed rates, global risk appetite, treasury yields
  - **Internal Forces:** Technology, community sentiment, adoption rate

### Key Empirical Finding

Bitcoin price changes can be decomposed into four categories:
1. External macro factors (Fed policy, VIX)
2. Crypto risk premium (stablecoin flows)
3. Adoption/growth dynamics
4. Institutional validation events

---

## Strategy Architecture

### 1. External Macro Signal

**Formula:**
```
External_Macro_Signal = Z-Score(Δ(US 2-Year Treasury Yield) + Δ(VIX Index))
```

**Components:**
- **US 2-Year Treasury Yield Change:** Proxy for Fed monetary policy stance
- **VIX Index Change:** Traditional market fear gauge

**Parameters:**
- Lookback period: 60 trading days for Z-Score calculation
- **Interpretation:** Positive score = strong external headwinds for crypto (bearish)

**Rationale:** Tightening monetary policy (rising yields) + rising fear (VIX) creates unfavorable conditions for risk assets like Bitcoin.

---

### 2. Crypto Risk Premium Signal

**Formula:**
```
Risk_Premium = Growth(Stablecoin Market Cap) / Growth(Total Crypto Market Cap)
```

**Measurement:**
- Stablecoin basket: USDT, USDC, BUSD, DAI
- Total crypto market cap excludes stablecoins initially

**Parameters:**
- Growth windows: 5-day and 20-day rolling
- **Interpretation:** Rising ratio = capital seeking safety within crypto (bearish for Bitcoin)

**Historical Validation:**
- **March 2020 COVID Crash:** Stablecoin ratio spiked
- **FTX Collapse (Nov 2022):** Massive spike as investors fled to USDT/USDC

**Economic Logic:** When crypto participants move to stablecoins, they're de-risking without leaving the ecosystem entirely. This signals internal fear.

---

### 3. Adoption/Growth Signal

**Formula:**
```
Adoption_Signal = Δ(Log(Total Crypto Market Cap ex-Stablecoins))
```

**Parameters:**
- Window: 30-day rolling growth rate
- **Interpretation:** Positive acceleration = bullish (organic capital inflow)

**Example Event:**
- **BlackRock ETF Launch (Jan 2024):** Massive capital inflow, decreased perceived risk premium

**Rationale:** Exponential growth in non-stablecoin crypto market cap indicates genuine adoption, not just speculative rotation.

---

### 4. Institutional Validation Signal

**Type:** Binary event flag

**Algorithm:**
- Set flag to **1** for 30 trading days following major institutional adoption event
- Examples:
  - Spot Bitcoin ETF launch (BlackRock, Fidelity)
  - Major corporate treasury allocation (MicroStrategy, Tesla)
  - Regulatory clarity (SEC approval)

**Parameters:**
- Duration: 30 days post-event
- **Interpretation:** Institutional entry decreases crypto risk premium (bullish)

**Paper Finding:** Major institutional events durably shift market perception and lower the risk premium.

---

## Machine Learning Model

### Model Selection

**Primary:** XGBoost Regressor  
**Alternative:** Random Forest

**Why Gradient Boosting?**
- Handles non-linear interactions between macro and crypto factors
- Captures regime changes (e.g., pre/post-2022 Fed pivot)
- Robust to outliers with proper tuning

### Target Variable

```
Target = Log(BTC_Price_t+1 / BTC_Price_t)
```

Next-day Bitcoin log return (not price level to ensure stationarity).

### Hyperparameter Optimization

**Method:** Bayesian Optimization (Optuna)

**Search Space:**
- `n_estimators`: [50, 100, 200, 300]
- `max_depth`: [3, 5, 7, 10]
- `learning_rate`: [0.01, 0.05, 0.1, 0.2]
- `subsample`: [0.6, 0.8, 1.0]
- `colsample_bytree`: [0.6, 0.8, 1.0]

**Objective:** Maximize **out-of-sample Sharpe Ratio** (not just MSE)

**Rationale:** Trading performance matters more than pure prediction accuracy.

---

## Validation Framework

### Walk-Forward Time Series Cross-Validation

**Training Window:** 2 years (~504 trading days)  
**Validation/Test Window:** 6 months (~126 trading days)  
**Step Size:** Roll forward 1 month (21 days) at a time

**Critical:** No data leakage. All features lagged by 1 day.

**Example:**
- Train: 2020-01-01 to 2021-12-31 → Test: 2022-01-01 to 2022-06-30
- Train: 2020-02-01 to 2022-01-31 → Test: 2022-02-01 to 2022-07-31
- Repeat...

---

## Portfolio Construction

### Position Sizing

**Formula:**
```
Position_Size = K * Model_Score
```

Where **K** is a volatility scaler such that portfolio targets **15% annualized volatility**.

**Calculation:**
```python
K = (target_vol / realized_vol) * capital
Position = K * model_predicted_return
```

### Leverage Rules

- **Maximum Leverage:** 2x (long or short)
- **Risk-Off Condition:** If both macro signal AND risk premium signal are in the top quartile (extreme stress):
  - Reduce to **1x leverage** (no leverage multiplier)

**Example:**
- Normal: Model predicts +2% return → Position = 2x * capital = $200k exposure on $100k capital
- Risk-Off: Same prediction → Position = 1x * capital = $100k exposure (no leverage)

### Execution

- **Order Type:** Market-on-Close (MOC)
- **Frequency:** Daily rebalancing
- **Latency:** Not critical (daily macro strategy)

---

## Risk Management

### 1. Stop-Loss Rule

**Trigger:** Portfolio experiences **10% drawdown** from last peak

**Action:** Reduce all positions by **50%**

**Rationale:** Protects against catastrophic losses during black swan events not captured by the model.

---

### 2. Stress Testing

**Historical Validation Periods:**

| Event | Period | Expected Signal Behavior |
|-------|--------|-------------------------|
| **COVID Crash** | Mar 2020 | External macro + risk premium spike |
| **Fed Tightening** | 2022 | Persistent negative macro signal (yielding 64% crash per paper) |
| **FTX Collapse** | Nov 2022 | Risk premium spike, internal shock |

**Test:** Run backtest specifically on these periods to validate model captures documented dynamics.

---

### 3. Real-Time Monitoring

#### Stablecoin Ratio Spike Alert

**Condition:**
```
5-day Risk Premium > 90-day MA + 1 * StdDev
```

**Action:** Email alert to portfolio manager (potential liquidation event brewing)

#### Model Decay Monitor

**Metric:** Rolling 3-month out-of-sample Sharpe Ratio

**Threshold:** If Sharpe < 0.0

**Action:** **Halt trading** (model relationship has broken down)

---

## Transaction Costs

### Cost Model

- **Commission:** 0.10% per trade (conservative retail crypto exchange estimate)
- **Slippage:** 0.05% (market-on-close orders)
- **Total:** **0.15% per trade**

### Numeric Example (from Blueprint)

**Scenario:** $100k portfolio, model signals 50% long position

**Trade Size:** $50,000

**Cost Calculation:**
```
Total Cost = $50,000 * 0.0015 = $75
```

**Impact on Return:**
```
Gross Return: +2%
Net Return: +2% - ($75 / $100,000) = +1.925%
```

---

## Evaluation Metrics

### Performance Metrics

1. **Annualized Return**
2. **Annualized Volatility**
3. **Sharpe Ratio** (primary)
4. **Sortino Ratio** (downside focus)
5. **Maximum Drawdown**
6. **Win Rate**
7. **Calmar Ratio** (Return / Max DD)

### Statistical Tests

**One-Sided T-Test:**
```
H0: Strategy Return ≤ Benchmark Return (BTC Buy-and-Hold)
Ha: Strategy Return > Benchmark Return
```

Report p-value. Significance level: α = 0.05.

### Capacity Analysis

**Method:** Simulate increasing capital with price impact

**Price Impact Model:**
```
Impact = 0.20% per $1M traded
```

**Test Levels:** $100k, $500k, $1M, $5M, $10M

**Report:** Asset size where Sharpe Ratio degrades by 10%

---

## Failure Modes & Mitigations

### 1. Regime Change (Rank 1)

**Failure:** Relationship between macro factors and crypto breaks down

**Severity:** High | **Likelihood:** Medium

**Mitigation:**
- Monitor 6-month rolling correlation between Macro Signal and BTC returns
- **Action:** If correlation falls below **0.1**, pause the macro component
- Continue trading using only crypto-internal signals

---

### 2. Stablecoin Peg Failure (Rank 2)

**Failure:** Major stablecoin (e.g., USDT) loses its peg, breaking the "safe haven" assumption

**Severity:** Critical | **Likelihood:** Low

**Mitigation:**
- **Immediate Action:** Exclude affected stablecoin from total stablecoin market cap
- Switch to basket of other audited stablecoins (USDC, DAI)
- Monitor USDT/USD price: alert if outside [0.98, 1.02] band

---

### 3. Model Overfitting (Rank 3)

**Failure:** Model learns spurious patterns from short, volatile crypto history

**Severity:** High | **Likelihood:** High

**Mitigation:**
- **Strict walk-forward validation** (no in-sample optimization)
- Feature importance analysis: prune features with consistently low importance
- Regularization: limit tree depth, use early stopping
- **Cross-validation:** Test on multiple non-overlapping periods

---

### 4. Exchange Risk (Rank 4)

**Failure:** Backtest assumes perfect execution, but trading venue fails (e.g., FTX)

**Severity:** Critical | **Likelihood:** Medium

**Mitigation:**
- **Mandatory:** Hold capital on **at least 2 top-tier, regulated exchanges**
- Position limits per exchange (e.g., max 50% on any single venue)
- Exchanges must have proof-of-reserves audits
- Prefer exchanges with regulatory oversight (e.g., Coinbase, Kraken, Gemini in US)

---

### 5. Crypto-Specific Black Swan (Rank 5)

**Failure:** Catastrophic protocol hack or regulatory ban not captured by modeled factors

**Severity:** High | **Likelihood:** Medium

**Mitigation:**
- **Hard stop-loss rule:** 10% drawdown triggers 50% position reduction
- **Portfolio allocation:** Allocate only **<20% of total fund capital** to this strategy
- Diversify across BTC and ETH (ETH may react differently to certain shocks)

---

## Data Requirements

### Crypto Data

| Data | Source | Frequency | Fields |
|------|--------|-----------|--------|
| **Bitcoin Price** | Yahoo Finance | Daily | BTC-USD Close |
| **Ethereum Price** | Yahoo Finance | Daily | ETH-USD Close |
| **Total Crypto Market Cap** | CoinMarketCap API | Daily | Total Market Cap |
| **Stablecoin Market Caps** | CoinMarketCap API | Daily | USDT, USDC, BUSD, DAI |

### Traditional Finance Data

| Data | Source | Frequency | Fields |
|------|--------|-----------|--------|
| **US 2-Year Treasury Yield** | FRED (DGS2) | Daily | Yield (%) |
| **VIX Index** | Yahoo Finance | Daily | ^VIX Close |

### Event Data

| Data | Source | Frequency | Format |
|------|--------|-----------|--------|
| **Institutional Events** | Manual curation | Ad-hoc | Date, Description, Impact (+/-) |

**Example Events:**
- 2024-01-10: BlackRock Spot Bitcoin ETF Launch (+)
- 2022-11-11: FTX Collapse (-)

---

## Feature Engineering Pipeline

### Step 1: Calculate Raw Signals

1. **External Macro:**
   ```python
   macro_raw = treasury_yield.diff() + vix.diff()
   macro_signal = (macro_raw - macro_raw.rolling(60).mean()) / macro_raw.rolling(60).std()
   ```

2. **Risk Premium:**
   ```python
   stablecoin_growth_5d = stablecoin_mcap.pct_change(5)
   crypto_growth_5d = total_crypto_mcap.pct_change(5)
   risk_premium_5d = stablecoin_growth_5d / crypto_growth_5d
   ```

3. **Adoption:**
   ```python
   adoption_signal = np.log(total_crypto_mcap).diff(30)
   ```

4. **Institutional:**
   ```python
   institutional_flag = create_event_windows(event_dates, duration=30)
   ```

### Step 2: Create Interaction Terms

```python
macro_risk_compound = macro_signal * risk_premium_5d
```

### Step 3: Lag All Features by 1 Day

```python
features_lagged = features.shift(1)
```

**Critical:** Avoids look-ahead bias.

### Step 4: Winsorization

```python
features_winsorized = features_lagged.clip(
    lower=features_lagged.quantile(0.025),
    upper=features_lagged.quantile(0.975)
)
```

**Rationale:** Mitigates outlier effects in volatile crypto data.

---

## Implementation Roadmap

### Phase 1: Data Acquisition
- [ ] Set up FRED API for treasury yields
- [ ] Fetch BTC/ETH prices from yfinance
- [ ] CoinMarketCap API integration (or scraping)
- [ ] Manual event calendar creation

### Phase 2: Feature Engineering
- [ ] Implement all 4 signal calculations
- [ ] Create interaction terms
- [ ] Apply lag and winsorization

### Phase 3: Model Development
- [ ] Walk-forward CV framework
- [ ] XGBoost training pipeline
- [ ] Bayesian hyperparameter optimization (Optuna)
- [ ] Feature importance analysis

### Phase 4: Backtesting
- [ ] Vectorized backtest engine
- [ ] Transaction cost model (0.15%)
- [ ] Position sizing with volatility scaling
- [ ] Leverage rules implementation

### Phase 5: Risk Management
- [ ] Stop-loss rule (10% drawdown)
- [ ] Stablecoin ratio spike alerts
- [ ] Model decay monitoring
- [ ] Stress test validation

### Phase 6: Evaluation
- [ ] Performance metrics calculation
- [ ] Statistical significance tests
- [ ] Capacity analysis
- [ ] Comparison to BTC buy-and-hold

---

## Expected Results

### Hypothesis Validation

If the research findings hold:

1. **2022 Performance:** Model should show strong negative macro signal throughout 2022 (Fed tightening)
2. **Stablecoin Events:** Risk premium signal should spike during March 2020 and Nov 2022 (FTX)
3. **ETF Launch:** Institutional signal should trigger around Jan 2024 BlackRock ETF

### Performance Targets

- **Sharpe Ratio:** > 1.0 (vs BTC buy-and-hold ~0.5-0.8)
- **Max Drawdown:** < 40% (vs BTC ~80%)
- **Win Rate:** > 55%
- **Capacity:** Estimate viable up to ~$10M AUM before material price impact

---

## Computational Requirements

- **Training Time:** ~30 minutes per walk-forward fold (with hyperparameter tuning)
- **Memory:** ~4 GB RAM (for full dataset + model)
- **Storage:** ~500 MB (data + models + results)

---

## Disclaimer

This is a **research prototype** for educational purposes. It is **not investment advice**.

**Key Risks:**
1. Crypto markets are highly volatile and speculative
2. Historical relationships may not persist
3. Regulatory risk (potential bans or restrictions)
4. Exchange and custody risk
5. Model overfitting to limited historical data

**Past performance does not guarantee future results.**

---

## References

**Primary Source:**
- Adams, Ibert, and Lea - "What Really Drives Crypto Prices?" (Quant Radio)
  - Key Finding: 2022 Bitcoin crash would have been 14% (not 64%) without Fed tightening
  - Empirical Framework: 4-factor decomposition of crypto price changes

**Data Sources:**
- Federal Reserve Economic Data (FRED): US Treasury Yields
- Yahoo Finance: BTC-USD, ETH-USD, VIX
- CoinMarketCap: Crypto and stablecoin market capitalizations

---

## Contact & Contributions

For questions, issues, or contributions, please refer to the main repository documentation.

**Last Updated:** January 2026
