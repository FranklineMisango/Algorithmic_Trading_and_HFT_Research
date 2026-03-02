# Deep Learning Options Trading Strategy Critique

## **CRITICAL STRATEGIC FLAWS**

### 1. **Fundamental Conceptual Errors**

**Delta-Neutral Straddle Misunderstanding**
- The strategy claims to be "delta-neutral" but the LSTM outputs position signals from -1 to 1, suggesting directional bets
- True delta-neutral straddles (long call + long put at same strike) are ALWAYS directionally neutral by construction
- The model appears to be predicting which direction to lean positions, contradicting the delta-neutral premise
- No actual delta calculation or hedging logic exists in the code

**Option Pricing vs. Returns Prediction Confusion**
- `create_features.py` line 119: Targets are set as `option_price` (next day's price)
- But `SharpeLoss` (line 106-130 in `lstm_model.py`) assumes targets are **returns**, not prices
- This fundamental mismatch means the model is trained on wrong objectives
- Portfolio returns calculated as `predictions * targets` only makes sense if targets are returns

### 2. **Data Quality Issues**

**Missing Critical Options Data**
- No strike prices in processed features
- No expiration dates tracked
- No implied volatility calculated (listed in config but not computed)
- No distinction between calls and puts
- "Moneyness" referenced in config but never computed in `create_features.py`
- Time to expiration bins mentioned but not created

**Straddle Pricing Fiction**
```python
# backtester.py line 119
atm_option = day_options.iloc[(day_options['moneyness'] - 1.0).abs().argmin()]
return atm_option['straddle_price']
```
- References `moneyness` and `straddle_price` columns that don't exist in actual data
- `create_features.py` only has `option_price` (single option), not straddle prices
- No logic to match calls and puts to form straddles

**Survivorship Bias Not Actually Controlled**
- README claims "point-in-time S&P 100 constituent lists" but only 3 tickers in actual data (AAPL, MSFT, AMZN)
- No historical constituent data loaded or used
- All 3 tickers are mega-cap survivors - extreme survivorship bias

### 3. **Model Architecture Problems**

**Inappropriate Loss Function**
```python
# lstm_model.py lines 106-130
sharpe_ratio = (mean_return / (std_return + 1e-8)) * np.sqrt(252)
loss = -sharpe_ratio + turnover_loss
```

Critical issues:
- **Batch Sharpe calculation is meaningless**: Computing Sharpe on 32-sample mini-batches (default batch size) produces extremely noisy, unreliable gradients
- **Non-differentiable in practice**: Sharpe ratio optimization requires large sample sizes to be statistically meaningful
- **Ignores sequential dependencies**: Each batch treated independently, breaking time-series structure
- **Turnover penalty broken**: `prev_positions` parameter exists but is never passed (always None)

**LSTM Architecture Mismatch**
```python
# lstm_model.py lines 82-84
lstm_out, (hn, cn) = self.lstm(x)
last_output = lstm_out[:, -1, :]
```
- Takes only the LAST time step output, throwing away all sequential information
- If only using final output, why use LSTM at all? A simple feedforward network would be more appropriate
- Defeats the purpose of sequence modeling
- 30-day sequences condensed to single output without leveraging temporal patterns

**Severe Data Leakage in Targets**
```python
# create_features.py lines 119-120
target = ticker_data.iloc[i]['option_price']
```
- Target is the SAME DAY's option price, not next day's
- The model sees the answer in the input sequence at index i-1
- Should be `ticker_data.iloc[i+1]` or better yet, actual forward returns

### 4. **Backtesting Unrealistic & Broken**

**Transaction Cost Model Wishful Thinking**
```python
# config.yaml line 46
transaction_cost_per_contract: 0.01  # $0.01 per contract
bid_ask_spread: 0.02  # 2% bid-ask spread
```
- $0.01/contract is broker fee only
- Real ATM straddle spreads on mega-caps are 0.5-2% at BEST
- Illiquid strikes can have 5-10% spreads
- 2% assumed spread is optimistic, especially for daily turnover
- No impact cost modeling for larger positions

**Slippage Model Inadequate**
```python
# backtester.py lines 190-202
if slippage_model == 'conservative':
    slippage = 0.001  # 0.1% slippage
```
- Fixed 0.1% slippage regardless of:
  - Trade size
  - Market volatility
  - Time of day
  - Liquidity conditions
- Real slippage scales non-linearly with size
- During volatile periods (when options traders make money), slippage explodes

**Portfolio Value Calculation Broken**
```python
# backtester.py lines 256-266
def _get_options_price_from_prices(self, prices_df, ticker, date):
    underlying_price = prices_df.loc[(ticker, date), 'Adj Close']
    return underlying_price * 0.05  # 5% of underlying price
```
- **USES SYNTHETIC PRICES**: Portfolio marked-to-market using fake 5% formula, not actual options data
- Completely disconnected from reality
- Makes all backtest metrics meaningless
- Why use Databento API if substituting with synthetic prices?

**Missing Gamma Risk**
- Delta-neutral strategies face gamma risk (P&L from underlying movement)
- No gamma calculation or hedging
- Straddles lose money from time decay (theta) unless volatility increases
- No theta modeling whatsoever

### 5. **Risk Management Theater**

**Position Sizing Contradictory**
```python
# config.yaml lines 47-49
max_position_size: 0.05  # Max 5% per straddle
max_single_stock_exposure: 0.10  # Max 10% per underlying
max_portfolio_delta: 0.1  # Max net delta
```
- Delta limit of ±10% contradicts "delta-neutral" claim
- If truly delta-neutral, delta should be near 0%, not ±10%
- Position limits applied but never checked for vega exposure (despite config)
- No actual Greek calculations anywhere in code

**Stress Testing Missing**
```python
# config.yaml lines 57-59
stress_test_periods:
  - "2008-01-01"  # Financial crisis
  - "2020-03-01"  # COVID crash
```
- Stress test periods defined but never actually used in `backtester.py`
- No stress testing code implemented
- Data doesn't even cover these periods (only has 2025-2026)

### 6. **Walk-Forward Validation Flawed**

```python
# lstm_model.py lines 359-400
def walk_forward_validation(...):
    train_periods = train_years * 252
    val_periods = val_years * 252
```
Problems:
- Assumes exactly 252 trading days per year (reality: 250-253)
- No gap between train and validation sets (look-ahead bias)
- Re-trains model completely each window (computationally wasteful, unstable)
- No test set - validates on same data used for early stopping
- Should use train/validation/test split with proper time gaps

### 7. **Feature Engineering Inadequacy**

**Missing Critical Features Listed in Docs**
From config.yaml:
```yaml
feature_list:
  - moneyness  # NOT COMPUTED
  - time_to_expiry  # NOT COMPUTED
  - implied_volatility  # NOT COMPUTED
```
Only 3 of 6 promised features actually created.

**Feature Leakage**
```python
# create_features.py lines 79-80
features['option_price'] = merged['close_option']
```
- Using option's closing price in features that predict... the option's price?
- Trivial prediction task, not useful for live trading

**No Options-Specific Features**
Missing:
- Put/call ratios
- Skew metrics
- Term structure
- Volume-weighted IV
- Open interest changes
- Actual Greeks

### 8. **Implementation Bugs**

**Scaler Shape Mismatch**
```python
# lstm_model.py lines 198-199
X_train_scaled = self.scaler.fit_transform(X_train.reshape(X_train.shape[0], -1))
X_train_scaled = X_train_scaled.reshape(X_train.shape)
```
- StandardScaler fit on flattened 2D array
- Reshapes back to 3D
- Destroys temporal structure - scaling done across time steps incorrectly
- Should scale each feature independently across time dimension

**Metadata Handling Error**
```python
# lstm_model.py lines 29-34
def __getitem__(self, idx):
    if self.metadata and idx < len(self.metadata):
        return self.X[idx], self.y[idx], self.metadata[idx]
    else:
        return self.X[idx], self.y[idx], {}
```
- Inconsistent return types based on metadata presence
- Will cause unpacking errors in training loop
- Should always return 3-tuple

**DataLoader Ignores Metadata**
```python
# lstm_model.py line 232
for batch_X, batch_y, _ in train_loader:
```
- Metadata returned but never used
- Can't track which predictions correspond to which dates/tickers
- Makes debugging impossible

### 9. **Market Reality Disconnect**

**Daily Rebalancing Impossible**
- Strategy assumes daily position updates
- Real options markets:
  - Spreads widen significantly intraday
  - Liquidity varies dramatically
  - Market makers widen quotes for known patterns
- High-frequency delta-neutral shops rebalance INTRADAY, not daily
- Daily rebalancing with transaction costs makes profitability extremely difficult

**Capacity Delusions**
```python
# config.yaml line 45
initial_capital: 1000000  # $1M
```
- With 3 tickers and 5% position sizes, this is $50k per straddle
- On liquid names like AAPL, feasible
- But scaling to 100 stocks (S&P 100 claim) means $50M capital
- At that size, market impact becomes serious
- No capacity analysis despite claiming it in README

**Volatility Trading Reality**
- Profitable options trading requires predicting volatility **changes**, not levels
- Model has no forward vol prediction
- No comparison to implied vs realized vol
- Missing the core edge in volatility trading

### 10. **Academic Paper Claims Unsubstantiated**

README claims this is "based on academic research" from a Quant Radio podcast, but:
- No citation to actual paper
- No replication of specific methodology details
- "Parsimonious features" claimed but critical features missing
- Claims Sharpe optimization but implementation is broken
- No validation that this actually reproduces research results

## **RECOMMENDATIONS FOR REDEMPTION**

### Immediate Fixes (Foundation)
1. **Fix target variable**: Use actual forward returns, not current prices
2. **Fix loss function**: Use MSE on returns, compute Sharpe only for evaluation
3. **Fix data**: Properly compute moneyness, time-to-expiry, IV
4. **Fix backtester**: Use real options prices, not synthetic
5. **Remove delta-neutral claims**: Strategy is NOT delta-neutral as implemented

### Medium-term Improvements
1. **Implement actual Greeks**: Calculate delta, gamma, vega, theta
2. **Fix walk-forward**: Add proper train/val/test splits with time gaps
3. **Realistic costs**: Model spread widening, market impact, actual execution
4. **Feature engineering**: Add skew, term structure, realized vol forecasts
5. **Expand data**: Multi-year history, full S&P 100 coverage

### Long-term Strategy Rethink
1. **Choose a clear objective**: Delta-neutral vol trading OR directional options?
2. **Think about edge**: Why would LSTM beat market makers with millisecond data?
3. **Consider simpler models**: Random forests often outperform LSTMs for tabular data
4. **Focus on prediction target**: IV forecast? Realized vol? Price dislocations?
5. **Study real vol trading**: Term structure arbitrage, variance risk premium, etc.

This strategy, as currently implemented, would lose money rapidly in live trading due to data issues, conceptual errors, and unrealistic assumptions. It requires fundamental redesign, not incremental fixes.