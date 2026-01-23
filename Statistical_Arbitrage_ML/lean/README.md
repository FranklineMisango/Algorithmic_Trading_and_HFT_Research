# QuantConnect Lean Integration - Statistical Arbitrage ML Strategy

This directory contains the QuantConnect Lean algorithm implementation for backtesting the statistical arbitrage ML strategy.

## Overview

The strategy is implemented in QuantConnect's Lean engine, which provides:
- Robust backtesting framework
- Realistic market simulation
- Transaction cost modeling
- Slippage and market impact
- Professional-grade execution

## Files

- `main.py` - Main algorithm implementation (QCAlgorithm class)
- `config.json` - Algorithm configuration
- `README.md` - This file
- `requirements.txt` - Python dependencies for Lean

## Algorithm Structure

### StatisticalArbitrageMLStrategy Class

**Key Components:**

1. **Universe Selection**
   - CoarseSelectionFunction: Filters by price ($5+) and volume ($10M+ daily)
   - FineSelectionFunction: Excludes specific sectors (small biotech, etc.)
   - Universe size: ~200-300 stocks (scalable to full Russell 3000)

2. **Feature Calculation**
   - Momentum indicators (5, 10, 20, 60, 126, 252 days)
   - Mean reversion indicators (distance from 10, 20, 50, 100, 200-day MAs)
   - Volume ratio (20-day vs 126-day average)
   - Volatility (20-day standard deviation)

3. **Prediction Generation**
   - Currently uses simple heuristic (placeholder)
   - **TODO**: Integrate pre-trained ML model from training notebook
   - Model should be loaded in `Initialize()` method
   - Predictions used in `GeneratePredictions()` method

4. **Portfolio Construction**
   - Ranks stocks by predicted returns
   - Selects top 20 for long, bottom 20 for short
   - Equal-weight positions (adjustable)
   - Max 4% per position
   - Market-neutral (50% long, 50% short)

5. **Rebalancing**
   - Every 3 days (configurable)
   - Liquidates old positions
   - Enters new positions
   - Maintains market neutrality

## Running the Backtest

### Option 1: QuantConnect Cloud

1. Create account at https://www.quantconnect.com
2. Create new algorithm project "Statistical Arbitrage ML"
3. Copy contents of `main.py` to the algorithm editor
4. Set backtest parameters (dates, capital)
5. Click "Backtest"

### Option 2: Lean CLI (Local)

#### Prerequisites

Install Lean CLI:
```bash
pip install lean
```

#### Setup

```bash
# Navigate to project root
cd /path/to/Statistical_Arbitrage_ML

# Initialize Lean (first time only)
lean init

# Login to QuantConnect (optional, for cloud features)
lean login
```

#### Run Backtest

```bash
# Run backtest
lean backtest "lean"

# Or with specific config
lean backtest "lean" --start 20200101 --end 20231231
```

#### View Results

Results will be saved in `lean/backtests/` directory:
- HTML report with charts
- JSON statistics
- Trades log
- Equity curve

## Configuration Parameters

Edit these in `main.py` `Initialize()` method:

```python
# Backtest period
self.SetStartDate(2020, 1, 1)
self.SetEndDate(2023, 12, 31)
self.SetCash(1000000)  # Starting capital

# Strategy parameters
self.n_long = 20          # Long positions
self.n_short = 20         # Short positions
self.holding_days = 3     # Rebalance frequency
self.max_position_size = 0.04  # Max 4% per stock

# Feature parameters
self.momentum_periods = [5, 10, 20, 60, 126, 252]
self.ma_periods = [10, 20, 50, 100, 200]
```

## Integrating Trained ML Model

To use the model trained in the notebook:

### Step 1: Export Model

In the training notebook, save model in Lean-compatible format:

```python
# In model_training_and_evaluation.ipynb
import joblib

# Save model
joblib.dump(best_trainer.model, '../lean/trained_model.pkl')

# Save feature names
with open('../lean/feature_names.txt', 'w') as f:
    f.write('\n'.join(feature_names))
```

### Step 2: Load Model in Lean

Modify `main.py`:

```python
def Initialize(self):
    # ... existing code ...
    
    # Load pre-trained model
    try:
        import joblib
        self.model = joblib.load('trained_model.pkl')
        
        with open('feature_names.txt', 'r') as f:
            self.feature_names = [line.strip() for line in f]
        
        self.Debug(f"Loaded ML model with {len(self.feature_names)} features")
    except Exception as e:
        self.Debug(f"Could not load model: {e}")
        self.Debug("Using simple heuristic instead")
```

### Step 3: Update Prediction Method

Replace `SimplePredictionHeuristic()` with actual model:

```python
def GeneratePredictions(self):
    predictions = {}
    
    for symbol, data_dict in self.securities_data.items():
        # Calculate features
        features = self.CalculateFeatures(symbol, history)
        
        if features is None or self.model is None:
            continue
        
        # Prepare feature vector in correct order
        feature_vector = np.array([features.get(name, 0) for name in self.feature_names])
        
        # Generate prediction using trained model
        predicted_return = self.model.predict([feature_vector])[0]
        
        predictions[symbol] = {
            'predicted_return': predicted_return,
            'current_price': history[0].Close
        }
    
    return predictions
```

## Performance Metrics

Lean automatically calculates:

- **Returns**: Total, annual, monthly
- **Risk**: Sharpe ratio, Sortino ratio, max drawdown
- **Market exposure**: Beta, alpha (vs SPY)
- **Trading**: Win rate, profit factor, turnover
- **Factor analysis**: Fama-French factors (if configured)

## Target Performance (from strategy blueprint)

| Metric | Target |
|--------|--------|
| Annual Return | 20-28% |
| Sharpe Ratio | >1.4 |
| Max Drawdown | <20% |
| Market Correlation | <0.15 (vs S&P 500) |
| Win Rate | ~50% |

## Troubleshooting

### Common Issues

1. **"Symbol not found"**
   - Some tickers may not be available in Lean's data
   - Solution: Filter universe or use alternative tickers

2. **"Insufficient buying power"**
   - Position sizes too large or too many positions
   - Solution: Reduce n_long/n_short or max_position_size

3. **"Model file not found"**
   - Model file not uploaded to Lean
   - Solution: Upload via Lean CLI or include in project files

4. **Slow backtests**
   - Large universe or high-frequency rebalancing
   - Solution: Reduce universe size or increase holding period

## Advanced Features

### Dynamic Position Sizing

Replace equal-weight with signal-based sizing:

```python
def ConstructPortfolio(self, predictions):
    # Weight by prediction strength
    long_weights = {}
    short_weights = {}
    
    for symbol, data in long_candidates:
        weight = min(abs(data['predicted_return']), self.max_position_size)
        long_weights[symbol] = weight
    
    # Normalize to sum to 0.5
    total = sum(long_weights.values())
    long_positions = {s: w/total * 0.5 for s, w in long_weights.items()}
    
    # Similar for shorts...
```

### Risk Management

Add position limits and circuit breakers:

```python
def ExecuteTrades(self, long_positions, short_positions):
    # Check portfolio risk
    if self.CalculatePortfolioRisk() > self.max_portfolio_risk:
        self.Debug("Risk limit exceeded, reducing positions")
        long_positions = {s: w*0.5 for s, w in long_positions.items()}
        short_positions = {s: w*0.5 for s, w in short_positions.items()}
    
    # Execute trades...
```

### Transaction Cost Analysis

Lean includes transaction costs by default. To adjust:

```python
def Initialize(self):
    # Set custom transaction cost model
    self.SetSecurityInitializer(lambda security: security.SetFeeModel(
        CustomFeeModel()  # Define your own fee model
    ))
```

## Next Steps

1. **Train Model**: Run `model_training_and_evaluation.ipynb` notebook
2. **Export Model**: Save trained model to `lean/` directory
3. **Integrate Model**: Update `main.py` to load and use model
4. **Run Backtest**: Execute via Lean CLI or QuantConnect cloud
5. **Analyze Results**: Review performance metrics and charts
6. **Optimize**: Tune parameters based on backtest results
7. **Paper Trade**: Deploy to paper trading for live testing

## Resources

- [QuantConnect Documentation](https://www.quantconnect.com/docs)
- [Lean GitHub Repository](https://github.com/QuantConnect/Lean)
- [QuantConnect Community](https://www.quantconnect.com/forum)
- [Lean CLI Documentation](https://www.quantconnect.com/docs/v2/lean-cli)

## Disclaimer

This algorithm is for educational and research purposes. Past performance does not guarantee future results. Thoroughly test and validate before live trading.
