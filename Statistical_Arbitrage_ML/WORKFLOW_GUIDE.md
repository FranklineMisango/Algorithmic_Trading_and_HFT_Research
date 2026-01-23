# Complete Workflow Guide - Statistical Arbitrage ML Strategy

This guide provides a step-by-step walkthrough of the complete workflow from setup to backtesting.

## Table of Contents

1. [Initial Setup](#initial-setup)
2. [Data Exploration](#data-exploration)
3. [Model Training](#model-training)
4. [Model Export](#model-export)
5. [Backtesting with Lean](#backtesting-with-lean)
6. [Results Analysis](#results-analysis)
7. [Optimization](#optimization)

---

## 1. Initial Setup

### Install Dependencies

```bash
# Navigate to project directory
cd Statistical_Arbitrage_ML

# Install Python dependencies
pip install -r requirements.txt

# Install QuantConnect Lean CLI
pip install lean
```

### Initialize Lean

```bash
# Initialize Lean environment (first time only)
lean init

# Login to QuantConnect (optional, enables cloud features)
lean login
```

### Verify Installation

```bash
# Check Lean installation
lean --version

# Check Python packages
python -c "import pandas, numpy, sklearn, xgboost, lightgbm; print('✓ All packages installed')"
```

---

## 2. Data Exploration

### Open Exploratory Notebook

```bash
# Option 1: Using Jupyter
jupyter notebook notebooks/exploratory_analysis.ipynb

# Option 2: Using VS Code
# Open notebooks/exploratory_analysis.ipynb in VS Code with Jupyter extension
```

### Run Cells Sequentially

The notebook covers:

1. **Data Acquisition** (Cells 1-5)
   - Load Russell 3000 universe
   - Download historical data
   - Apply quality filters

2. **Data Quality Analysis** (Cells 6-9)
   - Check for missing values
   - Analyze price distributions
   - Visualize sample time series

3. **Feature Engineering** (Cells 10-13)
   - Calculate momentum indicators
   - Calculate mean reversion indicators
   - Calculate volume features

4. **Feature Analysis** (Cells 14-18)
   - Feature distributions
   - Correlation analysis
   - Target variable analysis

### Key Outputs

- Understanding of data quality
- Feature distributions and relationships
- Target variable (3-day returns) characteristics

---

## 3. Model Training

### Open Training Notebook

```bash
# Option 1: Using Jupyter
jupyter notebook notebooks/model_training_and_evaluation.ipynb

# Option 2: Using VS Code
# Open notebooks/model_training_and_evaluation.ipynb in VS Code
```

### Execute Training Pipeline

The notebook includes:

#### Section 1-2: Setup and Configuration (Cells 1-4)
- Import libraries
- Set configuration parameters
- Define strategy settings

#### Section 3: Data Acquisition (Cells 5-9)
- Download historical data for training
- Quality analysis and validation
- Visualize price distributions

#### Section 4: Feature Engineering (Cells 10-15)
- Calculate all features
- Feature distribution analysis
- Correlation analysis

#### Section 5: ML Dataset Preparation (Cells 16-17)
- Create feature matrix (X) and target (y)
- Analyze target distribution

#### Section 6: Train/Test Split (Cell 18)
- Split data temporally (80/20)

#### Section 7: Model Training (Cells 19-22)
- Train multiple models:
  - Ridge Regression
  - Random Forest
  - XGBoost
  - LightGBM
- Compare performance
- Visualize results

#### Section 8: Feature Importance (Cell 23)
- Analyze feature contributions
- Identify top predictors

#### Section 9: Prediction Analysis (Cells 24-26)
- Predictions vs actuals
- Residual analysis

#### Section 10: Portfolio Simulation (Cells 27-29)
- Simulate portfolio construction
- Analyze exposures

#### Section 11: Model Persistence (Cell 30)
- Save best model to `models/` directory

### Expected Results

At the end of training, you should have:

- **Trained Models**: Multiple models compared
- **Best Model Identified**: Based on Test R²
- **Model File**: Saved to `models/[model_type]_best.pkl`
- **Performance Metrics**: R², RMSE, MAE for each model

### Example Output

```
MODEL PERFORMANCE COMPARISON
================================================================================
      Model  Train R²  Test R²  Test RMSE  Test MAE  Overfit Gap
      ridge    0.0234   0.0198    0.0523    0.0385         0.0036
random_forest  0.1245   0.0456    0.0498    0.0372         0.0789
     xgboost  0.0687   0.0512    0.0482    0.0361         0.0175
    lightgbm  0.0645   0.0498    0.0485    0.0364         0.0147
================================================================================

🏆 Best model (by Test R²): XGBOOST
```

---

## 4. Model Export

### Export Trained Model for Lean

After completing the training notebook:

```bash
# Export the best model (e.g., XGBoost)
python export_model_for_lean.py --model-path models/xgboost_best.pkl

# Or specify custom output directory
python export_model_for_lean.py --model-path models/xgboost_best.pkl --output-dir lean
```

### Verify Export

Check that these files were created in `lean/` directory:

```
lean/
├── trained_model.pkl       # The ML model
├── feature_names.txt       # Feature names in order
└── model_metadata.json     # Model information
```

### Review Model Metadata

```bash
# View model metadata
cat lean/model_metadata.json
```

Expected output:
```json
{
  "model_type": "XGBRegressor",
  "n_features": 20,
  "feature_names": ["momentum_5d", "momentum_10d", ...],
  "source_model": "models/xgboost_best.pkl"
}
```

---

## 5. Backtesting with Lean

### Review Lean Algorithm

Before running, review the algorithm:

```bash
# Open in editor
code lean/main.py

# Or view README
cat lean/README.md
```

### Configure Backtest Parameters

Edit `lean/main.py` if needed:

```python
# In Initialize() method:
self.SetStartDate(2020, 1, 1)     # Backtest start
self.SetEndDate(2023, 12, 31)     # Backtest end
self.SetCash(1000000)              # Starting capital

# Strategy parameters
self.n_long = 20                   # Long positions
self.n_short = 20                  # Short positions
self.holding_days = 3              # Rebalance frequency
```

### Run Backtest

#### Option 1: Local with Lean CLI

```bash
# Run backtest locally
lean backtest "lean"

# View progress
# Lean will show progress and save results when complete
```

#### Option 2: QuantConnect Cloud

1. Go to https://www.quantconnect.com
2. Create account/login
3. Create new algorithm project: "Statistical Arbitrage ML"
4. Copy contents of `lean/main.py`
5. Upload `trained_model.pkl` and `feature_names.txt` to project files
6. Click "Backtest"

### Monitor Backtest

Lean will display:
- Date progress
- Portfolio value
- Key events (rebalances, trades)
- Any errors or warnings

### Backtest Duration

Expected time:
- 3-year backtest: 5-15 minutes (local)
- 5-year backtest: 10-30 minutes (local)
- Faster on QuantConnect cloud

---

## 6. Results Analysis

### Locate Results

Results are saved to:

```
lean/backtests/[timestamp]/
├── report.html              # Visual report with charts
├── statistics.json          # Detailed metrics
├── orders.json              # All orders executed
└── log.txt                  # Algorithm logs
```

### View Report

```bash
# Open HTML report in browser
open lean/backtests/[latest]/report.html

# Or on Linux
xdg-open lean/backtests/[latest]/report.html
```

### Key Metrics to Review

#### Returns
- Total return
- Annual return
- Monthly returns
- Daily returns distribution

#### Risk-Adjusted Performance
- **Sharpe Ratio**: Target > 1.4
- **Sortino Ratio**: Downside risk-adjusted
- **Calmar Ratio**: Return/max drawdown

#### Drawdown
- **Maximum Drawdown**: Target < 20%
- Drawdown duration
- Underwater periods

#### Market Correlation
- **Beta vs SPY**: Target < 0.15 (market-neutral)
- **Alpha**: Excess returns
- **Correlation**: Low correlation desired

#### Trading Statistics
- Total trades
- **Win Rate**: Target ~50%
- Average win/loss
- Profit factor
- Turnover

### Compare Against Targets

| Metric | Target | Your Result |
|--------|--------|-------------|
| Annual Return | 20-28% | ? |
| Sharpe Ratio | >1.4 | ? |
| Max Drawdown | <20% | ? |
| Market Correlation | <0.15 | ? |
| Win Rate | ~50% | ? |

### Extract Statistics Programmatically

```python
import json

# Load statistics
with open('lean/backtests/[timestamp]/statistics.json', 'r') as f:
    stats = json.load(f)

# Print key metrics
print(f"Total Return: {stats['TotalReturn']}")
print(f"Sharpe Ratio: {stats['SharpeRatio']}")
print(f"Max Drawdown: {stats['MaxDrawdown']}")
```

---

## 7. Optimization

Based on backtest results, optimize the strategy:

### A. Hyperparameter Tuning

#### Model Parameters

Go back to training notebook and adjust:

```python
# In model_training_and_evaluation.ipynb

# XGBoost parameters
CONFIG['models']['xgboost_params'] = {
    'n_estimators': 200,      # Try: 100, 200, 300
    'max_depth': 5,           # Try: 3, 5, 7
    'learning_rate': 0.05,    # Try: 0.01, 0.05, 0.1
    'subsample': 0.8,
    'colsample_bytree': 0.8
}
```

Retrain → Export → Retest

#### Strategy Parameters

Edit `lean/main.py`:

```python
# Try different portfolio sizes
self.n_long = 30        # Try: 15, 20, 30, 40
self.n_short = 30

# Try different holding periods
self.holding_days = 5   # Try: 2, 3, 5, 7

# Try different position sizing
self.max_position_size = 0.03  # Try: 0.02, 0.03, 0.04, 0.05
```

### B. Feature Engineering

Add new features in `src/feature_engineering.py`:

```python
# Example: Add sector momentum
def calculate_sector_momentum(self, df):
    # Group by sector and calculate relative momentum
    pass

# Example: Add volume shock indicator
def calculate_volume_shock(self, df):
    # Identify unusual volume spikes
    pass
```

Retrain with new features → Export → Retest

### C. Risk Management

Enhance risk controls in `lean/main.py`:

```python
def CalculatePortfolioRisk(self):
    """Calculate current portfolio risk"""
    # Add sector concentration limits
    # Add factor exposure limits
    # Add volatility-based position sizing
    pass
```

### D. Universe Selection

Adjust universe filters:

```python
def CoarseSelectionFunction(self, coarse):
    # Try different size cutoffs
    filtered = [x for x in coarse if 
                x.Price > 10.0 and           # Higher minimum price
                x.DollarVolume > 20000000]   # Higher volume requirement
```

### Optimization Workflow

```
1. Identify underperforming aspect (e.g., high drawdown)
2. Hypothesize improvement (e.g., reduce position sizes)
3. Make change (edit code)
4. Retrain if needed (if model/features changed)
5. Retest (run backtest)
6. Compare results
7. Keep improvement or revert
8. Repeat
```

### Track Experiments

Keep a log of experiments:

```
experiments.md:

## Experiment 1: Increase holding period
- Date: 2024-01-23
- Change: holding_days 3 → 5
- Result: Sharpe 1.2 → 1.4, turnover reduced
- Decision: ✓ Keep

## Experiment 2: Add volume shock feature
- Date: 2024-01-24
- Change: Added volume_shock feature
- Result: Test R² 0.045 → 0.048, marginal improvement
- Decision: ✓ Keep
```

---

## Troubleshooting

### Common Issues

#### 1. "Model file not found" in Lean

**Problem**: Lean can't find trained_model.pkl

**Solution**:
```bash
# Verify file exists
ls -la lean/trained_model.pkl

# Re-export if missing
python export_model_for_lean.py --model-path models/xgboost_best.pkl
```

#### 2. Training notebook crashes (memory error)

**Problem**: Not enough RAM for full dataset

**Solution**:
```python
# Reduce universe size
CONFIG['data']['universe_size'] = 50  # Instead of 100

# Or reduce time period
CONFIG['data']['train_years'] = 5  # Instead of 10
```

#### 3. Backtest very slow

**Problem**: Large universe or high-frequency rebalancing

**Solution**:
```python
# In lean/main.py
# Reduce universe
return [x.Symbol for x in sorted_by_volume[:100]]  # Instead of 300

# Increase holding period
self.holding_days = 5  # Instead of 3
```

#### 4. Poor backtest performance

**Problem**: Model not performing well out-of-sample

**Solution**:
- Check for overfitting (compare train vs test R²)
- Try simpler models (Ridge instead of XGBoost)
- Add more training data
- Review feature engineering
- Check for data leakage

#### 5. "Insufficient buying power" errors

**Problem**: Position sizes too large

**Solution**:
```python
# In lean/main.py
self.n_long = 15          # Reduce positions
self.n_short = 15
self.max_position_size = 0.03  # Smaller position sizes
```

---

## Next Steps

After successful backtesting:

### 1. Paper Trading

Deploy to paper trading to test in real-time:

```bash
# With QuantConnect
# 1. Go to live trading page
# 2. Select paper trading
# 3. Deploy algorithm
# 4. Monitor performance
```

### 2. Walk-Forward Analysis

Test robustness with walk-forward optimization:

```python
# Train on 2015-2019
# Test on 2020

# Train on 2016-2020
# Test on 2021

# Train on 2017-2021
# Test on 2022

# Evaluate consistency across periods
```

### 3. Monte Carlo Simulation

Test robustness to different market conditions:

```python
# Shuffle trade sequences
# Analyze distribution of outcomes
# Calculate probability of meeting targets
```

### 4. Production Deployment

When ready for live trading:

1. Finalize risk management rules
2. Set up monitoring and alerts
3. Define circuit breakers
4. Start with small capital
5. Scale gradually based on performance

---

## Summary Checklist

- [ ] Environment setup complete
- [ ] Explored data with exploratory notebook
- [ ] Trained multiple ML models
- [ ] Identified best model
- [ ] Exported model for Lean
- [ ] Ran backtest successfully
- [ ] Analyzed results
- [ ] Compared against targets
- [ ] Optimized based on results
- [ ] Documented experiments
- [ ] Ready for paper trading (if performance meets targets)

---

## Resources

- **QuantConnect Docs**: https://www.quantconnect.com/docs
- **Lean GitHub**: https://github.com/QuantConnect/Lean
- **Forum**: https://www.quantconnect.com/forum
- **Project README**: ../README.md
- **Lean README**: ../lean/README.md

---

## Support

For issues or questions:

1. Check troubleshooting section above
2. Review Lean documentation
3. Check QuantConnect forum
4. Review project README files

**Good luck with your statistical arbitrage strategy! 🚀**
