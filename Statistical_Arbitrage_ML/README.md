# Market-Neutral Statistical Arbitrage Strategy using Machine Learning

## Overview

This research project implements a market-neutral statistical arbitrage trading strategy based on machine learning predictions of short-term stock returns. The strategy exploits temporary price discrepancies in the Russell 3000 universe using a mean-reversion framework.

## Strategy Objective

- **Goal**: Generate market-neutral, uncorrelated returns by exploiting temporary price discrepancies
- **Core Principle**: Mean reversion of relative prices
- **Method**: Machine learning forecasts of 2-3 day returns to construct long/short portfolios

## Key Features

### 1. Data & Universe
- **Stock Universe**: Russell 3000 constituents
- **Data Requirements**: Survivorship-bias-free historical data
- **Holding Period**: 2-3 days
- **Rebalancing**: Daily

### 2. Predictive Features

#### Momentum Indicators
- Price rate-of-change over multiple timeframes (short, medium, long-term up to 1 year)

#### Mean Reversion Indicators
- Distance from moving averages over various periods (up to 1 year)
- Standardized as percentage deviations

#### Volume Indicators
- Recent trading volume vs. 6-month average

### 3. Machine Learning Model

- **Task Type**: Regression (predicting continuous returns)
- **Target Variable**: 3-day log return
- **Training Regime**: Rolling window retraining
  - Window Size: 10 years of historical data
  - Retraining Frequency: Annual
- **Model Types**: Tested various algorithms (linear to complex)

### 4. Portfolio Construction

#### Position Selection
- **Long**: Top 10-20 stocks with highest predicted returns
- **Short**: Bottom 10-20 stocks with lowest predicted returns

#### Risk Management
- **Position Size Limit**: 3-4% per stock
- **Capital Allocation**: 2-3 staggered portfolios
- **Stock Exclusions**:
  - Penny stocks
  - Biotech stocks
  - "Meme" stocks (high unpredictable volatility)

### 5. Target Performance Metrics

| Metric | Target |
|--------|--------|
| Annual Return | 20-28% |
| Sharpe Ratio | >1.4 |
| Maximum Drawdown | <20% |
| Market Correlation | <0.15 (S&P 500) |
| Win Rate | ~50% |
| Profit/Loss Ratio | >1.2 |

### 6. Risk-Adjusted Performance

- **Factor Analysis**: Fama-French 3-Factor Model
- **Target Alpha**: High and statistically significant
- **Target R-squared**: <0.02 (low factor exposure)

## Implementation Structure

```
Statistical_Arbitrage_ML/
├── README.md                              # This file
├── requirements.txt                       # Python dependencies
├── config.yaml                            # Configuration parameters
├── export_model_for_lean.py              # Export trained model for Lean
├── src/
│   ├── __init__.py
│   ├── data_acquisition.py               # Data sourcing and preparation
│   ├── feature_engineering.py            # Feature calculation
│   ├── model_trainer.py                  # ML model training
│   ├── portfolio_builder.py              # Portfolio construction logic
│   ├── risk_manager.py                   # Risk management rules
│   └── utils.py                          # Utility functions
├── notebooks/
│   ├── exploratory_analysis.ipynb        # Data exploration
│   └── model_training_and_evaluation.ipynb  # Complete training pipeline
├── lean/                                  # QuantConnect Lean backtesting
│   ├── main.py                           # Lean algorithm implementation
│   ├── config.json                       # Lean configuration
│   ├── README.md                         # Lean setup guide
│   └── requirements.txt                  # Lean dependencies
├── models/                                # Saved trained models
└── tests/
    └── test_strategy.py
```

# Statistical Arbitrage ML Research - Quick Start Guide

## Project Overview

A complete implementation of a market-neutral statistical arbitrage strategy using machine learning to predict short-term stock returns and construct long/short portfolios.

## Project Structure

```
Statistical_Arbitrage_ML/
├── README.md                    # Comprehensive documentation
├── requirements.txt             # Python dependencies
├── config.yaml                  # Strategy configuration
├── main.py                      # Main execution script
├── src/
│   ├── __init__.py
│   ├── data_acquisition.py     # Data sourcing & preparation
│   ├── feature_engineering.py  # Feature calculation
│   ├── model_trainer.py        # ML model training
│   ├── portfolio_builder.py    # Portfolio construction
│   ├── backtester.py           # Backtesting engine
│   ├── risk_manager.py         # Risk management
│   └── utils.py                # Utility functions
├── notebooks/
│   └── exploratory_analysis.ipynb
└── data/                       # Data cache directory
```

## Installation

```bash
# Navigate to project directory
cd Statistical_Arbitrage_ML

# Install dependencies
pip install -r requirements.txt

# Note: Some packages may require additional system dependencies
# For TA-Lib on Ubuntu/Debian:
# sudo apt-get install build-essential python3-dev
```

## System Architecture

### 1. Run a Backtest

```bash
python main.py --mode backtest --start-date 2020-01-01 --end-date 2023-12-31
```

This will:
- Download historical data for Russell 3000 stocks
- Calculate momentum, mean reversion, and volume features
- Train an XGBoost model on rolling windows
- Build long/short portfolios
- Generate performance metrics and visualizations

### 2. View Results

After running, check the `results/` directory for:
- `backtest_report.txt` - Performance summary
- `equity_curve.png` - Equity curve visualization
- `returns_distribution.png` - Returns analysis
- `trades.csv` - All executed trades
- `equity_curve.csv` - Daily equity values

### 3. Customize Strategy

Edit `config.yaml` to adjust:

```yaml
# Change model type
model:
  type: "lightgbm"  # or "random_forest", "ridge", etc.

# Adjust portfolio size
portfolio:
  n_long: 30
  n_short: 30

# Modify features
features:
  momentum_periods: [5, 10, 20, 60, 126, 252]
  ma_periods: [10, 20, 50, 100, 200]
```

## Key Components

### 1. Data Acquisition (`data_acquisition.py`)

- Downloads OHLCV data for Russell 3000 universe
- Applies liquidity filters (min volume, price)
- Excludes high-risk stocks (biotech, penny stocks)
- Caches data locally for faster access

**Key function:**
```python
engine = DataAcquisitionEngine()
df = engine.get_training_data(tickers, start_date, end_date)
```

### 2. Feature Engineering (`feature_engineering.py`)

Calculates:
- **Momentum**: Rate of change over 5, 10, 20, 60, 126, 252 days
- **Mean Reversion**: Distance from moving averages
- **Volume**: Relative volume vs 6-month average
- **Volatility**: Historical volatility measures

**Key function:**
```python
engineer = FeatureEngineer()
df_features = engineer.calculate_all_features(df)
X, y = engineer.prepare_ml_dataset(df_features)
```

### 3. Model Training (`model_trainer.py`)

- Supports multiple algorithms (XGBoost, LightGBM, Random Forest, etc.)
- Rolling window training (10-year windows, annual retraining)
- Feature scaling and validation
- Model persistence

**Key function:**
```python
trainer = ModelTrainer(model_type='xgboost')
metrics = trainer.train(X_train, y_train, X_val, y_val)
predictions = trainer.predict(X_new)
```

### 4. Portfolio Construction (`portfolio_builder.py`)

- Ranks stocks by predicted returns
- Selects top N for long, bottom N for short
- Equal-weight or custom weighting
- Position size limits (4% max per stock)
- Staggered portfolios for smooth entry/exit

**Key function:**
```python
builder = PortfolioBuilder(n_long=20, n_short=20)
portfolio = builder.build_portfolio(predictions, prices, date)
```

### 5. Backtesting (`backtester.py`)

- Simulates historical performance
- Transaction costs (10 bps default)
- Comprehensive metrics (Sharpe, drawdown, etc.)
- Fama-French factor analysis
- Visualization tools

**Key function:**
```python
backtester = Backtester(initial_capital=1_000_000)
results = backtester.run_backtest(portfolios, price_data)
```

## Strategy Parameters

### Target Performance (from video source)

| Metric | Target |
|--------|--------|
| Annual Return | 20-28% |
| Sharpe Ratio | >1.4 |
| Max Drawdown | <20% |
| Market Correlation | <0.15 |
| Win Rate | ~50% |

### Default Configuration

- **Universe**: Russell 3000
- **Holding Period**: 3 days
- **Positions**: 20 long / 20 short
- **Capital**: $1,000,000
- **Max Position**: 4% per stock
- **Transaction Cost**: 10 basis points

## Example Usage

### Python Script

```python
from src.data_acquisition import DataAcquisitionEngine
from src.feature_engineering import FeatureEngineer
from src.model_trainer import ModelTrainer
from src.portfolio_builder import PortfolioBuilder
from datetime import datetime, timedelta

# 1. Get data
engine = DataAcquisitionEngine()
universe = engine.get_russell_3000_universe()[:100]  # Sample
df = engine.get_training_data(
    universe, 
    start_date=datetime(2020, 1, 1),
    end_date=datetime(2023, 12, 31)
)

# 2. Calculate features
engineer = FeatureEngineer()
df_features = engineer.calculate_all_features(df)
X, y = engineer.prepare_ml_dataset(df_features)

# 3. Train model
trainer = ModelTrainer(model_type='xgboost')
trainer.train(X, y)

# 4. Generate predictions
predictions = trainer.predict(X)

# 5. Build portfolio
builder = PortfolioBuilder(n_long=20, n_short=20)
portfolio = builder.build_portfolio(predictions, prices, date)
```

### Command Line Options

```bash
# Full backtest
python main.py --mode backtest --start-date 2020-01-01 --end-date 2023-12-31

# Use custom config
python main.py --mode backtest --config my_config.yaml

# Train model only
python main.py --mode train --window-years 10

# Generate predictions (for live trading)
python main.py --mode predict --date 2024-01-15
```

## Performance Monitoring

The strategy tracks:

1. **Return Metrics**
   - Total return
   - Annual return
   - Sharpe ratio
   - Sortino ratio

2. **Risk Metrics**
   - Maximum drawdown
   - Volatility
   - Market correlation
   - Beta

3. **Trade Statistics**
   - Win rate
   - Average win/loss
   - Profit factor
   - Total trades

4. **Factor Analysis**
   - Fama-French 3-factor model
   - Alpha (excess returns)
   - R-squared

## Important Notes

### Data Requirements

- **Survivorship bias**: Use quality data provider (Norgate Data recommended)
- **Historical constituents**: Russell 3000 composition changes over time
- **Corporate actions**: Data should be adjusted for splits/dividends

### Production Considerations

1. **Infrastructure**: Real-time data feeds, execution system
2. **Transaction costs**: Slippage, market impact, borrowing costs
3. **Model decay**: Regular retraining required
4. **Risk management**: Position limits, circuit breakers
5. **Compliance**: Regulatory requirements for short selling

## Troubleshooting

### Common Issues

1. **Missing data**: Some tickers may not have complete history
   - Solution: Increase universe size or use data with fewer gaps

2. **Memory errors**: Large universe with long history
   - Solution: Process in batches or reduce universe size

3. **Poor performance**: Model not predicting well
   - Solution: Try different model types, tune hyperparameters

4. **High transaction costs**: Frequent rebalancing
   - Solution: Increase holding period or reduce turnover

## Next Steps

1. **Optimize**: Hyperparameter tuning, feature selection
2. **Enhance**: Add alternative data, sentiment analysis
3. **Scale**: Larger universe, multiple strategies
4. **Deploy**: Paper trading, live execution

## References

- **Source**: Quant Radio Podcast - "Market Neutral Trading Strategy using Statistical Arbitrage"
- **Methodology**: Machine learning for mean reversion prediction
- **Framework**: Fama-French factor model

## Disclaimer

**Educational purposes only.** Not investment advice. Past performance does not guarantee future results. Requires robust infrastructure and risk management for live trading.

## Support

For issues or questions:
1. Check README.md for detailed documentation
2. Review config.yaml for parameter descriptions
3. Examine example notebooks for usage patterns
4. Consult source modules for API documentation
Workflow Overview

**This project uses a two-step workflow:**

1. **Training Phase**: Use Jupyter notebooks for model development and training
2. **Backtesting Phase**: Use QuantConnect Lean for robust backtesting

### Step 1: Environment Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Install Lean CLI for backtesting
pip install lean
```

### Step 2: Model Training

Use the comprehensive training notebook:

```bash
# Open Jupyter
jupyter notebook notebooks/model_training_and_evaluation.ipynb

# Or use VS Code with Jupyter extension
```

The notebook includes:
- Data acquisition and quality analysis
- Feature engineering and visualization
- Training multiple ML models (Ridge, Random Forest, XGBoost, LightGBM)
- Model comparison and evaluation
- Feature importance analysis
- Portfolio construction simulation
- Model persistence

### Step 3: Export Model for Backtesting

After training, export the model for Lean:

```bash
python export_model_for_lean.py --model-path models/xgboost_best.pkl
```

This creates:
- `lean/trained_model.pkl` - The trained model
- `lean/feature_names.txt` - Feature names in correct order
- `lean/model_metadata.json` - Model metadata

### Step 4: Run Backtest with QuantConnect Lean

```bash
# Initialize Lean (first time only)
lean init

# Run backtest
lean backtest "lean"

# Or specify custom parameters
lean backtest "lean" --start 20200101 --end 20231231
```

### Step 5: Analyze Results

Lean generates:
- HTML report with performance charts
- Detailed statistics (Sharpe, drawdown, returns)
- Trade-by-trade analysis
- Factor exposures

Results are saved in `lean/backtests/` directory.
### 5. Generate Predictions
```bash
python main.py --mode predict --date 2024-01-15
```

## Key Considerations

### Backtesting Framework

**Why QuantConnect Lean?**

This project uses QuantConnect Lean for backtesting instead of custom code because Lean provides:

1. **Realistic Simulation**: Proper handling of market microstructure, fills, and execution
2. **Transaction Costs**: Built-in models for commissions, slippage, and market impact
3. **Data Quality**: Professional-grade historical data with survivorship-bias adjustments
4. **Risk Management**: Real-time margin and buying power calculations
5. **Performance Metrics**: Industry-standard performance analytics
6. **Production Ready**: Same codebase can be deployed to paper/live trading

### Model Retraining
- AnnAdvanced ML Models**: 
   - Deep learning architectures (LSTM, Transformers)
   - Ensemble methods combining multiple models
   - Online learning for continuous adaptation

2. **Enhanced Features**: 
   - Alternative data (sentiment, news, satellite imagery)
   - Cross-sectional factors
   - Market microstructure signals

3. **Dynamic Position Sizing**: 
   - Kelly criterion
   - Signal strength-based weighting
   - Risk parity allocation

4. **Risk Management**: 
   - Sector exposure limits
   - Factor exposure constraints
   - Dynamic leverage adjustment

5. **Execution Optimization**: 
   - Smart order routing
   - VWAP/TWAP execution
   - Minimize market impact

6. **Multi-Strategy Approach**: 
   - Combine with other uncorrelated strategies
   - Dynamic strategy allocation
   - Risk budgeting across strategie
- Daily rebalancing maintains long/short balance
- Low correlation to broad market indices
- Target: <0.15 correlation with S&P 500

## Potential Improvements

1. **Weighting Schemes**: Optimize position weights beyond equal-weight
2. **Cash Management**: Earn interest on uninvested capital
3. **Universe Refinement**: Focus on more liquid or specific sectors
4. **Feature Enhancement**: Add alternative data sources
5. **Ensemble Models**: Combine multiple ML algorithms

## Data Sources

- **Primary**: Norgate Data (survivorship-bias-free)
- **Alternative**: 
  - Databento
  - Polygon.io
  - Alpha Vantage

## Disclaimer

**IMPORTANT**: This implementation is for educational and research purposes only, based on historical backtesting analysis from Quant Radio podcast. 

- Not intended as investment advice
- Past performance does not guarantee future results
- Live trading requires robust infrastructure, real-time data, and careful consideration of:
  - Transaction costs
  - Slippage
  - Model decay
  - Regulatory compliance

## References

- **Source**: Quant Radio Podcast - "Market Neutral Trading Strategy using Statistical Arbitrage"
- **Factor Model**: Fama-French 3-Factor Model
- **Universe**: Russell 3000 Index


## License

See LICENSE file in repository root.
