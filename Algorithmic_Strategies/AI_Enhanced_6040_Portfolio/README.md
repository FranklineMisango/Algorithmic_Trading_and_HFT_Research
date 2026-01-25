# AI-Enhanced 60/40 Portfolio Strategy

An AI-driven portfolio allocation system that modernizes the classic 60/40 portfolio using machine learning to dynamically adjust allocations based on economic indicators and market conditions.

## Overview

This strategy implements a decision tree regression model that analyzes key economic indicators (VIX, Yield Spread, Interest Rates) to make monthly portfolio rebalancing decisions across multiple asset classes including:
- **Traditional Assets**: Stocks (SPY), Bonds (TLT)
- **Alternative Assets**: Bitcoin (BTC-USD), Gold (GLD)

### Key Features
- **Machine Learning**: Decision tree models predict optimal asset allocations
- **Economic Indicators**: VIX, Yield Spread, Interest Rates drive decisions
- **Monthly Rebalancing**: Systematic portfolio adjustment
- **Performance Metrics**: Comprehensive risk-adjusted return analysis
- **Bias Mitigation**: Data-driven, unbiased decision framework

## Research Foundation

Based on quantitative research exploring how AI can overcome traditional portfolio limitations:
- **Problem**: Traditional 60/40 portfolios fail when stock-bond correlations break down
- **Solution**: AI dynamically identifies and allocates to complementary assets
- **Method**: Decision tree regression on economic indicators
- **Goal**: Improve risk-adjusted returns (Sharpe ratio) while managing drawdowns

## Project Structure

```
AI_Enhanced_6040_Portfolio/
├── config.yaml                          # Configuration parameters
├── requirements.txt                     # Python dependencies
├── README.md                           # This file
│
├── data_acquisition.py                 # Market data & indicator fetching
├── feature_engineering.py              # Economic indicator processing
├── ml_model.py                        # Decision tree ML models
├── backtester.py                      # Portfolio backtesting engine
├── main.py                            # Main orchestration script
│
├── 01_data_exploration.ipynb          # Data analysis notebook
├── 02_model_training_evaluation.ipynb # Model training notebook
├── 03_backtest_analysis.ipynb         # Backtest results notebook
├── 04_sensitivity_analysis.ipynb      # Parameter sensitivity analysis
│
├── results/                           # Output results
├── figures/                           # Generated visualizations
└── models/                            # Saved ML models
```

## Quick Start

### Prerequisites
- Python 3.8+
- pip package manager

### Installation

```bash
# Navigate to project directory
cd Algorithmic_Strategies/AI_Enhanced_6040_Portfolio

# Install dependencies
pip install -r requirements.txt
```

### Running the Strategy

#### Option 1: Run Complete Pipeline (Python Script)
```bash
python main.py
```

This will:
1. Fetch market data and economic indicators
2. Engineer features
3. Train ML models for each asset
4. Generate portfolio allocations
5. Run backtests
6. Create visualizations
7. Save results to `results/`, `figures/`, and `models/` directories

#### Option 2: Interactive Analysis (Jupyter Notebooks)

```bash
# Start Jupyter
jupyter notebook

# Open notebooks in order:
# 1. 01_data_exploration.ipynb
# 2. 02_model_training_evaluation.ipynb
# 3. 03_backtest_analysis.ipynb
# 4. 04_sensitivity_analysis.ipynb
```

## Configuration

Edit `config.yaml` to customize:

### Data Parameters
```yaml
data:
  start_date: "2017-01-01"
  end_date: "2024-12-31"
  lookback_period: 12  # months
  rebalance_frequency: "M"  # Monthly
```

### Asset Universe
```yaml
assets:
  traditional:
    - ticker: "SPY"
    - ticker: "TLT"
  alternative:
    - ticker: "BTC-USD"
      max_allocation: 0.30  # Maximum 30%
    - ticker: "GLD"
      max_allocation: 0.20
```

### Model Parameters
```yaml
model:
  type: "DecisionTreeRegressor"
  parameters:
    max_depth: 5
    min_samples_split: 10
    min_samples_leaf: 5
```

### Economic Indicators
```yaml
indicators:
  vix:
    ticker: "^VIX"
  yield_spread:
    long_term: "^TNX"   # 10-year Treasury
    short_term: "^IRX"  # 3-month Treasury
  interest_rate:
    ticker: "DFF"       # Fed Funds Rate
```

## Methodology

### 1. Data Acquisition
- Fetch historical prices for all assets
- Collect economic indicators (VIX, Yield Spread, Interest Rates)
- Resample to monthly frequency

### 2. Feature Engineering
Creates multiple feature types:
- **Lagged Features**: Historical values (1, 3, 6 months)
- **Rolling Features**: Moving averages, std dev, min/max
- **Change Features**: Absolute and percentage changes
- **Interaction Features**: VIX × Yield Spread, Rate × VIX, etc.
- **Regime Features**: High volatility, inverted curve, high rates

### 3. Model Training
- **One model per asset**: Separate decision tree for each asset's return prediction
- **Time series split**: Respects temporal ordering (80% train, 20% test)
- **Cross-validation**: 5-fold time series CV for robustness

### 4. Portfolio Allocation
- Predict next-period returns for all assets
- Calculate optimal weights based on predictions
- Apply constraints (max allocation limits)
- Normalize to sum to 100%

### 5. Backtesting
- Monthly rebalancing
- Transaction costs (0.1% per trade)
- Compare against benchmarks:
  - Buy & Hold SPY
  - Traditional 60/40 (60% SPY, 40% TLT)

## Performance Metrics

The strategy evaluates performance using:

- **Sharpe Ratio**: Risk-adjusted returns
- **Sortino Ratio**: Downside risk-adjusted returns
- **CAGR**: Compound Annual Growth Rate
- **Max Drawdown**: Largest peak-to-trough decline
- **Calmar Ratio**: CAGR / Max Drawdown
- **Volatility**: Annualized standard deviation
- **Directional Accuracy**: % of correct return direction predictions

## Jupyter Notebooks

### 01_data_exploration.ipynb
- Visualize asset prices and returns
- Analyze economic indicators
- Explore correlations
- Identify market regimes

### 02_model_training_evaluation.ipynb
- Feature engineering process
- Model training for all assets
- Feature importance analysis
- Cross-validation results
- Prediction quality assessment

### 03_backtest_analysis.ipynb
- Generate AI-driven allocations
- Run comprehensive backtest
- Compare vs benchmarks
- Portfolio value analysis
- Drawdown analysis
- Returns distribution

### 04_sensitivity_analysis.ipynb
- Lookback period sensitivity
- Bitcoin allocation limits
- Model hyperparameter tuning
- Combined parameter optimization
- Robustness testing

## Key Insights

### AI Advantages
1. **Dynamic Diversification**: Adapts to changing correlations
2. **Bias Mitigation**: Removes human confirmation bias
3. **Signal Integration**: Combines multiple indicators effectively
4. **Regime Adaptation**: Responds to different market environments

### Risk Considerations
- **Transaction Costs**: Monthly rebalancing incurs costs
- **Model Risk**: Performance depends on model accuracy
- **Market Changes**: Future regimes may differ from training period
- **Alternative Assets**: Higher volatility (especially Bitcoin)

## Module Details

### data_acquisition.py
- Fetches asset prices using yfinance
- Retrieves economic indicators (VIX, Treasuries, Fed Funds Rate)
- Handles missing data and resampling
- Aligns dates across all data sources

### feature_engineering.py
- Creates 50+ engineered features
- Implements lagged, rolling, change, interaction, and regime features
- Handles NaN values from transformations
- Provides feature scaling capabilities

### ml_model.py
- Trains separate decision tree for each asset
- Implements prediction and allocation logic
- Calculates feature importance
- Provides model persistence (save/load)
- Applies allocation constraints

### backtester.py
- Simulates portfolio performance
- Calculates transaction costs
- Computes comprehensive performance metrics
- Generates performance visualizations
- Supports multiple strategy comparison

### main.py
- Orchestrates entire workflow
- Provides progress tracking
- Saves all outputs
- Generates summary reports

## Output Files

### results/
- `allocations.csv`: Portfolio allocations over time
- `performance_comparison.csv`: Metrics for all strategies
- `model_evaluation.csv`: Model performance metrics

### figures/
- `portfolio_value.png`: Portfolio value comparison
- `drawdown.png`: Drawdown analysis
- `monthly_returns.png`: Returns distribution
- `allocations.png`: Allocation evolution
- `feature_importance.png`: Most important features

### models/
- `{asset}_model.pkl`: Trained models for each asset

## Customization

### Adding New Assets
1. Add ticker to `config.yaml` under `assets`
2. Set max allocation if alternative asset
3. Run the strategy - models will automatically train for new asset

### Adding New Indicators
1. Add indicator configuration to `config.yaml`
2. Implement fetching logic in `data_acquisition.py`
3. Features will be auto-generated in `feature_engineering.py`

### Changing Rebalancing Frequency
1. Modify `rebalance_frequency` in `config.yaml`
2. Adjust `periods_per_year` in backtester metrics

### Testing Different Models
1. Modify `model` section in `config.yaml`
2. Update model initialization in `ml_model.py`
3. Ensure compatibility with prediction interface

## Dependencies

Key libraries:
- **Data**: `pandas`, `numpy`, `yfinance`, `pandas-datareader`
- **ML**: `scikit-learn`
- **Visualization**: `matplotlib`, `seaborn`, `plotly`
- **Backtesting**: `bt`, `quantstats`
- **Config**: `pyyaml`

See `requirements.txt` for complete list.

## Disclaimer

**Important**: This project is for educational and research purposes only.

- Past performance does not guarantee future results
- This is NOT investment advice
- Cryptocurrency and alternative assets carry high risk
- Always consult with a financial advisor
- Use at your own risk

## Contributing

This is a research project. Suggestions for improvement:
- Additional economic indicators
- Alternative ML models (Random Forest, Gradient Boosting, Neural Networks)
- Different optimization objectives
- Risk parity approaches
- Multi-period optimization

## References

### Research Framework
Based on research exploring AI applications in portfolio management:
- Dynamic asset allocation using machine learning
- Economic indicator-based trading signals
- Decision tree regression for return prediction
- Risk-adjusted performance optimization

### Data Sources
- **Market Data**: Yahoo Finance (yfinance)
- **VIX**: CBOE Volatility Index
- **Treasury Yields**: Federal Reserve Economic Data
- **Bitcoin**: Cryptocurrency market data

## Learning Outcomes

This project demonstrates:
1. End-to-end ML pipeline for finance
2. Feature engineering for time series
3. Portfolio optimization techniques
4. Backtesting methodology
5. Performance evaluation metrics
6. Sensitivity analysis and robustness testing

## Support

For questions or issues:
1. Check the Jupyter notebooks for examples
2. Review the configuration file
3. Examine module docstrings
4. Run with sample data first

## Roadmap

Potential enhancements:
- [ ] Real-time data integration
- [ ] Additional ML models comparison
- [ ] Walk-forward optimization
- [ ] Risk parity implementation
- [ ] Transaction cost optimization
- [ ] Multi-objective optimization
- [ ] Regime detection improvements
- [ ] Live trading capabilities (paper trading)

---

**Happy Backtesting!**

*Remember: The goal is to learn and understand AI-driven portfolio management, not to guarantee returns.*
