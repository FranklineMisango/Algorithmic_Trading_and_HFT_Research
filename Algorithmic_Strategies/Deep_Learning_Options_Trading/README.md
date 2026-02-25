# Deep Learning for Delta-Neutral Options Trading

Implementation of a systematic options trading strategy using LSTM neural networks to optimize delta-neutral straddle positions on S&P 100 constituents

## Strategy Overview

This strategy implements a deep learning approach to options trading that learns directly from market data to maximize risk-adjusted returns. Unlike traditional pricing models, the LSTM model is trained end-to-end to optimize the Sharpe ratio of a simulated trading strategy, with turnover regularization to account for transaction costs.

### Key Features

- **Delta-Neutral Straddles**: Focus on volatility and relative value while minimizing directional risk
- **LSTM Architecture**: Sequential model excels at capturing time-dependent patterns in options data
- **Sharpe Ratio Optimization**: Direct maximization of risk-adjusted returns during training
- **Turnover Regularization**: Penalizes excessive trading to account for transaction costs
- **Walk-Forward Validation**: Rigorous out-of-sample testing to prevent overfitting
- **Benchmark Comparison**: Performance evaluation against buy-and-hold, momentum, and mean-reversion strategies

## Research Foundation

This implementation is based on the academic research discussed in the Quant Radio podcast:

> "They trained it to maximize the Sharpe ratio... turnover regularization... it worked especially well when the transaction costs were high"

The research tested various architectures and found LSTMs performed best due to their ability to capture sequential patterns in time-series data.

## Installation

1. Clone the repository and navigate to the strategy directory:
```bash
cd Algorithmic_Strategies/Deep_Learning_Options_Trading
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Ensure you have a CUDA-compatible GPU for efficient LSTM training (optional but recommended).

## Configuration

The strategy is configured via `config.yaml`. Key parameters include:

- **Data Settings**: Date ranges, liquidity filters, S&P 100 constituents
- **Model Architecture**: LSTM layers, hidden size, dropout, regularization
- **Backtesting**: Transaction costs, slippage, position limits, risk management
- **Benchmarks**: Which comparative strategies to run

## Usage

### Full Pipeline Execution

Run the complete strategy pipeline:
```bash
python main.py --mode full --start-date 2010-01-01 --end-date 2023-12-31
```

### Individual Pipeline Steps

1. **Data Acquisition & Feature Engineering**:
```bash
python main.py --mode data --start-date 2010-01-01 --end-date 2023-12-31
```

2. **Model Training**:
```bash
python main.py --mode train
```

3. **Backtesting**:
```bash
python main.py --mode backtest
```

## Data Requirements

The strategy requires historical options data for S&P 100 constituents:

- **Underlying Prices**: Daily stock prices and returns
- **Options Data**: Daily straddle prices, strikes, expirations, implied volatility
- **Liquidity Filters**: Minimum volume and open interest thresholds
- **Time Period**: Decade-plus of historical data for robust training

## Feature Engineering

The "parsimonious feature set" includes:

1. **Moneyness**: Strike price / underlying price
2. **Time to Expiration**: Days to expiry (annualized)
3. **Premium Normalized**: Straddle price / underlying price
4. **Implied Volatility**: Market expectation of future volatility
5. **Underlying Returns**: Recent price movements (1-day, 5-day)
6. **Underlying Volatility**: Rolling 30-day volatility

## Model Architecture

### LSTM Network
- Input: 30-day sequences of engineered features
- Hidden Layers: 2 LSTM layers with 64 units each
- Dropout: 20% for regularization
- Output: Position signal between -1 and 1

### Training Objective
- **Primary Loss**: Negative Sharpe ratio (maximization through minimization)
- **Regularization**: Turnover penalty to reduce transaction costs
- **Optimization**: Adam optimizer with learning rate decay

### Validation
- Walk-forward validation: Train on 3-year windows, validate on 1-year windows
- Early stopping based on validation Sharpe ratio
- Out-of-sample performance assessment

## Backtesting Framework

### Transaction Costs
- Per-contract cost: $0.01
- Bid-ask spread: 2% of straddle price
- Slippage: Conservative model (0.1% adverse price movement)

### Risk Management
- Maximum position size: 5% of portfolio per straddle
- Maximum single stock exposure: 10% of portfolio
- Portfolio delta limit: ±10%
- Vega exposure limit: 10,000
- Drawdown stop: 20% loss triggers position closure

### Benchmarks
1. **Buy-and-Hold Options**: Simple long straddle strategy
2. **Momentum**: Bet on continuation of underlying trends
3. **Mean Reversion**: Bet against extreme moves (noted to outperform momentum over short periods)

## Performance Evaluation

### Primary Metrics
- **Sharpe Ratio**: Risk-adjusted return measure
- **Total Return**: Cumulative portfolio performance
- **Maximum Drawdown**: Peak-to-trough loss

### Secondary Metrics
- Win Rate: Percentage of profitable trades
- Profit Factor: Gross profits / gross losses
- Turnover: Annual portfolio turnover rate

### Statistical Tests
- Diebold-Mariano test for forecast accuracy comparison
- Capacity analysis for different capital levels

## Risk Management & Monitoring

### Position Limits
- Strict delta neutrality maintenance
- Vega and gamma exposure monitoring
- Sector diversification requirements

### Stress Testing
- Historical crisis periods (2008 Financial Crisis, 2020 COVID crash)
- Extreme volatility scenarios
- Liquidity stress tests

### Monitoring Dashboard
- Daily Sharpe ratio tracking
- Turnover and cost analysis
- Sector performance breakdown
- Risk metric alerts

## Implementation Notes

### Survivorship Bias Control
- Point-in-time S&P 100 constituent lists
- Only includes options for stocks actually in the index on the trade date

### Computational Requirements
- GPU recommended for LSTM training
- Memory requirements scale with historical data size
- Training time: Hours for full decade of data

### Production Considerations
- Model retraining frequency (monthly/quarterly)
- Live data feed integration
- Order execution algorithms
- Compliance and regulatory requirements

## Failure Modes & Mitigations

| Risk | Severity | Likelihood | Mitigation |
|------|----------|------------|------------|
| Model Overfitting | High | Medium | Walk-forward validation, feature parsimony, regularization |
| Regime Change | High | Medium | Continuous monitoring, meta-model for regime detection |
| Transaction Costs | High | High | Turnover regularization, smart order routing |
| Liquidity Crisis | Medium | Low | Volume/open interest filters, liquidity factors |
| Black Swan Events | Critical | Low | Strict stop-losses, cash reserves |

## Research Validation

The implementation follows the methodology described in the Quant Radio episode:

- ✅ LSTM architecture for sequential pattern recognition
- ✅ Sharpe ratio direct optimization
- ✅ Turnover regularization for cost control
- ✅ Parsimonious feature set
- ✅ Delta-neutral straddle focus
- ✅ Walk-forward validation methodology

## Results & Analysis

After running the full pipeline, results are available in:

- `results/backtest_results.json`: Complete performance metrics
- `results/performance_report.md`: Comprehensive analysis report
- `results/backtest_results.png`: Performance visualization

## Next Steps

1. **Paper Trading**: Validate in simulated live environment
2. **Live Deployment**: Start with small capital allocation
3. **Model Monitoring**: Continuous performance tracking
4. **Enhancements**: Additional features, alternative architectures
5. **Capacity Analysis**: Scale testing with larger capital

## Disclaimer

This implementation is for research and educational purposes. The strategy is based on experimental academic research and should not be considered investment advice. Past performance does not guarantee future results. Always conduct thorough due diligence and risk assessment before deploying any trading strategy with real capital.