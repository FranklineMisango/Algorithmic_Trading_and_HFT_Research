# Deep Learning Momentum Trading Strategy

Implementation of "Applying Deep Learning to Enhance Momentum Trading Strategies in Stocks" (Taki & Lee, 2013)

## Overview

This project implements a deep learning approach to momentum trading using a Feed-Forward Neural Network (FFNN) to rank stocks based on predicted outperformance probability. The strategy uses cross-sectional analysis and binary classification to identify which stocks will outperform the median return.

## Key Results (From Original Study)

- **Annualized Return**: 12.8% (vs S&P 500: 7.0%)
- **Sharpe Ratio**: 1.03 (vs S&P 500: 0.5)
- **Maximum Drawdown**: 24% (vs S&P 500: 52.6%)
- **Market Correlation**: Negative (diversification benefit)
- **Prediction Accuracy**: ~52% (but ranking creates profitable spread)

## Methodology

### 1. Data Preparation
- **Universe**: US stocks (NYSE, AMEX, NASDAQ)
- **Filter**: Price > $5 to avoid microstructure noise
- **Period**: 1990-present (daily data)

### 2. Feature Engineering (33 Features)
- **12 Long-Term Features**: Monthly returns from t-13 to t-2
- **20 Short-Term Features**: Daily returns from most recent month (t)
- **1 Anomaly Feature**: January Effect dummy variable

### 3. Preprocessing
- **Cross-Sectional Z-Score Standardization**: Each day, normalize features relative to the cross-section of all stocks

### 4. Target Definition
- **Binary Classification**: 
  - Label 1: Stock return > median return next month
  - Label 0: Stock return ≤ median return next month

### 5. Model Architecture
- **Type**: Feed-Forward Neural Network
- **Key Feature**: Bottleneck layer (4 units) for compressed representation
- **Training**: End-to-end with Adam optimizer
- **Loss**: Binary cross-entropy

### 6. Validation
- **Rolling Window Cross-Validation**:
  - Train on historical chunk
  - Validate on next chunk (model selection)
  - Test on subsequent chunk (performance evaluation)
  - Roll forward and repeat

### 7. Trading Strategy
- Rank stocks by predicted probability
- Sort into 10 quantiles (deciles)
- **LONG**: Top quantile (Q10) - most confident to outperform
- **SHORT**: Bottom quantile (Q1) - least confident to outperform
- **Rebalance**: Monthly

## Project Structure

```
Deep_Learning_Momentum/
├── README.md
├── requirements.txt
├── config.yaml
├── data_processor.py       # Data acquisition & feature engineering
├── model.py                # Neural network architecture
├── trainer.py              # Training & rolling window validation
├── strategy.py             # Trading signal generation & execution
├── evaluator.py            # Performance metrics & visualization
├── notebooks/
│   ├── 1_data_preparation.ipynb
│   ├── 2_model_training.ipynb
│   ├── 3_strategy_backtesting.ipynb
│   └── 4_end_to_end_pipeline.ipynb
├── results/
└── models/
```

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

### Option 1: Run Individual Notebooks
1. `1_data_preparation.ipynb` - Download and prepare data
2. `2_model_training.ipynb` - Train the neural network
3. `3_strategy_backtesting.ipynb` - Backtest the strategy
4. `4_end_to_end_pipeline.ipynb` - Complete workflow

### Option 2: Run Python Pipeline
```python
from data_processor import DataProcessor
from model import MomentumRanker
from trainer import RollingWindowTrainer
from strategy import LongShortStrategy
from evaluator import PerformanceEvaluator

# See notebooks for detailed examples
```

## Key Concepts

### The Bitter Lesson
This implementation demonstrates Richard Sutton's "Bitter Lesson": general-purpose, computation-heavy methods (deep learning) applied to relatively raw data can discover patterns without complex hand-crafted financial rules. The value is in the model's ability to **rank stocks consistently**, not predict returns with 100% accuracy.

### Cross-Sectional Analysis
By standardizing features relative to the cross-section of stocks each day, the model learns to identify **relative outperformance**, which is more stable than predicting absolute returns.

### Bottleneck Architecture
The bottleneck layer forces the network to learn a compressed, efficient representation of the 33 input features, focusing on the most salient patterns for momentum prediction.

## Important Notes

⚠️ **Disclaimer**: This is for educational and research purposes only. Not investment advice.

⚠️ **Replication**: Results will differ from the original paper due to:
- Different data sources and time periods
- Market evolution
- End-to-end training vs. original RBM pre-training
- Potential survivorship bias

⚠️ **Transaction Costs**: Backtests don't account for:
- Trading commissions
- Slippage
- Market impact
- Borrowing costs for shorts

## References

1. Taki, D., & Lee, A. (2013). "Applying Deep Learning to Enhance Momentum Trading Strategies in Stocks"
2. Sutton, R. (2019). "The Bitter Lesson"
3. Jegadeesh, N., & Titman, S. (1993). "Returns to Buying Winners and Selling Losers"

## License

MIT License - See LICENSE file
