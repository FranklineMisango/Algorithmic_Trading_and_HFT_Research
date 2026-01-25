# Statistical Arbitrage with Reinforcement Learning

## Overview

Advanced pairs trading strategy using Reinforcement Learning to predict and exploit mean-reverting price relationships between co-moving stocks. Based on research from Purdue University (Ning & Lee, 2024).

## Strategy Logic

### Core Innovation
- **Empirical Mean Reversion Time (EMRT)**: Custom data-driven metric measuring actual historical time for spread reversion
- **RL-Based Trading**: Deep Q-Network learns optimal entry/exit timing from profit/loss feedback
- **Sector-Focused**: Targets Technology, Healthcare, and Consumer Goods sectors with proven mean-reversion characteristics

### Hypothesis
Stock pairs within the same sector exhibit stable long-term equilibrium relationships. Short-term deviations are temporary and will revert to the mean. An RL agent can learn timing and direction of reversions more effectively than static model-based approaches.

### Economic Rationale
Mean-reversion driven by:
- Market microstructure and temporary liquidity imbalances
- Fundamental linkage between companies in similar industries
- Co-movement patterns ("dance partners" in same sector)

## Implementation

### 1. Pair Selection (Grid Search)
```
For each sector:
  - Calculate pairwise correlations over lookback window
  - Filter pairs with correlation > threshold
  - Calculate EMRT for each candidate pair
  - Select pairs with minimum EMRT (fastest mean reversion)
```

### 2. EMRT Calculation
```
For each pair over historical window:
  - Calculate rolling spread (log price ratio)
  - Identify deviation events (spread > 2σ)
  - Measure time until reversion to mean
  - Aggregate to empirical mean reversion time
```

### 3. RL Agent Training
- **State**: Recent price trends, spread z-score, momentum, volume ratio
- **Action**: {Buy Stock A, Sell Stock A, Hold}
- **Reward**: Realized PnL from action
- **Algorithm**: Deep Q-Network with experience replay

### 4. Execution
- Dollar-neutral positions (equal $ long/short)
- Daily rebalancing based on RL agent signals
- Strict risk controls: stop-losses, correlation monitoring, position limits

## Files

### Python Modules
- `data_acquisition.py`: Fetch S&P 500 prices and sector classifications
- `feature_engineering.py`: Calculate spreads, z-scores, momentum features
- `pair_selection.py`: Grid search for minimum EMRT pairs
- `emrt_calculator.py`: Custom EMRT metric calculation
- `rl_agent.py`: DQN implementation with profit-based rewards
- `backtester.py`: Event-driven backtest with realistic costs
- `main.py`: End-to-end pipeline orchestration

### Jupyter Notebooks
- `01_data_exploration.ipynb`: S&P 500 universe analysis, sector distributions
- `02_pair_selection.ipynb`: Correlation analysis, EMRT visualization, pair ranking
- `03_rl_training.ipynb`: Agent training, learning curves, policy evaluation
- `04_backtest_analysis.ipynb`: Performance metrics, drawdowns, sector attribution

### Configuration
- `config.yaml`: All strategy parameters
- `requirements.txt`: Python dependencies

### QuantConnect
- `lean_algorithm.py`: Event-driven implementation for production backtesting

## Research Results (Paper)
- **Simulation**: 600%+ returns in controlled experiments
- **Real-World**: Outperformed benchmarks on Sharpe ratio in 2023 S&P 500 test
- **Best Sectors**: Technology, Healthcare, Consumer Goods
- **Training Period**: 2022
- **Test Period**: 2023

## Risk Considerations

### Critical Failure Modes
1. **Model Overfitting**: RL agent learns patterns specific to training regime
   - *Mitigation*: Robust walk-forward validation across market cycles
   
2. **Structural Break**: Fundamental changes break historical correlation
   - *Mitigation*: Real-time correlation monitoring, hard stops on extreme spreads
   
3. **Data Snooping**: Pair selection influenced by hindsight
   - *Mitigation*: Purely rolling, out-of-sample pair selection
   
4. **Reward Function Limitation**: True long-term mean unknowable in practice
   - *Mitigation*: Robust mean estimators, sensitivity testing
   
5. **Sector Dependency**: Performance varies significantly by sector
   - *Mitigation*: Limit to proven sectors, develop sector-specific models

## Usage

### Training
```bash
python main.py --mode train --config config.yaml
```

### Backtesting
```bash
python main.py --mode backtest --config config.yaml
```

### Lean Backtest
```bash
lean backtest --algorithm-location Statistical_Arbitrage_RL/lean_algorithm.py
```

## Dependencies
- Python 3.9+
- PyTorch (RL agent)
- stable-baselines3 (DQN implementation)
- pandas, numpy
- yfinance (data)
- matplotlib, seaborn (visualization)

## References
- Ning, B. & Lee, K. (2024). "Advanced Statistical Arbitrage with Reinforcement Learning"
- Quant Radio Podcast: Statistical Arbitrage with Reinforcement Learning
