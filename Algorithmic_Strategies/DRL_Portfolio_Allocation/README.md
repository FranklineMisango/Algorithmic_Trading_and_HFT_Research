# Deep Reinforcement Learning for Portfolio Allocation

## Overview

A research implementation of Deep Reinforcement Learning (DRL) for multi-asset portfolio optimization. The agent learns through trial-and-error to dynamically allocate capital across stocks, bonds, commodities, and real estate to achieve superior risk-adjusted returns compared to traditional mean-variance optimization.

## Research Objective

**Goal**: Demonstrate that a DRL-based portfolio management system can outperform the Markowitz mean-variance model by achieving:
- **Higher average annual returns** (+10% vs benchmark)
- **Lower portfolio volatility** (-15% vs benchmark)
- **Superior Sharpe ratio** through adaptive, multi-period optimization

## Core Hypothesis

Traditional portfolio optimization (Markowitz) relies on static historical estimates of returns and covariances. Deep RL can:
1. **Adapt continuously** to evolving market conditions
2. **Discover non-linear relationships** between features and optimal allocations
3. **Optimize for complex objectives** (multi-period, risk-adjusted)
4. **Learn from experience** without explicit financial models

## Methodology

### 1. Reinforcement Learning Framework

**State (S_t)**:
- Current portfolio weights [w_SPY, w_AGG, w_GLD, w_VNQ]
- 20-day asset returns
- 20-day rolling volatility
- 20-day momentum signals
- Asset correlation matrix

**Action (A_t)**:
- Target portfolio weights (continuous, sum to 1.0)
- Constraints: 0% ≤ weight ≤ 40% per asset

**Reward (R_t)**:
```
R_t = log(portfolio_return_t) - λ × volatility_t - transaction_costs_t
```
- Primary: Logarithmic portfolio return
- Penalty: Volatility (λ = 0.5)
- Penalty: Transaction costs (10 bps commission + 5 bps slippage)

**Agent**:
- **Algorithm**: Proximal Policy Optimization (PPO)
- **Architecture**: Actor-Critic with dual MLP networks
  - Policy network: [128, 128] → Softmax weights
  - Value network: [128, 128] → State value estimate
- **Exploration**: Entropy regularization (coef = 0.01)

### 2. Asset Universe

| Symbol | Asset Class | Allocation Range | Expected Vol |
|--------|-------------|------------------|--------------|
| SPY | U.S. Equities | 0-40% | 15% |
| AGG | U.S. Aggregate Bonds | 0-40% | 5% |
| GLD | Gold / Commodities | 0-40% | 18% |
| VNQ | Real Estate | 0-40% | 20% |

### 3. Training Procedure

**Data Split**:
- Training: 70% (2010-2019)
- Validation: 10% (2019-2021)
- Test: 20% (2021-2024) - **strictly out-of-sample**

**Training Loop**:
1. Initialize random policy
2. Collect experience (2048 steps per update)
3. Calculate advantages using GAE (λ = 0.95)
4. Update policy via PPO (10 epochs, clip = 0.2)
5. Evaluate on validation set every 10k steps
6. Save best model by Sharpe ratio

**Hyperparameter Optimization**:
- Framework: Optuna
- Trials: 50
- Search space: Learning rate, γ, entropy coef, volatility penalty
- Metric: Validation Sharpe ratio

### 4. Benchmark Strategies

**Markowitz Mean-Variance**:
- Covariance estimation: 252-day rolling window
- Optimization: Maximize Sharpe ratio
- Rebalance: Monthly

**60/40 Portfolio**:
- 60% SPY, 40% AGG
- Rebalance: Quarterly

**Equal Weight**:
- 25% each asset
- Rebalance: Monthly

## Risk Management

### Position Constraints
- Min weight: 0% (no short selling)
- Max weight: 40% (concentration limit)
- Max leverage: 1.0 (cash-only)

### Circuit Breakers
- **15% drawdown**: Halt trading, review model
- **Action sanity check**: Override if >50% in single asset
- **Volatility filter**: Reduce exposure if VIX > 30

### Transaction Costs
- Commission: 10 bps per trade
- Slippage: 5 bps (liquid ETFs)
- Min trade size: 1% position change

## Evaluation Metrics

### Performance
- Total return
- Annualized return
- Annualized volatility
- Sharpe ratio
- Sortino ratio
- Calmar ratio (return/max drawdown)

### Risk
- Maximum drawdown
- Value at Risk (95%)
- Conditional VaR
- Drawdown duration

### Statistical Tests
- **T-test**: Significance of return difference vs benchmark (p < 0.05)
- **Diebold-Mariano**: Forecast comparison
- **Bootstrap**: Sharpe ratio confidence intervals

## Explainability & Monitoring

### XAI Techniques (Addressing "Black Box" Concern)
- **SHAP values**: Feature importance for each decision
- **LIME**: Local interpretable explanations
- **Attention weights**: Which state features drive actions

### Live Monitoring
- Daily tracking error vs 60/40
- Rolling Sharpe ratio (30-day window)
- Agent entropy (exploration vs exploitation)
- Position concentration (Herfindahl index)

## Failure Modes & Mitigations

| Risk | Severity | Mitigation |
|------|----------|-----------|
| **Overfitting** | Critical | Strict train/test split, walk-forward validation, simplified architecture |
| **Non-stationarity** | High | Regime indicators in state, periodic fine-tuning, ensemble methods |
| **Look-ahead bias** | High | Lagged features only, peer review data pipeline, strict time-series split |
| **Implementation gap** | Medium | Conservative cost assumptions, gradual capital scaling, paper trading |
| **Catastrophic actions** | Medium | Hard constraints on action space, sanity check layer, kill switches |

## Research Questions

1. **Reward Design**: Does volatility penalty improve risk-adjusted returns?
2. **Architecture**: Do deeper networks (256-128-64) outperform shallow (128-128)?
3. **Lookback**: Optimal state window (10d vs 20d vs 60d)?
4. **Rebalancing**: Daily vs weekly vs monthly execution?
5. **Stress**: Does DRL maintain advantage during 2008, 2020 crises?

## Computational Requirements

- **Training**: ~2-4 hours on CPU (500k timesteps), ~30min on GPU
- **Memory**: ~4GB RAM for experience buffer
- **Storage**: ~500MB for trained models + logs
- **Dependencies**: TensorFlow/PyTorch, Stable-Baselines3, Gymnasium

## Files

### Core Modules
- `config.yaml` - Full experimental configuration
- `data_acquisition.py` - Multi-asset data fetching (yfinance)
- `feature_engineering.py` - State representation, normalization
- `portfolio_env.py` - Custom Gym environment (PortfolioGym-v0)
- `rl_agent.py` - DRL agent (PPO, A2C, SAC wrappers)
- `benchmark.py` - Markowitz, 60/40, equal weight
- `backtester.py` - Walk-forward validation with costs
- `explainability.py` - SHAP, LIME implementations
- `main.py` - Training and evaluation pipeline

### Research Notebooks
- `01_data_exploration.ipynb` - Asset statistics, correlations, regime analysis
- `02_environment_validation.ipynb` - Gym env testing, reward design
- `03_training_analysis.ipynb` - Learning curves, hyperparameter sensitivity
- `04_performance_evaluation.ipynb` - Backtest results, XAI, statistical tests

## Usage

### Training
```bash
# Train PPO agent with default config
python main.py --mode train --config config.yaml

# Hyperparameter optimization
python main.py --mode hyperopt --n_trials 50

# Train with custom reward
python main.py --mode train --reward_type sharpe_ratio
```

### Evaluation
```bash
# Backtest on out-of-sample data
python main.py --mode backtest --model trained_models/best_model.zip

# Compare to benchmarks
python main.py --mode compare --models trained_models/*.zip

# Stress testing
python main.py --mode stress --period covid
```

### Explainability
```bash
# Generate SHAP explanations
python main.py --mode explain --method shap --n_samples 1000

# Visualize policy heatmap
python main.py --mode visualize --type policy_surface
```

## Expected Results (Based on Research)

| Metric | DRL (Target) | Markowitz | 60/40 |
|--------|--------------|-----------|-------|
| Annual Return | 12-15% | 10-12% | 8-10% |
| Volatility | 8-10% | 12-14% | 10-12% |
| Sharpe Ratio | 1.2-1.5 | 0.8-1.0 | 0.7-0.9 |
| Max Drawdown | -12% | -18% | -15% |

**Note**: Actual results will vary. This is a research prototype, not investment advice.

## References

- Research: "Smart Portfolios with Deep Reinforcement Learning" (Quant Radio)
- Algorithm: Schulman et al. (2017) - Proximal Policy Optimization
- Framework: Stable-Baselines3 (Raffin et al.)
- Benchmark: Markowitz (1952) - Portfolio Selection

## Disclaimer

This is a **research implementation** for educational purposes. Deep RL for portfolio management is an active research area with significant challenges:
- Overfitting to historical data
- Non-stationarity of financial markets
- High sensitivity to hyperparameters
- "Black box" nature of neural networks

**Do not use for live trading without extensive additional validation, risk management, and regulatory compliance.**
