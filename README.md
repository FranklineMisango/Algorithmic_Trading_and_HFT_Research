# Algorithmic Trading and High-Frequency Trading Research

A comprehensive research repository implementing quantitative trading strategies, market making models, derivatives pricing, and machine learning-based prediction systems. This repository contains production-grade implementations of academic research papers and original trading strategies across multiple asset classes.

## Table of Contents

- [Overview](#overview)
- [Technology Stack](#technology-stack)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Project Divisions](#project-divisions)
  - [Algorithmic Trading Strategies](#algorithmic-trading-strategies)
  - [Market Making Models](#market-making-models)
  - [Derivatives Pricing](#derivatives-pricing)
  - [Data Infrastructure](#data-infrastructure)
  - [Research Notebooks](#research-notebooks)
- [Build Instructions](#build-instructions)
- [Development Tools](#development-tools)
- [Contributing](#contributing)
- [License](#license)

## Overview

This repository serves as a comprehensive collection of quantitative finance research, spanning:

- **Statistical arbitrage** using machine learning
- **Momentum strategies** enhanced with deep learning
- **Volatility-based predictive trading** using Granger causality
- **Market making** with optimal control theory (HJB, Avellaneda-Stoikov)
- **Structured derivatives pricing** and client pitch simulation
- **Options arbitrage** exploiting put-call parity
- **Data pipelines** for multiple exchanges and data vendors

All implementations include backtesting frameworks, performance analytics, and visualization tools.

## Technology Stack

### Core Languages
- **Python 3.8+** - Primary language for all implementations
- **LaTeX** - Research paper documentation

### Machine Learning & Deep Learning
- **PyTorch 2.0+** - Deep learning models (LSTM, CNN, FFNN)
- **TensorFlow 2.x** - Alternative ML framework
- **scikit-learn** - Classical ML algorithms
- **XGBoost, LightGBM, CatBoost** - Gradient boosting frameworks

### Quantitative Libraries
- **NumPy** - Numerical computing
- **pandas** - Data manipulation and time series
- **SciPy** - Scientific computing and optimization
- **statsmodels** - Statistical modeling and econometrics
- **cvxpy** - Convex optimization

### Financial Data APIs
- **yfinance** - Yahoo Finance data
- **alpaca-py** - Alpaca Markets API (US equities)
- **polygon-api-client** - Polygon.io market data
- **ccxt** - Cryptocurrency exchange integration
- **python-binance** - Binance API
- **pandas_datareader** - Multiple data sources

### Backtesting & Analysis
- **Backtrader** - Event-driven backtesting
- **vectorbt** - Vectorized backtesting
- **pyfolio-reloaded** - Performance analysis
- **empyrical** - Financial metrics
- **LEAN/QuantConnect** - Institutional-grade backtesting

### Visualization
- **matplotlib** - Static plotting
- **seaborn** - Statistical visualization
- **plotly** - Interactive charts
- **Dash** - Web-based dashboards

### Development Tools
- **Jupyter Lab/Notebook** - Interactive development
- **pytest** - Unit testing
- **loguru** - Logging
- **python-dotenv** - Environment management
- **numba** - JIT compilation for performance

### Market Microstructure
- **websocket-client, websockets** - Real-time data streaming
- **ta-lib** - Technical analysis indicators
- **ARCH** - Volatility modeling

## Project Structure

```
Algorithmic_Trading_and_HFT_Research/
├── Algorithmic_Strategies/          # Trading strategy implementations
│   ├── Volts_Volatility_Strategy/   # Volatility-based Granger causality
│   ├── Statistical_Arbitrage_ML/    # ML-based market-neutral arbitrage
│   ├── Futures_Prediction_ML/       # Order book ML prediction
│   ├── Put_Futures_Spread_Arb/      # Options arbitrage
│   └── Leveraged_Index_Funds/       # Index fund strategies
├── Deep_Learning_Momentum/          # Deep learning momentum ranking
├── Market_Making/                   # Market making models
│   ├── Avellaneda-Stoikov/          # AS optimal market making
│   ├── Grossman-Miller-Model/       # Grossman-Miller framework
│   ├── HJB_DP_MM_Optimisation/      # HJB equation solutions
│   ├── MM_With_Info_Disadvantage/   # Adverse selection models
│   └── Sentiment_HJB/               # Sentiment-aware market making
├── Derivatives_Pricing_Research/    # Structured products
│   ├── structured_products/         # Product library
│   └── notebooks/                   # Pricing notebooks
├── data_pipeline/                   # Multi-source data pipeline
├── Extending_Classical_Strategies/  # Benjamin Graham, etc.
├── HFT_Binance_data_fetcher/       # Real-time crypto data
├── Hidden_Orders/                   # Hidden order simulations
├── Sharpe_Research/                 # Sharpe ratio optimizations
├── Pairs_Trading/                   # Pairs trading research
├── Base_Latex_Template/            # Research paper template
├── data/                           # Data storage
├── requirements.txt                # Global dependencies
└── README.md                       # This file
```

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Virtual environment (recommended)
- TA-Lib (system-level installation required)

### System Dependencies

#### Ubuntu/Debian
```bash
sudo apt-get update
sudo apt-get install python3-dev python3-pip
sudo apt-get install build-essential
sudo apt-get install ta-lib
```

#### macOS
```bash
brew install python3
brew install ta-lib
```

#### Windows
Download TA-Lib from [unofficial binaries](https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib)

### Python Environment Setup

```bash
# Clone repository
git clone https://github.com/yourusername/Algorithmic_Trading_and_HFT_Research.git
cd Algorithmic_Trading_and_HFT_Research

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install global dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### Project-Specific Setup

Each strategy has its own dependencies. Navigate to the project directory and install:

```bash
# Example: Deep Learning Momentum
cd Deep_Learning_Momentum
pip install -r requirements.txt

# Example: VOLTS Strategy
cd Algorithmic_Strategies/Volts_Volatility_Strategy
pip install -r requirements.txt
```

## Project Divisions

### Algorithmic Trading Strategies

#### 1. VOLTS Volatility Strategy
**Location**: `Algorithmic_Strategies/Volts_Volatility_Strategy/`

Implements volatility-based predictive trading using Granger causality tests to identify stocks where one's volatility predicts another's price movements.

**Key Features**:
- 4 robust volatility estimators (Parkinson, Garman-Klass, Rogers-Satchell, Yang-Zhang)
- K-means++ clustering for volatility regimes
- Granger causality network analysis
- Trend-following signal generation

**Tech Stack**: pandas, scipy, statsmodels, scikit-learn, yfinance, alpaca-py

**Usage**:
```bash
cd Algorithmic_Strategies/Volts_Volatility_Strategy
python main.py --config config.yaml
```

#### 2. Statistical Arbitrage with Machine Learning
**Location**: `Algorithmic_Strategies/Statistical_Arbitrage_ML/`

Market-neutral strategy exploiting mean reversion in Russell 3000 stocks using ML predictions of 2-3 day returns.

**Key Features**:
- Rolling window model retraining (10-year windows)
- Momentum + mean reversion features
- Long/short portfolio construction
- Integration with QuantConnect LEAN

**Performance Targets**:
- Annual Return: 20-28%
- Sharpe Ratio: >1.4
- Max Drawdown: <20%

**Tech Stack**: scikit-learn, XGBoost, LEAN, pandas

**Usage**:
```bash
cd Algorithmic_Strategies/Statistical_Arbitrage_ML
python main.py
```

#### 3. Deep Learning Momentum Strategy
**Location**: `Deep_Learning_Momentum/`

Implements Taki & Lee (2013) neural network approach to rank stocks by predicted outperformance probability.

**Key Features**:
- 33 momentum features (12 long-term + 20 short-term + 1 anomaly)
- FFNN with bottleneck layer (64→4→32→16→1)
- Cross-sectional z-score standardization
- Quantile-based long-short strategy

**Expected Performance**:
- Annual Return: 12.8%
- Sharpe Ratio: 1.03
- Max Drawdown: 24%

**Tech Stack**: PyTorch, scikit-learn, yfinance, alpaca-py

**Usage**:
```bash
cd Deep_Learning_Momentum
jupyter notebook notebooks/4_end_to_end_pipeline.ipynb
```

#### 4. Futures Prediction Using Order Book ML
**Location**: `Algorithmic_Strategies/Futures_Prediction_Arbitrage_ML/`

Predicts short-term futures price movements using real-time order book imbalance and microstructure features.

**Key Features**:
- WebSocket streaming from Binance
- Order book imbalance features
- XGBoost, LightGBM, LSTM models
- Real-time prediction API

**Tech Stack**: websockets, ccxt, XGBoost, LSTM, Flask

**Usage**:
```bash
cd Algorithmic_Strategies/Futures_Prediction_Arbitrage_ML
python data_fetcher.py  # Start background data collection
jupyter notebook ml_orderbook_prediction.ipynb
```

#### 5. Put-Futures Spread Arbitrage
**Location**: `Algorithmic_Strategies/Put_Futures_Spread_Arbitrage/`

Exploits put-call parity violations between options and futures contracts.

**Key Features**:
- Real-time options data from yfinance
- Put-call parity arbitrage detection
- Synthetic futures construction

**Tech Stack**: yfinance, numpy, pandas

### Market Making Models

#### 1. Avellaneda-Stoikov Model
**Location**: `Market_Making/Avellaneda-Stoikov/`

Optimal market making using stochastic control theory. Solves for bid-ask spreads that maximize utility while managing inventory risk.

**Key Features**:
- Closed-form solutions for optimal quotes
- Inventory risk management
- Real-time Binance implementation

**Tech Stack**: numpy, scipy, binance API

#### 2. HJB Dynamic Programming
**Location**: `Market_Making/HJB_DP_MM_Optimisation/`

Solves Hamilton-Jacobi-Bellman equation for optimal market making policies.

**Key Features**:
- Numerical PDE solutions
- Dynamic programming optimization
- Value function iteration

**Tech Stack**: numpy, scipy, matplotlib

#### 3. Market Making with Informational Disadvantage
**Location**: `Market_Making/MM_With_Informational_Disadvantage/`

Models adverse selection costs when market makers face informed traders.

**Key Features**:
- Glosten-Milgrom framework
- Adverse selection modeling
- Optimal quote adjustment

#### 4. Sentiment-Aware Market Making
**Location**: `Market_Making/Sentiment_HJB/`

Incorporates market sentiment into HJB framework for dynamic quote adjustment.

### Derivatives Pricing

#### Structured Products Library
**Location**: `Derivatives_Pricing_Research/`

Comprehensive library for pricing structured equity products with Monte Carlo and analytical methods.

**Products Supported**:
- Autocallables (early redemption notes)
- Reverse Convertibles (capital protection)
- Barrier Options (knock-in/knock-out)
- Range Accruals (coupon-based notes)

**Key Features**:
- Monte Carlo simulation engine
- Greeks calculation (Delta, Gamma, Vega, Theta, Rho)
- Client pitch simulator
- Interactive payoff diagrams

**Tech Stack**: numpy, scipy, matplotlib, plotly

**Usage**:
```bash
cd Derivatives_Pricing_Research
jupyter notebook notebooks/payoff_visualizations.ipynb
```

### Data Infrastructure

#### Multi-Source Data Pipeline
**Location**: `data_pipeline/`

Unified data pipeline supporting multiple exchanges and data vendors with automatic format conversion for LEAN backtesting.

**Data Sources**:
- Alpaca Markets (US equities)
- Binance (crypto)
- Polygon.io (options, futures)
- Databento (professional market data)
- yfinance (equities, options)

**Features**:
- Automatic LEAN format conversion
- Rate limiting and retry logic
- Data validation and quality checks
- Multi-resolution support (minute, hour, daily)

**Tech Stack**: alpaca-py, ccxt, polygon-api-client, databento, yfinance

**Usage**:
```bash
cd data_pipeline
# Set up environment variables
cp .env.example .env  # Edit with your API keys
./setup.sh
source data_pipeline_env/bin/activate
python main.py --source alpaca --resolution daily
```

### Research Notebooks

#### Classical Strategy Extensions
**Location**: `Extending_Classical_Strategies/`

Implementations of classic value investing strategies:
- Benjamin Graham defensive investor strategy
- P/E ratio-based stock selection

#### Sharpe Research
**Location**: `Sharpe_Research/`

Cointegration-based portfolio optimization and Sharpe ratio calculations.

#### Pairs Trading
**Location**: `Pairs_Trading/`

Statistical pairs trading research and implementations.

#### Hidden Orders
**Location**: `Hidden_Orders/`

Simulations of hidden order execution and market impact.

## Build Instructions

### Running Individual Strategies

Most strategies follow this pattern:

```bash
# 1. Navigate to project directory
cd <project_directory>

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure (edit config.yaml or .env)
nano config.yaml

# 4. Run
python main.py
# OR
jupyter notebook
```

### Running Backtests

#### LEAN/QuantConnect Backtests
```bash
cd Algorithmic_Strategies/Statistical_Arbitrage_ML/lean
lean backtest main.py
```

#### Backtrader Backtests
```python
# In Python script or notebook
from backtester import Backtester

backtester = Backtester(config)
results = backtester.run()
backtester.plot()
```

### Data Pipeline Setup

```bash
cd data_pipeline
chmod +x setup.sh
./setup.sh

# Configure API keys
export ALPACA_API_KEY="your_key"
export ALPACA_SECRET_KEY="your_secret"

# Run pipeline
python main.py --source alpaca --resolution daily --test
```

### Docker Deployment (Futures Prediction)

```bash
cd Algorithmic_Strategies/Futures_Prediction_Arbitrage_ML
docker-compose up -d
```

## Development Tools

### Testing
```bash
# Run all tests
pytest

# Run specific project tests
cd Deep_Learning_Momentum
pytest tests/
```

### Code Quality
```bash
# Format code
black .

# Lint
flake8 .

# Type checking
mypy .
```

### Jupyter Notebooks
```bash
# Start Jupyter Lab
jupyter lab

# Convert notebook to script
jupyter nbconvert --to script notebook.ipynb
```

### Performance Profiling
```bash
# Profile Python script
python -m cProfile -o output.prof main.py

# Analyze profile
snakeviz output.prof
```

## Environment Variables

Create a `.env` file in project directories:

```bash
# Alpaca (US Equities)
ALPACA_API_KEY=your_alpaca_key
ALPACA_SECRET_KEY=your_alpaca_secret

# Binance (Crypto)
BINANCE_API_KEY=your_binance_key
BINANCE_SECRET_KEY=your_binance_secret

# Polygon.io
POLYGON_API_KEY=your_polygon_key

# Databento
DATABENTO_API_KEY=your_databento_key

# QuantConnect
QC_USER_ID=your_qc_user
QC_API_TOKEN=your_qc_token
```

## Performance Metrics

All strategies track standard quantitative metrics:

- **Returns**: Annualized, cumulative, monthly
- **Risk**: Sharpe ratio, Sortino ratio, max drawdown, volatility
- **Trading**: Win rate, profit factor, average trade duration
- **Market Exposure**: Beta, correlation, factor loadings

## Research Papers Implemented

1. Taki & Lee (2013) - "Applying Deep Learning to Enhance Momentum Trading"
2. Avellaneda & Stoikov (2008) - "High-frequency trading in a limit order book"
3. Grossman & Miller (1988) - "Liquidity and Market Structure"
4. Jegadeesh & Titman (1993) - "Returns to Buying Winners and Selling Losers"
5. Graham (1949) - "The Intelligent Investor"

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Disclaimer

This repository is for educational and research purposes only. None of the strategies, models, or code should be interpreted as financial advice. Past performance does not guarantee future results. Trading involves substantial risk of loss.

## Contact

For questions, suggestions, or collaboration opportunities, please open an issue on GitHub.

---

**Last Updated**: January 2026