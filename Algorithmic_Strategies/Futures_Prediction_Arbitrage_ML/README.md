# ML-Based Futures Price Prediction Using Order Book Data

## Project Overview

This project implements a machine learning pipeline for predicting short-term price movements in cryptocurrency futures (e.g., BTC/USDT) using real-time order book data from Binance. The goal is to develop a predictive model that can inform high-frequency trading (HFT) strategies by forecasting whether the mid-price will increase or decrease in the next time step.

The notebook (`ml_orderbook_prediction.ipynb`) demonstrates the end-to-end process: data loading from background fetcher, feature engineering, model training with multiple algorithms, and backtesting a simple trading strategy.

## Setup and Usage

### Background Data Fetcher
To collect real-time order book data continuously:

1. Install dependencies: `pip install websocket-client pandas numpy`
2. Run the background fetcher: `python data_fetcher.py`
   - This starts a WebSocket connection and saves data to `live_order_book_data.csv` every minute.
   - Press Ctrl+C to stop.
3. The fetcher runs in the background, allowing the notebook to load fresh data without blocking.

### Running the Notebook
- **Option 1: Manual Start**: Run `python data_fetcher.py &` in terminal to start background data collection
- **Option 2: From Notebook**: Use cell 2.5 in the notebook to start the fetcher programmatically
- The notebook loads data from `live_order_book_data.csv` created by the background fetcher
- If no live data, it falls back to historical fetches

## Methodology

### Data Collection
- **Real-Time Data**: Background script (`data_fetcher.py`) uses WebSocket to stream order book updates (20 depth levels, 50ms intervals) and saves to CSV.
- **Historical Data**: CCXT library fetches additional snapshots for backtesting (used as fallback).
- **Data Processing**: Parses bids/asks into structured DataFrames with timestamps.

### Feature Engineering
- **Bid-Ask Imbalance**: Ratio of bid vs. ask quantities in top k levels.
- **Mid-Price**: Average of best bid and ask prices.
- **Spread**: Difference between best ask and bid.
- **Volatility**: Rolling standard deviation of mid-price.
- **Target**: Binary classification (1 if mid-price increases next period, 0 otherwise).

### Models Trained
- **Tree-Based**: XGBoost, LightGBM, CatBoost, Random Forest (with hyperparameter tuning and class weighting for imbalance).
- **Ensemble**: Voting classifier combining multiple models.
- **Deep Learning**: LSTM for temporal sequences, CNN treating order book as 2D image.

### Evaluation
- Accuracy, precision, recall, F1-score.
- Trading simulation: Simple long/short strategy based on predictions, compared to buy-and-hold.

## Results

### Model Performance (Example from Notebook)
- XGBoost (tuned): ~55-60% accuracy (varies by run).
- Ensemble methods show marginal improvements.
- Deep learning models (LSTM/CNN) perform similarly but require more data.

### Trading Simulation
- Simulated profits: Variable, often outperforming buy-and-hold in short periods due to overfitting.
- Number of trades: Depends on prediction frequency.
- Caveat: Unrealistic assumptions (no fees, slippage).

## Critique

### Strengths
- **Comprehensive Approach**: Tests multiple algorithms, including novel CNN application to order books.
- **Real-Time Integration**: WebSocket for live data is suitable for HFT.
- **Imbalance Handling**: Uses class weights effectively.
- **Exploratory Analysis**: EDA plots provide insights into data distributions.
- **Trading Backtest**: Includes economic evaluation beyond pure prediction.

### Weaknesses
- **Data Limitations**: Short collection periods (~20-30 minutes) yield insufficient samples for robust ML. Historical data is sparse.
- **Feature Engineering**: Basic features miss depth (e.g., order book slope, cumulative volumes). CNN ignores price information.
- **Evaluation Metrics**: Accuracy is misleading for imbalanced data; better to use F1/AUC. No time-series validation (random split ignores chronology).
- **Overfitting Risk**: Small dataset + complex models likely overfit. Deep learning trained for 100 epochs without regularization.
- **Trading Simulation**: Unrealistic—no transaction costs, slippage, market impact. Fixed position sizing. Simulation on test set isn't walk-forward.
- **Practicality**: Real-time prediction may be slow for HFT. No model interpretability or production monitoring.

### Recommendations
- **Expand Data**: Collect months of historical data using Binance API dumps or multiple assets.
- **Enhance Features**: Add LOBSTER-style metrics (e.g., order book imbalance at different depths).
- **Improve Validation**: Use time-series splits (e.g., `TimeSeriesSplit`). Switch to regression for price change magnitude.
- **Realistic Backtesting**: Integrate libraries like `backtrader` with fees/slippage. Use out-of-sample data.
- **Production Readiness**: Add retraining pipelines, risk controls, and latency profiling.
- **Further Research**: Test on other pairs (e.g., ETH/USDT). Explore reinforcement learning for dynamic strategies.

## Requirements

- Python 3.8+
- Libraries: websocket-client, pandas, numpy, scikit-learn, xgboost, lightgbm, catboost, tensorflow, ccxt, matplotlib
- Install via `pip install -r ../requirements.txt`

## Usage

1. Run the notebook cells sequentially.
2. Adjust `SYMBOL`, `DEPTH_LEVELS`, etc., for different assets.
3. For live data, ensure stable internet for WebSocket.

## Files

- `ml_orderbook_prediction.ipynb`: Main notebook.
- `order_book_data.csv`: Sample collected data.
- `catboost_info/`: Model training artifacts.

## License

See root LICENSE file.

## Contact

For questions, refer to the notebook or repository issues.