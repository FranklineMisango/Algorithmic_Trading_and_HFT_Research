# Deep Learning Options Trading - Notebooks

Interactive Jupyter notebooks for exploring, training, and backtesting the LSTM options trading strategy.

##  Notebooks Overview

###  [00_quick_start.ipynb](00_quick_start.ipynb) **← START HERE**
Get up and running in 5 minutes:
- Configure your data parameters (tickers, dates)
- Fetch options + underlying data from Databento
- Quick validation and visualization
- Next steps guidance

###  [01_data_exploration.ipynb](01_data_exploration.ipynb) (NEW VERSION)
Comprehensive data analysis:
- **UPDATED:** Now with customizable data fetching
- **NEW:** Interactive ticker and date selection
- Price and options liquidity analysis
- Feature distributions
- Time series patterns
- Data quality reports

**Old version** backed up as `01_data_exploration_OLD.ipynb`

###  [02_feature_analysis.ipynb](02_feature_analysis.ipynb)
Feature engineering for LSTM:
- Moneyness calculation
- Time to expiration features
- Implied volatility
- Rolling statistics
- Sequential data creation (30-day windows)
- Feature importance analysis

###  [03_model_training.ipynb](03_model_training.ipynb)
Train the LSTM model:
- Model architecture configuration
- Sharpe ratio optimization
- Turnover regularization
- Walk-forward validation
- Hyperparameter tuning
- Training visualization

###  [04_backtest_evaluation.ipynb](04_backtest_evaluation.ipynb)
Strategy backtesting:
- Performance metrics (Sharpe, returns, drawdown)
- Transaction cost modeling
- Risk management simulation
- Benchmark comparisons
- Position analysis

##  Quick Start Guide

### First Time Setup

1. **Open Quick Start notebook:**
   ```bash
   jupyter notebook 00_quick_start.ipynb
   ```

2. **Configure your analysis:**
   - Edit the `TICKERS` list (e.g., `['AAPL', 'MSFT', 'AMZN']`)
   - Set date range (`START_DATE`, `END_DATE`)
   - Run the data fetcher

3. **Explore and analyze:**
   - Open `01_data_exploration.ipynb` for deep dive
   - Progress through notebooks sequentially

### Customizing Data

In either `00_quick_start.ipynb` or `01_data_exploration.ipynb`:

```python
# Example: Fetch more tickers and longer history
TICKERS = [
    'AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META',  # Tech
    'JPM', 'BAC', 'GS', 'C',                   # Finance  
    'JNJ', 'PFE', 'UNH', 'ABBV'                # Healthcare
]

START_DATE = '2024-01-01'  # 1+ year history
END_DATE = '2026-02-23'

FETCH_NEW_DATA = True  # Download fresh data
```

Then run the data fetcher cell.

##  Data Files

After running the data fetcher, you'll have:

```
data/
 underlying_prices/
    underlying_prices.csv    # Stock OHLCV + returns + volatility
 options_data/
    options_data.csv         # Options OHLCV + volume from Databento
 data_summary.csv            # Quick reference stats
```

##  Workflow

### Typical Analysis Flow:

1. **Quick Start** → Get data loaded
2. **Data Exploration** → Understand your dataset
3. **Feature Analysis** → Engineer ML features
4. **Model Training** → Train LSTM
5. **Backtest** → Evaluate strategy

### Want Different Data?

Just re-run the configuration + fetch cells in any notebook!

##  Pro Tips

### For Quick Testing (3-5 minutes)
```python
TICKERS = ['AAPL', 'MSFT', 'AMZN']  # 3 stocks
START_DATE = '2025-10-01'            # ~4 months
END_DATE = '2026-02-23'
```

### For Serious Backtesting (20-30 minutes)
```python
TICKERS = [top 20 S&P 100 stocks]    # 20+ stocks
START_DATE = '2023-01-01'            # 2+ years
END_DATE = '2026-02-23'
```

### For Production Research (1-2 hours)
```python
TICKERS = [all S&P 100]              # ~100 stocks
START_DATE = '2020-01-01'            # 5+ years
END_DATE = '2026-02-23'
```

##  Common Issues

### "No data found"
→ Run the data fetcher cell in notebook with `FETCH_NEW_DATA = True`

### "Databento API error"
→ Check your API key in `../config.yaml`

### "Empty data file"
→ Date range might be too recent (historical data has ~2 day lag)

##  Data Requirements

- **Databento API key:** Free tier works great for testing
- **Date range:** End date must be ≥2 days ago (historical data delay)
- **Tickers:** S&P 100 stocks recommended (best options liquidity)
- **Min history:** 60 days for testing, 1+ years for production

##  Need Help?

1. Start with `00_quick_start.ipynb` - it's the most beginner-friendly
2. Check output messages - they guide you on what to do next
3. Run cells sequentially (don't skip steps)
4. Restart kernel if you get import errors

##  Learning Path

**Beginner:** Focus on notebooks 00 and 01
**Intermediate:** All notebooks in sequence  
**Advanced:** Customize feature engineering (02) and model architecture (03)

---

**Happy Trading! ** Remember: Past performance doesn't guarantee future results. This is for research and education only.
