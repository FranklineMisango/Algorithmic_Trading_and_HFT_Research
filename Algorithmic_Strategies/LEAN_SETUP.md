# QuantConnect Lean Configuration for Carry Strategies

## Installation

```bash
# Install Lean CLI
pip install lean

# Initialize Lean project
cd Algorithmic_Strategies
lean init

# Link strategy files
ln -s FX_Carry/lean_algorithm.py lean/Algorithm.Python/FXCarryStrategy/main.py
ln -s Rates_Carry/lean_algorithm.py lean/Algorithm.Python/RatesCarryStrategy/main.py
ln -s Credit_Carry/lean_algorithm.py lean/Algorithm.Python/CreditCarryStrategy/main.py
ln -s Commodity_Carry/lean_algorithm.py lean/Algorithm.Python/CommodityCarryStrategy/main.py
```

## Running Backtests

All strategies now use Lean event-driven backtesting. Run any strategy:

```bash
# Carry Strategies
lean backtest --algorithm-location FX_Carry/lean_algorithm.py
lean backtest --algorithm-location Rates_Carry/lean_algorithm.py
lean backtest --algorithm-location Credit_Carry/lean_algorithm.py
lean backtest --algorithm-location Commodity_Carry/lean_algorithm.py

# Equity Strategies
lean backtest --algorithm-location Copy_Congress_Strategy/lean_algorithm.py
lean backtest --algorithm-location AI_Enhanced_6040_Portfolio/lean_algorithm.py
lean backtest --algorithm-location Pairs_Trading/lean_algorithm.py
lean backtest --algorithm-location Statistical_Arbitrage_ML/lean_algorithm.py

# Reinforcement Learning
lean backtest --algorithm-location Statistical_Arbitrage_RL/lean_algorithm.py

# Intraday/HFT
lean backtest --algorithm-location Intraday_Momentum_Breakout/lean_algorithm.py

# Derivatives/Volatility
lean backtest --algorithm-location Volts_Volatility_Strategy/lean_algorithm.py
lean backtest --algorithm-location Futures_Prediction_Arbitrage_ML/lean_algorithm.py
lean backtest --algorithm-location Put_Futures_Spread_Arbitrage/lean_algorithm.py
lean backtest --algorithm-location Leveraged_Index_Funds/lean_algorithm.py

# Alternative Assets
lean backtest --algorithm-location Music_Royalties_Strategy/lean_algorithm.py
lean backtest --algorithm-location DP_Ratio_Market_Timing/lean_algorithm.py
```

## Live Trading

```bash
# Deploy to QuantConnect cloud
lean cloud push --project FXCarryStrategy

# Or run locally with live data
lean live FXCarryStrategy
```

## Key Advantages of Lean

1. **Event-Driven Architecture**: Handles data chronologically, avoiding look-ahead bias
2. **Realistic Execution**: Models slippage, market impact, and realistic fills
3. **Multiple Asset Classes**: Supports FX, equities, futures, options, crypto
4. **Data Quality**: Professional-grade tick data from multiple providers
5. **Production Ready**: Same engine for backtesting and live trading
6. **Risk Management**: Built-in margin, liquidation, and portfolio tracking

## Configuration Files

Each strategy needs a `config.json`:

```json
{
    "algorithm-type-name": "FXCarryAlgorithm",
    "algorithm-language": "Python",
    "algorithm-location": "lean_algorithm.py",
    "data-folder": "/Data",
    "debugging": false,
    "debugging-method": "LocalCmdline",
    "log-handler": "ConsoleLogHandler",
    "messaging-handler": "StreamingMessageHandler",
    "job-queue-handler": "JobQueue",
    "api-handler": "Api",
    "map-file-provider": "LocalDiskMapFileProvider",
    "factor-file-provider": "LocalDiskFactorFileProvider",
    "data-provider": "DefaultDataProvider",
    "alpha-handler": "DefaultAlphaHandler",
    "data-channel-provider": "DataChannelProvider",
    "object-store": "LocalObjectStore",
    "data-aggregator": "QuantConnect.Lean.Engine.DataFeeds.AggregationManager"
}
```

## Strategy Overview

### Carry Strategies (4)
1. **FX_Carry**: Currency interest rate differentials
2. **Rates_Carry**: Government bond roll-down yield
3. **Credit_Carry**: CDS spread carry
4. **Commodity_Carry**: Futures convenience yield

### Equity Strategies (4)
5. **Copy_Congress_Strategy**: Congressional trade replication
6. **AI_Enhanced_6040_Portfolio**: ML-driven asset allocation
7. **Pairs_Trading**: Statistical arbitrage pairs
8. **Statistical_Arbitrage_ML**: ML mean reversion

### Reinforcement Learning (1)
9. **Statistical_Arbitrage_RL**: DQN-based pairs trading with EMRT selection

### Intraday/HFT (1)
10. **Intraday_Momentum_Breakout**: Opening range breakouts

### Derivatives/Volatility (4)
11. **Volts_Volatility_Strategy**: VIX regime trading
12. **Futures_Prediction_Arbitrage_ML**: Futures basis ML
13. **Put_Futures_Spread_Arbitrage**: Put-call parity arbitrage
14. **Leveraged_Index_Funds**: Leveraged ETF decay

### Alternative Assets (2)
15. **Music_Royalties_Strategy**: Music streaming royalties
16. **DP_Ratio_Market_Timing**: Dividend/price valuation timing

## Data Requirements

All strategies use QuantConnect data feeds:
- **Equities**: US stocks, ETFs (1-minute to daily)
- **Forex**: OANDA spot rates with bid/ask
- **Futures**: CME, ICE, NYMEX contracts
- **Options**: US equity options chains
- **Crypto**: Major cryptocurrencies
- **Custom Data**: CBOE VIX, FRED economic data

## Performance Monitoring

Lean provides real-time metrics:
- Sharpe Ratio
- Maximum Drawdown
- Win Rate
- Turnover
- Transaction Costs
- Capacity Estimates

## References
- [QuantConnect Documentation](https://www.quantconnect.com/docs)
- [Lean Engine GitHub](https://github.com/QuantConnect/Lean)
- [Algorithm Framework](https://www.quantconnect.com/docs/v2/writing-algorithms/algorithm-framework)
