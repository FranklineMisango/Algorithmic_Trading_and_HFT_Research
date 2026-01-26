# Perpetual Futures Funding Rate Arbitrage

Cross-market statistical arbitrage strategy exploiting funding rate inefficiencies in cryptocurrency perpetual futures contracts.

## Strategy Overview

**Objective**: Capture risk-adjusted returns by trading the price differential between perpetual futures and spot when it breaches theoretically derived, clamp-adjusted no-arbitrage bounds.

**Economic Rationale**: The clamping function in funding rate mechanisms creates a "dead zone" where small deviations aren't corrected, leading to wider no-arbitrage bounds than traditional models suggest. When the perp/spot ratio moves outside these bounds, cash-and-carry arbitrage becomes profitable.

## Key Features

- **Clamp-Adjusted Bounds**: Incorporates 5 bps clamping factor from Binance contract specifications
- **Dynamic No-Arbitrage Bounds**: Real-time calculation using transaction fees, interest rates, and funding intervals
- **Delta-Neutral Execution**: Long cheap asset, short expensive asset
- **Multi-Asset Support**: BTC, ETH perpetuals (both USDT-margined and inverse contracts)

## Signal Definition

**Premium/Discount Index**: `I = (PerpPrice / SpotPrice) - 1`

**Entry Signals**:
- Short Perp/Long Spot: `I > δ + (cs + cp + (rc - rf)Δt)`
- Long Perp/Short Spot: `I < -δ + (cs + cp + (rf - rc)Δt)`

**Exit Signal**: When `I` returns within bounds

**Parameters**:
- δ (Clamp): 5 bps (0.0005)
- cs, cp (Fees): 0.9 bps spot, 0 bps perp (maker)
- rf: USD risk-free rate (T-Bill)
- rc: Crypto borrowing rate (Binance Borrow)
- Δt: Funding interval (8 hours)

## Performance Targets

- **Sharpe Ratio**: > 1.5
- **Max Drawdown**: < 10%
- **Win Rate**: > 60%
- **Strategy Capacity**: Determined via slippage analysis

## Data Requirements

- Perpetual futures prices (hourly+)
- Spot prices (hourly+)
- Funding rates (8-hour frequency)
- USD risk-free rate (daily)
- Crypto borrowing rates (daily)

## Installation

```bash
cd Perp_Futures_Funding_Arbitrage
pip install -r requirements.txt
```

## Usage

### Research & Analysis
```bash
jupyter notebook notebooks/01_data_exploration.ipynb
```

### Backtesting
```bash
python main.py --config config.yaml
```

### LEAN Backtesting
```bash
lean backtest lean_algorithm.py
```

## Project Structure

```
Perp_Futures_Funding_Arbitrage/
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_bound_validation.ipynb
│   ├── 03_backtest_analysis.ipynb
│   └── 04_capacity_analysis.ipynb
├── src/
│   ├── data_acquisition.py
│   ├── signal_generator.py
│   ├── backtester.py
│   └── risk_manager.py
├── config.yaml
├── main.py
├── lean_algorithm.py
├── requirements.txt
└── README.md
```

## Risk Management

- **Stop-Loss**: Exit if I moves 10 bps beyond entry bound
- **Position Sizing**: Max 2% annualized volatility per position
- **Liquidity Monitoring**: Pause if spread > 2x normal or volume < $5M
- **Stress Tests**: Funding shock, liquidity crisis, rate spike scenarios

## References

- "Arbitrage in Perpetual Crypto Contracts" - Quant Radio
- Binance Perpetual Futures Contract Specifications

## Disclaimer

Educational and research purposes only. Not financial advice. Trading involves substantial risk of loss.
