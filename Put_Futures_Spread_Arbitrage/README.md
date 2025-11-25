# Put Futures Spread Arbitrage Research

This research explores put futures spread arbitrage strategies, focusing on exploiting price discrepancies between put options, call options, and futures contracts through put-call parity.

## Overview

Put-call parity states that for European options:  
\[ C + PV(K) = P + F \]  
where:  
- \( C \): Call option price  
- \( P \): Put option price  
- \( PV(K) \): Present value of strike price \( K \)  
- \( F \): Futures price  

Arbitrage opportunities arise when this relationship breaks down.

## Strategy

1. **Fetch Real Data**: Use yfinance to get current SPY options and ES futures prices.
2. **Identify Mispricing**: Compare actual futures price to theoretical price derived from options.  
3. **Create Synthetic Futures**: Use options to replicate futures positions.  
4. **Execute Arbitrage**: Trade actual futures against synthetic positions.  
5. **Risk Management**: Monitor and close positions when mispricing corrects.

## Files

- `put_futures_spread_arbitrage.ipynb`: Jupyter notebook with implementation, real data fetching, and simulations.

## Requirements

- Python 3.x  
- NumPy, Pandas, Matplotlib, yfinance  

Install dependencies:  
```bash  
pip install numpy pandas matplotlib yfinance  
```

## Notes

- Real data is fetched for SPY (ETF) options and ES futures, but they are not identical underlyings, leading to apparent mispricings.
- For true arbitrage, use instruments on the same underlying.