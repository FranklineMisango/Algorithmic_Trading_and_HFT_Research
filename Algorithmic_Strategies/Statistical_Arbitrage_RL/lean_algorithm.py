"""
QuantConnect Lean Algorithm: Statistical Arbitrage with Reinforcement Learning

Event-driven implementation of pairs trading strategy using RL-trained policy.
"""

from AlgorithmImports import *
import numpy as np
from collections import deque


class StatisticalArbitrageRLAlgorithm(QCAlgorithm):
    """
    Statistical Arbitrage using Reinforcement Learning for pair selection and trading.
    
    Strategy:
    1. Select pairs with minimum Empirical Mean Reversion Time (EMRT)
    2. Calculate real-time spread features (z-score, momentum, volatility)
    3. Use trained RL policy to decide: Buy, Sell, or Hold
    4. Execute dollar-neutral positions with risk management
    """
    
    def Initialize(self):
        """Initialize algorithm parameters and data."""
        # Backtest period
        self.SetStartDate(2023, 1, 1)
        self.SetEndDate(2023, 12, 31)
        self.SetCash(100000)
        
        # Selected pairs (from grid search in research)
        self.pairs = [
            ("MSFT", "GOOGL"),  # Technology
            ("CVS", "JNJ"),     # Healthcare
            ("PG", "KO"),       # Consumer Goods
            ("JPM", "BAC")      # Financials
        ]
        
        # Add securities
        self.symbols = {}
        for ticker1, ticker2 in self.pairs:
            sym1 = self.AddEquity(ticker1, Resolution.Daily).Symbol
            sym2 = self.AddEquity(ticker2, Resolution.Daily).Symbol
            
            pair_id = f"{ticker1}_{ticker2}"
            self.symbols[pair_id] = {
                'sym1': sym1,
                'sym2': sym2,
                'ticker1': ticker1,
                'ticker2': ticker2
            }
        
        # Strategy parameters
        self.lookback_window = 20  # Days for rolling statistics
        self.zscore_entry = 2.0    # Entry threshold
        self.zscore_exit = 0.5     # Exit threshold
        self.position_size = 0.1   # 10% per leg
        
        # State tracking
        self.spread_history = {pair: deque(maxlen=126) for pair in self.symbols.keys()}
        self.positions = {pair: 0 for pair in self.symbols.keys()}  # 0=none, 1=long, -1=short
        self.entry_prices = {pair: {'price1': 0, 'price2': 0} for pair in self.symbols.keys()}
        
        # Performance tracking
        self.trade_count = 0
        self.winning_trades = 0
        
        # Schedule rebalancing
        self.Schedule.On(
            self.DateRules.EveryDay(),
            self.TimeRules.AfterMarketOpen("MSFT", 30),
            self.Rebalance
        )
        
        self.Debug(f"Initialized {len(self.pairs)} pairs for trading")
    
    def Rebalance(self):
        """Daily rebalancing based on RL policy (simplified rule-based proxy)."""
        
        for pair_id, pair_info in self.symbols.items():
            sym1 = pair_info['sym1']
            sym2 = pair_info['sym2']
            
            # Check if we have price data
            if not self.Securities[sym1].HasData or not self.Securities[sym2].HasData:
                continue
            
            # Get current prices
            price1 = self.Securities[sym1].Price
            price2 = self.Securities[sym2].Price
            
            if price1 == 0 or price2 == 0:
                continue
            
            # Calculate spread (log price ratio)
            spread = np.log(price1 / price2)
            self.spread_history[pair_id].append(spread)
            
            # Need minimum history
            if len(self.spread_history[pair_id]) < self.lookback_window:
                continue
            
            # Calculate features
            spread_array = np.array(self.spread_history[pair_id])
            mean = spread_array[-self.lookback_window:].mean()
            std = spread_array[-self.lookback_window:].std()
            
            if std == 0:
                continue
            
            zscore = (spread - mean) / std
            
            # Current position
            current_position = self.positions[pair_id]
            
            # RL-inspired decision logic (simplified from trained policy)
            action = self._get_action(zscore, current_position, spread_array)
            
            # Execute action
            if action == 'buy' and current_position == 0:
                # Long spread: Buy stock1, Sell stock2
                self._open_position(pair_id, sym1, sym2, price1, price2, direction=1)
                
            elif action == 'sell' and current_position == 0:
                # Short spread: Sell stock1, Buy stock2
                self._open_position(pair_id, sym1, sym2, price1, price2, direction=-1)
                
            elif action == 'close' and current_position != 0:
                # Close position
                self._close_position(pair_id, sym1, sym2, price1, price2)
    
    def _get_action(self, zscore, current_position, spread_array):
        """
        Simplified RL policy.
        
        In production, this would use the trained DQN to predict optimal action.
        Here we use rule-based proxy that mimics mean-reversion logic.
        """
        # Calculate momentum
        momentum = spread_array[-5] - spread_array[-10] if len(spread_array) >= 10 else 0
        
        # No position: look for entry signals
        if current_position == 0:
            if zscore > self.zscore_entry and momentum < 0:
                # Spread high and reverting down -> short spread
                return 'sell'
            elif zscore < -self.zscore_entry and momentum > 0:
                # Spread low and reverting up -> long spread
                return 'buy'
        
        # Has position: look for exit signals
        else:
            if abs(zscore) < self.zscore_exit:
                # Spread reverted to mean
                return 'close'
            
            # Stop loss: spread diverged further
            if current_position == 1 and zscore < -3:
                return 'close'
            elif current_position == -1 and zscore > 3:
                return 'close'
        
        return 'hold'
    
    def _open_position(self, pair_id, sym1, sym2, price1, price2, direction):
        """
        Open dollar-neutral pair position.
        
        direction: 1 = long spread, -1 = short spread
        """
        portfolio_value = self.Portfolio.TotalPortfolioValue
        position_value = portfolio_value * self.position_size
        
        # Calculate share quantities for dollar neutrality
        shares1 = int(position_value / price1) * direction
        shares2 = int(position_value / price2) * (-direction)
        
        # Execute orders
        self.MarketOrder(sym1, shares1)
        self.MarketOrder(sym2, shares2)
        
        # Update tracking
        self.positions[pair_id] = direction
        self.entry_prices[pair_id] = {'price1': price1, 'price2': price2}
        self.trade_count += 1
        
        self.Debug(f"Opened {pair_id}: Direction={direction}, Z-score={self._get_current_zscore(pair_id):.2f}")
    
    def _close_position(self, pair_id, sym1, sym2, price1, price2):
        """Close existing pair position."""
        
        # Liquidate both legs
        self.Liquidate(sym1)
        self.Liquidate(sym2)
        
        # Calculate PnL
        entry1 = self.entry_prices[pair_id]['price1']
        entry2 = self.entry_prices[pair_id]['price2']
        direction = self.positions[pair_id]
        
        ret1 = (price1 - entry1) / entry1
        ret2 = (price2 - entry2) / entry2
        
        if direction == 1:
            pnl_pct = ret1 - ret2
        else:
            pnl_pct = ret2 - ret1
        
        if pnl_pct > 0:
            self.winning_trades += 1
        
        self.Debug(f"Closed {pair_id}: PnL={pnl_pct*100:.2f}%, Z-score={self._get_current_zscore(pair_id):.2f}")
        
        # Reset tracking
        self.positions[pair_id] = 0
    
    def _get_current_zscore(self, pair_id):
        """Get current z-score for pair."""
        if len(self.spread_history[pair_id]) < self.lookback_window:
            return 0
        
        spread_array = np.array(self.spread_history[pair_id])
        mean = spread_array[-self.lookback_window:].mean()
        std = spread_array[-self.lookback_window:].std()
        
        if std == 0:
            return 0
        
        return (spread_array[-1] - mean) / std
    
    def OnEndOfAlgorithm(self):
        """Log final performance."""
        win_rate = (self.winning_trades / self.trade_count * 100) if self.trade_count > 0 else 0
        
        self.Debug(f"\n=== BACKTEST SUMMARY ===")
        self.Debug(f"Total Trades: {self.trade_count}")
        self.Debug(f"Winning Trades: {self.winning_trades}")
        self.Debug(f"Win Rate: {win_rate:.1f}%")
        self.Debug(f"Final Portfolio Value: ${self.Portfolio.TotalPortfolioValue:,.2f}")
        
        total_return = ((self.Portfolio.TotalPortfolioValue - 100000) / 100000) * 100
        self.Debug(f"Total Return: {total_return:.2f}%")
