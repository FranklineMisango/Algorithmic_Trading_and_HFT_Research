import numpy as np
import pandas as pd
from loguru import logger

class Backtester:
    def __init__(self, config):
        self.config = config
        self.initial_capital = config['backtest']['initial_capital']
        self.max_positions = config['parameters']['max_positions']
        self.stop_loss_bps = config['parameters']['stop_loss_bps'] / 10000
        self.max_spread_mult = config['parameters']['max_spread_multiplier']
        self.min_volume = config['parameters']['min_volume_usd']
        
    def calculate_position_size(self, capital, volatility, max_vol_pct=0.02):
        """Size position to target max volatility"""
        if volatility == 0:
            return 0
        return (capital * max_vol_pct) / volatility
    
    def calculate_slippage(self, trade_size, avg_volume, slippage_factor=2.0):
        """Linear slippage model"""
        if avg_volume == 0:
            return 0.0005  # 5 bps max
        
        slippage_bps = (trade_size / avg_volume) * slippage_factor
        return min(slippage_bps / 10000, 0.0005)
    
    def run_backtest(self, df, signals):
        """Execute backtest with position management"""
        logger.info("Running backtest")
        
        results = pd.DataFrame(index=df.index)
        results['signal'] = signals['signal']
        results['premium_index'] = signals['premium_index']
        results['upper_bound'] = signals['upper_bound']
        results['lower_bound'] = signals['lower_bound']
        
        # Portfolio state
        capital = self.initial_capital
        position = 0  # 1 = short perp/long spot, -1 = long perp/short spot
        entry_index = 0
        entry_bound = 0
        
        # Track performance
        equity_curve = []
        trades = []
        
        for i, (timestamp, row) in enumerate(df.iterrows()):
            signal = signals.loc[timestamp, 'signal']
            premium_idx = signals.loc[timestamp, 'premium_index']
            
            # Check liquidity conditions
            volume_ok = row['perp_volume'] * row['perp_price'] > self.min_volume
            
            # Position management
            if position == 0 and signal != 0 and volume_ok:
                # Enter position
                position = signal
                entry_index = premium_idx
                entry_bound = signals.loc[timestamp, 'upper_bound'] if signal == 1 else signals.loc[timestamp, 'lower_bound']
                
                # Calculate position size
                volatility = df['perp_price'].pct_change().rolling(24).std().iloc[i]
                position_size = self.calculate_position_size(capital, volatility)
                
                # Calculate slippage
                avg_volume = (row['perp_volume'] + row['spot_volume']) / 2
                slippage = self.calculate_slippage(position_size, avg_volume)
                
                entry_price_perp = row['perp_price'] * (1 + slippage * position)
                entry_price_spot = row['spot_price'] * (1 - slippage * position)
                
                logger.debug(f"Enter {position} at {timestamp}")
                
            elif position != 0:
                # Check exit conditions
                exit_signal = False
                
                # Exit if premium returns within bounds
                if position == 1 and premium_idx <= signals.loc[timestamp, 'upper_bound']:
                    exit_signal = True
                elif position == -1 and premium_idx >= signals.loc[timestamp, 'lower_bound']:
                    exit_signal = True
                
                # Stop loss: exit if moves 10 bps beyond entry bound
                if position == 1 and premium_idx > entry_bound + self.stop_loss_bps:
                    exit_signal = True
                    logger.warning(f"Stop loss triggered at {timestamp}")
                elif position == -1 and premium_idx < entry_bound - self.stop_loss_bps:
                    exit_signal = True
                    logger.warning(f"Stop loss triggered at {timestamp}")
                
                if exit_signal:
                    # Exit position
                    avg_volume = (row['perp_volume'] + row['spot_volume']) / 2
                    slippage = self.calculate_slippage(position_size, avg_volume)
                    
                    exit_price_perp = row['perp_price'] * (1 - slippage * position)
                    exit_price_spot = row['spot_price'] * (1 + slippage * position)
                    
                    # Calculate P&L
                    pnl_perp = position_size * (entry_price_perp - exit_price_perp) * position
                    pnl_spot = position_size * (exit_price_spot - entry_price_spot) * (-position)
                    pnl = pnl_perp + pnl_spot
                    
                    capital += pnl
                    
                    trades.append({
                        'entry_time': df.index[i - 1],
                        'exit_time': timestamp,
                        'position': position,
                        'pnl': pnl,
                        'return': pnl / position_size
                    })
                    
                    position = 0
                    logger.debug(f"Exit at {timestamp}, PnL: {pnl:.2f}")
            
            equity_curve.append(capital)
        
        results['equity'] = equity_curve
        results['returns'] = results['equity'].pct_change()
        
        trades_df = pd.DataFrame(trades)
        
        logger.info(f"Backtest complete: {len(trades)} trades")
        return results, trades_df
    
    def calculate_metrics(self, results, trades_df):
        """Calculate performance metrics"""
        metrics = {}
        
        # Returns
        total_return = (results['equity'].iloc[-1] / self.initial_capital - 1) * 100
        annual_return = ((results['equity'].iloc[-1] / self.initial_capital) ** (252 / len(results)) - 1) * 100
        
        # Risk
        returns = results['returns'].dropna()
        sharpe = returns.mean() / returns.std() * np.sqrt(252 * 24) if returns.std() > 0 else 0
        sortino = returns.mean() / returns[returns < 0].std() * np.sqrt(252 * 24) if len(returns[returns < 0]) > 0 else 0
        
        # Drawdown
        cummax = results['equity'].cummax()
        drawdown = (results['equity'] - cummax) / cummax
        max_drawdown = drawdown.min() * 100
        
        # Trading
        if len(trades_df) > 0:
            win_rate = (trades_df['pnl'] > 0).sum() / len(trades_df) * 100
            avg_win = trades_df[trades_df['pnl'] > 0]['pnl'].mean() if len(trades_df[trades_df['pnl'] > 0]) > 0 else 0
            avg_loss = trades_df[trades_df['pnl'] < 0]['pnl'].mean() if len(trades_df[trades_df['pnl'] < 0]) > 0 else 0
            profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else 0
        else:
            win_rate = 0
            profit_factor = 0
        
        metrics = {
            'Total Return (%)': total_return,
            'Annual Return (%)': annual_return,
            'Sharpe Ratio': sharpe,
            'Sortino Ratio': sortino,
            'Max Drawdown (%)': max_drawdown,
            'Win Rate (%)': win_rate,
            'Profit Factor': profit_factor,
            'Total Trades': len(trades_df)
        }
        
        return metrics
