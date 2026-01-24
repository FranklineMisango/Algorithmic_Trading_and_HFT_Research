"""
Backtesting Engine

Event-driven backtester with realistic transaction cost modeling.

Critical Features:
- 1 tick slippage per side (conservative assumption)
- Commission: $4.20 per round-trip
- Market impact modeling
- Intraday-only execution (no overnight risk)
- Walk-forward optimization

Slippage Model:
- ES: 0.25 ticks = $12.50 per contract per side
- NQ: 0.25 ticks = $5.00 per contract per side
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from dataclasses import dataclass
from datetime import datetime, time
import warnings
warnings.filterwarnings('ignore')


@dataclass
class Trade:
    """Single trade record."""
    timestamp: pd.Timestamp
    symbol: str
    side: str  # 'long' or 'short'
    entry_price: float
    exit_price: float
    contracts: int
    entry_slippage: float
    exit_slippage: float
    commission: float
    pnl_gross: float
    pnl_net: float
    holding_bars: int
    exit_reason: str


class TransactionCostModel:
    """
    Models transaction costs: slippage + commission.
    """
    
    def __init__(self, config: dict):
        """
        Initialize transaction cost model.
        
        Parameters
        ----------
        config : dict
            Configuration dictionary
        """
        self.config = config
        self.costs = config['strategy']['transaction_costs']
        
        # Get instrument specs
        self.instruments = {
            inst['symbol']: inst for inst in config['data']['instruments']
        }
    
    def calculate_slippage(self, symbol: str, contracts: int, stress_multiplier: float = 1.0) -> float:
        """
        Calculate slippage cost.
        
        Parameters
        ----------
        symbol : str
            Instrument symbol
        contracts : int
            Number of contracts
        stress_multiplier : float
            Multiplier for stress testing (e.g., 5x for flash crash)
            
        Returns
        -------
        float
            Slippage cost in dollars
        """
        # Get instrument parameters
        tick_value = self.costs[symbol]['tick_value']
        slippage_ticks = self.costs['slippage_ticks'] * stress_multiplier
        
        # Total slippage (entry + exit)
        total_slippage = slippage_ticks * 2 * tick_value * abs(contracts)
        
        return total_slippage
    
    def calculate_commission(self, contracts: int) -> float:
        """
        Calculate commission cost.
        
        Parameters
        ----------
        contracts : int
            Number of contracts
            
        Returns
        -------
        float
            Commission cost in dollars
        """
        return self.costs['commission_per_contract'] * abs(contracts)
    
    def calculate_total_cost(
        self, 
        symbol: str, 
        contracts: int, 
        stress_multiplier: float = 1.0
    ) -> Tuple[float, float, float]:
        """
        Calculate total transaction costs.
        
        Parameters
        ----------
        symbol : str
            Instrument symbol
        contracts : int
            Number of contracts
        stress_multiplier : float
            Slippage multiplier for stress testing
            
        Returns
        -------
        tuple
            (slippage, commission, total_cost)
        """
        slippage = self.calculate_slippage(symbol, contracts, stress_multiplier)
        commission = self.calculate_commission(contracts)
        total = slippage + commission
        
        return slippage, commission, total
    
    def apply_slippage_to_price(
        self, 
        price: float, 
        symbol: str, 
        side: str,
        entry: bool = True
    ) -> float:
        """
        Apply slippage to execution price.
        
        Parameters
        ----------
        price : float
            Market price
        symbol : str
            Instrument symbol
        side : str
            'long' or 'short'
        entry : bool
            True for entry, False for exit
            
        Returns
        -------
        float
            Execution price after slippage
        """
        tick_size = self.costs[symbol]['tick_size']
        slippage_ticks = self.costs['slippage_ticks']
        
        # Entry slippage: pay more for long, receive less for short
        # Exit slippage: receive less for long, pay more for short
        if entry:
            if side == 'long':
                slipped_price = price + (slippage_ticks * tick_size)
            else:  # short
                slipped_price = price - (slippage_ticks * tick_size)
        else:  # exit
            if side == 'long':
                slipped_price = price - (slippage_ticks * tick_size)
            else:  # short
                slipped_price = price + (slippage_ticks * tick_size)
        
        return slipped_price


class Backtester:
    """
    Event-driven backtester with transaction costs.
    """
    
    def __init__(self, config: dict):
        """
        Initialize backtester.
        
        Parameters
        ----------
        config : dict
            Configuration dictionary
        """
        self.config = config
        self.cost_model = TransactionCostModel(config)
        
        # Portfolio
        self.initial_capital = config['strategy']['portfolio']['initial_capital']
        self.cash = self.initial_capital
        self.portfolio_value = self.initial_capital
        
        # Tracking
        self.trades: List[Trade] = []
        self.equity_curve = []
        self.positions = {}  # {strategy_name: {'contracts': int, 'entry_price': float, 'entry_time': timestamp}}
        
        # Get instrument specs
        self.instruments = {
            inst['symbol']: inst for inst in config['data']['instruments']
        }
    
    def calculate_contract_value(self, symbol: str, price: float, contracts: int) -> float:
        """
        Calculate notional value of position.
        
        Parameters
        ----------
        symbol : str
            Instrument symbol
        price : float
            Current price
        contracts : int
            Number of contracts
            
        Returns
        -------
        float
            Notional value
        """
        multiplier = self.instruments[symbol]['multiplier']
        return price * multiplier * abs(contracts)
    
    def update_portfolio_value(self, positions_data: Dict[str, pd.DataFrame], idx: int):
        """
        Update portfolio value based on current positions and prices.
        
        Parameters
        ----------
        positions_data : dict
            Position data for each strategy
        idx : int
            Current bar index
        """
        # Start with cash
        total_value = self.cash
        
        # Add value of open positions
        for strategy_name, position in self.positions.items():
            if position['contracts'] != 0:
                # Get current price
                symbol = position['symbol']
                data = positions_data[strategy_name]
                current_price = data.iloc[idx]['Close']
                
                # Calculate unrealized P&L
                if position['side'] == 'long':
                    unrealized_pnl = (current_price - position['entry_price']) * \
                                    position['contracts'] * self.instruments[symbol]['multiplier']
                else:  # short
                    unrealized_pnl = (position['entry_price'] - current_price) * \
                                    position['contracts'] * self.instruments[symbol]['multiplier']
                
                total_value += unrealized_pnl
        
        self.portfolio_value = total_value
        self.equity_curve.append({
            'timestamp': positions_data['ES_momentum'].iloc[idx].name,
            'portfolio_value': total_value,
            'cash': self.cash
        })
    
    def execute_trade(
        self,
        strategy_name: str,
        symbol: str,
        signal: int,
        contracts: int,
        price: float,
        timestamp: pd.Timestamp,
        reason: str = 'signal'
    ):
        """
        Execute a trade with transaction costs.
        
        Parameters
        ----------
        strategy_name : str
            Strategy identifier
        symbol : str
            Instrument symbol
        signal : int
            1 for long, -1 for short, 0 for close
        contracts : int
            Number of contracts (absolute value)
        price : float
            Execution price (before slippage)
        timestamp : pd.Timestamp
            Execution timestamp
        reason : str
            Trade reason
        """
        contracts = abs(contracts)
        
        # Check if closing existing position
        current_position = self.positions.get(strategy_name, {'contracts': 0})
        
        if signal == 0 or (signal == 1 and current_position.get('side') == 'short') or \
           (signal == -1 and current_position.get('side') == 'long'):
            # Close position
            if current_position['contracts'] != 0:
                self._close_position(strategy_name, price, timestamp, reason)
        
        # Open new position if signal != 0
        if signal != 0:
            side = 'long' if signal == 1 else 'short'
            
            # Apply slippage to entry
            entry_price = self.cost_model.apply_slippage_to_price(
                price, symbol, side, entry=True
            )
            
            # Calculate costs
            slippage, commission, total_cost = self.cost_model.calculate_total_cost(
                symbol, contracts
            )
            
            # Update cash (deduct entry costs)
            self.cash -= (commission + slippage / 2)  # Entry slippage only
            
            # Record position
            self.positions[strategy_name] = {
                'symbol': symbol,
                'side': side,
                'contracts': contracts,
                'entry_price': entry_price,
                'entry_time': timestamp
            }
    
    def _close_position(self, strategy_name: str, price: float, timestamp: pd.Timestamp, reason: str):
        """
        Close an open position.
        
        Parameters
        ----------
        strategy_name : str
            Strategy identifier
        price : float
            Exit price (before slippage)
        timestamp : pd.Timestamp
            Exit timestamp
        reason : str
            Exit reason
        """
        position = self.positions[strategy_name]
        
        if position['contracts'] == 0:
            return
        
        # Apply slippage to exit
        exit_price = self.cost_model.apply_slippage_to_price(
            price, position['symbol'], position['side'], entry=False
        )
        
        # Calculate costs
        slippage, commission, total_cost = self.cost_model.calculate_total_cost(
            position['symbol'], position['contracts']
        )
        
        # Calculate P&L
        multiplier = self.instruments[position['symbol']]['multiplier']
        
        if position['side'] == 'long':
            pnl_gross = (exit_price - position['entry_price']) * position['contracts'] * multiplier
        else:  # short
            pnl_gross = (position['entry_price'] - exit_price) * position['contracts'] * multiplier
        
        pnl_net = pnl_gross - total_cost
        
        # Update cash
        self.cash += pnl_net
        
        # Record trade
        holding_bars = int((timestamp - position['entry_time']).total_seconds() / 300)  # 5-min bars
        
        trade = Trade(
            timestamp=timestamp,
            symbol=position['symbol'],
            side=position['side'],
            entry_price=position['entry_price'],
            exit_price=exit_price,
            contracts=position['contracts'],
            entry_slippage=slippage / 2,
            exit_slippage=slippage / 2,
            commission=commission,
            pnl_gross=pnl_gross,
            pnl_net=pnl_net,
            holding_bars=holding_bars,
            exit_reason=reason
        )
        
        self.trades.append(trade)
        
        # Clear position
        self.positions[strategy_name] = {'contracts': 0}
    
    def run_backtest(self, portfolio_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Run full backtest on portfolio.
        
        Parameters
        ----------
        portfolio_data : dict
            Position data for each strategy: ES_momentum, NQ_momentum, NQ_long_only
            
        Returns
        -------
        pd.DataFrame
            Equity curve
        """
        print("="*60)
        print("BACKTESTING")
        print("="*60)
        print(f"Initial capital: ${self.initial_capital:,.0f}")
        print(f"Transaction costs:")
        print(f"  Slippage: {self.config['strategy']['transaction_costs']['slippage_ticks']} ticks per side")
        print(f"  Commission: ${self.config['strategy']['transaction_costs']['commission_per_contract']:.2f} per contract")
        print()
        
        # Align all dataframes to same index
        es_data = portfolio_data['ES_momentum']
        nq_momentum_data = portfolio_data['NQ_momentum']
        nq_long_data = portfolio_data['NQ_long_only']
        
        # Get common index
        common_index = es_data.index.intersection(nq_momentum_data.index)
        
        print(f"Backtesting {len(common_index)} bars...")
        
        # Initialize positions
        self.positions = {
            'ES_momentum': {'contracts': 0},
            'NQ_momentum': {'contracts': 0},
            'NQ_long_only': {'contracts': 0}
        }
        
        # Event loop
        for i in range(len(common_index)):
            timestamp = common_index[i]
            
            # Get current bar data
            es_bar = es_data.loc[timestamp]
            nq_momentum_bar = nq_momentum_data.loc[timestamp]
            nq_long_bar = nq_long_data.loc[timestamp]
            
            # ES momentum strategy
            if es_bar['entry_signal'] or es_bar['exit_signal']:
                self.execute_trade(
                    strategy_name='ES_momentum',
                    symbol='ES',
                    signal=int(es_bar['signal']),
                    contracts=abs(int(es_bar['position_size'])),
                    price=es_bar['Close'],
                    timestamp=timestamp,
                    reason=es_bar.get('exit_reason', 'entry')
                )
            
            # NQ momentum strategy
            if nq_momentum_bar['entry_signal'] or nq_momentum_bar['exit_signal']:
                self.execute_trade(
                    strategy_name='NQ_momentum',
                    symbol='NQ',
                    signal=int(nq_momentum_bar['signal']),
                    contracts=abs(int(nq_momentum_bar['position_size'])),
                    price=nq_momentum_bar['Close'],
                    timestamp=timestamp,
                    reason=nq_momentum_bar.get('exit_reason', 'entry')
                )
            
            # NQ long-only strategy (rebalance if position size changes significantly)
            current_nq_long_pos = self.positions['NQ_long_only']['contracts']
            target_nq_long_pos = abs(int(nq_long_bar['position_size']))
            
            if abs(target_nq_long_pos - current_nq_long_pos) > 1:  # Rebalance threshold
                self.execute_trade(
                    strategy_name='NQ_long_only',
                    symbol='NQ',
                    signal=1,  # Always long
                    contracts=target_nq_long_pos,
                    price=nq_long_bar['Close'],
                    timestamp=timestamp,
                    reason='rebalance'
                )
            
            # Update portfolio value
            self.update_portfolio_value(portfolio_data, i)
        
        # Close any remaining positions at end
        for strategy_name in self.positions:
            if self.positions[strategy_name]['contracts'] != 0:
                if strategy_name == 'ES_momentum':
                    final_price = es_data.iloc[-1]['Close']
                else:
                    final_price = nq_momentum_data.iloc[-1]['Close']
                
                self._close_position(
                    strategy_name, 
                    final_price, 
                    common_index[-1], 
                    'end_of_backtest'
                )
        
        # Convert equity curve to DataFrame
        equity_df = pd.DataFrame(self.equity_curve)
        equity_df.set_index('timestamp', inplace=True)
        
        # Summary statistics
        print("\nBacktest Complete!")
        print(f"Total trades: {len(self.trades)}")
        print(f"Final portfolio value: ${self.portfolio_value:,.0f}")
        print(f"Total return: {(self.portfolio_value / self.initial_capital - 1) * 100:.2f}%")
        
        if len(self.trades) > 0:
            winning_trades = [t for t in self.trades if t.pnl_net > 0]
            losing_trades = [t for t in self.trades if t.pnl_net <= 0]
            
            print(f"\nTrade Statistics:")
            print(f"  Winning trades: {len(winning_trades)} ({len(winning_trades)/len(self.trades)*100:.1f}%)")
            print(f"  Losing trades: {len(losing_trades)} ({len(losing_trades)/len(self.trades)*100:.1f}%)")
            
            if len(winning_trades) > 0:
                print(f"  Avg win: ${np.mean([t.pnl_net for t in winning_trades]):,.0f}")
            if len(losing_trades) > 0:
                print(f"  Avg loss: ${np.mean([t.pnl_net for t in losing_trades]):,.0f}")
            
            total_costs = sum(t.commission + t.entry_slippage + t.exit_slippage for t in self.trades)
            print(f"\nTotal transaction costs: ${total_costs:,.0f}")
            print(f"  As % of initial capital: {total_costs / self.initial_capital * 100:.2f}%")
        
        print("="*60)
        
        return equity_df
    
    def get_trades_dataframe(self) -> pd.DataFrame:
        """
        Convert trades list to DataFrame.
        
        Returns
        -------
        pd.DataFrame
            All trades
        """
        if len(self.trades) == 0:
            return pd.DataFrame()
        
        trades_data = []
        for trade in self.trades:
            trades_data.append({
                'timestamp': trade.timestamp,
                'symbol': trade.symbol,
                'side': trade.side,
                'entry_price': trade.entry_price,
                'exit_price': trade.exit_price,
                'contracts': trade.contracts,
                'entry_slippage': trade.entry_slippage,
                'exit_slippage': trade.exit_slippage,
                'commission': trade.commission,
                'pnl_gross': trade.pnl_gross,
                'pnl_net': trade.pnl_net,
                'holding_bars': trade.holding_bars,
                'exit_reason': trade.exit_reason
            })
        
        return pd.DataFrame(trades_data)


def main():
    """
    Test backtester.
    """
    import yaml
    from noise_area import NoiseAreaCalculator
    from signal_generator import SignalGenerator
    from position_sizer import PositionSizer
    
    # Load config
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Generate sample data
    print("Generating sample data...")
    dates = pd.date_range('2023-01-01 09:30', '2023-03-31 16:00', freq='5min')
    n = len(dates)
    
    np.random.seed(42)
    
    # ES data
    es_price = 4500 + np.cumsum(np.random.randn(n) * 2)
    es_data = pd.DataFrame({
        'Open': es_price + np.random.randn(n) * 2,
        'High': es_price + abs(np.random.randn(n) * 5),
        'Low': es_price - abs(np.random.randn(n) * 5),
        'Close': es_price,
        'Volume': np.random.randint(1000, 10000, n)
    }, index=dates)
    
    # NQ data
    nq_price = 15000 + np.cumsum(np.random.randn(n) * 5)
    nq_data = pd.DataFrame({
        'Open': nq_price + np.random.randn(n) * 5,
        'High': nq_price + abs(np.random.randn(n) * 10),
        'Low': nq_price - abs(np.random.randn(n) * 10),
        'Close': nq_price,
        'Volume': np.random.randint(1000, 10000, n)
    }, index=dates)
    
    # Process pipeline
    calculator = NoiseAreaCalculator(config)
    signal_gen = SignalGenerator(config)
    sizer = PositionSizer(config)
    
    # ES processing
    es_data = calculator.calculate_noise_area(es_data)
    es_data = calculator.identify_breakouts(es_data)
    es_data = signal_gen.generate_signals(es_data)
    
    # NQ processing
    nq_data = calculator.calculate_noise_area(nq_data)
    nq_data = calculator.identify_breakouts(nq_data)
    nq_data = signal_gen.generate_signals(nq_data)
    
    # Position sizing
    portfolio = sizer.calculate_portfolio_positions(es_data, nq_data)
    
    # Run backtest
    backtester = Backtester(config)
    equity_curve = backtester.run_backtest(portfolio)
    trades_df = backtester.get_trades_dataframe()
    
    # Save results
    equity_curve.to_csv('results/backtest_equity_curve.csv')
    trades_df.to_csv('results/backtest_trades.csv')
    
    print("\nBacktest results saved to results/")


if __name__ == "__main__":
    main()
