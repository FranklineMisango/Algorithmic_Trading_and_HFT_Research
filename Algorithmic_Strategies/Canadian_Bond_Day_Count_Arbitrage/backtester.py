"""
Backtester for Canadian Bond Day Count Arbitrage Strategy

Implements the full strategy including:
- Signal generation
- Duration hedging
- Transaction cost modeling
- Risk management
- Performance measurement
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import yaml
import logging
from pathlib import Path
from dataclasses import dataclass

from data_acquisition import CanadianBondDataAcquisition
from feature_engineering import CanadianBondFeatureEngineering, DayCountCalculator


@dataclass
class Trade:
    """Representation of a single arbitrage trade."""
    trade_id: str
    bond_identifier: str
    entry_date: datetime
    exit_date: datetime
    coupon_date: datetime
    
    # Entry details
    entry_price_clean: float
    entry_price_dirty: float
    accrued_interest_entry: float
    
    # Exit details
    exit_price_clean: float
    exit_price_dirty: float
    accrued_interest_exit: float
    coupon_received: float
    
    # Position details
    notional: float
    position_size: float  # Number of bonds
    
    # Hedge details
    hedge_composition: Dict
    hedge_pnl: float
    
    # Costs
    transaction_costs: float
    financing_costs: float
    slippage: float
    
    # P&L
    gross_pnl: float
    net_pnl: float
    return_pct: float
    return_bps: float
    
    # Metadata
    coupon_period_length: int
    days_held: int
    success: bool
    failure_reason: Optional[str] = None


class CanadianBondArbitrageBacktester:
    """
    Backtester for Canadian Bond Day Count Arbitrage Strategy.
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize backtester with configuration."""
        self.config = self._load_config(config_path)
        self.logger = self._setup_logging()
        
        # Initialize components
        self.data_acquirer = CanadianBondDataAcquisition(config_path)
        self.feature_engineer = CanadianBondFeatureEngineering(config_path)
        
        # Strategy parameters
        self.initial_capital = self.config['backtest']['initial_capital']
        self.max_position_pct = self.config['risk']['max_single_position_pct'] / 100
        self.max_exposure_pct = self.config['risk']['max_total_exposure_pct'] / 100
        
        # Cost parameters
        self.commission_pct = self.config['costs']['commission_pct'] / 100
        self.base_slippage_bps = self.config['costs'].get('base_slippage_bps', 2.0)
        
        # Track state
        self.current_capital = self.initial_capital
        self.trades: List[Trade] = []
        self.active_positions: List[Dict] = []
    
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _setup_logging(self) -> logging.Logger:
        """Set up logging."""
        log_dir = Path(self.config['logging']['log_dir'])
        log_dir.mkdir(exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / 'backtest.log'),
                logging.StreamHandler()
            ]
        )
        return logging.getLogger(__name__)
    
    def run(self, start_date: str, end_date: str, 
            run_control: bool = True) -> Dict:
        """
        Run backtest over specified period.
        
        Args:
            start_date: Backtest start date (YYYY-MM-DD)
            end_date: Backtest end date (YYYY-MM-DD)
            run_control: Whether to run control test on non-target periods
        
        Returns:
            Dictionary with backtest results
        """
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        
        self.logger.info(f"Starting backtest from {start_date} to {end_date}")
        self.logger.info(f"Initial capital: ${self.initial_capital:,.2f}")
        
        # Reset state
        self.current_capital = self.initial_capital
        self.trades = []
        self.active_positions = []
        
        # Run main backtest
        main_results = self._run_backtest_period(start_dt, end_dt, target_periods_only=True)
        
        # Run control test if requested
        control_results = None
        if run_control:
            self.logger.info("\n=== Running Control Test ===")
            self.current_capital = self.initial_capital
            self.trades = []
            self.active_positions = []
            control_results = self._run_backtest_period(start_dt, end_dt, target_periods_only=False)
        
        # Combine results
        results = {
            'main_backtest': main_results,
            'control_test': control_results,
            'strategy_alpha': self._calculate_alpha(main_results, control_results)
        }
        
        # Save results
        self._save_results(results)
        
        return results
    
    def _run_backtest_period(self, start_date: datetime, end_date: datetime,
                            target_periods_only: bool = True) -> Dict:
        """Run backtest for a specific period."""
        
        # Generate trading dates (assuming daily evaluation)
        trading_dates = pd.bdate_range(start_date, end_date, freq='B')
        
        self.logger.info(f"Evaluating {len(trading_dates)} trading days")
        
        for current_date in trading_dates:
            self._process_trading_day(current_date, target_periods_only)
        
        # Calculate performance metrics
        results = self._calculate_performance_metrics()
        
        return results
    
    def _process_trading_day(self, current_date: datetime, 
                            target_periods_only: bool):
        """Process a single trading day."""
        
        # 1. Check for exits (bonds past coupon date)
        self._process_exits(current_date)
        
        # 2. Get current bond universe
        bonds = self.data_acquirer.get_canadian_government_bonds(current_date)
        
        if bonds is None or len(bonds) == 0:
            return
        
        # 3. Get detailed bond data
        bond_details = self.data_acquirer.get_bond_details(
            bonds['identifier'].tolist(),
            current_date
        )
        
        # 4. Engineer features
        features = self.feature_engineer.engineer_features(bond_details, current_date)
        
        # 5. Generate signals
        if target_periods_only:
            signals = self.feature_engineer.generate_signals(features)
        else:
            # For control test, use non-target periods
            signals = self._generate_control_signals(features)
        
        # 6. Execute trades based on signals
        self._execute_signals(signals, current_date, features)
    
    def _generate_control_signals(self, features: pd.DataFrame) -> pd.DataFrame:
        """Generate signals for control test (non-181/182 day periods)."""
        df = features.copy()
        
        # Use similar logic but for NON-target periods
        control_periods = self.config['backtest'].get('control_periods', [180, 183, 184])
        
        df['signal'] = 0
        df['signal_strength'] = 0.0
        
        buy_conditions = (
            df['coupon_period_length'].isin(control_periods) &
            df['in_entry_window'] &
            (df['arbitrage_profit_bps'] >= self.config['signals']['min_profit_bps'])
        )
        
        df.loc[buy_conditions, 'signal'] = 1
        df.loc[buy_conditions, 'signal_strength'] = df.loc[buy_conditions, 'arbitrage_profit_bps']
        
        return df
    
    def _execute_signals(self, signals: pd.DataFrame, current_date: datetime,
                        all_bonds: pd.DataFrame):
        """Execute trades based on signals."""
        
        buy_signals = signals[signals['signal'] == 1].copy()
        
        if len(buy_signals) == 0:
            return
        
        # Sort by signal strength (expected profit)
        buy_signals = buy_signals.sort_values('signal_strength', ascending=False)
        
        for _, signal_bond in buy_signals.iterrows():
            # Check if we can take another position
            if not self._can_add_position():
                self.logger.debug(f"Position limits reached on {current_date}")
                break
            
            # Calculate position size
            position_size = self._calculate_position_size(signal_bond)
            
            if position_size <= 0:
                continue
            
            # Create duration hedge
            hedge = self.feature_engineer.create_duration_hedge(
                signal_bond,
                all_bonds
            )
            
            # Validate hedge quality
            if not self._validate_hedge(hedge):
                self.logger.warning(f"Hedge validation failed for {signal_bond['identifier']}")
                continue
            
            # Execute the trade
            self._enter_position(signal_bond, position_size, hedge, current_date)
    
    def _can_add_position(self) -> bool:
        """Check if we can add another position based on risk limits."""
        # Check concurrent position limit
        if len(self.active_positions) >= self.config['risk']['max_concurrent_trades']:
            return False
        
        # Check total exposure
        total_exposure = sum(pos['notional'] for pos in self.active_positions)
        if total_exposure >= self.current_capital * self.max_exposure_pct:
            return False
        
        return True
    
    def _calculate_position_size(self, bond: pd.Series) -> float:
        """Calculate position size based on risk management rules."""
        # Maximum notional based on position limit
        max_notional = self.current_capital * self.max_position_pct
        
        # Bond price (assume $100 face value per bond)
        bond_price = bond['PX_DIRTY']
        
        # Number of bonds
        num_bonds = max_notional / bond_price
        
        return num_bonds
    
    def _validate_hedge(self, hedge: Dict) -> bool:
        """Validate hedge quality."""
        max_tracking_error = self.config['signals']['duration_match_tolerance']
        
        # Check tracking error relative to target PV01
        relative_error = hedge['tracking_error'] / hedge['target_pv01']
        
        if relative_error > max_tracking_error:
            return False
        
        return True
    
    def _enter_position(self, bond: pd.Series, position_size: float,
                       hedge: Dict, entry_date: datetime):
        """Enter a new arbitrage position."""
        
        # Calculate costs
        notional = position_size * bond['PX_DIRTY']
        transaction_cost = notional * self.commission_pct
        
        # Estimate slippage
        slippage = (notional / 1_000_000) * self.base_slippage_bps / 10000 * notional
        
        # Record position
        position = {
            'bond_identifier': bond['identifier'],
            'entry_date': entry_date,
            'exit_date': bond['NXT_CPN_DT'],
            'coupon_date': bond['NXT_CPN_DT'],
            'entry_price_dirty': bond['PX_DIRTY'],
            'entry_price_clean': bond['PX_CLEAN'],
            'position_size': position_size,
            'notional': notional,
            'hedge': hedge,
            'transaction_cost': transaction_cost,
            'slippage': slippage,
            'coupon_rate': bond['CPN'],
            'coupon_period_length': bond['coupon_period_length']
        }
        
        self.active_positions.append(position)
        self.current_capital -= (notional + transaction_cost + slippage)
        
        self.logger.info(
            f"Entered position: {bond['identifier']} "
            f"Notional: ${notional:,.2f} "
            f"Entry: {entry_date.date()} "
            f"Exit: {bond['NXT_CPN_DT'].date()}"
        )
    
    def _process_exits(self, current_date: datetime):
        """Process exits for positions past their coupon date."""
        
        positions_to_exit = []
        
        for i, position in enumerate(self.active_positions):
            if current_date >= position['coupon_date']:
                positions_to_exit.append(i)
        
        # Exit positions in reverse order to maintain indices
        for i in sorted(positions_to_exit, reverse=True):
            self._exit_position(self.active_positions[i], current_date)
            del self.active_positions[i]
    
    def _exit_position(self, position: Dict, exit_date: datetime):
        """Exit an arbitrage position and calculate P&L."""
        
        # Simulate receiving coupon
        coupon_payment = (position['coupon_rate'] / 100 / 2) * 100 * position['position_size']
        
        # Assume we sell bond at clean price (simplified)
        # In reality, would need to fetch actual exit price
        exit_price_clean = position['entry_price_clean']  # Simplified assumption
        exit_proceeds = exit_price_clean * position['position_size']
        
        # Calculate costs
        exit_transaction_cost = exit_proceeds * self.commission_pct
        exit_slippage = (exit_proceeds / 1_000_000) * self.base_slippage_bps / 10000 * exit_proceeds
        
        # Calculate financing costs (time value of money)
        days_held = (exit_date - position['entry_date']).days
        financing_rate_annual = self.config['costs']['financing_rate_spread_bps'] / 10000
        financing_cost = position['notional'] * financing_rate_annual * (days_held / 365)
        
        # Hedge P&L (assume hedge is duration-neutral, so minimal P&L)
        # In reality, would need to track hedge performance
        hedge_pnl = 0  # Simplified
        
        # Total costs
        total_transaction_costs = (
            position['transaction_cost'] + exit_transaction_cost +
            position['slippage'] + exit_slippage
        )
        
        # Gross P&L from arbitrage
        # The arbitrage profit comes from the day count mismatch
        theoretical_arbitrage = (
            position['position_size'] * 
            (position.get('arbitrage_profit_per_100', 0))
        )
        
        # Actual P&L components
        gross_pnl = (
            coupon_payment +  # Received coupon
            exit_proceeds -   # Sold bond
            position['notional']  # Original investment
        )
        
        # Net P&L after costs
        net_pnl = gross_pnl - total_transaction_costs - financing_cost + hedge_pnl
        
        # Calculate returns
        return_pct = (net_pnl / position['notional']) * 100
        return_bps = return_pct * 100
        
        # Create trade record
        trade = Trade(
            trade_id=f"{position['bond_identifier']}_{position['entry_date'].strftime('%Y%m%d')}",
            bond_identifier=position['bond_identifier'],
            entry_date=position['entry_date'],
            exit_date=exit_date,
            coupon_date=position['coupon_date'],
            entry_price_clean=position['entry_price_clean'],
            entry_price_dirty=position['entry_price_dirty'],
            accrued_interest_entry=position['entry_price_dirty'] - position['entry_price_clean'],
            exit_price_clean=exit_price_clean,
            exit_price_dirty=exit_price_clean,  # Simplified
            accrued_interest_exit=0,
            coupon_received=coupon_payment,
            notional=position['notional'],
            position_size=position['position_size'],
            hedge_composition=position['hedge'],
            hedge_pnl=hedge_pnl,
            transaction_costs=total_transaction_costs,
            financing_costs=financing_cost,
            slippage=position['slippage'] + exit_slippage,
            gross_pnl=gross_pnl,
            net_pnl=net_pnl,
            return_pct=return_pct,
            return_bps=return_bps,
            coupon_period_length=position['coupon_period_length'],
            days_held=days_held,
            success=net_pnl > 0
        )
        
        self.trades.append(trade)
        self.current_capital += (position['notional'] + coupon_payment + exit_proceeds - 
                                exit_transaction_cost - exit_slippage - financing_cost)
        
        self.logger.info(
            f"Exited position: {position['bond_identifier']} "
            f"Net P&L: ${net_pnl:,.2f} ({return_bps:.2f} bps) "
            f"Days held: {days_held}"
        )
    
    def _calculate_performance_metrics(self) -> Dict:
        """Calculate comprehensive performance metrics."""
        
        if len(self.trades) == 0:
            return {
                'num_trades': 0,
                'win_rate': 0,
                'avg_return_bps': 0,
                'total_pnl': 0,
                'sharpe_ratio': 0
            }
        
        # Convert trades to DataFrame
        trades_df = pd.DataFrame([vars(t) for t in self.trades])
        
        # Basic metrics
        num_trades = len(trades_df)
        num_wins = trades_df['success'].sum()
        win_rate = num_wins / num_trades
        
        total_pnl = trades_df['net_pnl'].sum()
        avg_return_bps = trades_df['return_bps'].mean()
        median_return_bps = trades_df['return_bps'].median()
        
        # Risk metrics
        return_std = trades_df['return_bps'].std()
        sharpe_ratio = (avg_return_bps / return_std) if return_std > 0 else 0
        
        # Drawdown
        cumulative_pnl = trades_df['net_pnl'].cumsum()
        running_max = cumulative_pnl.cummax()
        drawdown = cumulative_pnl - running_max
        max_drawdown = drawdown.min()
        
        # Final capital
        final_capital = self.current_capital
        total_return_pct = ((final_capital - self.initial_capital) / self.initial_capital) * 100
        
        metrics = {
            'num_trades': num_trades,
            'num_wins': int(num_wins),
            'win_rate': win_rate,
            'avg_return_bps': avg_return_bps,
            'median_return_bps': median_return_bps,
            'std_return_bps': return_std,
            'sharpe_ratio': sharpe_ratio,
            'total_pnl': total_pnl,
            'max_drawdown': max_drawdown,
            'initial_capital': self.initial_capital,
            'final_capital': final_capital,
            'total_return_pct': total_return_pct,
            'avg_transaction_costs': trades_df['transaction_costs'].mean(),
            'avg_slippage': trades_df['slippage'].mean(),
            'avg_days_held': trades_df['days_held'].mean(),
            'coupon_period_breakdown': trades_df['coupon_period_length'].value_counts().to_dict()
        }
        
        return metrics
    
    def _calculate_alpha(self, main_results: Dict, control_results: Optional[Dict]) -> Optional[float]:
        """Calculate strategy alpha (excess return vs control)."""
        if control_results is None or main_results['num_trades'] == 0:
            return None
        
        main_return = main_results['avg_return_bps']
        control_return = control_results.get('avg_return_bps', 0)
        
        alpha = main_return - control_return
        
        return alpha
    
    def _save_results(self, results: Dict):
        """Save backtest results to disk."""
        # Save trades to CSV
        if len(self.trades) > 0:
            trades_df = pd.DataFrame([vars(t) for t in self.trades])
            trades_df.to_csv('data/backtest_trades.csv', index=False)
            self.logger.info("Saved trades to data/backtest_trades.csv")
        
        # Save summary metrics
        summary_df = pd.DataFrame([results['main_backtest']])
        summary_df.to_csv('data/backtest_summary.csv', index=False)
        self.logger.info("Saved summary to data/backtest_summary.csv")


if __name__ == "__main__":
    # Run backtest
    backtester = CanadianBondArbitrageBacktester()
    
    results = backtester.run(
        start_date='2023-01-01',
        end_date='2023-12-31',
        run_control=True
    )
    
    print("\n" + "="*60)
    print("CANADIAN BOND DAY COUNT ARBITRAGE - BACKTEST RESULTS")
    print("="*60)
    
    main = results['main_backtest']
    print(f"\nTotal Trades: {main['num_trades']}")
    print(f"Win Rate: {main['win_rate']:.2%}")
    print(f"Avg Return: {main['avg_return_bps']:.2f} bps")
    print(f"Sharpe Ratio: {main['sharpe_ratio']:.2f}")
    print(f"Total P&L: ${main['total_pnl']:,.2f}")
    print(f"Total Return: {main['total_return_pct']:.2f}%")
    
    if results['strategy_alpha'] is not None:
        print(f"\nStrategy Alpha: {results['strategy_alpha']:.2f} bps")
