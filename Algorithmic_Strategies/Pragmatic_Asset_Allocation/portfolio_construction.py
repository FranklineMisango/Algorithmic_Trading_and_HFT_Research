"""
Pragmatic Asset Allocation Model - Portfolio Construction Module
Implements tranched rebalancing, position sizing, and hedging portfolio management.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import yaml
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PragmaticAssetAllocationPortfolio:
    """
    Portfolio construction for Pragmatic Asset Allocation Model.
    Implements quarterly tranched rebalancing with 12-month holding periods.
    """

    def __init__(self, config_path: str = "config.yaml"):
        """Initialize with configuration."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.portfolio_config = self.config['portfolio']
        self.costs_config = self.config['costs']
        self.assets = self.config['assets']

    def get_quarterly_rebalance_dates(self, start_date: str, end_date: str) -> List[datetime]:
        """
        Generate quarterly rebalance dates (end of March, June, September, December).

        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)

        Returns:
            List of rebalance dates
        """
        try:
            start = pd.to_datetime(start_date)
            end = pd.to_datetime(end_date)

            # Generate all quarter ends in the date range
            quarter_ends = pd.date_range(start=start, end=end, freq='Q')

            # Filter to ensure we have dates within range
            rebalance_dates = [date for date in quarter_ends if start <= date <= end]

            return sorted(rebalance_dates)

        except Exception as e:
            logger.error(f"Error generating rebalance dates: {str(e)}")
            return []

    def apply_tranche_system(self, rebalance_dates: List[datetime]) -> Dict[int, List[datetime]]:
        """
        Apply 4-tranche system where each tranche is rebalanced every 3 months.

        Args:
            rebalance_dates: List of all quarterly rebalance dates

        Returns:
            Dictionary mapping tranche number to its rebalance dates
        """
        try:
            tranches = self.portfolio_config['tranches']
            tranche_schedule = {}

            for tranche in range(1, tranches + 1):
                # Each tranche rebalances every 4th quarter, offset by tranche number
                tranche_dates = rebalance_dates[tranche-1::tranches]
                tranche_schedule[tranche] = tranche_dates

            return tranche_schedule

        except Exception as e:
            logger.error(f"Error applying tranche system: {str(e)}")
            return {}

    def calculate_position_sizes(self, signals_summary: Dict[str, any],
                               current_portfolio_value: float) -> Dict[str, float]:
        """
        Calculate position sizes based on current signals.

        Args:
            signals_summary: Current signal summary
            current_portfolio_value: Current total portfolio value

        Returns:
            Dictionary of position sizes by asset
        """
        try:
            position_sizes = {}

            # Extract allocation percentages from signals
            risky_allocation_pct = signals_summary.get('risky_allocation_pct', 0)
            hedging_allocation_pct = signals_summary.get('hedging_allocation_pct', 0)
            cash_allocation_pct = signals_summary.get('total_cash_pct', 0)

            # Selected risky assets
            selected_risky = signals_summary.get('selected_risky_assets', [])
            trending_up = signals_summary.get('trending_up_assets', [])

            # Only invest in selected assets that are trending up
            eligible_risky_assets = [asset for asset in selected_risky if asset in trending_up]

            # Calculate risky asset positions
            if eligible_risky_assets and risky_allocation_pct > 0:
                risky_value = current_portfolio_value * risky_allocation_pct
                per_asset_value = risky_value / len(eligible_risky_assets)

                for asset in eligible_risky_assets:
                    position_sizes[asset] = per_asset_value

            # Calculate hedging positions
            if hedging_allocation_pct > 0:
                hedging_value = current_portfolio_value * hedging_allocation_pct

                # Split between bonds and gold
                bonds_allocation = signals_summary.get('bonds_allocation', hedging_allocation_pct * 0.5)
                gold_allocation = signals_summary.get('gold_allocation', hedging_allocation_pct * 0.5)

                bonds_value = current_portfolio_value * bonds_allocation
                gold_value = current_portfolio_value * gold_allocation

                # Find hedging asset tickers
                bonds_ticker = next((asset['ticker'] for asset in self.assets['hedging']
                                   if 'bond' in asset['name'].lower()), None)
                gold_ticker = next((asset['ticker'] for asset in self.assets['hedging']
                                  if 'gold' in asset['name'].lower()), None)

                if bonds_ticker:
                    position_sizes[bonds_ticker] = bonds_value
                if gold_ticker:
                    position_sizes[gold_ticker] = gold_value

            # Cash position
            cash_value = current_portfolio_value * cash_allocation_pct
            position_sizes['CASH'] = cash_value

            return position_sizes

        except Exception as e:
            logger.error(f"Error calculating position sizes: {str(e)}")
            return {}

    def apply_transaction_costs(self, position_changes: Dict[str, float],
                              price_data: pd.DataFrame, current_date: str) -> Dict[str, float]:
        """
        Apply transaction costs to position changes.

        Args:
            position_changes: Dictionary of position changes (positive = buy, negative = sell)
            price_data: Current price data
            current_date: Current date for price lookup

        Returns:
            Dictionary of costs by asset
        """
        try:
            costs = {}
            total_costs = 0

            for asset, change_value in position_changes.items():
                if asset == 'CASH' or change_value == 0:
                    costs[asset] = 0
                    continue

                # Get current price
                if asset in price_data.columns.levels[0]:
                    current_price = price_data[asset]['Adj Close'].loc[current_date]

                    # Calculate trade value and shares
                    trade_value = abs(change_value)
                    shares = trade_value / current_price

                    # Apply costs
                    trading_cost_bps = self.costs_config['etf_trading_cost_bps']

                    # Additional cost for illiquid assets
                    if asset in ['EEM']:  # MSCI EM is less liquid
                        trading_cost_bps += self.costs_config['illiquid_penalty_bps']

                    # Total cost in basis points
                    total_cost_bps = trading_cost_bps + self.costs_config['slippage_bps']

                    # Calculate dollar cost
                    cost_dollars = trade_value * (total_cost_bps / 10000)  # Convert bps to decimal

                    costs[asset] = cost_dollars
                    total_costs += cost_dollars
                else:
                    costs[asset] = 0

            costs['TOTAL_COSTS'] = total_costs
            return costs

        except Exception as e:
            logger.error(f"Error applying transaction costs: {str(e)}")
            return {}

    def execute_rebalance(self, current_positions: Dict[str, float],
                         target_positions: Dict[str, float],
                         price_data: pd.DataFrame, current_date: str) -> Dict[str, any]:
        """
        Execute portfolio rebalance from current to target positions.

        Args:
            current_positions: Current position values
            target_positions: Target position values
            price_data: Current price data
            current_date: Current rebalance date

        Returns:
            Dictionary with rebalance results
        """
        try:
            rebalance_results = {
                'date': current_date,
                'current_positions': current_positions.copy(),
                'target_positions': target_positions.copy(),
                'trades': {},
                'costs': {},
                'final_positions': {}
            }

            # Calculate position changes
            all_assets = set(current_positions.keys()) | set(target_positions.keys())
            position_changes = {}

            for asset in all_assets:
                current_value = current_positions.get(asset, 0)
                target_value = target_positions.get(asset, 0)
                change = target_value - current_value
                position_changes[asset] = change

                if abs(change) > 1e-6:  # Only record meaningful changes
                    rebalance_results['trades'][asset] = change

            # Apply transaction costs
            costs = self.apply_transaction_costs(position_changes, price_data, current_date)
            rebalance_results['costs'] = costs

            # Calculate final positions after costs
            final_positions = {}
            total_costs = costs.get('TOTAL_COSTS', 0)

            # Distribute costs proportionally across the portfolio
            total_portfolio_value = sum(target_positions.values())

            if total_portfolio_value > 0:
                cost_distribution_factor = total_costs / total_portfolio_value

                for asset, target_value in target_positions.items():
                    if asset != 'CASH':
                        # Reduce position value by proportional cost
                        adjusted_value = target_value * (1 - cost_distribution_factor)
                        final_positions[asset] = adjusted_value
                    else:
                        # Cash position absorbs the costs
                        final_positions[asset] = target_value - total_costs
            else:
                final_positions = target_positions.copy()

            rebalance_results['final_positions'] = final_positions

            return rebalance_results

        except Exception as e:
            logger.error(f"Error executing rebalance: {str(e)}")
            return {}

    def track_tranche_holdings(self, tranche_schedule: Dict[int, List[datetime]],
                             rebalance_date: datetime) -> Dict[int, datetime]:
        """
        Track which tranches should be rebalanced and their holding periods.

        Args:
            tranche_schedule: Dictionary of tranche rebalance dates
            rebalance_date: Current rebalance date

        Returns:
            Dictionary mapping tranche to next rebalance date
        """
        try:
            tranche_status = {}

            for tranche, dates in tranche_schedule.items():
                # Find the most recent rebalance date for this tranche
                past_dates = [date for date in dates if date <= rebalance_date]
                if past_dates:
                    last_rebalance = max(past_dates)
                    # Next rebalance is 4 quarters later
                    next_rebalance = last_rebalance + pd.DateOffset(months=12)
                    tranche_status[tranche] = next_rebalance

            return tranche_status

        except Exception as e:
            logger.error(f"Error tracking tranche holdings: {str(e)}")
            return {}

    def run_portfolio_construction(self, signals_dict: Dict[str, pd.DataFrame],
                                 price_data: pd.DataFrame,
                                 start_date: str = None, end_date: str = None) -> Dict[str, pd.DataFrame]:
        """
        Run complete portfolio construction process.

        Args:
            signals_dict: Dictionary of signal DataFrames
            price_data: Combined price data for all assets
            start_date: Start date (defaults to config)
            end_date: End date (defaults to config)

        Returns:
            Dictionary with portfolio construction results
        """
        try:
            if start_date is None:
                start_date = self.config['backtest']['start_date']
            if end_date is None:
                end_date = self.config['backtest']['end_date']

            logger.info(f"Running portfolio construction from {start_date} to {end_date}")

            # Get rebalance dates and tranche schedule
            rebalance_dates = self.get_quarterly_rebalance_dates(start_date, end_date)
            tranche_schedule = self.apply_tranche_system(rebalance_dates)

            # Initialize portfolio tracking
            portfolio_history = []
            current_positions = {'CASH': self.config['backtest']['initial_capital']}
            portfolio_value = self.config['backtest']['initial_capital']

            # Process each rebalance date
            for rebalance_date in rebalance_dates:
                date_str = rebalance_date.strftime('%Y-%m-%d')

                logger.info(f"Processing rebalance on {date_str}")

                # Get signals as of rebalance date
                from signal_generation import PragmaticAssetAllocationSignals
                signal_gen = PragmaticAssetAllocationSignals()
                signals_summary = signal_gen.get_signal_summary(signals_dict, date_str)

                # Calculate target positions
                target_positions = self.calculate_position_sizes(signals_summary, portfolio_value)

                # Execute rebalance
                rebalance_results = self.execute_rebalance(
                    current_positions, target_positions, price_data, date_str
                )

                # Update current positions and portfolio value
                if rebalance_results:
                    current_positions = rebalance_results['final_positions']
                    portfolio_value = sum(current_positions.values())

                # Track tranche status
                tranche_status = self.track_tranche_holdings(tranche_schedule, rebalance_date)

                # Record portfolio state
                portfolio_record = {
                    'date': rebalance_date,
                    'portfolio_value': portfolio_value,
                    'positions': current_positions.copy(),
                    'signals_summary': signals_summary,
                    'rebalance_results': rebalance_results,
                    'tranche_status': tranche_status
                }

                portfolio_history.append(portfolio_record)

            # Convert to DataFrames
            results = {
                'portfolio_history': pd.DataFrame(portfolio_history),
                'tranche_schedule': pd.DataFrame.from_dict(tranche_schedule, orient='index').T,
                'final_positions': current_positions
            }

            logger.info("Portfolio construction complete")
            return results

        except Exception as e:
            logger.error(f"Error in portfolio construction: {str(e)}")
            return {}