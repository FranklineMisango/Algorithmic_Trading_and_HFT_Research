"""
Backtester for Music Royalties Strategy
Simulates buying/selling music royalty assets with realistic transaction costs
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class Trade:
    """Represents a single trade"""
    date: datetime
    asset_id: str
    action: str  # 'buy' or 'sell'
    price: float
    quantity: float  # Weight in portfolio
    buyer_fee: float
    seller_commission: float
    slippage: float
    total_cost: float


class RoyaltyBacktester:
    """
    Backtests music royalty asset strategy with realistic costs
    """
    
    def __init__(self, config: Dict):
        """
        Initialize backtester
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.initial_capital = config['backtest']['initial_capital']
        self.buyer_fee = config['transaction_costs']['buyer_fee']
        self.seller_commission = config['transaction_costs']['seller_commission']
        self.base_slippage = config['transaction_costs']['slippage']['base_rate']
        self.rebalancing_freq = config['portfolio']['rebalancing_frequency']
        
        # State
        self.portfolio = pd.DataFrame()
        self.cash = self.initial_capital
        self.portfolio_value = self.initial_capital
        self.trades = []
        self.equity_curve = []
        
    def run_backtest(self, universe_by_date: Dict[str, pd.DataFrame],
                    model, constructor) -> pd.DataFrame:
        """
        Run full backtest across time periods
        
        Args:
            universe_by_date: Dict mapping rebalance dates to available assets
            model: Trained price prediction model
            constructor: Portfolio constructor instance
            
        Returns:
            DataFrame with equity curve
        """
        logger.info("=== Starting Backtest ===")
        logger.info(f"Initial capital: ${self.initial_capital:,.0f}")
        
        rebalance_dates = sorted(universe_by_date.keys())
        
        for i, date in enumerate(rebalance_dates):
            logger.info(f"\n--- Rebalance {i+1}/{len(rebalance_dates)}: {date} ---")
            
            # Get available universe at this date
            universe = universe_by_date[date]
            
            if len(universe) == 0:
                logger.warning(f"No assets available on {date}")
                continue
            
            # Calculate mispricing
            universe = model.calculate_mispricing(universe)
            
            # Construct new portfolio
            new_portfolio = constructor.construct_portfolio(universe)
            
            if len(new_portfolio) == 0:
                logger.warning(f"No portfolio constructed on {date}")
                continue
            
            # Execute trades to rebalance
            self._rebalance(date, new_portfolio)
            
            # Calculate portfolio value (mark-to-market)
            self._update_portfolio_value(date, universe)
            
            # Record equity point
            self.equity_curve.append({
                'date': date,
                'portfolio_value': self.portfolio_value,
                'cash': self.cash,
                'n_holdings': len(self.portfolio)
            })
        
        # Convert to DataFrame
        equity_df = pd.DataFrame(self.equity_curve)
        
        logger.info(f"\n=== Backtest Complete ===")
        logger.info(f"Final portfolio value: ${self.portfolio_value:,.0f}")
        logger.info(f"Total return: {(self.portfolio_value/self.initial_capital - 1)*100:.2f}%")
        logger.info(f"Total trades: {len(self.trades)}")
        
        return equity_df
    
    def _rebalance(self, date: datetime, new_portfolio: pd.DataFrame) -> None:
        """
        Rebalance portfolio to new target
        
        Args:
            date: Rebalance date
            new_portfolio: Target portfolio with weights
        """
        # Current holdings
        current_assets = set(self.portfolio['asset_id']) if len(self.portfolio) > 0 else set()
        new_assets = set(new_portfolio['asset_id'])
        
        # Calculate target dollar amounts
        new_portfolio['target_value'] = new_portfolio['weight'] * self.portfolio_value
        
        # 1. SELL assets not in new portfolio
        for asset_id in current_assets - new_assets:
            self._sell_asset(date, asset_id)
        
        # 2. BUY new assets not in current portfolio
        for asset_id in new_assets - current_assets:
            asset_data = new_portfolio[new_portfolio['asset_id'] == asset_id].iloc[0]
            target_value = asset_data['target_value']
            price = asset_data['transaction_price']
            self._buy_asset(date, asset_id, target_value, price, asset_data)
        
        # 3. ADJUST positions for assets in both
        for asset_id in current_assets & new_assets:
            current_row = self.portfolio[self.portfolio['asset_id'] == asset_id].iloc[0]
            new_row = new_portfolio[new_portfolio['asset_id'] == asset_id].iloc[0]
            
            current_value = current_row['value']
            target_value = new_row['target_value']
            
            # Only rebalance if difference is significant
            if abs(target_value - current_value) > 0.01 * self.portfolio_value:
                if target_value > current_value:
                    # Increase position
                    add_value = target_value - current_value
                    self._buy_asset(date, asset_id, add_value, 
                                  new_row['transaction_price'], new_row, is_adjustment=True)
                else:
                    # Decrease position
                    reduce_value = current_value - target_value
                    self._sell_partial_asset(date, asset_id, reduce_value)
    
    def _buy_asset(self, date: datetime, asset_id: str, target_value: float,
                  price: float, asset_data: pd.Series, is_adjustment: bool = False) -> None:
        """
        Buy an asset
        
        Args:
            date: Transaction date
            asset_id: Asset identifier
            target_value: Target dollar value to buy
            price: Transaction price
            asset_data: Asset data row
            is_adjustment: Whether this is an adjustment of existing position
        """
        # Calculate transaction costs
        buyer_fee = self.buyer_fee
        slippage_cost = target_value * self.base_slippage
        total_cost = target_value + buyer_fee + slippage_cost
        
        # Check if we have enough cash
        if total_cost > self.cash:
            logger.warning(f"Insufficient cash for {asset_id}: need ${total_cost:.0f}, have ${self.cash:.0f}")
            return
        
        # Execute trade
        self.cash -= total_cost
        
        # Add to portfolio
        new_holding = {
            'asset_id': asset_id,
            'value': target_value,
            'price': price,
            'genre': asset_data.get('genre', 'Unknown'),
            'revenue_ltm': asset_data.get('revenue_ltm', 0),
            'catalog_age': asset_data.get('catalog_age', 0),
            'entry_date': date
        }
        
        if len(self.portfolio) == 0:
            self.portfolio = pd.DataFrame([new_holding])
        else:
            self.portfolio = pd.concat([self.portfolio, pd.DataFrame([new_holding])], 
                                      ignore_index=True)
        
        # Record trade
        trade = Trade(
            date=date,
            asset_id=asset_id,
            action='buy',
            price=price,
            quantity=target_value / self.portfolio_value,
            buyer_fee=buyer_fee,
            seller_commission=0,
            slippage=slippage_cost,
            total_cost=total_cost
        )
        self.trades.append(trade)
        
        action_type = "adjusted" if is_adjustment else "bought"
        logger.debug(f"  {action_type.capitalize()} {asset_id}: ${target_value:,.0f} (cost: ${total_cost:,.0f})")
    
    def _sell_asset(self, date: datetime, asset_id: str) -> None:
        """
        Sell entire position in an asset
        
        Args:
            date: Transaction date
            asset_id: Asset identifier
        """
        if len(self.portfolio) == 0:
            return
        
        # Find asset in portfolio
        asset_row = self.portfolio[self.portfolio['asset_id'] == asset_id]
        if len(asset_row) == 0:
            return
        
        asset_row = asset_row.iloc[0]
        asset_value = asset_row['value']
        
        # Calculate transaction costs
        seller_commission = asset_value * self.seller_commission
        slippage_cost = asset_value * self.base_slippage
        proceeds = asset_value - seller_commission - slippage_cost
        
        # Execute trade
        self.cash += proceeds
        
        # Remove from portfolio
        self.portfolio = self.portfolio[self.portfolio['asset_id'] != asset_id].reset_index(drop=True)
        
        # Record trade
        trade = Trade(
            date=date,
            asset_id=asset_id,
            action='sell',
            price=asset_row['price'],
            quantity=asset_value / self.portfolio_value,
            buyer_fee=0,
            seller_commission=seller_commission,
            slippage=slippage_cost,
            total_cost=seller_commission + slippage_cost
        )
        self.trades.append(trade)
        
        logger.debug(f"  Sold {asset_id}: ${asset_value:,.0f} (proceeds: ${proceeds:,.0f})")
    
    def _sell_partial_asset(self, date: datetime, asset_id: str, 
                           reduce_value: float) -> None:
        """
        Sell partial position in an asset
        
        Args:
            date: Transaction date
            asset_id: Asset identifier
            reduce_value: Dollar value to sell
        """
        # Similar to _sell_asset but reduces position instead of eliminating
        asset_idx = self.portfolio[self.portfolio['asset_id'] == asset_id].index[0]
        current_value = self.portfolio.loc[asset_idx, 'value']
        
        # Calculate costs
        seller_commission = reduce_value * self.seller_commission
        slippage_cost = reduce_value * self.base_slippage
        proceeds = reduce_value - seller_commission - slippage_cost
        
        # Update portfolio
        self.portfolio.loc[asset_idx, 'value'] = current_value - reduce_value
        self.cash += proceeds
        
        # Record trade
        trade = Trade(
            date=date,
            asset_id=asset_id,
            action='sell_partial',
            price=self.portfolio.loc[asset_idx, 'price'],
            quantity=reduce_value / self.portfolio_value,
            buyer_fee=0,
            seller_commission=seller_commission,
            slippage=slippage_cost,
            total_cost=seller_commission + slippage_cost
        )
        self.trades.append(trade)
    
    def _update_portfolio_value(self, date: datetime, universe: pd.DataFrame) -> None:
        """
        Update portfolio value based on current market prices
        
        Args:
            date: Current date
            universe: Current universe with prices
        """
        if len(self.portfolio) == 0:
            self.portfolio_value = self.cash
            return
        
        # Mark to market (use current prices if available)
        total_holdings_value = 0
        
        for idx, row in self.portfolio.iterrows():
            asset_id = row['asset_id']
            
            # Try to find current market price
            current_data = universe[universe['asset_id'] == asset_id]
            if len(current_data) > 0:
                current_price = current_data.iloc[0]['transaction_price']
                # Update to current market value (assuming value tracks revenue)
                # In reality, would need historical price data
                self.portfolio.loc[idx, 'value'] = row['value']  # Keep book value for now
            
            total_holdings_value += row['value']
        
        self.portfolio_value = total_holdings_value + self.cash
    
    def get_trades_df(self) -> pd.DataFrame:
        """
        Get all trades as DataFrame
        
        Returns:
            DataFrame of trades
        """
        if len(self.trades) == 0:
            return pd.DataFrame()
        
        trades_data = []
        for trade in self.trades:
            trades_data.append({
                'date': trade.date,
                'asset_id': trade.asset_id,
                'action': trade.action,
                'price': trade.price,
                'quantity': trade.quantity,
                'buyer_fee': trade.buyer_fee,
                'seller_commission': trade.seller_commission,
                'slippage': trade.slippage,
                'total_cost': trade.total_cost
            })
        
        return pd.DataFrame(trades_data)
    
    def calculate_transaction_cost_drag(self) -> float:
        """
        Calculate total transaction cost as % of portfolio
        
        Returns:
            Transaction cost ratio
        """
        if len(self.trades) == 0:
            return 0.0
        
        total_costs = sum(trade.total_cost for trade in self.trades)
        cost_ratio = total_costs / self.initial_capital
        
        return cost_ratio


def prepare_universe_by_date(data: pd.DataFrame, rebalancing_freq: str) -> Dict[str, pd.DataFrame]:
    """
    Prepare universe of assets grouped by rebalancing dates
    
    Args:
        data: Full dataset
        rebalancing_freq: 'annual', 'quarterly', or 'monthly'
        
    Returns:
        Dictionary mapping dates to available assets
    """
    data = data.sort_values('transaction_date')
    
    if rebalancing_freq == 'annual':
        # Group by year
        data['rebalance_period'] = data['transaction_date'].dt.year
    elif rebalancing_freq == 'quarterly':
        data['rebalance_period'] = data['transaction_date'].dt.to_period('Q')
    elif rebalancing_freq == 'monthly':
        data['rebalance_period'] = data['transaction_date'].dt.to_period('M')
    else:
        raise ValueError(f"Unknown rebalancing frequency: {rebalancing_freq}")
    
    # Get first date of each period
    universe_by_date = {}
    for period, group in data.groupby('rebalance_period'):
        period_start = group['transaction_date'].min()
        universe_by_date[period_start] = group
    
    return universe_by_date


if __name__ == '__main__':
    import yaml
    from pathlib import Path
    from data_loader import load_and_prepare_data
    from feature_engineering import engineer_all_features
    from model_trainer import train_and_validate_model
    from portfolio_constructor import PortfolioConstructor
    
    # Load config
    config_path = Path(__file__).parent / 'config.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Load and prepare data
    data_splits = load_and_prepare_data(config)
    
    # Engineer features
    train_df = engineer_all_features(data_splits['train'], config)
    val_df = engineer_all_features(data_splits['validation'], config)
    test_df = engineer_all_features(data_splits['test'], config)
    
    # Train model
    model, _ = train_and_validate_model(train_df, val_df, config)
    
    # Prepare universe
    universe_by_date = prepare_universe_by_date(test_df, config['portfolio']['rebalancing_frequency'])
    
    # Run backtest
    backtester = RoyaltyBacktester(config)
    constructor = PortfolioConstructor(config)
    equity_curve = backtester.run_backtest(universe_by_date, model, constructor)
    
    print("\n=== Backtest Results ===")
    print(f"Total return: {(backtester.portfolio_value/backtester.initial_capital - 1)*100:.2f}%")
    print(f"Transaction cost drag: {backtester.calculate_transaction_cost_drag()*100:.2f}%")
