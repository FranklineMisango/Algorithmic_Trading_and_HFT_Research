"""
Pragmatic Asset Allocation Model - Signal Generation Module
Implements momentum ranking, trend filters, market health signals, and yield curve analysis.
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

class PragmaticAssetAllocationSignals:
    """
    Signal generation for Pragmatic Asset Allocation Model.
    Implements all six steps of the strategy.
    """

    def __init__(self, config_path: str = "config.yaml"):
        """Initialize with configuration."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.signals_config = self.config['signals']
        self.assets = self.config['assets']

    def generate_momentum_signals(self, price_data: pd.DataFrame) -> pd.DataFrame:
        """
        Step 1: Rank risky assets by 12-month momentum and select top 2.

        Args:
            price_data: DataFrame with price data for risky assets

        Returns:
            DataFrame with momentum rankings and selection signals
        """
        try:
            lookback_months = self.signals_config['momentum']['lookback_months']
            lookback_days = lookback_months * 21  # Approximate trading days
            selection_count = self.signals_config['momentum']['selection_count']

            momentum_signals = pd.DataFrame(index=price_data.index)

            # Calculate momentum for each risky asset
            risky_assets = [asset['ticker'] for asset in self.assets['risky']]

            for ticker in risky_assets:
                if ticker in price_data.columns.levels[0]:
                    prices = price_data[ticker]['Close']
                    # Total return over lookback period
                    momentum = (prices / prices.shift(lookback_days) - 1)
                    momentum_signals[f'{ticker}_Momentum'] = momentum

            # Rank assets by momentum (higher = better)
            momentum_cols = [col for col in momentum_signals.columns if 'Momentum' in col]
            rankings = momentum_signals[momentum_cols].rank(axis=1, ascending=False, method='dense')

            # Select top performers
            for i in range(1, selection_count + 1):
                selected_assets = rankings.apply(lambda x: x.index[x == i].tolist(), axis=1)
                momentum_signals[f'Top_{i}_Asset'] = selected_assets.apply(
                    lambda x: x[0].replace('_Momentum', '') if x else None
                )

            # Binary selection signal
            momentum_signals['Momentum_Selected'] = rankings.apply(
                lambda x: [asset.replace('_Momentum', '') for asset in x.index[x <= selection_count]],
                axis=1
            )

            return momentum_signals

        except Exception as e:
            logger.error(f"Error generating momentum signals: {str(e)}")
            return pd.DataFrame()

    def generate_trend_signals(self, price_data: pd.DataFrame) -> pd.DataFrame:
        """
        Step 2: Apply trend filter - only invest if price > 12-month SMA.

        Args:
            price_data: DataFrame with price data

        Returns:
            DataFrame with trend filter signals
        """
        try:
            lookback_months = self.signals_config['trend_filter']['lookback_months']
            lookback_days = lookback_months * 21

            trend_signals = pd.DataFrame(index=price_data.index)

            # Apply trend filter to all assets
            all_assets = ([asset['ticker'] for asset in self.assets['risky']] +
                         [asset['ticker'] for asset in self.assets['hedging']])

            for ticker in all_assets:
                if ticker in price_data.columns.levels[0]:
                    prices = price_data[ticker]['Close']
                    sma = prices.rolling(window=lookback_days).mean()

                    trend_signals[f'{ticker}_SMA_{lookback_months}M'] = sma
                    trend_signals[f'{ticker}_Price_vs_SMA'] = prices / sma
                    trend_signals[f'{ticker}_Trend_Up'] = prices > sma

            return trend_signals

        except Exception as e:
            logger.error(f"Error generating trend signals: {str(e)}")
            return pd.DataFrame()

    def generate_market_health_signals(self, trend_signals: pd.DataFrame) -> pd.DataFrame:
        """
        Step 3: Cash allocation based on market health.
        If 2+ risky assets are in downtrend, allocate portion to cash.

        Args:
            trend_signals: DataFrame with trend signals

        Returns:
            DataFrame with market health and cash allocation signals
        """
        try:
            threshold = self.signals_config['market_health']['downtrend_threshold']
            cash_pct = self.signals_config['market_health']['cash_allocation_pct']

            market_health_signals = pd.DataFrame(index=trend_signals.index)

            # Count risky assets in downtrend
            risky_assets = [asset['ticker'] for asset in self.assets['risky']]
            downtrend_cols = [f'{ticker}_Trend_Up' for ticker in risky_assets]

            if all(col in trend_signals.columns for col in downtrend_cols):
                downtrend_count = (~trend_signals[downtrend_cols]).sum(axis=1)
                market_health_signals['Risky_Assets_Downtrend_Count'] = downtrend_count
                market_health_signals['Market_Stress_Signal'] = downtrend_count >= threshold
                market_health_signals['Cash_Allocation_Pct'] = np.where(
                    market_health_signals['Market_Stress_Signal'], cash_pct, 0
                )

            return market_health_signals

        except Exception as e:
            logger.error(f"Error generating market health signals: {str(e)}")
            return pd.DataFrame()

    def generate_yield_curve_signals(self, macroeconomic_data: pd.DataFrame) -> pd.DataFrame:
        """
        Step 4: Yield curve signal for full hedging/cash allocation.
        If yield curve inverted, move entire portfolio to cash.

        Args:
            macroeconomic_data: DataFrame with yield curve data

        Returns:
            DataFrame with yield curve signals
        """
        try:
            persistence_months = self.signals_config['yield_curve'].get('persistence_months', 1)

            yield_curve_signals = pd.DataFrame(index=macroeconomic_data.index)

            if 'Yield_Curve_Inverted' in macroeconomic_data.columns:
                # Require inversion to persist for specified months
                inverted = macroeconomic_data['Yield_Curve_Inverted']
                if persistence_months > 1:
                    # Rolling check for persistence
                    persistence_window = persistence_months * 21  # Approximate days
                    persistent_inversion = inverted.rolling(window=persistence_window).sum() == persistence_window
                    yield_curve_signals['Yield_Curve_Inverted_Persistent'] = persistent_inversion
                    yield_curve_signals['Full_Cash_Signal'] = persistent_inversion
                else:
                    yield_curve_signals['Yield_Curve_Inverted_Persistent'] = inverted
                    yield_curve_signals['Full_Cash_Signal'] = inverted

                yield_curve_signals['Yield_Curve_Spread'] = macroeconomic_data['Yield_Curve_Spread']

            return yield_curve_signals

        except Exception as e:
            logger.error(f"Error generating yield curve signals: {str(e)}")
            return pd.DataFrame()

    def generate_hedging_signals(self, yield_curve_signals: pd.DataFrame,
                               market_health_signals: pd.DataFrame) -> pd.DataFrame:
        """
        Step 5: Hedging portfolio construction.
        Maintain separate hedging portfolio when not fully in cash.

        Args:
            yield_curve_signals: DataFrame with yield curve signals
            market_health_signals: DataFrame with market health signals

        Returns:
            DataFrame with hedging allocation signals
        """
        try:
            hedging_signals = pd.DataFrame(index=yield_curve_signals.index if not yield_curve_signals.empty else market_health_signals.index)

            # Hedging allocation is inverse of cash allocation
            full_cash = yield_curve_signals.get('Full_Cash_Signal', pd.Series(False, index=hedging_signals.index))
            stress_cash_pct = market_health_signals.get('Cash_Allocation_Pct', pd.Series(0.0, index=hedging_signals.index))

            # Ensure they're Series with the same index
            if not isinstance(full_cash, pd.Series):
                full_cash = pd.Series(full_cash, index=hedging_signals.index)
            if not isinstance(stress_cash_pct, pd.Series):
                stress_cash_pct = pd.Series(stress_cash_pct, index=hedging_signals.index)

            # Total cash allocation
            total_cash_pct = np.where(full_cash, 1.0, stress_cash_pct)

            # Hedging allocation (when not in full cash)
            hedging_allocation_pct = np.where(full_cash, 0.0, 1.0 - stress_cash_pct)

            hedging_signals['Total_Cash_Allocation'] = total_cash_pct
            hedging_signals['Hedging_Allocation'] = hedging_allocation_pct
            hedging_signals['Risky_Allocation'] = 1.0 - total_cash_pct

            # Hedging basket split (bonds vs gold)
            hedging_split = self.config['portfolio']['allocation']['hedging_split']
            hedging_signals['Bonds_Allocation'] = hedging_allocation_pct * hedging_split['bonds']
            hedging_signals['Gold_Allocation'] = hedging_allocation_pct * hedging_split['gold']

            return hedging_signals

        except Exception as e:
            logger.error(f"Error generating hedging signals: {str(e)}")
            return pd.DataFrame()

    def generate_stop_loss_signals(self, price_data: pd.DataFrame,
                                 current_positions: pd.DataFrame) -> pd.DataFrame:
        """
        Step 6: Stop-loss mechanism for damage control.

        Args:
            price_data: DataFrame with current price data
            current_positions: DataFrame with current positions

        Returns:
            DataFrame with stop-loss signals
        """
        try:
            stop_loss_pct = self.signals_config['stop_loss']['level_pct']

            stop_loss_signals = pd.DataFrame(index=price_data.index)

            # Track peak prices for each position
            if not current_positions.empty:
                for ticker in current_positions.columns:
                    if ticker in price_data.columns.levels[0]:
                        prices = price_data[ticker]['Close']

                        # Calculate stop loss levels (only evaluated quarterly per config)
                        peak_price = prices.expanding().max()
                        stop_loss_level = peak_price * (1 - stop_loss_pct)

                        stop_loss_signals[f'{ticker}_Peak_Price'] = peak_price
                        stop_loss_signals[f'{ticker}_Stop_Loss_Level'] = stop_loss_level
                        stop_loss_signals[f'{ticker}_Stop_Loss_Triggered'] = prices <= stop_loss_level

            return stop_loss_signals

        except Exception as e:
            logger.error(f"Error generating stop-loss signals: {str(e)}")
            return pd.DataFrame()

    def generate_all_signals(self, data_dict: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        Generate all signals for the Pragmatic Asset Allocation Model.

        Args:
            data_dict: Dictionary containing price and macroeconomic data

        Returns:
            Dictionary containing all signal DataFrames
        """
        logger.info("Generating all signals for Pragmatic Asset Allocation Model")

        signals_dict = {}

        # Extract data
        risky_assets = data_dict.get('risky_assets')
        hedging_assets = data_dict.get('hedging_assets')
        macroeconomic = data_dict.get('macroeconomic')

        if risky_assets is None:
            logger.error("Risky assets data not found")
            return signals_dict

        # Combine price data
        all_price_data = risky_assets.copy()
        if hedging_assets is not None:
            all_price_data = pd.concat([all_price_data, hedging_assets], axis=1)

        # Step 1: Momentum signals
        logger.info("Step 1: Generating momentum signals")
        momentum_signals = self.generate_momentum_signals(risky_assets)
        signals_dict['momentum'] = momentum_signals

        # Step 2: Trend signals
        logger.info("Step 2: Generating trend signals")
        trend_signals = self.generate_trend_signals(all_price_data)
        signals_dict['trend'] = trend_signals

        # Step 3: Market health signals
        logger.info("Step 3: Generating market health signals")
        market_health_signals = self.generate_market_health_signals(trend_signals)
        signals_dict['market_health'] = market_health_signals

        # Step 4: Yield curve signals
        if macroeconomic is not None:
            logger.info("Step 4: Generating yield curve signals")
            yield_curve_signals = self.generate_yield_curve_signals(macroeconomic)
            signals_dict['yield_curve'] = yield_curve_signals
        else:
            logger.warning("Macroeconomic data not available for yield curve signals")
            yield_curve_signals = pd.DataFrame()

        # Step 5: Hedging signals
        logger.info("Step 5: Generating hedging signals")
        hedging_signals = self.generate_hedging_signals(yield_curve_signals, market_health_signals)
        signals_dict['hedging'] = hedging_signals

        # Step 6: Stop-loss signals (placeholder - requires position data)
        logger.info("Step 6: Stop-loss signals prepared (requires position data)")

        logger.info("All signals generated successfully")
        return signals_dict

    def get_signal_summary(self, signals_dict: Dict[str, pd.DataFrame],
                          as_of_date: str = None) -> Dict[str, any]:
        """
        Generate a summary of current signals.

        Args:
            signals_dict: Dictionary of signal DataFrames
            as_of_date: Date for signal summary (defaults to latest)

        Returns:
            Dictionary with signal summary
        """
        try:
            if as_of_date is None:
                # Use latest date
                dates = []
                for df in signals_dict.values():
                    if not df.empty:
                        dates.extend(df.index)
                if dates:
                    as_of_date = max(dates)
                else:
                    return {}

            summary = {'as_of_date': as_of_date}

            # Momentum summary
            if 'momentum' in signals_dict and not signals_dict['momentum'].empty:
                momentum_filtered = signals_dict['momentum'].loc[:as_of_date]
                if not momentum_filtered.empty:
                    momentum_data = momentum_filtered.iloc[-1]
                    selected_assets = momentum_data.get('Momentum_Selected', [])
                    summary['selected_risky_assets'] = selected_assets

            # Trend summary
            if 'trend' in signals_dict and not signals_dict['trend'].empty:
                trend_filtered = signals_dict['trend'].loc[:as_of_date]
                if not trend_filtered.empty:
                    trend_data = trend_filtered.iloc[-1]
                    risky_assets = [asset['ticker'] for asset in self.assets['risky']]
                    trending_up = []
                    for ticker in risky_assets:
                        if f'{ticker}_Trend_Up' in trend_data.index and trend_data[f'{ticker}_Trend_Up']:
                            trending_up.append(ticker)
                    summary['trending_up_assets'] = trending_up

            # Market health summary
            if 'market_health' in signals_dict and not signals_dict['market_health'].empty:
                health_filtered = signals_dict['market_health'].loc[:as_of_date]
                if not health_filtered.empty:
                    health_data = health_filtered.iloc[-1]
                    summary['market_stress'] = health_data.get('Market_Stress_Signal', False)
                    summary['cash_allocation_pct'] = health_data.get('Cash_Allocation_Pct', 0)

            # Yield curve summary
            if 'yield_curve' in signals_dict and not signals_dict['yield_curve'].empty:
                yc_filtered = signals_dict['yield_curve'].loc[:as_of_date]
                if not yc_filtered.empty:
                    yc_data = yc_filtered.iloc[-1]
                    summary['yield_curve_inverted'] = yc_data.get('Yield_Curve_Inverted_Persistent', False)
                    summary['full_cash_signal'] = yc_data.get('Full_Cash_Signal', False)

            # Hedging summary
            if 'hedging' in signals_dict and not signals_dict['hedging'].empty:
                hedge_filtered = signals_dict['hedging'].loc[:as_of_date]
                if not hedge_filtered.empty:
                    hedge_data = hedge_filtered.iloc[-1]
                    summary['total_cash_pct'] = hedge_data.get('Total_Cash_Allocation', 0)
                    summary['risky_allocation_pct'] = hedge_data.get('Risky_Allocation', 0)
                    summary['hedging_allocation_pct'] = hedge_data.get('Hedging_Allocation', 0)

            return summary

        except Exception as e:
            logger.error(f"Error generating signal summary: {str(e)}")
            return {}