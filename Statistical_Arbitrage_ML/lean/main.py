# region imports
from AlgorithmImports import *
import numpy as np
from datetime import timedelta
import pickle
import joblib
# endregion

class StatisticalArbitrageMLStrategy(QCAlgorithm):
    """
    Market-Neutral Statistical Arbitrage Strategy using Machine Learning
    
    Strategy Overview:
    - Predicts 3-day returns using ML model trained on momentum, mean reversion, and volume features
    - Constructs market-neutral long/short portfolios
    - Rebalances every 3 days (holding period)
    - Target: 20-28% annual return, Sharpe > 1.4, max drawdown < 20%
    
    Universe: Russell 3000 (or similar large-cap/mid-cap universe)
    """

    def Initialize(self):
        """Initialize algorithm parameters and data structures"""
        
        # Backtest period
        self.SetStartDate(2020, 1, 1)
        self.SetEndDate(2023, 12, 31)
        self.SetCash(1000000)  # $1M starting capital
        
        # Strategy parameters
        self.n_long = 20  # Number of long positions
        self.n_short = 20  # Number of short positions
        self.holding_days = 3  # Rebalance every 3 days
        self.max_position_size = 0.04  # Max 4% per position
        
        # Feature calculation parameters
        self.momentum_periods = [5, 10, 20, 60, 126, 252]
        self.ma_periods = [10, 20, 50, 100, 200]
        self.lookback_days = 300  # Days of historical data needed
        
        # Model placeholder (in production, load pre-trained model)
        self.model = None
        self.feature_names = []
        
        # Universe selection
        self.UniverseSettings.Resolution = Resolution.Daily
        self.AddUniverse(self.CoarseSelectionFunction, self.FineSelectionFunction)
        
        # Data structures
        self.securities_data = {}  # Store historical data per symbol
        self.current_portfolio = {}  # Current positions
        self.next_rebalance_time = self.Time
        
        # Schedule rebalancing
        self.Schedule.On(
            self.DateRules.EveryDay(),
            self.TimeRules.AfterMarketOpen("SPY", 30),
            self.CheckRebalance
        )
        
        # Performance tracking
        self.portfolio_value_history = []
        
        self.Debug(f"Algorithm initialized: {self.n_long} long, {self.n_short} short positions")
        self.Debug(f"Rebalancing every {self.holding_days} days")

    def CoarseSelectionFunction(self, coarse):
        """
        Coarse universe selection - filter by liquidity and price
        """
        # Filter criteria
        filtered = [x for x in coarse if 
                    x.HasFundamentalData and
                    x.Price > 5.0 and  # Minimum price
                    x.DollarVolume > 10000000]  # Minimum $10M daily volume
        
        # Sort by dollar volume and take top N
        sorted_by_volume = sorted(filtered, key=lambda x: x.DollarVolume, reverse=True)
        
        # Return top 300 (similar to Russell 3000 but manageable for backtesting)
        return [x.Symbol for x in sorted_by_volume[:300]]

    def FineSelectionFunction(self, fine):
        """
        Fine universe selection - additional filters
        """
        # Filter out specific sectors/industries if needed
        filtered = []
        
        for f in fine:
            # Exclude penny stocks, biotech, etc.
            if (f.AssetClassification.MorningstarSectorCode != MorningstarSectorCode.Healthcare or
                f.MarketCap > 1000000000):  # Exclude small biotech unless large cap
                filtered.append(f.Symbol)
        
        return filtered[:200]  # Limit universe size for computational efficiency

    def OnSecuritiesChanged(self, changes):
        """
        Handle universe changes - initialize data structures for new securities
        """
        for security in changes.AddedSecurities:
            symbol = security.Symbol
            
            if symbol not in self.securities_data:
                self.securities_data[symbol] = {
                    'history': RollingWindow[TradeBar](self.lookback_days),
                    'last_update': self.Time
                }
                
                # Request historical data
                history = self.History(symbol, self.lookback_days, Resolution.Daily)
                
                if not history.empty:
                    for index, row in history.iterrows():
                        bar = TradeBar()
                        bar.Symbol = symbol
                        bar.Time = index[1] if isinstance(index, tuple) else index
                        bar.Open = row['open']
                        bar.High = row['high']
                        bar.Low = row['low']
                        bar.Close = row['close']
                        bar.Volume = row['volume']
                        
                        self.securities_data[symbol]['history'].Add(bar)
        
        for security in changes.RemovedSecurities:
            symbol = security.Symbol
            if symbol in self.securities_data:
                del self.securities_data[symbol]
            
            # Liquidate if holding
            if self.Portfolio[symbol].Invested:
                self.Liquidate(symbol)

    def OnData(self, data):
        """
        Process incoming data and update rolling windows
        """
        for symbol in data.Keys:
            if symbol in self.securities_data and data[symbol] is not None:
                self.securities_data[symbol]['history'].Add(data[symbol])
                self.securities_data[symbol]['last_update'] = self.Time

    def CheckRebalance(self):
        """
        Check if it's time to rebalance the portfolio
        """
        if self.Time >= self.next_rebalance_time:
            self.Rebalance()
            self.next_rebalance_time = self.Time + timedelta(days=self.holding_days)

    def Rebalance(self):
        """
        Main rebalancing logic:
        1. Calculate features for all securities
        2. Generate predictions
        3. Construct long/short portfolio
        4. Execute trades
        """
        self.Debug(f"Rebalancing portfolio on {self.Time.date()}")
        
        # Calculate features and predictions
        predictions = self.GeneratePredictions()
        
        if len(predictions) < (self.n_long + self.n_short):
            self.Debug(f"Insufficient predictions: {len(predictions)}. Skipping rebalance.")
            return
        
        # Build portfolio
        long_positions, short_positions = self.ConstructPortfolio(predictions)
        
        # Execute trades
        self.ExecuteTrades(long_positions, short_positions)
        
        # Track performance
        self.portfolio_value_history.append({
            'time': self.Time,
            'value': self.Portfolio.TotalPortfolioValue
        })

    def GeneratePredictions(self):
        """
        Calculate features and generate return predictions for all securities
        """
        predictions = {}
        
        for symbol, data_dict in self.securities_data.items():
            history = data_dict['history']
            
            # Need sufficient data
            if not history.IsReady or history.Count < max(self.momentum_periods + self.ma_periods) + 10:
                continue
            
            # Calculate features
            features = self.CalculateFeatures(symbol, history)
            
            if features is None:
                continue
            
            # Generate prediction
            # Note: In production, load pre-trained model and use it here
            # For now, use a simple heuristic based on momentum
            predicted_return = self.SimplePredictionHeuristic(features)
            
            predictions[symbol] = {
                'predicted_return': predicted_return,
                'current_price': history[0].Close,
                'features': features
            }
        
        return predictions

    def CalculateFeatures(self, symbol, history):
        """
        Calculate momentum, mean reversion, and volume features
        """
        try:
            features = {}
            
            # Convert rolling window to lists
            closes = [bar.Close for bar in history]
            volumes = [bar.Volume for bar in history]
            
            if len(closes) < max(self.momentum_periods + self.ma_periods) + 10:
                return None
            
            # Current price
            current_price = closes[0]
            
            # Momentum features (rate of change)
            for period in self.momentum_periods:
                if period < len(closes):
                    past_price = closes[period]
                    if past_price > 0:
                        momentum = (current_price - past_price) / past_price
                        features[f'momentum_{period}d'] = momentum
            
            # Mean reversion features (distance from moving averages)
            for period in self.ma_periods:
                if period < len(closes):
                    ma = np.mean(closes[:period])
                    if ma > 0:
                        ma_dist = (current_price - ma) / ma
                        features[f'ma_dist_{period}d'] = ma_dist
            
            # Volume features
            if len(volumes) >= 126:
                recent_volume = np.mean(volumes[:20])
                avg_volume = np.mean(volumes[:126])
                if avg_volume > 0:
                    features['volume_ratio'] = recent_volume / avg_volume
            
            # Volatility
            if len(closes) >= 20:
                returns = [(closes[i] - closes[i+1]) / closes[i+1] for i in range(min(20, len(closes)-1))]
                features['volatility_20d'] = np.std(returns) if returns else 0
            
            return features
            
        except Exception as e:
            self.Debug(f"Error calculating features for {symbol}: {str(e)}")
            return None

    def SimplePredictionHeuristic(self, features):
        """
        Simple prediction heuristic (placeholder for ML model)
        
        In production, replace this with:
            feature_vector = np.array([features[name] for name in self.feature_names])
            prediction = self.model.predict([feature_vector])[0]
        """
        # Combine momentum and mean reversion signals
        score = 0.0
        
        # Short-term momentum (positive)
        if 'momentum_5d' in features:
            score += features['momentum_5d'] * 0.3
        
        # Mean reversion from 20-day MA (negative - bet on reversion)
        if 'ma_dist_20d' in features:
            score -= features['ma_dist_20d'] * 0.5
        
        # Long-term momentum (positive)
        if 'momentum_126d' in features:
            score += features['momentum_126d'] * 0.2
        
        return score

    def ConstructPortfolio(self, predictions):
        """
        Construct long/short portfolio based on predictions
        """
        # Sort by predicted return
        sorted_predictions = sorted(predictions.items(), 
                                   key=lambda x: x[1]['predicted_return'], 
                                   reverse=True)
        
        # Select top N for long, bottom N for short
        long_candidates = sorted_predictions[:self.n_long]
        short_candidates = sorted_predictions[-self.n_short:]
        
        # Equal weight positions
        long_weight = 0.5 / self.n_long  # 50% allocated to longs
        short_weight = 0.5 / self.n_short  # 50% allocated to shorts
        
        # Enforce position size limits
        long_weight = min(long_weight, self.max_position_size)
        short_weight = min(short_weight, self.max_position_size)
        
        # Build position dictionaries
        long_positions = {symbol: long_weight for symbol, data in long_candidates}
        short_positions = {symbol: -short_weight for symbol, data in short_candidates}
        
        self.Debug(f"Portfolio: {len(long_positions)} long, {len(short_positions)} short")
        
        return long_positions, short_positions

    def ExecuteTrades(self, long_positions, short_positions):
        """
        Execute trades to achieve target portfolio
        """
        target_positions = {**long_positions, **short_positions}
        
        # Liquidate positions not in target portfolio
        for symbol in list(self.current_portfolio.keys()):
            if symbol not in target_positions:
                if self.Portfolio[symbol].Invested:
                    self.Liquidate(symbol)
                    self.Debug(f"Liquidated {symbol}")
                del self.current_portfolio[symbol]
        
        # Execute new positions
        for symbol, target_weight in target_positions.items():
            if self.Securities.ContainsKey(symbol):
                try:
                    self.SetHoldings(symbol, target_weight)
                    self.current_portfolio[symbol] = target_weight
                except Exception as e:
                    self.Debug(f"Error trading {symbol}: {str(e)}")

    def OnEndOfAlgorithm(self):
        """
        Calculate and report final statistics
        """
        self.Debug("="*80)
        self.Debug("BACKTEST COMPLETE")
        self.Debug("="*80)
        
        # Calculate performance metrics
        total_return = (self.Portfolio.TotalPortfolioValue - 1000000) / 1000000
        num_years = (self.EndDate - self.StartDate).days / 365.25
        annual_return = (1 + total_return) ** (1/num_years) - 1
        
        self.Debug(f"Total Return: {total_return*100:.2f}%")
        self.Debug(f"Annual Return: {annual_return*100:.2f}%")
        self.Debug(f"Final Portfolio Value: ${self.Portfolio.TotalPortfolioValue:,.2f}")
        
        # Get Sharpe ratio from backtest statistics (if available)
        self.Debug(f"Sharpe Ratio: {self.Statistics.SharpeRatio:.2f}" if hasattr(self.Statistics, 'SharpeRatio') else "")
        self.Debug(f"Max Drawdown: {self.Statistics.Drawdown*100:.2f}%" if hasattr(self.Statistics, 'Drawdown') else "")
        
        self.Debug("="*80)
