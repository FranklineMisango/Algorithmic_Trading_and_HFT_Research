from AlgorithmImports import *
import numpy as np
from collections import deque

class MLMeanReversionAlgorithm(QCAlgorithm):
    """
    ML-Enhanced Mean Reversion Strategy for Russell 3000
    - Entry: QPI_3day < 15 AND ML_Probability > 0.60
    - VIX Regime Filter: Dynamic allocation based on volatility
    - Position Limits: Max 20 long, 20 short
    - Risk: 5% stop loss, 6-day time stop
    """
    
    def Initialize(self):
        self.SetStartDate(2020, 1, 1)
        self.SetEndDate(2023, 12, 31)
        self.SetCash(1000000)
        
        # VIX for regime detection
        self.vix = self.AddData(CBOE, "VIX", Resolution.Daily).Symbol
        self.vix_sma15 = SimpleMovingAverage(15)
        self.vix_history = deque(maxlen=15)
        
        # Universe: Russell 3000 (simplified to liquid stocks)
        self.UniverseSettings.Resolution = Resolution.Daily
        self.AddUniverse(self.CoarseSelectionFunction)
        
        # Strategy parameters
        self.qpi_threshold = 15
        self.ml_prob_threshold = 0.60
        self.max_long_positions = 20
        self.max_short_positions = 20
        self.stop_loss_pct = 0.05
        self.holding_period = 6
        
        # Position tracking
        self.long_positions = {}
        self.short_positions = {}
        
        # Rebalance daily
        self.Schedule.On(
            self.DateRules.EveryDay(self.vix),
            self.TimeRules.AfterMarketOpen(self.vix, 30),
            self.Rebalance
        )
        
    def CoarseSelectionFunction(self, coarse):
        """Select liquid Russell 3000 stocks"""
        filtered = [x for x in coarse if x.HasFundamentalData 
                    and x.Price > 1.0 
                    and x.DollarVolume > 1000000]
        
        # Sort by dollar volume, take top 500 as proxy for Russell 3000
        sorted_by_volume = sorted(filtered, key=lambda x: x.DollarVolume, reverse=True)
        return [x.Symbol for x in sorted_by_volume[:500]]
    
    def OnData(self, data):
        """Update VIX data"""
        if self.vix in data:
            vix_close = data[self.vix].Close
            self.vix_history.append(vix_close)
            self.vix_sma15.Update(data[self.vix].EndTime, vix_close)
    
    def Rebalance(self):
        """Main rebalancing logic"""
        # 1. Determine VIX regime
        is_bear_market = self.IsBearMarket()
        
        # 2. Set allocation based on regime
        if is_bear_market:
            long_allocation = 0.1
            short_allocation = 0.2
        else:
            long_allocation = 1.1
            short_allocation = 0.2
        
        # 3. Exit positions (stop loss or time stop)
        self.ExitPositions()
        
        # 4. Generate new signals
        long_candidates = []
        short_candidates = []
        
        for symbol in self.ActiveSecurities.Keys:
            if symbol == self.vix:
                continue
                
            if not self.Securities[symbol].HasData:
                continue
            
            # Calculate QPI and features
            qpi = self.CalculateQPI(symbol)
            if qpi is None:
                continue
            
            features = self.CalculateFeatures(symbol)
            if features is None:
                continue
            
            # ML probability (simplified - in production, load trained model)
            ml_prob_long = self.PredictProbabilityLong(features)
            ml_prob_short = self.PredictProbabilityShort(features)
            
            # Long signal
            if qpi < self.qpi_threshold and ml_prob_long > self.ml_prob_threshold:
                long_candidates.append((symbol, ml_prob_long))
            
            # Short signal (inverse QPI logic)
            if qpi > (100 - self.qpi_threshold) and ml_prob_short > self.ml_prob_threshold:
                short_candidates.append((symbol, ml_prob_short))
        
        # 5. Enter new positions
        self.EnterPositions(long_candidates, short_candidates, long_allocation, short_allocation)
    
    def IsBearMarket(self):
        """VIX regime filter: Bear if VIX > SMA15 * 1.15"""
        if not self.vix_sma15.IsReady or len(self.vix_history) == 0:
            return False
        
        current_vix = self.vix_history[-1]
        threshold = self.vix_sma15.Current.Value * 1.15
        return current_vix > threshold
    
    def CalculateQPI(self, symbol):
        """Calculate 3-day QPI"""
        history = self.History(symbol, 25, Resolution.Daily)
        if history.empty or len(history) < 25:
            return None
        
        closes = history['close'].values
        volumes = history['volume'].values
        
        # Returns
        ret_1d = (closes[-1] / closes[-2] - 1) if len(closes) > 1 else 0
        ret_3d = (closes[-1] / closes[-4] - 1) if len(closes) > 3 else 0
        
        # Volume ratio
        vol_ratio = volumes[-1] / np.mean(volumes[-20:]) if len(volumes) >= 20 else 1.0
        
        # Volatility
        returns = np.diff(closes) / closes[:-1]
        volatility = np.std(returns[-20:]) if len(returns) >= 20 else 0.01
        
        # QPI formula
        raw_qpi = 50 + (ret_3d / (volatility + 1e-6)) * 10 - (vol_ratio - 1) * 5
        qpi = np.clip(raw_qpi, 0, 100)
        
        return qpi
    
    def CalculateFeatures(self, symbol):
        """Calculate ML features"""
        history = self.History(symbol, 25, Resolution.Daily)
        if history.empty or len(history) < 25:
            return None
        
        closes = history['close'].values
        volumes = history['volume'].values
        
        # RSI
        rsi = self.CalculateRSI(closes, 14)
        
        # Bollinger Band position
        sma_20 = np.mean(closes[-20:])
        std_20 = np.std(closes[-20:])
        bb_position = (closes[-1] - sma_20) / (std_20 + 1e-6)
        
        # Volume surge
        volume_surge = volumes[-1] / np.mean(volumes[-5:]) if len(volumes) >= 5 else 1.0
        
        # Momentum
        mom_5 = (closes[-1] / closes[-6] - 1) if len(closes) > 5 else 0
        mom_10 = (closes[-1] / closes[-11] - 1) if len(closes) > 10 else 0
        mom_20 = (closes[-1] / closes[-21] - 1) if len(closes) > 20 else 0
        
        vol_ratio = volumes[-1] / np.mean(volumes[-20:]) if len(volumes) >= 20 else 1.0
        
        return {
            'rsi_14': rsi,
            'bb_position': bb_position,
            'volume_surge': volume_surge,
            'mom_5': mom_5,
            'mom_10': mom_10,
            'mom_20': mom_20,
            'vol_ratio': vol_ratio
        }
    
    def CalculateRSI(self, prices, period=14):
        """Calculate RSI"""
        if len(prices) < period + 1:
            return 50
        
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])
        
        if avg_loss == 0:
            return 100
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def PredictProbabilityLong(self, features):
        """Simplified ML prediction (replace with trained model)"""
        # Heuristic: oversold conditions favor mean reversion
        score = 0.5
        if features['rsi_14'] < 30:
            score += 0.15
        if features['bb_position'] < -1.5:
            score += 0.10
        if features['mom_5'] < -0.03:
            score += 0.10
        return min(score, 1.0)
    
    def PredictProbabilityShort(self, features):
        """Simplified ML prediction for short"""
        score = 0.5
        if features['rsi_14'] > 70:
            score += 0.15
        if features['bb_position'] > 1.5:
            score += 0.10
        if features['mom_5'] > 0.03:
            score += 0.10
        return min(score, 1.0)
    
    def EnterPositions(self, long_candidates, short_candidates, long_alloc, short_alloc):
        """Enter new positions"""
        # Sort by probability
        long_candidates.sort(key=lambda x: x[1], reverse=True)
        short_candidates.sort(key=lambda x: x[1], reverse=True)
        
        # Long positions
        available_long_slots = self.max_long_positions - len(self.long_positions)
        for symbol, prob in long_candidates[:available_long_slots]:
            if symbol not in self.long_positions:
                weight = long_alloc / self.max_long_positions
                self.SetHoldings(symbol, weight)
                self.long_positions[symbol] = {
                    'entry_price': self.Securities[symbol].Price,
                    'entry_time': self.Time,
                    'days_held': 0
                }
        
        # Short positions
        available_short_slots = self.max_short_positions - len(self.short_positions)
        for symbol, prob in short_candidates[:available_short_slots]:
            if symbol not in self.short_positions:
                weight = -short_alloc / self.max_short_positions
                self.SetHoldings(symbol, weight)
                self.short_positions[symbol] = {
                    'entry_price': self.Securities[symbol].Price,
                    'entry_time': self.Time,
                    'days_held': 0
                }
    
    def ExitPositions(self):
        """Exit positions based on stop loss or time stop"""
        # Check long positions
        for symbol in list(self.long_positions.keys()):
            position = self.long_positions[symbol]
            current_price = self.Securities[symbol].Price
            entry_price = position['entry_price']
            days_held = (self.Time - position['entry_time']).days
            
            # Stop loss
            if current_price < entry_price * (1 - self.stop_loss_pct):
                self.Liquidate(symbol)
                del self.long_positions[symbol]
                continue
            
            # Time stop
            if days_held >= self.holding_period:
                self.Liquidate(symbol)
                del self.long_positions[symbol]
        
        # Check short positions
        for symbol in list(self.short_positions.keys()):
            position = self.short_positions[symbol]
            current_price = self.Securities[symbol].Price
            entry_price = position['entry_price']
            days_held = (self.Time - position['entry_time']).days
            
            # Stop loss (inverse for shorts)
            if current_price > entry_price * (1 + self.stop_loss_pct):
                self.Liquidate(symbol)
                del self.short_positions[symbol]
                continue
            
            # Time stop
            if days_held >= self.holding_period:
                self.Liquidate(symbol)
                del self.short_positions[symbol]
