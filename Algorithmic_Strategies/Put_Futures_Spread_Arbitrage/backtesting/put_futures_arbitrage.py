"""
Put-Call Parity Arbitrage Algorithm for QuantConnect Lean
Implements arbitrage between SPX options and ES futures based on put-call parity
"""

from AlgorithmImports import *
import numpy as np
import pandas as pd

class PutFuturesSpreadArbitrage(QCAlgorithm):

    def Initialize(self):
        """Initialize the algorithm"""
        self.SetStartDate(2023, 1, 1)  # Set start date
        self.SetEndDate(2024, 12, 31)  # Set end date
        self.SetCash(1000000)  # Set initial cash

        # Add SPX options and ES futures
        self.spx = self.AddEquity("SPY", Resolution.Daily).Symbol  # Use SPY as proxy for SPX
        self.es = self.AddFuture(Futures.Indices.SP500EMini, Resolution.Daily).Symbol

        # Risk-free rate (use 3-month T-bill as proxy)
        self.rf_rate = 0.05  # Initial estimate, will update

        # Trading parameters
        self.arbitrage_threshold = 0.01  # Minimum mispricing to trade
        self.max_position_size = 10  # Max contracts per trade
        self.commission_per_contract = 0.50  # Estimated commission

        # Track positions
        self.current_positions = {}

        # Schedule daily execution
        self.Schedule.On(self.DateRules.EveryDay(), self.TimeRules.AfterMarketOpen("SPY", 30), self.ExecuteArbitrage)

    def ExecuteArbitrage(self):
        """Main arbitrage execution logic"""
        try:
            # Get current prices
            spx_price = self.Securities[self.spx].Price
            es_price = self.Securities[self.es].Price

            # Get risk-free rate (simplified - in practice use actual T-bill data)
            self.rf_rate = self.GetRiskFreeRate()

            # Get options chain for nearest expiration
            options = self.OptionChain(self.spx)
            if not options:
                self.Log("No options data available")
                return

            # Find nearest expiration
            expirations = sorted(set([opt.Expiry for opt in options]))
            if not expirations:
                return
            nearest_exp = expirations[0]

            # Time to expiration
            T = (nearest_exp - self.Time).days / 365.0
            if T <= 0:
                return

            # Find at-the-money strike
            atm_strike = min(options, key=lambda x: abs(x.Strike - spx_price)).Strike

            # Get call and put at this strike
            call = next((opt for opt in options if opt.Strike == atm_strike and opt.Right == OptionRight.Call), None)
            put = next((opt for opt in options if opt.Strike == atm_strike and opt.Right == OptionRight.Put), None)

            if not call or not put:
                self.Log(f"Missing options for strike {atm_strike}")
                return

            C = call.AskPrice if call.AskPrice > 0 else call.LastPrice
            P = put.AskPrice if put.AskPrice > 0 else put.LastPrice
            K = atm_strike

            if C <= 0 or P <= 0:
                return

            # Calculate theoretical futures price
            PV_K = K * np.exp(-self.rf_rate * T)
            F_theoretical = C - P + PV_K

            # Check parity
            diff = es_price - F_theoretical
            parity_check = abs(diff)

            self.Log(f"SPX: {spx_price:.2f}, ES: {es_price:.2f}, F_theoretical: {F_theoretical:.2f}, Diff: {diff:.4f}")

            # Execute arbitrage if mispricing exceeds threshold
            if parity_check > self.arbitrage_threshold:
                self.ExecuteTrade(es_price, F_theoretical, diff, C, P, K, T)
            else:
                self.Log("No arbitrage opportunity")

        except Exception as e:
            self.Log(f"Error in ExecuteArbitrage: {str(e)}")

    def ExecuteTrade(self, F_actual, F_theoretical, diff, C, P, K, T):
        """Execute the arbitrage trade"""
        try:
            if diff > 0:  # Futures overpriced
                self.Log("Futures overpriced: Sell futures, buy synthetic long")
                # Sell actual futures, buy synthetic (long call, short put)
                self.SellFutures()
                self.BuySyntheticLong(C, P, K)
            else:  # Futures underpriced
                self.Log("Futures underpriced: Buy futures, sell synthetic long")
                # Buy actual futures, sell synthetic (short call, long put)
                self.BuyFutures()
                self.SellSyntheticLong(C, P, K)

        except Exception as e:
            self.Log(f"Error executing trade: {str(e)}")

    def SellFutures(self):
        """Sell ES futures"""
        quantity = min(self.max_position_size, self.Portfolio[self.es].Quantity)
        if quantity > 0:
            self.MarketOrder(self.es, -quantity)
            self.Log(f"Sold {quantity} ES futures")

    def BuyFutures(self):
        """Buy ES futures"""
        quantity = self.max_position_size
        self.MarketOrder(self.es, quantity)
        self.Log(f"Bought {quantity} ES futures")

    def BuySyntheticLong(self, C, P, K):
        """Buy synthetic long futures: Buy call, sell put"""
        # In practice, need to check liquidity and position limits
        call_symbol = self.GetCallOption(K)
        put_symbol = self.GetPutOption(K)

        if call_symbol and put_symbol:
            # Buy call
            self.MarketOrder(call_symbol, 1)
            # Sell put
            self.MarketOrder(put_symbol, -1)
            self.Log("Executed synthetic long: Bought call, sold put")

    def SellSyntheticLong(self, C, P, K):
        """Sell synthetic long futures: Sell call, buy put"""
        call_symbol = self.GetCallOption(K)
        put_symbol = self.GetPutOption(K)

        if call_symbol and put_symbol:
            # Sell call
            self.MarketOrder(call_symbol, -1)
            # Buy put
            self.MarketOrder(put_symbol, 1)
            self.Log("Executed synthetic short: Sold call, bought put")

    def GetCallOption(self, strike):
        """Get call option symbol for given strike"""
        # Simplified - in practice need to find exact option contract
        options = self.OptionChain(self.spx)
        call = next((opt for opt in options if opt.Strike == strike and opt.Right == OptionRight.Call), None)
        return call.Symbol if call else None

    def GetPutOption(self, strike):
        """Get put option symbol for given strike"""
        options = self.OptionChain(self.spx)
        put = next((opt for opt in options if opt.Strike == strike and opt.Right == OptionRight.Put), None)
        return put.Symbol if put else None

    def GetRiskFreeRate(self):
        """Get current risk-free rate (simplified)"""
        # In practice, use Treasury bill data
        return 0.05  # 5% annual rate

    def OnOrderEvent(self, orderEvent):
        """Handle order events"""
        if orderEvent.Status == OrderStatus.Filled:
            self.Log(f"Order filled: {orderEvent.Symbol} {orderEvent.FillQuantity} @ {orderEvent.FillPrice}")

    def OnEndOfDay(self):
        """Daily cleanup and reporting"""
        # Close any open arbitrage positions if needed
        # Calculate daily P&L
        pass