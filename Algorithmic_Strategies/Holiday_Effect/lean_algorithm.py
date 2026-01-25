"""
QuantConnect Lean Algorithm for Holiday Effect Strategy

Trades AMZN long during 10-day windows before Black Friday and Prime Day.
"""

from AlgorithmImports import *
from datetime import datetime, timedelta
from calendar import monthrange


class HolidayEffectAlgorithm(QCAlgorithm):
    """Holiday Effect strategy on Amazon stock."""
    
    def Initialize(self):
        """Initialize algorithm parameters."""
        # Set dates and capital
        self.SetStartDate(1998, 1, 1)
        self.SetEndDate(2024, 12, 31)
        self.SetCash(1000000)
        
        # Add securities
        self.amzn = self.AddEquity("AMZN", Resolution.Daily)
        self.spy = self.AddEquity("SPY", Resolution.Daily)
        
        # Set benchmark
        self.SetBenchmark("SPY")
        
        # Strategy parameters
        self.lookback_days = 10  # 10 trading days before event
        self.stop_loss_pct = 0.08  # 8% stop-loss
        self.vix_threshold = 25
        
        # State variables
        self.in_position = False
        self.entry_price = 0
        self.current_event = None
        self.next_entry_date = None
        self.next_exit_date = None
        
        # Track events for current year
        self.UpdateEventDates()
        
        # Schedule rebalancing before market open
        self.Schedule.On(
            self.DateRules.EveryDay("AMZN"),
            self.TimeRules.At(9, 30),
            self.Rebalance
        )
        
        # Year-end event update
        self.Schedule.On(
            self.DateRules.MonthEnd("AMZN"),
            self.TimeRules.At(16, 0),
            self.CheckYearEnd
        )
        
    def UpdateEventDates(self):
        """Calculate Black Friday and Prime Day for current year."""
        year = self.Time.year
        
        # Black Friday: Friday after 4th Thursday in November
        black_friday = self.GetBlackFriday(year)
        
        # Prime Day: Historical dates (simplified - use mid-July)
        prime_day = self.GetPrimeDay(year)
        
        # Store events
        self.events = []
        
        if black_friday:
            self.events.append({
                'name': 'Black Friday',
                'date': black_friday
            })
        
        if prime_day:
            self.events.append({
                'name': 'Prime Day',
                'date': prime_day
            })
        
        self.Log(f"Events for {year}: {len(self.events)} events scheduled")
        
    def GetBlackFriday(self, year):
        """Calculate Black Friday date for given year."""
        # Find first day of November
        nov_first = datetime(year, 11, 1)
        
        # Find first Thursday
        days_until_thursday = (3 - nov_first.weekday()) % 7
        first_thursday = nov_first + timedelta(days=days_until_thursday)
        
        # Fourth Thursday (Thanksgiving)
        thanksgiving = first_thursday + timedelta(weeks=3)
        
        # Black Friday (next day)
        black_friday = thanksgiving + timedelta(days=1)
        
        return black_friday
    
    def GetPrimeDay(self, year):
        """Get Prime Day date for given year."""
        # Prime Day started in 2015
        if year < 2015:
            return None
        
        # Historical Prime Day dates (mid-July default)
        prime_day_dates = {
            2015: datetime(2015, 7, 15),
            2016: datetime(2016, 7, 12),
            2017: datetime(2017, 7, 11),
            2018: datetime(2018, 7, 16),
            2019: datetime(2019, 7, 15),
            2020: datetime(2020, 10, 13),  # COVID delay
            2021: datetime(2021, 6, 21),
            2022: datetime(2022, 7, 12),
            2023: datetime(2023, 7, 11),
            2024: datetime(2024, 7, 16)
        }
        
        return prime_day_dates.get(year, datetime(year, 7, 15))
    
    def CheckYearEnd(self):
        """Update events at year-end."""
        if self.Time.month == 12:
            self.UpdateEventDates()
    
    def Rebalance(self):
        """Daily rebalancing logic."""
        # Skip if no data
        if not self.amzn.HasData or not self.spy.HasData:
            return
        
        current_date = self.Time
        
        # Check if we need to calculate new entry/exit dates
        if self.next_entry_date is None:
            self.CalculateNextWindow()
        
        # ENTRY LOGIC
        if not self.in_position and self.next_entry_date:
            if current_date.date() >= self.next_entry_date.date():
                # Check market filters
                if self.CheckMarketFilters():
                    # Enter position
                    self.SetHoldings("AMZN", 1.0)
                    self.entry_price = self.Securities["AMZN"].Price
                    self.in_position = True
                    self.Log(f"ENTRY: {current_date} - {self.current_event} at ${self.entry_price:.2f}")
                else:
                    self.Log(f"SKIPPED: Market filters failed for {self.current_event}")
                    # Skip this event
                    self.CalculateNextWindow()
        
        # EXIT LOGIC
        if self.in_position:
            current_price = self.Securities["AMZN"].Price
            
            # Stop-loss check
            if current_price <= self.entry_price * (1 - self.stop_loss_pct):
                self.Liquidate("AMZN")
                self.in_position = False
                self.Log(f"STOP-LOSS: {current_date} at ${current_price:.2f} (-{self.stop_loss_pct*100:.1f}%)")
                self.CalculateNextWindow()
                return
            
            # Normal exit on exit date
            if self.next_exit_date and current_date.date() >= self.next_exit_date.date():
                self.Liquidate("AMZN")
                pnl = (current_price - self.entry_price) / self.entry_price * 100
                self.in_position = False
                self.Log(f"EXIT: {current_date} at ${current_price:.2f} (+{pnl:.2f}%)")
                self.CalculateNextWindow()
    
    def CalculateNextWindow(self):
        """Calculate next event window dates."""
        current_date = self.Time
        
        # Find next upcoming event
        upcoming_events = [
            event for event in self.events
            if event['date'].date() > current_date.date()
        ]
        
        if not upcoming_events:
            # No more events this year, update for next year
            self.next_entry_date = None
            self.next_exit_date = None
            self.current_event = None
            return
        
        # Get nearest event
        next_event = min(upcoming_events, key=lambda x: x['date'])
        
        self.current_event = next_event['name']
        event_date = next_event['date']
        
        # Calculate entry date (10 trading days before)
        # Approximate: 10 trading days ≈ 14 calendar days
        self.next_entry_date = event_date - timedelta(days=14)
        
        # Calculate exit date (day before event)
        self.next_exit_date = event_date - timedelta(days=1)
        
        self.Log(f"Next event: {self.current_event} on {event_date.strftime('%Y-%m-%d')}")
        self.Log(f"  Entry: {self.next_entry_date.strftime('%Y-%m-%d')}")
        self.Log(f"  Exit: {self.next_exit_date.strftime('%Y-%m-%d')}")
    
    def CheckMarketFilters(self):
        """Check if market conditions are favorable."""
        # SPY 200-day MA filter
        spy_history = self.History("SPY", 200, Resolution.Daily)
        
        if spy_history.empty:
            return True  # Default to allowing trade if no data
        
        spy_ma200 = spy_history['close'].mean()
        spy_current = self.Securities["SPY"].Price
        
        if spy_current <= spy_ma200:
            self.Log(f"Market filter: SPY below 200MA (${spy_current:.2f} vs ${spy_ma200:.2f})")
            return False
        
        # VIX filter (if available - simplified)
        # In practice, would add VIX as security
        
        return True
    
    def OnEndOfAlgorithm(self):
        """Log final statistics."""
        self.Log(f"=== BACKTEST COMPLETE ===")
        self.Log(f"Final Portfolio Value: ${self.Portfolio.TotalPortfolioValue:,.2f}")
        
        # Calculate returns
        initial_value = 1000000
        final_value = self.Portfolio.TotalPortfolioValue
        total_return = (final_value - initial_value) / initial_value * 100
        
        self.Log(f"Total Return: {total_return:.2f}%")
