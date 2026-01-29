from AlgorithmImports import *
from datetime import timedelta

class HolidayEffectAlgorithm(QCAlgorithm):
    
    def Initialize(self):
        self.SetStartDate(1998, 1, 1)
        self.SetEndDate(2024, 12, 31)
        self.SetCash(1000000)
        
        self.amzn = self.AddEquity("AMZN", Resolution.Daily)
        self.spy = self.AddEquity("SPY", Resolution.Daily)
        self.SetBenchmark("SPY")
        
        self.lookback_days = 10
        self.in_position = False
        self.entry_date = None
        self.exit_date = None
        
        self.Schedule.On(self.DateRules.EveryDay("AMZN"), 
                        self.TimeRules.AfterMarketOpen("AMZN", 1), 
                        self.CheckEntry)
        
    def CheckEntry(self):
        if not self.amzn.HasData or not self.spy.HasData:
            return
            
        current_date = self.Time
        
        # Calculate next event dates
        if self.entry_date is None or current_date > self.exit_date:
            self.CalculateNextEvent()
        
        # Entry logic
        if not self.in_position and self.entry_date:
            if current_date.date() >= self.entry_date.date():
                if self.CheckFilters():
                    self.SetHoldings("AMZN", 1.0)
                    self.in_position = True
                    self.Log(f"ENTRY: {current_date} at ${self.amzn.Price:.2f}")
                else:
                    self.CalculateNextEvent()
        
        # Exit logic
        if self.in_position and self.exit_date:
            if current_date.date() >= self.exit_date.date():
                self.Liquidate("AMZN")
                self.in_position = False
                self.Log(f"EXIT: {current_date} at ${self.amzn.Price:.2f}")
                self.CalculateNextEvent()
    
    def CalculateNextEvent(self):
        year = self.Time.year
        
        # Black Friday
        bf_date = self.GetBlackFriday(year)
        bf_entry = bf_date - timedelta(days=14)
        bf_exit = bf_date - timedelta(days=1)
        
        # Prime Day
        pd_date = self.GetPrimeDay(year)
        
        if pd_date:
            pd_entry = pd_date - timedelta(days=14)
            pd_exit = pd_date - timedelta(days=1)
            
            # Choose next upcoming event
            if self.Time < bf_entry:
                self.entry_date = bf_entry
                self.exit_date = bf_exit
            elif self.Time < pd_entry:
                self.entry_date = pd_entry
                self.exit_date = pd_exit
            else:
                self.entry_date = None
                self.exit_date = None
        else:
            if self.Time < bf_entry:
                self.entry_date = bf_entry
                self.exit_date = bf_exit
            else:
                self.entry_date = None
                self.exit_date = None
    
    def GetBlackFriday(self, year):
        nov_first = datetime(year, 11, 1)
        days_until_thursday = (3 - nov_first.weekday()) % 7
        first_thursday = nov_first + timedelta(days=days_until_thursday)
        thanksgiving = first_thursday + timedelta(weeks=3)
        return thanksgiving + timedelta(days=1)
    
    def GetPrimeDay(self, year):
        dates = {
            2015: datetime(2015, 7, 15), 2016: datetime(2016, 7, 12),
            2017: datetime(2017, 7, 11), 2018: datetime(2018, 7, 16),
            2019: datetime(2019, 7, 15), 2020: datetime(2020, 10, 13),
            2021: datetime(2021, 6, 21), 2022: datetime(2022, 7, 12),
            2023: datetime(2023, 7, 11), 2024: datetime(2024, 7, 16)
        }
        return dates.get(year, datetime(year, 7, 15) if year >= 2015 else None)
    
    def CheckFilters(self):
        spy_history = self.History(self.spy.Symbol, 200, Resolution.Daily)
        if spy_history.empty:
            return True
        spy_ma200 = spy_history['close'].mean()
        return self.spy.Price > spy_ma200
