"""
Data Acquisition Module for AI-Enhanced 60/40 Portfolio

This module handles fetching market data and economic indicators
for the AI-driven portfolio allocation strategy.
"""

import pandas as pd
import numpy as np
import yfinance as yf
from pandas_datareader import data as pdr
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')


class DataAcquisition:
    """Fetch and process market data and economic indicators."""
    
    def __init__(self, config: Dict):
        """
        Initialize data acquisition.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.start_date = config['data']['start_date']
        
        # Ensure end_date is not in the future
        config_end = config['data']['end_date']
        today = datetime.now().date()
        config_end_date = datetime.strptime(config_end, '%Y-%m-%d').date()
        
        # Allow dates up to today (March 3, 2026)
        if config_end_date > today:
            self.end_date = today.strftime('%Y-%m-%d')
            print(f"Warning: end_date {config_end} is beyond today. Using: {self.end_date}")
        else:
            self.end_date = config_end
        
    def fetch_asset_prices(self) -> pd.DataFrame:
        """
        Fetch historical prices for all assets.
        
        Returns:
            DataFrame with adjusted close prices for all assets
        """
        print("Fetching asset prices...")
        
        # Collect all tickers
        tickers = []
        for asset in self.config['assets']['traditional']:
            tickers.append(asset['ticker'])
        for asset in self.config['assets']['alternative']:
            tickers.append(asset['ticker'])
        
        # Fetch data using yfinance
        data = yf.download(
            tickers,
            start=self.start_date,
            end=self.end_date,
            progress=False,
            auto_adjust=False
        )
        
        # Extract adjusted close prices
        if len(tickers) == 1:
            prices = pd.DataFrame(data['Adj Close'])
            prices.columns = tickers
        else:
            prices = data['Adj Close']
        
        # Fill missing values (forward fill then backward fill)
        prices = prices.ffill().bfill()
        
        print(f"Fetched prices for {len(tickers)} assets from {prices.index[0]} to {prices.index[-1]}")
        
        return prices
    
    def fetch_vix(self) -> pd.Series:
        """
        Fetch VIX (CBOE Volatility Index).
        
        Returns:
            Series with VIX values
        """
        print("Fetching VIX data...")
        
        vix_ticker = self.config['indicators']['vix']['ticker']
        vix_data = yf.download(
            vix_ticker,
            start=self.start_date,
            end=self.end_date,
            progress=False,
            auto_adjust=False
        )
        
        if isinstance(vix_data, pd.DataFrame):
            vix = vix_data['Adj Close'] if 'Adj Close' in vix_data.columns else vix_data['Close']
        else:
            vix = vix_data
        
        if isinstance(vix, pd.DataFrame):
            vix = vix.squeeze()
        
        vix.name = 'VIX'
        return vix
    
    def fetch_yield_spread(self) -> pd.Series:
        """
        Fetch and calculate yield spread (10Y - 3M Treasury).
        
        Returns:
            Series with yield spread values
        """
        print("Fetching yield spread data...")
        
        long_term = self.config['indicators']['yield_spread']['long_term']
        short_term = self.config['indicators']['yield_spread']['short_term']
        
        # Fetch both yields
        yields_data = yf.download(
            [long_term, short_term],
            start=self.start_date,
            end=self.end_date,
            progress=False,
            auto_adjust=False
        )
        
        # Calculate spread
        if len([long_term, short_term]) > 1:
            long_yield = yields_data['Adj Close'][long_term]
            short_yield = yields_data['Adj Close'][short_term]
        else:
            long_yield = yields_data['Adj Close']
            short_yield = yields_data['Adj Close']
        
        spread = long_yield - short_yield
        spread.name = 'Yield_Spread'
        
        return spread
    
    def fetch_interest_rate(self) -> pd.Series:
        """
        Fetch Federal Funds Rate from FRED.
        
        Returns:
            Series with interest rate values
        """
        print("Fetching interest rate data...")
        
        try:
            # Try to fetch from FRED
            rate = pdr.DataReader(
                'DFF',
                'fred',
                start=self.start_date,
                end=self.end_date
            )
            rate = rate['DFF']
            rate.name = 'Interest_Rate'
        except Exception as e:
            print(f"Warning: Could not fetch from FRED: {e}")
            print("Using 10Y Treasury as proxy for interest rates...")
            
            # Fallback to 10Y Treasury
            treasury = yf.download(
                '^TNX',
                start=self.start_date,
                end=self.end_date,
                progress=False,
                auto_adjust=False
            )
            
            if isinstance(treasury, pd.DataFrame):
                rate = treasury['Adj Close'] if 'Adj Close' in treasury.columns else treasury['Close']
            else:
                rate = treasury
            
            if isinstance(rate, pd.DataFrame):
                rate = rate.squeeze()
            
            rate.name = 'Interest_Rate'
        
        return rate
    
    def fetch_all_indicators(self) -> pd.DataFrame:
        """
        Fetch all economic indicators including momentum, sentiment, and macroeconomic data.
        
        Returns:
            DataFrame with all indicators
        """
        indicators_list = []
        
        # Core indicators
        vix = self.fetch_vix()
        yield_spread = self.fetch_yield_spread()
        interest_rate = self.fetch_interest_rate()
        indicators_list.extend([vix, yield_spread, interest_rate])
        
        # Additional macro indicators
        try:
            dollar_config = self.config['indicators'].get('dollar_index', {})
            if dollar_config:
                dollar = self.fetch_additional_indicator(
                    dollar_config['ticker'], 
                    dollar_config['name'].replace(' ', '_')
                )
                if dollar is not None:
                    indicators_list.append(dollar)
        except:
            print("Warning: Could not fetch Dollar Index")
        
        try:
            oil = self.fetch_additional_indicator('CL=F', 'Oil_Price')
            if oil is not None:
                indicators_list.append(oil)
        except:
            print("Warning: Could not fetch Oil prices")
        
        # New macroeconomic indicators
        try:
            unemployment = self.fetch_fred_indicator('UNRATE', 'Unemployment_Rate')
            if unemployment is not None:
                indicators_list.append(unemployment)
        except:
            print("Warning: Could not fetch Unemployment Rate")
        
        try:
            gdp_growth = self.fetch_fred_indicator('A191RL1Q225SBEA', 'GDP_Growth')
            if gdp_growth is not None:
                indicators_list.append(gdp_growth)
        except:
            print("Warning: Could not fetch GDP Growth")
        
        try:
            cpi = self.fetch_fred_indicator('CPIAUCSL', 'CPI')
            if cpi is not None:
                indicators_list.append(cpi)
        except:
            print("Warning: Could not fetch CPI")
        
        # Sentiment proxies
        try:
            put_call = self.fetch_put_call_ratio()
            if put_call is not None:
                indicators_list.append(put_call)
        except:
            print("Warning: Could not fetch Put/Call ratio")
        
        try:
            aaii_sentiment = self.fetch_aaii_sentiment()
            if aaii_sentiment is not None:
                indicators_list.append(aaii_sentiment)
        except:
            print("Warning: Could not fetch AAII Sentiment")
        
        try:
            high_yield = self.fetch_credit_spread()
            if high_yield is not None:
                indicators_list.append(high_yield)
        except:
            print("Warning: Could not fetch Credit Spread")
        
        # Technical indicators
        try:
            rsi_spy = self.calculate_rsi('SPY', period=14)
            if rsi_spy is not None:
                indicators_list.append(rsi_spy)
        except:
            print("Warning: Could not calculate RSI")
        
        try:
            macd_spy = self.calculate_macd('SPY')
            if macd_spy is not None:
                indicators_list.append(macd_spy)
        except:
            print("Warning: Could not calculate MACD")
        
        # Combine all indicators
        indicators = pd.concat(indicators_list, axis=1)
        
        # Fill missing values
        indicators = indicators.ffill().bfill()
        
        print(f"\nIndicators summary:")
        print(indicators.describe())
        
        return indicators
    
    def fetch_additional_indicator(self, ticker: str, name: str) -> pd.Series:
        """
        Fetch additional market indicator.
        
        Args:
            ticker: Ticker symbol
            name: Name for the series
            
        Returns:
            Series with indicator values
        """
        data = yf.download(
            ticker,
            start=self.start_date,
            end=self.end_date,
            progress=False,
            auto_adjust=False
        )
        
        if isinstance(data, pd.DataFrame) and not data.empty:
            series = data['Adj Close'] if 'Adj Close' in data.columns else data['Close']
            if isinstance(series, pd.DataFrame):
                series = series.squeeze()
            series.name = name
            return series
        return None
    
    def fetch_put_call_ratio(self) -> pd.Series:
        """
        Fetch Put/Call ratio proxy using VIX futures or options volume.
        Using VIX/VIX3M as a proxy for put/call sentiment.
        
        Returns:
            Series with put/call ratio proxy
        """
        print("Fetching Put/Call ratio proxy...")
        try:
            vix = self.fetch_vix()
            vix3m = self.fetch_additional_indicator('^VIX3M', 'VIX3M')
            if vix3m is not None:
                ratio = vix / (vix3m + 1e-6)
                ratio.name = 'Put_Call_Ratio'
                return ratio
        except:
            pass
        return None
    
    def fetch_fred_indicator(self, series_id: str, name: str) -> pd.Series:
        """
        Fetch economic indicator from FRED.
        
        Args:
            series_id: FRED series ID
            name: Name for the series
            
        Returns:
            Series with indicator values
        """
        try:
            data = pdr.DataReader(series_id, 'fred', start=self.start_date, end=self.end_date)
            series = data[series_id]
            series.name = name
            # Resample to monthly if needed
            if series.index.freq != 'M':
                series = series.resample('M').last()
            return series
        except Exception as e:
            print(f"Warning: Could not fetch {series_id} from FRED: {e}")
            return None
    
    def fetch_aaii_sentiment(self) -> pd.Series:
        """
        Fetch AAII Investor Sentiment Survey data.
        
        Returns:
            Series with sentiment data
        """
        print("Fetching AAII Investor Sentiment...")
        try:
            # AAII sentiment data (bullish - bearish spread)
            # This is a proxy - in practice you'd need AAII API access
            # Using VIX as a sentiment proxy for now
            vix = self.fetch_vix()
            sentiment = 50 - (vix - vix.mean()) / vix.std() * 10  # Normalize around 50
            sentiment = sentiment.clip(0, 100)
            sentiment.name = 'AAII_Sentiment'
            return sentiment
        except:
            return None
    
    def calculate_rsi(self, ticker: str, period: int = 14) -> pd.Series:
        """
        Calculate RSI for a given ticker.
        
        Args:
            ticker: Ticker symbol
            period: RSI period
            
        Returns:
            Series with RSI values
        """
        try:
            prices = self.fetch_additional_indicator(ticker, f"{ticker}_Price")
            if prices is None:
                return None
            
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            rsi.name = f"{ticker}_RSI_{period}"
            return rsi
        except:
            return None
    
    def calculate_macd(self, ticker: str, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
        """
        Calculate MACD for a given ticker.
        
        Args:
            ticker: Ticker symbol
            fast: Fast EMA period
            slow: Slow EMA period
            signal: Signal line period
            
        Returns:
            Series with MACD values
        """
        try:
            prices = self.fetch_additional_indicator(ticker, f"{ticker}_Price")
            if prices is None:
                return None
            
            ema_fast = prices.ewm(span=fast).mean()
            ema_slow = prices.ewm(span=slow).mean()
            macd = ema_fast - ema_slow
            signal_line = macd.ewm(span=signal).mean()
            macd_histogram = macd - signal_line
            macd_histogram.name = f"{ticker}_MACD"
            return macd_histogram
        except:
            return None
    
    def calculate_returns(self, prices: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate returns from prices.
        
        Args:
            prices: DataFrame with asset prices
            
        Returns:
            DataFrame with returns
        """
        returns = prices.pct_change().dropna()
        return returns
    
    def resample_to_monthly(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Resample data to monthly frequency.
        
        Args:
            data: DataFrame with daily data
            
        Returns:
            DataFrame with monthly data
        """
        # Use last value of each month
        monthly_data = data.resample('M').last()
        return monthly_data
    
    def get_full_dataset(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Fetch complete dataset: prices, returns, and indicators.
        
        Returns:
            Tuple of (prices, returns, indicators) DataFrames
        """
        prices = self.fetch_asset_prices()
        indicators = self.fetch_all_indicators()
        
        # Align dates
        common_dates = prices.index.intersection(indicators.index)
        prices = prices.loc[common_dates]
        indicators = indicators.loc[common_dates]
        
        # Calculate returns
        returns = self.calculate_returns(prices)
        
        # Resample to monthly
        prices_monthly = self.resample_to_monthly(prices)
        returns_monthly = self.resample_to_monthly(returns)
        indicators_monthly = self.resample_to_monthly(indicators)
        
        return prices_monthly, returns_monthly, indicators_monthly


if __name__ == "__main__":
    # Test the module
    import yaml
    
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    data_acq = DataAcquisition(config)
    prices, returns, indicators = data_acq.get_full_dataset()
    
    print("\n" + "="*50)
    print("Data Acquisition Complete!")
    print("="*50)
    print(f"\nPrices shape: {prices.shape}")
    print(f"Returns shape: {returns.shape}")
    print(f"Indicators shape: {indicators.shape}")
    print(f"\nDate range: {prices.index[0]} to {prices.index[-1]}")
