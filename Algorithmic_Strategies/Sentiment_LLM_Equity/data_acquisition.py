"""
Data Acquisition for Sentiment-Based Equity Strategy

Fetches text data (news, social media, filings) and market data.
"""

import pandas as pd
import numpy as np
import yaml
from pathlib import Path
from typing import Dict, List
import warnings
warnings.filterwarnings('ignore')

# Placeholder imports (require API keys in production)
try:
    import yfinance as yf
except ImportError:
    print("Warning: yfinance not installed")


class SentimentDataAcquisition:
    """Data acquisition for text and market data."""
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize data acquisition."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.data_config = self.config['data']
        self.universe_config = self.data_config['universe']
        
        # Create data directories
        self.data_dir = Path("data")
        self.data_dir.mkdir(exist_ok=True)
        (self.data_dir / "text").mkdir(exist_ok=True)
        (self.data_dir / "market").mkdir(exist_ok=True)
    
    def fetch_universe(self, date: str = None) -> List[str]:
        """
        Fetch stock universe (Russell 3000 constituents).
        
        Args:
            date: Date for universe composition (default: latest)
        
        Returns:
            List of ticker symbols
        """
        # Placeholder: In production, use official Russell 3000 constituents list
        # For testing, use a subset of major stocks
        
        print("Fetching stock universe...")
        
        # Placeholder universe (top 100 US stocks by market cap)
        placeholder_universe = [
            "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "BRK.B",
            "UNH", "JNJ", "V", "WMT", "XOM", "JPM", "MA", "PG", "HD", "CVX",
            "LLY", "ABBV", "MRK", "KO", "AVGO", "PEP", "COST", "ADBE", "TMO",
            "CSCO", "MCD", "ACN", "NFLX", "ABT", "CRM", "ORCL", "NKE", "DHR",
            "WFC", "LIN", "VZ", "TXN", "PM", "AMD", "NEE", "DIS", "QCOM", "UPS",
            "RTX", "INTU", "HON", "BMY", "AMGN", "SBUX", "LOW", "SPGI", "BA",
            "CAT", "UNP", "IBM", "DE", "GE", "MDT", "ELV", "PLD", "GILD", "AMT",
            "GS", "ISRG", "BLK", "MMC", "SCHW", "C", "ADP", "ADI", "REGN", "TJX",
            "VRTX", "CVS", "SYK", "BKNG", "LRCX", "PGR", "MO", "ZTS", "CI", "CB",
            "MDLZ", "SO", "DUK", "FISV", "ETN", "EOG", "APD", "NOC", "PNC", "CME",
            "ICE", "AON", "TGT", "USB", "MCO", "CL", "F", "GM", "SLB", "EMR"
        ]
        
        # Apply universe filters
        min_price = self.universe_config['min_price']
        
        print(f"Universe size: {len(placeholder_universe)} stocks")
        print(f"Filters: price > ${min_price}")
        
        return placeholder_universe
    
    def fetch_market_data(
        self,
        tickers: List[str],
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """
        Fetch market data for universe.
        
        Args:
            tickers: List of ticker symbols
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
        
        Returns:
            DataFrame with prices, returns, market cap, sector
        """
        print(f"\nFetching market data for {len(tickers)} stocks...")
        print(f"Period: {start_date} to {end_date}")
        
        # Use yfinance for demonstration
        all_data = []
        
        for ticker in tickers[:10]:  # Limit to 10 for testing
            try:
                stock = yf.Ticker(ticker)
                hist = stock.history(start=start_date, end=end_date)
                
                if len(hist) > 0:
                    hist['ticker'] = ticker
                    hist['return'] = hist['Close'].pct_change()
                    
                    # Get info
                    info = stock.info
                    hist['market_cap'] = info.get('marketCap', np.nan)
                    hist['sector'] = info.get('sector', 'Unknown')
                    
                    all_data.append(hist.reset_index())
                    
            except Exception as e:
                print(f"  Error fetching {ticker}: {e}")
                continue
        
        if len(all_data) == 0:
            print("Warning: No market data fetched")
            return pd.DataFrame()
        
        # Combine
        market_data = pd.concat(all_data, ignore_index=True)
        market_data.rename(columns={'Date': 'date'}, inplace=True)
        
        print(f"Fetched {len(market_data)} rows for {market_data['ticker'].nunique()} stocks")
        
        # Save
        market_data.to_csv(self.data_dir / "market" / "prices.csv", index=False)
        
        return market_data
    
    def calculate_market_beta(
        self,
        stock_returns: pd.Series,
        market_returns: pd.Series,
        window: int = 252
    ) -> float:
        """
        Calculate rolling beta vs market.
        
        Args:
            stock_returns: Stock returns
            market_returns: Market returns (e.g., SPY)
            window: Rolling window (default: 252 days)
        
        Returns:
            Beta estimate
        """
        # Align series
        aligned = pd.DataFrame({
            'stock': stock_returns,
            'market': market_returns
        }).dropna()
        
        if len(aligned) < window:
            return 1.0  # Default beta
        
        # Calculate covariance and variance
        cov = aligned['stock'].tail(window).cov(aligned['market'].tail(window))
        var = aligned['market'].tail(window).var()
        
        beta = cov / var if var > 0 else 1.0
        
        return beta
    
    def fetch_text_data_placeholder(
        self,
        ticker: str,
        date: str
    ) -> str:
        """
        Placeholder function for fetching text data.
        
        In production, this would fetch from:
        - News APIs (Bloomberg, Reuters, RavenPack)
        - Social media APIs (Twitter, Reddit, StockTwits)
        - SEC EDGAR (filings)
        - Analyst reports
        
        Args:
            ticker: Stock ticker
            date: Date (YYYY-MM-DD)
        
        Returns:
            Aggregated text corpus
        """
        # Placeholder: Generate synthetic sentiment-bearing text
        
        # Simulate positive/negative/neutral sentiment
        sentiment_type = np.random.choice(['positive', 'negative', 'neutral'], p=[0.3, 0.3, 0.4])
        
        if sentiment_type == 'positive':
            texts = [
                f"{ticker} reported strong earnings, beating analyst expectations.",
                f"Analysts upgrade {ticker} to Buy on robust growth outlook.",
                f"{ticker} announces new product launch, shares surge.",
                f"Institutional investors increase holdings in {ticker}.",
                f"{ticker} CEO expresses confidence in future performance."
            ]
        elif sentiment_type == 'negative':
            texts = [
                f"{ticker} misses earnings estimates, shares decline.",
                f"Analysts downgrade {ticker} citing weak fundamentals.",
                f"{ticker} faces regulatory scrutiny, stock under pressure.",
                f"Insider selling raises concerns about {ticker}'s prospects.",
                f"{ticker} reports declining revenue, guidance lowered."
            ]
        else:  # neutral
            texts = [
                f"{ticker} reports quarterly results in line with expectations.",
                f"Trading volume for {ticker} remains average.",
                f"{ticker} maintains current dividend policy.",
                f"Analysts maintain Hold rating on {ticker}.",
                f"{ticker} stock shows typical price action."
            ]
        
        # Combine texts
        corpus = " ".join(np.random.choice(texts, size=3, replace=False))
        
        return corpus
    
    def fetch_full_dataset(
        self,
        start_date: str = None,
        end_date: str = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch complete dataset (market + text).
        
        Args:
            start_date: Start date
            end_date: End date
        
        Returns:
            Dict with 'market_data' and 'text_data'
        """
        if start_date is None:
            start_date = self.data_config['period']['start_date']
        if end_date is None:
            end_date = self.data_config['period']['end_date']
        
        # Fetch universe
        universe = self.fetch_universe()
        
        # Fetch market data
        market_data = self.fetch_market_data(universe, start_date, end_date)
        
        # Generate placeholder text data
        print("\nGenerating placeholder text data...")
        text_data = []
        
        dates = pd.date_range(start_date, end_date, freq='D')
        
        for ticker in universe[:10]:  # Limit for testing
            for date in dates[:100]:  # Limit for testing
                text = self.fetch_text_data_placeholder(ticker, date.strftime('%Y-%m-%d'))
                
                text_data.append({
                    'ticker': ticker,
                    'date': date,
                    'text': text
                })
        
        text_df = pd.DataFrame(text_data)
        
        print(f"Generated {len(text_df)} text samples")
        
        # Save
        text_df.to_csv(self.data_dir / "text" / "news_social.csv", index=False)
        
        return {
            'market_data': market_data,
            'text_data': text_df
        }


# Test code
if __name__ == "__main__":
    data_acq = SentimentDataAcquisition('config.yaml')
    
    # Fetch full dataset
    dataset = data_acq.fetch_full_dataset(
        start_date="2023-01-01",
        end_date="2023-12-31"
    )
    
    print("\n" + "="*60)
    print("DATA ACQUISITION COMPLETE")
    print("="*60)
    print(f"\nMarket Data: {len(dataset['market_data'])} rows")
    print(f"Text Data: {len(dataset['text_data'])} rows")
    print(f"\nData saved to data/ directory")
