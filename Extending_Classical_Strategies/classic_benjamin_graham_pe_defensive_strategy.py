import pandas as pd
import numpy as np
import statsmodels.api as sm
import yfinance as yf
import pandas_datareader.data as web
from datetime import datetime
from cvxpy import Variable, Minimize, sum_squares, Problem
import matplotlib.pyplot as plt
import time
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Configuration
START_DATE = '2000-01-01'
END_DATE = '2025-04-14'
INITIAL_CAPITAL = 10000
CPI_SERIES = 'CPIAUCSL'

def fetch_inflation_data():
    """Fetch and preprocess CPI data from FRED"""
    try:
        cpi = web.DataReader(CPI_SERIES, 'fred', START_DATE, END_DATE)
        # Calculate inflation rate instead of using X13
        cpi_clean = cpi[CPI_SERIES].dropna()
        inflation_rate = cpi_clean.pct_change(12).fillna(0)  # Year-over-year inflation
        return cpi_clean
    except Exception as e:
        print(f"Error fetching CPI data: {e}")
        # Return dummy data if FRED is unavailable
        dates = pd.date_range(START_DATE, END_DATE, freq='M')
        return pd.Series(index=dates, data=np.linspace(100, 150, len(dates)))

def adjust_for_inflation(series, cpi):
    """Adjust nominal values using CPI"""
    if cpi is None or len(cpi) == 0:
        return series
    
    # Align series with CPI data
    common_dates = series.index.intersection(cpi.index)
    if len(common_dates) == 0:
        return series
    
    base_cpi = cpi.loc[common_dates[0]]
    aligned_cpi = cpi.reindex(series.index, method='ffill')
    return series * (base_cpi / aligned_cpi)

class GrahamBacktester:
    def __init__(self):
        self.cpi = fetch_inflation_data()
        self.sp500 = self._get_sp500_constituents()
        self.data = self._fetch_fundamental_data()
        self._preprocess_data()

    def _get_sp500_constituents(self):
        """Get current S&P 500 constituents"""
        try:
            constituents = pd.read_html(
                'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
            )[0]['Symbol'].tolist()
            print(f"Successfully fetched {len(constituents)} S&P 500 constituents")
            return constituents  # Use all constituents instead of limiting to 50
        except Exception as e:
            print(f"Error fetching S&P 500 constituents: {e}")
            # Return a larger set of well-known stocks as fallback
            return [
                'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'JPM', 'V', 'JNJ',
                'UNH', 'HD', 'PG', 'MA', 'BAC', 'ABBV', 'PFE', 'KO', 'AVGO', 'PEP',
                'TMO', 'COST', 'MRK', 'ABT', 'ACN', 'NFLX', 'ADBE', 'CRM', 'LLY', 'CSCO',
                'XOM', 'NKE', 'DHR', 'VZ', 'QCOM', 'TXN', 'BMY', 'UNP', 'T', 'CMCSA',
                'NEE', 'LOW', 'CVX', 'ORCL', 'MDT', 'IBM', 'HON', 'SBUX', 'AMT', 'INTU'
            ]

    def _fetch_fundamental_data(self):
        """Fetch historical price data with robust error handling"""
        try:
            print(f"Fetching data for {len(self.sp500)} tickers...")
            
            # Configure session with retry strategy
            session = requests.Session()
            retry_strategy = Retry(
                total=3,
                backoff_factor=1,
                status_forcelist=[429, 500, 502, 503, 504],
                allowed_methods=["HEAD", "GET", "OPTIONS"]
            )
            adapter = HTTPAdapter(max_retries=retry_strategy)
            session.mount("http://", adapter)
            session.mount("https://", adapter)
            
            # Split into smaller chunks to avoid overwhelming the API
            chunk_size = 20  # Reduced chunk size
            all_data = []
            successful_tickers = []
            failed_tickers = []
            
            for i in range(0, len(self.sp500), chunk_size):
                chunk = self.sp500[i:i+chunk_size]
                chunk_num = i//chunk_size + 1
                total_chunks = (len(self.sp500)-1)//chunk_size + 1
                
                print(f"Fetching chunk {chunk_num}/{total_chunks}: {len(chunk)} tickers (Running total: {len(successful_tickers)})")
                
                # Retry mechanism for each chunk
                max_retries = 3
                chunk_successful = False
                
                for attempt in range(max_retries):
                    try:
                        # Add delay between requests
                        if attempt > 0:
                            time.sleep(2 ** attempt)  # Exponential backoff
                        
                        chunk_data = yf.download(
                            chunk,
                            start=START_DATE,
                            end=END_DATE,
                            threads=False,  # Disable threading to avoid connection issues
                            group_by='ticker',
                            auto_adjust=True,
                            prepost=False,
                            repair=True,
                            timeout=30
                        )
                        
                        if not chunk_data.empty:
                            all_data.append(chunk_data)
                            successful_tickers.extend(chunk)
                            chunk_successful = True
                            print(f"  ✓ Successfully fetched {len(chunk)} tickers (Total so far: {len(successful_tickers)})")
                            break
                        else:
                            print(f"  ✗ No data returned for chunk {chunk_num}")
                            
                    except Exception as e:
                        print(f"  ✗ Attempt {attempt + 1} failed for chunk {chunk_num}: {str(e)[:80]}...")
                        if attempt == max_retries - 1:
                            failed_tickers.extend(chunk)
                            print(f"  ✗ Failed to fetch chunk {chunk_num} after {max_retries} attempts")
                
                if not chunk_successful:
                    print(f"  ✗ Chunk {chunk_num} completely failed - added {len(chunk)} tickers to failed list")
                
                # Add delay between chunks
                time.sleep(1)
                
                # Progress update every 5 chunks
                if chunk_num % 5 == 0:
                    print(f"  Progress: {chunk_num}/{total_chunks} chunks completed, {len(successful_tickers)} tickers successful, {len(failed_tickers)} failed")
            
            print(f"\n=== DATA FETCHING SUMMARY ===")
            print(f"Total S&P 500 tickers attempted: {len(self.sp500)}")
            print(f"Successfully fetched: {len(successful_tickers)} tickers ({len(successful_tickers)/len(self.sp500)*100:.1f}%)")
            print(f"Failed to fetch: {len(failed_tickers)} tickers ({len(failed_tickers)/len(self.sp500)*100:.1f}%)")
            
            if failed_tickers:
                print(f"Failed tickers (first 20): {failed_tickers[:20]}")
                if len(failed_tickers) > 20:
                    print(f"... and {len(failed_tickers) - 20} more")
            
            if not all_data:
                print("No data fetched successfully")
                return pd.DataFrame()
            
            # Combine all chunks
            print(f"\nCombining {len(all_data)} data chunks...")
            data = pd.concat(all_data, axis=1)
            print(f"Combined data shape: {data.shape}")
            
            # Handle different data structures
            if len(successful_tickers) == 1:
                # Single ticker case - create MultiIndex
                ticker = successful_tickers[0]
                data = pd.concat({ticker: data}, axis=1)
                data = data.swaplevel(axis=1).sort_index(axis=1)
            else:
                # Multiple tickers - check if already in correct format
                if isinstance(data.columns, pd.MultiIndex):
                    if data.columns.names == [None, None]:
                        # Need to swap levels
                        data = data.swaplevel(axis=1).sort_index(axis=1)
                else:
                    # Single level columns, need to create MultiIndex
                    if len(successful_tickers) == 1:
                        data = pd.concat({successful_tickers[0]: data}, axis=1)
                        data = data.swaplevel(axis=1).sort_index(axis=1)
            
            # Check data availability and quality
            if isinstance(data.columns, pd.MultiIndex):
                available_tickers = data.columns.get_level_values(0).unique()
                print(f"Final result: Data available for {len(available_tickers)} unique tickers")
                
                # Check data quality
                if 'Close' in data.columns.get_level_values(1):
                    closes = data.xs('Close', axis=1, level=1)
                    data_quality = closes.notna().sum()
                    print(f"Data quality summary:")
                    print(f"  Average data points per ticker: {data_quality.mean():.0f}")
                    print(f"  Min data points: {data_quality.min()}")
                    print(f"  Max data points: {data_quality.max()}")
                    print(f"  Tickers with >1000 data points: {(data_quality > 1000).sum()}")
                    print(f"  Tickers with >5000 data points: {(data_quality > 5000).sum()}")
                    
                    # Filter out tickers with insufficient data
                    min_data_points = 252  # At least 1 year of data
                    good_tickers = data_quality[data_quality >= min_data_points].index
                    print(f"  Tickers with sufficient data (>{min_data_points} points): {len(good_tickers)}")
                    
                    if len(good_tickers) < len(available_tickers):
                        # Filter the data to include only good tickers
                        good_columns = []
                        for ticker in good_tickers:
                            for col_type in data.columns.get_level_values(1).unique():
                                if (ticker, col_type) in data.columns:
                                    good_columns.append((ticker, col_type))
                        
                        if good_columns:
                            data = data[good_columns]
                            print(f"  Filtered data to {len(good_tickers)} tickers with sufficient data")
                        else:
                            print("  No tickers with sufficient data found")
                            return pd.DataFrame()
                    
                    # Update successful tickers list to only include those with good data
                    self.successful_tickers = good_tickers.tolist()
                    print(f"Final tickers for analysis: {len(self.successful_tickers)}")
                else:
                    print("  No Close price data found")
                    return pd.DataFrame()
            else:
                print("  Data structure is not MultiIndex as expected")
                return pd.DataFrame()
            
            print(f"Final data processing completed successfully")
            return data
            
        except Exception as e:
            print(f"Critical error in data fetching: {e}")
            return pd.DataFrame()

    def _get_ticker_info(self, ticker):
        """Get current fundamental data for a ticker with error handling"""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                if attempt > 0:
                    time.sleep(1)  # Add delay between retries
                
                stock = yf.Ticker(ticker)
                info = stock.info
                
                # Validate that we got some data
                if info and isinstance(info, dict):
                    return {
                        'pe_ratio': info.get('trailingPE', np.nan),
                        'pb_ratio': info.get('priceToBook', np.nan),
                        'current_ratio': info.get('currentRatio', np.nan),
                        'market_cap': info.get('marketCap', np.nan),
                        'revenue': info.get('totalRevenue', np.nan)
                    }
                else:
                    if attempt == max_retries - 1:
                        print(f"    No fundamental data available for {ticker}")
                    
            except Exception as e:
                if attempt == max_retries - 1:
                    print(f"    Error fetching fundamental data for {ticker}: {str(e)[:50]}...")
                
        return {}

    def _preprocess_data(self):
        """Clean and normalize data with available information"""
        if self.data.empty:
            print("No data available for preprocessing")
            return
        
        print(f"Data preprocessing started...")
        print(f"Data shape: {self.data.shape}")
        
        # Get current fundamentals for screening (only for tickers that have price data)
        if hasattr(self, 'successful_tickers'):
            available_tickers = self.successful_tickers
        elif isinstance(self.data.columns, pd.MultiIndex):
            available_tickers = self.data.columns.get_level_values(0).unique().tolist()
        else:
            available_tickers = self.sp500
        
        print(f"Getting fundamental data for {len(available_tickers)} tickers...")
        self.fundamentals = {}
        
        # Process in batches to avoid overwhelming the API
        batch_size = 10
        for i in range(0, len(available_tickers), batch_size):
            batch = available_tickers[i:i+batch_size]
            print(f"  Processing fundamental data batch {i//batch_size + 1}/{(len(available_tickers)-1)//batch_size + 1}")
            
            for ticker in batch:
                self.fundamentals[ticker] = self._get_ticker_info(ticker)
            
            # Add delay between batches
            time.sleep(0.5)
        
        print("Fundamental data collection completed")
        
        # Try to extract Close and Volume data with error handling
        try:
            # Check if we have the expected MultiIndex structure
            if isinstance(self.data.columns, pd.MultiIndex):
                # Try to get Close data
                closes = None
                volumes = None
                
                # Look for Close data in different possible locations
                if 'Close' in self.data.columns.get_level_values(1):
                    closes = self.data.xs('Close', axis=1, level=1)
                elif 'Adj Close' in self.data.columns.get_level_values(1):
                    closes = self.data.xs('Adj Close', axis=1, level=1)
                
                # Look for Volume data
                if 'Volume' in self.data.columns.get_level_values(1):
                    volumes = self.data.xs('Volume', axis=1, level=1)
                
                if closes is not None:
                    # Remove tickers with all NaN values
                    closes = closes.dropna(axis=1, how='all')
                    print(f"Close prices available for {len(closes.columns)} tickers")
                    
                    self.closes = closes
                    if volumes is not None:
                        volumes = volumes.dropna(axis=1, how='all')
                        # Align volumes with closes
                        common_tickers = closes.columns.intersection(volumes.columns)
                        if len(common_tickers) > 0:
                            self.market_cap_proxy = closes[common_tickers] * volumes[common_tickers].rolling(252).mean()
                        else:
                            self.market_cap_proxy = closes  # Just use price as proxy
                    else:
                        self.market_cap_proxy = closes  # Just use price as proxy
                else:
                    print("Could not find Close price data")
                    self.closes = pd.DataFrame()
                    self.market_cap_proxy = pd.DataFrame()
            else:
                # Single level columns
                if 'Close' in self.data.columns:
                    self.closes = self.data[['Close']].copy()
                    if 'Volume' in self.data.columns:
                        self.market_cap_proxy = self.data['Close'] * self.data['Volume'].rolling(252).mean()
                    else:
                        self.market_cap_proxy = self.data['Close']
                else:
                    print("Could not find Close price data in single-level columns")
                    self.closes = pd.DataFrame()
                    self.market_cap_proxy = pd.DataFrame()
                    
        except Exception as e:
            print(f"Error in data preprocessing: {e}")
            self.closes = pd.DataFrame()
            self.market_cap_proxy = pd.DataFrame()
        
        print(f"Data preprocessing completed. Final close prices shape: {self.closes.shape}")

    def _graham_filters(self, date):
        """Simplified Graham criteria filters using available data"""
        dt = pd.to_datetime(date)
        
        # Use actual column names from the data instead of original sp500 list
        if hasattr(self, 'closes') and not self.closes.empty:
            available_tickers = self.closes.columns.tolist()
        else:
            available_tickers = self.sp500
        
        filtered = pd.Series(index=available_tickers, dtype=bool)
        
        if self.data.empty:
            return filtered.fillna(False)
        
        eligible_count = 0
        for ticker in available_tickers:
            try:
                ticker_data = self.fundamentals.get(ticker, {})
                
                # Size filter (use market cap if available)
                market_cap = ticker_data.get('market_cap', 0)
                size_ok = market_cap > 2e9 if market_cap else False  # More strict
                
                # PE ratio filter - more strict Graham criteria
                pe_ratio = ticker_data.get('pe_ratio', np.nan)
                pe_ok = (pe_ratio < 15 and pe_ratio > 0) if not np.isnan(pe_ratio) else False
                
                # PB ratio filter
                pb_ratio = ticker_data.get('pb_ratio', np.nan)
                pb_ok = (pb_ratio < 1.5 and pb_ratio > 0) if not np.isnan(pb_ratio) else False
                
                # Current ratio filter
                current_ratio = ticker_data.get('current_ratio', np.nan)
                cr_ok = current_ratio > 2 if not np.isnan(current_ratio) else False
                
                # Additional filters for quality
                revenue = ticker_data.get('revenue', 0)
                revenue_ok = revenue > 1e9 if revenue else False
                
                # Stock must pass at least 3 out of 5 criteria (more strict)
                criteria_passed = sum([size_ok, pe_ok, pb_ok, cr_ok, revenue_ok])
                passed = criteria_passed >= 3
                
                if passed:
                    eligible_count += 1
                    
                filtered[ticker] = passed
                
            except:
                filtered[ticker] = False
        
        print(f"    Graham filters: {eligible_count} stocks passed out of {len(available_tickers)}")
        
        # If too few stocks pass, relax criteria slightly
        if eligible_count < 5:
            print("    Too few stocks passed, relaxing criteria...")
            eligible_count = 0
            for ticker in available_tickers:
                try:
                    ticker_data = self.fundamentals.get(ticker, {})
                    
                    # More lenient criteria
                    market_cap = ticker_data.get('market_cap', 0)
                    size_ok = market_cap > 1e9 if market_cap else False
                    
                    pe_ratio = ticker_data.get('pe_ratio', np.nan)
                    pe_ok = (pe_ratio < 20 and pe_ratio > 0) if not np.isnan(pe_ratio) else False
                    
                    pb_ratio = ticker_data.get('pb_ratio', np.nan)
                    pb_ok = (pb_ratio < 2.0 and pb_ratio > 0) if not np.isnan(pb_ratio) else False
                    
                    # At least 2 out of 3 criteria
                    criteria_passed = sum([size_ok, pe_ok, pb_ok])
                    passed = criteria_passed >= 2
                    
                    if passed:
                        eligible_count += 1
                        
                    filtered[ticker] = passed
                    
                except:
                    filtered[ticker] = False
                    
            print(f"    Relaxed filters: {eligible_count} stocks passed")
        
        return filtered.fillna(False)

    def _risk_parity_weights(self, returns):
        """Risk parity portfolio optimization with error handling"""
        try:
            returns_clean = returns.dropna(axis=1)
            if returns_clean.empty or returns_clean.shape[1] < 2:
                return np.array([])
            
            cov = returns_clean.cov().values
            n = cov.shape[0]
            
            # Add small diagonal term for numerical stability
            cov += np.eye(n) * 1e-8
            
            w = Variable(n)
            risk_contrib = w @ cov @ w  # Use @ instead of * for matrix multiplication
            
            prob = Problem(
                Minimize(sum_squares(risk_contrib - 1/n)),
                [sum(w) == 1, w >= 0]
            )
            prob.solve(solver='ECOS', verbose=False)
            
            if prob.status == 'optimal':
                weights = np.array(w.value).flatten()
                # Ensure weights are valid and sum to 1
                if np.any(weights < 0) or np.any(np.isnan(weights)):
                    return np.ones(n) / n
                return weights / weights.sum()  # Normalize
            else:
                # Equal weights fallback
                return np.ones(n) / n
                
        except:
            # Equal weights fallback
            n = returns.shape[1]
            return np.ones(n) / n

    def backtest(self):
        """Main backtesting engine with error handling"""
        if self.data.empty:
            print("No data available for backtesting")
            return pd.Series([INITIAL_CAPITAL])
        
        # Use the preprocessed closes data
        if hasattr(self, 'closes') and not self.closes.empty:
            closes = self.closes
        else:
            print("No close price data available")
            return pd.Series([INITIAL_CAPITAL])
        
        returns = closes.pct_change()
        
        portfolio = pd.Series(index=closes.index, dtype=float)
        portfolio.iloc[0] = INITIAL_CAPITAL
        
        print(f"Starting backtest with {len(closes.columns)} tickers")
        print(f"Date range: {closes.index[0]} to {closes.index[-1]}")
        
        # Rebalance quarterly - use 'QE' instead of deprecated 'Q'
        rebalance_dates = pd.date_range(start=closes.index[0], end=closes.index[-1], freq='QE')
        rebalance_dates = rebalance_dates.intersection(closes.index)
        
        print(f"Rebalancing on {len(rebalance_dates)} dates")
        
        current_weights = None
        rebalance_count = 0
        
        for i in range(1, len(closes.index)):
            dt = closes.index[i]
            
            # Check if rebalancing is needed
            if dt in rebalance_dates:
                rebalance_count += 1
                print(f"Rebalancing {rebalance_count}/{len(rebalance_dates)}: {dt.strftime('%Y-%m-%d')}")
                
                eligible = self._graham_filters(dt)
                
                if eligible.any():
                    eligible_tickers = eligible[eligible].index.tolist()
                    print(f"  Eligible tickers: {len(eligible_tickers)}")
                    
                    # Limit to maximum 20 stocks for better risk management
                    if len(eligible_tickers) > 20:
                        # Select top 20 by market cap
                        market_caps = []
                        for ticker in eligible_tickers:
                            mc = self.fundamentals.get(ticker, {}).get('market_cap', 0)
                            market_caps.append((ticker, mc))
                        market_caps.sort(key=lambda x: x[1], reverse=True)
                        eligible_tickers = [x[0] for x in market_caps[:20]]
                        print(f"  Limited to top 20 by market cap: {len(eligible_tickers)}")
                    
                    # Ensure eligible tickers exist in returns data
                    available_eligible = [t for t in eligible_tickers if t in returns.columns]
                    print(f"  Available eligible tickers: {len(available_eligible)}")
                    
                    if len(available_eligible) > 0:
                        # Get returns for available eligible tickers
                        eligible_returns = returns.loc[:dt, available_eligible]
                        
                        # Remove tickers with insufficient data
                        min_observations = min(60, len(eligible_returns) // 2)
                        eligible_returns = eligible_returns.dropna(thresh=min_observations, axis=1)
                        
                        if not eligible_returns.empty and len(eligible_returns.columns) > 0:
                            # Use more recent data for weight calculation
                            recent_returns = eligible_returns.tail(min(252, len(eligible_returns)))
                            recent_returns = recent_returns.dropna(axis=1)
                            
                            if not recent_returns.empty and len(recent_returns.columns) > 0:
                                print(f"  Using {len(recent_returns.columns)} tickers for weight calculation")
                                
                                weights = self._risk_parity_weights(recent_returns)
                                if len(weights) > 0 and len(weights) == len(recent_returns.columns):
                                    current_weights = pd.Series(weights, index=recent_returns.columns)
                                    print(f"  Portfolio weights assigned to {len(current_weights)} stocks")
                                else:
                                    # Equal weights fallback
                                    current_weights = pd.Series(
                                        1.0 / len(recent_returns.columns),
                                        index=recent_returns.columns
                                    )
                                    print(f"  Using equal weights for {len(current_weights)} stocks")
                            else:
                                current_weights = None
                                print("  No valid recent returns data")
                        else:
                            current_weights = None
                            print("  No valid returns data for eligible stocks")
                    else:
                        current_weights = None
                        print("  No available eligible tickers in returns data")
                else:
                    current_weights = None
                    print("  No eligible stocks found")
            
            # Calculate portfolio return
            if current_weights is not None:
                # Get daily returns for current holdings
                available_tickers = current_weights.index.intersection(returns.columns)
                if len(available_tickers) > 0:
                    daily_returns = returns.loc[dt, available_tickers].fillna(0)
                    weights_aligned = current_weights.loc[available_tickers]
                    # Renormalize weights to ensure they sum to 1
                    weights_aligned = weights_aligned / weights_aligned.sum()
                    port_return = (daily_returns * weights_aligned).sum()
                    
                    # Add more conservative realism - cap daily returns at +/- 5%
                    port_return = max(-0.05, min(0.05, port_return))
                    
                    portfolio.iloc[i] = portfolio.iloc[i-1] * (1 + port_return)
                else:
                    portfolio.iloc[i] = portfolio.iloc[i-1]
            else:
                portfolio.iloc[i] = portfolio.iloc[i-1]
        
        print("Backtest completed")
        return portfolio

    def analyze_performance(self, portfolio):
        """Performance analysis and visualization"""
        if len(portfolio) <= 1:
            print("Insufficient data for performance analysis")
            return "No analysis available"
        
        # Calculate metrics
        returns = portfolio.pct_change().dropna()
        
        if len(returns) == 0:
            print("No returns data available")
            return "No analysis available"
        
        # Calculate performance metrics
        total_return = (portfolio.iloc[-1] / portfolio.iloc[0]) - 1
        years = len(portfolio) / 252
        annualized_return = (1 + total_return) ** (1/years) - 1
        sharpe = np.sqrt(252) * returns.mean() / returns.std() if returns.std() > 0 else 0
        max_dd = (portfolio / portfolio.cummax() - 1).min()
        
        print(f"Total Return: {total_return:.2%}")
        print(f"Annualized Return: {annualized_return:.2%}")
        print(f"Sharpe Ratio: {sharpe:.2f}")
        print(f"Max Drawdown: {max_dd:.2%}")
        
        # Fetch S&P 500 for comparison
        try:
            spy_data = yf.download('SPY', start=START_DATE, end=END_DATE, auto_adjust=True)
            
            # Handle different data structures returned by yfinance
            if isinstance(spy_data, pd.DataFrame):
                if 'Close' in spy_data.columns:
                    spy = spy_data['Close']
                elif len(spy_data.columns) == 1:
                    spy = spy_data.iloc[:, 0]
                else:
                    spy = spy_data['Adj Close'] if 'Adj Close' in spy_data.columns else spy_data.iloc[:, 0]
            else:
                spy = spy_data
            
            # Ensure we have valid data
            if len(spy) > 0:
                spy_returns = spy.pct_change().dropna()
                spy_total_return = (spy.iloc[-1] / spy.iloc[0]) - 1
                spy_years = len(spy) / 252
                spy_annualized_return = (1 + spy_total_return) ** (1/spy_years) - 1
                spy_sharpe = np.sqrt(252) * spy_returns.mean() / spy_returns.std() if spy_returns.std() > 0 else 0
                spy_max_dd = (spy / spy.cummax() - 1).min()
                
                print(f"\nS&P 500 Comparison:")
                print(f"SPY Total Return: {spy_total_return:.2%}")
                print(f"SPY Annualized Return: {spy_annualized_return:.2%}")
                print(f"SPY Sharpe Ratio: {spy_sharpe:.2f}")
                print(f"SPY Max Drawdown: {spy_max_dd:.2%}")
                
                # Calculate alpha and beta
                portfolio_returns = portfolio.pct_change().dropna()
                common_dates = portfolio_returns.index.intersection(spy_returns.index)
                if len(common_dates) > 252:  # At least 1 year of common data
                    port_ret_common = portfolio_returns.loc[common_dates]
                    spy_ret_common = spy_returns.loc[common_dates]
                    
                    # Calculate beta
                    covariance = np.cov(port_ret_common, spy_ret_common)[0, 1]
                    spy_variance = np.var(spy_ret_common)
                    beta = covariance / spy_variance if spy_variance > 0 else 0
                    
                    # Calculate alpha (annualized)
                    alpha = (annualized_return - spy_annualized_return * beta) * 100
                    
                    print(f"Alpha: {alpha:.2f}%")
                    print(f"Beta: {beta:.2f}")
                
            else:
                print("No valid SPY data found")
                spy = pd.Series()
                
        except Exception as e:
            print(f"Error fetching SPY data: {e}")
            spy = pd.Series()
        
        # Plotting
        plt.figure(figsize=(12, 8))
        
        # Normalize both series to start at 1
        portfolio_norm = portfolio / portfolio.iloc[0]
        portfolio_norm.plot(label='Graham Strategy', linewidth=2, color='blue')
        
        if len(spy) > 0:
            spy_norm = spy / spy.iloc[0]
            spy_norm.plot(label='S&P 500 (SPY)', linewidth=2, color='red', alpha=0.7)
        
        plt.title(f"Benjamin Graham Defensive Strategy Performance\nSharpe: {sharpe:.2f} | Max DD: {max_dd:.2%} | Ann. Return: {annualized_return:.2%}")
        plt.ylabel('Normalized Value (Starting at 1.0)')
        plt.xlabel('Date')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.yscale('log')  # Use log scale for better visualization of long-term returns
        plt.tight_layout()
        plt.show()
        
        return f"Sharpe: {sharpe:.2f}, Max DD: {max_dd:.2%}, Ann. Return: {annualized_return:.2%}"

if __name__ == "__main__":
    backtester = GrahamBacktester()
    results = backtester.backtest()
    analysis = backtester.analyze_performance(results)
    print(analysis)