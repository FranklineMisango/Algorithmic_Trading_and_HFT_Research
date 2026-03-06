"""
Data acquisition module for Strait of Hormuz geopolitical strategy.
Fetches shipping traffic, geopolitical events, and multi-asset market data.
Uses: Databento (futures), Alpaca (equities/forex/news), yfinance (fallback)
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from pathlib import Path
import yaml
import requests
from typing import Dict, List, Tuple
import warnings
import os
from dotenv import load_dotenv

warnings.filterwarnings('ignore')

# Load environment variables
load_dotenv()

# Optional imports (install if available)
try:
    import databento as db
    DATABENTO_AVAILABLE = True
except ImportError:
    DATABENTO_AVAILABLE = False
    print("⚠ Databento not installed. Run: pip install databento")

try:
    from alpaca.data.historical import StockHistoricalDataClient, CryptoHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest, CryptoBarsRequest, NewsRequest
    from alpaca.data.timeframe import TimeFrame
    ALPACA_AVAILABLE = True
except ImportError:
    ALPACA_AVAILABLE = False
    print("⚠ Alpaca not installed. Run: pip install alpaca-py")


class DataAcquisition:
    """Fetch and process data for geopolitical risk strategy."""
    
    def __init__(self, config_path: str = 'config.yaml'):
        """Initialize with configuration."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.start_date = self.config['data']['start_date']
        self.end_date = self.config['data']['end_date']
        
        # Initialize API clients
        self._init_api_clients()
    
    def _init_api_clients(self):
        """Initialize API clients with credentials."""
        # Databento
        self.databento_client = None
        if DATABENTO_AVAILABLE:
            api_key = os.getenv('DATABENTO_API_KEY')
            if api_key:
                try:
                    self.databento_client = db.Historical(api_key)
                    print("✓ Databento client initialized")
                except Exception as e:
                    print(f"⚠ Databento initialization failed: {e}")
        
        # Alpaca
        self.alpaca_stock_client = None
        self.alpaca_crypto_client = None
        if ALPACA_AVAILABLE:
            api_key = os.getenv('ALPACA_API_KEY')
            secret_key = os.getenv('ALPACA_SECRET_KEY')
            if api_key and secret_key:
                try:
                    self.alpaca_stock_client = StockHistoricalDataClient(api_key, secret_key)
                    self.alpaca_crypto_client = CryptoHistoricalDataClient(api_key, secret_key)
                    print("✓ Alpaca clients initialized")
                except Exception as e:
                    print(f"⚠ Alpaca initialization failed: {e}")
        
        # yfinance always available (no key needed)
        print("✓ yfinance available (no key required)")
    
    def fetch_market_data(self) -> Dict[str, pd.DataFrame]:
        """Fetch multi-asset market data across regions."""
        print("Fetching market data...")
        
        market_data = {}
        
        # 1. Equities (Alpaca primary, yfinance fallback)
        market_data['equities'] = self._fetch_equities()
        
        # 2. Fixed Income (yfinance)
        market_data['fixed_income'] = self._fetch_fixed_income()
        
        # 3. Futures (Databento)
        market_data['futures'] = self._fetch_futures()
        
        # 4. Commodities ETFs (yfinance)
        market_data['commodities'] = self._fetch_commodities()
        
        # 5. Currencies (Alpaca)
        market_data['currencies'] = self._fetch_currencies()
        
        # 6. Crypto (Alpaca)
        market_data['crypto'] = self._fetch_crypto()
        
        return market_data
    
    def _fetch_equities(self) -> Dict[str, pd.DataFrame]:
        """Fetch equity data from local files or APIs."""
        print("  → Fetching equities...")
        
        equities = {}
        equity_config = self.config['data']['markets']['equities']
        
        # Check if using local data
        if equity_config.get('source') == 'local':
            print("    → Loading from local files...")
            
            # Load US equities
            us_tickers = equity_config.get('us', [])
            if us_tickers:
                us_data = self._load_local_equities('US', us_tickers)
                if not us_data.empty:
                    equities['us'] = us_data
            
            # Load Europe equities
            europe_tickers = equity_config.get('europe', [])
            if europe_tickers:
                europe_data = self._load_local_equities('Europe', europe_tickers)
                if not europe_data.empty:
                    equities['europe'] = europe_data
            
            # Load Asia equities
            asia_tickers = equity_config.get('asia', [])
            if asia_tickers:
                asia_data = self._load_local_equities('Asia', asia_tickers)
                if not asia_data.empty:
                    equities['asia'] = asia_data
            
            # Load additional categories
            for category in ['asia_importers_short', 'europe_vulnerable_short', 'beneficiaries_long']:
                if category in self.config['data']['markets']:
                    cat_config = self.config['data']['markets'][category]
                    if isinstance(cat_config, dict):
                        for region, tickers in cat_config.items():
                            if region == 'source':
                                continue
                            if isinstance(tickers, list):
                                # Determine folder based on region
                                folder = self._get_equity_folder(region)
                                data = self._load_local_equities(folder, tickers)
                                if not data.empty:
                                    equities[f"{category}_{region}"] = data
            
            print(f"    ✓ Loaded {len(equities)} equity groups from local files")
            return equities
        
        # Original API-based fetching code follows...
        # Try Alpaca first if available
        if self.alpaca_stock_client and equity_config.get('source') == 'alpaca':
            try:
                for region, tickers in equity_config.items():
                    if region in ['source', 'fallback']:
                        continue
                    
                    if isinstance(tickers, list):
                        request_params = StockBarsRequest(
                            symbol_or_symbols=tickers,
                            timeframe=TimeFrame.Day,
                            start=datetime.strptime(self.start_date, '%Y-%m-%d'),
                            end=datetime.strptime(self.end_date, '%Y-%m-%d')
                        )
                        bars = self.alpaca_stock_client.get_stock_bars(request_params)
                        
                        # Convert to DataFrame
                        df_list = []
                        for ticker in tickers:
                            if ticker in bars.data:
                                ticker_data = bars.data[ticker]
                                df = pd.DataFrame([{
                                    'timestamp': bar.timestamp,
                                    'close': bar.close
                                } for bar in ticker_data])
                                df.set_index('timestamp', inplace=True)
                                df.columns = [ticker]
                                df_list.append(df)
                        
                        if df_list:
                            equities[region] = pd.concat(df_list, axis=1)
                
                print(f"    ✓ Fetched via Alpaca")
                return equities
            
            except Exception as e:
                print(f"    ⚠ Alpaca failed: {e}, falling back to yfinance")
        
        # Fallback to yfinance
        for region, tickers in equity_config.items():
            if region in ['source', 'fallback']:
                continue
            
            if isinstance(tickers, list):
                try:
                    data = yf.download(tickers, start=self.start_date, end=self.end_date, progress=False)
                    if len(tickers) > 1:
                        equities[region] = data['Adj Close']
                    else:
                        # Handle single ticker case
                        adj_close = data['Adj Close'] if 'Adj Close' in data.columns else data['Close']
                        equities[region] = pd.DataFrame(adj_close)
                        equities[region].columns = tickers
                except Exception as e:
                    print(f"    ⚠ Error fetching {region}: {e}")
        
        print(f"    ✓ Fetched via yfinance")
        return equities
    
    def _get_equity_folder(self, region: str) -> str:
        """Map region name to folder name."""
        region_map = {
            'us': 'US',
            'us_lng_exporters': 'US',
            'us_oil_producers': 'US',
            'us_oil_services': 'US',
            'us_tankers': 'US',
            'japan': 'Asia',
            'south_korea': 'Asia',
            'india': 'Asia',
            'china': 'Asia',
            'taiwan': 'Asia',
            'germany': 'Europe',
            'italy': 'Europe',
            'poland': 'Europe',
            'norway': 'Europe',
            'australia': 'Asia',
        }
        return region_map.get(region.lower(), 'US')
    
    def _load_local_equities(self, folder: str, tickers: List[str]) -> pd.DataFrame:
        """Load equity data from local parquet/csv files."""
        data_frames = []
        
        for ticker in tickers:
            # Try parquet first, then csv
            parquet_path = Path(f'Data/Equities/{folder}/{ticker}.parquet')
            csv_path = Path(f'Data/Equities/{folder}/{ticker}.csv')
            
            try:
                if parquet_path.exists():
                    df = pd.read_parquet(parquet_path)
                    if 'Date' in df.columns:
                        df.set_index('Date', inplace=True)
                    # Get close price column
                    close_col = 'Close' if 'Close' in df.columns else 'Adj Close' if 'Adj Close' in df.columns else df.columns[0]
                    series = df[close_col]
                    series.name = ticker
                    data_frames.append(series)
                elif csv_path.exists():
                    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
                    close_col = 'Close' if 'Close' in df.columns else 'Adj Close' if 'Adj Close' in df.columns else df.columns[0]
                    series = df[close_col]
                    series.name = ticker
                    data_frames.append(series)
            except Exception as e:
                print(f"      ⚠ Could not load {ticker}: {e}")
        
        if data_frames:
            result = pd.concat(data_frames, axis=1)
            # Filter by date range
            result = result.loc[self.start_date:self.end_date]
            return result
        
        return pd.DataFrame()
    
    def _load_local_data(self, folder: str, tickers: List[str]) -> pd.DataFrame:
        """Generic method to load data from local parquet/csv files."""
        data_frames = []
        
        for ticker in tickers:
            # Try parquet first, then csv
            parquet_path = Path(f'Data/{folder}/{ticker}.parquet')
            csv_path = Path(f'Data/{folder}/{ticker}.csv')
            
            try:
                if parquet_path.exists():
                    df = pd.read_parquet(parquet_path)
                    if 'Date' in df.columns:
                        df.set_index('Date', inplace=True)
                    close_col = 'Close' if 'Close' in df.columns else 'Adj Close' if 'Adj Close' in df.columns else df.columns[0]
                    series = df[close_col]
                    series.name = ticker
                    data_frames.append(series)
                elif csv_path.exists():
                    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
                    close_col = 'Close' if 'Close' in df.columns else 'Adj Close' if 'Adj Close' in df.columns else df.columns[0]
                    series = df[close_col]
                    series.name = ticker
                    data_frames.append(series)
            except Exception as e:
                print(f"      ⚠ Could not load {ticker}: {e}")
        
        if data_frames:
            result = pd.concat(data_frames, axis=1)
            # Filter by date range
            result = result.loc[self.start_date:self.end_date]
            return result
        
        return pd.DataFrame()

    
    def _fetch_fixed_income(self) -> pd.DataFrame:
        """Fetch fixed income data from local files or yfinance."""
        print("  → Fetching fixed income...")
        
        fixed_income_config = self.config['data']['markets']['fixed_income']
        
        # Check if using local data
        if fixed_income_config.get('source') == 'local':
            print("    → Loading from local files...")
            
            # Get all tickers from safe_havens_long and vulnerable_short
            tickers = []
            if 'safe_havens_long' in fixed_income_config:
                tickers.extend(fixed_income_config['safe_havens_long'])
            if 'vulnerable_short' in fixed_income_config:
                tickers.extend(fixed_income_config['vulnerable_short'])
            
            if tickers:
                df = self._load_local_data('Fixed_Income', tickers)
                if not df.empty:
                    print(f"    ✓ Loaded {len(tickers)} instruments from local files")
                    return df
        
        # Original API-based code
        if isinstance(fixed_income_config, dict):
            tickers = fixed_income_config.get('symbols', fixed_income_config.get('tickers', []))
        else:
            tickers = fixed_income_config
        
        if not tickers:
            print("    ⚠ No fixed income tickers configured")
            return pd.DataFrame()
        
        try:
            data = yf.download(tickers, start=self.start_date, end=self.end_date, 
                             progress=False, auto_adjust=False)
            if len(tickers) > 1:
                df = data['Adj Close']
            else:
                # Handle single ticker case
                df = pd.DataFrame(data['Adj Close'])
                df.columns = tickers
            print(f"    ✓ Fetched {len(tickers)} instruments")
            return df
        except Exception as e:
            print(f"    ⚠ Error: {e}")
            return pd.DataFrame()
    
    def _fetch_futures(self) -> pd.DataFrame:
        """Fetch futures data from local files or APIs."""
        print("  → Fetching futures...")
        
        futures_config = self.config['data']['markets']['futures']
        
        # Check if using local data
        if futures_config.get('source') == 'local':
            print("    → Loading from local Futures folder...")
            
            # Map symbols to files
            futures_map = {
                'CL.FUT': 'CL',
                'BZ.FUT': 'BZ',
                'NG.FUT': 'NG',
                'ES.FUT': 'ES',
                'NQ.FUT': 'NQ',
            }
            
            data_frames = []
            for config_symbol, file_symbol in futures_map.items():
                parquet_path = Path(f'Data/Futures/{file_symbol}.parquet')
                csv_path = Path(f'Data/Futures/{file_symbol}.csv')
                
                try:
                    if parquet_path.exists():
                        df = pd.read_parquet(parquet_path)
                        if 'Date' in df.columns:
                            df.set_index('Date', inplace=True)
                        close_col = 'Close' if 'Close' in df.columns else 'close' if 'close' in df.columns else df.columns[0]
                        series = df[close_col]
                        series.name = file_symbol
                        data_frames.append(series)
                    elif csv_path.exists():
                        df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
                        close_col = 'Close' if 'Close' in df.columns else 'close' if 'close' in df.columns else df.columns[0]
                        series = df[close_col]
                        series.name = file_symbol
                        data_frames.append(series)
                except Exception as e:
                    print(f"      ⚠ Could not load {file_symbol}: {e}")
            
            if data_frames:
                result = pd.concat(data_frames, axis=1)
                result = result.loc[self.start_date:self.end_date]
                print(f"    ✓ Loaded {len(data_frames)} futures from local files")
                return result
            else:
                print("    ⚠ No local futures files found, checking cache...")
        
        # Check cache
        cache_dir = Path('.cache/databento')
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = cache_dir / f'futures_{self.start_date}_{self.end_date}.parquet'
        
        if cache_file.exists():
            print("    ✓ Loading from cache")
            return pd.read_parquet(cache_file)
        
        if not self.databento_client:
            print("    ⚠ Databento not available, using yfinance proxies")
            # Use futures ETFs as proxy
            proxies = {
                'CL': 'CL=F',  # WTI
                'BZ': 'BZ=F',  # Brent
                'NG': 'NG=F',  # Natural Gas
            }
            try:
                data = yf.download(list(proxies.values()), start=self.start_date, 
                                 end=self.end_date, progress=False)
                df = data['Adj Close']
                df.columns = list(proxies.keys())
                print(f"    ✓ Fetched {len(proxies)} futures proxies")
                return df
            except Exception as e:
                print(f"    ⚠ Error: {e}")
                return pd.DataFrame()
        
        # Use Databento for high-quality futures data
        try:
            futures_config = self.config['data']['markets']['futures']
            all_futures = {}
            
            for category, symbols in futures_config.items():
                if category == 'source':
                    continue
                
                for symbol in symbols:
                    # Databento symbol format
                    db_symbol = symbol.replace('.FUT', '')
                    
                    # Fetch data
                    data = self.databento_client.timeseries.get_range(
                        dataset='GLBX.MDP3',  # CME Globex
                        symbols=[db_symbol],
                        schema='ohlcv-1d',
                        start=self.start_date,
                        end=self.end_date
                    )
                    
                    df = data.to_df()
                    all_futures[db_symbol] = df['close']
            
            futures_df = pd.DataFrame(all_futures)
            
            # Save to cache
            futures_df.to_parquet(cache_file)
            print(f"    ✓ Fetched {len(all_futures)} futures via Databento (cached)")
            return futures_df
        
        except Exception as e:
            print(f"    ⚠ Databento error: {e}, using proxies")
            return self._fetch_futures()  # Recursive call to use proxies
    
    def _fetch_commodities(self) -> pd.DataFrame:
        """Fetch commodity ETFs from local files or yfinance."""
        print("  → Fetching commodities...")
        
        commodities_config = self.config['data']['markets']['commodities']
        
        # Check if using local data
        if commodities_config.get('source') == 'local':
            print("    → Loading from local files...")
            tickers = commodities_config.get('symbols', [])
            
            if tickers:
                df = self._load_local_data('Commodities', tickers)
                if not df.empty:
                    print(f"    ✓ Loaded {len(tickers)} commodities from local files")
                    return df
        
        # Original API-based code
        if isinstance(commodities_config, dict):
            tickers = commodities_config.get('symbols', commodities_config.get('tickers', []))
        else:
            tickers = commodities_config
        
        if not tickers:
            print("    ⚠ No commodity tickers configured")
            return pd.DataFrame()
        
        try:
            data = yf.download(tickers, start=self.start_date, end=self.end_date, 
                             progress=False, auto_adjust=False)
            if len(tickers) > 1:
                df = data['Adj Close']
            else:
                # Handle single ticker case
                df = pd.DataFrame(data['Adj Close'])
                df.columns = tickers
            print(f"    ✓ Fetched {len(tickers)} commodities")
            return df
        except Exception as e:
            print(f"    ⚠ Error: {e}")
            return pd.DataFrame()
    
    def _fetch_currencies(self) -> pd.DataFrame:
        """Fetch currency data from local files or APIs."""
        print("  → Fetching currencies...")
        
        currencies_config = self.config['data']['markets']['currencies']
        
        # Check if using local data
        if currencies_config.get('source') == 'local':
            print("    → Loading from local files...")
            
            # Map currency pairs to file names
            currency_files = {
                'NOK/USD': 'USD_NOK_Gas_Exporter',
                'CAD/USD': 'USD_CAD_Oil_Exporter',
                'AUD/USD': 'AUD_USD_LNG_Exporter',
                'JPY/USD': 'USD_JPY_Oil_Importer',
                'KRW/USD': 'USD_KRW_Oil_Importer',
                'INR/USD': 'USD_INR_Oil_Importer',
                'CNH/USD': 'USD_CNH_Strategic_Importer',
            }
            
            data_frames = []
            for pair, filename in currency_files.items():
                parquet_path = Path(f'Data/Forex/{filename}.parquet')
                csv_path = Path(f'Data/Forex/{filename}.csv')
                
                try:
                    if parquet_path.exists():
                        df = pd.read_parquet(parquet_path)
                        if 'Date' in df.columns:
                            df.set_index('Date', inplace=True)
                        close_col = 'Close' if 'Close' in df.columns else df.columns[0]
                        series = df[close_col]
                        series.name = pair
                        data_frames.append(series)
                    elif csv_path.exists():
                        df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
                        close_col = 'Close' if 'Close' in df.columns else df.columns[0]
                        series = df[close_col]
                        series.name = pair
                        data_frames.append(series)
                except Exception as e:
                    print(f"      ⚠ Could not load {pair}: {e}")
            
            if data_frames:
                result = pd.concat(data_frames, axis=1)
                result = result.loc[self.start_date:self.end_date]
                print(f"    ✓ Loaded {len(data_frames)} currency pairs from local files")
                return result
        
        # Original API-based code - Use currency ETFs as proxy
        fx_etfs = ['FXE', 'FXY', 'FXC', 'FXA']  # EUR, JPY, CAD, AUD
        try:
            data = yf.download(fx_etfs, start=self.start_date, end=self.end_date, 
                             progress=False, auto_adjust=False)
            if len(fx_etfs) > 1:
                df = data['Adj Close']
            else:
                df = pd.DataFrame(data['Adj Close'])
                df.columns = fx_etfs
            print(f"    ✓ Fetched {len(fx_etfs)} FX ETFs")
            return df
        except Exception as e:
            print(f"    ⚠ Error: {e}")
            return pd.DataFrame()
    
    def _fetch_crypto(self) -> pd.DataFrame:
        """Fetch crypto data using Binance, Alpaca, or yfinance."""
        print("  → Fetching crypto...")
        
        # Try Binance first (free, no API key needed for historical data)
        try:
            import ccxt
            exchange = ccxt.binance()
            
            symbols = ['BTC/USDT', 'ETH/USDT']
            dfs = []
            
            for symbol in symbols:
                # Fetch OHLCV data
                since = int(pd.Timestamp(self.start_date).timestamp() * 1000)
                ohlcv = exchange.fetch_ohlcv(symbol, '1d', since=since)
                
                df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
                df.set_index('date', inplace=True)
                dfs.append(df['close'].rename(symbol.split('/')[0]))
            
            result = pd.concat(dfs, axis=1)
            print(f"    ✓ Fetched {len(symbols)} cryptos via Binance")
            return result
            
        except Exception as e:
            print(f"    ⚠ Binance error: {e}")
        
        # Fallback to Alpaca
        if self.alpaca_crypto_client:
            try:
                crypto_config = self.config['data']['markets']['crypto']
                if isinstance(crypto_config, dict):
                    crypto_config = crypto_config.get('tickers', ['BTC/USD', 'ETH/USD'])
                
                # Fetch from Alpaca
                dfs = []
                for symbol in crypto_config:
                    bars = self.alpaca_crypto_client.get_crypto_bars(
                        symbol,
                        TimeFrame.Day,
                        start=self.start_date,
                        end=self.end_date
                    ).df
                    dfs.append(bars['close'].rename(symbol.split('/')[0]))
                
                df = pd.concat(dfs, axis=1)
                print(f"    ✓ Fetched {len(crypto_config)} cryptos via Alpaca")
                return df
            except Exception as e:
                print(f"    ⚠ Alpaca crypto error: {e}")
        
        # Final fallback to yfinance
        print("    ⚠ Using yfinance for crypto")
        crypto_tickers = ['BTC-USD', 'ETH-USD']
        try:
            data = yf.download(crypto_tickers, start=self.start_date, 
                             end=self.end_date, progress=False)
            if len(crypto_tickers) > 1:
                df = data['Adj Close']
            else:
                adj_close = data['Adj Close'] if 'Adj Close' in data.columns else data['Close']
                df = pd.DataFrame(adj_close)
                df.columns = ['BTC']
            df.columns = ['BTC', 'ETH']
            print(f"    ✓ Fetched {len(crypto_tickers)} cryptos via yfinance")
            return df
        except Exception as e:
                print(f"    ⚠ Error: {e}")
                return pd.DataFrame()
        
        # Use Alpaca for crypto
        try:
            crypto_symbols = self.config['data']['markets']['crypto']
            if isinstance(crypto_symbols, dict):
                crypto_symbols = crypto_symbols.get('symbols', ['BTC/USD', 'ETH/USD'])
            
            request_params = CryptoBarsRequest(
                symbol_or_symbols=crypto_symbols,
                timeframe=TimeFrame.Day,
                start=datetime.strptime(self.start_date, '%Y-%m-%d'),
                end=datetime.strptime(self.end_date, '%Y-%m-%d')
            )
            
            bars = self.alpaca_crypto_client.get_crypto_bars(request_params)
            
            # Convert to DataFrame
            df_list = []
            for symbol in crypto_symbols:
                if symbol in bars.data:
                    symbol_data = bars.data[symbol]
                    df = pd.DataFrame([{
                        'timestamp': bar.timestamp,
                        'close': bar.close
                    } for bar in symbol_data])
                    df.set_index('timestamp', inplace=True)
                    df.columns = [symbol.split('/')[0]]
                    df_list.append(df)
            
            if df_list:
                crypto_df = pd.concat(df_list, axis=1)
                print(f"    ✓ Fetched {len(df_list)} cryptos via Alpaca")
                return crypto_df
        
        except Exception as e:
            print(f"    ⚠ Alpaca error: {e}, using yfinance")
            return self._fetch_crypto()  # Recursive to use yfinance
    
    def fetch_shipping_data(self) -> pd.DataFrame:
        """
        Fetch shipping traffic data through Strait of Hormuz.
        Uses real PortWatch IMF data if available, otherwise generates synthetic data.
        """
        print("Fetching shipping traffic data...")
        
        # Try to load real PortWatch data
        try:
            from portwatch_loader import PortWatchLoader
            
            loader = PortWatchLoader()
            shipping_df = loader.get_tanker_traffic(
                start_date=self.start_date,
                end_date=self.end_date
            )
            
            # Rename index to 'date' for consistency
            shipping_df.index.name = 'date'
            
            print(f"✓ Loaded REAL PortWatch IMF data: {len(shipping_df)} days")
            return shipping_df
        
        except (FileNotFoundError, ImportError) as e:
            print(f"  ⚠ PortWatch data not available: {e}")
            print(f"  → Using synthetic data for demonstration")
            return self._generate_synthetic_shipping_data()
    
    def _generate_synthetic_shipping_data(self) -> pd.DataFrame:
        """Generate synthetic shipping data for demonstration."""
        # Generate synthetic shipping data for demonstration
        dates = pd.date_range(self.start_date, self.end_date, freq='D')
        
        # Baseline: ~25 tankers per day through Strait of Hormuz
        baseline = 25
        
        # Add seasonal variation
        seasonal = 3 * np.sin(2 * np.pi * np.arange(len(dates)) / 365)
        
        # Add crisis events (reduced traffic)
        traffic = baseline + seasonal + np.random.normal(0, 2, len(dates))
        
        # Simulate crisis periods
        crisis_events = self.config['backtest']['crisis_events']
        for event in crisis_events:
            start = pd.to_datetime(event['start'])
            end = pd.to_datetime(event['end'])
            mask = (dates >= start) & (dates <= end)
            # Reduce traffic by 30-70% during crisis
            traffic[mask] *= np.random.uniform(0.3, 0.7)
        
        shipping_df = pd.DataFrame({
            'date': dates,
            'tanker_transits': np.maximum(traffic, 0),
            'lng_transits': np.maximum(traffic * 0.4, 0),  # LNG carriers
            'total_transits': np.maximum(traffic * 1.4, 0)
        })
        
        shipping_df.set_index('date', inplace=True)
        
        print(f"✓ Generated synthetic shipping data: {len(shipping_df)} days")
        return shipping_df
    
    def fetch_geopolitical_events(self) -> pd.DataFrame:
        """
        Fetch geopolitical events and news sentiment.
        Uses multi-source news sentiment (GDELT, NewsAPI, Google, Alpaca).
        NO SYNTHETIC DATA - will raise error if real data unavailable.
        """
        print("Fetching geopolitical events...")
        
        dates = pd.date_range(self.start_date, self.end_date, freq='D')
        
        # Try to fetch real news sentiment from multiple sources
        news_sentiment = self._fetch_news_sentiment_multisource()
        
        if news_sentiment is not None and len(news_sentiment) > 0:
            print("  ✓ Using REAL news sentiment from multiple sources")
            # Merge with date range
            events_df = pd.DataFrame({'date': dates})
            events_df.set_index('date', inplace=True)
            events_df = events_df.join(news_sentiment, how='left')
            
            # Forward fill missing days (reasonable for news sentiment)
            events_df.fillna(method='ffill', inplace=True)
            # Backfill any remaining NaNs at the start
            events_df.fillna(method='bfill', inplace=True)
            
            # Ensure required columns exist
            if 'news_sentiment' not in events_df.columns:
                events_df['news_sentiment'] = events_df.get('sentiment_mean', 0)
            if 'conflict_events' not in events_df.columns:
                events_df['conflict_events'] = events_df.get('article_count', 0)
            if 'military_activity' not in events_df.columns:
                # Estimate from risk score
                events_df['military_activity'] = events_df.get('risk_score', 50)
            
            # If still mostly empty, raise error
            if events_df['news_sentiment'].isna().sum() > len(events_df) * 0.9:
                raise ValueError("Insufficient real news data - over 90% missing values")
            
            return events_df
            
        else:
            # NO SYNTHETIC DATA - raise error
            raise ValueError(
                "Cannot fetch real geopolitical data. Please check:\n"
                "  1. NewsAPI key is configured and valid\n"
                "  2. Date range is within API limits (NewsAPI free: last 30 days)\n"
                "  3. Google Search API is configured for historical data\n"
                "  4. GDELT or Alpaca News APIs are available\n"
                "  Consider using a shorter date range or upgrading API plans."
            )
            
            events_df = pd.DataFrame({
                'date': dates,
                'risk_score': np.clip(risk_scores, 0, 100),
                'news_sentiment': np.random.uniform(-1, 1, len(dates)),
                'conflict_events': np.random.poisson(0.5, len(dates)),
                'military_activity': np.random.uniform(0, 100, len(dates))
            })
            
            events_df.set_index('date', inplace=True)
        
        print(f"✓ Geopolitical events data: {len(events_df)} days")
        return events_df
    
    def _fetch_news_sentiment_multisource(self) -> pd.DataFrame:
        """Fetch news sentiment from GDELT with OpenAI analysis."""
        try:
            from news_sentiment_llm import EnhancedNewsSentimentAnalyzer
            
            print("  → Fetching news sentiment with GDELT + OpenAI...")
            
            analyzer = EnhancedNewsSentimentAnalyzer(
                config_path='appsettings.json',
                fetch_full_articles=True
            )
            
            keywords = self.config['data']['geopolitical']['keywords']
            
            sentiment_df = analyzer.get_geopolitical_sentiment(
                start_date=self.start_date,
                end_date=self.end_date,
                keywords=keywords
            )
            
            if sentiment_df is None or len(sentiment_df) == 0:
                print("    ⚠ No news data returned from GDELT")
                return None
            
            # Rename columns to match expected format
            if 'sentiment_mean' in sentiment_df.columns:
                sentiment_df['news_sentiment'] = sentiment_df['sentiment_mean']
            if 'article_count' in sentiment_df.columns:
                sentiment_df['conflict_events'] = sentiment_df['article_count']
            
            print(f"    ✓ GDELT + OpenAI sentiment analysis complete: {len(sentiment_df)} days")
            return sentiment_df
        
        except ImportError as e:
            print(f"    ⚠ Sentiment module not available: {e}")
            return None
        except Exception as e:
            print(f"    ⚠ Error fetching news sentiment: {e}")
            return None
    
    def fetch_macro_indicators(self) -> pd.DataFrame:
        """Fetch macro indicators (VIX, oil prices, yields)."""
        print("Fetching macro indicators...")
        
        macro_df = pd.DataFrame()
        
        try:
            # VIX - Volatility Index
            print("  → Fetching VIX...")
            vix_data = yf.download('^VIX', start=self.start_date, end=self.end_date, 
                                 progress=False, auto_adjust=False)
            vix = vix_data['Adj Close'] if 'Adj Close' in vix_data.columns else vix_data['Close']
            macro_df['vix'] = vix
            
            # Oil prices - Use cached futures data if available
            print("  → Fetching oil futures...")
            futures_df = self._fetch_futures()
            
            if not futures_df.empty:
                # Map futures symbols to macro columns
                if 'CL' in futures_df.columns:
                    macro_df['wti_oil'] = futures_df['CL']
                if 'BZ' in futures_df.columns:
                    macro_df['brent_oil'] = futures_df['BZ']
                print("    ✓ Using futures data for oil prices")
            else:
                # Fallback to yfinance
                print("  → Fetching oil prices from yfinance...")
                wti_data = yf.download('CL=F', start=self.start_date, end=self.end_date, 
                                     progress=False, auto_adjust=False)
                brent_data = yf.download('BZ=F', start=self.start_date, end=self.end_date, 
                                       progress=False, auto_adjust=False)
                wti = wti_data['Adj Close'] if 'Adj Close' in wti_data.columns else wti_data['Close']
                brent = brent_data['Adj Close'] if 'Adj Close' in brent_data.columns else brent_data['Close']
                macro_df['wti_oil'] = wti
                macro_df['brent_oil'] = brent
        
        except Exception as e:
            print(f"    ⚠ Error fetching oil data: {e}")
        
        try:
            # Treasury yields
            print("  → Fetching Treasury yields...")
            tnx_data = yf.download('^TNX', start=self.start_date, end=self.end_date, 
                                 progress=False, auto_adjust=False)
            tnx = tnx_data['Adj Close'] if 'Adj Close' in tnx_data.columns else tnx_data['Close']
            macro_df['treasury_10y'] = tnx
            
            print(f"✓ Fetched macro indicators")
            
        except Exception as e:
            print(f"    ⚠ Error fetching Treasury data: {e}")
            dates = pd.date_range(self.start_date, self.end_date, freq='D')
            if 'vix' not in macro_df:
                macro_df['vix'] = pd.Series(15 + np.random.normal(0, 5, len(dates)), index=dates)
            if 'wti_oil' not in macro_df:
                macro_df['wti_oil'] = pd.Series(70 + np.random.normal(0, 10, len(dates)), index=dates)
            if 'brent_oil' not in macro_df:
                macro_df['brent_oil'] = pd.Series(75 + np.random.normal(0, 10, len(dates)), index=dates)
            if 'treasury_10y' not in macro_df:
                macro_df['treasury_10y'] = pd.Series(2.5 + np.random.normal(0, 0.5, len(dates)), index=dates)
        
        return macro_df
    
    def _generate_synthetic_market_data(self) -> Dict[str, pd.DataFrame]:
        """Generate synthetic market data for testing."""
        print("Generating synthetic market data...")
        
        dates = pd.date_range(self.start_date, self.end_date, freq='D')
        n_days = len(dates)
        
        market_data = {
            'equities': {
                'us': pd.DataFrame({
                    'SPY': 100 * np.exp(np.cumsum(np.random.normal(0.0003, 0.01, n_days))),
                    'XLE': 100 * np.exp(np.cumsum(np.random.normal(0.0004, 0.015, n_days))),
                }, index=dates),
                'europe': pd.DataFrame({
                    'EZU': 100 * np.exp(np.cumsum(np.random.normal(0.0002, 0.01, n_days))),
                }, index=dates),
                'asia': pd.DataFrame({
                    'EWJ': 100 * np.exp(np.cumsum(np.random.normal(0.0002, 0.012, n_days))),
                }, index=dates)
            },
            'fixed_income': pd.DataFrame({
                'TLT': 100 * np.exp(np.cumsum(np.random.normal(0.0001, 0.008, n_days))),
            }, index=dates),
            'commodities': pd.DataFrame({
                'USO': 100 * np.exp(np.cumsum(np.random.normal(0.0002, 0.02, n_days))),
            }, index=dates)
        }
        
        return market_data
    
    def fetch_all_data(self) -> Dict:
        """Fetch all required data."""
        print("\n" + "="*60)
        print("STRAIT OF HORMUZ STRATEGY - DATA ACQUISITION")
        print("="*60 + "\n")
        
        data = {
            'market': self.fetch_market_data(),
            'shipping': self.fetch_shipping_data(),
            'geopolitical': self.fetch_geopolitical_events(),
            'macro': self.fetch_macro_indicators()
        }
        
        print("\n" + "="*60)
        print("DATA ACQUISITION COMPLETE")
        print("="*60 + "\n")
        
        return data


if __name__ == "__main__":
    acq = DataAcquisition()
    data = acq.fetch_all_data()
    
    print("\nData Summary:")
    print(f"Market data: {len(data['market'])} asset classes")
    print(f"Shipping data: {len(data['shipping'])} days")
    print(f"Geopolitical events: {len(data['geopolitical'])} days")
    print(f"Macro indicators: {len(data['macro'])} days")
