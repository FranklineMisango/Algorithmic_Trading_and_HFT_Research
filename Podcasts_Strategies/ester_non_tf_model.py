import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import mean_squared_error
import yfinance as yf
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class EsterStrategy:
    def __init__(self, lookback_period=21, holding_period=21, transaction_cost=0.001, 
                 max_drawdown_limit=0.15, volatility_target=0.12):
        self.lookback = lookback_period
        self.holding = holding_period
        self.transaction_cost = transaction_cost
        self.max_drawdown_limit = max_drawdown_limit
        self.volatility_target = volatility_target
        
        # Use RobustScaler for better outlier handling
        self.scaler = RobustScaler()
        
        # Model initialization with better hyperparameters
        self.gbt_model = GradientBoostingRegressor(
            n_estimators=50, 
            max_depth=3,
            learning_rate=0.1,
            subsample=0.8,
            random_state=42
        )
        self.nn_model = None
        
        # Dynamic model weighting with regime detection
        self.model_weights = [0.5, 0.5]
        self.performance_history = {'gbt': [], 'nn': []}
        self.regime_indicator = 0.5  # 0 = low vol, 1 = high vol
        
        # Risk management
        self.current_drawdown = 0.0
        self.peak_value = 1.0
        self.volatility_window = 20
        
        # Performance tracking
        self.performance_metrics = {
            'returns': [],
            'dates': [],
            'positions': [],
            'drawdowns': [],
            'sharpe_ratios': [],
            'volatilities': []
        }

    def detect_regime(self, returns_series):
        """Detect market regime based on volatility and skewness"""
        if len(returns_series) < self.volatility_window:
            return self.regime_indicator
            
        recent_returns = returns_series[-self.volatility_window:]
        volatility = recent_returns.std()
        skewness = stats.skew(recent_returns)
        
        # High volatility or negative skewness indicates stress regime
        vol_threshold = returns_series.std() * 1.5
        regime = 1.0 if (volatility > vol_threshold or skewness < -0.5) else 0.0
        
        # Smooth regime transitions
        self.regime_indicator = 0.8 * self.regime_indicator + 0.2 * regime
        return self.regime_indicator

    def update_model_weights(self, gbt_error, nn_error):
        """Dynamic model weighting based on recent performance"""
        self.performance_history['gbt'].append(gbt_error)
        self.performance_history['nn'].append(nn_error)
        
        # Keep only recent performance (last 10 predictions)
        if len(self.performance_history['gbt']) > 10:
            self.performance_history['gbt'] = self.performance_history['gbt'][-10:]
            self.performance_history['nn'] = self.performance_history['nn'][-10:]
        
        if len(self.performance_history['gbt']) >= 3:
            gbt_avg_error = np.mean(self.performance_history['gbt'])
            nn_avg_error = np.mean(self.performance_history['nn'])
            
            # Weight models inversely to their errors
            total_error = gbt_avg_error + nn_avg_error
            if total_error > 0:
                gbt_weight = nn_avg_error / total_error
                nn_weight = gbt_avg_error / total_error
                
                # Smooth weight transitions
                self.model_weights[0] = 0.7 * self.model_weights[0] + 0.3 * gbt_weight
                self.model_weights[1] = 0.7 * self.model_weights[1] + 0.3 * nn_weight
                
                # Ensure weights sum to 1
                weight_sum = sum(self.model_weights)
                self.model_weights = [w/weight_sum for w in self.model_weights]

    def train_models(self, X_train, y_train, incremental=False):
        """Train models with incremental learning support"""
        if len(X_train) == 0:
            return
            
        # Train Gradient Boosted Trees (always full retrain)
        self.gbt_model.fit(X_train, y_train)
        
        # Neural Network with proper architecture
        sample_size = len(X_train)
        
        if self.nn_model is None or not incremental:
            # Create new model
            hidden_size = min(64, max(8, sample_size // 4))
            self.nn_model = MLPRegressor(
                hidden_layer_sizes=(hidden_size, hidden_size // 2),
                activation='relu',
                solver='adam',
                alpha=0.01,  # Increased regularization
                batch_size=min(32, max(8, sample_size // 4)),
                learning_rate_init=0.001,
                max_iter=200,
                early_stopping=True,
                validation_fraction=0.1,
                random_state=42
            )
        
        # Train Neural Network
        self.nn_model.fit(X_train, y_train)

    @staticmethod
    def fetch_data(tickers, start_date, end_date):
        """Enhanced data fetching with error handling"""
        try:
            data = yf.download(tickers, start=start_date, end=end_date, auto_adjust=True)
            
            if data.empty:
                print("No data retrieved from yfinance")
                return pd.DataFrame()
            
            if len(tickers) == 1:
                # Single ticker case
                if 'Close' in data.columns:
                    adj_close = data['Close']
                else:
                    adj_close = data.iloc[:, 0]  # Take first column if Close not available
                
                returns = adj_close.pct_change().dropna()
                
                result = pd.DataFrame({
                    'date': returns.index,
                    'stock_id': tickers[0],
                    'returns': returns.values
                })
            else:
                # Multiple tickers case
                if 'Close' in data.columns.levels[0] if hasattr(data.columns, 'levels') else 'Close' in data.columns:
                    adj_close = data['Close'] if hasattr(data.columns, 'levels') else data
                else:
                    print("Close price data not found")
                    return pd.DataFrame()
                
                returns = adj_close.pct_change().dropna()
                
                # Check if returns is empty
                if returns.empty:
                    print("No return data after processing")
                    return pd.DataFrame()
                
                result_dfs = []
                for ticker in tickers:
                    if ticker in returns.columns:
                        ticker_returns = returns[ticker].dropna()
                        if len(ticker_returns) > 0:
                            ticker_data = pd.DataFrame({
                                'date': ticker_returns.index,
                                'stock_id': ticker,
                                'returns': ticker_returns.values
                            })
                            result_dfs.append(ticker_data)
                
                if not result_dfs:
                    print("No valid ticker data found")
                    return pd.DataFrame()
                
                result = pd.concat(result_dfs, ignore_index=True)
            
            # Data validation and cleaning
            if result.empty:
                print("Result dataframe is empty")
                return pd.DataFrame()
            
            # Remove extreme outliers (>5 standard deviations)
            returns_std = result['returns'].std()
            if returns_std > 0:
                result = result[np.abs(result['returns']) < 5 * returns_std]
            
            # Final cleanup
            result = result.dropna()
            
            # Ensure we have sufficient data
            if len(result) < 100:
                print(f"Insufficient data: only {len(result)} records")
                return pd.DataFrame()
            
            print(f"Successfully fetched {len(result)} records for {result['stock_id'].nunique()} stocks")
            return result
            
        except Exception as e:
            print(f"Error fetching data: {e}")
            import traceback
            traceback.print_exc()
            return pd.DataFrame()

    def preprocess_data(self, df, fit_scaler=True):
        """Enhanced feature engineering with better technical indicators"""
        if len(df) == 0:
            return np.array([]), np.array([]), pd.DataFrame()
            
        features_df = df.copy()
        
        # Initialize feature columns with NaN
        feature_columns = ['sma_3', 'sma_10', 'sma_20', 'volatility', 'momentum_3', 
                          'momentum_10', 'mean_reversion', 'regime']
        
        for col in feature_columns:
            features_df[col] = np.nan
        
        for stock in df['stock_id'].unique():
            stock_data = df[df['stock_id'] == stock].copy().sort_values('date')
            
            if len(stock_data) < 5:
                continue
                
            returns = stock_data['returns'].values
            
            # Enhanced technical indicators
            stock_data = stock_data.copy()  # Ensure we're working with a copy
            
            # Calculate features
            stock_data['sma_3'] = pd.Series(returns).rolling(3, min_periods=1).mean().values
            stock_data['sma_10'] = pd.Series(returns).rolling(10, min_periods=1).mean().values
            stock_data['sma_20'] = pd.Series(returns).rolling(20, min_periods=1).mean().values
            
            # Volatility with better handling
            vol_series = pd.Series(returns).rolling(10, min_periods=2).std()
            stock_data['volatility'] = vol_series.fillna(pd.Series(returns).std()).values
            
            # Momentum indicators
            stock_data['momentum_3'] = pd.Series(returns).rolling(3, min_periods=1).sum().values
            stock_data['momentum_10'] = pd.Series(returns).rolling(10, min_periods=1).sum().values
            
            # Mean reversion indicators
            mean_rev = (returns - stock_data['sma_10'].values) / stock_data['volatility'].values
            stock_data['mean_reversion'] = np.where(stock_data['volatility'].values > 0, mean_rev, 0)
            
            # Regime indicator (constant for all rows of this stock)
            regime_val = self.detect_regime(pd.Series(returns))
            stock_data['regime'] = regime_val
            
            # Update main dataframe using proper indexing
            stock_mask = features_df['stock_id'] == stock
            stock_indices = features_df[stock_mask].index
            
            if len(stock_indices) == len(stock_data):
                for col in feature_columns:
                    if col in stock_data.columns:
                        features_df.loc[stock_indices, col] = stock_data[col].values
        
        # Remove rows with too many missing features
        features_df = features_df.dropna(subset=feature_columns, thresh=len(feature_columns)//2)
        
        if len(features_df) == 0:
            return np.array([]), np.array([]), pd.DataFrame()
        
        # Fill remaining NaN values
        for col in feature_columns:
            features_df[col] = features_df[col].fillna(0)
        
        features = features_df[feature_columns].values
        targets = features_df['returns'].values
        
        # Robust scaling
        if len(features) > 0:
            try:
                if fit_scaler:
                    features = self.scaler.fit_transform(features)
                else:
                    features = self.scaler.transform(features)
            except Exception as e:
                print(f"Scaling error: {e}")
                return np.array([]), np.array([]), pd.DataFrame()
        
        return features, targets, features_df

    def calculate_ester(self, X, actual_returns):
        """Enhanced Ester calculation with error tracking"""
        if len(X) == 0:
            return np.array([])
            
        # Get model predictions
        gbt_pred = self.gbt_model.predict(X)
        nn_pred = self.nn_model.predict(X)
        
        # Calculate individual errors for weight updating
        gbt_error = np.mean((actual_returns - gbt_pred) ** 2)
        nn_error = np.mean((actual_returns - nn_pred) ** 2)
        
        # Update model weights
        self.update_model_weights(gbt_error, nn_error)
        
        # Combine predictions with dynamic weights
        combined_pred = (self.model_weights[0] * gbt_pred + 
                        self.model_weights[1] * nn_pred)
        
        # Calculate Ester (excess return)
        ester = actual_returns - combined_pred
        return ester

    def calculate_position_sizes(self, signals, current_vol, target_vol=None):
        """Volatility-targeted position sizing"""
        if target_vol is None:
            target_vol = self.volatility_target
            
        if current_vol <= 0:
            return signals
            
        # Scale positions based on volatility
        vol_scalar = min(2.0, target_vol / current_vol)
        
        # Apply drawdown scaling
        drawdown_scalar = max(0.1, 1.0 - self.current_drawdown / self.max_drawdown_limit)
        
        return signals * vol_scalar * drawdown_scalar

    def generate_signals(self, ester_scores, returns_history=None):
        """Enhanced signal generation with quantile-based thresholds"""
        if len(ester_scores) == 0:
            return np.array([])
            
        # Use more aggressive thresholds for signal generation
        base_threshold = 0.4  # Increased from 0.3
        regime_adj = self.regime_indicator * 0.05  # Reduced adjustment
        threshold = base_threshold + regime_adj
        
        if len(ester_scores) >= 2:
            # Use quantile-based thresholds
            long_threshold = np.quantile(ester_scores, threshold)
            short_threshold = np.quantile(ester_scores, 1 - threshold)
            
            signals = np.zeros(len(ester_scores))
            signals[ester_scores <= long_threshold] = 1
            signals[ester_scores >= short_threshold] = -1
            
            # Position sizing based on signal strength
            if len(signals) > 0:
                # Normalize signal strength
                signal_strength = np.abs(ester_scores - np.median(ester_scores))
                if signal_strength.std() > 0:
                    signal_strength = signal_strength / signal_strength.std()
                    signals = signals * np.clip(signal_strength, 0.3, 1.0)  # Increased minimum
            
            return signals
        else:
            # Fallback for small datasets - always generate some signal
            median_score = np.median(ester_scores)
            return np.where(ester_scores < median_score, 0.8, -0.8)

    def calculate_comprehensive_metrics(self, returns_df):
        """Calculate comprehensive performance metrics"""
        if len(returns_df) == 0:
            return {}
            
        returns = returns_df['return'].values
        
        # Basic metrics
        total_return = np.prod(1 + returns) - 1
        annualized_return = np.prod(1 + returns) ** (252 / len(returns)) - 1
        volatility = np.std(returns) * np.sqrt(252)
        
        # Risk metrics
        sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
        
        # Drawdown calculation
        cumulative = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = (cumulative - running_max) / running_max
        max_drawdown = np.min(drawdowns)
        
        # Additional metrics
        positive_returns = returns[returns > 0]
        negative_returns = returns[returns < 0]
        
        win_rate = len(positive_returns) / len(returns) if len(returns) > 0 else 0
        avg_win = np.mean(positive_returns) if len(positive_returns) > 0 else 0
        avg_loss = np.mean(negative_returns) if len(negative_returns) > 0 else 0
        
        profit_factor = (len(positive_returns) * avg_win) / (-len(negative_returns) * avg_loss) if avg_loss != 0 else 0
        
        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'total_trades': len(returns)
        }

    def backtest_strategy(self, data):
        """Enhanced backtesting with comprehensive risk management"""
        # Initialize results
        results = []
        current_positions = {}
        position_entry_dates = {}
        
        # Get unique dates
        dates = sorted(data['date'].unique())
        print(f"Total dates to process: {len(dates)}")
        print(f"Backtest period: {dates[0]} to {dates[-1]}")
        
        # Initialize tracking
        portfolio_value = 1.0
        returns_history = []
        debug_counter = 0
        
        # Rolling window backtest
        for i in range(self.lookback, len(dates) - self.holding):
            if i % 100 == 0:
                print(f"Processing date {i}/{len(dates)-self.holding}: {dates[i]}")
                
            try:
                current_date = dates[i]
                
                # Risk management: Check drawdown
                if self.current_drawdown > self.max_drawdown_limit:
                    print(f"Maximum drawdown exceeded at {current_date}")
                    current_positions.clear()
                    position_entry_dates.clear()
                    continue
                
                # Position management
                positions_to_close = []
                for stock, entry_date in position_entry_dates.items():
                    days_held = (pd.to_datetime(current_date) - pd.to_datetime(entry_date)).days
                    if days_held >= self.holding:
                        positions_to_close.append(stock)
                
                for stock in positions_to_close:
                    if stock in current_positions:
                        del current_positions[stock]
                        del position_entry_dates[stock]
                
                # Training data
                train_end_date = dates[i]
                train_start_date = dates[i-self.lookback]
                
                train_data = data[
                    (data['date'] >= train_start_date) & 
                    (data['date'] < train_end_date)
                ]
                
                if len(train_data) < 20:
                    if debug_counter < 5:
                        print(f"Debug: Insufficient training data at {current_date}: {len(train_data)} records")
                        debug_counter += 1
                    continue
                
                # Preprocess and train
                X_train, y_train, _ = self.preprocess_data(train_data, fit_scaler=True)
                
                if len(X_train) < 10:
                    if debug_counter < 5:
                        print(f"Debug: Insufficient processed training data at {current_date}: {len(X_train)} records")
                        debug_counter += 1
                    continue
                
                # Train models (incremental after first time)
                incremental = i > self.lookback
                self.train_models(X_train, y_train, incremental=incremental)
                
                # Current predictions
                current_data = data[data['date'] == current_date]
                
                if len(current_data) == 0:
                    if debug_counter < 5:
                        print(f"Debug: No current data at {current_date}")
                        debug_counter += 1
                    continue
                
                X_current, _, current_features_df = self.preprocess_data(current_data, fit_scaler=False)
                
                if len(X_current) == 0:
                    if debug_counter < 5:
                        print(f"Debug: No processed current data at {current_date}")
                        debug_counter += 1
                    continue
                
                # Generate signals
                actual_returns = current_features_df['returns'].values
                ester = self.calculate_ester(X_current, actual_returns)
                
                # Calculate current portfolio volatility
                current_vol = np.std(returns_history[-20:]) if len(returns_history) >= 20 else 0.02
                
                signals = self.generate_signals(ester, returns_history)
                
                # Debug first few iterations
                if debug_counter < 3:
                    print(f"Debug at {current_date}:")
                    print(f"  Ester scores: {ester}")
                    print(f"  Raw signals: {signals}")
                    print(f"  Current vol: {current_vol}")
                    debug_counter += 1
                
                # Position sizing
                signals = self.calculate_position_sizes(signals, current_vol)
                
                # Create position mapping
                stock_signals = {}
                stock_returns = {}
                
                for idx, (_, row) in enumerate(current_features_df.iterrows()):
                    if idx < len(signals):
                        stock_signals[row['stock_id']] = signals[idx]
                        stock_returns[row['stock_id']] = row['returns']
                
                # Update positions - be more aggressive about taking positions
                new_positions = 0
                for stock, signal in stock_signals.items():
                    if abs(signal) > 0.01 and stock not in current_positions:  # Lowered threshold
                        current_positions[stock] = signal
                        position_entry_dates[stock] = current_date
                        new_positions += 1
                
                # Calculate portfolio return
                if current_positions:
                    portfolio_return = 0
                    position_count = 0
                    total_weight = sum(abs(pos) for pos in current_positions.values())
                    
                    for stock, position in current_positions.items():
                        if stock in stock_returns:
                            weight = abs(position) / total_weight if total_weight > 0 else 0
                            portfolio_return += position * stock_returns[stock] * weight
                            position_count += 1
                    
                    # Apply transaction costs
                    portfolio_return -= self.transaction_cost * new_positions  # Only on new positions
                    
                    # Update portfolio tracking
                    portfolio_value *= (1 + portfolio_return)
                    returns_history.append(portfolio_return)
                    
                    # Update drawdown
                    if portfolio_value > self.peak_value:
                        self.peak_value = portfolio_value
                        self.current_drawdown = 0
                    else:
                        self.current_drawdown = (self.peak_value - portfolio_value) / self.peak_value
                    
                    results.append({
                        'date': current_date,
                        'return': portfolio_return,
                        'num_positions': position_count,
                        'portfolio_value': portfolio_value,
                        'drawdown': self.current_drawdown,
                        'regime': self.regime_indicator
                    })
                    
                    if debug_counter < 3:
                        print(f"  Generated signal! Positions: {position_count}, Return: {portfolio_return:.4f}")
                
            except Exception as e:
                print(f"Error processing date {dates[i]}: {str(e)}")
                import traceback
                traceback.print_exc()
                continue
        
        print(f"Backtest completed. Generated {len(results)} trading periods.")
        return pd.DataFrame(results)


if __name__ == "__main__":
    # Use simpler parameters for testing
    tickers = ["NVDA", "AAPL"]
    start_date = "2019-01-01"
    end_date = "2024-12-31"
    
    try:
        data = EsterStrategy.fetch_data(tickers, start_date, end_date)
        
        if data.empty:
            print("Failed to fetch data. Exiting.")
            exit(1)
        
        print(f"Fetched data shape: {data.shape}")
        print(f"Date range: {data['date'].min()} to {data['date'].max()}")
        print(f"Unique stocks: {data['stock_id'].unique()}")
        
        # Initialize strategy with more aggressive parameters
        strategy = EsterStrategy(
            lookback_period=21,  # Reduced from 30
            holding_period=10,   # Reduced from 15
            transaction_cost=0.001,  # Reduced from 0.002
            max_drawdown_limit=0.25,  # Increased from 0.20
            volatility_target=0.20    # Increased from 0.15
        )
        
        portfolio_returns = strategy.backtest_strategy(data)
        
        if len(portfolio_returns) > 0:
            # Calculate comprehensive metrics
            metrics = strategy.calculate_comprehensive_metrics(portfolio_returns)
            
            print("\n=== STRATEGY PERFORMANCE METRICS ===")
            print(f"Total Return: {metrics['total_return']:.2%}")
            print(f"Annualized Return: {metrics['annualized_return']:.2%}")
            print(f"Volatility: {metrics['volatility']:.2%}")
            print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
            print(f"Maximum Drawdown: {metrics['max_drawdown']:.2%}")
            print(f"Win Rate: {metrics['win_rate']:.2%}")
            print(f"Profit Factor: {metrics['profit_factor']:.2f}")
            print(f"Total Trades: {metrics['total_trades']}")
            
            # Additional analysis
            avg_positions = portfolio_returns['num_positions'].mean()
            regime_dist = portfolio_returns['regime'].mean()
            
            print(f"\nAverage Positions: {avg_positions:.1f}")
            print(f"Average Regime Indicator: {regime_dist:.2f}")
            
            # Show sample of results
            print(f"\nSample trading periods:")
            print(portfolio_returns.head(10))
            
        else:
            print("No valid trading signals generated")
            
    except Exception as e:
        print(f"Error running strategy: {e}")
        import traceback
        traceback.print_exc()