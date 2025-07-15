import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import yfinance as yf

class EsterStrategy:
    def __init__(self, lookback_period=21, holding_period=21):
        self.lookback = lookback_period
        self.holding = holding_period
        self.scaler = StandardScaler()
        self.gbt_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
        self.nn_model = None  # Initialize as None, will be created dynamically
        self.combined_weights = [0.5, 0.5]  # Equal weights for model combination

    def train_models(self, X_train, y_train):
        if len(X_train) > 0:
            # Train Gradient Boosted Trees
            self.gbt_model.fit(X_train, y_train)
            
            # Create neural network with adaptive batch size
            sample_size = len(X_train)
            batch_size = min(32, max(1, sample_size))  # Simplified batch size
            
            self.nn_model = MLPRegressor(
                hidden_layer_sizes=(64, 32),  # Smaller network
                activation='relu',
                solver='adam',
                alpha=0.001,
                batch_size=batch_size,
                learning_rate_init=0.001,
                max_iter=100,  # Reduced iterations
                random_state=42
            )
            
            # Train Neural Network
            self.nn_model.fit(X_train, y_train)

    @staticmethod
    def fetch_data(tickers, start_date, end_date):
        """Fetch historical stock data using yfinance."""
        data = yf.download(tickers, start=start_date, end=end_date)
        
        # Handle single vs multiple tickers
        if len(tickers) == 1:
            # Single ticker case
            adj_close = data['Close']
            returns = adj_close.pct_change().dropna()
            result = pd.DataFrame({
                'date': returns.index,
                'stock_id': tickers[0],
                'returns': returns.values
            })
        else:
            # Multiple tickers case
            adj_close = data['Close']
            returns = adj_close.pct_change().dropna()
            
            # Melt the dataframe to long format
            result = returns.reset_index().melt(
                id_vars=['Date'], 
                var_name='stock_id', 
                value_name='returns'
            )
            result.rename(columns={'Date': 'date'}, inplace=True)
        
        # Remove any remaining NaN values
        result = result.dropna()
        return result
    
    def preprocess_data(self, df, fit_scaler=True):
        """Process features and calculate returns"""
        # Create simple features from returns data
        features_df = df.copy()
        
        # Add basic technical indicators as features
        for stock in df['stock_id'].unique():
            stock_data = df[df['stock_id'] == stock].copy().sort_values('date')
            
            # Rolling statistics - use shorter windows to preserve more data
            stock_data['sma_3'] = stock_data['returns'].rolling(3, min_periods=1).mean()
            stock_data['sma_5'] = stock_data['returns'].rolling(5, min_periods=1).mean()
            stock_data['volatility'] = stock_data['returns'].rolling(5, min_periods=1).std()
            stock_data['momentum'] = stock_data['returns'].rolling(3, min_periods=1).sum()
            
            # Fill NaN values with 0 for volatility
            stock_data['volatility'] = stock_data['volatility'].fillna(0)
            
            # Update features_df
            mask = features_df['stock_id'] == stock
            features_df.loc[mask, 'sma_3'] = stock_data['sma_3'].values
            features_df.loc[mask, 'sma_5'] = stock_data['sma_5'].values
            features_df.loc[mask, 'volatility'] = stock_data['volatility'].values
            features_df.loc[mask, 'momentum'] = stock_data['momentum'].values
        
        # Only drop rows where all feature columns are NaN
        feature_columns = ['sma_3', 'sma_5', 'volatility', 'momentum']
        features_df = features_df.dropna(subset=feature_columns)
        
        # Prepare features and targets
        features = features_df[feature_columns]
        targets = features_df['returns']
        
        # Normalize features - fix data leakage
        if len(features) > 0:
            if fit_scaler:
                features = self.scaler.fit_transform(features)
            else:
                features = self.scaler.transform(features)
        
        return features, targets, features_df

    def calculate_ester(self, X, actual_returns):
        if len(X) == 0:
            return np.array([])
            
        # Get model predictions
        gbt_pred = self.gbt_model.predict(X)
        nn_pred = self.nn_model.predict(X)
        
        # Combine predictions
        combined_pred = (self.combined_weights[0] * gbt_pred +
                        self.combined_weights[1] * nn_pred)
        
        # Calculate Ester (excess return)
        ester = actual_returns - combined_pred
        return ester

    def generate_signals(self, ester_scores):
        if len(ester_scores) == 0:
            return np.array([])
            
        # Use more relaxed thresholds for signal generation
        if len(ester_scores) >= 2:
            # Sort scores and take extremes
            sorted_indices = np.argsort(ester_scores)
            signals = np.zeros(len(ester_scores))
            
            # Take bottom 30% as longs, top 30% as shorts
            n_long = max(1, int(len(ester_scores) * 0.3))
            n_short = max(1, int(len(ester_scores) * 0.3))
            
            signals[sorted_indices[:n_long]] = 1   # Long positions
            signals[sorted_indices[-n_short:]] = -1  # Short positions
            
            return signals
        else:
            # If only one or two stocks, use median split
            median_score = np.median(ester_scores)
            signals = np.where(ester_scores < median_score, 1, -1)
            return signals

    def backtest_strategy(self, data):
        # Initialize results
        results = []
        current_positions = {}
        position_entry_dates = {}
        
        # Get unique dates for time series analysis
        dates = sorted(data['date'].unique())
        print(f"Total dates to process: {len(dates)}")
        print(f"Backtest period: {dates[0]} to {dates[-1]}")
        
        # Rolling window backtest
        for i in range(self.lookback, len(dates)-self.holding):
            if i % 50 == 0:  # Progress indicator
                print(f"Processing date {i}/{len(dates)-self.holding}: {dates[i]}")
                
            try:
                current_date = dates[i]
                
                # Check if we need to close existing positions (holding period expired)
                positions_to_close = []
                for stock, entry_date in position_entry_dates.items():
                    days_held = (pd.to_datetime(current_date) - pd.to_datetime(entry_date)).days
                    if days_held >= self.holding:
                        positions_to_close.append(stock)
                
                # Close expired positions
                for stock in positions_to_close:
                    if stock in current_positions:
                        del current_positions[stock]
                        del position_entry_dates[stock]
                
                # Training period
                train_end_date = dates[i]
                train_start_date = dates[i-self.lookback]
                
                train_data = data[
                    (data['date'] >= train_start_date) & 
                    (data['date'] < train_end_date)
                ]
                
                if len(train_data) < 10:  # Reduced minimum data requirement
                    continue
                    
                X_train, y_train, _ = self.preprocess_data(train_data, fit_scaler=True)
                
                if len(X_train) == 0:
                    continue
                
                # Train models
                self.train_models(X_train, y_train)
                
                # Current data for prediction
                current_data = data[data['date'] == current_date]
                
                if len(current_data) == 0:
                    continue
                
                X_current, _, current_features_df = self.preprocess_data(current_data, fit_scaler=False)
                
                if len(X_current) == 0:
                    continue
                
                # Calculate Ester and generate signals
                actual_returns = current_features_df['returns'].values
                ester = self.calculate_ester(X_current, actual_returns)
                signals = self.generate_signals(ester)
                
                # Debug: Print signal generation info
                if i == self.lookback:  # First iteration
                    print(f"Debug - Ester scores: {ester[:5] if len(ester) > 5 else ester}")
                    print(f"Debug - Signals: {signals[:5] if len(signals) > 5 else signals}")
                    print(f"Debug - Non-zero signals: {np.sum(signals != 0)}")
                
                # Create signal mapping by stock
                stock_signals = {}
                stock_returns = {}
                for idx, (_, row) in enumerate(current_features_df.iterrows()):
                    if idx < len(signals):
                        stock_signals[row['stock_id']] = signals[idx]
                        stock_returns[row['stock_id']] = row['returns']
                
                # Update positions based on signals
                for stock, signal in stock_signals.items():
                    if signal != 0 and stock not in current_positions:
                        current_positions[stock] = signal
                        position_entry_dates[stock] = current_date
                
                # Calculate portfolio return for this period
                if current_positions:
                    portfolio_return = 0
                    position_count = 0
                    
                    for stock, position in current_positions.items():
                        if stock in stock_returns:
                            portfolio_return += position * stock_returns[stock]
                            position_count += 1
                    
                    if position_count > 0:
                        portfolio_return /= position_count  # Equal weight portfolio
                        
                        results.append({
                            'date': current_date,
                            'return': portfolio_return,
                            'num_positions': position_count
                        })
                
            except Exception as e:
                print(f"Error processing date {dates[i]}: {str(e)}")
                continue
        
        print(f"Backtest completed. Generated {len(results)} trading signals.")
        return pd.DataFrame(results)


if __name__ == "__main__": 
    tickers = ["NVDA", "PLTR"]  # Fix: Use list instead of string
    start_date = "2020-01-01"
    end_date = "2024-12-31"
    
    try:
        data = EsterStrategy.fetch_data(tickers, start_date, end_date)
        print(f"Fetched data shape: {data.shape}")
        print(f"Date range: {data['date'].min()} to {data['date'].max()}")
        print(f"Unique stocks: {data['stock_id'].unique()}")
        
        # Initialize and backtest the strategy
        strategy = EsterStrategy()
        portfolio_returns = strategy.backtest_strategy(data)
        
        if len(portfolio_returns) > 0:
            # Calculate performance metrics
            returns = portfolio_returns['return']
            cumulative_returns = (1 + returns).cumprod()
            annualized_return = np.prod(1 + returns)**(252/len(returns)) - 1
            
            print(f"Strategy Annualized Return: {annualized_return:.2%}")
            print(f"Total trades: {len(portfolio_returns)}")
            print(f"Average positions per trade: {portfolio_returns['num_positions'].mean():.1f}")
        else:
            print("No valid trading signals generated")
            
    except Exception as e:
        print(f"Error running strategy: {e}")