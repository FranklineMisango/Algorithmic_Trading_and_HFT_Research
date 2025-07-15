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
    
    def preprocess_data(self, df):
        """Process features and calculate returns"""
        # Create simple features from returns data
        features_df = df.copy()
        
        # Add basic technical indicators as features
        for stock in df['stock_id'].unique():
            stock_data = df[df['stock_id'] == stock].copy()
            
            # Rolling statistics
            stock_data['sma_5'] = stock_data['returns'].rolling(5).mean()
            stock_data['sma_10'] = stock_data['returns'].rolling(10).mean()
            stock_data['volatility'] = stock_data['returns'].rolling(10).std()
            stock_data['momentum'] = stock_data['returns'].rolling(5).sum()
            
            # Update features_df
            features_df.loc[features_df['stock_id'] == stock, 'sma_5'] = stock_data['sma_5']
            features_df.loc[features_df['stock_id'] == stock, 'sma_10'] = stock_data['sma_10']
            features_df.loc[features_df['stock_id'] == stock, 'volatility'] = stock_data['volatility']
            features_df.loc[features_df['stock_id'] == stock, 'momentum'] = stock_data['momentum']
        
        # Drop rows with NaN values
        features_df = features_df.dropna()
        
        # Prepare features and targets
        feature_columns = ['sma_5', 'sma_10', 'volatility', 'momentum']
        features = features_df[feature_columns]
        targets = features_df['returns']
        
        # Normalize features
        if len(features) > 0:
            features = self.scaler.fit_transform(features)
        
        return features, targets

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
            
        # Rank stocks by Ester scores
        ranked = pd.Series(ester_scores).rank(pct=True)
        
        # Long bottom decile, short top decile
        long_threshold = 0.1
        short_threshold = 0.9
        
        signals = np.zeros(len(ranked))
        signals[ranked <= long_threshold] = 1    # Buy signals
        signals[ranked >= short_threshold] = -1  # Sell signals
        return signals

    def backtest_strategy(self, data):
        # Initialize results
        results = []
        
        # Get unique dates for time series analysis
        dates = sorted(data['date'].unique())
        print(f"Total dates to process: {len(dates)}")
        print(f"Backtest period: {dates[0]} to {dates[-1]}")
        
        # Rolling window backtest
        for i in range(self.lookback, len(dates)-self.holding):
            if i % 50 == 0:  # Progress indicator
                print(f"Processing date {i}/{len(dates)-self.holding}: {dates[i]}")
                
            try:
                # Training period
                train_end_date = dates[i]
                train_start_date = dates[i-self.lookback]
                
                train_data = data[
                    (data['date'] >= train_start_date) & 
                    (data['date'] < train_end_date)
                ]
                
                if len(train_data) < 20:  # Increased minimum data requirement
                    continue
                    
                X_train, y_train = self.preprocess_data(train_data)
                
                if len(X_train) == 0:
                    continue
                
                # Train models
                self.train_models(X_train, y_train)
                
                # Current data for prediction
                current_date = dates[i]
                current_data = data[data['date'] == current_date]
                
                if len(current_data) == 0:
                    continue
                
                X_current, _ = self.preprocess_data(current_data)
                actual_returns = current_data['returns'].values
                
                if len(X_current) == 0:
                    continue
                
                # Calculate Ester and generate signals
                ester = self.calculate_ester(X_current, actual_returns)
                signals = self.generate_signals(ester)
                
                # Calculate strategy return
                if len(signals) > 0:
                    strategy_return = np.mean(signals * actual_returns)
                    results.append({
                        'date': current_date,
                        'return': strategy_return
                    })
                
            except Exception as e:
                print(f"Error processing date {dates[i]}: {e}")
                continue
        
        print(f"Backtest completed. Generated {len(results)} trading signals.")
        return pd.DataFrame(results)


if __name__ == "__main__": 
    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN"]  # Fix: Use list instead of string
    start_date = "2020-01-01"
    end_date = "2023-12-31"
    
    try:
        data = EsterStrategy.fetch_data(tickers, start_date, end_date)
        
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
        else:
            print("No valid trading signals generated")
            
    except Exception as e:
        print(f"Error running strategy: {e}")