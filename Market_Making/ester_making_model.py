import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import yfinance as yf

class EsterStrategy:
    def __init__(self, lookback_period=21, holding_period=21):
        self.lookback = lookback_period
        self.holding = holding_period
        self.scaler = StandardScaler()
        self.gbt_model = GradientBoostingRegressor()
        self.nn_model = self._build_nn()
        self.combined_weights = [0.5, 0.5]  # Equal weights for model combination

    @staticmethod
    def fetch_data(tickers, start_date, end_date):
        """Fetch historical stock data using yfinance."""
        data = yf.download(tickers, start=start_date, end=end_date)
        # Ensure the data is in the required format
        data = data['Adj Close'].pct_change().dropna().reset_index()
        data = data.melt(id_vars=['Date'], var_name='stock_id', value_name='returns')
        data.rename(columns={'Date': 'date'}, inplace=True)
        return data
    
    def _build_nn(self):
        model = Sequential([
            Dense(256, activation='relu', input_shape=(200,)),
            Dropout(0.3),
            Dense(128, activation='relu'),
            Dropout(0.2),
            Dense(64, activation='relu'),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        return model

    def preprocess_data(self, df):
        """Process 200+ factors and calculate returns"""
        features = df.drop(['returns', 'stock_id', 'date'], axis=1)
        targets = df['returns'].shift(-self.holding)  # Future returns
        
        # Handle missing values and normalization
        features = features.fillna(method='ffill').dropna(axis=1)
        features = self.scaler.fit_transform(features)
        
        return features, targets

    def train_models(self, X_train, y_train):
        # Train Gradient Boosted Trees
        self.gbt_model.fit(X_train, y_train)
        
        # Train Neural Network
        self.nn_model.fit(X_train, y_train, 
                         epochs=50,
                         batch_size=64,
                         validation_split=0.2,
                         verbose=0)

    def calculate_ester(self, X, actual_returns):
        # Get model predictions
        gbt_pred = self.gbt_model.predict(X)
        nn_pred = self.nn_model.predict(X).flatten()
        
        # Combine predictions
        combined_pred = (self.combined_weights[0] * gbt_pred +
                        self.combined_weights[1] * nn_pred)
        
        # Calculate Ester (excess return)
        ester = actual_returns - combined_pred
        return ester

    def generate_signals(self, ester_scores):
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
        # Initialize portfolio
        portfolio = pd.DataFrame(index=data.index)
        portfolio['returns'] = 0
        
        # Rolling window backtest
        for i in range(self.lookback, len(data)-self.holding):
            # Training period
            train_data = data.iloc[i-self.lookback:i]
            X_train, y_train = self.preprocess_data(train_data)
            
            # Train models
            self.train_models(X_train, y_train)
            
            # Current universe for prediction
            current_data = data.iloc[i]
            X_current, _ = self.preprocess_data(current_data)
            actual_returns = current_data['returns']
            
            # Calculate Ester and generate signals
            ester = self.calculate_ester(X_current, actual_returns)
            signals = self.generate_signals(ester)
            
            # Calculate strategy returns
            portfolio.iloc[i+self.holding] = (
                signals * data['returns'].iloc[i+self.holding]
            ).mean()
        
        return portfolio

if __name__ == "__main__":
    # Fetch data using yfinance
    tickers = "AAPL MSFT GOOGL AMZN"  # Example tickers
    start_date = "2020-01-01"
    end_date = "2023-12-31"
    data = EsterStrategy.fetch_data(tickers, start_date, end_date)
    
    # Initialize and backtest the strategy
    strategy = EsterStrategy()
    portfolio_returns = strategy.backtest_strategy(data)
    
    # Calculate performance metrics
    cumulative_returns = (1 + portfolio_returns).cumprod()
    annualized_return = np.prod(1 + portfolio_returns)**(252/len(portfolio_returns)) - 1
    
    print(f"Strategy Annualized Return: {annualized_return:.2%}")
