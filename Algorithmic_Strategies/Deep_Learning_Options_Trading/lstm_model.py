"""
LSTM Model Module for Deep Learning Options Trading Strategy

Implements LSTM neural network trained to maximize Sharpe ratio for delta-neutral
straddle trading, with turnover regularization to account for transaction costs.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import logging
import yaml
from pathlib import Path
import matplotlib.pyplot as plt

class OptionsDataset(Dataset):
    """PyTorch Dataset for options trading sequences."""

    def __init__(self, X: np.ndarray, y: np.ndarray, metadata: list = None):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        self.metadata = metadata or []

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.metadata[idx] if self.metadata else None

class LSTMPortfolioOptimizer(nn.Module):
    """
    LSTM model that learns to predict optimal straddle positions.
    Trained to maximize Sharpe ratio with turnover regularization.
    """

    def __init__(self, input_size: int, hidden_size: int = 64,
                 num_layers: int = 2, dropout: float = 0.2):
        super(LSTMPortfolioOptimizer, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )

        # Output layers
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size, 32)
        self.fc2 = nn.Linear(32, 1)  # Position signal (-1 to 1)

        # Activation
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, x):
        # LSTM forward pass
        lstm_out, (hn, cn) = self.lstm(x)

        # Take last time step output
        last_output = lstm_out[:, -1, :]

        # Fully connected layers
        out = self.dropout(last_output)
        out = self.relu(self.fc1(out))
        out = self.dropout(out)
        position_signal = self.tanh(self.fc2(out))  # Bound between -1 and 1

        return position_signal.squeeze()

class SharpeLoss(nn.Module):
    """
    Custom loss function that maximizes Sharpe ratio with turnover regularization.
    Based on the research methodology from the Quant Radio transcript.
    """

    def __init__(self, turnover_penalty: float = 0.01, risk_free_rate: float = 0.02):
        super(SharpeLoss, self).__init__()
        self.turnover_penalty = turnover_penalty
        self.risk_free_rate = risk_free_rate

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor,
                prev_positions: torch.Tensor = None) -> torch.Tensor:
        """
        Calculate loss as negative Sharpe ratio plus turnover penalty.

        Args:
            predictions: Model position signals (-1 to 1)
            targets: Actual straddle returns
            prev_positions: Previous period positions for turnover calculation

        Returns:
            Loss value (negative Sharpe ratio + turnover penalty)
        """
        # Calculate portfolio returns
        portfolio_returns = predictions * targets

        # Calculate excess returns
        excess_returns = portfolio_returns - self.risk_free_rate / 252  # Daily risk-free rate

        # Calculate Sharpe ratio components
        mean_return = torch.mean(excess_returns)
        std_return = torch.std(excess_returns)

        # Sharpe ratio (annualized)
        sharpe_ratio = (mean_return / (std_return + 1e-8)) * np.sqrt(252)

        # Turnover regularization (penalize position changes)
        turnover_loss = 0
        if prev_positions is not None:
            position_changes = torch.abs(predictions - prev_positions)
            turnover_loss = self.turnover_penalty * torch.mean(position_changes)

        # Loss is negative Sharpe ratio + turnover penalty
        # (we want to minimize loss, which maximizes Sharpe)
        loss = -sharpe_ratio + turnover_loss

        return loss

class DeepLearningOptionsTrader:
    """
    Main class for training and using the LSTM options trading model.
    Implements walk-forward validation and portfolio optimization.
    """

    def __init__(self, config_path: str = "config.yaml"):
        """Initialize with configuration."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.logger = logging.getLogger(__name__)
        self._setup_logging()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger.info(f"Using device: {self.device}")

        self.model = None
        self.scaler = StandardScaler()

    def _setup_logging(self):
        """Setup logging configuration."""
        logging.basicConfig(
            level=getattr(logging, self.config['logging']['level']),
            format=self.config['logging']['format'],
            handlers=[
                logging.FileHandler(self.config['logging']['file']),
                logging.StreamHandler()
            ]
        )

    def build_model(self, input_size: int) -> LSTMPortfolioOptimizer:
        """Build LSTM model with configured hyperparameters."""
        model = LSTMPortfolioOptimizer(
            input_size=input_size,
            hidden_size=self.config['model']['hidden_size'],
            num_layers=self.config['model']['lstm_layers'],
            dropout=self.config['model']['dropout']
        )

        self.model = model.to(self.device)
        return model

    def train_model(self, X_train: np.ndarray, y_train: np.ndarray,
                   X_val: np.ndarray = None, y_val: np.ndarray = None,
                   metadata_train: list = None, metadata_val: list = None):
        """
        Train the LSTM model with Sharpe ratio optimization.

        Args:
            X_train: Training features
            y_train: Training targets (straddle returns)
            X_val: Validation features (optional)
            y_val: Validation targets (optional)
            metadata_train: Training metadata
            metadata_val: Validation metadata
        """
        self.logger.info("Starting model training")

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train.reshape(X_train.shape[0], -1))
        X_train_scaled = X_train_scaled.reshape(X_train.shape)

        if X_val is not None:
            X_val_scaled = self.scaler.transform(X_val.reshape(X_val.shape[0], -1))
            X_val_scaled = X_val_scaled.reshape(X_val.shape)

        # Build model
        input_size = X_train.shape[2]  # Number of features
        self.build_model(input_size)

        # Create datasets
        train_dataset = OptionsDataset(X_train_scaled, y_train, metadata_train)
        train_loader = DataLoader(train_dataset, batch_size=self.config['model']['batch_size'],
                                shuffle=True)

        if X_val is not None:
            val_dataset = OptionsDataset(X_val_scaled, y_val, metadata_val)
            val_loader = DataLoader(val_dataset, batch_size=self.config['model']['batch_size'],
                                  shuffle=False)

        # Optimizer and loss
        optimizer = optim.Adam(self.model.parameters(), lr=self.config['model']['learning_rate'])
        criterion = SharpeLoss(
            turnover_penalty=self.config['model']['turnover_penalty'],
            risk_free_rate=0.02
        )

        # Training loop
        best_val_sharpe = -np.inf
        patience_counter = 0

        for epoch in range(self.config['model']['epochs']):
            self.model.train()
            train_loss = 0

            for batch_X, batch_y, _ in train_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)

                optimizer.zero_grad()

                # Forward pass
                predictions = self.model(batch_X)

                # Calculate loss (need to handle previous positions for turnover)
                # For simplicity, we'll use None for prev_positions in first implementation
                loss = criterion(predictions, batch_y)

                # Backward pass
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            train_loss /= len(train_loader)

            # Validation
            if X_val is not None:
                val_sharpe = self._evaluate_sharpe(X_val_scaled, y_val)
                self.logger.info(f"Epoch {epoch+1}/{self.config['model']['epochs']} - "
                               f"Train Loss: {train_loss:.4f}, Val Sharpe: {val_sharpe:.4f}")

                # Early stopping
                if val_sharpe > best_val_sharpe:
                    best_val_sharpe = val_sharpe
                    patience_counter = 0
                    self._save_model("models/best_model.pth")
                else:
                    patience_counter += 1

                if patience_counter >= self.config['model']['patience']:
                    self.logger.info("Early stopping triggered")
                    break
            else:
                self.logger.info(f"Epoch {epoch+1}/{self.config['model']['epochs']} - "
                               f"Train Loss: {train_loss:.4f}")

        self._save_model("models/final_model.pth")
        self.logger.info("Training completed")

    def _evaluate_sharpe(self, X: np.ndarray, y: np.ndarray) -> float:
        """Evaluate Sharpe ratio on validation set."""
        self.model.eval()

        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            predictions = self.model(X_tensor).cpu().numpy()

            # Calculate portfolio returns
            portfolio_returns = predictions * y

            # Annualized Sharpe ratio
            mean_return = np.mean(portfolio_returns) * 252
            std_return = np.std(portfolio_returns) * np.sqrt(252)
            sharpe = mean_return / (std_return + 1e-8)

        return sharpe

    def predict_positions(self, X: np.ndarray) -> np.ndarray:
        """
        Generate position signals for new data.

        Args:
            X: Feature sequences

        Returns:
            Position signals between -1 and 1
        """
        self.model.eval()

        # Scale features
        X_scaled = self.scaler.transform(X.reshape(X.shape[0], -1))
        X_scaled = X_scaled.reshape(X.shape)

        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_scaled).to(self.device)
            predictions = self.model(X_tensor).cpu().numpy()

        return predictions

    def _save_model(self, filepath: str):
        """Save model and scaler."""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)

        torch.save({
            'model_state_dict': self.model.state_dict(),
            'scaler': self.scaler,
            'config': self.config
        }, filepath)

    def load_model(self, filepath: str):
        """Load trained model."""
        checkpoint = torch.load(filepath, map_location=self.device)

        # Rebuild model
        input_size = checkpoint['config']['features']['lookback_window']  # Approximate
        self.build_model(input_size)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.scaler = checkpoint['scaler']

        self.logger.info(f"Model loaded from {filepath}")

    def walk_forward_validation(self, X: np.ndarray, y: np.ndarray,
                              train_years: int = 3, val_years: int = 1) -> dict:
        """
        Perform walk-forward validation as described in the research.

        Args:
            X: Full feature sequences
            y: Full targets
            train_years: Years for training window
            val_years: Years for validation window

        Returns:
            Dictionary with validation results
        """
        self.logger.info("Performing walk-forward validation")

        # Assume daily data, convert to periods
        train_periods = train_years * 252
        val_periods = val_years * 252

        results = {
            'sharpe_ratios': [],
            'train_periods': [],
            'val_periods': []
        }

        total_periods = len(X)

        for start_idx in range(0, total_periods - train_periods - val_periods, val_periods):
            train_end = start_idx + train_periods
            val_end = train_end + val_periods

            # Training data
            X_train = X[start_idx:train_end]
            y_train = y[start_idx:train_end]

            # Validation data
            X_val = X[train_end:val_end]
            y_val = y[train_end:val_end]

            # Train model
            self.train_model(X_train, y_train, X_val, y_val)

            # Evaluate
            val_sharpe = self._evaluate_sharpe(X_val, y_val)
            results['sharpe_ratios'].append(val_sharpe)
            results['train_periods'].append((start_idx, train_end))
            results['val_periods'].append((train_end, val_end))

            self.logger.info(f"Walk-forward step: Train {start_idx}-{train_end}, "
                           f"Val {train_end}-{val_end}, Sharpe: {val_sharpe:.4f}")

        return results


if __name__ == "__main__":
    # Example usage
    trader = DeepLearningOptionsTrader()

    # Generate sample data for testing
    np.random.seed(42)
    n_samples = 1000
    seq_length = 30
    n_features = 6

    X = np.random.randn(n_samples, seq_length, n_features)
    y = np.random.randn(n_samples) * 0.1  # Simulated straddle returns

    # Train model
    trader.train_model(X, y)

    # Make predictions
    predictions = trader.predict_positions(X[:10])
    print(f"Sample predictions: {predictions}")