"""
Neural Network Model for Momentum Stock Ranking

Implements a Feed-Forward Neural Network with bottleneck layer for learning
compressed representations of momentum signals.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict
import numpy as np


class MomentumRankerModel(nn.Module):
    """
    Feed-Forward Neural Network for ranking stocks by momentum.
    
    Key Feature: Bottleneck layer (e.g., 4 units) forces the network
    to learn a compressed, efficient representation of the input features.
    """
    
    def __init__(
        self,
        input_size: int = 33,
        hidden_layers: List[int] = [64, 4, 32, 16],
        dropout: float = 0.3
    ):
        """
        Initialize the model architecture.
        
        Parameters:
        -----------
        input_size : int
            Number of input features (default: 33)
        hidden_layers : List[int]
            List of hidden layer sizes
            Example: [64, 4, 32, 16] creates 4 hidden layers with bottleneck at layer 2
        dropout : float
            Dropout probability for regularization
        """
        super(MomentumRankerModel, self).__init__()
        
        self.input_size = input_size
        self.hidden_layers_sizes = hidden_layers
        self.dropout_prob = dropout
        
        # Build network layers
        layers = []
        prev_size = input_size
        
        for i, hidden_size in enumerate(hidden_layers):
            # Linear layer
            layers.append(nn.Linear(prev_size, hidden_size))
            
            # Batch normalization
            layers.append(nn.BatchNorm1d(hidden_size))
            
            # Activation
            layers.append(nn.ReLU())
            
            # Dropout (except for bottleneck layer)
            if dropout > 0 and hidden_size > 4:  # Skip dropout for very small layers
                layers.append(nn.Dropout(dropout))
            
            prev_size = hidden_size
        
        # Output layer (binary classification)
        layers.append(nn.Linear(prev_size, 1))
        layers.append(nn.Sigmoid())  # Probability output [0, 1]
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        """
        Forward pass through the network.
        
        Parameters:
        -----------
        x : torch.Tensor
            Input tensor of shape (batch_size, input_size)
            
        Returns:
        --------
        torch.Tensor : Output probabilities of shape (batch_size, 1)
        """
        return self.network(x)
    
    def predict_proba(self, x):
        """
        Predict probabilities for input data.
        
        Parameters:
        -----------
        x : torch.Tensor or np.ndarray
            Input data
            
        Returns:
        --------
        np.ndarray : Predicted probabilities
        """
        self.eval()
        
        with torch.no_grad():
            if isinstance(x, np.ndarray):
                x = torch.FloatTensor(x)
            
            if torch.cuda.is_available():
                x = x.cuda()
            
            probs = self.forward(x)
            
            return probs.cpu().numpy().flatten()
    
    def get_bottleneck_representation(self, x):
        """
        Extract the compressed representation from the bottleneck layer.
        
        Useful for visualization and understanding what the model learns.
        
        Parameters:
        -----------
        x : torch.Tensor
            Input tensor
            
        Returns:
        --------
        torch.Tensor : Bottleneck activations
        """
        self.eval()
        
        with torch.no_grad():
            # Find the bottleneck layer (smallest hidden layer)
            bottleneck_idx = self.hidden_layers_sizes.index(min(self.hidden_layers_sizes))
            
            # Calculate layer index in sequential (accounting for BN, ReLU, Dropout)
            layer_idx = bottleneck_idx * 4 + 2  # After BN and ReLU
            
            # Forward pass up to bottleneck
            for i, layer in enumerate(self.network):
                x = layer(x)
                if i == layer_idx:
                    return x
        
        return None


def create_model(config: Dict) -> MomentumRankerModel:
    """
    Create model from configuration.
    
    Parameters:
    -----------
    config : Dict
        Configuration dictionary
        
    Returns:
    --------
    MomentumRankerModel : Initialized model
    """
    model = MomentumRankerModel(
        input_size=config['model']['input_size'],
        hidden_layers=config['model']['hidden_layers'],
        dropout=config['model']['dropout']
    )
    
    # Move to GPU if available
    if torch.cuda.is_available():
        model = model.cuda()
        print("Model moved to GPU")
    
    return model


def count_parameters(model: nn.Module) -> int:
    """
    Count trainable parameters in the model.
    
    Parameters:
    -----------
    model : nn.Module
        PyTorch model
        
    Returns:
    --------
    int : Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_model_summary(model: MomentumRankerModel):
    """
    Print detailed model summary.
    
    Parameters:
    -----------
    model : MomentumRankerModel
        Model to summarize
    """
    print("="*80)
    print("MODEL ARCHITECTURE")
    print("="*80)
    print(f"\nInput Size: {model.input_size}")
    print(f"Hidden Layers: {model.hidden_layers_sizes}")
    print(f"Dropout: {model.dropout_prob}")
    print(f"\nTotal Parameters: {count_parameters(model):,}")
    print("\nLayer Details:")
    print("-"*80)
    
    for i, layer in enumerate(model.network):
        print(f"  Layer {i}: {layer}")
    
    print("="*80)


class EarlyStopping:
    """
    Early stopping to prevent overfitting.
    
    Stops training when validation loss stops improving.
    """
    
    def __init__(self, patience: int = 10, min_delta: float = 0.0001):
        """
        Initialize early stopping.
        
        Parameters:
        -----------
        patience : int
            Number of epochs to wait before stopping
        min_delta : float
            Minimum change in loss to qualify as improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
    
    def __call__(self, val_loss: float) -> bool:
        """
        Check if training should stop.
        
        Parameters:
        -----------
        val_loss : float
            Current validation loss
            
        Returns:
        --------
        bool : True if training should stop
        """
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
        
        return self.early_stop


if __name__ == "__main__":
    import yaml
    
    # Load configuration
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Create and summarize model
    model = create_model(config)
    print_model_summary(model)
    
    # Test forward pass
    batch_size = 32
    input_size = config['model']['input_size']
    
    dummy_input = torch.randn(batch_size, input_size)
    if torch.cuda.is_available():
        dummy_input = dummy_input.cuda()
    
    output = model(dummy_input)
    print(f"\nTest forward pass:")
    print(f"  Input shape: {dummy_input.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Output range: [{output.min():.4f}, {output.max():.4f}]")
