"""
Sentiment Model: Market-Labeled BERT for Financial Sentiment

Fine-tunes BERT using abnormal returns as labels.
"""

import pandas as pd
import numpy as np
import yaml
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from sklearn.model_selection import TimeSeriesSplit
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')


class FinancialTextDataset(Dataset):
    """Dataset for financial text with abnormal return labels."""
    
    def __init__(self, texts: List[str], labels: np.ndarray, tokenizer, max_length: int = 512):
        """Initialize dataset."""
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = float(self.labels[idx])
        
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.float)
        }


class SmartyBERT:
    """Market-labeled BERT for sentiment prediction."""
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize model."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.model_config = self.config['model']
        
        # Initialize tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(self.model_config['variant'])
        
        # Initialize model (regression task)
        self.model = BertForSequenceClassification.from_pretrained(
            self.model_config['variant'],
            num_labels=1  # Regression output
        )
        
        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        print(f"Model initialized on {self.device}")
    
    def construct_labels(
        self,
        stock_returns: pd.Series,
        market_returns: pd.Series,
        beta: float = 1.0
    ) -> np.ndarray:
        """
        Construct market-labeled targets.
        
        Label = AbnormalReturn(t+1) = Return(t+1) - Beta * MarketReturn(t+1)
        
        Args:
            stock_returns: Stock returns (t+1)
            market_returns: Market returns (t+1)
            beta: Stock beta
        
        Returns:
            Abnormal returns array
        """
        # Calculate abnormal returns
        abnormal_returns = stock_returns - beta * market_returns
        
        # Winsorize
        lower = np.percentile(abnormal_returns, 1)
        upper = np.percentile(abnormal_returns, 99)
        abnormal_returns = np.clip(abnormal_returns, lower, upper)
        
        return abnormal_returns.values
    
    def train_model(
        self,
        texts: List[str],
        labels: np.ndarray,
        validation_split: float = 0.2
    ) -> Dict:
        """
        Fine-tune BERT on market-labeled data.
        
        Args:
            texts: List of text corpuses
            labels: Abnormal returns (market-labeled)
            validation_split: Validation set proportion
        
        Returns:
            Training history
        """
        # Split data
        split_idx = int(len(texts) * (1 - validation_split))
        train_texts = texts[:split_idx]
        train_labels = labels[:split_idx]
        val_texts = texts[split_idx:]
        val_labels = labels[split_idx:]
        
        # Create datasets
        train_dataset = FinancialTextDataset(train_texts, train_labels, self.tokenizer)
        val_dataset = FinancialTextDataset(val_texts, val_labels, self.tokenizer)
        
        # Create dataloaders
        batch_size = self.model_config['finetuning']['hyperparameters']['batch_size']
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Optimizer
        optimizer = AdamW(
            self.model.parameters(),
            lr=self.model_config['finetuning']['hyperparameters']['learning_rate'],
            weight_decay=self.model_config['finetuning']['hyperparameters']['weight_decay']
        )
        
        # Scheduler
        epochs = self.model_config['finetuning']['hyperparameters']['epochs']
        total_steps = len(train_loader) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.model_config['finetuning']['hyperparameters']['warmup_steps'],
            num_training_steps=total_steps
        )
        
        # Training loop
        history = {'train_loss': [], 'val_loss': [], 'val_mse': []}
        
        print(f"\nTraining for {epochs} epochs...")
        
        for epoch in range(epochs):
            # Train
            self.model.train()
            train_loss = 0
            
            for batch in train_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device).unsqueeze(1)
                
                optimizer.zero_grad()
                
                outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.model_config['finetuning']['hyperparameters']['max_grad_norm']
                )
                optimizer.step()
                scheduler.step()
                
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            
            # Validate
            self.model.eval()
            val_loss = 0
            val_mse = 0
            
            with torch.no_grad():
                for batch in val_loader:
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    labels = batch['label'].to(self.device).unsqueeze(1)
                    
                    outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
                    loss = outputs.loss
                    
                    val_loss += loss.item()
                    val_mse += torch.mean((outputs.logits - labels) ** 2).item()
            
            val_loss /= len(val_loader)
            val_mse /= len(val_loader)
            
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['val_mse'].append(val_mse)
            
            print(f"Epoch {epoch+1}/{epochs}: Train Loss={train_loss:.6f}, Val Loss={val_loss:.6f}, Val MSE={val_mse:.6f}")
        
        return history
    
    def predict(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Generate sentiment scores for texts.
        
        Args:
            texts: List of text corpuses
            batch_size: Batch size for inference
        
        Returns:
            Sentiment scores (continuous)
        """
        dataset = FinancialTextDataset(
            texts,
            np.zeros(len(texts)),  # Dummy labels
            self.tokenizer
        )
        
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        self.model.eval()
        predictions = []
        
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                outputs = self.model(input_ids, attention_mask=attention_mask)
                predictions.extend(outputs.logits.squeeze().cpu().numpy())
        
        return np.array(predictions)
    
    def save_model(self, path: str):
        """Save model checkpoint."""
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        print(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """Load model checkpoint."""
        self.model = BertForSequenceClassification.from_pretrained(path)
        self.tokenizer = BertTokenizer.from_pretrained(path)
        self.model.to(self.device)
        print(f"Model loaded from {path}")


# Test code
if __name__ == "__main__":
    from data_acquisition import SentimentDataAcquisition
    
    # Load data
    data_acq = SentimentDataAcquisition('config.yaml')
    dataset = data_acq.fetch_full_dataset("2023-01-01", "2023-03-31")
    
    # Initialize model
    smarty_bert = SmartyBERT('config.yaml')
    
    # Prepare training data
    text_data = dataset['text_data']
    market_data = dataset['market_data']
    
    # Merge to get next-day returns
    merged = text_data.merge(
        market_data[['ticker', 'date', 'return']],
        on=['ticker', 'date'],
        how='inner'
    )
    
    # Construct labels (simplified - no beta adjustment)
    texts = merged['text'].tolist()
    labels = smarty_bert.construct_labels(
        merged['return'],
        pd.Series(np.random.randn(len(merged)) * 0.01),  # Placeholder market returns
        beta=1.0
    )
    
    # Train
    print("\nTraining Smarty BERT (simplified demo)...")
    history = smarty_bert.train_model(texts[:100], labels[:100], validation_split=0.2)
    
    # Predict
    print("\nGenerating predictions...")
    predictions = smarty_bert.predict(texts[:10])
    
    print("\nPredictions (sentiment scores):")
    for i, (text, pred) in enumerate(zip(texts[:5], predictions[:5])):
        print(f"  Text {i+1}: {text[:50]}...")
        print(f"  Sentiment: {pred:.6f}\n")
