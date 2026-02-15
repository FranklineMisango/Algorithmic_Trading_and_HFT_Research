#!/usr/bin/env python3
"""
Interactive Sentiment Scorer using Mistral-7B-Instruct-v0.2
Downloads model, processes data, and scores economic sentiments (1-5 scale)
Optimized for remote GPU usage
"""

import os
import sys
import json
import torch
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import warnings
warnings.filterwarnings('ignore')

class InteractiveSentimentScorer:
    def __init__(self):
        self.model_name = "mistralai/Mistral-7B-Instruct-v0.2"
        self.model = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.use_quantization = False
        
        print("=" * 60)
        print("Interactive Sentiment Scorer - Mistral 7B")
        print("=" * 60)
        print(f"Device detected: {self.device.upper()}")
        if self.device == "cuda":
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        print("=" * 60)
    
    def setup_model(self):
        """Download and setup Mistral-7B model"""
        print("\n[1/5] Setting up Mistral-7B-Instruct-v0.2...")
        
        # Ask about quantization for memory efficiency
        if self.device == "cuda":
            use_quant = input("\nUse 4-bit quantization for memory efficiency? (y/n) [recommended: y]: ").strip().lower()
            self.use_quantization = use_quant in ['y', 'yes', '']
        
        try:
            print(f"\nDownloading tokenizer from Hugging Face...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
            
            print(f"Downloading model weights (this may take 10-15 minutes)...")
            
            if self.use_quantization:
                # 4-bit quantization config for memory efficiency
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
                
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    quantization_config=quantization_config,
                    device_map="auto",
                    trust_remote_code=True
                )
                print("✓ Model loaded with 4-bit quantization")
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    device_map="auto" if self.device == "cuda" else None,
                    trust_remote_code=True
                )
                if self.device == "cpu":
                    self.model = self.model.to(self.device)
                print("✓ Model loaded successfully")
            
            self.model.eval()
            return True
            
        except Exception as e:
            print(f"✗ Error loading model: {e}")
            return False
    
    def load_data(self) -> pd.DataFrame:
        """Interactive data loading from multiple sources"""
        print("\n[2/5] Loading Dataset...")
        print("\nData source options:")
        print("1. S&P 500 Earnings Transcripts (kurry/sp500_earnings_transcripts) - RECOMMENDED")
        print("2. Local CSV file")
        print("3. Local JSON/JSONL file")
        print("4. Other Hugging Face dataset")
        print("5. Download Fed transcripts (FRED API)")
        
        choice = input("\nSelect option (1-5): ").strip()
        
        if choice == "1" or choice == "":
            return self._load_sp500_transcripts()
        elif choice == "2":
            return self._load_csv()
        elif choice == "3":
            return self._load_json()
        elif choice == "4":
            return self._load_huggingface()
        elif choice == "5":
            return self._download_fed_transcripts()
        else:
            print("Invalid choice. Defaulting to S&P 500 transcripts.")
            return self._load_sp500_transcripts()
    
    def _load_sp500_transcripts(self) -> pd.DataFrame:
        """Load S&P 500 earnings transcripts from Hugging Face"""
        from datasets import load_dataset
        
        print("\nLoading S&P 500 Earnings Transcripts...")
        print("Dataset: kurry/sp500_earnings_transcripts (2005-2025)")
        
        try:
            # Load the dataset
            print("Downloading dataset (this may take a few minutes on first run)...")
            dataset = load_dataset("kurry/sp500_earnings_transcripts", split="train")
            
            # Convert to pandas
            df = dataset.to_pandas()
            print(f"✓ Loaded {len(df):,} total transcripts")
            
            # Show available columns
            print(f"\nAvailable columns: {', '.join(df.columns)}")
            
            # Filter by date
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'], errors='coerce')
                
                # Ask user for date range
                print(f"\nDate range in dataset: {df['date'].min()} to {df['date'].max()}")
                
                start_year = input("Enter start year (e.g., 2015) or press Enter for all: ").strip()
                end_year = input("Enter end year (e.g., 2020) or press Enter for all: ").strip()
                
                if start_year:
                    df = df[df['date'].dt.year >= int(start_year)]
                if end_year:
                    df = df[df['date'].dt.year <= int(end_year)]
                
                print(f"✓ Filtered to {len(df):,} transcripts")
            
            # Identify text column
            text_columns = [col for col in df.columns if 'text' in col.lower() or 'transcript' in col.lower()]
            
            if len(text_columns) == 1:
                text_col = text_columns[0]
                print(f"Using text column: '{text_col}'")
            elif len(text_columns) > 1:
                print(f"\nMultiple text columns found: {', '.join(text_columns)}")
                text_col = input("Enter column name to use: ").strip()
            else:
                text_col = input(f"Enter text column name (columns: {', '.join(df.columns[:5])}...): ").strip()
            
            if text_col not in df.columns:
                print(f"✗ Column '{text_col}' not found")
                return pd.DataFrame()
            
            # Rename for consistency
            df = df.rename(columns={text_col: 'text'})
            
            # Drop rows with empty text
            df = df[df['text'].notna() & (df['text'].str.len() > 0)]
            
            print(f"✓ Ready to process {len(df):,} transcripts")
            
            return df
            
        except Exception as e:
            print(f"✗ Error loading dataset: {e}")
            print("Make sure 'datasets' library is installed: pip install datasets")
            return pd.DataFrame()
    
    def _load_csv(self) -> pd.DataFrame:
        """Load data from CSV file"""
        file_path = input("Enter CSV file path: ").strip()
        
        if not os.path.exists(file_path):
            print(f"✗ File not found: {file_path}")
            return pd.DataFrame()
        
        df = pd.read_csv(file_path)
        text_col = input(f"Enter text column name (columns: {', '.join(df.columns)}): ").strip()
        
        if text_col not in df.columns:
            print(f"✗ Column '{text_col}' not found")
            return pd.DataFrame()
        
        print(f"✓ Loaded {len(df)} rows from CSV")
        return df.rename(columns={text_col: 'text'})
    
    def _load_json(self) -> pd.DataFrame:
        """Load data from JSON/JSONL file"""
        file_path = input("Enter JSON/JSONL file path: ").strip()
        
        if not os.path.exists(file_path):
            print(f"✗ File not found: {file_path}")
            return pd.DataFrame()
        
        if file_path.endswith('.jsonl'):
            df = pd.read_json(file_path, lines=True)
        else:
            df = pd.read_json(file_path)
        
        text_col = input(f"Enter text column name (columns: {', '.join(df.columns)}): ").strip()
        
        if text_col not in df.columns:
            print(f"✗ Column '{text_col}' not found")
            return pd.DataFrame()
        
        print(f"✓ Loaded {len(df)} rows from JSON")
        return df.rename(columns={text_col: 'text'})
    
    def _load_huggingface(self) -> pd.DataFrame:
        """Load dataset from Hugging Face"""
        from datasets import load_dataset
        
        dataset_name = input("Enter Hugging Face dataset name (e.g., 'news_articles'): ").strip()
        
        try:
            dataset = load_dataset(dataset_name)
            df = pd.DataFrame(dataset['train'])
            
            text_col = input(f"Enter text column name (columns: {', '.join(df.columns)}): ").strip()
            
            if text_col not in df.columns:
                print(f"✗ Column '{text_col}' not found")
                return pd.DataFrame()
            
            print(f"✓ Loaded {len(df)} rows from Hugging Face")
            return df.rename(columns={text_col: 'text'})
        except Exception as e:
            print(f"✗ Error loading dataset: {e}")
            return pd.DataFrame()
    
    def _download_fed_transcripts(self) -> pd.DataFrame:
        """Download Federal Reserve transcripts from 2015"""
        print("\nDownloading Fed transcripts from 2015...")
        
        try:
            # Using yfinance or fredapi for economic data
            # This is a placeholder - you might need specific API
            from fredapi import Fred
            
            api_key = input("Enter FRED API key (get free at https://fred.stlouisfed.org/): ").strip()
            
            if not api_key:
                print("✗ API key required")
                return pd.DataFrame()
            
            fred = Fred(api_key=api_key)
            
            # Example: Download economic indicators
            series = input("Enter FRED series ID (e.g., 'GDP', 'UNRATE', 'CPIAUCSL'): ").strip()
            data = fred.get_series(series, observation_start='2015-01-01', observation_end='2015-12-31')
            
            df = pd.DataFrame({
                'date': data.index,
                'value': data.values,
                'text': [f"Economic indicator {series} on {date.strftime('%Y-%m-%d')} was {value:.2f}" 
                        for date, value in zip(data.index, data.values)]
            })
            
            print(f"✓ Downloaded {len(df)} data points from FRED")
            return df
            
        except Exception as e:
            print(f"✗ Error downloading data: {e}")
            print("Consider using option 1-3 with pre-downloaded data")
            return pd.DataFrame()
    
    def create_sentiment_prompt(self, text: str) -> str:
        """Create prompt for economic sentiment scoring"""
        prompt = f"""[INST] You are an expert economic analyst. Rate the economic sentiment of the following text on a scale of 1-5:

1 = Very Negative (recession, crisis, severe downturn)
2 = Negative (slowdown, weakness, concerns)
3 = Neutral (mixed signals, stable, uncertain)
4 = Positive (growth, improvement, optimism)
5 = Very Positive (strong growth, boom, excellent conditions)

Text: {text[:1000]}

Provide ONLY a single number (1-5) as your response. [/INST]

Sentiment Score:"""
        return prompt
    
    def score_sentiment(self, text: str) -> int:
        """Score a single text for economic sentiment"""
        prompt = self.create_sentiment_prompt(text)
        
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=10,
                temperature=0.1,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract score from response
        try:
            # Look for a number 1-5 in the response
            score_text = response.split("Sentiment Score:")[-1].strip()
            score = int(score_text[0])
            if 1 <= score <= 5:
                return score
        except:
            pass
        
        # Default to neutral if parsing fails
        return 3
    
    def process_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process entire dataset with progress tracking"""
        print("\n[3/5] Processing Dataset...")
        
        batch_size = int(input("Enter batch size (recommended: 1-10): ").strip() or "5")
        
        # Filter by year if date column exists
        if 'date' in df.columns:
            filter_year = input("Filter by year? (e.g., 2015, or press Enter to skip): ").strip()
            if filter_year:
                df['date'] = pd.to_datetime(df['date'])
                df = df[df['date'].dt.year == int(filter_year)]
                print(f"Filtered to {len(df)} rows from year {filter_year}")
        
        if len(df) == 0:
            print("✗ No data to process")
            return df
        
        print(f"\nProcessing {len(df)} texts...")
        
        scores = []
        for i in tqdm(range(0, len(df), batch_size), desc="Scoring"):
            batch = df.iloc[i:i+batch_size]
            
            for _, row in batch.iterrows():
                score = self.score_sentiment(row['text'])
                scores.append(score)
        
        df['sentiment_score'] = scores
        
        print(f"\n✓ Scored {len(df)} texts")
        print(f"\nScore Distribution:")
        print(df['sentiment_score'].value_counts().sort_index())
        print(f"\nAverage Score: {df['sentiment_score'].mean():.2f}")
        
        return df
    
    def save_results(self, df: pd.DataFrame):
        """Save scored results"""
        print("\n[4/5] Saving Results...")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_path = f"sentiment_scores_{timestamp}.csv"
        
        save_path = input(f"Enter output file path (default: {default_path}): ").strip() or default_path
        
        df.to_csv(save_path, index=False)
        print(f"✓ Results saved to: {save_path}")
        
        # Also save summary statistics
        summary_path = save_path.replace('.csv', '_summary.json')
        summary = {
            'total_texts': len(df),
            'average_score': float(df['sentiment_score'].mean()),
            'score_distribution': df['sentiment_score'].value_counts().to_dict(),
            'timestamp': timestamp,
            'model': self.model_name
        }
        
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"✓ Summary saved to: {summary_path}")
    
    def run(self):
        """Main execution flow"""
        try:
            # Step 1: Setup model
            if not self.setup_model():
                print("Failed to setup model. Exiting.")
                return
            
            # Step 2: Load data
            df = self.load_data()
            if len(df) == 0:
                print("No data loaded. Exiting.")
                return
            
            # Step 3: Process dataset
            scored_df = self.process_dataset(df)
            
            # Step 4: Save results
            self.save_results(scored_df)
            
            # Step 5: Done
            print("\n[5/5] Complete!")
            print("=" * 60)
            print("Sentiment scoring completed successfully!")
            print("=" * 60)
            
        except KeyboardInterrupt:
            print("\n\n✗ Process interrupted by user")
        except Exception as e:
            print(f"\n✗ Error during execution: {e}")
            import traceback
            traceback.print_exc()

def main():
    """Entry point"""
    scorer = InteractiveSentimentScorer()
    scorer.run()

if __name__ == "__main__":
    main()
