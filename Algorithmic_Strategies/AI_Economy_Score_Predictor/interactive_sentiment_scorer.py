#!/usr/bin/env python3
"""
Advanced Interactive Sentiment Scorer using Mistral-7B-Instruct-v0.2
Multi-dimensional sentiment analysis with contextual scoring
Optimized for remote GPU usage with comprehensive analysis
"""

import os
import sys
import json
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')

class InteractiveSentimentScorer:
    def __init__(self):
        self.model_name = "mistralai/Mistral-7B-Instruct-v0.2"
        self.model = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.use_quantization = False
        self.use_advanced_scoring = True
        self.chunk_size = 4000  # Characters per chunk
        self.historical_cache = {}  # Cache for historical data
        
        print("=" * 70)
        print("Advanced Multi-Dimensional Sentiment Scorer - Mistral 7B")
        print("=" * 70)
        print(f"Device detected: {self.device.upper()}")
        if self.device == "cuda":
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        print("=" * 70)
    
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
            
            # Identify text column - auto-detect common names
            text_column_candidates = ['content', 'text', 'transcript', 'transcript_text']
            text_col = None
            
            # Try to auto-detect
            for candidate in text_column_candidates:
                if candidate in df.columns:
                    text_col = candidate
                    print(f"✓ Auto-detected text column: '{text_col}'")
                    break
            
            # If not found, ask user
            if text_col is None:
                print(f"\nCouldn't auto-detect text column.")
                print(f"Available columns: {', '.join(df.columns)}")
                
                # Show sample of first few rows to help identify
                print("\nFirst row sample:")
                for col in df.columns[:5]:
                    sample = str(df[col].iloc[0])[:50]
                    print(f"  {col}: {sample}...")
                
                text_col = input("\nEnter text column name (usually 'content' for this dataset): ").strip()
            
            if not text_col or text_col not in df.columns:
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
    
    def create_aspect_prompt(self, text: str, aspect: str, aspect_description: str) -> str:
        """Create prompt for specific aspect scoring with chain-of-thought"""
        prompt = f"""[INST] You are an expert financial analyst. Analyze the following earnings call transcript excerpt focusing on: {aspect}

{aspect_description}

Text: {text}

First, identify 2-3 key facts about {aspect} from the text.
Then, rate the sentiment for this aspect on a scale of 1-5:

1 = Very Negative
2 = Negative
3 = Neutral
4 = Positive
5 = Very Positive

Respond in this format:
Key Facts:
- [fact 1]
- [fact 2]

Score: [1-5]
Reasoning: [brief explanation]
[/INST]"""
        return prompt
    
    def parse_aspect_response(self, response: str) -> Dict[str, any]:
        """Parse chain-of-thought response"""
        try:
            lines = response.split('\n')
            score = 3
            facts = []
            reasoning = ""
            
            for line in lines:
                if 'Score:' in line:
                    score_text = line.split('Score:')[-1].strip()
                    score = int(score_text[0]) if score_text[0].isdigit() else 3
                elif line.strip().startswith('-'):
                    facts.append(line.strip()[1:].strip())
                elif 'Reasoning:' in line:
                    reasoning = line.split('Reasoning:')[-1].strip()
            
            return {
                'score': score,
                'facts': facts,
                'reasoning': reasoning
            }
        except:
            return {'score': 3, 'facts': [], 'reasoning': 'Parsing failed'}
    
    def split_transcript_sections(self, text: str) -> Dict[str, str]:
        """Split transcript into sections: prepared remarks vs Q&A"""
        text_lower = text.lower()
        
        # Common markers for Q&A section
        qa_markers = ['question-and-answer', 'q&a', 'questions and answers', 
                      'operator', 'first question', 'begin the question']
        
        qa_start = len(text)
        for marker in qa_markers:
            pos = text_lower.find(marker)
            if pos > 0:
                qa_start = min(qa_start, pos)
        
        if qa_start < len(text) * 0.3:  # Q&A shouldn't start too early
            qa_start = len(text)
        
        return {
            'prepared_remarks': text[:qa_start],
            'qa_session': text[qa_start:] if qa_start < len(text) else ""
        }
    
    def chunk_text(self, text: str, chunk_size: int = 4000) -> List[str]:
        """Split text into overlapping chunks"""
        chunks = []
        overlap = 500
        
        for i in range(0, len(text), chunk_size - overlap):
            chunk = text[i:i + chunk_size]
            if len(chunk) > 100:  # Skip tiny chunks
                chunks.append(chunk)
        
        return chunks
    
    def score_text_aspect(self, text: str, aspect: str, description: str) -> Dict[str, any]:
        """Score a specific aspect of text"""
        # Chunk long text
        chunks = self.chunk_text(text, self.chunk_size)
        
        if len(chunks) == 0:
            return {'score': 3, 'facts': [], 'reasoning': 'Empty text'}
        
        # For very long transcripts, sample representative chunks
        if len(chunks) > 5:
            # Take first, middle, and last chunks
            sample_chunks = [chunks[0], chunks[len(chunks)//2], chunks[-1]]
        else:
            sample_chunks = chunks
        
        # Score each chunk
        chunk_results = []
        for chunk in sample_chunks[:3]:  # Max 3 chunks per aspect
            prompt = self.create_aspect_prompt(chunk[:2000], aspect, description)
            
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=150,
                    temperature=0.3,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            result = self.parse_aspect_response(response)
            chunk_results.append(result)
        
        # Aggregate results
        avg_score = np.mean([r['score'] for r in chunk_results])
        all_facts = []
        for r in chunk_results:
            all_facts.extend(r['facts'])
        
        return {
            'score': round(avg_score, 2),
            'score_int': int(round(avg_score)),
            'facts': all_facts[:5],  # Top 5 facts
            'reasoning': chunk_results[0]['reasoning'] if chunk_results else ''
        }
    
    def score_comprehensive(self, text: str, symbol: str = None, date: str = None) -> Dict[str, any]:
        """Comprehensive multi-aspect sentiment scoring"""
        
        # Split into sections
        sections = self.split_transcript_sections(text)
        
        # Define aspects to score
        aspects = {
            'revenue_growth': 'Focus on revenue trends, sales growth, top-line performance',
            'profitability': 'Focus on profit margins, costs, operating efficiency, bottom-line performance',
            'forward_guidance': 'Focus on future outlook, guidance, predictions, management expectations',
            'management_confidence': 'Focus on management tone, confidence level, certainty in statements',
            'competitive_position': 'Focus on market share, competitive advantages, industry positioning'
        }
        
        results = {
            'symbol': symbol,
            'date': date,
            'timestamp': datetime.now().isoformat()
        }
        
        # Score prepared remarks
        print(f"  Analyzing prepared remarks ({len(sections['prepared_remarks'])} chars)...")
        prepared_scores = {}
        for aspect, description in aspects.items():
            aspect_result = self.score_text_aspect(sections['prepared_remarks'], aspect, description)
            prepared_scores[aspect] = aspect_result
        
        results['prepared_remarks'] = prepared_scores
        
        # Score Q&A if exists
        if sections['qa_session']:
            print(f"  Analyzing Q&A session ({len(sections['qa_session'])} chars)...")
            qa_scores = {}
            for aspect, description in aspects.items():
                aspect_result = self.score_text_aspect(sections['qa_session'], aspect, description)
                qa_scores[aspect] = aspect_result
            
            results['qa_session'] = qa_scores
        else:
            results['qa_session'] = None
        
        # Calculate weighted composite scores
        # Q&A weighted more heavily (70%) as it's less scripted
        composite_scores = {}
        for aspect in aspects.keys():
            prepared = prepared_scores[aspect]['score']
            
            if results['qa_session']:
                qa = qa_scores[aspect]['score']
                composite = 0.3 * prepared + 0.7 * qa
            else:
                composite = prepared
            
            composite_scores[aspect] = round(composite, 2)
        
        results['composite_scores'] = composite_scores
        
        # Overall sentiment (average of all aspects)
        results['overall_sentiment'] = round(np.mean(list(composite_scores.values())), 2)
        results['overall_sentiment_int'] = int(round(results['overall_sentiment']))
        
        return results
    
    def fetch_market_context(self, symbol: str, date: str) -> Dict[str, any]:
        """Fetch market data from Yahoo Finance for context"""
        try:
            # Convert date to datetime
            target_date = pd.to_datetime(date)
            start_date = target_date - timedelta(days=90)  # 90 days prior
            end_date = target_date + timedelta(days=30)  # 30 days after
            
            # Fetch stock data
            ticker = yf.Ticker(symbol)
            stock_data = ticker.history(start=start_date, end=end_date)
            
            if stock_data.empty:
                return None
            
            # Get price at earnings date
            earnings_price = stock_data.loc[stock_data.index >= target_date].iloc[0]['Close'] if len(stock_data.loc[stock_data.index >= target_date]) > 0 else None
            
            # Calculate returns
            if earnings_price and len(stock_data) > 30:
                pre_earnings_price = stock_data.loc[stock_data.index < target_date].iloc[-1]['Close']
                post_earnings_prices = stock_data.loc[stock_data.index > target_date]
                
                if len(post_earnings_prices) >= 5:
                    week_later_price = post_earnings_prices.iloc[4]['Close']
                    week_return = ((week_later_price - earnings_price) / earnings_price) * 100
                else:
                    week_return = None
                
                if len(post_earnings_prices) >= 20:
                    month_later_price = post_earnings_prices.iloc[19]['Close']
                    month_return = ((month_later_price - earnings_price) / earnings_price) * 100
                else:
                    month_return = None
            else:
                week_return = None
                month_return = None
            
            # Fetch S&P 500 for relative performance
            spy = yf.Ticker('SPY')
            spy_data = spy.history(start=start_date, end=end_date)
            
            market_context = {
                'symbol': symbol,
                'date': date,
                'price_at_earnings': float(earnings_price) if earnings_price else None,
                'return_1week': round(week_return, 2) if week_return else None,
                'return_1month': round(month_return, 2) if month_return else None,
                'has_market_data': True
            }
            
            return market_context
            
        except Exception as e:
            print(f"    Warning: Could not fetch market data for {symbol}: {e}")
            return {'symbol': symbol, 'date': date, 'has_market_data': False}
    
    def score_sentiment(self, text: str, symbol: str = None, date: str = None, use_simple: bool = False) -> any:
        """Score sentiment - either simple or comprehensive"""
        if use_simple:
            # Simple legacy scoring
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
            
            try:
                score_text = response.split("Sentiment Score:")[-1].strip()
                score = int(score_text[0])
                if 1 <= score <= 5:
                    return score
            except:
                pass
            
            return 3
        else:
            # Comprehensive multi-aspect scoring
            return self.score_comprehensive(text, symbol, date)
    
    def process_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process entire dataset with progress tracking"""
        print("\n[3/5] Processing Dataset...")
        
        # Ask about scoring mode
        print("\nScoring modes:")
        print("1. Simple scoring (fast, 1-5 score only)")
        print("2. Comprehensive multi-aspect scoring (slower, detailed analysis)")
        
        mode = input("\nSelect mode (1/2) [default: 2]: ").strip()
        use_simple = (mode == "1")
        
        if not use_simple:
            print("\nComprehensive mode will analyze:")
            print("  - 5 sentiment aspects (revenue, profitability, guidance, confidence, competitive)")
            print("  - Prepared remarks vs Q&A sessions separately")
            print("  - Market context from Yahoo Finance")
            print("  - ~30-60 seconds per transcript")
        
        batch_size = int(input("\nEnter batch size (recommended: 1-5 for comprehensive, 1-10 for simple): ").strip() or "3")
        
        # Ask about market data fetching
        fetch_market = False
        if not use_simple and 'symbol' in df.columns and 'date' in df.columns:
            fetch_market_input = input("\nFetch market data from Yahoo Finance? (y/n) [default: y]: ").strip().lower()
            fetch_market = fetch_market_input in ['y', 'yes', '']
        
        # Filter by year if date column exists
        if 'date' in df.columns:
            filter_year = input("\nFilter by year? (e.g., 2015, or press Enter to skip): ").strip()
            if filter_year:
                df['date'] = pd.to_datetime(df['date'])
                df = df[df['date'].dt.year == int(filter_year)]
                print(f"Filtered to {len(df)} rows from year {filter_year}")
        
        if len(df) == 0:
            print("✗ No data to process")
            return df
        
        print(f"\nProcessing {len(df)} texts...")
        
        results = []
        market_data = []
        
        for i in tqdm(range(0, len(df), batch_size), desc="Scoring"):
            batch = df.iloc[i:i+batch_size]
            
            for idx, row in batch.iterrows():
                symbol = row.get('symbol', None)
                date = row.get('date', None)
                
                if date and isinstance(date, pd.Timestamp):
                    date = date.strftime('%Y-%m-%d')
                
                print(f"\n  Processing {symbol} ({date})..." if symbol else f"\n  Processing row {idx}...")
                
                # Score sentiment
                result = self.score_sentiment(row['text'], symbol, date, use_simple=use_simple)
                
                if use_simple:
                    results.append({'sentiment_score': result})
                else:
                    results.append(result)
                    
                    # Fetch market data if requested
                    if fetch_market and symbol and date:
                        market_ctx = self.fetch_market_context(symbol, date)
                        market_data.append(market_ctx)
        
        # Add results to dataframe
        if use_simple:
            df['sentiment_score'] = [r['sentiment_score'] for r in results]
            
            print(f"\n✓ Scored {len(df)} texts")
            print(f"\nScore Distribution:")
            print(df['sentiment_score'].value_counts().sort_index())
            print(f"\nAverage Score: {df['sentiment_score'].mean():.2f}")
        else:
            # Add comprehensive scores
            df['overall_sentiment'] = [r['overall_sentiment'] for r in results]
            df['overall_sentiment_int'] = [r['overall_sentiment_int'] for r in results]
            df['revenue_growth'] = [r['composite_scores']['revenue_growth'] for r in results]
            df['profitability'] = [r['composite_scores']['profitability'] for r in results]
            df['forward_guidance'] = [r['composite_scores']['forward_guidance'] for r in results]
            df['management_confidence'] = [r['composite_scores']['management_confidence'] for r in results]
            df['competitive_position'] = [r['composite_scores']['competitive_position'] for r in results]
            
            # Store full results as JSON
            df['detailed_results'] = [json.dumps(r) for r in results]
            
            # Add market data if fetched
            if market_data:
                market_df = pd.DataFrame(market_data)
                df = df.merge(market_df, on=['symbol', 'date'], how='left')
            
            print(f"\n✓ Scored {len(df)} texts with comprehensive analysis")
            print(f"\nOverall Sentiment Distribution:")
            print(df['overall_sentiment_int'].value_counts().sort_index())
            print(f"\nAverage Overall Sentiment: {df['overall_sentiment'].mean():.2f}")
            print(f"\nAspect Averages:")
            for aspect in ['revenue_growth', 'profitability', 'forward_guidance', 'management_confidence', 'competitive_position']:
                print(f"  {aspect}: {df[aspect].mean():.2f}")
        
        return df
    
    def save_results(self, df: pd.DataFrame):
        """Save scored results"""
        print("\n[4/5] Saving Results...")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_path = f"sentiment_scores_{timestamp}.csv"
        
        save_path = input(f"Enter output file path (default: {default_path}): ").strip() or default_path
        
        # Save main CSV
        df.to_csv(save_path, index=False)
        print(f"✓ Results saved to: {save_path}")
        
        # Also save summary statistics
        summary_path = save_path.replace('.csv', '_summary.json')
        
        summary = {
            'total_texts': len(df),
            'timestamp': timestamp,
            'model': self.model_name
        }
        
        # Add appropriate summary stats based on scoring mode
        if 'overall_sentiment' in df.columns:
            summary['scoring_mode'] = 'comprehensive'
            summary['overall_sentiment_avg'] = float(df['overall_sentiment'].mean())
            summary['overall_sentiment_distribution'] = df['overall_sentiment_int'].value_counts().to_dict()
            summary['aspect_averages'] = {
                'revenue_growth': float(df['revenue_growth'].mean()),
                'profitability': float(df['profitability'].mean()),
                'forward_guidance': float(df['forward_guidance'].mean()),
                'management_confidence': float(df['management_confidence'].mean()),
                'competitive_position': float(df['competitive_position'].mean())
            }
            
            # Add market data summary if available
            if 'return_1week' in df.columns:
                valid_returns = df['return_1week'].dropna()
                if len(valid_returns) > 0:
                    summary['market_data'] = {
                        'avg_1week_return': float(valid_returns.mean()),
                        'avg_1month_return': float(df['return_1month'].dropna().mean()) if 'return_1month' in df.columns else None
                    }
        else:
            summary['scoring_mode'] = 'simple'
            summary['average_score'] = float(df['sentiment_score'].mean())
            summary['score_distribution'] = df['sentiment_score'].value_counts().to_dict()
        
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"✓ Summary saved to: {summary_path}")
        
        # Save detailed results if comprehensive mode
        if 'detailed_results' in df.columns:
            detailed_path = save_path.replace('.csv', '_detailed.jsonl')
            with open(detailed_path, 'w') as f:
                for _, row in df.iterrows():
                    f.write(row['detailed_results'] + '\n')
            print(f"✓ Detailed results saved to: {detailed_path}")
    
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
