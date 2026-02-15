#!/usr/bin/env python3
"""
Optimized vLLM Sentiment Scorer with proper memory management for 16GB GPU
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from tqdm import tqdm
import yfinance as yf
import warnings
import re
warnings.filterwarnings('ignore')

class VLLMSentimentScorer:
    def __init__(self):
        self.model_name = "mistralai/Mistral-7B-Instruct-v0.2"
        self.llm = None
        self.batch_size = 32
        self.historical_cache = {}
        
        print("=" * 70)
        print("vLLM Optimized Sentiment Scorer")
        print("=" * 70)
        print("Optimized for 16GB GPU (RTX 4080)")
        print("=" * 70)
    
    def setup_model(self):
        """Setup vLLM with proper memory settings for 16GB GPU"""
        print("\n[1/4] Setting up vLLM...")
        
        try:
            from vllm import LLM, SamplingParams
            import torch
            
            if not torch.cuda.is_available():
                print("✗ CUDA not available")
                return False
            
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            
            print(f"\n✓ GPU: {gpu_name}")
            print(f"✓ VRAM: {gpu_memory:.1f} GB")
            
            print(f"\nLoading {self.model_name}...")
            print("This may take 5-10 minutes on first run...")
            
            # Conservative memory settings for 16GB GPU
            self.llm = LLM(
                model=self.model_name,
                tensor_parallel_size=1,
                gpu_memory_utilization=0.75,  # Use 75% instead of 90%
                max_model_len=2048,  # Reduced from 4096
                max_num_seqs=64,  # Limit concurrent sequences
                trust_remote_code=True,
                enforce_eager=False,
                swap_space=4,  # GB of CPU swap space
            )
            
            print("✓ vLLM loaded successfully")
            print(f"✓ Max context: 2048 tokens")
            print(f"✓ GPU memory: 75% utilization")
            return True
            
        except ImportError:
            print("✗ vLLM not installed. Run: pip install vllm")
            return False
        except Exception as e:
            print(f"✗ Error: {e}")
            print("\nTroubleshooting:")
            print("1. Try: pip install vllm --upgrade")
            print("2. Or use: python interactive_sentiment_scorer.py (slower but works)")
            return False
    
    def load_data(self) -> pd.DataFrame:
        """Load earnings transcripts dataset"""
        print("\n[2/4] Loading Dataset...")
        print("\nData source:")
        print("1. S&P 500 Earnings Transcripts - RECOMMENDED")
        print("2. Local CSV file")
        
        choice = input("\nSelect option (1-2): ").strip() or "1"
        
        if choice == "1":
            return self._load_sp500_transcripts()
        else:
            return self._load_local_csv()
    
    def _load_sp500_transcripts(self) -> pd.DataFrame:
        """Load S&P 500 transcripts"""
        print("\nLoading S&P 500 Earnings Transcripts...")
        
        try:
            from datasets import load_dataset
            
            print("Downloading (first run may take a few minutes)...")
            dataset = load_dataset("kurry/sp500_earnings_transcripts", split="train")
            df = dataset.to_pandas()
            
            print(f"✓ Loaded {len(df):,} transcripts")
            
            # Filter by year
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
                print(f"\nDate range: {df['date'].min().year} to {df['date'].max().year}")
                
                start_year = input("Start year (or Enter for all): ").strip()
                if start_year:
                    df = df[df['date'].dt.year >= int(start_year)]
                
                end_year = input("End year (or Enter for all): ").strip()
                if end_year:
                    df = df[df['date'].dt.year <= int(end_year)]
                
                print(f"✓ Filtered to {len(df):,} transcripts")
            
            # Rename text column
            text_col = 'content' if 'content' in df.columns else 'text'
            df = df.rename(columns={text_col: 'text'})
            
            # Ask about sampling
            if len(df) > 1000:
                sample = input(f"\nDataset has {len(df):,} rows. Sample for testing? (e.g., 100, 500) [Enter for all]: ").strip()
                if sample:
                    sample_size = int(sample)
                    df = df.sample(n=min(sample_size, len(df)), random_state=42)
                    print(f"✓ Sampled {len(df)} transcripts")
            
            return df
            
        except Exception as e:
            print(f"✗ Error: {e}")
            return pd.DataFrame()
    
    def _load_local_csv(self) -> pd.DataFrame:
        """Load from CSV"""
        file_path = input("\nCSV file path: ").strip()
        
        try:
            df = pd.read_csv(file_path)
            print(f"✓ Loaded {len(df):,} rows")
            
            text_col = input(f"Text column name (available: {', '.join(df.columns)}): ").strip()
            if text_col not in df.columns:
                print(f"✗ Column not found")
                return pd.DataFrame()
            
            return df.rename(columns={text_col: 'text'})
        except Exception as e:
            print(f"✗ Error: {e}")
            return pd.DataFrame()
    
    def create_sentiment_prompt(self, text: str) -> str:
        """Create scoring prompt"""
        # Take first 1500 chars for speed
        text_sample = text[:1500]
        
        prompt = f"""[INST] Rate the sentiment of this earnings call (1-5):

1 = Very Negative
2 = Negative  
3 = Neutral
4 = Positive
5 = Very Positive

Text: {text_sample}

Answer with ONLY the number. [/INST]

Score:"""
        return prompt
    
    def create_comprehensive_prompt(self, text: str) -> str:
        """Create multi-aspect prompt"""
        text_sample = text[:2500]
        
        prompt = f"""[INST] Rate these 5 aspects (1-5 each):

Text: {text_sample}

Format:
Revenue Growth: [1-5]
Profitability: [1-5]
Forward Guidance: [1-5]
Management Confidence: [1-5]
Competitive Position: [1-5]
[/INST]

Scores:
"""
        return prompt
    
    def parse_comprehensive_response(self, response: str) -> Dict[str, int]:
        """Parse multi-aspect scores"""
        aspects = {
            'revenue_growth': 3,
            'profitability': 3,
            'forward_guidance': 3,
            'management_confidence': 3,
            'competitive_position': 3
        }
        
        try:
            patterns = {
                'revenue_growth': r'Revenue Growth:\s*(\d)',
                'profitability': r'Profitability:\s*(\d)',
                'forward_guidance': r'Forward Guidance:\s*(\d)',
                'management_confidence': r'Management Confidence:\s*(\d)',
                'competitive_position': r'Competitive Position:\s*(\d)'
            }
            
            for aspect, pattern in patterns.items():
                match = re.search(pattern, response, re.IGNORECASE)
                if match:
                    score = int(match.group(1))
                    if 1 <= score <= 5:
                        aspects[aspect] = score
        except:
            pass
        
        return aspects
    
    def extract_score(self, response: str) -> int:
        """Extract score from response"""
        try:
            digits = re.findall(r'\b([1-5])\b', response)
            if digits:
                return int(digits[0])
        except:
            pass
        return 3
    
    def score_batch(self, texts: List[str], mode: str = 'simple') -> List:
        """Score a batch of texts"""
        from vllm import SamplingParams
        
        if mode == 'simple':
            prompts = [self.create_sentiment_prompt(text) for text in texts]
            sampling_params = SamplingParams(
                temperature=0.0,
                max_tokens=5,
                stop=["\n", "."]
            )
        else:
            prompts = [self.create_comprehensive_prompt(text) for text in texts]
            sampling_params = SamplingParams(
                temperature=0.0,
                max_tokens=100,
                stop=["\n\n"]
            )
        
        # Generate
        outputs = self.llm.generate(prompts, sampling_params)
        
        # Parse results
        results = []
        for output in outputs:
            response = output.outputs[0].text
            
            if mode == 'simple':
                score = self.extract_score(response)
                results.append({'sentiment_score': score})
            else:
                aspects = self.parse_comprehensive_response(response)
                overall = np.mean(list(aspects.values()))
                results.append({
                    'overall_sentiment': round(overall, 2),
                    'overall_sentiment_int': round(overall),
                    **aspects
                })
        
        return results
    
    def process_dataset(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """Process dataset with vLLM"""
        print("\n[3/4] Configuring Processing...")
        
        # Choose mode
        print("\nMode:")
        print("1. Simple (1-5 score) - FASTEST")
        print("2. Comprehensive (5 aspects) - Detailed")
        
        mode_choice = input("\nSelect (1/2) [default: 1]: ").strip() or "1"
        mode = 'simple' if mode_choice == "1" else 'comprehensive'
        
        # Batch size
        if mode == 'simple':
            default_batch = 64
        else:
            default_batch = 32
        
        batch_input = input(f"\nBatch size [default: {default_batch}]: ").strip()
        batch_size = int(batch_input) if batch_input else default_batch
        
        print(f"\n[4/4] Processing {len(df):,} transcripts...")
        print(f"Mode: {mode}")
        print(f"Batch size: {batch_size}")
        
        # Process in batches
        results = []
        
        for i in tqdm(range(0, len(df), batch_size), desc="Scoring"):
            batch_df = df.iloc[i:i+batch_size]
            batch_texts = batch_df['text'].tolist()
            
            batch_results = self.score_batch(batch_texts, mode=mode)
            results.extend(batch_results)
        
        # Add to dataframe
        if mode == 'simple':
            df['sentiment_score'] = [r['sentiment_score'] for r in results]
            
            print(f"\n✓ Scored {len(df):,} transcripts")
            print(f"\nDistribution:")
            print(df['sentiment_score'].value_counts().sort_index())
            print(f"Average: {df['sentiment_score'].mean():.2f}")
            
            summary = {
                'mode': 'simple',
                'total': len(df),
                'average': float(df['sentiment_score'].mean()),
                'distribution': df['sentiment_score'].value_counts().to_dict()
            }
        else:
            df['overall_sentiment'] = [r['overall_sentiment'] for r in results]
            df['overall_sentiment_int'] = [r['overall_sentiment_int'] for r in results]
            df['revenue_growth'] = [r['revenue_growth'] for r in results]
            df['profitability'] = [r['profitability'] for r in results]
            df['forward_guidance'] = [r['forward_guidance'] for r in results]
            df['management_confidence'] = [r['management_confidence'] for r in results]
            df['competitive_position'] = [r['competitive_position'] for r in results]
            
            print(f"\n✓ Scored {len(df):,} transcripts")
            print(f"\nAverage Scores:")
            for col in ['overall_sentiment', 'revenue_growth', 'profitability', 
                       'forward_guidance', 'management_confidence', 'competitive_position']:
                print(f"  {col}: {df[col].mean():.2f}")
            
            summary = {
                'mode': 'comprehensive',
                'total': len(df),
                'averages': {col: float(df[col].mean()) 
                           for col in ['overall_sentiment', 'revenue_growth', 'profitability',
                                     'forward_guidance', 'management_confidence', 'competitive_position']}
            }
        
        return df, summary
    
    def save_results(self, df: pd.DataFrame, summary: Dict):
        """Save results"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # CSV
        csv_file = f"sentiment_vllm_{timestamp}.csv"
        df.to_csv(csv_file, index=False)
        print(f"\n✓ Saved: {csv_file}")
        
        # Summary
        json_file = f"summary_vllm_{timestamp}.json"
        with open(json_file, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"✓ Saved: {json_file}")
        
        print("\n" + "=" * 70)
        print("Complete!")
        print("=" * 70)
    
    def run(self):
        """Main execution"""
        if not self.setup_model():
            return
        
        df = self.load_data()
        if df.empty:
            print("\nNo data loaded")
            return
        
        df_scored, summary = self.process_dataset(df)
        self.save_results(df_scored, summary)


def main():
    try:
        scorer = VLLMSentimentScorer()
        scorer.run()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
