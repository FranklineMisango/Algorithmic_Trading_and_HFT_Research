#!/usr/bin/env python3
"""
Comprehensive vLLM Sentiment Scorer - Full Feature Set
Combines the speed of vLLM with the sophistication of the original scorer:
- Section splitting (Prepared Remarks vs Q&A)
- Chain-of-thought reasoning
- Market data integration
- Weighted composite scoring
- Full transcript analysis
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

class ComprehensiveVLLMScorer:
    def __init__(self):
        self.model_name = "TheBloke/Mistral-7B-Instruct-v0.2-AWQ"
        self.llm = None
        self.batch_size = 16  # Smaller batches for comprehensive analysis
        self.chunk_size = 3500  # Characters per chunk
        self.historical_cache = {}
        
        print("=" * 70)
        print("Comprehensive vLLM Sentiment Scorer")
        print("=" * 70)
        print("Full feature set with vLLM speed optimization")
        print("=" * 70)
    
    def setup_model(self):
        """Setup vLLM with AWQ quantization"""
        print("\n[1/5] Setting up vLLM...")
        
        try:
            from vllm import LLM, SamplingParams
            import torch
            
            if not torch.cuda.is_available():
                print("✗ CUDA not available")
                return False
            
            print(f"\n✓ GPU: {torch.cuda.get_device_name(0)}")
            print(f"✓ VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
            
            print(f"\nLoading AWQ quantized model...")
            print("First run will download ~4GB model...")
            
            self.llm = LLM(
                model=self.model_name,
                quantization="awq",
                tensor_parallel_size=1,
                gpu_memory_utilization=0.85,
                max_model_len=2048,
                max_num_seqs=128,
                trust_remote_code=True,
                dtype="half",
            )
            
            print("✓ Model loaded successfully")
            return True
            
        except Exception as e:
            print(f"✗ Error: {e}")
            return False
    
    def load_data(self) -> pd.DataFrame:
        """Load earnings transcripts"""
        print("\n[2/5] Loading Dataset...")
        
        try:
            from datasets import load_dataset
            
            print("\nLoading S&P 500 Earnings Transcripts...")
            dataset = load_dataset("kurry/sp500_earnings_transcripts", split="train")
            df = dataset.to_pandas()
            
            print(f"✓ Loaded {len(df):,} transcripts")
            
            # Filter by year
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
                print(f"\nDate range: {df['date'].min().year} to {df['date'].max().year}")
                
                start_year = input("Start year [Enter for all]: ").strip()
                if start_year:
                    df = df[df['date'].dt.year >= int(start_year)]
                
                end_year = input("End year [Enter for all]: ").strip()
                if end_year:
                    df = df[df['date'].dt.year <= int(end_year)]
                
                print(f"✓ Filtered to {len(df):,} transcripts")
            
            # Rename text column
            text_col = 'content' if 'content' in df.columns else 'text'
            df = df.rename(columns={text_col: 'text'})
            
            # Optional sampling
            if len(df) > 1000:
                sample = input(f"\nSample for testing? (e.g., 100) [Enter for all]: ").strip()
                if sample:
                    df = df.sample(n=min(int(sample), len(df)), random_state=42)
                    print(f"✓ Sampled {len(df)} transcripts")
            
            return df
            
        except Exception as e:
            print(f"✗ Error: {e}")
            return pd.DataFrame()
    
    def split_transcript_sections(self, text: str) -> Dict[str, str]:
        """Split transcript into prepared remarks vs Q&A session"""
        qa_markers = [
            'question-and-answer',
            'questions and answers',
            'q&a session',
            'operator: ',
            'question:',
            'analyst:',
            'questions from analysts'
        ]
        
        text_lower = text.lower()
        qa_start = len(text)
        
        for marker in qa_markers:
            pos = text_lower.find(marker)
            if pos != -1 and pos < qa_start:
                qa_start = pos
        
        # If Q&A section found, split; otherwise, all is prepared
        if qa_start < len(text) * 0.8:
            return {
                'prepared_remarks': text[:qa_start],
                'qa_session': text[qa_start:]
            }
        else:
            return {
                'prepared_remarks': text,
                'qa_session': ''
            }
    
    def chunk_text(self, text: str, chunk_size: int = 3500) -> List[str]:
        """Split text into overlapping chunks"""
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        overlap = 500
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            chunks.append(text[start:end])
            start = end - overlap
            
            # Limit to 3 chunks for speed
            if len(chunks) >= 3:
                break
        
        return chunks
    
    def create_aspect_prompt(self, text: str, aspect: str, description: str) -> str:
        """Create chain-of-thought prompt for aspect scoring"""
        # Take chunk for analysis
        text_sample = text[:self.chunk_size]
        
        prompt = f"""[INST] You are an expert financial analyst. Analyze this earnings call transcript focusing on: {aspect}

{description}

Text: {text_sample}

Provide your analysis in this format:
Key Facts: [2-3 specific facts about {aspect}]
Score: [1-5 where 1=Very Negative, 3=Neutral, 5=Very Positive]
Reasoning: [Brief explanation]
[/INST]

Analysis:
"""
        return prompt
    
    def parse_aspect_response(self, response: str) -> Dict[str, any]:
        """Parse chain-of-thought response"""
        result = {
            'score': 3,
            'facts': [],
            'reasoning': ''
        }
        
        try:
            # Extract score
            score_match = re.search(r'Score:\s*(\d)', response, re.IGNORECASE)
            if score_match:
                score = int(score_match.group(1))
                if 1 <= score <= 5:
                    result['score'] = score
            
            # Extract facts
            facts_match = re.search(r'Key Facts?:\s*(.+?)(?=Score:|Reasoning:|$)', response, re.IGNORECASE | re.DOTALL)
            if facts_match:
                facts_text = facts_match.group(1).strip()
                # Extract bullet points or sentences
                facts = [f.strip('- •*').strip() for f in facts_text.split('\n') if f.strip()]
                result['facts'] = [f for f in facts if len(f) > 10][:3]
            
            # Extract reasoning
            reasoning_match = re.search(r'Reasoning:\s*(.+?)$', response, re.IGNORECASE | re.DOTALL)
            if reasoning_match:
                result['reasoning'] = reasoning_match.group(1).strip()[:200]
        
        except Exception as e:
            pass
        
        return result
    
    def score_section_aspects(self, text: str, section_name: str) -> Dict[str, Dict]:
        """Score all 5 aspects for a section using vLLM batch processing"""
        from vllm import SamplingParams
        
        aspects = {
            'revenue_growth': 'Revenue trends, sales growth, top-line performance',
            'profitability': 'Profit margins, costs, operating efficiency, bottom-line',
            'forward_guidance': 'Future outlook, guidance, predictions, expectations',
            'management_confidence': 'Management tone, confidence, certainty in statements',
            'competitive_position': 'Market share, competitive advantages, industry positioning'
        }
        
        # Create prompts for all aspects
        prompts = []
        aspect_names = []
        for aspect, description in aspects.items():
            prompt = self.create_aspect_prompt(text, aspect, description)
            prompts.append(prompt)
            aspect_names.append(aspect)
        
        # Batch inference with vLLM
        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=150,
            stop=["\n\n"]
        )
        
        outputs = self.llm.generate(prompts, sampling_params)
        
        # Parse results
        results = {}
        for i, output in enumerate(outputs):
            response = output.outputs[0].text
            aspect_name = aspect_names[i]
            results[aspect_name] = self.parse_aspect_response(response)
        
        return results
    
    def score_comprehensive(self, text: str, symbol: str = None, date: str = None) -> Dict:
        """Comprehensive multi-aspect scoring with section weighting"""
        # Split sections
        sections = self.split_transcript_sections(text)
        
        print(f"    Prepared remarks: {len(sections['prepared_remarks']):,} chars")
        print(f"    Q&A session: {len(sections['qa_session']):,} chars")
        
        results = {
            'symbol': symbol,
            'date': date,
            'timestamp': datetime.now().isoformat()
        }
        
        # Score prepared remarks
        print(f"    Scoring prepared remarks (5 aspects)...")
        prepared_scores = self.score_section_aspects(sections['prepared_remarks'], 'prepared')
        results['prepared_remarks'] = prepared_scores
        
        # Score Q&A if exists
        if sections['qa_session'] and len(sections['qa_session']) > 200:
            print(f"    Scoring Q&A session (5 aspects)...")
            qa_scores = self.score_section_aspects(sections['qa_session'], 'qa')
            results['qa_session'] = qa_scores
        else:
            results['qa_session'] = None
        
        # Calculate weighted composite (30% prepared, 70% Q&A)
        composite_scores = {}
        for aspect in ['revenue_growth', 'profitability', 'forward_guidance', 
                      'management_confidence', 'competitive_position']:
            prepared = prepared_scores[aspect]['score']
            
            if results['qa_session']:
                qa = qa_scores[aspect]['score']
                composite = 0.3 * prepared + 0.7 * qa
            else:
                composite = prepared
            
            composite_scores[aspect] = round(composite, 2)
        
        results['composite_scores'] = composite_scores
        
        # Overall sentiment
        results['overall_sentiment'] = round(np.mean(list(composite_scores.values())), 2)
        results['overall_sentiment_int'] = int(round(results['overall_sentiment']))
        
        return results
    
    def fetch_market_context(self, symbol: str, date: str) -> Dict:
        """Fetch market data from Yahoo Finance"""
        if symbol in self.historical_cache:
            return self.historical_cache[symbol]
        
        try:
            ticker = yf.Ticker(symbol)
            earnings_date = pd.to_datetime(date)
            
            # Fetch data window
            start = earnings_date - timedelta(days=7)
            end = earnings_date + timedelta(days=35)
            
            hist = ticker.history(start=start, end=end)
            
            if len(hist) == 0:
                return {}
            
            # Price at earnings
            earnings_price = hist.loc[hist.index >= earnings_date].iloc[0]['Close'] if len(hist.loc[hist.index >= earnings_date]) > 0 else None
            
            if earnings_price:
                # Calculate returns
                week_later = earnings_date + timedelta(days=7)
                month_later = earnings_date + timedelta(days=30)
                
                week_price = hist.loc[hist.index >= week_later].iloc[0]['Close'] if len(hist.loc[hist.index >= week_later]) > 0 else None
                month_price = hist.loc[hist.index >= month_later].iloc[0]['Close'] if len(hist.loc[hist.index >= month_later]) > 0 else None
                
                return_1week = ((week_price / earnings_price) - 1) * 100 if week_price else None
                return_1month = ((month_price / earnings_price) - 1) * 100 if month_price else None
                
                result = {
                    'price_at_earnings': round(earnings_price, 2),
                    'return_1week': round(return_1week, 2) if return_1week else None,
                    'return_1month': round(return_1month, 2) if return_1month else None
                }
                
                self.historical_cache[symbol] = result
                return result
        except:
            pass
        
        return {}
    
    def process_dataset(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict, List[Dict]]:
        """Process dataset with comprehensive analysis"""
        print("\n[3/5] Configuring Processing...")
        
        # Batch size
        batch_input = input(f"\nBatch size [default: 16, recommended: 8-32]: ").strip()
        batch_size = int(batch_input) if batch_input else 16
        
        # Market data
        fetch_market = False
        if 'symbol' in df.columns and 'date' in df.columns:
            market_input = input("\nFetch market data from Yahoo Finance? (y/n) [default: y]: ").strip().lower()
            fetch_market = market_input in ['y', 'yes', '']
        
        print(f"\n[4/5] Processing {len(df):,} transcripts...")
        print(f"Comprehensive mode with chain-of-thought")
        print(f"Batch size: {batch_size}")
        print(f"Market data: {'Yes' if fetch_market else 'No'}")
        print("\nThis will take ~3-4 hours for the full dataset")
        print("=" * 70)
        
        # Process
        results = []
        detailed_results = []
        market_data_list = []
        
        for i in tqdm(range(0, len(df), batch_size), desc="Processing batches"):
            batch_df = df.iloc[i:i+batch_size]
            
            for idx, row in batch_df.iterrows():
                symbol = row.get('symbol', None)
                date = row.get('date', None)
                if isinstance(date, pd.Timestamp):
                    date = date.strftime('%Y-%m-%d')
                
                print(f"\n  [{idx+1}/{len(df)}] {symbol} ({date})")
                
                # Comprehensive scoring
                detailed = self.score_comprehensive(row['text'], symbol, date)
                detailed_results.append(detailed)
                
                # Extract for CSV
                result = {
                    'overall_sentiment': detailed['overall_sentiment'],
                    'overall_sentiment_int': detailed['overall_sentiment_int'],
                    **detailed['composite_scores']
                }
                results.append(result)
                
                # Market data
                if fetch_market and symbol and date:
                    market_ctx = self.fetch_market_context(symbol, date)
                    market_data_list.append(market_ctx)
                else:
                    market_data_list.append({})
        
        # Add to dataframe
        df['overall_sentiment'] = [r['overall_sentiment'] for r in results]
        df['overall_sentiment_int'] = [r['overall_sentiment_int'] for r in results]
        df['revenue_growth'] = [r['revenue_growth'] for r in results]
        df['profitability'] = [r['profitability'] for r in results]
        df['forward_guidance'] = [r['forward_guidance'] for r in results]
        df['management_confidence'] = [r['management_confidence'] for r in results]
        df['competitive_position'] = [r['competitive_position'] for r in results]
        
        # Add market data
        if fetch_market:
            df['price_at_earnings'] = [m.get('price_at_earnings') for m in market_data_list]
            df['return_1week'] = [m.get('return_1week') for m in market_data_list]
            df['return_1month'] = [m.get('return_1month') for m in market_data_list]
        
        print(f"\n✓ Scored {len(df):,} transcripts")
        print(f"\nAverage Scores:")
        for col in ['overall_sentiment', 'revenue_growth', 'profitability',
                   'forward_guidance', 'management_confidence', 'competitive_position']:
            print(f"  {col}: {df[col].mean():.2f}")
        
        summary = {
            'mode': 'comprehensive',
            'total': len(df),
            'market_data_included': fetch_market,
            'averages': {col: float(df[col].mean())
                       for col in ['overall_sentiment', 'revenue_growth', 'profitability',
                                 'forward_guidance', 'management_confidence', 'competitive_position']}
        }
        
        return df, summary, detailed_results
    
    def save_results(self, df: pd.DataFrame, summary: Dict, detailed_results: List[Dict]):
        """Save comprehensive results"""
        print("\n[5/5] Saving Results...")
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save CSV
        csv_file = f"comprehensive_scores_{timestamp}.csv"
        df.to_csv(csv_file, index=False)
        print(f"✓ CSV: {csv_file}")
        
        # Save summary
        json_file = f"summary_{timestamp}.json"
        with open(json_file, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"✓ Summary: {json_file}")
        
        # Save detailed results with chain-of-thought
        jsonl_file = f"detailed_results_{timestamp}.jsonl"
        with open(jsonl_file, 'w') as f:
            for result in detailed_results:
                f.write(json.dumps(result) + '\n')
        print(f"✓ Detailed chain-of-thought: {jsonl_file}")
        
        print("\n" + "=" * 70)
        print("Processing Complete!")
        print("=" * 70)
        print(f"\nFiles created:")
        print(f"  1. {csv_file} - Sentiment scores with market data")
        print(f"  2. {json_file} - Summary statistics")
        print(f"  3. {jsonl_file} - Detailed reasoning and facts")
    
    def run(self):
        """Main execution"""
        if not self.setup_model():
            return
        
        df = self.load_data()
        if df.empty:
            return
        
        df_scored, summary, detailed = self.process_dataset(df)
        self.save_results(df_scored, summary, detailed)


def main():
    try:
        scorer = ComprehensiveVLLMScorer()
        scorer.run()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
