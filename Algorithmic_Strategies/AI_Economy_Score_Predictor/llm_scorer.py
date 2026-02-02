"""
LLM Scoring Module for AI Economy Score Predictor

Implements prompt engineering and LLM-based sentiment scoring of earnings transcripts.
"""

import os
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import yaml
import re
from datetime import datetime
import time
import warnings
from dotenv import load_dotenv
warnings.filterwarnings('ignore')

# Load environment variables
load_dotenv()

try:
    import openai
except ImportError:
    print("Warning: openai not installed. Install with: pip install openai")

try:
    import anthropic
except ImportError:
    print("Warning: anthropic not installed. Install with: pip install anthropic")


class LLMScorer:
    """Handles LLM-based scoring of earnings call transcripts."""
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize with configuration."""
        with open(config_path, 'r') as f:
            config_content = f.read()
            # Expand environment variables in config
            config_content = os.path.expandvars(config_content)
            self.config = yaml.safe_load(config_content)
        
        self.llm_config = self.config['llm']
        self.provider = self.llm_config['provider']
        
        # Initialize LLM client
        if self.provider == 'openai':
            api_key = self.llm_config.get('api_key')
            if api_key and api_key != "YOUR_OPENAI_API_KEY":
                self.client = openai.OpenAI(api_key=api_key)
            else:
                self.client = None
                print("Warning: OpenAI API key not configured")
        
        elif self.provider == 'anthropic':
            api_key = self.llm_config.get('api_key')
            if api_key and api_key != "YOUR_ANTHROPIC_API_KEY":
                self.client = anthropic.Anthropic(api_key=api_key)
            else:
                self.client = None
                print("Warning: Anthropic API key not configured")
        
        else:
            self.client = None
            print(f"Warning: {self.provider} not yet supported")
    
    def clean_transcript(self, text: str) -> str:
        """
        Clean transcript text before scoring.
        
        Removes:
        - Boilerplate safe harbor statements
        - Presenter lists
        - Excessive whitespace
        
        Args:
            text: Raw transcript text
            
        Returns:
            Cleaned text
        """
        # Only remove truly boilerplate content, keep speaker tags minimal
        # Most cleaning was too aggressive - transcripts need context
        
        # Remove operator instructions in brackets
        text = re.sub(r'\[Operator Instructions\]', '', text, flags=re.IGNORECASE)
        
        # Remove excessive whitespace
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r' {2,}', ' ', text)
        
        return text.strip()
    
    def extract_md_and_a(self, text: str) -> str:
        """
        Extract Management Discussion & Analysis section.
        
        Args:
            text: Full transcript
            
        Returns:
            MD&A section text
        """
        # Look for common section markers
        md_a_markers = [
            r"prepared remarks",
            r"management discussion",
            r"opening remarks",
            r"CEO remarks"
        ]
        
        qa_markers = [
            r"question-and-answer",
            r"Q&A",
            r"questions and answers",
            r"operator.*questions"
        ]
        
        # Find MD&A start
        md_a_start = 0
        for marker in md_a_markers:
            match = re.search(marker, text, re.IGNORECASE)
            if match:
                md_a_start = match.start()
                break
        
        # Find Q&A start (MD&A ends here)
        qa_start = len(text)
        for marker in qa_markers:
            match = re.search(marker, text, re.IGNORECASE)
            if match:
                qa_start = match.start()
                break
        
        return text[md_a_start:qa_start]
    
    def chunk_text(
        self,
        text: str,
        chunk_size: Optional[int] = None
    ) -> List[str]:
        """
        Split text into chunks for LLM processing.
        Handles both paragraph-based and sentence-based splitting.
        
        Args:
            text: Text to chunk
            chunk_size: Max characters per chunk (from config if None)
            
        Returns:
            List of text chunks
        """
        if chunk_size is None:
            chunk_size = self.llm_config['chunk_size']
        
        # If text is shorter than chunk_size, return as single chunk
        if len(text) <= chunk_size:
            return [text]
        
        # Try splitting on paragraph boundaries first
        paragraphs = text.split('\n\n')
        
        # If we have meaningful paragraphs, use them
        if len(paragraphs) > 1 and min(len(p) for p in paragraphs) > 100:
            chunks = []
            current_chunk = ""
            
            for para in paragraphs:
                if len(current_chunk) + len(para) <= chunk_size:
                    current_chunk += para + "\n\n"
                else:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    current_chunk = para + "\n\n"
            
            if current_chunk:
                chunks.append(current_chunk.strip())
            
            return chunks
        
        # No paragraph breaks - split on sentences or character boundaries
        # Split on common sentence boundaries
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence) <= chunk_size:
                current_chunk += sentence + " "
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                # If single sentence is too long, force split it
                if len(sentence) > chunk_size:
                    for i in range(0, len(sentence), chunk_size):
                        chunks.append(sentence[i:i+chunk_size])
                    current_chunk = ""
                else:
                    current_chunk = sentence + " "
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks if chunks else [text]
    
    def score_text_openai(self, text: str, timeout: int = 30, max_retries: int = 5) -> Optional[int]:
        """
        Score text using OpenAI API with timeout, rate limit handling, and retries.
        
        Args:
            text: Text to score
            timeout: API timeout in seconds
            max_retries: Maximum retry attempts for rate limits
            
        Returns:
            Score (1-5) or None if error
        """
        if self.client is None:
            return None
        
        # Format prompt once
        prompt = self.llm_config['prompt_template'].format(text=text)
        
        for attempt in range(max_retries):
            try:
                # Call API with timeout and optimizations
                response = self.client.chat.completions.create(
                    model=self.llm_config['model'],
                    messages=[
                        {"role": "system", "content": "You are an expert economic analyst."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=self.llm_config['temperature'],
                    max_tokens=self.llm_config['max_tokens'],
                    timeout=timeout
                )
                
                # Extract score
                score_text = response.choices[0].message.content.strip()
            
                # Parse score - extract first digit found
                # Handle cases where LLM returns "5" or "The score is 5" or just text
                match = re.search(r'\b([1-5])\b', score_text)
                if match:
                    score = int(match.group(1))
                    return score
                else:
                    # Try to find any digit
                    digits = re.findall(r'\d', score_text)
                    if digits:
                        score = int(digits[0])
                        if 1 <= score <= 5:
                            return score
                    
                    print(f"Warning: Could not parse score from: {score_text[:100]}")
                    return None
                    
            except openai.RateLimitError as e:
                # Exponential backoff for rate limits
                wait_time = (2 ** attempt) + (np.random.random() * 0.1)  # 1, 2, 4, 8, 16 seconds + jitter
                if attempt < max_retries - 1:
                    print(f"Rate limit hit, waiting {wait_time:.1f}s (attempt {attempt + 1}/{max_retries})...")
                    time.sleep(wait_time)
                else:
                    print(f"Rate limit exceeded after {max_retries} attempts: {e}")
                    return None
                    
            except openai.APITimeoutError as e:
                # Retry timeouts with shorter backoff
                wait_time = 1 + attempt
                if attempt < max_retries - 1:
                    print(f"Timeout, retrying in {wait_time}s (attempt {attempt + 1}/{max_retries})...")
                    time.sleep(wait_time)
                else:
                    print(f"Timeout after {max_retries} attempts: {e}")
                    return None
                    
            except openai.APIError as e:
                # Retry API errors with backoff
                if "overloaded" in str(e).lower() or "capacity" in str(e).lower():
                    wait_time = (2 ** attempt) + np.random.random()
                    if attempt < max_retries - 1:
                        print(f"API overloaded, waiting {wait_time:.1f}s (attempt {attempt + 1}/{max_retries})...")
                        time.sleep(wait_time)
                    else:
                        print(f"API error after {max_retries} attempts: {e}")
                        return None
                else:
                    print(f"API error: {e}")
                    return None
                    
            except Exception as e:
                print(f"Error scoring with OpenAI: {e}")
                return None
        
        return None
    
    def score_text_anthropic(self, text: str) -> Optional[int]:
        """
        Score text using Anthropic Claude API.
        
        Args:
            text: Text to score
            
        Returns:
            Score (1-5) or None if error
        """
        if self.client is None:
            return None
        
        try:
            # Format prompt
            prompt = self.llm_config['prompt_template'].format(text=text)
            
            # Call API
            message = self.client.messages.create(
                model=self.llm_config['model'],
                max_tokens=self.llm_config['max_tokens'],
                temperature=self.llm_config['temperature'],
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            # Extract score
            score_text = message.content[0].text.strip()
            score = int(score_text)
            
            if 1 <= score <= 5:
                return score
            else:
                print(f"Warning: Invalid score {score}, expected 1-5")
                return None
                
        except Exception as e:
            print(f"Error scoring with Anthropic: {e}")
            return None
    
    def score_text(self, text: str) -> Optional[int]:
        """
        Score text using configured LLM provider.
        
        Args:
            text: Text to score
            
        Returns:
            Score (1-5) or None if error
        """
        if self.provider == 'openai':
            return self.score_text_openai(text)
        elif self.provider == 'anthropic':
            return self.score_text_anthropic(text)
        else:
            # Placeholder for local models or other providers
            return np.random.randint(1, 6)  # Random 1-5 for testing
    
    def score_transcript(
        self,
        transcript: Dict[str, str],
        use_md_a_only: bool = True
    ) -> Dict[str, float]:
        """
        Score an entire transcript.
        
        Args:
            transcript: Dict with 'full_text', 'md&a', 'qa' keys
            use_md_a_only: If True, score only MD&A section
            
        Returns:
            Dict with 'firm_score', 'chunk_scores', 'confidence'
        """
        # Select text to score
        if use_md_a_only and 'md&a' in transcript:
            text = transcript['md&a']
        else:
            text = transcript['full_text']
        
        # Clean text
        text = self.clean_transcript(text)
        
        # If focus on MD&A, extract it
        if use_md_a_only:
            text = self.extract_md_and_a(text)
        
        # Chunk text
        chunks = self.chunk_text(text)
        
        # Score each chunk
        chunk_scores = []
        for chunk in chunks:
            score = self.score_text(chunk)
            if score is not None:
                chunk_scores.append(score)
        
        if not chunk_scores:
            return {
                'firm_score': None,
                'chunk_scores': [],
                'confidence': 0.0,
                'aggregation_method': 'none'
            }
        
        # Advanced aggregation methods
        aggregation = self._aggregate_scores(chunk_scores)
        
        return {
            'firm_score': aggregation['firm_score'],
            'chunk_scores': chunk_scores,
            'confidence': aggregation['confidence'],
            'num_chunks': len(chunks),
            'aggregation_method': aggregation['method'],
            'score_std': aggregation['std'],
            'score_range': aggregation['range'],
            'sentiment_trend': aggregation.get('trend', 0.0)
        }
    
    def _aggregate_scores(self, scores: List[int]) -> Dict:
        """
        Advanced score aggregation with multiple statistical methods.
        
        Methods applied:
        1. Trimmed mean (remove outliers)
        2. Position-weighted (early chunks = guidance, more weight)
        3. Variance-based confidence
        4. Sentiment trajectory analysis
        
        Args:
            scores: List of chunk scores
            
        Returns:
            Dict with aggregated score and metadata
        """
        scores_array = np.array(scores)
        
        # Method 1: Simple mean (baseline)
        simple_mean = np.mean(scores_array)
        
        # Method 2: Trimmed mean (remove top/bottom 10% if enough samples)
        if len(scores) >= 10:
            trim_pct = 0.1
            from scipy import stats
            trimmed_mean = stats.trim_mean(scores_array, trim_pct)
        else:
            trimmed_mean = simple_mean
        
        # Method 3: Weighted by position (early chunks = forward guidance)
        # Weight decays: first 25% chunks get 1.5x, middle 50% get 1.0x, last 25% get 0.8x
        n = len(scores)
        weights = np.ones(n)
        
        # Early chunks (first 25%) - forward-looking statements
        early_cutoff = max(1, n // 4)
        weights[:early_cutoff] = 1.5
        
        # Late chunks (last 25%) - Q&A, often repetitive
        late_cutoff = max(1, n * 3 // 4)
        weights[late_cutoff:] = 0.8
        
        weights = weights / weights.sum()  # Normalize
        weighted_mean = np.average(scores_array, weights=weights)
        
        # Method 4: Median (robust to outliers)
        median_score = np.median(scores_array)
        
        # Method 5: Sentiment trajectory (is sentiment improving/declining?)
        if len(scores) >= 3:
            # Fit linear trend
            x = np.arange(len(scores))
            slope, _ = np.polyfit(x, scores_array, 1)
            sentiment_trend = slope  # Positive = improving, negative = declining
        else:
            sentiment_trend = 0.0
        
        # Choose final score based on context
        # Use weighted mean as primary, adjusted by trend
        if len(scores) >= 10:
            # For longer transcripts, use trimmed + weighted
            firm_score = (trimmed_mean * 0.6 + weighted_mean * 0.4)
        else:
            # For shorter transcripts, use weighted mean
            firm_score = weighted_mean
        
        # Adjust slightly for strong trends (momentum matters)
        if abs(sentiment_trend) > 0.1:
            trend_adjustment = sentiment_trend * 0.1  # Max ±0.2 adjustment
            firm_score += trend_adjustment
        
        # Clip to valid range
        firm_score = np.clip(firm_score, 1.0, 5.0)
        
        # Calculate confidence metrics
        std = np.std(scores_array)
        score_range = (float(np.min(scores_array)), float(np.max(scores_array)))
        
        # Confidence: higher when scores agree (low variance)
        # and when we have many samples
        sample_confidence = min(1.0, len(scores) / 20.0)  # Max at 20+ chunks
        variance_confidence = 1.0 / (1.0 + std)
        confidence = (sample_confidence * 0.4 + variance_confidence * 0.6)
        
        return {
            'firm_score': float(firm_score),
            'method': 'weighted_trimmed_trend',
            'confidence': float(confidence),
            'std': float(std),
            'range': score_range,
            'trend': float(sentiment_trend),
            'components': {
                'simple_mean': float(simple_mean),
                'trimmed_mean': float(trimmed_mean),
                'weighted_mean': float(weighted_mean),
                'median': float(median_score)
            }
        }
    
    def score_multiple_transcripts(
        self,
        transcripts: List[Dict[str, any]]
    ) -> pd.DataFrame:
        """
        Score multiple transcripts.
        
        Args:
            transcripts: List of dicts with 'symbol', 'date', 'text' keys
            
        Returns:
            DataFrame with scores
        """
        results = []
        
        for i, transcript in enumerate(transcripts):
            print(f"Scoring transcript {i+1}/{len(transcripts)}: {transcript.get('symbol', 'Unknown')}")
            
            # Score transcript
            score_result = self.score_transcript(transcript)
            
            # Combine metadata with score
            result = {
                'symbol': transcript.get('symbol'),
                'date': transcript.get('date'),
                'quarter': transcript.get('quarter'),
                'year': transcript.get('year'),
                'firm_score': score_result['firm_score'],
                'confidence': score_result['confidence'],
                'num_chunks': score_result['num_chunks']
            }
            
            results.append(result)
        
        return pd.DataFrame(results)
    
    def aggregate_national_score(
        self,
        firm_scores: pd.DataFrame,
        weighting: str = 'value'
    ) -> pd.DataFrame:
        """
        Aggregate firm scores into national AGG_t score.
        
        Args:
            firm_scores: DataFrame with firm-level scores
            weighting: 'value' (market cap) or 'equal'
            
        Returns:
            DataFrame with quarterly AGG_t scores
        """
        # Group by quarter
        if weighting == 'value':
            # Would use actual market cap weights
            # Placeholder: equal weight for now
            agg = firm_scores.groupby(['year', 'quarter'])['firm_score'].mean()
        else:
            agg = firm_scores.groupby(['year', 'quarter'])['firm_score'].mean()
        
        agg_df = agg.reset_index()
        agg_df.columns = ['year', 'quarter', 'agg_score']
        
        return agg_df
    
    def aggregate_industry_scores(
        self,
        firm_scores: pd.DataFrame,
        industry_mapping: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Aggregate firm scores by GICS sector/industry.
        
        Args:
            firm_scores: DataFrame with firm-level scores
            industry_mapping: DataFrame mapping symbols to GICS sectors
            
        Returns:
            DataFrame with industry-level IND_k,t scores
        """
        # Merge with industry mapping
        merged = firm_scores.merge(
            industry_mapping[['symbol', 'gics_sector']],
            on='symbol',
            how='left'
        )
        
        # Group by industry and quarter
        ind_scores = merged.groupby(
            ['gics_sector', 'year', 'quarter']
        )['firm_score'].mean().reset_index()
        
        ind_scores.columns = ['gics_sector', 'year', 'quarter', 'ind_score']
        
        return ind_scores


# Test code
if __name__ == "__main__":
    scorer = LLMScorer('config.yaml')
    
    # Test text cleaning
    sample_text = """
    Forward-looking statements: This call contains forward-looking statements...
    
    CEO: I'm pleased to report strong financial performance this quarter.
    The US economy continues to show resilience despite headwinds.
    
    Question-and-answer session begins now.
    """
    
    cleaned = scorer.clean_transcript(sample_text)
    print(f"Cleaned text:\n{cleaned}\n")
    
    # Test MD&A extraction
    md_a = scorer.extract_md_and_a(sample_text)
    print(f"MD&A section:\n{md_a}\n")
    
    # Test chunking
    chunks = scorer.chunk_text(cleaned, chunk_size=100)
    print(f"Chunks: {len(chunks)}")
    
    # Test scoring (with placeholder)
    score = scorer.score_text(cleaned)
    print(f"Score: {score}")
