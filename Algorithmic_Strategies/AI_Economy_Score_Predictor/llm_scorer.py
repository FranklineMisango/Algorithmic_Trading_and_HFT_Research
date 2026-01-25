"""
LLM Scoring Module for AI Economy Score Predictor

Implements prompt engineering and LLM-based sentiment scoring of earnings transcripts.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import yaml
import re
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

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
            self.config = yaml.safe_load(f)
        
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
        # Remove safe harbor statements (common patterns)
        patterns_to_remove = [
            r"forward-looking statements.*?(?=\n\n|\Z)",
            r"safe harbor.*?(?=\n\n|\Z)",
            r"GAAP.*?reconciliation.*?(?=\n\n|\Z)",
            r"operator:.*?(?=\n)",
            r"\\[.*?\\]",  # Remove [Operator Instructions]
        ]
        
        for pattern in patterns_to_remove:
            text = re.sub(pattern, "", text, flags=re.IGNORECASE | re.DOTALL)
        
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
        
        Args:
            text: Text to chunk
            chunk_size: Max characters per chunk (from config if None)
            
        Returns:
            List of text chunks
        """
        if chunk_size is None:
            chunk_size = self.llm_config['chunk_size']
        
        # Split on paragraph boundaries
        paragraphs = text.split('\n\n')
        
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
    
    def score_text_openai(self, text: str) -> Optional[int]:
        """
        Score text using OpenAI API.
        
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
            response = self.client.chat.completions.create(
                model=self.llm_config['model'],
                messages=[
                    {"role": "system", "content": "You are an expert economic analyst."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.llm_config['temperature'],
                max_tokens=self.llm_config['max_tokens']
            )
            
            # Extract score
            score_text = response.choices[0].message.content.strip()
            
            # Parse score (should be single digit 1-5)
            score = int(score_text)
            
            if 1 <= score <= 5:
                return score
            else:
                print(f"Warning: Invalid score {score}, expected 1-5")
                return None
                
        except Exception as e:
            print(f"Error scoring with OpenAI: {e}")
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
                'confidence': 0.0
            }
        
        # Aggregate chunk scores (mean)
        firm_score = np.mean(chunk_scores)
        
        # Calculate confidence (inverse of variance)
        if len(chunk_scores) > 1:
            confidence = 1.0 / (1.0 + np.var(chunk_scores))
        else:
            confidence = 0.5
        
        return {
            'firm_score': firm_score,
            'chunk_scores': chunk_scores,
            'confidence': confidence,
            'num_chunks': len(chunks)
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
