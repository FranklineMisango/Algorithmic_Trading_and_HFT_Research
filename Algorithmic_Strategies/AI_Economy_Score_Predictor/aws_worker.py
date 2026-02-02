"""
AWS Worker Script for LLM Scoring on EC2 Spot Instances

This script runs on EC2 spot instances to:
1. Pull transcript batches from S3
2. Score them using LLM
3. Push results back to S3
4. Update job status in DynamoDB
"""

import os
import sys
import json
import boto3
import pandas as pd
from datetime import datetime
import argparse
import logging
from pathlib import Path
import time
import signal

# Local imports
from llm_scorer import LLMScorer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('/tmp/aws_worker.log')
    ]
)
logger = logging.getLogger(__name__)

class SpotInstanceHandler:
    """Handle EC2 spot instance interruption signals"""
    
    def __init__(self):
        self.interrupted = False
        signal.signal(signal.SIGTERM, self._handle_interruption)
        
    def _handle_interruption(self, signum, frame):
        logger.warning("Received spot instance interruption signal!")
        self.interrupted = True
        
    def is_interrupted(self):
        return self.interrupted


class AWSWorker:
    """Worker that processes scoring jobs from S3"""
    
    def __init__(self, bucket_name: str, config_path: str = "config.yaml"):
        """Initialize AWS worker with S3 bucket and config"""
        self.bucket_name = bucket_name
        self.s3_client = boto3.client('s3')
        self.dynamodb = boto3.resource('dynamodb')
        
        # Initialize LLM Scorer
        self.scorer = LLMScorer(config_path)
        
        # Spot instance handler
        self.spot_handler = SpotInstanceHandler()
        
        # Job tracking
        self.job_table = self.dynamodb.Table('transcript-scoring-jobs')
        
        logger.info(f"AWS Worker initialized with bucket: {bucket_name}")
        
    def get_job_from_queue(self) -> dict:
        """Get next job from DynamoDB queue"""
        try:
            # Scan for pending jobs
            response = self.job_table.scan(
                FilterExpression='job_status = :status',
                ExpressionAttributeValues={':status': 'pending'}
            )
            
            jobs = response.get('Items', [])
            if not jobs:
                return None
                
            # Get oldest job
            job = sorted(jobs, key=lambda x: x['created_at'])[0]
            
            # Mark as in-progress
            self.job_table.update_item(
                Key={'job_id': job['job_id']},
                UpdateExpression='SET job_status = :status, started_at = :time',
                ExpressionAttributeValues={
                    ':status': 'in_progress',
                    ':time': datetime.now().isoformat()
                }
            )
            
            return job
            
        except Exception as e:
            logger.error(f"Error getting job from queue: {e}")
            return None
    
    def download_batch_from_s3(self, s3_key: str) -> pd.DataFrame:
        """Download transcript batch from S3"""
        try:
            logger.info(f"Downloading batch from s3://{self.bucket_name}/{s3_key}")
            
            # Download to temp file
            local_path = f"/tmp/{Path(s3_key).name}"
            self.s3_client.download_file(self.bucket_name, s3_key, local_path)
            
            # Read into DataFrame
            if s3_key.endswith('.csv'):
                df = pd.read_csv(local_path)
            elif s3_key.endswith('.parquet'):
                df = pd.read_parquet(local_path)
            else:
                raise ValueError(f"Unsupported file format: {s3_key}")
                
            logger.info(f"Downloaded {len(df)} transcripts")
            return df
            
        except Exception as e:
            logger.error(f"Error downloading from S3: {e}")
            raise
    
    def upload_results_to_s3(self, df: pd.DataFrame, s3_key: str):
        """Upload scored results to S3"""
        try:
            logger.info(f"Uploading results to s3://{self.bucket_name}/{s3_key}")
            
            # Save to temp file
            local_path = f"/tmp/{Path(s3_key).name}"
            
            if s3_key.endswith('.csv'):
                df.to_csv(local_path, index=False)
            elif s3_key.endswith('.parquet'):
                df.to_parquet(local_path, index=False)
            else:
                raise ValueError(f"Unsupported file format: {s3_key}")
            
            # Upload to S3
            self.s3_client.upload_file(local_path, self.bucket_name, s3_key)
            
            logger.info(f"Uploaded {len(df)} scored transcripts")
            
        except Exception as e:
            logger.error(f"Error uploading to S3: {e}")
            raise
    
    def score_batch(self, transcripts_df: pd.DataFrame) -> pd.DataFrame:
        """Score a batch of transcripts"""
        scored_results = []
        errors = []
        
        # Determine transcript column
        transcript_col = None
        for col in ['transcript', 'text', 'content', 'full_text', 'body']:
            if col in transcripts_df.columns:
                transcript_col = col
                break
        
        if not transcript_col:
            raise ValueError("No transcript column found in DataFrame")
        
        logger.info(f"Scoring {len(transcripts_df)} transcripts using column: {transcript_col}")
        
        for idx, row in transcripts_df.iterrows():
            # Check for spot interruption
            if self.spot_handler.is_interrupted():
                logger.warning("Spot interruption detected, saving progress...")
                break
            
            try:
                symbol = row['symbol']
                date = row['date']
                transcript = row[transcript_col]
                
                logger.info(f"Scoring {symbol} on {date} ({idx+1}/{len(transcripts_df)})")
                
                # Score transcript
                score = self.scorer.score_transcript(transcript)
                
                # Store result
                scored_results.append({
                    'symbol': symbol,
                    'date': date,
                    'score': score,
                    'scored_at': datetime.now().isoformat()
                })
                
                # Rate limiting
                time.sleep(0.5)
                
            except Exception as e:
                logger.error(f"Error scoring {symbol} on {date}: {e}")
                errors.append({
                    'symbol': symbol,
                    'date': date,
                    'error': str(e)
                })
        
        # Convert to DataFrame
        scored_df = pd.DataFrame(scored_results)
        
        logger.info(f"Completed: {len(scored_df)} scored, {len(errors)} errors")
        
        return scored_df
    
    def update_job_status(self, job_id: str, status: str, **kwargs):
        """Update job status in DynamoDB"""
        try:
            update_expr = 'SET job_status = :status, updated_at = :time'
            expr_values = {
                ':status': status,
                ':time': datetime.now().isoformat()
            }
            
            # Add optional fields
            for key, value in kwargs.items():
                update_expr += f', {key} = :{key}'
                expr_values[f':{key}'] = value
            
            self.job_table.update_item(
                Key={'job_id': job_id},
                UpdateExpression=update_expr,
                ExpressionAttributeValues=expr_values
            )
            
            logger.info(f"Updated job {job_id} to status: {status}")
            
        except Exception as e:
            logger.error(f"Error updating job status: {e}")
    
    def process_job(self, job: dict):
        """Process a single job"""
        job_id = job['job_id']
        input_s3_key = job['input_s3_key']
        output_s3_key = job['output_s3_key']
        
        logger.info(f"Processing job: {job_id}")
        
        try:
            # Download batch
            transcripts_df = self.download_batch_from_s3(input_s3_key)
            
            # Score transcripts
            scored_df = self.score_batch(transcripts_df)
            
            # Upload results
            self.upload_results_to_s3(scored_df, output_s3_key)
            
            # Update job status
            self.update_job_status(
                job_id,
                'completed',
                completed_at=datetime.now().isoformat(),
                transcripts_scored=len(scored_df),
                output_s3_key=output_s3_key
            )
            
            logger.info(f"Job {job_id} completed successfully")
            
        except Exception as e:
            logger.error(f"Job {job_id} failed: {e}")
            self.update_job_status(
                job_id,
                'failed',
                error_message=str(e),
                failed_at=datetime.now().isoformat()
            )
    
    def run(self, max_jobs: int = None):
        """Main worker loop"""
        logger.info("AWS Worker started")
        
        jobs_processed = 0
        
        while True:
            # Check if max jobs reached
            if max_jobs and jobs_processed >= max_jobs:
                logger.info(f"Reached max jobs limit: {max_jobs}")
                break
            
            # Check for spot interruption
            if self.spot_handler.is_interrupted():
                logger.warning("Spot instance interrupted, shutting down gracefully")
                break
            
            # Get next job
            job = self.get_job_from_queue()
            
            if not job:
                logger.info("No pending jobs, waiting...")
                time.sleep(30)
                continue
            
            # Process job
            self.process_job(job)
            jobs_processed += 1
        
        logger.info(f"Worker finished. Processed {jobs_processed} jobs")


def main():
    parser = argparse.ArgumentParser(description='AWS Worker for LLM Scoring')
    parser.add_argument('--bucket', required=True, help='S3 bucket name')
    parser.add_argument('--config', default='config.yaml', help='Config file path')
    parser.add_argument('--max-jobs', type=int, default=None, help='Max jobs to process')
    
    args = parser.parse_args()
    
    # Initialize and run worker
    worker = AWSWorker(args.bucket, args.config)
    worker.run(max_jobs=args.max_jobs)


if __name__ == "__main__":
    main()
