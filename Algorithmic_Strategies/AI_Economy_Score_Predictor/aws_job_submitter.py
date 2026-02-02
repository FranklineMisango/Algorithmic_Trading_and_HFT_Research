"""
AWS Job Submitter - Submit scoring jobs to be processed by EC2 spot instances

This script:
1. Splits transcripts into batches
2. Uploads batches to S3
3. Creates jobs in DynamoDB
4. Optionally launches EC2 spot instances
"""

import os
import boto3
import pandas as pd
from datetime import datetime
import argparse
import logging
from pathlib import Path
import uuid
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AWSJobSubmitter:
    """Submit transcript scoring jobs to AWS"""
    
    def __init__(self, bucket_name: str, region: str = 'us-east-1'):
        """Initialize AWS job submitter"""
        self.bucket_name = bucket_name
        self.region = region
        self.s3_client = boto3.client('s3', region_name=region)
        self.dynamodb = boto3.resource('dynamodb', region_name=region)
        self.ec2_client = boto3.client('ec2', region_name=region)
        
        # DynamoDB table for job tracking
        self.job_table = self.dynamodb.Table('transcript-scoring-jobs')
        
        logger.info(f"Initialized job submitter for bucket: {bucket_name}")
    
    def create_batch_files(self, transcripts_df: pd.DataFrame, batch_size: int = 50) -> list:
        """Split transcripts into batches and upload to S3"""
        logger.info(f"Creating batches of {batch_size} transcripts each")
        
        batch_keys = []
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        for i in range(0, len(transcripts_df), batch_size):
            batch_df = transcripts_df.iloc[i:i+batch_size]
            batch_num = i // batch_size + 1
            
            # Create S3 key
            s3_key = f"input/batches/{timestamp}/batch_{batch_num:04d}.csv"
            
            # Save to temp file
            temp_file = f"/tmp/batch_{batch_num:04d}.csv"
            batch_df.to_csv(temp_file, index=False)
            
            # Upload to S3
            self.s3_client.upload_file(temp_file, self.bucket_name, s3_key)
            
            logger.info(f"Uploaded batch {batch_num} ({len(batch_df)} transcripts) to s3://{self.bucket_name}/{s3_key}")
            
            batch_keys.append(s3_key)
            
            # Clean up temp file
            os.remove(temp_file)
        
        logger.info(f"Created {len(batch_keys)} batches")
        return batch_keys
    
    def create_jobs(self, batch_keys: list) -> list:
        """Create jobs in DynamoDB for each batch"""
        logger.info(f"Creating {len(batch_keys)} jobs in DynamoDB")
        
        job_ids = []
        
        for batch_key in batch_keys:
            job_id = str(uuid.uuid4())
            
            # Output S3 key
            output_key = batch_key.replace('input/batches/', 'output/scored/')
            
            # Create job record
            self.job_table.put_item(Item={
                'job_id': job_id,
                'input_s3_key': batch_key,
                'output_s3_key': output_key,
                'job_status': 'pending',
                'created_at': datetime.now().isoformat(),
                'updated_at': datetime.now().isoformat()
            })
            
            job_ids.append(job_id)
            logger.info(f"Created job {job_id} for batch {batch_key}")
        
        logger.info(f"Created {len(job_ids)} jobs")
        return job_ids
    
    def launch_spot_instances(self, 
                             num_instances: int = 2, 
                             instance_type: str = 't3.medium',
                             ami_id: str = None,
                             key_name: str = None,
                             security_group_id: str = None,
                             iam_instance_profile: str = None):
        """Launch EC2 spot instances to process jobs"""
        
        if not ami_id:
            # Use Ubuntu 22.04 LTS AMI for us-east-1
            ami_id = 'ami-0030e4319cbf4dbf2'  # Ubuntu 22.04 LTS us-east-1
        
        logger.info(f"Launching {num_instances} spot instances of type {instance_type}")
        
        # User data script to install dependencies and run worker
        user_data = f"""#!/bin/bash
set -e

# Update system
yum update -y

# Install Python 3.11
yum install -y python3.11 python3.11-pip git

# Install AWS CLI
pip3.11 install --upgrade awscli

# Clone repo or download code
cd /home/ec2-user
aws s3 cp s3://{self.bucket_name}/code/worker-code.zip ./worker-code.zip
unzip worker-code.zip
cd worker

# Install Python dependencies
pip3.11 install -r requirements.txt

# Run worker
python3.11 aws_worker.py --bucket {self.bucket_name} --config config.yaml > /var/log/worker.log 2>&1
"""
        
        # Build launch specification (TagSpecifications not supported in spot requests)
        launch_specification = {
            'ImageId': ami_id,
            'InstanceType': instance_type,
            'UserData': user_data,
        }
        
        # Add optional parameters only if provided
        if key_name:
            launch_specification['KeyName'] = key_name
        
        if security_group_id:
            launch_specification['SecurityGroupIds'] = [security_group_id]
        
        if iam_instance_profile:
            launch_specification['IamInstanceProfile'] = {'Name': iam_instance_profile}
        
        try:
            # Request spot instances
            response = self.ec2_client.request_spot_instances(
                InstanceCount=num_instances,
                Type='one-time',
                LaunchSpecification=launch_specification
            )
            
            spot_request_ids = [req['SpotInstanceRequestId'] for req in response['SpotInstanceRequests']]
            
            logger.info(f"Launched spot instance requests: {spot_request_ids}")
            logger.info("Instances will start processing jobs once they're running")
            
            return spot_request_ids
            
        except Exception as e:
            logger.error(f"Error launching spot instances: {e}")
            logger.info("You can manually launch instances and run the worker script")
            return []
    
    def get_job_status(self, job_ids: list = None) -> pd.DataFrame:
        """Get status of all jobs or specific jobs"""
        if job_ids:
            # Get specific jobs
            jobs = []
            for job_id in job_ids:
                response = self.job_table.get_item(Key={'job_id': job_id})
                if 'Item' in response:
                    jobs.append(response['Item'])
        else:
            # Get all jobs
            response = self.job_table.scan()
            jobs = response.get('Items', [])
        
        if not jobs:
            return pd.DataFrame()
        
        return pd.DataFrame(jobs)
    
    def download_results(self, output_dir: str = './results') -> pd.DataFrame:
        """Download all completed results from S3"""
        logger.info(f"Downloading results to {output_dir}")
        
        Path(output_dir).mkdir(exist_ok=True, parents=True)
        
        # List all output files
        response = self.s3_client.list_objects_v2(
            Bucket=self.bucket_name,
            Prefix='output/scored/'
        )
        
        all_results = []
        
        for obj in response.get('Contents', []):
            s3_key = obj['Key']
            
            if not s3_key.endswith('.csv'):
                continue
            
            # Download file
            local_path = os.path.join(output_dir, Path(s3_key).name)
            self.s3_client.download_file(self.bucket_name, s3_key, local_path)
            
            # Read and append
            df = pd.read_csv(local_path)
            all_results.append(df)
            
            logger.info(f"Downloaded {len(df)} results from {s3_key}")
        
        if all_results:
            combined_df = pd.concat(all_results, ignore_index=True)
            logger.info(f"Total results: {len(combined_df)}")
            return combined_df
        else:
            logger.warning("No results found")
            return pd.DataFrame()


def main():
    parser = argparse.ArgumentParser(description='Submit transcript scoring jobs to AWS')
    parser.add_argument('--bucket', required=True, help='S3 bucket name')
    parser.add_argument('--input-file', required=True, help='Input CSV file with transcripts')
    parser.add_argument('--batch-size', type=int, default=50, help='Transcripts per batch')
    parser.add_argument('--launch-instances', action='store_true', help='Launch EC2 spot instances')
    parser.add_argument('--num-instances', type=int, default=2, help='Number of spot instances')
    parser.add_argument('--instance-type', default='t3.medium', help='EC2 instance type')
    parser.add_argument('--region', default='us-east-1', help='AWS region')
    
    args = parser.parse_args()
    
    # Load transcripts
    logger.info(f"Loading transcripts from {args.input_file}")
    transcripts_df = pd.read_csv(args.input_file)
    logger.info(f"Loaded {len(transcripts_df)} transcripts")
    
    # Initialize submitter
    submitter = AWSJobSubmitter(args.bucket, args.region)
    
    # Create batches and upload to S3
    batch_keys = submitter.create_batch_files(transcripts_df, args.batch_size)
    
    # Create jobs in DynamoDB
    job_ids = submitter.create_jobs(batch_keys)
    
    # Optionally launch spot instances
    if args.launch_instances:
        submitter.launch_spot_instances(
            num_instances=args.num_instances,
            instance_type=args.instance_type
        )
    
    logger.info(f"\nJob submission complete!")
    logger.info(f"Created {len(job_ids)} jobs")
    logger.info(f"\nMonitor jobs with:")
    logger.info(f"  python aws_job_submitter.py --bucket {args.bucket} --status")
    logger.info(f"\nDownload results with:")
    logger.info(f"  python aws_job_submitter.py --bucket {args.bucket} --download")


if __name__ == "__main__":
    main()
