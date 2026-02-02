# AWS Deployment Guide for Transcript Scoring

This guide explains how to deploy the transcript scoring pipeline to AWS using EC2 spot instances and S3 storage.

## Architecture Overview

```
┌─────────────────┐
│  Your Notebook  │
│   (Submitter)   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   DynamoDB      │◄──────┐
│  (Job Queue)    │       │
└────────┬────────┘       │
         │                │
         ▼                │
┌─────────────────┐       │
│  EC2 Spot       │       │
│  Instances      │───────┘
│  (Workers)      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   S3 Bucket     │
│  Input/Output   │
└─────────────────┘
```

## Benefits

- **Cost Savings**: EC2 spot instances are ~70% cheaper than on-demand
- **Scalability**: Add more workers as needed
- **Fault Tolerance**: Jobs automatically retry if spot instances terminate
- **No Local Resources**: Offload CPU/memory intensive work to cloud

## Setup Instructions

### 1. Prerequisites

Install required packages:
```bash
pip install boto3 rich pandas
```

Configure AWS credentials:
```bash
aws configure
# Enter your AWS Access Key ID, Secret Access Key, and region
```

### 2. Run Setup Script

```bash
bash aws_setup.sh
```

This will:
- Create S3 bucket for storing transcripts and results
- Create DynamoDB table for job tracking
- Create IAM roles with proper permissions
- Upload worker code to S3

Or use CloudFormation:
```bash
aws cloudformation create-stack \
  --stack-name transcript-scoring \
  --template-body file://aws_infrastructure.yaml \
  --capabilities CAPABILITY_NAMED_IAM
```

### 3. Prepare Your API Keys

Store your OpenAI/Anthropic API key in AWS Secrets Manager:
```bash
aws secretsmanager create-secret \
  --name transcript-scorer/openai-api-key \
  --secret-string "your-api-key-here"
```

Update `config.yaml` to use Secrets Manager:
```yaml
llm:
  provider: openai
  api_key: ${AWS_SECRET:transcript-scorer/openai-api-key}
```

## Usage

### Submit Jobs

From your notebook or Python script:

```python
from aws_job_submitter import AWSJobSubmitter
import pandas as pd

# Load your transcripts
transcripts_df = pd.read_csv('scoring_transcripts.csv')

# Initialize submitter
submitter = AWSJobSubmitter(
    bucket_name='transcript-scoring-123456789',
    region='us-east-1'
)

# Split into batches and upload to S3
batch_keys = submitter.create_batch_files(transcripts_df, batch_size=50)

# Create jobs in DynamoDB
job_ids = submitter.create_jobs(batch_keys)

# Launch spot instances to process jobs
submitter.launch_spot_instances(
    num_instances=3,
    instance_type='t3.medium'
)

print(f"Submitted {len(job_ids)} jobs")
```

Or via command line:
```bash
python aws_job_submitter.py \
  --bucket transcript-scoring-123456789 \
  --input-file scoring_transcripts.csv \
  --batch-size 50 \
  --launch-instances \
  --num-instances 3 \
  --instance-type t3.medium
```

### Monitor Progress

Live monitoring:
```bash
python aws_monitor.py --bucket transcript-scoring-123456789 --watch
```

One-time status check:
```bash
python aws_monitor.py --bucket transcript-scoring-123456789
```

### Download Results

```bash
python aws_monitor.py \
  --bucket transcript-scoring-123456789 \
  --download \
  --output scored_transcripts_final.csv
```

Or in Python:
```python
from aws_monitor import AWSJobMonitor

monitor = AWSJobMonitor('transcript-scoring-123456789')
results_df = monitor.download_results('scored_transcripts.csv')

print(f"Downloaded {len(results_df)} scored transcripts")
print(results_df.head())
```

## Cost Estimation

### EC2 Spot Instances
- **t3.medium**: ~$0.01/hour (spot) vs $0.042/hour (on-demand)
- **t3.large**: ~$0.02/hour (spot) vs $0.083/hour (on-demand)

### Example: Scoring 1000 transcripts
- Batch size: 50 (20 batches)
- Workers: 3 instances (t3.medium)
- Time per transcript: 2 seconds
- Total time: ~11 minutes

**Cost breakdown:**
- EC2: 3 instances × $0.01/hr × 0.18 hours = **$0.005**
- S3: ~10MB storage = **$0.0002**
- DynamoDB: 20 writes + 60 reads = **$0.0003**
- OpenAI API: 1000 × $0.001 = **$1.00**

**Total: ~$1.01** (vs running locally with no cost savings)

### For 10,000 transcripts:
- EC2: **$0.05**
- API calls: **$10.00**
- **Total: ~$10.05**

## Advanced Configuration

### Spot Instance Best Practices

1. **Use multiple instance types** for better availability:
```python
submitter.launch_spot_instances(
    num_instances=5,
    instance_type='t3.medium,t3.large',  # Comma-separated
)
```

2. **Set appropriate max price**:
```python
# Default: on-demand price
# Set to 30-40% of on-demand for typical spot pricing
spot_price = '0.02'  # USD per hour
```

3. **Use diversified availability zones**:
- CloudFormation template handles this automatically
- Reduces likelihood of all instances terminating

### Scaling Workers

Start with fewer instances and add more if needed:
```bash
# Start with 2 workers
python aws_job_submitter.py --num-instances 2 ...

# Later, add 3 more workers
python aws_job_submitter.py --num-instances 3 --launch-instances
```

### Handling Failures

Jobs automatically retry if:
- Spot instance terminates mid-job
- API call fails
- Network issues

Failed jobs remain in DynamoDB with status='failed'. Review and resubmit:
```python
from aws_monitor import AWSJobMonitor

monitor = AWSJobMonitor('your-bucket')
jobs_df = monitor.get_all_jobs()
failed_jobs = jobs_df[jobs_df['job_status'] == 'failed']

print(f"Failed jobs: {len(failed_jobs)}")
print(failed_jobs[['job_id', 'error_message']])
```

## Integration with Existing Notebook

Replace the scoring cell in `00_full_pipeline.ipynb`:

```python
# OLD: Local scoring
# scored_data = score_quarter_transcripts(
#     scoring_transcripts, 
#     scorer, 
#     save_path=save_path
# )

# NEW: AWS-based scoring
from aws_job_submitter import AWSJobSubmitter
from aws_monitor import AWSJobMonitor

print("Submitting to AWS...")
submitter = AWSJobSubmitter('transcript-scoring-123456789')

# Submit jobs
batch_keys = submitter.create_batch_files(scoring_transcripts, batch_size=50)
job_ids = submitter.create_jobs(batch_keys)
submitter.launch_spot_instances(num_instances=2)

print(f"Submitted {len(job_ids)} jobs to AWS")
print("Monitor progress with: python aws_monitor.py --bucket transcript-scoring-123456789 --watch")

# Wait for completion (or check later)
monitor = AWSJobMonitor('transcript-scoring-123456789')

import time
while True:
    summary = monitor.get_job_summary()
    if summary['pending'] == 0 and summary['in_progress'] == 0:
        break
    print(f"Progress: {summary['completed']}/{summary['total']} completed")
    time.sleep(30)

# Download results
scored_data = monitor.download_results('scored_transcripts.csv')
print(f"Downloaded {len(scored_data)} scored transcripts")
```

## Cleanup

Remove all AWS resources:
```bash
# Delete CloudFormation stack
aws cloudformation delete-stack --stack-name transcript-scoring

# Or manually:
aws s3 rb s3://transcript-scoring-123456789 --force
aws dynamodb delete-table --table-name transcript-scoring-jobs
aws iam delete-instance-profile --instance-profile-name transcript-scorer-profile
aws iam delete-role --role-name transcript-scorer-ec2-role
```

## Troubleshooting

### No spot capacity
If spot instances can't launch:
- Try different instance types (t3.large, c5.large)
- Try different regions (us-east-1, us-west-2)
- Increase max spot price

### Jobs not processing
Check worker logs:
```bash
# SSH into instance
ssh -i your-key.pem ec2-user@instance-ip

# View logs
tail -f /var/log/worker.log
```

### API rate limits
If hitting OpenAI rate limits:
- Reduce number of workers
- Add delay between API calls in `aws_worker.py`
- Request rate limit increase from OpenAI

## Security Notes

- Never commit AWS credentials to git
- Use IAM roles instead of access keys where possible
- Restrict S3 bucket access with bucket policies
- Enable S3 encryption at rest
- Use VPC endpoints for S3/DynamoDB access (reduces data transfer costs)

## Support

For issues or questions:
1. Check logs in `/var/log/worker.log` on EC2 instances
2. Review CloudWatch logs
3. Check DynamoDB for job status
4. Review S3 bucket for uploaded files
