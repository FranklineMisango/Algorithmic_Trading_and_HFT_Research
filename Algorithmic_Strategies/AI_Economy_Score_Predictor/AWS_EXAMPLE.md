# AWS Spot Instance Integration - Example Usage

## Quick Start

This shows how to replace local scoring with AWS spot instance processing.

### 1. Setup (one-time)
```bash
# Make setup script executable
chmod +x aws_setup.sh

# Run setup
bash aws_setup.sh

# Load environment
source .env.aws
```

### 2. Submit Jobs from Notebook

```python
from aws_job_submitter import AWSJobSubmitter
from aws_monitor import AWSJobMonitor

# Configuration (from .env.aws)
BUCKET_NAME = 'transcript-scoring-123456789'  # Your bucket from setup

# Initialize submitter
submitter = AWSJobSubmitter(BUCKET_NAME)

# Submit your transcripts for scoring
print(f"Submitting {len(scoring_transcripts)} transcripts to AWS...")

batch_keys = submitter.create_batch_files(
    scoring_transcripts, 
    batch_size=50  # 50 transcripts per batch
)

job_ids = submitter.create_jobs(batch_keys)

# Launch spot instances (cheap!)
spot_request_ids = submitter.launch_spot_instances(
    num_instances=2,           # Start with 2 workers
    instance_type='t3.medium'  # ~$0.01/hour (spot pricing)
)

print(f"✓ Submitted {len(job_ids)} jobs")
print(f"✓ Launched {len(spot_request_ids)} spot instances")
print(f"✓ Estimated cost: ${len(scoring_transcripts) * 0.001:.2f} (API) + $0.05 (EC2)")
```

### 3. Monitor Progress

```python
# Initialize monitor
monitor = AWSJobMonitor(BUCKET_NAME)

# Check status
monitor.display_summary()

# Live monitoring (updates every 5 seconds)
# monitor.watch_progress()  # Uncomment to use
```

### 4. Download Results

```python
# Wait for completion and download
import time

while True:
    summary = monitor.get_job_summary()
    completed = summary['completed']
    total = summary['total']
    
    print(f"Progress: {completed}/{total} jobs completed", end='\r')
    
    if summary['pending'] == 0 and summary['in_progress'] == 0:
        print("\n✓ All jobs completed!")
        break
    
    time.sleep(30)  # Check every 30 seconds

# Download all results
scored_data = monitor.download_results('scored_transcripts_aws.csv')

print(f"\n✓ Downloaded {len(scored_data)} scored transcripts")
print(f"✓ Date range: {scored_data['date'].min()} to {scored_data['date'].max()}")
print(f"✓ Average score: {scored_data['score'].mean():.2f}")
```

## Cost Comparison

### Local Processing:
- Time: ~33 minutes (1000 transcripts)
- Cost: $1.00 (API calls only)
- Resources: Uses your computer

### AWS Spot Instances:
- Time: ~11 minutes (with 3 workers)
- Cost: $1.01 (API + $0.01 EC2)
- Resources: Runs in cloud
- **Benefit**: Frees your computer, 3x faster

### For 10,000 transcripts:
- Local: ~5.5 hours, $10
- AWS (5 workers): ~1.1 hours, $10.05
- **5x faster for $0.05 more**

## Command Line Usage

### Submit jobs:
```bash
python aws_job_submitter.py \
  --bucket $AWS_BUCKET_NAME \
  --input-file scoring_transcripts.csv \
  --batch-size 50 \
  --launch-instances \
  --num-instances 2
```

### Monitor:
```bash
# Live monitoring
python aws_monitor.py --bucket $AWS_BUCKET_NAME --watch

# One-time check
python aws_monitor.py --bucket $AWS_BUCKET_NAME
```

### Download results:
```bash
python aws_monitor.py \
  --bucket $AWS_BUCKET_NAME \
  --download \
  --output scored_transcripts_final.csv
```

## Advanced: Batch Processing

For very large datasets, use batch mode:

```python
# Process 50,000 transcripts in batches
all_transcripts = pd.read_csv('all_transcripts.csv')

for i in range(0, len(all_transcripts), 10000):
    batch = all_transcripts.iloc[i:i+10000]
    
    print(f"Processing batch {i//10000 + 1}")
    
    batch_keys = submitter.create_batch_files(batch, batch_size=50)
    job_ids = submitter.create_jobs(batch_keys)
    
    print(f"Submitted {len(job_ids)} jobs")

# Launch workers once (they'll process all jobs)
submitter.launch_spot_instances(num_instances=5)
```

## Troubleshooting

### If spot instances don't launch:
```python
# Try different instance type
submitter.launch_spot_instances(
    num_instances=2,
    instance_type='t3.large'  # Slightly more expensive but more available
)
```

### If jobs are slow:
```python
# Add more workers
submitter.launch_spot_instances(num_instances=5)
```

### If you need to stop everything:
```bash
# Terminate all spot instances
aws ec2 describe-instances \
  --filters "Name=tag:Name,Values=transcript-scorer-worker" \
  --query "Reservations[*].Instances[*].InstanceId" \
  --output text | xargs aws ec2 terminate-instances --instance-ids
```
