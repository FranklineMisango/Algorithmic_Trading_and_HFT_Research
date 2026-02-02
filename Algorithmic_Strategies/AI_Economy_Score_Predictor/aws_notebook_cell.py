"""
Add this cell to your notebook to use AWS spot instances for scoring
Replace the existing score_quarter_transcripts() call with this code
"""

# ============================================================================
# AWS SPOT INSTANCE SCORING - Drop-in Replacement
# ============================================================================

# Option 1: AWS Spot Instances (Recommended for large datasets)
USE_AWS = True  # Set to False to use local scoring

if USE_AWS:
    from aws_job_submitter import AWSJobSubmitter
    from aws_monitor import AWSJobMonitor
    import time
    
    # AWS Configuration
    AWS_BUCKET = "transcript-scoring-123456789"  # From aws_setup.sh
    AWS_REGION = "us-east-1"
    
    print("=" * 70)
    print("AWS SPOT INSTANCE SCORING")
    print("=" * 70)
    print(f"Starting at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Transcripts to score: {len(scoring_transcripts)}")
    print(f"Estimated cost: ${len(scoring_transcripts) * 0.001:.2f} (API) + $0.05 (EC2)")
    print()
    
    # Initialize submitter
    submitter = AWSJobSubmitter(AWS_BUCKET, AWS_REGION)
    
    # Split into batches and upload to S3
    print("📦 Creating batches...")
    batch_keys = submitter.create_batch_files(
        scoring_transcripts, 
        batch_size=50
    )
    print(f"✓ Created {len(batch_keys)} batches")
    
    # Create jobs in DynamoDB
    print("\n📝 Creating jobs in queue...")
    job_ids = submitter.create_jobs(batch_keys)
    print(f"✓ Created {len(job_ids)} jobs")
    
    # Launch spot instances
    print("\n🚀 Launching spot instances...")
    spot_requests = submitter.launch_spot_instances(
        num_instances=2,           # Adjust based on workload
        instance_type='t3.medium'  # ~$0.01/hour spot pricing
    )
    print(f"✓ Launched spot instance requests")
    print(f"  Workers will start processing shortly...")
    
    # Initialize monitor
    monitor = AWSJobMonitor(AWS_BUCKET, AWS_REGION)
    
    # Live monitoring
    print("\n" + "=" * 70)
    print("MONITORING PROGRESS")
    print("=" * 70)
    print("(Will auto-update every 30 seconds)")
    print()
    
    last_completed = 0
    start_time = time.time()
    
    while True:
        summary = monitor.get_job_summary()
        
        completed = summary['completed']
        total = summary['total']
        failed = summary['failed']
        in_progress = summary['in_progress']
        pending = summary['pending']
        
        # Calculate progress
        progress = (completed / total * 100) if total > 0 else 0
        elapsed = time.time() - start_time
        
        # Estimate time remaining
        if completed > 0:
            rate = completed / elapsed  # jobs per second
            remaining_jobs = total - completed
            eta_seconds = remaining_jobs / rate if rate > 0 else 0
            eta_mins = eta_seconds / 60
        else:
            eta_mins = 0
        
        # Display progress
        print(f"\r{completed}/{total} jobs completed ({progress:.1f}%) | "
              f"In Progress: {in_progress} | Pending: {pending} | "
              f"Failed: {failed} | ETA: {eta_mins:.1f} min", 
              end='', flush=True)
        
        # Check if done
        if pending == 0 and in_progress == 0:
            print("\n\n✓ All jobs completed!")
            break
        
        # Check for progress (warn if stuck)
        if completed == last_completed and elapsed > 300:  # 5 minutes no progress
            print(f"\n⚠️  Warning: No progress in 5 minutes. Check AWS console.")
        
        last_completed = completed
        time.sleep(30)  # Check every 30 seconds
    
    # Download results
    print("\n" + "=" * 70)
    print("DOWNLOADING RESULTS")
    print("=" * 70)
    
    scored_data = monitor.download_results(save_path)
    
    print(f"\n✓ Downloaded {len(scored_data)} scored transcripts")
    print(f"✓ Saved to: {save_path}")
    
else:
    # Original local scoring
    print("=" * 70)
    print("LOCAL SCORING")
    print("=" * 70)
    print(f"Starting at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    scored_data = score_quarter_transcripts(
        scoring_transcripts, 
        scorer, 
        save_path=save_path
    )

# Continue with rest of pipeline...
print("\n" + "=" * 70)
print("SCORING COMPLETE")
print("=" * 70)
print(f"Completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"\nFinal Results:")
print(f"  Total scored: {len(scored_data)}")
print(f"  Date range: {scored_data['date'].min()} to {scored_data['date'].max()}")
print(f"  Average score: {scored_data['score'].mean():.2f}")
print(f"  Score distribution:")
print(scored_data['score'].value_counts().sort_index())
