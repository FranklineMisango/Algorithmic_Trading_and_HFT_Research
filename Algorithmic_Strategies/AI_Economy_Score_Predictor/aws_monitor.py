"""
AWS Job Monitor - Monitor scoring job progress and download results
"""

import boto3
import pandas as pd
from datetime import datetime
import argparse
import logging
import time
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.live import Live
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
console = Console()


class AWSJobMonitor:
    """Monitor and manage AWS scoring jobs"""
    
    def __init__(self, bucket_name: str, region: str = 'us-east-1'):
        self.bucket_name = bucket_name
        self.region = region
        self.dynamodb = boto3.resource('dynamodb', region_name=region)
        self.s3_client = boto3.client('s3', region_name=region)
        self.job_table = self.dynamodb.Table('transcript-scoring-jobs')
    
    def get_job_summary(self) -> dict:
        """Get summary of all jobs"""
        response = self.job_table.scan()
        jobs = response.get('Items', [])
        
        summary = {
            'total': len(jobs),
            'pending': sum(1 for j in jobs if j['job_status'] == 'pending'),
            'in_progress': sum(1 for j in jobs if j['job_status'] == 'in_progress'),
            'completed': sum(1 for j in jobs if j['job_status'] == 'completed'),
            'failed': sum(1 for j in jobs if j['job_status'] == 'failed'),
        }
        
        return summary
    
    def get_all_jobs(self) -> pd.DataFrame:
        """Get details of all jobs"""
        response = self.job_table.scan()
        jobs = response.get('Items', [])
        
        if not jobs:
            return pd.DataFrame()
        
        df = pd.DataFrame(jobs)
        
        # Sort by created_at
        if 'created_at' in df.columns:
            df = df.sort_values('created_at', ascending=False)
        
        return df
    
    def display_summary(self):
        """Display job summary in a nice table"""
        summary = self.get_job_summary()
        
        table = Table(title="Job Summary")
        table.add_column("Status", style="cyan")
        table.add_column("Count", style="magenta", justify="right")
        
        table.add_row("Total", str(summary['total']))
        table.add_row("Pending", str(summary['pending']), style="yellow")
        table.add_row("In Progress", str(summary['in_progress']), style="blue")
        table.add_row("Completed", str(summary['completed']), style="green")
        table.add_row("Failed", str(summary['failed']), style="red")
        
        console.print(table)
    
    def display_jobs(self, limit: int = 20):
        """Display recent jobs in a table"""
        jobs_df = self.get_all_jobs()
        
        if jobs_df.empty:
            console.print("[yellow]No jobs found[/yellow]")
            return
        
        # Display only recent jobs
        jobs_df = jobs_df.head(limit)
        
        table = Table(title=f"Recent Jobs (showing {len(jobs_df)})")
        table.add_column("Job ID", style="cyan")
        table.add_column("Status", style="magenta")
        table.add_column("Created", style="blue")
        table.add_column("Input", style="green")
        
        for _, job in jobs_df.iterrows():
            status = job['job_status']
            status_style = {
                'pending': 'yellow',
                'in_progress': 'blue',
                'completed': 'green',
                'failed': 'red'
            }.get(status, 'white')
            
            created = job.get('created_at', 'N/A')[:19]
            input_key = Path(job.get('input_s3_key', '')).name
            
            table.add_row(
                job['job_id'][:8] + "...",
                f"[{status_style}]{status}[/{status_style}]",
                created,
                input_key
            )
        
        console.print(table)
    
    def watch_progress(self, refresh_seconds: int = 5):
        """Live monitoring of job progress"""
        console.print("\n[cyan]Monitoring job progress... (Press Ctrl+C to exit)[/cyan]\n")
        
        try:
            while True:
                # Clear screen
                console.clear()
                
                # Show summary
                self.display_summary()
                console.print()
                
                # Show recent jobs
                self.display_jobs()
                
                # Show timestamp
                console.print(f"\n[dim]Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}[/dim]")
                
                # Check if all jobs are done
                summary = self.get_job_summary()
                if summary['pending'] == 0 and summary['in_progress'] == 0:
                    console.print("\n[green]All jobs completed![/green]")
                    break
                
                # Wait before refresh
                time.sleep(refresh_seconds)
                
        except KeyboardInterrupt:
            console.print("\n[yellow]Monitoring stopped[/yellow]")
    
    def download_results(self, output_file: str = 'scored_transcripts.csv'):
        """Download and combine all results"""
        console.print(f"[cyan]Downloading results...[/cyan]")
        
        # Get completed jobs
        response = self.job_table.scan(
            FilterExpression='job_status = :status',
            ExpressionAttributeValues={':status': 'completed'}
        )
        
        completed_jobs = response.get('Items', [])
        
        if not completed_jobs:
            console.print("[yellow]No completed jobs found[/yellow]")
            return None
        
        console.print(f"Found {len(completed_jobs)} completed jobs")
        
        all_results = []
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            console=console
        ) as progress:
            
            task = progress.add_task("Downloading...", total=len(completed_jobs))
            
            for job in completed_jobs:
                output_key = job['output_s3_key']
                
                try:
                    # Download file to temp
                    temp_file = f"/tmp/{Path(output_key).name}"
                    self.s3_client.download_file(self.bucket_name, output_key, temp_file)
                    
                    # Read and append
                    df = pd.read_csv(temp_file)
                    all_results.append(df)
                    
                except Exception as e:
                    console.print(f"[red]Error downloading {output_key}: {e}[/red]")
                
                progress.update(task, advance=1)
        
        if all_results:
            # Combine all results
            combined_df = pd.concat(all_results, ignore_index=True)
            
            # Remove duplicates
            combined_df = combined_df.drop_duplicates(subset=['symbol', 'date'])
            
            # Save to file
            combined_df.to_csv(output_file, index=False)
            
            console.print(f"\n[green]✓ Downloaded {len(combined_df)} scored transcripts[/green]")
            console.print(f"[green]✓ Saved to: {output_file}[/green]")
            
            # Show summary stats
            table = Table(title="Results Summary")
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="magenta")
            
            table.add_row("Total transcripts", str(len(combined_df)))
            table.add_row("Date range", f"{combined_df['date'].min()} to {combined_df['date'].max()}")
            table.add_row("Average score", f"{combined_df['score'].mean():.2f}")
            
            console.print()
            console.print(table)
            
            return combined_df
        else:
            console.print("[yellow]No results to combine[/yellow]")
            return None


def main():
    parser = argparse.ArgumentParser(description='Monitor AWS scoring jobs')
    parser.add_argument('--bucket', required=True, help='S3 bucket name')
    parser.add_argument('--region', default='us-east-1', help='AWS region')
    parser.add_argument('--watch', action='store_true', help='Live monitoring mode')
    parser.add_argument('--download', action='store_true', help='Download all results')
    parser.add_argument('--output', default='scored_transcripts.csv', help='Output file for results')
    parser.add_argument('--refresh', type=int, default=5, help='Refresh interval for watch mode')
    
    args = parser.parse_args()
    
    monitor = AWSJobMonitor(args.bucket, args.region)
    
    if args.watch:
        monitor.watch_progress(args.refresh)
    elif args.download:
        monitor.download_results(args.output)
    else:
        # Default: show summary and recent jobs
        monitor.display_summary()
        console.print()
        monitor.display_jobs()


if __name__ == "__main__":
    main()
