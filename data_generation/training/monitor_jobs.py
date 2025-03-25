#!/usr/bin/env python
"""
Monitor Fine-Tuning Jobs

Utility script to monitor and manage fine-tuning jobs.

Usage:
    python -m data_generation.monitor_jobs list
    python -m data_generation.monitor_jobs status [job_id]
    python -m data_generation.monitor_jobs cancel [job_id]
"""

import os
import sys
import json
import argparse
from datetime import datetime

# Add src to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dotenv import load_dotenv
from openai import OpenAI
from tabulate import tabulate

# Load environment variables
load_dotenv()

def list_jobs(client, limit=10):
    """List fine-tuning jobs"""
    try:
        response = client.fine_tuning.jobs.list(limit=limit)
        jobs = response.data
        
        if not jobs:
            print("No fine-tuning jobs found")
            return
        
        table_data = []
        for job in jobs:
            created_at = datetime.fromtimestamp(job.created_at).strftime('%Y-%m-%d %H:%M:%S')
            table_data.append([
                job.id,
                job.model,
                job.fine_tuned_model or "N/A",
                job.status,
                created_at
            ])
        
        headers = ["Job ID", "Base Model", "Fine-tuned Model", "Status", "Created At"]
        print(tabulate(table_data, headers=headers, tablefmt="grid"))
        
    except Exception as e:
        print(f"Error listing jobs: {e}")

def get_job_status(client, job_id):
    """Get detailed status of a fine-tuning job"""
    try:
        job = client.fine_tuning.jobs.retrieve(job_id)
        
        print(f"\nJob ID: {job.id}")
        print(f"Status: {job.status}")
        print(f"Base Model: {job.model}")
        print(f"Fine-tuned Model: {job.fine_tuned_model or 'Not yet available'}")
        
        created_at = datetime.fromtimestamp(job.created_at).strftime('%Y-%m-%d %H:%M:%S')
        print(f"Created At: {created_at}")
        
        if job.finished_at:
            finished_at = datetime.fromtimestamp(job.finished_at).strftime('%Y-%m-%d %H:%M:%S')
            print(f"Finished At: {finished_at}")
        
        print(f"Training File: {job.training_file}")
        if job.validation_file:
            print(f"Validation File: {job.validation_file}")
        
        if job.trained_tokens:
            print(f"Trained Tokens: {job.trained_tokens}")
        
        if hasattr(job, 'error') and job.error:
            print(f"\nError: {job.error.message}")
        
        # Get training metrics if available
        if job.status in ["succeeded", "running"]:
            try:
                metrics = client.fine_tuning.jobs.list_metrics(job_id)
                results = metrics.data
                
                if results:
                    print("\nTraining Metrics:")
                    for result in results:
                        timestamp = datetime.fromtimestamp(result.timestamp).strftime('%H:%M:%S')
                        print(f"  [{timestamp}] Step {result.step}: Loss = {result.train_loss:.4f}")
                        if hasattr(result, 'test_loss'):
                            print(f"              Test Loss = {result.test_loss:.4f}")
            except Exception as e:
                print(f"Could not retrieve metrics: {e}")
        
        return job
        
    except Exception as e:
        print(f"Error retrieving job status: {e}")
        return None

def cancel_job(client, job_id):
    """Cancel a fine-tuning job"""
    try:
        job = client.fine_tuning.jobs.retrieve(job_id)
        
        if job.status in ["succeeded", "failed", "cancelled"]:
            print(f"Job already {job.status}, cannot cancel")
            return False
        
        print(f"Cancelling job {job_id}...")
        response = client.fine_tuning.jobs.cancel(job_id)
        print(f"Job status: {response.status}")
        return True
        
    except Exception as e:
        print(f"Error cancelling job: {e}")
        return False

def register_model(client, job):
    """Register a fine-tuned model in the config"""
    if not job.fine_tuned_model:
        print("No fine-tuned model available for this job")
        return False
    
    try:
        # Update Config.FINE_TUNED_MODEL in .env
        env_file = ".env"
        
        # First check if model ID already in .env
        current_model = None
        if os.path.exists(env_file):
            with open(env_file, 'r') as f:
                for line in f:
                    if line.startswith("FINE_TUNED_MODEL="):
                        current_model = line.strip().split("=", 1)[1].strip('"\'')
        
        if current_model == job.fine_tuned_model:
            print(f"Model {job.fine_tuned_model} already registered in .env")
            return True
        
        # Confirm with user
        confirm = input(f"Update FINE_TUNED_MODEL in .env to {job.fine_tuned_model}? [y/N] ")
        if confirm.lower() != 'y':
            print("Operation cancelled")
            return False
        
        # Update .env file
        lines = []
        model_updated = False
        
        if os.path.exists(env_file):
            with open(env_file, 'r') as f:
                lines = f.readlines()
                
            for i, line in enumerate(lines):
                if line.startswith("FINE_TUNED_MODEL="):
                    lines[i] = f"FINE_TUNED_MODEL={job.fine_tuned_model}\n"
                    model_updated = True
        
        if not model_updated:
            lines.append(f"FINE_TUNED_MODEL={job.fine_tuned_model}\n")
        
        with open(env_file, 'w') as f:
            f.writelines(lines)
        
        print(f"Updated FINE_TUNED_MODEL in .env to {job.fine_tuned_model}")
        return True
        
    except Exception as e:
        print(f"Error registering model: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Monitor and manage fine-tuning jobs")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # List command
    list_parser = subparsers.add_parser("list", help="List fine-tuning jobs")
    list_parser.add_argument("--limit", type=int, default=10,
                           help="Maximum number of jobs to list")
    
    # Status command
    status_parser = subparsers.add_parser("status", help="Get job status")
    status_parser.add_argument("job_id", help="ID of the fine-tuning job")
    status_parser.add_argument("--register", action="store_true",
                             help="Register the fine-tuned model in .env if successful")
    
    # Cancel command
    cancel_parser = subparsers.add_parser("cancel", help="Cancel a job")
    cancel_parser.add_argument("job_id", help="ID of the fine-tuning job to cancel")
    
    args = parser.parse_args()
    
    # Check if API key is set
    if not os.environ.get("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable not set")
        sys.exit(1)
    
    # Initialize OpenAI client
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    
    if args.command == "list":
        list_jobs(client, args.limit)
    elif args.command == "status":
        job = get_job_status(client, args.job_id)
        if job and args.register and job.status == "succeeded":
            register_model(client, job)
    elif args.command == "cancel":
        cancel_job(client, args.job_id)
    else:
        parser.print_help()

if __name__ == "__main__":
    main() 