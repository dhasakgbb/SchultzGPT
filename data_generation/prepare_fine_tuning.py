#!/usr/bin/env python3
"""
Prepare Jon's generated data for fine-tuning with OpenAI.
This script takes the generated data from jon_data_generator.py and 
formats it for OpenAI's fine-tuning endpoint.
"""

import os
import json
import argparse
import glob
from datetime import datetime
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Prepare Jon's data for fine-tuning")
    parser.add_argument("--input-dir", type=str, default="data_generation/output", 
                       help="Directory containing the generated data")
    parser.add_argument("--output-file", type=str, default="data_generation/fine_tuning/jon_fine_tuning.jsonl", 
                       help="Output file for the fine-tuning data")
    parser.add_argument("--upload", action="store_true", 
                       help="Upload the data to OpenAI for fine-tuning")
    parser.add_argument("--model", type=str, default="gpt-3.5-turbo", 
                       help="Base model for fine-tuning")
    return parser.parse_args()

def find_latest_data_files(input_dir):
    """Find the latest QA, conversation, and statement files."""
    # Look for regular output files first
    qa_files = glob.glob(os.path.join(input_dir, "qa_data_*.json"))
    conv_files = glob.glob(os.path.join(input_dir, "conversation_data_*.json"))
    stmt_files = glob.glob(os.path.join(input_dir, "statement_data_*.json"))
    
    # If no regular files found, look for checkpoint files
    if not qa_files:
        qa_files = glob.glob(os.path.join(input_dir, "checkpoints", "checkpoint_qa_data_*.jsonl"))
    if not conv_files:
        conv_files = glob.glob(os.path.join(input_dir, "checkpoints", "checkpoint_conversation_data_*.jsonl"))
    if not stmt_files:
        stmt_files = glob.glob(os.path.join(input_dir, "checkpoints", "checkpoint_statement_data_*.jsonl"))
    
    latest_qa = max(qa_files, key=os.path.getctime) if qa_files else None
    latest_conv = max(conv_files, key=os.path.getctime) if conv_files else None
    latest_stmt = max(stmt_files, key=os.path.getctime) if stmt_files else None
    
    print(f"Latest QA file: {latest_qa}")
    print(f"Latest conversation file: {latest_conv}")
    print(f"Latest statement file: {latest_stmt}")
    
    return latest_qa, latest_conv, latest_stmt

def load_json_data(file_path):
    """Load data from a JSON or JSONL file."""
    if not file_path or not os.path.exists(file_path):
        return []
    
    data = []
    # Check if it's a JSONL file
    if file_path.endswith('.jsonl'):
        with open(file_path, 'r') as f:
            for line in f:
                try:
                    data.append(json.loads(line.strip()))
                except json.JSONDecodeError:
                    continue
    else:
        # Regular JSON file
        with open(file_path, 'r') as f:
            data = json.load(f)
    
    return data

def format_qa_for_fine_tuning(qa_data):
    """Format QA pairs for fine-tuning."""
    formatted_data = []
    
    for item in qa_data:
        if "question" not in item or "answer" not in item:
            continue
            
        formatted_item = {
            "messages": [
                {"role": "user", "content": item["question"]},
                {"role": "assistant", "content": item["answer"]}
            ]
        }
        formatted_data.append(formatted_item)
    
    return formatted_data

def format_conversations_for_fine_tuning(conv_data):
    """Format conversations for fine-tuning."""
    formatted_data = []
    
    for item in conv_data:
        if "messages" not in item:
            continue
            
        formatted_item = {
            "messages": item["messages"]
        }
        formatted_data.append(formatted_item)
    
    return formatted_data

def format_statements_for_fine_tuning(stmt_data):
    """Format statements for fine-tuning."""
    formatted_data = []
    
    for item in stmt_data:
        if "statement" not in item:
            continue
            
        # Create a synthetic prompt for the statement
        prompt = "Tell me what's on your mind, Jon."
        
        formatted_item = {
            "messages": [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": item["statement"]}
            ]
        }
        formatted_data.append(formatted_item)
    
    return formatted_data

def save_fine_tuning_data(data, output_file):
    """Save the formatted data to a JSONL file."""
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')
    
    print(f"Saved {len(data)} items to {output_file}")
    return output_file

def upload_for_fine_tuning(file_path, model):
    """Upload the data to OpenAI for fine-tuning."""
    if not os.environ.get("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY environment variable not set")
    
    client = OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY"),
        organization=os.environ.get("OPENAI_ORG_ID")
    )
    
    print("Uploading file for fine-tuning...")
    with open(file_path, 'rb') as f:
        response = client.files.create(
            file=f,
            purpose="fine-tune"
        )
    
    file_id = response.id
    print(f"File uploaded with ID: {file_id}")
    
    print(f"Creating fine-tuning job with model: {model}")
    response = client.fine_tuning.jobs.create(
        training_file=file_id,
        model=model,
        suffix=f"jon-schultz-{datetime.now().strftime('%Y%m%d')}"
    )
    
    job_id = response.id
    print(f"Fine-tuning job created with ID: {job_id}")
    print(f"You can check the status with: openai api fine_tunes.get -i {job_id}")
    
    return job_id

def main():
    """Main entry point."""
    args = parse_args()
    
    # Find the latest data files
    qa_file, conv_file, stmt_file = find_latest_data_files(args.input_dir)
    
    # Load the data
    qa_data = load_json_data(qa_file)
    conv_data = load_json_data(conv_file)
    stmt_data = load_json_data(stmt_file)
    
    print(f"Loaded {len(qa_data)} QA pairs, {len(conv_data)} conversations, and {len(stmt_data)} statements")
    
    # Format the data for fine-tuning
    formatted_qa = format_qa_for_fine_tuning(qa_data)
    formatted_conv = format_conversations_for_fine_tuning(conv_data)
    formatted_stmt = format_statements_for_fine_tuning(stmt_data)
    
    # Combine all the formatted data
    all_formatted_data = formatted_qa + formatted_conv + formatted_stmt
    
    # Save the formatted data to a JSONL file
    output_file = save_fine_tuning_data(all_formatted_data, args.output_file)
    
    # Upload for fine-tuning if requested
    if args.upload:
        upload_for_fine_tuning(output_file, args.model)
    else:
        print("Data prepared for fine-tuning. To upload and start fine-tuning, run again with --upload")
        print(f"Command: python {os.path.basename(__file__)} --output-file {args.output_file} --upload --model {args.model}")

if __name__ == "__main__":
    main() 