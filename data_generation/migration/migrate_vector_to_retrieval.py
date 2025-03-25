#!/usr/bin/env python
"""
Migrate Vector Store Data to Retrieval API

This script migrates data from the legacy vector store to the new OpenAI Retrieval API.
It's intended for users upgrading from the old vector store implementation.

Usage:
    python -m data_generation.migration.migrate_vector_to_retrieval --vector-dir ./vector_store --assistant-id asst_abc123
"""

import os
import sys
import json
import argparse
from typing import List, Dict, Any, Optional
import time

# Add root to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

# Load environment variables
load_dotenv()

def load_vector_data(vector_dir: str) -> List[Dict[str, Any]]:
    """
    Load data from the vector store directory
    """
    print(f"Loading data from vector store: {vector_dir}")
    
    if not os.path.exists(vector_dir):
        print(f"Error: Vector store directory not found: {vector_dir}")
        return []
    
    # Check for data.json or embeddings.jsonl
    data_file = os.path.join(vector_dir, "data.json")
    embeddings_file = os.path.join(vector_dir, "embeddings.jsonl")
    
    data = []
    
    # Try to load data.json
    if os.path.exists(data_file):
        try:
            with open(data_file, 'r') as f:
                data_json = json.load(f)
                print(f"Loaded {len(data_json)} entries from data.json")
                data.extend(data_json)
        except Exception as e:
            print(f"Error loading data.json: {e}")
    
    # Try to load embeddings.jsonl
    if os.path.exists(embeddings_file):
        try:
            with open(embeddings_file, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    if line.strip():
                        data.append(json.loads(line))
                print(f"Loaded {len(lines)} entries from embeddings.jsonl")
        except Exception as e:
            print(f"Error loading embeddings.jsonl: {e}")
    
    # Check if we found any data
    if not data:
        print("No data found in vector store")
        return []
    
    print(f"Successfully loaded {len(data)} entries from vector store")
    return data

def format_for_retrieval(data: List[Dict[str, Any]]) -> str:
    """
    Format the data for the retrieval API
    """
    retrieval_data = []
    
    for item in data:
        # Handle different data formats
        if "text" in item and "metadata" in item:
            # This looks like it's already in retrieval format
            retrieval_data.append(item)
        elif "question" in item and "answer" in item:
            # This is a QA pair, format for retrieval
            retrieval_item = {
                "text": f"Question: {item['question']}\nAnswer: {item['answer']}",
                "metadata": {
                    "type": "qa_pair",
                    "topic": item.get("topic", ""),
                    "source": "vector_store_migration",
                    "timestamp": time.time()
                }
            }
            
            # Copy any metadata if present
            if "metadata" in item and isinstance(item["metadata"], dict):
                for key, value in item["metadata"].items():
                    retrieval_item["metadata"][key] = value
            
            retrieval_data.append(retrieval_item)
        elif "content" in item:
            # This might be a raw message
            retrieval_item = {
                "text": item["content"],
                "metadata": {
                    "type": "message",
                    "source": "vector_store_migration",
                    "timestamp": time.time()
                }
            }
            
            # Copy any metadata if present
            if "metadata" in item and isinstance(item["metadata"], dict):
                for key, value in item["metadata"].items():
                    retrieval_item["metadata"][key] = value
            
            retrieval_data.append(retrieval_item)
    
    # Convert to JSONL format
    jsonl_data = "\n".join(json.dumps(item) for item in retrieval_data)
    
    print(f"Formatted {len(retrieval_data)} items for retrieval API")
    return jsonl_data

def upload_to_retrieval_api(client: OpenAI, data: str, assistant_id: Optional[str] = None, 
                           batch_size: int = 20) -> str:
    """
    Upload the data to the OpenAI Retrieval API
    """
    print("Uploading data to Retrieval API...")
    
    # Create a temporary file with the data
    temp_file = "temp_migration_data.jsonl"
    with open(temp_file, 'w') as f:
        f.write(data)
    
    try:
        # Get or create assistant
        if assistant_id:
            # Verify the assistant exists
            try:
                assistant = client.beta.assistants.retrieve(assistant_id=assistant_id)
                print(f"Using existing assistant: {assistant.name} ({assistant_id})")
            except Exception as e:
                print(f"Error accessing assistant {assistant_id}: {e}")
                print("Creating a new assistant instead")
                assistant_id = None
        
        if not assistant_id:
            # Create a new assistant
            assistant = client.beta.assistants.create(
                name="Jon Memory (Migrated)",
                instructions="You are Jon, a persona with specific knowledge and memories. Your responses should reflect Jon's personality, knowledge, and past interactions.",
                model="gpt-4o",
                tools=[{"type": "retrieval"}]
            )
            assistant_id = assistant.id
            print(f"Created new assistant: {assistant.name} ({assistant_id})")
            
            # Save assistant ID to .env
            try:
                env_file = ".env"
                env_text = ""
                if os.path.exists(env_file):
                    with open(env_file, 'r') as f:
                        env_text = f.read()
                
                # Check if OPENAI_ASSISTANT_ID is already set
                if "OPENAI_ASSISTANT_ID" in env_text:
                    # Replace existing value
                    env_lines = env_text.splitlines()
                    for i, line in enumerate(env_lines):
                        if line.startswith("OPENAI_ASSISTANT_ID="):
                            env_lines[i] = f"OPENAI_ASSISTANT_ID={assistant_id}"
                            break
                    env_text = "\n".join(env_lines)
                else:
                    # Add new value
                    env_text += f"\nOPENAI_ASSISTANT_ID={assistant_id}"
                
                with open(env_file, 'w') as f:
                    f.write(env_text)
                
                print(f"Saved assistant ID to .env file")
            except Exception as e:
                print(f"Warning: Failed to save assistant ID to .env: {e}")
        
        # Upload the file
        with open(temp_file, 'rb') as f:
            file = client.files.create(
                file=f,
                purpose="assistants"
            )
            print(f"Uploaded file with ID: {file.id}")
        
        # Attach file to the assistant
        client.beta.assistants.files.create(
            assistant_id=assistant_id,
            file_id=file.id
        )
        print(f"Attached file to assistant")
        
        print(f"Migration complete! Assistant ID: {assistant_id}")
        return assistant_id
        
    finally:
        # Clean up the temporary file
        if os.path.exists(temp_file):
            os.remove(temp_file)

def main():
    """Main entry point for the script"""
    parser = argparse.ArgumentParser(description="Migrate data from vector store to retrieval API")
    parser.add_argument("--vector-dir", type=str, default="vector_store", help="Vector store directory")
    parser.add_argument("--assistant-id", type=str, help="Assistant ID to use (creates new one if not provided)")
    parser.add_argument("--batch-size", type=int, default=20, help="Batch size for uploads")
    parser.add_argument("--dry-run", action="store_true", help="Don't upload, just show what would be uploaded")
    
    args = parser.parse_args()
    
    # Initialize OpenAI client
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    
    # Load data from vector store
    vector_data = load_vector_data(args.vector_dir)
    
    if not vector_data:
        print("No data to migrate. Exiting.")
        return
    
    # Format data for retrieval API
    retrieval_data = format_for_retrieval(vector_data)
    
    # Save formatted data to file for reference
    output_file = "migration_data.jsonl"
    with open(output_file, 'w') as f:
        f.write(retrieval_data)
    print(f"Saved formatted data to {output_file} for reference")
    
    # Upload to retrieval API if not dry run
    if not args.dry_run:
        upload_to_retrieval_api(client, retrieval_data, args.assistant_id, args.batch_size)
    else:
        print("Dry run complete. Data not uploaded.")
        print(f"To upload this data, run again without --dry-run")

if __name__ == "__main__":
    main() 