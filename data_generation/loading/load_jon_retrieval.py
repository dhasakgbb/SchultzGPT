#!/usr/bin/env python
"""
Data loading script for the OpenAI Retrieval API.
Loads Q&A pairs and statements into the Retrieval API for use with the assistant.

Usage:
    python -m data_generation.loading.load_jon_retrieval --file data/jon_data.jsonl
    python -m data_generation.loading.load_jon_retrieval --file data/jon_data.jsonl --assistant-id asst_abc123
"""

import os
import sys
import json
import argparse
import time
import asyncio
from typing import Dict, List, Any, Union, Optional
from datetime import datetime
from tqdm import tqdm
from dotenv import load_dotenv, set_key

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import RetrievalStore
try:
    from src.services.retrieval_store import RetrievalStore
except ImportError:
    print("Error importing RetrievalStore. Make sure you're running this from the project root.")
    print("You can also try without the retrieval store functionality by using --dry-run")
    RetrievalStore = None

# Import OpenAI
from openai import OpenAI

# Load environment variables
load_dotenv()

def load_data(file_path: str, dry_run: bool = False) -> bool:
    """
    Load data from a JSONL file into the Retrieval API.
    
    Args:
        file_path: Path to the JSONL file
        dry_run: Whether to test without uploading
        
    Returns:
        Success status
    """
    try:
        # Read and validate data first
        with open(file_path, 'r') as f:
            data = [json.loads(line) for line in f]
            
        if dry_run:
            print(f"Found {len(data)} items to load")
            print("Sample items:")
            for item in data[:3]:
                print(json.dumps(item, indent=2))
            return True
            
        # Initialize retrieval store
        store = RetrievalStore()
        
        # Load data into retrieval store
        for item in tqdm(data, desc="Loading data"):
            store.add_text(
                text=item["text"],
                metadata=item.get("metadata", {})
            )
            
        print(f"Successfully loaded {len(data)} items")
        return True
        
    except FileNotFoundError:
        print(f"Error: File not found: {file_path}")
        return False
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in file: {e}")
        return False
    except Exception as e:
        print(f"Error loading data: {e}")
        return False

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Load data into the OpenAI Retrieval API")
    parser.add_argument("--file", required=True, help="Path to JSONL file to load")
    parser.add_argument("--dry-run", action="store_true", help="Test without uploading")
    args = parser.parse_args()
    
    success = load_data(args.file, args.dry_run)
    if not success:
        exit(1)

if __name__ == "__main__":
    main() 