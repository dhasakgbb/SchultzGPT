#!/usr/bin/env python
"""
Jon Tools - Unified CLI for SchultzGPT Data Tools

This script provides a unified interface to all the Jon data generation,
loading, and management tools through subcommands.

The data generation now uses OpenAI's official Batch API for efficient bulk generation 
when one-shot generation is enabled and the client supports it. This significantly improves
performance for large generation jobs and reduces API call overhead.

Usage:
    python -m data_generation.jon_tools generate --qa-pairs 1000 --one-shot --batch-size 15 --max-concurrent 10
    python -m data_generation.jon_tools load --file data/jon_retrieval_data.jsonl
    python -m data_generation.jon_tools train --file data/jon_fine_tuning.jsonl
    python -m data_generation.jon_tools assistants list
    python -m data_generation.jon_tools test --file data/jon_retrieval_data.jsonl --interactive
"""

import os
import sys
import argparse
import importlib
from typing import Dict, List, Any, Optional, Callable
import traceback

# Add src to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import tools using relative imports
try:
    from data_generation.generation.jon_data_generator import main as generate_main
    from data_generation.loading.load_jon_retrieval import main as load_retrieval_main
    from data_generation.training.prepare_fine_tuning import main as prepare_ft_main
    from data_generation.training.monitor_jobs import main as monitor_jobs_main
    from data_generation.utils.test_jon_data import main as test_main
    from data_generation.loading.manage_assistants import main as manage_assistants_main
except ImportError as e:
    # Unable to import from the new structure
    print(f"Error importing module: {str(e)}")
    print(f"Error location: {traceback.format_exc()}")
    print("The files have been moved to their respective subdirectories.")
    print("Please run this script from the project root:")
    print("python -m data_generation.jon_tools [command]")
    sys.exit(1)

def setup_generate_parser(subparsers):
    """Set up the parser for the generate command"""
    generate_parser = subparsers.add_parser(
        'generate', 
        help='Generate synthetic Jon data for various purposes'
    )
    
    # Data generation arguments
    generate_parser.add_argument("--qa-pairs", type=int, default=1000, help="Number of QA pairs to generate")
    generate_parser.add_argument("--conversations", type=int, default=100, help="Number of conversations to generate")
    generate_parser.add_argument("--statements", type=int, default=150, help="Number of statements to generate")
    generate_parser.add_argument("--temperature", type=float, default=0.85, help="Temperature for generation")
    generate_parser.add_argument("--batch-size", type=int, default=15, help="Batch size for parallel/bulk generation")
    generate_parser.add_argument("--max-concurrent", type=int, default=8, help="Maximum concurrent batch requests")
    generate_parser.add_argument("--variations", type=int, default=2, help="Number of contextual variations per Q&A pair")
    generate_parser.add_argument("--output-dir", type=str, default="data_generation/output", help="Output directory")
    generate_parser.add_argument("--parallel", action="store_true", help="Use parallel processing for generation")
    generate_parser.add_argument("--dynamic-batching", action="store_true", help="Use dynamic batch sizing")
    generate_parser.add_argument("--enrich-metadata", action="store_true", help="Add enhanced metadata to generated items")
    generate_parser.add_argument("--one-shot", action="store_true", help="Use one-shot bulk generation (default for large datasets)")
    generate_parser.add_argument("--use-real-data", action="store_true", help="Use real Jon data for examples")
    generate_parser.add_argument("--verify", action="store_true", help="Verify output data")
    generate_parser.add_argument("--checkpoint", type=int, default=100, dest="checkpoint_frequency", help="Save checkpoints every N items")
    generate_parser.add_argument("--incremental-save", action="store_true", help="Save data incrementally for large generations")
    generate_parser.add_argument("--dry-run", action="store_true", help="Skip retrieval store functionality")
    
    return generate_parser

def setup_load_parser(subparsers):
    """Set up the parser for the load command"""
    load_parser = subparsers.add_parser(
        'load', 
        help='Load data into the Retrieval API'
    )
    
    # Add load arguments
    load_parser.add_argument("--file", type=str, required=True, help="Path to the data file to load")
    load_parser.add_argument("--dry-run", action="store_true", help="Validate without loading")
    load_parser.add_argument("--assistant-id", type=str, help="Assistant ID for loading")
    load_parser.add_argument("--batch-size", type=int, default=20, help="Batch size for retrieval API loads")
    
    return load_parser

def setup_train_parser(subparsers):
    """Set up the parser for the train command"""
    train_parser = subparsers.add_parser(
        'train', 
        help='Prepare and manage fine-tuning jobs'
    )
    
    train_subparsers = train_parser.add_subparsers(dest='train_cmd', required=True)
    
    # Prepare fine-tuning data
    prepare_parser = train_subparsers.add_parser('prepare', help='Prepare data for fine-tuning')
    prepare_parser.add_argument("--file", type=str, required=True, help="Input JSONL file")
    prepare_parser.add_argument("--output", type=str, help="Output file path")
    prepare_parser.add_argument("--test-split", type=float, default=0.1, help="Test set ratio")
    prepare_parser.add_argument("--fine-tune", action="store_true", help="Start fine-tuning job after preparation")
    prepare_parser.add_argument("--model", type=str, default="gpt-3.5-turbo", help="Base model for fine-tuning")
    prepare_parser.add_argument("--suffix", type=str, default="jon", help="Suffix for the fine-tuned model")
    
    # List fine-tuning jobs
    list_parser = train_subparsers.add_parser('list', help='List fine-tuning jobs')
    list_parser.add_argument("--limit", type=int, default=10, help="Maximum number of jobs to list")
    
    # Status of fine-tuning job
    status_parser = train_subparsers.add_parser('status', help='Check status of a fine-tuning job')
    status_parser.add_argument("job_id", type=str, help="Fine-tuning job ID")
    status_parser.add_argument("--register", action="store_true", help="Register the model in .env when complete")
    
    # Cancel fine-tuning job
    cancel_parser = train_subparsers.add_parser('cancel', help='Cancel a fine-tuning job')
    cancel_parser.add_argument("job_id", type=str, help="Fine-tuning job ID")
    
    return train_parser

def setup_assistants_parser(subparsers):
    """Set up the parser for the assistants command"""
    assistants_parser = subparsers.add_parser(
        'assistants', 
        help='Manage OpenAI Assistants'
    )
    
    assistants_subparsers = assistants_parser.add_subparsers(dest='assistants_cmd', required=True)
    
    # List assistants
    list_parser = assistants_subparsers.add_parser('list', help='List assistants')
    list_parser.add_argument("--limit", type=int, default=10, help="Maximum number of assistants to list")
    
    # Get details
    details_parser = assistants_subparsers.add_parser('details', help='Get assistant details')
    details_parser.add_argument("--id", type=str, required=True, help="Assistant ID")
    
    # Create assistant
    create_parser = assistants_subparsers.add_parser('create', help='Create new assistant')
    create_parser.add_argument("--name", type=str, required=True, help="Assistant name")
    create_parser.add_argument("--model", type=str, default="gpt-4o", help="Model to use")
    create_parser.add_argument("--instructions", type=str, help="Assistant instructions")
    create_parser.add_argument("--save", action="store_true", help="Save assistant ID to .env")
    
    # Clean assistant
    clean_parser = assistants_subparsers.add_parser('clean', help='Clean assistant files')
    clean_parser.add_argument("--id", type=str, required=True, help="Assistant ID")
    clean_parser.add_argument("--age", type=int, default=30, help="Delete files older than X days")
    clean_parser.add_argument("--dry-run", action="store_true", help="Report without deleting")
    
    return assistants_parser

def setup_test_parser(subparsers):
    """Set up the parser for the test command"""
    test_parser = subparsers.add_parser(
        'test', 
        help='Test generated data interactively'
    )
    
    test_parser.add_argument("--file", type=str, help="Data file to test")
    test_parser.add_argument("--type", type=str, choices=["qa", "conversation", "retrieval"], 
                         default="qa", help="Type of data")
    test_parser.add_argument("--interactive", action="store_true", help="Interactive testing mode")
    test_parser.add_argument("--model", type=str, help="Model to use for testing")
    test_parser.add_argument("--assistant-id", type=str, help="Assistant ID for testing")
    test_parser.add_argument("--limit", type=int, default=5, help="Number of items to display")
    
    return test_parser

def main():
    """Main entry point for the CLI"""
    parser = argparse.ArgumentParser(
        description="Unified CLI for Jon data tools",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Set up command parsers
    setup_generate_parser(subparsers)
    setup_load_parser(subparsers)
    setup_train_parser(subparsers)
    setup_assistants_parser(subparsers)
    setup_test_parser(subparsers)
    
    # Parse arguments
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
        
    # Execute command
    if args.command == 'generate':
        generate_main(args)
    elif args.command == 'load':
        load_retrieval_main(args)
    elif args.command == 'train':
        if args.train_cmd == 'prepare':
            prepare_ft_main(args)
        elif args.train_cmd == 'list' or args.train_cmd == 'status' or args.train_cmd == 'cancel':
            monitor_jobs_main(args)
    elif args.command == 'assistants':
        manage_assistants_main(args)
    elif args.command == 'test':
        test_main(args)

if __name__ == '__main__':
    main() 