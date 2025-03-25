#!/usr/bin/env python
"""
Jon Tools - Unified CLI for SchultzGPT Data Tools

This script provides a unified interface to all the Jon data generation,
loading, and management tools through subcommands.

Usage:
    python -m data_generation.jon_tools generate --qa-pairs 100 --batch-size 10
    python -m data_generation.jon_tools load --file data/jon_retrieval_data.jsonl
    python -m data_generation.jon_tools train --file data/jon_fine_tuning.jsonl
    python -m data_generation.jon_tools assistants list
    python -m data_generation.jon_tools visualize --input data/jon_raw_data.json
    python -m data_generation.jon_tools test --file data/jon_retrieval_data.jsonl --interactive
"""

import os
import sys
import argparse
import importlib
from typing import Dict, List, Any, Optional, Callable

# Add src to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import tools using relative imports
try:
    from data_generation.generation.jon_data_generator import main as generate_main
    from data_generation.loading.load_jon_data import main as load_vector_main
    from data_generation.loading.load_jon_retrieval import main as load_retrieval_main
    from data_generation.training.prepare_fine_tuning import main as prepare_ft_main
    from data_generation.training.monitor_jobs import main as monitor_jobs_main
    from data_generation.utils.test_jon_data import main as test_main
    from data_generation.loading.manage_assistants import main as manage_assistants_main
    from data_generation.visualization.visualize_memory import main as visualize_main
    from data_generation.migration.migrate_vector_to_retrieval import main as migrate_main
except ImportError as e:
    # Unable to import from the new structure
    print(f"Error importing module: {str(e)}")
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
    generate_parser.add_argument("--qa-pairs", type=int, default=100, help="Number of Q&A pairs to generate")
    generate_parser.add_argument("--conversations", type=int, default=20, help="Number of conversations to generate")
    generate_parser.add_argument("--statements", type=int, default=50, help="Number of standalone Jon statements to generate")
    generate_parser.add_argument("--batch-size", type=int, default=10, help="Number of items per API call")
    generate_parser.add_argument("--variations", type=int, default=2, help="Number of contextual variations per Q&A pair")
    generate_parser.add_argument("--contrastive", action="store_true", help="Generate contrastive examples")
    generate_parser.add_argument("--output-dir", type=str, default="data_generation/output", help="Output directory")
    generate_parser.add_argument("--optimize-chunks", action="store_true", help="Optimize chunk sizes for embedding performance")
    generate_parser.add_argument("--parallel", action="store_true", help="Use parallel processing for generation")
    generate_parser.add_argument("--dynamic-batching", action="store_true", help="Dynamically adjust batch sizes")
    generate_parser.add_argument("--enrich-metadata", action="store_true", help="Add enhanced metadata to generated items")
    
    return generate_parser

def setup_load_parser(subparsers):
    """Set up the parser for the load command"""
    load_parser = subparsers.add_parser(
        'load', 
        help='Load data into vector store or retrieval API'
    )
    
    # Add load arguments
    load_parser.add_argument("--file", type=str, required=True, help="Path to the data file to load")
    load_parser.add_argument("--target", type=str, choices=["vector", "retrieval"], default="retrieval",
                            help="Where to load the data (legacy vector store or retrieval API)")
    load_parser.add_argument("--dry-run", action="store_true", help="Validate without loading")
    load_parser.add_argument("--store-dir", type=str, help="Vector store directory (for vector target)")
    load_parser.add_argument("--assistant-id", type=str, help="Assistant ID (for retrieval target)")
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

def setup_visualize_parser(subparsers):
    """Set up the parser for the visualize command"""
    visualize_parser = subparsers.add_parser(
        'visualize', 
        help='Visualize Jon\'s memory structure'
    )
    
    visualize_parser.add_argument("--assistant-id", type=str, help="Assistant ID to visualize")
    visualize_parser.add_argument("--input", "-i", type=str, help="Input file with memory data (JSON/JSONL)")
    visualize_parser.add_argument("--output", "-o", type=str, help="Output directory for visualizations")
    visualize_parser.add_argument("--format", type=str, choices=["html", "png", "json"], 
                              default="html", help="Output format")
    visualize_parser.add_argument("--type", type=str, choices=["graph", "topics", "sentiment", "dashboard", "all"],
                              default="all", help="Type of visualization to generate")
    
    return visualize_parser

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

def setup_migrate_parser(subparsers):
    """Set up the parser for the migrate command"""
    migrate_parser = subparsers.add_parser(
        'migrate', 
        help='Migrate data from vector store to retrieval API'
    )
    
    migrate_parser.add_argument("--vector-dir", type=str, default="vector_store", help="Vector store directory")
    migrate_parser.add_argument("--assistant-id", type=str, help="Assistant ID to use (creates new one if not provided)")
    migrate_parser.add_argument("--batch-size", type=int, default=20, help="Batch size for uploads")
    migrate_parser.add_argument("--dry-run", action="store_true", help="Don't upload, just show what would be uploaded")
    
    return migrate_parser

def handle_generate(args):
    """Handle the generate command"""
    # Convert args to sys.argv format for the original script
    sys_args = ["data_generation.generation.jon_data_generator"]
    
    if args.qa_pairs is not None:
        sys_args.extend(["--qa-pairs", str(args.qa_pairs)])
    if args.conversations is not None:
        sys_args.extend(["--conversations", str(args.conversations)])
    if args.statements is not None:
        sys_args.extend(["--statements", str(args.statements)])
    if args.batch_size is not None:
        sys_args.extend(["--batch-size", str(args.batch_size)])
    if args.variations is not None:
        sys_args.extend(["--variations", str(args.variations)])
    if args.contrastive:
        sys_args.append("--contrastive")
    if args.output_dir is not None:
        sys_args.extend(["--output-dir", args.output_dir])
    if args.optimize_chunks:
        sys_args.append("--optimize-chunks")
    if args.parallel:
        sys_args.append("--parallel")
    if args.dynamic_batching:
        sys_args.append("--dynamic-batching")
    if args.enrich_metadata:
        sys_args.append("--enrich-metadata")
    
    # Temporarily replace sys.argv
    old_argv = sys.argv
    sys.argv = sys_args
    
    try:
        # Call the original main function
        generate_main()
    finally:
        # Restore sys.argv
        sys.argv = old_argv

def handle_load(args):
    """Handle the load command"""
    if args.target == "vector":
        # Vector store loading
        sys_args = ["data_generation.loading.load_jon_data", "--file", args.file]
        
        if args.dry_run:
            sys_args.append("--dry-run")
        if args.store_dir:
            sys_args.extend(["--store-dir", args.store_dir])
        
        # Temporarily replace sys.argv
        old_argv = sys.argv
        sys.argv = sys_args
        
        try:
            # Call the original main function
            load_vector_main()
        finally:
            # Restore sys.argv
            sys.argv = old_argv
    else:
        # Retrieval API loading
        sys_args = ["data_generation.loading.load_jon_retrieval", "--file", args.file]
        
        if args.dry_run:
            sys_args.append("--dry-run")
        if args.assistant_id:
            sys_args.extend(["--assistant-id", args.assistant_id])
        if args.batch_size is not None:
            sys_args.extend(["--batch-size", str(args.batch_size)])
        
        # Temporarily replace sys.argv
        old_argv = sys.argv
        sys.argv = sys_args
        
        try:
            # Call the original main function
            load_retrieval_main()
        finally:
            # Restore sys.argv
            sys.argv = old_argv

def handle_train(args):
    """Handle the train command"""
    if args.train_cmd == "prepare":
        # Prepare fine-tuning data
        sys_args = ["data_generation.training.prepare_fine_tuning", "--file", args.file]
        
        if args.output:
            sys_args.extend(["--output", args.output])
        if args.test_split is not None:
            sys_args.extend(["--test-split", str(args.test_split)])
        if args.fine_tune:
            sys_args.append("--fine-tune")
            if args.model:
                sys_args.extend(["--model", args.model])
            if args.suffix:
                sys_args.extend(["--suffix", args.suffix])
        
        # Temporarily replace sys.argv
        old_argv = sys.argv
        sys.argv = sys_args
        
        try:
            # Call the original main function
            prepare_ft_main()
        finally:
            # Restore sys.argv
            sys.argv = old_argv
    else:
        # Monitor jobs commands
        if args.train_cmd == "list":
            sys_args = ["data_generation.training.monitor_jobs", "list"]
            if args.limit:
                sys_args.extend(["--limit", str(args.limit)])
        elif args.train_cmd == "status":
            sys_args = ["data_generation.training.monitor_jobs", "status", args.job_id]
            if args.register:
                sys_args.append("--register")
        elif args.train_cmd == "cancel":
            sys_args = ["data_generation.training.monitor_jobs", "cancel", args.job_id]
        
        # Temporarily replace sys.argv
        old_argv = sys.argv
        sys.argv = sys_args
        
        try:
            # Call the original main function
            monitor_jobs_main()
        finally:
            # Restore sys.argv
            sys.argv = old_argv

def handle_assistants(args):
    """Handle the assistants command"""
    # Map commands to monitor_jobs.py commands
    if args.assistants_cmd == "list":
        sys_args = ["data_generation.loading.manage_assistants", "list"]
        if args.limit:
            sys_args.extend(["--limit", str(args.limit)])
    elif args.assistants_cmd == "details":
        sys_args = ["data_generation.loading.manage_assistants", "details", "--id", args.id]
    elif args.assistants_cmd == "create":
        sys_args = ["data_generation.loading.manage_assistants", "create", "--name", args.name]
        if args.model:
            sys_args.extend(["--model", args.model])
        if args.instructions:
            sys_args.extend(["--instructions", args.instructions])
        if args.save:
            sys_args.append("--save")
    elif args.assistants_cmd == "clean":
        sys_args = ["data_generation.loading.manage_assistants", "clean", "--id", args.id]
        if args.age:
            sys_args.extend(["--age", str(args.age)])
        if args.dry_run:
            sys_args.append("--dry-run")
    
    # Temporarily replace sys.argv
    old_argv = sys.argv
    sys.argv = sys_args
    
    try:
        # Call the original main function
        manage_assistants_main()
    finally:
        # Restore sys.argv
        sys.argv = old_argv

def handle_visualize(args):
    """Handle the visualize command"""
    sys_args = ["data_generation.visualization.visualize_memory"]
    
    if args.assistant_id:
        sys_args.extend(["--assistant-id", args.assistant_id])
    if args.input:
        sys_args.extend(["--input", args.input])
    if args.output:
        sys_args.extend(["--output", args.output])
    if args.format:
        sys_args.extend(["--format", args.format])
    if args.type:
        sys_args.extend(["--type", args.type])
    
    # Temporarily replace sys.argv
    old_argv = sys.argv
    sys.argv = sys_args
    
    try:
        # Call the original main function
        visualize_main()
    finally:
        # Restore sys.argv
        sys.argv = old_argv

def handle_test(args):
    """Handle the test command"""
    sys_args = ["data_generation.utils.test_jon_data"]
    
    if args.file:
        sys_args.extend(["--file", args.file])
    if args.type:
        sys_args.extend(["--type", args.type])
    if args.interactive:
        sys_args.append("--interactive")
    if args.model:
        sys_args.extend(["--model", args.model])
    if args.assistant_id:
        sys_args.extend(["--assistant-id", args.assistant_id])
    if args.limit:
        sys_args.extend(["--limit", str(args.limit)])
    
    # Temporarily replace sys.argv
    old_argv = sys.argv
    sys.argv = sys_args
    
    try:
        # Call the original main function
        test_main()
    finally:
        # Restore sys.argv
        sys.argv = old_argv

def handle_migrate(args):
    """Handle the migrate command"""
    if migrate_main is None:
        print("Error: Migration tool is not available in this installation.")
        print("Please try updating your SchultzGPT installation.")
        return
    
    sys_args = ["data_generation.migration.migrate_vector_to_retrieval"]
    
    if args.vector_dir:
        sys_args.extend(["--vector-dir", args.vector_dir])
    if args.assistant_id:
        sys_args.extend(["--assistant-id", args.assistant_id])
    if args.batch_size:
        sys_args.extend(["--batch-size", str(args.batch_size)])
    if args.dry_run:
        sys_args.append("--dry-run")
    
    # Temporarily replace sys.argv
    old_argv = sys.argv
    sys.argv = sys_args
    
    try:
        # Call the original main function
        migrate_main()
    finally:
        # Restore sys.argv
        sys.argv = old_argv

def main():
    """Main entry point for the script"""
    # Set up the argument parser
    parser = argparse.ArgumentParser(
        description="Jon Tools - Unified CLI for SchultzGPT Data Tools",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m data_generation.jon_tools generate --qa-pairs 100 --batch-size 10
  python -m data_generation.jon_tools load --file data/jon_retrieval_data.jsonl
  python -m data_generation.jon_tools train prepare --file data/jon_fine_tuning.jsonl
  python -m data_generation.jon_tools train list
  python -m data_generation.jon_tools train status ft-abc123 --register
  python -m data_generation.jon_tools assistants list
  python -m data_generation.jon_tools assistants create --name "Jon Memory" --save
  python -m data_generation.jon_tools visualize --input data/jon_raw_data.json
  python -m data_generation.jon_tools test --file data/jon_retrieval_data.jsonl --interactive
  python -m data_generation.jon_tools migrate --vector-dir vector_store --dry-run
        """
    )
    
    # Add subparsers for the various commands
    subparsers = parser.add_subparsers(dest='command', required=True)
    
    # Set up parsers for each command
    generate_parser = setup_generate_parser(subparsers)
    load_parser = setup_load_parser(subparsers)
    train_parser = setup_train_parser(subparsers)
    assistants_parser = setup_assistants_parser(subparsers)
    visualize_parser = setup_visualize_parser(subparsers)
    test_parser = setup_test_parser(subparsers)
    
    # Only set up migrate parser if the migrate function is available
    if migrate_main is not None:
        migrate_parser = setup_migrate_parser(subparsers)
    
    # Parse arguments
    args = parser.parse_args()
    
    # Handle the specified command
    if args.command == 'generate':
        handle_generate(args)
    elif args.command == 'load':
        handle_load(args)
    elif args.command == 'train':
        handle_train(args)
    elif args.command == 'assistants':
        handle_assistants(args)
    elif args.command == 'visualize':
        handle_visualize(args)
    elif args.command == 'test':
        handle_test(args)
    elif args.command == 'migrate' and migrate_main is not None:
        handle_migrate(args)

if __name__ == "__main__":
    main() 