#!/usr/bin/env python
"""
Manage Assistants Utility

This script provides utilities for managing OpenAI Assistants used by SchultzGPT.
It allows listing, creating, and cleaning up assistants.

Usage:
    python -m data_generation.manage_assistants [command]
"""

import os
import sys
import json
import argparse
import time
from typing import List, Dict, Any, Optional
from tabulate import tabulate

# Add src to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()

def list_assistants(client: OpenAI, verbose: bool = False) -> List[Dict[str, Any]]:
    """List all assistants associated with the account"""
    try:
        assistants = client.beta.assistants.list(limit=100)
        
        result = []
        for assistant in assistants.data:
            asst_data = {
                "id": assistant.id,
                "name": assistant.name,
                "description": assistant.description,
                "model": assistant.model,
                "created_at": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(assistant.created_at)),
                "file_count": len(assistant.file_ids) if hasattr(assistant, 'file_ids') else 0
            }
            
            # Add verbose data if requested
            if verbose:
                tools = [tool.type for tool in assistant.tools] if hasattr(assistant, 'tools') else []
                asst_data["tools"] = ", ".join(tools) if tools else "none"
                
                # Get file details if available
                if hasattr(assistant, 'file_ids') and assistant.file_ids:
                    asst_data["files"] = assistant.file_ids
            
            result.append(asst_data)
        
        return result
    except Exception as e:
        print(f"Error listing assistants: {e}")
        return []

def create_assistant(client: OpenAI, name: str, model: str, description: str = None, 
                    instructions: str = None) -> Optional[Dict[str, Any]]:
    """Create a new assistant"""
    try:
        # Set default description if none provided
        if not description:
            description = f"Assistant created by SchultzGPT on {time.strftime('%Y-%m-%d')}"
        
        # Set default instructions if none provided
        if not instructions:
            instructions = "You are a helpful assistant that provides information and assistance."
        
        # Create the assistant
        assistant = client.beta.assistants.create(
            name=name,
            description=description,
            instructions=instructions,
            model=model,
            tools=[{"type": "retrieval"}]
        )
        
        # Format response
        return {
            "id": assistant.id,
            "name": assistant.name,
            "description": assistant.description,
            "model": assistant.model,
            "created_at": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(assistant.created_at))
        }
    except Exception as e:
        print(f"Error creating assistant: {e}")
        return None

def get_assistant_details(client: OpenAI, assistant_id: str) -> Optional[Dict[str, Any]]:
    """Get detailed information about a specific assistant"""
    try:
        # Retrieve the assistant
        assistant = client.beta.assistants.retrieve(assistant_id)
        
        # Get all files associated with this assistant
        files = []
        if hasattr(assistant, 'file_ids') and assistant.file_ids:
            for file_id in assistant.file_ids:
                try:
                    file = client.files.retrieve(file_id)
                    files.append({
                        "id": file.id,
                        "filename": file.filename,
                        "purpose": file.purpose,
                        "created_at": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(file.created_at)),
                        "bytes": file.bytes
                    })
                except Exception as file_err:
                    files.append({"id": file_id, "error": str(file_err)})
        
        # Format response
        details = {
            "id": assistant.id,
            "name": assistant.name,
            "description": assistant.description,
            "model": assistant.model,
            "created_at": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(assistant.created_at)),
            "instructions": assistant.instructions,
            "tools": [tool.type for tool in assistant.tools] if hasattr(assistant, 'tools') else [],
            "file_count": len(assistant.file_ids) if hasattr(assistant, 'file_ids') else 0,
            "files": files
        }
        
        return details
    except Exception as e:
        print(f"Error retrieving assistant details: {e}")
        return None

def delete_files(client: OpenAI, assistant_id: str, age_days: int = None) -> Dict[str, Any]:
    """Delete files from an assistant based on age"""
    try:
        # Get the assistant details first
        assistant = client.beta.assistants.retrieve(assistant_id)
        
        if not hasattr(assistant, 'file_ids') or not assistant.file_ids:
            return {"status": "no_files", "message": "No files to delete"}
        
        file_ids = assistant.file_ids
        original_count = len(file_ids)
        
        # If no age specified, don't delete anything
        if age_days is None:
            return {
                "status": "skipped", 
                "message": f"No files deleted. Use --age parameter to specify file age in days.",
                "file_count": original_count
            }
        
        # Determine cutoff time
        cutoff_time = time.time() - (age_days * 24 * 60 * 60)
        files_to_keep = []
        files_to_delete = []
        
        # Check each file's age
        for file_id in file_ids:
            try:
                file = client.files.retrieve(file_id)
                if file.created_at < cutoff_time:
                    files_to_delete.append(file_id)
                else:
                    files_to_keep.append(file_id)
            except Exception:
                # If we can't retrieve the file, assume it's already deleted
                pass
        
        # Update the assistant with remaining files
        if files_to_delete:
            client.beta.assistants.update(
                assistant_id=assistant_id,
                file_ids=files_to_keep
            )
            
            # Try to delete the files from OpenAI
            deleted_count = 0
            for file_id in files_to_delete:
                try:
                    client.files.delete(file_id)
                    deleted_count += 1
                except Exception:
                    # File might be in use or already deleted
                    pass
        
        return {
            "status": "success",
            "original_count": original_count,
            "deleted_count": len(files_to_delete),
            "remaining_count": len(files_to_keep),
            "actually_deleted": deleted_count
        }
    except Exception as e:
        print(f"Error deleting files: {e}")
        return {"status": "error", "message": str(e)}

def save_assistant_id_to_env(assistant_id: str) -> bool:
    """Save an assistant ID to the .env file"""
    try:
        # Check if .env file exists
        if os.path.exists(".env"):
            # Read current .env file
            with open(".env", "r") as f:
                env_content = f.read()
            
            # Check if OPENAI_ASSISTANT_ID is already in the file
            if "OPENAI_ASSISTANT_ID=" in env_content:
                # Replace the existing assistant ID
                new_env_content = []
                for line in env_content.split("\n"):
                    if line.startswith("OPENAI_ASSISTANT_ID="):
                        new_env_content.append(f"OPENAI_ASSISTANT_ID={assistant_id}")
                    else:
                        new_env_content.append(line)
                
                # Write updated content
                with open(".env", "w") as f:
                    f.write("\n".join(new_env_content))
            else:
                # Append the assistant ID to the file
                with open(".env", "a") as f:
                    f.write(f"\n# Added automatically by SchultzGPT\nOPENAI_ASSISTANT_ID={assistant_id}\n")
            
            print(f"Saved Assistant ID to .env file: {assistant_id}")
            return True
        else:
            # Create a new .env file
            with open(".env", "w") as f:
                f.write(f"OPENAI_ASSISTANT_ID={assistant_id}\n")
            
            print(f"Created .env file with Assistant ID: {assistant_id}")
            return True
    except Exception as e:
        print(f"Error saving Assistant ID to .env file: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Manage OpenAI Assistants for SchultzGPT")
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # List command
    list_parser = subparsers.add_parser("list", help="List all assistants")
    list_parser.add_argument("--verbose", "-v", action="store_true", help="Show detailed information")
    list_parser.add_argument("--output", "-o", type=str, help="Output file path (JSON format)")
    
    # Create command
    create_parser = subparsers.add_parser("create", help="Create a new assistant")
    create_parser.add_argument("--name", "-n", type=str, required=True, help="Assistant name")
    create_parser.add_argument("--model", "-m", type=str, default="gpt-4o", help="Model to use")
    create_parser.add_argument("--description", "-d", type=str, help="Assistant description")
    create_parser.add_argument("--instructions", "-i", type=str, help="Assistant instructions")
    create_parser.add_argument("--save", "-s", action="store_true", help="Save assistant ID to .env file")
    
    # Get details command
    details_parser = subparsers.add_parser("details", help="Get details about an assistant")
    details_parser.add_argument("--id", type=str, required=True, help="Assistant ID")
    details_parser.add_argument("--output", "-o", type=str, help="Output file path (JSON format)")
    
    # Delete files command
    delete_parser = subparsers.add_parser("clean", help="Delete files from an assistant")
    delete_parser.add_argument("--id", type=str, required=True, help="Assistant ID")
    delete_parser.add_argument("--age", type=int, help="Delete files older than X days")
    delete_parser.add_argument("--all", action="store_true", help="Delete all files")
    
    args = parser.parse_args()
    
    # Initialize OpenAI client
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    
    # Process commands
    if args.command == "list":
        assistants = list_assistants(client, args.verbose)
        
        if assistants:
            # Prepare table headers based on verbosity
            headers = ["ID", "Name", "Description", "Model", "Created", "Files"]
            if args.verbose:
                headers.extend(["Tools"])
            
            # Prepare table rows
            rows = []
            for asst in assistants:
                row = [
                    asst["id"],
                    asst["name"],
                    asst["description"][:30] + "..." if len(asst["description"]) > 30 else asst["description"],
                    asst["model"],
                    asst["created_at"],
                    asst["file_count"]
                ]
                
                if args.verbose:
                    row.append(asst.get("tools", ""))
                
                rows.append(row)
            
            # Print table
            print(tabulate(rows, headers=headers, tablefmt="grid"))
            print(f"\nTotal assistants: {len(assistants)}")
            
            # Save to file if requested
            if args.output:
                with open(args.output, "w") as f:
                    json.dump(assistants, f, indent=2)
                print(f"Results saved to {args.output}")
        else:
            print("No assistants found or error occurred.")
    
    elif args.command == "create":
        assistant = create_assistant(
            client, 
            name=args.name,
            model=args.model,
            description=args.description,
            instructions=args.instructions
        )
        
        if assistant:
            print("\nAssistant created successfully:")
            for key, value in assistant.items():
                print(f"{key}: {value}")
            
            # Save to .env if requested
            if args.save:
                save_assistant_id_to_env(assistant["id"])
        else:
            print("Failed to create assistant.")
    
    elif args.command == "details":
        details = get_assistant_details(client, args.id)
        
        if details:
            print("\nAssistant Details:")
            print(f"ID: {details['id']}")
            print(f"Name: {details['name']}")
            print(f"Description: {details['description']}")
            print(f"Model: {details['model']}")
            print(f"Created: {details['created_at']}")
            print(f"Instructions: {details['instructions'][:100]}..." if len(details['instructions']) > 100 else details['instructions'])
            print(f"Tools: {', '.join(details['tools'])}")
            print(f"Files: {details['file_count']}")
            
            if details['files']:
                print("\nFile Details:")
                file_table = []
                for file in details['files']:
                    if "error" in file:
                        file_table.append([file["id"], "ERROR", file["error"], "", ""])
                    else:
                        file_table.append([
                            file["id"], 
                            file["filename"], 
                            file["purpose"], 
                            file["created_at"],
                            f"{file['bytes'] / 1024:.1f} KB"
                        ])
                
                print(tabulate(file_table, headers=["ID", "Filename", "Purpose", "Created", "Size"], tablefmt="grid"))
            
            # Save to file if requested
            if args.output:
                with open(args.output, "w") as f:
                    json.dump(details, f, indent=2)
                print(f"Details saved to {args.output}")
        else:
            print(f"Failed to retrieve details for assistant {args.id}")
    
    elif args.command == "clean":
        if not args.age and not args.all:
            print("Error: Must specify either --age or --all")
            return
        
        age = None if args.all else args.age
        result = delete_files(client, args.id, age)
        
        if result["status"] == "success":
            print(f"Files cleaning complete for assistant {args.id}:")
            print(f"Original file count: {result['original_count']}")
            print(f"Files removed: {result['deleted_count']}")
            print(f"Remaining files: {result['remaining_count']}")
        else:
            print(f"Status: {result['status']}")
            print(f"Message: {result.get('message', 'No additional information')}")
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main() 