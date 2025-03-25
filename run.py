#!/usr/bin/env python
"""
Run script for SchultzGPT - execute from project root.
"""

import os
import sys
import traceback
from dotenv import load_dotenv

# Add the src directory to the path
src_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
sys.path.insert(0, src_path)

# Import components
from controllers.controller import SchultzController
from ui.terminal import TerminalUI, console


def setup_environment():
    """Check and setup the environment for SchultzGPT."""
    # Load environment variables
    load_dotenv()
    
    # Check for required environment variables
    required_vars = ["OPENAI_API_KEY"]
    missing_vars = [var for var in required_vars if not os.environ.get(var)]
    
    if missing_vars:
        console.print("[bold red]Error:[/bold red] Missing required environment variables:")
        for var in missing_vars:
            console.print(f"  - {var}")
        console.print("\nPlease set these variables in your .env file or environment.")
        return False
    
    # Create required directories if they don't exist
    directories = [
        ".cache",
        ".cache/openai",
        ".retrieval_store",
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    
    return True


def main():
    """Main entry point for SchultzGPT."""
    try:
        # Setup UI
        ui = TerminalUI()
        
        # Show startup message
        ui.clear_screen()
        ui.render_header()
        ui.render_info("Initializing SchultzGPT...")
        
        # Setup environment
        if not setup_environment():
            ui.render_error("Failed to initialize. Exiting.")
            sys.exit(1)
        
        # Initialize controller
        controller = SchultzController()
        
        # Run the application
        controller.run()
        
    except KeyboardInterrupt:
        # Handle Ctrl+C gracefully
        console.print("\n[yellow]SchultzGPT terminated by user. Goodbye![/yellow]")
        sys.exit(0)
        
    except Exception as e:
        # Handle unexpected errors
        console.print("[bold red]Unhandled exception:[/bold red]")
        console.print(str(e))
        console.print(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main() 