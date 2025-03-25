import os
import sys
from typing import Dict, List, Optional

from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.align import Align
from rich.rule import Rule
from rich.box import Box

# Create console for rich text rendering
console = Console()

class TerminalUI:
    """Terminal user interface components for SchultzGPT"""
    
    @staticmethod
    def clear_screen():
        """Clear the terminal screen"""
        os.system('cls' if os.name == 'nt' else 'clear')
    
    @staticmethod
    def render_header():
        """Render a simple header"""
        console.print("\n[bold cyan]SchultzGPT[/bold cyan]")
        console.print("-------------")
    
    @staticmethod
    def render_message(content: str, is_user: bool):
        """Render a simple message"""
        try:
            if is_user:
                console.print(f"[cyan]You:[/cyan] {content}")
            else:
                console.print(f"[green]Jon:[/green] {content}")
        except Exception as e:
            # Fallback to plaintext if rich fails
            if is_user:
                print(f"You: {content}")
            else:
                print(f"Jon: {content}")
            print(f"UI Error: {str(e)}")

    @staticmethod
    def render_command_output(command: str, output: str):
        """Render a command and its output clearly"""
        try:
            console.print(f"[yellow]{command}:[/yellow] {output}")
        except Exception as e:
            # Fallback to simpler rendering
            print(f"{command}: {output}")
            print(f"UI Error: {str(e)}")

    @staticmethod
    def render_footer(jon_state, message_manager):
        """Render a simple footer with essential status information"""
        try:
            # Determine memory status
            if message_manager.vector_store_available:
                mem_status = "Vector store: Connected"
            else:
                mem_status = "Vector store: Disconnected"
                
            # Simple status line
            console.print(f"[dim]{mem_status} | Type /help for commands[/dim]")
        except Exception as e:
            print("Status: Active | Type /help for commands")

    @staticmethod
    def handle_resize():
        """Handle terminal resize event"""
        try:
            # Get new terminal dimensions
            new_width = os.get_terminal_size().columns
            new_height = os.get_terminal_size().lines
            
            # Redraw the UI elements as needed
            TerminalUI.clear_screen()
            TerminalUI.render_header()
            console.print(f"[dim]Terminal resized to {new_width}x{new_height}[/dim]")
        except Exception as e:
            print(f"Resize error: {str(e)}")
            
    @staticmethod
    def show_help(commands):
        """Show help information for available commands"""
        try:
            help_text = "Available commands:\n\n"
            for cmd, desc in commands:
                help_text += f"[cyan]{cmd}[/cyan] - {desc}\n"
                
            console.print(help_text)
        except Exception as e:
            # Fallback simple help
            print("COMMANDS:")
            for cmd, desc in commands:
                print(f"{cmd} - {desc}")
            print(f"Error showing help: {str(e)}")
    
    @staticmethod
    def render_structured_message(content: str, is_user: bool, metadata: Optional[Dict] = None):
        """Render a message with minimal formatting"""
        try:
            if is_user:
                # Simple user message
                console.print(f"[cyan]You:[/cyan] {content}")
            else:
                # Simple Jon message with minimal metadata
                mood_str = ""
                if metadata and "mood" in metadata and metadata["mood"] != "neutral":
                    mood_str = f" [{metadata['mood']}]"
                
                console.print(f"[green]Jon{mood_str}:[/green] {content}")
        except Exception as e:
            # Fallback to plaintext if rich fails
            if is_user:
                print(f"You: {content}")
            else:
                print(f"Jon: {content}")

    @staticmethod
    def stream_structured_response(cached_chat_completion, messages: List[dict], temperature_modifier: float = 0, mood: str = "neutral", model=None) -> Optional[Dict]:
        """Get response with minimal complexity"""
        try:
            # Simple typing indicator
            with console.status("Processing response..."):
                # Get user message and system message
                user_message = messages[-1]["content"]
                system_message = messages[0]["content"]
                
                # Use fixed temperature with simple modifier
                temperature = 0.9 + temperature_modifier
                
                # Add structured output format to the system message
                structured_system = system_message + "\n\nReturn a JSON response with 'response' text and metadata including 'mood' and 'topics'."
                
                # Replace the system message with the structured version
                structured_messages = [{"role": "system", "content": structured_system}]
                structured_messages.extend(messages[1:])
                
                try:
                    # Use cached chat completion
                    response = cached_chat_completion(
                        model=model,
                        messages=structured_messages,
                        temperature=temperature,
                        max_tokens=150,  # Should be configurable
                        response_format={"type": "json_object"}
                    )
                    
                    # Process the structured response
                    import json
                    result = json.loads(response.choices[0].message.content)
                    
                    # Extract the main response text
                    response_text = result.get("response", "")
                    if not response_text and "text" in result:
                        response_text = result.get("text", "")
                    
                    # If we still don't have a response, use fallback
                    if not response_text:
                        response_text = "sorry i cant really focus right now"
                    
                    # Build the result with metadata
                    structured_result = {
                        "response": response_text,
                        "mood": result.get("mood", mood),
                        "topics": result.get("topics", [])
                    }
                    
                    # Render the response
                    TerminalUI.render_structured_message(response_text, False, structured_result)
                    
                    return structured_result
                
                except Exception as api_error:
                    console.print(f"Error: {str(api_error)}")
                    fallback = {"response": "sorry, having trouble connecting", "mood": mood}
                    TerminalUI.render_structured_message(fallback["response"], False, fallback)
                    return fallback
                    
        except Exception as e:
            console.print(f"Error: {str(e)}")
            fallback = {"response": "something's wrong with the system", "mood": "neutral"}
            TerminalUI.render_structured_message(fallback["response"], False, fallback)
            return fallback

    @staticmethod
    def get_user_input():
        """Get input from the user with a nice prompt"""
        try:
            # Show a clear prompt for user input
            console.print()
            prompt = "[bold cyan]You: [/bold cyan]"
            user_message = input(prompt)
            return user_message.strip()
        except KeyboardInterrupt:
            # Handle Ctrl+C gracefully
            console.print("\n[yellow]Input interrupted. Type 'exit' to quit.[/yellow]")
            return ""
        except EOFError:
            # Handle Ctrl+D gracefully
            console.print("\n[yellow]Exiting SchultzGPT. Goodbye![/yellow]")
            sys.exit(0)
            
    @staticmethod
    def render_error(error_message):
        """Render an error message"""
        console.print(f"[bold red]Error: {error_message}[/bold red]")
        
    @staticmethod
    def render_info(message):
        """Render an informational message"""
        console.print(f"[yellow]{message}[/yellow]") 