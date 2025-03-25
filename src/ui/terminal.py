import os
import sys
from typing import Dict, List, Optional, Any, Callable

from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.align import Align
from rich.rule import Rule
from rich.box import Box
from rich.markdown import Markdown
from rich.prompt import Prompt
from rich.live import Live
from rich.layout import Layout
from rich.table import Table

from config.config import Config
from src.ui.style_provider import StyleProvider

# Create console for rich text rendering
console = Console()

class TerminalUI:
    """Terminal user interface components for SchultzGPT"""
    
    def __init__(self, message_manager, controller):
        self.message_manager = message_manager
        self.controller = controller
        self.console = console
        self.style_provider = StyleProvider()
    
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
            # Determine memory status icon and text
            if message_manager.retrieval_store_available:
                mem_icon = "🧠"
                mem_status = "connected"
            else:
                mem_icon = "💭"
                mem_status = "limited"
            
            # Get mood emoji based on current mood
            mood = jon_state.mood.lower()
            mood_emoji = {
                "neutral": "😐",
                "happy": "😊",
                "amused": "😄",
                "sarcastic": "😏",
                "cynical": "🙄",
                "thoughtful": "🤔",
                "supportive": "👍",
                "irritated": "😒",
                "tired": "😴",
                "spiral": "🌀",
                "anxious": "😰",
                "philosophical": "🧐"
            }.get(mood, "🤖")
            
            # Get model name (abbreviated if needed)
            model_name = Config.FINE_TUNED_MODEL
            model_short = model_name
            if len(model_name) > 12:
                model_short = model_name.split(":")[-1] if ":" in model_name else model_name[:10] + "..."
            
            # Build stylish status line with emojis
            status_line = f"{mood_emoji} jon: {mood} | {mem_icon} memory: {mem_status} | 📝 model: {model_short} | /help for commands"
            
            # Display the footer
            console.print(f"[dim]{status_line}[/dim]")
        except Exception as e:
            # Fallback status if something fails
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

    def render_footer(self) -> None:
        """Render the footer with status information."""
        # Get current status
        storage_status = "connected" if self.message_manager.retrieval_store_available else "disconnected"
        model_name = self.controller.model_name
        
        # Get Jon's current mood
        mood = self.controller.state.mood
        
        # Map moods to emojis
        mood_emojis = {
            "happy": "😊",
            "sad": "😔",
            "angry": "😠",
            "confused": "😕",
            "excited": "😃",
            "thoughtful": "🤔",
            "neutral": "😐",
            "curious": "🧐",
            "anxious": "😰",
            "bored": "😒",
            "suspicious": "🤨",
            "mischievous": "😏",
            "surprised": "😮",
            "tired": "😴",
            "amused": "😄",
            "concerned": "🙁",
            "calm": "😌",
            "frustrated": "😤",
            "numb": "😶",
            "reflective": "🙂"
        }
        
        # Get emoji for current mood
        mood_emoji = mood_emojis.get(mood.lower(), "😐")
        
        # Style for different sections
        mood_style = self.style_provider.mood_style(mood)
        storage_style = self.style_provider.status_style(storage_status == "connected")
        model_style = self.style_provider.model_style(model_name)
        
        # Create status line with emojis
        footer_text = f"{mood_emoji} jon: {self.console.stylize(mood, mood_style)} | 🧠 memory: {self.console.stylize(storage_status, storage_style)} | 📝 model: {self.console.stylize(model_name, model_style)} | /help for commands"
        
        # Print the footer with padding
        footer_width = self.console.width
        self.console.print("─" * footer_width, style="dim")
        self.console.print(footer_text, justify="center")

    def render_welcome(self) -> None:
        """Render the welcome message"""
        # Create a nice looking welcome panel
        welcome_text = f"""
# Welcome to SchultzGPT

A terminal-based AI persona chatbot with OpenAI Retrieval API-backed memory.

- Type a message to start chatting with Jon
- Type /help to see available commands
- Press Ctrl+C to exit
        """
        
        panel = Panel(
            Markdown(welcome_text),
            title="SchultzGPT",
            subtitle=f"v{Config.VERSION}",
            border_style="cyan"
        )
        
        self.console.print(panel)
    
    def render_help(self, commands: Dict[str, str]) -> None:
        """Render the help message with available commands"""
        # Create a table for commands
        table = Table(title="Available Commands", show_header=True, header_style="bold")
        table.add_column("Command", style=self.style_provider.command_style())
        table.add_column("Description")
        
        # Add each command to the table
        for command, description in commands.items():
            table.add_row(command, description)
        
        # Print the table
        self.console.print(table)
    
    def render_status(self, state) -> None:
        """Render the current application status"""
        # Create a table for status information
        table = Table(title="SchultzGPT Status", show_header=True, header_style="bold")
        table.add_column("Setting", style="cyan")
        table.add_column("Value")
        
        # Add status information
        table.add_row("Jon's Mood", state.mood)
        table.add_row("Context Window Size", str(state.context_window_size))
        table.add_row("Temperature Modifier", f"{state.temperature_modifier:+.1f}")
        table.add_row("Caching Enabled", "Yes" if state.caching_enabled else "No")
        table.add_row("Debug Mode", "Enabled" if state.debug_mode else "Disabled")
        table.add_row("Retrieval Memory", "Connected" if self.message_manager.retrieval_store_available else "Disconnected")
        table.add_row("Retrieval Enabled", "Yes" if state.retrieval_store_enabled else "No")
        table.add_row("Model", self.controller.model_name)
        
        # Print the table
        self.console.print(table)
    
    def render_performance(self, metrics: Dict[str, Any]) -> None:
        """Render performance metrics"""
        # Create a table for performance metrics
        table = Table(title="Performance Metrics", show_header=True, header_style="bold")
        table.add_column("Metric", style="cyan")
        table.add_column("Value")
        
        # Add metrics to the table
        for metric, value in metrics.items():
            # Format metrics nicely
            if metric == "avg_response_time":
                formatted_value = f"{value:.2f} seconds"
            elif metric == "total_tokens_used":
                formatted_value = f"{value:,}"
            else:
                formatted_value = str(value)
            
            # Convert snake_case to Title Case for display
            display_metric = " ".join(word.capitalize() for word in metric.split("_"))
            table.add_row(display_metric, formatted_value)
        
        # Print the table
        self.console.print(table)
    
    def get_user_input(self, prompt: str = "You: ") -> str:
        """Get input from the user"""
        return Prompt.ask(prompt) 