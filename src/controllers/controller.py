"""
Main controller for SchultzGPT.
Handles orchestration between UI, services, and models.
"""

import os
import time
import asyncio
from typing import Dict, List, Any, Optional, Union, Callable, Tuple
import traceback

from dotenv import load_dotenv
from openai import OpenAI

from models.state import SchultzState
from models.message import Message, MessageHistory
from services.message_manager import MessageManager
from services.retrieval_store import RetrievalStore
from services.openai import chat_completion, async_chat_completion, run_async
from services.performance import PerformanceTracker, timed, async_timed
from config.config import Config
from ui.terminal import TerminalUI


class SchultzController:
    """Main controller for SchultzGPT application."""
    
    def __init__(self):
        """Initialize the SchultzGPT controller."""
        # Load environment variables
        load_dotenv()
        
        # Initialize state
        self.state = SchultzState()
        
        # Initialize retrieval store
        self.retrieval_store = RetrievalStore()
        
        # Initialize performance tracker
        self.performance_tracker = PerformanceTracker(
            enable_file_logging=self.state.debug_mode
        )
        
        # Initialize message manager
        self.message_manager = MessageManager(
            state=self.state,
            system_message=Config.SYSTEM_PROMPT,
            retrieval_store=self.retrieval_store
        )
        
        # Initialize UI
        self.ui = TerminalUI()
        
        # Initialize OpenAI client
        self.openai_client = OpenAI(
            api_key=os.environ.get("OPENAI_API_KEY"),
            organization=os.environ.get("OPENAI_ORG_ID")
        )
        
        # Register commands
        self.commands = {
            "/help": self.show_help,
            "/clear": self.clear_screen,
            "/exit": self.exit_app,
            "/quit": self.exit_app,
            "/reset": self.reset_conversation,
            "/toggle-cache": self.toggle_cache,
            "/toggle-debug": self.toggle_debug,
            "/toggle-retrieval": self.toggle_retrieval_store,
            "/context": self.show_context_size,
            "/set-context": self.set_context_size,
            "/set-temp": self.set_temperature,
            "/performance": self.show_performance,
            "/clear-metrics": self.clear_metrics,
            "/status": self.show_status,
            "/summarize": self.summarize_conversation,
            "/load": self.load_data
        }
        
        # Set initialized flag
        self.initialized = True
    
    @timed
    def show_help(self, args: List[str] = None) -> str:
        """Show help information."""
        commands = [
            ("/help", "Show available commands"),
            ("/clear", "Clear the terminal screen"),
            ("/exit", "Exit the application"),
            ("/reset", "Reset the conversation history"),
            ("/toggle-cache", "Toggle response caching on/off"),
            ("/toggle-debug", "Toggle debug mode on/off"),
            ("/toggle-retrieval", "Toggle retrieval memory on/off"),
            ("/context", "Show current context window size"),
            ("/set-context <size>", "Set context window size"),
            ("/set-temp <value>", "Set temperature modifier (-0.5 to 0.5)"),
            ("/performance", "Show performance metrics"),
            ("/clear-metrics", "Clear performance metrics"),
            ("/status", "Show application status"),
            ("/summarize", "Summarize the conversation"),
            ("/load <file>", "Load data from JSONL file into retrieval store")
        ]
        
        self.ui.show_help(commands)
        return "Showing commands."
    
    @timed
    def clear_screen(self, args: List[str] = None) -> str:
        """Clear the terminal screen."""
        self.ui.clear_screen()
        self.ui.render_header()
        return "Screen cleared."
    
    @timed
    def exit_app(self, args: List[str] = None) -> str:
        """Exit the application."""
        # Save state before exiting
        self.state.save_state()
        # Save retrieval store
        self.retrieval_store.save_store()
        # Show exit message
        exit_msg = "Exiting SchultzGPT. Goodbye!"
        # Exit program
        exit(0)
        return exit_msg
    
    @timed
    def reset_conversation(self, args: List[str] = None) -> str:
        """Reset the conversation history."""
        self.message_manager.clear_history()
        return "Conversation history reset."
    
    @timed
    def toggle_cache(self, args: List[str] = None) -> str:
        """Toggle response caching on/off."""
        self.state.caching_enabled = not self.state.caching_enabled
        return f"Caching {'enabled' if self.state.caching_enabled else 'disabled'}."
    
    @timed
    def toggle_debug(self, args: List[str] = None) -> str:
        """Toggle debug mode on/off."""
        self.state.debug_mode = not self.state.debug_mode
        # Update performance tracker logging
        self.performance_tracker.enable_file_logging = self.state.debug_mode
        return f"Debug mode {'enabled' if self.state.debug_mode else 'disabled'}."
    
    @timed
    def toggle_retrieval_store(self, args: List[str] = None) -> str:
        """Toggle retrieval memory on/off."""
        self.state.retrieval_store_enabled = not self.state.retrieval_store_enabled
        return f"Retrieval memory {'enabled' if self.state.retrieval_store_enabled else 'disabled'}."
    
    @timed
    def show_context_size(self, args: List[str] = None) -> str:
        """Show the current context window size."""
        return f"Current context window size: {self.state.context_window_size} messages."
    
    @timed
    def set_context_size(self, args: List[str] = None) -> str:
        """Set the context window size."""
        if not args:
            return "Usage: /set-context <size>"
        
        try:
            size = int(args[0])
            if size < 1 or size > 50:
                return "Context window size must be between 1 and 50."
            
            self.state.context_window_size = size
            return f"Context window size set to {size}."
        except ValueError:
            return "Invalid context window size. Must be a number between 1 and 50."
    
    @timed
    def set_temperature(self, args: List[str] = None) -> str:
        """Set temperature modifier (-0.5 to 0.5)."""
        if not args:
            return "Usage: /set-temp <value>"
        
        try:
            value = float(args[0])
            if value < -0.5 or value > 0.5:
                return "Temperature modifier must be between -0.5 and 0.5."
            
            self.state.set_temperature_modifier(value)
            return f"Temperature modifier set to {value}."
        except ValueError:
            return "Invalid temperature value. Must be a number between -0.5 and 0.5."
    
    @timed
    def show_performance(self, args: List[str] = None) -> str:
        """Show performance metrics."""
        return self.performance_tracker.get_formatted_report()
    
    @timed
    def clear_metrics(self, args: List[str] = None) -> str:
        """Clear performance metrics."""
        self.performance_tracker.clear_metrics()
        return "Performance metrics cleared."
    
    @timed
    def show_status(self, args: List[str] = None) -> str:
        """Show application status."""
        status = (
            f"SchultzGPT Status:\n"
            f"  Caching: {'Enabled' if self.state.caching_enabled else 'Disabled'}\n"
            f"  Debug Mode: {'Enabled' if self.state.debug_mode else 'Disabled'}\n"
            f"  Retrieval Memory: {'Enabled' if self.state.retrieval_store_enabled else 'Disabled'}\n"
            f"  Context Window: {self.state.context_window_size} messages\n"
            f"  Temperature Modifier: {self.state.temperature_modifier}\n"
            f"  Retrieval Store: {'Connected' if self.message_manager.retrieval_store_available else 'Disconnected'}\n"
            f"  Assistant ID: {os.environ.get('OPENAI_ASSISTANT_ID', 'Not Set')}\n"
            f"  Model: {Config.FINE_TUNED_MODEL}\n"
            f"  Retrieval API: {'Enabled' if self.state.retrieval_store_enabled else 'Disabled'}\n"
        )
        return status
    
    @timed
    def summarize_conversation(self, args: List[str] = None) -> str:
        """Summarize the conversation."""
        summary = self.message_manager.summarize_conversation()
        if not summary:
            return "No conversation to summarize."
        
        self.ui.render_structured_message(f"Conversation Summary:\n{summary}", is_user=False)
        return "Conversation summarized."
    
    @timed
    def load_data(self, args: List[str] = None) -> str:
        """Load data from JSONL file into retrieval store."""
        if not args:
            return "Usage: /load <file>"
            
        file_path = args[0]
        if not os.path.exists(file_path):
            return f"File not found: {file_path}"
            
        success = self.retrieval_store.load_from_jsonl(file_path)
        if success:
            # Update availability
            self.message_manager.retrieval_store_available = self.retrieval_store.available
            return f"Data loaded from {file_path} into retrieval store."
        else:
            return f"Error loading data from {file_path}."
    
    @timed
    def handle_command(self, command: str) -> str:
        """
        Handle a command from the user.
        
        Args:
            command: The command to handle
            
        Returns:
            Response message
        """
        # Split command and arguments
        parts = command.strip().split()
        cmd = parts[0].lower()
        args = parts[1:] if len(parts) > 1 else None
        
        # Check if this is a recognized command
        if cmd in self.commands:
            return self.commands[cmd](args)
        else:
            return f"Unknown command: {cmd}. Type /help for available commands."
            
    @timed
    def handle_message(self, user_message: str) -> Dict[str, Any]:
        """
        Process a user message and get a response.
        
        Args:
            user_message: The user's message
            
        Returns:
            Response data
        """
        # Check if this is a command
        if user_message.startswith("/"):
            response = self.handle_command(user_message)
            self.ui.render_command_output(user_message, response)
            return {
                "response": response,
                "is_command": True
            }
            
        # Display user message
        self.ui.render_message(user_message, is_user=True)
        
        # Process asynchronously
        return asyncio.run(self._process_message_async(user_message))
        
    @async_timed
    async def _process_message_async(self, user_message: str) -> Dict[str, Any]:
        """
        Process a user message asynchronously.
        
        Args:
            user_message: The user's message
            
        Returns:
            Response data
        """
        try:
            # Add user message to conversation
            user_msg = self.message_manager.add_user_message(user_message)
            
            # Get context window size
            context_size = self.state.context_window_size
            
            # Get relevant context asynchronously
            context_items = await self.message_manager.get_context_async(
                query=user_message,
                count=max(3, context_size // 2)
            )
            
            # Build prompt with context
            context_instruction = "Use this context to inform your response if relevant."
            messages = self.message_manager.build_prompt_with_context(
                user_query=user_message,
                instruction=context_instruction,
                context_items=context_items
            )
            
            # Get model from config instead of environment variable
            model = Config.FINE_TUNED_MODEL
            
            # Determine temperature (base + modifier)
            temperature = 0.7 + self.state.temperature_modifier
            
            # Call OpenAI API asynchronously
            response = await async_chat_completion(
                messages=messages,
                model=model,
                temperature=temperature
            )
            
            # Process response
            assistant_message = response.choices[0].message.content
            
            # Add assistant response to conversation
            self.message_manager.add_assistant_message(
                content=assistant_message,
                metadata={"mood": self.state.mood, "model": model}
            )
            
            # Display the response
            self.ui.render_message(assistant_message, is_user=False)
            
            return {
                "response": assistant_message,
                "is_command": False,
                "mood": self.state.mood,
                "model": model
            }
            
        except Exception as e:
            # Get error details
            error_msg = str(e)
            if self.state.debug_mode:
                error_msg = f"{error_msg}\n\n{traceback.format_exc()}"
                
            self.ui.render_error(f"Error processing message asynchronously: {error_msg}")
            return {"response": f"Error: {error_msg}", "is_command": False, "error": True}
    
    def run(self):
        """Run the SchultzGPT application."""
        # Clear screen
        self.ui.clear_screen()
        
        # Show header
        self.ui.render_header()
        
        # Show startup message
        startup_message = "SchultzGPT initialized. Type a message to chat or /help for commands."
        self.ui.render_info(startup_message)
        
        # Main loop
        running = True
        while running:
            try:
                # Render footer with status
                self.ui.render_footer(self.state, self.message_manager)
                
                # Get user input
                user_message = self.ui.get_user_input()
                
                # Exit if requested
                if user_message.lower() in ["/exit", "/quit"]:
                    self.exit_app()
                    running = False
                    continue
                
                # Skip empty messages
                if not user_message.strip():
                    continue
                
                # Handle message
                response_data = self.handle_message(user_message)
                
                # Update metrics if not a command
                if not response_data.get("is_command", False):
                    # Record any token usage
                    tokens = response_data.get("tokens", 0)
                    if tokens > 0:
                        self.state.update_performance_metric("total_tokens_used", tokens)
                
            except KeyboardInterrupt:
                # Handle Ctrl+C gracefully
                self.ui.render_info("Press Ctrl+C again to exit, or type /exit")
                
            except Exception as e:
                # Handle unhandled exceptions
                error_msg = str(e)
                if self.state.debug_mode:
                    error_msg = f"{error_msg}\n\n{traceback.format_exc()}"
                    
                self.ui.render_error(f"Unhandled error: {error_msg}")
                
        # Final cleanup
        self.state.save_state() 