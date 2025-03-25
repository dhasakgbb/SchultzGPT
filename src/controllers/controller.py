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

from models.state import SchultzState
from models.message import Message, MessageHistory
from services.message_manager import MessageManager
from services.vector_store import VectorStore
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
        
        # Initialize vector store
        self.vector_store = VectorStore(
            store_dir=os.environ.get("VECTOR_STORE_PATH", ".vector_store"),
            embedding_model=os.environ.get("EMBEDDING_MODEL", "text-embedding-ada-002")
        )
        
        # Initialize performance tracker
        self.performance_tracker = PerformanceTracker(
            enable_file_logging=self.state.debug_mode
        )
        
        # Initialize message manager
        self.message_manager = MessageManager(
            state=self.state,
            system_message=Config.SYSTEM_PROMPT,
            vector_store=self.vector_store
        )
        
        # Initialize UI
        self.ui = TerminalUI()
        
        # Register commands
        self.commands = {
            "/help": self.show_help,
            "/clear": self.clear_screen,
            "/exit": self.exit_app,
            "/reset": self.reset_conversation,
            "/toggle-cache": self.toggle_cache,
            "/toggle-debug": self.toggle_debug,
            "/toggle-vector": self.toggle_vector_store,
            "/context": self.show_context_window,
            "/set-context": self.set_context_window,
            "/set-temp": self.set_temperature,
            "/performance": self.show_performance,
            "/clear-metrics": self.clear_metrics,
            "/status": self.show_status,
            "/summarize": self.summarize_conversation,
            "/reindex": self.reindex_vector_store
        }
        
        # Set initialized flag
        self.initialized = True
    
    @timed
    def show_help(self, args: List[str] = None) -> str:
        """Show help information."""
        help_text = "Available commands:\n"
        for cmd, func in self.commands.items():
            help_text += f"{cmd}: {func.__doc__}\n"
        return help_text
    
    @timed
    def clear_screen(self, args: List[str] = None) -> str:
        """Clear the terminal screen."""
        self.ui.clear_screen()
        return "Screen cleared."
    
    @timed
    def exit_app(self, args: List[str] = None) -> str:
        """Exit the application."""
        # Save state before exiting
        self.state.save_state()
        # Save vector store
        self.vector_store.save_store()
        # Show exit message
        exit_msg = "Exiting SchultzGPT. Goodbye!"
        # Exit program
        exit(0)
        return exit_msg
    
    @timed
    def reset_conversation(self, args: List[str] = None) -> str:
        """Reset the conversation history."""
        self.message_manager.clear_history()
        return "Conversation history has been reset."
    
    @timed
    def toggle_cache(self, args: List[str] = None) -> str:
        """Toggle response caching on/off."""
        new_state = self.state.toggle_caching()
        return f"Caching is now {'enabled' if new_state else 'disabled'}."
    
    @timed
    def toggle_debug(self, args: List[str] = None) -> str:
        """Toggle debug mode on/off."""
        new_state = self.state.toggle_debug()
        # Update performance tracker logging
        self.performance_tracker.enable_file_logging = new_state
        return f"Debug mode is now {'enabled' if new_state else 'disabled'}."
    
    @timed
    def toggle_vector_store(self, args: List[str] = None) -> str:
        """Toggle vector store on/off."""
        new_state = self.state.toggle_vector_store()
        return f"Vector store is now {'enabled' if new_state else 'disabled'}."
    
    @timed
    def show_context_window(self, args: List[str] = None) -> str:
        """Show current context window size."""
        return f"Current context window size: {self.state.context_window_size} messages."
    
    @timed
    def set_context_window(self, args: List[str] = None) -> str:
        """Set context window size (number of messages)."""
        if not args or not args[0].isdigit():
            return "Usage: /set-context <number>"
        
        size = int(args[0])
        if size < 1:
            return "Context window size must be at least 1."
        
        self.state.set_context_window(size)
        return f"Context window size set to {size} messages."
    
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
            f"  Vector Store: {'Enabled' if self.state.vector_store_enabled else 'Disabled'}\n"
            f"  Context Window: {self.state.context_window_size} messages\n"
            f"  Temperature Modifier: {self.state.temperature_modifier}\n"
            f"  Vector Store Entries: {self.vector_store.get_stats().get('entry_count', 0)}\n"
        )
        return status
    
    @timed
    def summarize_conversation(self, args: List[str] = None) -> str:
        """Summarize the conversation."""
        summary = self.message_manager.summarize_conversation()
        return f"Conversation Summary:\n\n{summary}"
    
    @timed
    def reindex_vector_store(self, args: List[str] = None) -> str:
        """Rebuild the vector store from conversation history."""
        # Track operation time
        self.performance_tracker.start_timer("reindex_vector_store")
        
        # Reindex
        success = self.message_manager.reindex_vector_store()
        
        # Stop timer
        self.performance_tracker.stop_timer("reindex_vector_store", success=success)
        
        if success:
            return "Vector store successfully reindexed."
        else:
            return "Error reindexing vector store."
    
    @timed
    def handle_command(self, command: str) -> str:
        """
        Handle a command.
        
        Args:
            command: The command to handle
            
        Returns:
            Command output
        """
        # Split command and arguments
        parts = command.strip().split()
        cmd = parts[0].lower()
        args = parts[1:] if len(parts) > 1 else []
        
        # Execute command if registered
        if cmd in self.commands:
            return self.commands[cmd](args)
        else:
            return f"Unknown command: {cmd}. Type /help for available commands."
    
    @timed
    def handle_message(self, user_message: str) -> Dict[str, Any]:
        """
        Handle a user message.
        
        Args:
            user_message: Message from the user
            
        Returns:
            Response data
        """
        # Check if this is a command
        if user_message.startswith("/"):
            # Handle command
            response = self.handle_command(user_message)
            return {"response": response, "is_command": True}
            
        try:
            # Start timer for overall message handling
            self.performance_tracker.start_timer("handle_message")
            
            # Add user message to conversation
            user_msg = self.message_manager.add_user_message(user_message)
            
            # Get context window size
            context_size = self.state.context_window_size
            
            # Get relevant context
            context_items = self.message_manager.get_context(
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
            
            # Get base model from environment or use default
            model = os.environ.get("OPENAI_MODEL", "gpt-3.5-turbo")
            
            # Determine temperature (base + modifier)
            temperature = 0.7 + self.state.temperature_modifier
            
            # Use structured response with UI component
            response_data = self.ui.stream_structured_response(
                cached_chat_completion=chat_completion,
                messages=messages,
                temperature_modifier=self.state.temperature_modifier,
                mood=self.state.mood,
                model=model
            )
            
            # Add assistant response to conversation
            if response_data:
                self.message_manager.add_assistant_message(
                    content=response_data.get("response", ""),
                    metadata=response_data
                )
            
            # Stop timer
            self.performance_tracker.stop_timer("handle_message", success=True)
            
            return {"response": response_data.get("response", ""), "is_command": False, **response_data}
            
        except Exception as e:
            # Record error
            self.performance_tracker.stop_timer("handle_message", success=False)
            
            # Get error details
            error_msg = str(e)
            if self.state.debug_mode:
                error_msg = f"{error_msg}\n\n{traceback.format_exc()}"
                
            self.ui.render_error(f"Error processing message: {error_msg}")
            return {"response": f"Error: {error_msg}", "is_command": False, "error": True}
    
    @async_timed
    async def handle_message_async(self, user_message: str) -> Dict[str, Any]:
        """
        Handle a user message asynchronously.
        
        Args:
            user_message: Message from the user
            
        Returns:
            Response data
        """
        # Check if this is a command
        if user_message.startswith("/"):
            # Handle command
            response = self.handle_command(user_message)
            return {"response": response, "is_command": True}
            
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
            
            # Get base model from environment or use default
            model = os.environ.get("OPENAI_MODEL", "gpt-3.5-turbo")
            
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
                metadata={"mood": self.state.mood}
            )
            
            return {
                "response": assistant_message,
                "is_command": False,
                "mood": self.state.mood
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