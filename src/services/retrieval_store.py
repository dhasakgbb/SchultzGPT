"""
Retrieval API service for SchultzGPT.
Uses OpenAI's Retrieval API for storing and retrieving conversation context.
"""

import os
import json
import time
from typing import Dict, List, Any, Optional, Union
from datetime import datetime

from openai import OpenAI
from config.config import Config

class RetrievalStore:
    """Interface to OpenAI's Retrieval API for storing and retrieving conversation context"""
    
    def __init__(
        self,
        assistant_id: Optional[str] = None,
        model: str = "gpt-4-turbo-preview",
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        client: Optional[OpenAI] = None
    ):
        """Initialize the RetrievalStore
        
        Args:
            assistant_id: Optional ID of an existing assistant to use
            model: Model to use for chat completions
            temperature: Response variability
            max_tokens: Maximum tokens in response
            client: Optional OpenAI client
        """
        self.client = client or OpenAI()
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.assistant_id = assistant_id or os.environ.get("OPENAI_ASSISTANT_ID")
        self.file_ids = []
        
        # Initialize assistant if needed
        if not self.assistant_id:
            self._create_assistant()
        else:
            # Verify assistant exists
            try:
                self.assistant = self.client.beta.assistants.retrieve(self.assistant_id)
                self.file_ids = self.assistant.file_ids
            except Exception as e:
                print(f"Error retrieving assistant: {e}")
                self._create_assistant()
    
    def _create_assistant(self):
        """Create a new assistant with the Retrieval API"""
        try:
            self.assistant = self.client.beta.assistants.create(
                name="Jon",
                description="A millennial who works at a retirement community, dealing with life's challenges with humor and authenticity.",
                model=self.model,
                tools=[{"type": "retrieval"}],
                file_ids=self.file_ids
            )
            self.assistant_id = self.assistant.id
            print(f"Created new assistant with ID: {self.assistant_id}")
        except Exception as e:
            raise RuntimeError(f"Failed to create assistant: {e}")
    
    def add_file(self, file_path: str) -> bool:
        """Add a file to the assistant
        
        Args:
            file_path: Path to the file to add
            
        Returns:
            Success status
        """
        try:
            # Upload the file to OpenAI
            with open(file_path, "rb") as f:
                file = self.client.files.create(
                    file=f,
                    purpose="assistants"
                )
            
            # Add the file to the assistant
            self.assistant = self.client.beta.assistants.update(
                self.assistant_id,
                file_ids=[*self.file_ids, file.id]
            )
            self.file_ids.append(file.id)
            return True
        except Exception as e:
            print(f"Failed to add file: {e}")
            return False
    
    def remove_files(self, file_ids: List[str]):
        """Remove files from the assistant
        
        Args:
            file_ids: List of file IDs to remove
        """
        try:
            remaining_files = [f for f in self.file_ids if f not in file_ids]
            self.assistant = self.client.beta.assistants.update(
                self.assistant_id,
                file_ids=remaining_files
            )
            self.file_ids = remaining_files
        except Exception as e:
            raise RuntimeError(f"Failed to remove files: {e}")
    
    def get_relevant_context(self, query: str, thread_id: Optional[str] = None) -> str:
        """Get relevant context for a query using the Retrieval API
        
        Args:
            query: The query to get context for
            thread_id: Optional thread ID for conversation context
            
        Returns:
            Retrieved context as a string
        """
        try:
            # Create a thread if not provided
            if not thread_id:
                thread = self.client.beta.threads.create()
                thread_id = thread.id
            
            # Add the message to the thread
            message = self.client.beta.threads.messages.create(
                thread_id=thread_id,
                role="user",
                content=query
            )
            
            # Run the assistant to get context
            run = self.client.beta.threads.runs.create(
                thread_id=thread_id,
                assistant_id=self.assistant_id
            )
            
            # Wait for completion
            while run.status in ["queued", "in_progress"]:
                run = self.client.beta.threads.runs.retrieve(
                    thread_id=thread_id,
                    run_id=run.id
                )
                time.sleep(0.5)
            
            if run.status == "completed":
                # Get the assistant's response
                messages = self.client.beta.threads.messages.list(thread_id=thread_id)
                context = []
                for msg in messages.data:
                    if msg.role == "assistant":
                        for content in msg.content:
                            if hasattr(content, 'text'):
                                context.append(content.text.value)
                return "\n".join(context)
            else:
                raise RuntimeError(f"Run failed with status: {run.status}")
                
        except Exception as e:
            raise RuntimeError(f"Failed to get context: {e}")
    
    def get_assistant_info(self) -> Dict[str, Any]:
        """Get information about the current assistant
        
        Returns:
            Dictionary with assistant information
        """
        try:
            return {
                "id": self.assistant_id,
                "model": self.model,
                "file_ids": self.file_ids,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens
            }
        except Exception as e:
            raise RuntimeError(f"Failed to get assistant info: {e}")
    
    def save_store(self) -> bool:
        """
        Save the current state of the retrieval store.
        
        For the Retrieval API implementation, this saves the assistant ID
        and file IDs to a local JSON file for reference and backup.
        
        Returns:
            Success status
        """
        try:
            # Create a state object with assistant and file information
            state = {
                "assistant_id": self.assistant_id,
                "file_ids": self.file_ids,
                "model": self.model,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "last_updated": datetime.now().isoformat()
            }
            
            # Ensure the directory exists
            os.makedirs(".retrieval_store", exist_ok=True)
            
            # Save to a JSON file
            with open(".retrieval_store/state.json", "w") as f:
                json.dump(state, f, indent=2)
                
            return True
        except Exception as e:
            print(f"Failed to save store state: {e}")
            return False
    
    def clear_store(self) -> bool:
        """Clear all files from the retrieval store."""
        try:
            # Delete each file
            for file_id in self.file_ids:
                try:
                    self.client.files.delete(file_id)
                except Exception as e:
                    print(f"Error deleting file {file_id}: {e}")
            
            # Clear our file list
            self.file_ids = []
            
            # Update the assistant
            if self.assistant_id:
                self.client.beta.assistants.update(
                    assistant_id=self.assistant_id,
                    file_ids=[]
                )
            
            return True
        except Exception as e:
            print(f"Error clearing retrieval store: {e}")
            return False 