"""
Message Manager service for SchultzGPT.
Manages conversation history, context retrieval, and conversation segmentation.
"""

import os
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Callable, Tuple
import json

from models.message import Message, MessageHistory
from models.state import SchultzState
from services.retrieval_store import RetrievalStore
from services.openai import chat_completion, async_chat_completion, run_async


class ConversationSegment:
    """Represents a segment of conversation history."""
    
    def __init__(self, start_time: datetime, end_time: Optional[datetime] = None):
        self.start_time = start_time
        self.end_time = end_time or start_time
        self.messages: List[Dict[str, Any]] = []
        self.summary: str = ""
        self.topics: List[str] = []
        
    def add_message(self, message: Dict[str, Any]) -> None:
        """Add a message to this segment."""
        self.messages.append(message)
        self.end_time = datetime.now()
        
    def set_summary(self, summary: str) -> None:
        """Set the summary for this segment."""
        self.summary = summary
        
    def set_topics(self, topics: List[str]) -> None:
        """Set the topics for this segment."""
        self.topics = topics
        
    @property
    def duration(self) -> timedelta:
        """Get the duration of this segment."""
        return self.end_time - self.start_time
        
    @property
    def message_count(self) -> int:
        """Get the number of messages in this segment."""
        return len(self.messages)
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert this segment to a dictionary."""
        return {
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat(),
            "message_count": self.message_count,
            "summary": self.summary,
            "topics": self.topics,
            "messages": self.messages
        }


class MessageManager:
    """Manager for conversation messages, context retrieval, and conversation segmentation."""
    
    def __init__(self, state: SchultzState):
        """Initialize the message manager.
        
        Args:
            state: Application state
        """
        self.state = state
        self.message_history = MessageHistory()
        self.current_segment: Optional[ConversationSegment] = None
        self.segments: List[ConversationSegment] = []
        
        # Initialize retrieval store if enabled
        self.retrieval_store = None
        if self.state.retrieval_store_enabled:
            try:
                self.retrieval_store = RetrievalStore()
            except Exception as e:
                print(f"Error initializing retrieval store: {e}")
    
    @property
    def retrieval_store_available(self) -> bool:
        """Check if retrieval store is available."""
        return self.retrieval_store is not None
    
    def add_user_message(self, 
                        content: str, 
                        metadata: Optional[Dict[str, Any]] = None) -> Message:
        """
        Add a user message to the conversation history.
        
        Args:
            content: Message content
            metadata: Optional metadata (mood, topics, etc.)
            
        Returns:
            The created message
        """
        # Add to message history
        message = self.message_history.add_user_message(content, metadata)
        
        # Add to retrieval store if available
        if self.retrieval_store_available and self.state.retrieval_store_enabled:
            retrieval_metadata = {
                "role": "user",
                "timestamp": datetime.now().isoformat(),
                "type": "message"
            }
            
            # Add additional metadata if provided
            if metadata:
                retrieval_metadata.update(metadata)
                
            self.retrieval_store.add_text(content, retrieval_metadata)
            
        # Add to current segment
        if self.current_segment:
            self.current_segment.add_message(message.to_dict())
            
        return message
    
    def add_assistant_message(self, 
                            content: str, 
                            metadata: Optional[Dict[str, Any]] = None) -> Message:
        """
        Add an assistant message to the conversation history.
        
        Args:
            content: Message content
            metadata: Optional metadata (mood, topics, etc.)
            
        Returns:
            The created message
        """
        # Add to message history
        message = self.message_history.add_assistant_message(content, metadata)
        
        # Add to retrieval store if available
        if self.retrieval_store_available and self.state.retrieval_store_enabled:
            retrieval_metadata = {
                "role": "assistant",
                "timestamp": datetime.now().isoformat(),
                "type": "message"
            }
            
            # Add additional metadata if provided
            if metadata:
                retrieval_metadata.update(metadata)
                
            self.retrieval_store.add_text(content, retrieval_metadata)
            
        # Add to current segment
        if self.current_segment:
            self.current_segment.add_message(message.to_dict())
            
        return message
    
    def get_relevant_context(self, query: str) -> str:
        """
        Get relevant context for a query from the retrieval store.
        
        Args:
            query: The query to get context for
            
        Returns:
            Retrieved context as a string
        """
        if not self.retrieval_store_available or not self.state.retrieval_store_enabled:
            return ""
            
        try:
            return self.retrieval_store.get_relevant_context(query)
        except Exception as e:
            print(f"Error getting context: {e}")
            return ""
    
    def start_new_segment(self) -> None:
        """Start a new conversation segment."""
        # Save current segment if it exists
        if self.current_segment and self.current_segment.message_count > 0:
            self.segments.append(self.current_segment)
            
        # Create new segment
        self.current_segment = ConversationSegment(datetime.now())
    
    def clear_history(self) -> None:
        """Clear conversation history."""
        # Clear message history
        self.message_history.clear()
        
        # Clear segments
        self.segments = []
        self.current_segment = None
        
        # Clear retrieval store if available
        if self.retrieval_store_available and self.state.retrieval_store_enabled:
            self.retrieval_store.clear_store()
    
    def summarize_conversation(self) -> str:
        """
        Generate a summary of the conversation.
        
        Returns:
            Conversation summary
        """
        # If no messages, return empty string
        if not self.message_history.messages or len(self.message_history.messages) < 3:
            return ""
            
        # Format messages for the summary prompt
        message_texts = []
        for msg in self.message_history.messages[1:]:  # Skip system message
            role = "User" if msg.role == "user" else "Jon"
            message_texts.append(f"{role}: {msg.content}")
            
        message_history = "\n".join(message_texts)
        
        # Create the summary prompt
        prompt = [
            {"role": "system", "content": "You are a helpful assistant who summarizes conversations."},
            {"role": "user", "content": f"Please summarize this conversation in a few sentences.\n\nConversation:\n{message_history}"}
        ]
        
        try:
            # Get the summary
            response = chat_completion(
                messages=prompt,
                model="gpt-4",
                temperature=0.7
            )
            
            summary = response.choices[0].message.content
            
            # Add to retrieval store with special metadata
            if self.retrieval_store_available and self.state.retrieval_store_enabled:
                metadata = {
                    "type": "conversation_summary",
                    "timestamp": datetime.now().isoformat(),
                    "message_count": len(self.message_history.messages) - 1  # Exclude system
                }
                self.retrieval_store.add_text(summary, metadata)
                
            return summary
        except Exception as e:
            print(f"Error summarizing conversation: {e}")
            return "Error generating summary."
    
    def get_conversation_topics(self) -> List[str]:
        """
        Extract topics from the conversation.
        
        Returns:
            List of conversation topics
        """
        # If no messages, return empty list
        if not self.message_history.messages or len(self.message_history.messages) < 3:
            return []
            
        # Format messages for the topic prompt
        message_texts = []
        for msg in self.message_history.messages[1:]:  # Skip system message
            role = "User" if msg.role == "user" else "Jon"
            message_texts.append(f"{role}: {msg.content}")
            
        message_history = "\n".join(message_texts)
        
        # Create the topics prompt
        prompt = [
            {"role": "system", "content": "You are a helpful assistant who identifies the main topics in conversations."},
            {"role": "user", "content": f"Please list the 3-5 main topics in this conversation. Return the topics as a JSON array of strings.\n\nConversation:\n{message_history}"}
        ]
        
        try:
            # Get the topics
            response = chat_completion(
                messages=prompt,
                model="gpt-4",
                temperature=0.7,
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response.choices[0].message.content)
            
            # Handle different response formats
            if isinstance(result, list):
                return result
            elif isinstance(result, dict) and "topics" in result:
                return result["topics"]
            else:
                return []
        except Exception as e:
            print(f"Error extracting topics: {e}")
            return [] 