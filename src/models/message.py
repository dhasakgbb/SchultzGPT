"""
Message models for SchultzGPT.
"""

from typing import Dict, List, Optional, Any
from datetime import datetime

class Message:
    """Base message class for SchultzGPT conversations."""
    
    def __init__(self, content: str, role: str, timestamp: Optional[datetime] = None):
        self.content = content
        self.role = role  # "user", "assistant", or "system"
        self.timestamp = timestamp or datetime.now()
        self.metadata: Dict[str, Any] = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary format for API calls."""
        return {
            "role": self.role,
            "content": self.content
        }
    
    def to_retrieval_store_format(self) -> Dict[str, Any]:
        """Convert message to retrieval store format with metadata."""
        return {
            "text": self.content,
            "metadata": {
                "role": self.role,
                "timestamp": self.timestamp.isoformat(),
                **self.metadata
            }
        }
    
    # Alias for backward compatibility
    to_vector_store_format = to_retrieval_store_format
    
    def set_metadata(self, key: str, value: Any) -> None:
        """Set metadata for the message."""
        self.metadata[key] = value
    
    def __str__(self) -> str:
        """String representation of the message."""
        return f"{self.role.capitalize()}: {self.content}"


class MessageHistory:
    """Manages a collection of messages in a conversation."""
    
    def __init__(self, system_message: str = ""):
        self.messages: List[Message] = []
        if system_message:
            self.add_system_message(system_message)
    
    def add_system_message(self, content: str) -> None:
        """Add a system message to the conversation."""
        self.messages.append(Message(content, "system"))
    
    def add_user_message(self, content: str) -> Message:
        """Add a user message to the conversation."""
        msg = Message(content, "user")
        self.messages.append(msg)
        return msg
    
    def add_assistant_message(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> Message:
        """Add an assistant message to the conversation."""
        msg = Message(content, "assistant")
        if metadata:
            msg.metadata = metadata
        self.messages.append(msg)
        return msg
    
    def get_last_n_messages(self, n: int) -> List[Message]:
        """Get the last n messages from the conversation."""
        return self.messages[-n:] if n < len(self.messages) else self.messages[:]
    
    def clear(self) -> None:
        """Clear all messages except system messages."""
        self.messages = [msg for msg in self.messages if msg.role == "system"]
    
    def to_openai_format(self) -> List[Dict[str, str]]:
        """Convert all messages to OpenAI API format."""
        return [msg.to_dict() for msg in self.messages]
    
    def to_retrieval_store_format(self) -> List[Dict[str, Any]]:
        """Convert all messages to retrieval store format."""
        return [msg.to_retrieval_store_format() for msg in self.messages 
                if msg.role != "system"]  # Typically don't store system messages
    
    # Alias for backward compatibility
    to_vector_store_format = to_retrieval_store_format 