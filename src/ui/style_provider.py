"""
Style provider for SchultzGPT UI components.
"""

from typing import Dict, Any, Optional


class StyleProvider:
    """Provides consistent styling for terminal UI components."""
    
    def __init__(self):
        # Mood style mapping - maps moods to their display style
        self.mood_styles = {
            "happy": "green",
            "sad": "cyan",
            "angry": "red",
            "confused": "yellow",
            "excited": "bright_green",
            "thoughtful": "blue",
            "neutral": "white",
            "curious": "purple",
            "anxious": "bright_magenta",
            "bored": "dim white",
            "suspicious": "orange3",
            "mischievous": "magenta",
            "surprised": "bright_blue",
            "tired": "dim cyan",
            "amused": "bright_yellow",
            "concerned": "dim yellow",
            "calm": "sky_blue2",
            "frustrated": "red3",
            "numb": "dim white",
            "reflective": "bright_cyan",
            "sarcastic": "magenta",
            "cynical": "grey69",
            "supportive": "green",
            "irritated": "orange_red1",
            "relaxed": "sky_blue1",
            "energetic": "spring_green3",
            "philosophical": "deep_sky_blue1",
            "spiral": "bright_magenta",
        }
        
        # Default styles
        self.default_mood_style = "white"
        self.default_user_style = "bright_blue"
        self.default_assistant_style = "bright_green"
        self.default_system_style = "yellow"
        
    def mood_style(self, mood: str) -> str:
        """Get the style for a particular mood."""
        return self.mood_styles.get(mood.lower(), self.default_mood_style)
    
    def status_style(self, is_connected: bool) -> str:
        """Get style for connection status."""
        return "green" if is_connected else "red"
    
    def model_style(self, model_name: str) -> str:
        """Get style for model name based on model type."""
        model_lower = model_name.lower()
        
        if "gpt-4" in model_lower:
            return "bright_magenta"
        elif "gpt-3.5" in model_lower:
            return "bright_cyan"
        elif "ft:" in model_lower:  # Fine-tuned model
            return "bright_green"
        else:
            return "bright_white"
    
    def user_message_style(self) -> str:
        """Style for user messages."""
        return self.default_user_style
    
    def assistant_message_style(self, mood: Optional[str] = None) -> str:
        """Style for assistant messages, optionally affected by mood."""
        if mood and mood.lower() in self.mood_styles:
            return self.mood_styles[mood.lower()]
        return self.default_assistant_style
    
    def system_message_style(self) -> str:
        """Style for system messages."""
        return self.default_system_style
    
    def error_style(self) -> str:
        """Style for error messages."""
        return "bold red"
    
    def info_style(self) -> str:
        """Style for informational messages."""
        return "yellow"
    
    def success_style(self) -> str:
        """Style for success messages."""
        return "green"
    
    def warning_style(self) -> str:
        """Style for warning messages."""
        return "orange3"
    
    def command_style(self) -> str:
        """Style for command text."""
        return "bold cyan" 