# SchultzGPT

A terminal-based AI persona chatbot with vector store-backed memory.

## Features

- Terminal-based UI with rich text formatting
- Conversation memory using vector embeddings
- Asynchronous API processing for improved performance
- Performance tracking and metrics
- Conversation summarization for long-term context
- Smart context retrieval for more relevant responses
- Configurable via environment variables

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/SchultzGPT.git
cd SchultzGPT
```

2. Set up a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the dependencies:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file with your API keys (or copy from .env.example):
```
OPENAI_API_KEY=your_api_key_here
```

## Usage

Run the application using:

```bash
./run.py
```

Or install as a package:

```bash
pip install -e .
schultzgpt
```

## Commands

The application supports the following commands:

- `/help` - Show available commands
- `/clear` - Clear the terminal screen
- `/exit` - Exit the application
- `/reset` - Reset the conversation history
- `/toggle-cache` - Toggle response caching on/off
- `/toggle-debug` - Toggle debug mode on/off
- `/toggle-vector` - Toggle vector store on/off
- `/context` - Show current context window size
- `/set-context <size>` - Set context window size
- `/set-temp <value>` - Set temperature modifier (-0.5 to 0.5)
- `/performance` - Show performance metrics
- `/clear-metrics` - Clear performance metrics
- `/status` - Show application status
- `/summarize` - Summarize the conversation
- `/reindex` - Rebuild the vector store from conversation history

## Architecture

SchultzGPT is built with a clean, modular architecture:

- `src/` - Main source code directory
  - `models/` - Data models and state management
    - `message.py` - Message data structures
    - `state.py` - Application state and caching
  - `services/` - Core functionality
    - `message_manager.py` - Unified conversation and message handling
    - `openai.py` - OpenAI API integration
    - `vector_store.py` - Vector-based semantic search
    - `performance.py` - Performance tracking
  - `controllers/` - Application logic
    - `controller.py` - Main controller coordinating services
  - `ui/` - User interface components
    - `terminal.py` - Terminal-based UI
  - `config/` - Configuration settings
    - `config.py` - Application constants and settings
- `run.py` - Main entry point

## Performance Features

- Asynchronous API processing
- Response caching
- Performance tracking with detailed metrics
- Vector store for semantic search

## License

MIT 