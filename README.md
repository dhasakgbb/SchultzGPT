# SchultzGPT

A terminal-based AI persona chatbot with OpenAI Retrieval API-backed memory.

## Features

- Terminal-based UI with rich text formatting
- Conversation memory using OpenAI's Retrieval API
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
- `/toggle-retrieval` - Toggle retrieval memory on/off
- `/context` - Show current context window size
- `/set-context <size>` - Set context window size
- `/set-temp <value>` - Set temperature modifier (-0.5 to 0.5)
- `/performance` - Show performance metrics
- `/clear-metrics` - Clear performance metrics
- `/status` - Show application status
- `/summarize` - Summarize the conversation
- `/load <file>` - Load data from JSONL file into retrieval store

## Architecture

SchultzGPT is built with a clean, modular architecture:

- `src/` - Main source code directory
  - `models/` - Data models and state management
    - `message.py` - Message data structures
    - `state.py` - Application state and caching
  - `services/` - Core functionality
    - `message_manager.py` - Unified conversation and message handling
    - `openai.py` - OpenAI API integration
    - `retrieval_store.py` - OpenAI Retrieval API integration
    - `performance.py` - Performance tracking
  - `controllers/` - Application logic
    - `controller.py` - Main controller coordinating services
  - `ui/` - User interface components
    - `terminal.py` - Terminal-based UI
  - `config/` - Configuration settings
    - `config.py` - Application constants and settings
- `data_generation/` - Data generation and management toolkit
  - `jon_tools.py` - Unified CLI interface for all data tools
  - `generation/` - Data generation scripts
    - `jon_data_generator.py` - Synthetic Jon data generator
  - `loading/` - Data loading utilities
    - `load_jon_retrieval.py` - Retrieval API loader
    - `manage_assistants.py` - OpenAI Assistant management
  - `training/` - Fine-tuning utilities
    - `prepare_fine_tuning.py` - Prepares data for fine-tuning
    - `monitor_jobs.py` - Manages fine-tuning jobs
  - `utils/` - Shared utilities
    - `test_jon_data.py` - Data testing and exploration
  - `visualization/` - Memory visualization tools
    - `visualize_memory.py` - Jon's memory visualizer
- `run.py` - Main entry point

## Enhanced UI

SchultzGPT features an enhanced terminal UI with:

- **Expressive Footer Display**: Shows Jon's current emotional state with matching emoji
- **Memory Status Indicator**: Displays whether retrieval memory is connected
- **Model Information**: Shows which model is currently active
- **Command Help**: Reminds users that `/help` is available for command options

The status line format: `😊 jon: happy | 🧠 memory: connected | 📝 model: gpt-4o | /help for commands`

## Performance Features

- Asynchronous API processing
- Response caching
- Performance tracking with detailed metrics
- OpenAI Retrieval API for semantic memory

## OpenAI Retrieval API Integration

SchultzGPT uses OpenAI's Retrieval API to create a persistent semantic memory system:

- **Assistant-backed Memory**: Uses the Assistants API with retrieval capability
- **File Storage**: Conversation history, messages, and summaries are stored as files in OpenAI's system
- **Semantic Search**: Dynamically finds relevant past conversations based on semantic similarity
- **Metadata Enrichment**: Stores rich metadata with each memory for better retrieval

When you first run SchultzGPT, it creates an Assistant automatically. The Assistant ID is saved for future sessions.
You can also specify your own Assistant ID in the `.env` file.

## SchultzGPT Data Generation

The `data_generation` directory contains a comprehensive toolkit for generating synthetic data to train and enhance Jon's knowledge. It has been reorganized into a more modular structure:

- `jon_tools.py`: A unified CLI interface to all data generation and management functions
- `generation/`: Scripts for generating Jon data (Q&A pairs, conversations, statements)
- `loading/`: Tools for loading data into the Retrieval API and managing assistants
- `training/`: Scripts for preparing data for fine-tuning and managing fine-tuning jobs
- `utils/`: Utilities for testing and exploring Jon data
- `visualization/`: Tools for visualizing Jon's memory structure

For detailed usage instructions, see the [Jon Data Generation Toolkit README](data_generation/README.md).

## License

MIT 