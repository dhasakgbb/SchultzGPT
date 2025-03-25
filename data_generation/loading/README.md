# Jon Data Loading Utilities

This directory contains scripts for loading Jon data into various storage systems and managing OpenAI assistants:

## Scripts

- `load_jon_data.py`: Loads Q&A pairs into the legacy vector store
- `load_jon_retrieval.py`: Loads data into the OpenAI Retrieval API
- `manage_assistants.py`: Manages OpenAI Assistants (create, list, get details, clean)

## Usage Examples

### Loading Data into Legacy Vector Store

```bash
# Load data into vector store
python -m data_generation.loading.load_jon_data --file path/to/jon_vector_data.jsonl

# Validate data without loading
python -m data_generation.loading.load_jon_data --file path/to/jon_vector_data.jsonl --dry-run
```

### Loading Data into Retrieval API

```bash
# Load data into retrieval API
python -m data_generation.loading.load_jon_retrieval --file path/to/jon_retrieval_data.jsonl

# Use a specific Assistant ID
python -m data_generation.loading.load_jon_retrieval --file path/to/data.jsonl --assistant-id asst_abc123
```

### Managing Assistants

```bash
# List all assistants
python -m data_generation.loading.manage_assistants list

# Create a new assistant
python -m data_generation.loading.manage_assistants create --name "Jon Memory" --save

# Clean up old files from an assistant
python -m data_generation.loading.manage_assistants clean --id asst_abc123
```

For more details, see the main [Jon Data Generation Toolkit README](../README.md). 