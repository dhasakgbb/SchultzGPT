# Data Loading

This directory contains scripts for loading generated data into the OpenAI Retrieval API.

## Scripts

- `load_jon_retrieval.py`: Loads Q&A pairs and statements into the OpenAI Retrieval API

## Usage

### Loading Data into Retrieval API

```bash
# Load data into Retrieval API
python -m data_generation.loading.load_jon_retrieval --file path/to/jon_data.jsonl

# Test data loading without actually uploading (dry run)
python -m data_generation.loading.load_jon_retrieval --file path/to/jon_data.jsonl --dry-run
```

### Options

- `--file`: Path to the JSONL file containing data to load
- `--dry-run`: Test data loading without actually uploading
- `--batch-size`: Number of items to upload in each batch (default: 100)
- `--max-concurrent`: Maximum number of concurrent uploads (default: 5)

For more details, see the main [Jon Data Generation Toolkit README](../README.md). 