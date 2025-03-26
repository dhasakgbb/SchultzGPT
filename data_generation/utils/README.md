# Data Generation Utilities

This directory contains utility scripts for working with generated data.

## Test Jon Data

The `test_jon_data.py` script helps validate generated data files.

### Usage

```bash
# Test a Q&A data file
python -m data_generation.utils.test_jon_data --file path/to/qa_data.jsonl --type qa

# Test a conversation data file
python -m data_generation.utils.test_jon_data --file path/to/conversation_data.jsonl --type conversation

# Test a statements data file
python -m data_generation.utils.test_jon_data --file path/to/statements_data.jsonl --type statements
```

### Options

- `--file`: Path to the data file to test
- `--type`: Type of data to test (qa, conversation, or statements)
- `--verbose`: Show detailed test results

## Usage Examples

### Testing Generated Data

```bash
# View sample Q&A pairs
python -m data_generation.utils.test_jon_data --file path/to/vector_data.jsonl --type qa

# View retrieval format data
python -m data_generation.utils.test_jon_data --file path/to/retrieval_data.jsonl --type retrieval

# View conversations
python -m data_generation.utils.test_jon_data --file path/to/raw_conversations.json --type conversation

# Test with your fine-tuned model interactively
python -m data_generation.utils.test_jon_data --file path/to/data.jsonl --interactive --model ft:gpt-3.5-turbo-0613:personal:jon-v1

# Test with retrieval-backed assistant interactively
python -m data_generation.utils.test_jon_data --file path/to/retrieval_data.jsonl --interactive --assistant-id asst_abc123
```

The test utility supports:
- Displaying samples from different data formats
- Interactive testing with fine-tuned models
- Interactive testing with OpenAI Assistants 
- Custom display limits

For more details, see the main [Jon Data Generation Toolkit README](../README.md). 