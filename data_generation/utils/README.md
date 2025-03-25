# Jon Data Utilities

This directory contains utility scripts for testing and exploring Jon data:

## Scripts

- `test_jon_data.py`: Displays and interactively tests generated Jon data

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