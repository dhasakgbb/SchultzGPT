# Jon Fine-Tuning Utilities

This directory contains scripts for preparing data for fine-tuning and managing fine-tuning jobs with OpenAI:

## Scripts

- `prepare_fine_tuning.py`: Validates and prepares data for fine-tuning
- `monitor_jobs.py`: Lists, monitors, and manages fine-tuning jobs

## Usage Examples

### Preparing Data for Fine-Tuning

```bash
# Prepare data for fine-tuning
python -m data_generation.training.prepare_fine_tuning --file path/to/jon_fine_tuning.jsonl

# Specify output file and test split
python -m data_generation.training.prepare_fine_tuning --file path/to/data.jsonl --output prepared_data.jsonl --test-split 0.2

# Start fine-tuning job directly
python -m data_generation.training.prepare_fine_tuning --file path/to/data.jsonl --fine-tune --model gpt-3.5-turbo
```

### Managing Fine-Tuning Jobs

```bash
# List all fine-tuning jobs
python -m data_generation.training.monitor_jobs list

# Check status of a specific job
python -m data_generation.training.monitor_jobs status ft-abc123

# Register a completed model in your .env file
python -m data_generation.training.monitor_jobs status ft-abc123 --register

# Cancel a running job
python -m data_generation.training.monitor_jobs cancel ft-abc123
```

For more details, see the main [Jon Data Generation Toolkit README](../README.md). 