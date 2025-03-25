# SchultzGPT

A personalized AI chatbot trained on conversation data to emulate natural dialogue. Built with Python and OpenAI's GPT-3.5 Turbo.

## Features

- 💬 Natural conversation flow with context awareness
- 🔍 Semantic search for relevant responses
- ⚡ Real-time streaming responses
- 👍 User feedback system
- 💾 Response saving functionality
- 🎨 Clean, terminal-based interface
- 🤖 Character-specific personality emulation
- 📊 Synthetic data generation for training
- 🎯 Authentic texting style simulation

## Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/SchultzGPT.git
cd SchultzGPT
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file with your OpenAI API key:
```env
OPENAI_API_KEY=your_api_key_here
CURRENT_MODEL_ID=ft:gpt-3.5-turbo-0125:personal::BEqahUc4
```

4. Run the application:
```bash
python schultz.py
```

## Project Structure

- `schultz.py` - Main application file
- `requirements.txt` - Python dependencies
- `jon_embed.json` - Embeddings data for semantic search
- `feedback.jsonl` - User feedback storage
- `synthetic_prompts.jsonl` - Generated conversation starters
- `generate_jon_synthetic.py` - Synthetic data generation script
- `.env` - Environment variables

## Technical Details

- Terminal-based interface with rich text formatting
- Uses OpenAI's GPT-3.5 Turbo fine-tuned model
- Implements semantic search using text embeddings
- Token-aware context management
- Real-time response streaming
- Persistent feedback storage
- Character-specific style emulation
- Synthetic data generation with authentic texting patterns

## Usage

1. Start the application using `python schultz.py`
2. Type your message and press Enter
3. Receive AI-generated responses
4. Provide feedback using 👍/👎 buttons
5. Save important responses using the ★ button

## Development

### Synthetic Data Generation

The project includes tools for generating synthetic conversation data that maintains authentic texting patterns:
- Realistic typos and common texting mistakes
- Natural filler words and expressions
- Character-specific vocabulary and style
- Contextually appropriate responses

### Character Emulation

The chatbot is designed to emulate specific character traits:
- Authentic texting style
- Consistent personality traits
- Natural conversation flow
- Context-aware responses

## Contributing

Feel free to submit issues and enhancement requests! 

## License

MIT License - feel free to use this project for your own purposes. 