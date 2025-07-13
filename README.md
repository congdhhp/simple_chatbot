# Simple CLI Chatbot 🤖

A flexible command-line chatbot powered by Hugging Face Transformers with CUDA support. Easily switch between different LLM models and manage conversations with a rich CLI interface.

## Features

- 🚀 **Multiple Model Support**: Easy switching between different LLM models
- ⚡ **CUDA Acceleration**: Optimized for GPU inference with CUDA 11.8
- 🔧 **Flexible Configuration**: YAML-based configuration for easy model management
- 💬 **Conversation Management**: Save, load, and manage conversation history
- 🎨 **Rich CLI Interface**: Beautiful command-line interface with syntax highlighting
- 📦 **Virtual Environment**: Isolated Python environment for clean dependency management
- 🔄 **Memory Optimization**: Automatic memory management and quantization support

## Quick Start

### 1. Setup Virtual Environment

```bash
# Create and activate virtual environment
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

### 2. Install Dependencies

```bash
# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
pip install -r requirements.txt
```

### 3. Run the Chatbot

```bash
# Start with default model (Llama 3.2 3B)
python chatbot.py

# Start with specific model
python chatbot.py -m llama-3.2-1b-instruct

# Use custom configuration
python chatbot.py -c config/custom.yaml
```

## Supported Models

The chatbot comes pre-configured with several models:

| Model | Size | Description |
|-------|------|-------------|
| `llama-3.2-3b-instruct` | 3B | Meta's Llama 3.2 3B (default) |
| `llama-3.2-1b-instruct` | 1B | Lighter version for lower memory |
| `mistral-7b-instruct` | 7B | Mistral's instruction-tuned model |

## CLI Commands

Once the chatbot is running, you can use these commands:

- `/help` - Show help message
- `/models` - List available models
- `/switch` - Switch to a different model
- `/info` - Show current model information
- `/config` - View/modify model configuration
- `/clear` - Clear conversation history
- `/save` - Save current conversation
- `/load` - Load a saved conversation
- `/list` - List all saved conversations
- `/quit` - Exit the chatbot

## Configuration

### Model Configuration

Models are configured in `config/models.yaml`. Each model has its own configuration and settings:

```yaml
models:
  model-name:
    model_id: "huggingface/model-id"
    display_name: "Human Readable Name"
    description: "Model description"
    device: "cuda"
    torch_dtype: "float16"
    generation_config:
      max_new_tokens: 512
      temperature: 0.7
      top_p: 0.9
      top_k: 50
      do_sample: true
      repetition_penalty: 1.1
    system_prompt: |
      Your system prompt here
    settings:
      max_conversation_history: 10
      save_conversations: true
      conversation_dir: "conversations"
      use_hf_cache: true  # Use Hugging Face default cache (shared across projects)
      log_level: "INFO"
```

### Per-Model Settings

Each model can have its own customized settings:

- **max_conversation_history**: Number of previous messages to keep in context
- **save_conversations**: Whether to save conversation history to files
- **conversation_dir**: Directory to save conversations
- **use_hf_cache**: Whether to use shared HuggingFace cache or local cache
- **log_level**: Logging level for this model

## Memory Requirements

| Model Size | Minimum GPU Memory | Recommended |
|------------|-------------------|-------------|
| 1B | 2GB | 4GB |
| 3B | 4GB | 6GB |
| 7B | 8GB | 12GB |

The chatbot automatically applies 4-bit quantization for GPUs with less than 8GB memory.

## Model Caching

The chatbot uses Hugging Face's default cache system (`~/.cache/huggingface/`) which provides several benefits:

- **🔄 Shared Cache**: Models downloaded once can be used by all HF-based projects
- **💾 Space Efficient**: No duplicate model downloads across different projects
- **⚡ Faster Setup**: If you already have models cached from other projects, they'll be instantly available
- **🛠️ Standard Location**: Follows HF conventions and integrates with HF ecosystem tools

To check your cache location and size:
```bash
# View cache info
huggingface-cli scan-cache

# Clean cache if needed
huggingface-cli delete-cache
```

## Project Structure

```
simple_chatbot/
├── chatbot.py              # Main entry point
├── requirements.txt        # Python dependencies
├── config/
│   └── models.yaml        # Model configurations
├── src/
│   ├── __init__.py
│   ├── cli.py             # CLI interface
│   ├── config_manager.py  # Configuration management
│   ├── model_manager.py   # Model loading and inference
│   └── conversation_manager.py  # Conversation handling
├── conversations/         # Saved conversations (auto-created)
└── venv/                 # Virtual environment

# Models cached in: ~/.cache/huggingface/ (shared across projects)
```

## Usage Examples

### Basic Chat

```bash
$ python chatbot.py
🤖 Simple CLI Chatbot
Powered by Hugging Face Transformers

Loading default model...
✓ Model loaded successfully!

You: Hello! How are you?

🤖 Assistant
Hello! I'm doing well, thank you for asking. I'm here and ready to help you with any questions or tasks you might have. How are you doing today?
```

### Switching Models

```bash
You: /switch

Available models:
  1. Llama 3.2 3B Instruct (llama-3.2-3b-instruct)
  2. Llama 3.2 1B Instruct (llama-3.2-1b-instruct)
  3. Mistral 7B Instruct (mistral-7b-instruct)

Enter model number or name: 2

Loading model: Llama 3.2 1B Instruct...
✓ Model loaded successfully!
```

### Saving Conversations

```bash
You: /save
Enter filename [conversation.json]: my_chat_2025
✓ Conversation saved to my_chat_2025.json
```

## Adding New Models

To add a new model, edit `config/models.yaml`:

```yaml
models:
  your-new-model:
    model_id: "organization/model-name"
    display_name: "Your Model Name"
    description: "Description of your model"
    device: "cuda"
    torch_dtype: "float16"
    generation_config:
      max_new_tokens: 512
      temperature: 0.7
      # ... other parameters
    system_prompt: |
      Your custom system prompt
    settings:
      max_conversation_history: 10
      save_conversations: true
      conversation_dir: "conversations"
      use_hf_cache: true
      log_level: "INFO"
```

## Troubleshooting

### CUDA Issues

If you encounter CUDA-related errors:

1. Verify CUDA installation: `nvidia-smi`
2. Check PyTorch CUDA support: `python -c "import torch; print(torch.cuda.is_available())"`
3. Reinstall PyTorch with correct CUDA version

### Memory Issues

For out-of-memory errors:

1. Use smaller models (1B instead of 3B)
2. Enable quantization (automatic for <8GB GPU)
3. Reduce `max_new_tokens` in model config

### Model Loading Issues

If models fail to load:

1. Check internet connection (first download)
2. Verify Hugging Face model ID
3. Check available disk space
4. Review logs in `chatbot.log`

## Requirements

- Python 3.9+
- CUDA 11.8 (for GPU acceleration)
- 4GB+ GPU memory (recommended)
- 10GB+ free disk space (for model cache)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## Acknowledgments

- [Hugging Face Transformers](https://huggingface.co/transformers/) for the model infrastructure
- [Rich](https://rich.readthedocs.io/) for the beautiful CLI interface
- [Click](https://click.palletsprojects.com/) for command-line interface framework