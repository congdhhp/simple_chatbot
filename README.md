# LLM Chatbot with GPU Support

A flexible CLI chatbot that supports different LLM models with GPU acceleration using CUDA.

## Features

- 🚀 **GPU Acceleration**: Automatic CUDA detection and GPU utilization
- 🔄 **Universal Model Support**: Works with ANY HuggingFace model or local model
- 🎯 **Easy Model Switching**: Change models instantly with simple commands
- ⚙️ **Configurable**: JSON-based configuration system
- 💬 **Interactive CLI**: User-friendly command-line interface
- 📝 **Conversation History**: Track and review chat history
- 🎛️ **Generation Parameters**: Customizable text generation settings
- 🔧 **Extensible Architecture**: Add any model type with minimal configuration

## Quick Start

1. **Setup Virtual Environment**:
   ```bash
   python -m venv venv
   ./venv/Scripts/activate  # Windows
   # source venv/bin/activate  # Linux/Mac
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Chatbot**:
   ```bash
   python chatbot.py
   ```

## Usage

### Basic Commands

- `help` - Show available commands
- `info` - Display current model information
- `models` - List available models
- `switch <model>` - Switch to a different model
- `clear` - Clear conversation history
- `history` - Show conversation history
- `quit/exit/q` - Exit the chatbot

### Command Line Options

```bash
python chatbot.py --help
python chatbot.py --model gpt2-medium  # Use specific model
python chatbot.py --no-gpu             # Disable GPU
python chatbot.py --config my_config.json  # Custom config file
```

### Supported Model Types

**This chatbot works with ANY LLM model!** The architecture is completely general and supports:

#### **Popular Model Families:**
- **GPT Models**: GPT-2, GPT-3.5, GPT-4, CodeGPT
- **LLaMA Models**: LLaMA, LLaMA 2, Code Llama, Alpaca, Vicuna
- **Mistral Models**: Mistral 7B, Mixtral 8x7B
- **Google Models**: Gemma, PaLM, T5, FLAN-T5
- **Microsoft Models**: Phi-2, Phi-3, DialoGPT
- **Code Models**: StarCoder, CodeGen, WizardCoder
- **Other Models**: BLOOM, OPT, Falcon, MPT, and more!

#### **Pre-configured Models:**
- `gpt2-large` - GPT-2 Large (774M parameters) - **Default**
- `gpt2-medium` - GPT-2 Medium (355M parameters)
- `gpt2` - GPT-2 Base (124M parameters)
- `distilgpt2` - DistilGPT-2 (82M parameters, faster)
- `llama2-7b` - LLaMA 2 7B Chat
- `mistral-7b` - Mistral 7B Instruct
- `phi-2` - Microsoft Phi-2
- `codegen` - Salesforce CodeGen

#### **Model Switching:**
```bash
# During chat
switch mistral-7b
switch llama2-7b
switch phi-2

# From command line
python chatbot.py --model mistral-7b
```

#### **Adding New Models:**
```bash
# Run the example script to add popular models
python add_models_example.py

# Or add manually in chat
# Just use any HuggingFace model name!
switch microsoft/DialoGPT-large
switch bigcode/starcoder
switch meta-llama/Llama-2-13b-chat-hf
```

## Configuration

The chatbot uses `chatbot_config.json` for configuration. Key settings:

```json
{
  "current_model": "openai-community/gpt2-large",
  "use_gpu": true,
  "generation_params": {
    "max_new_tokens": 100,
    "temperature": 0.7,
    "top_p": 0.9,
    "do_sample": true
  }
}
```

## GPU Requirements

- NVIDIA GPU with CUDA support
- CUDA 11.8 (or compatible version)
- Sufficient GPU memory for the model

## Files

- `chatbot.py` - Main CLI chatbot interface
- `model_manager.py` - LLM model management and loading
- `config.py` - Configuration system
- `chatbot_config.json` - Configuration file (auto-generated)
- `requirements.txt` - Python dependencies

## Adding New Models

You can easily add new models by editing the configuration:

```python
from config import ChatbotConfig

config = ChatbotConfig()
config.add_model(
    alias="llama2",
    model_name="meta-llama/Llama-2-7b-chat-hf",
    description="Llama 2 7B Chat model"
)
```

Or directly edit `chatbot_config.json`:

```json
{
  "available_models": {
    "custom-model": {
      "name": "path/to/your/model",
      "description": "Your custom model",
      "type": "causal_lm"
    }
  }
}
```
