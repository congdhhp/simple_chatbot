# Simple CLI Chatbot 🤖

A flexible command-line chatbot powered by Hugging Face Transformers with CUDA support. Optimized for Ubuntu 24.04 WSL2 + RTX 5060 Ti 16GB + CUDA 12.0.

## Features

- 🚀 **Multiple Model Support**: Easy switching between different LLM models
- ⚡ **CUDA 12.0 Acceleration**: Optimized for RTX 5060 Ti with 16GB VRAM
- 🔥 **Flash Attention**: Memory-efficient attention computation for better performance
- 🔧 **Flexible Configuration**: YAML-based configuration for easy model management
- 💬 **Conversation Management**: Save, load, and manage conversation history
- 🎨 **Rich CLI Interface**: Beautiful command-line interface with syntax highlighting
- 📦 **Virtual Environment**: Isolated Python environment for clean dependency management
- 🔄 **Memory Optimization**: Automatic memory management optimized for 16GB VRAM
- 🐧 **WSL2 Optimized**: Specifically configured for Ubuntu 24.04 in WSL2

## Quick Start for Ubuntu 24.04 WSL2 + RTX 5060 Ti

### Prerequisites

Ensure you have:
- Ubuntu 24.04 in WSL2
- NVIDIA drivers installed on Windows host
- CUDA 12.0 toolkit: `nvcc --version` should show release 12.0
- Python 3.12.3 (recommended): `python3 --version`

### 1. Automated Setup (Recommended)

```bash
# Run the automated setup script
python3 setup.py
```

This will:
- Create virtual environment
- Install PyTorch with CUDA 12.1 support (compatible with CUDA 12.0)
- Install flash-attention and other optimizations
- Configure environment for RTX 5060 Ti

### 2. Manual Setup (Alternative)

```bash
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install PyTorch with CUDA 12.1 support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install flash-attention (may take several minutes)
pip install flash-attn --no-build-isolation

# Install other dependencies
pip install -r requirements.txt
```

### 3. Run the Chatbot

```bash
# Use the optimized launcher script
./run_chatbot.sh

# Or manually activate and run
source venv/bin/activate
python3 chatbot.py

# Start with specific model
python3 chatbot.py -m llama-3.2-1b-instruct

# Use custom configuration
python3 chatbot.py -c config/custom.yaml
```

## Supported Models (RTX 5060 Ti Optimized)

The chatbot comes pre-configured with several models optimized for 16GB VRAM:

| Model | Size | VRAM Usage | Max Tokens | Quantization | Description |
|-------|------|------------|------------|--------------|-------------|
| `llama-3.2-1b-instruct` | 1B | ~4GB | 2048 | FP16 | Fastest, most efficient |
| `llama-3.2-3b-instruct` | 3B | ~8GB | 1024 | FP16 | Balanced performance (default) |
| `mistral-7b-instruct` | 7B | ~12GB | 1024 | FP16 | High quality responses |
| `llama-3.1-8b-instruct` | 8B | ~15GB | 1024 | FP16 | Maximum quality, uses full VRAM |
| `llama-3.1-8b-instruct-8bit` | 8B | ~10GB | 1024 | 8-bit | Balanced performance and memory |
| `llama-3.1-8b-instruct-4bit` | 8B | ~8GB | 1024 | 4-bit | Large model, efficient memory |
| `codellama-13b-instruct-8bit` | 13B | ~14GB | 1024 | 8-bit | Largest model, better quality |
| `codellama-13b-instruct-4bit` | 13B | ~10GB | 1024 | 4-bit | Largest model, maximum efficiency |

All models include:
- ⚡ Flash Attention support for memory efficiency
- 🔥 Optimized generation parameters for RTX 5060 Ti
- 📈 Increased context length taking advantage of 16GB VRAM
- 🗜️ **Quantization Support**: Available for large models (7B+ parameters)

### Quantization Support

The chatbot now supports both 4-bit and 8-bit quantization using BitsAndBytesConfig for efficient memory usage:

- **Benefits**: Run 13B models on 16GB GPU (normally requires 24GB+)
- **4-bit Quantization**: ~60-70% memory reduction, maximum efficiency
- **8-bit Quantization**: ~40-50% memory reduction, better quality than 4-bit
- **Performance**: Minimal inference impact, slight loading overhead

**Quantized Models Available:**
- `llama-3.1-8b-instruct-8bit`: 8B model in ~10GB VRAM (vs 15GB)  
- `llama-3.1-8b-instruct-4bit`: 8B model in ~8GB VRAM (vs 15GB)
- `codellama-13b-instruct-8bit`: 13B model in ~14GB VRAM (vs 26GB)
- `codellama-13b-instruct-4bit`: 13B model in ~10GB VRAM (vs 26GB)

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

## Testing Quantization

To test quantization support and verify your setup:

```bash
# Test all model quantization settings
python test_quantization.py

# Test specific quantized models
python test_quantization.py codellama-13b-instruct-4bit
python test_quantization.py codellama-13b-instruct-8bit
python test_quantization.py llama-3.1-8b-instruct-4bit
python test_quantization.py llama-3.1-8b-instruct-8bit
```

The test script will:
- Show quantization status for all models (🔵 FP16, 🟡 8-bit, 🟢 4-bit)
- Load and test the specified model
- Verify memory usage and generation quality
- Display device placement and configuration

## Memory Requirements (RTX 5060 Ti 16GB)

| Model Size | Precision | VRAM Usage | Context Length | Performance |
|------------|-----------|------------|----------------|-------------|
| 1B | FP16 | ~4GB | 25 messages | ⚡⚡⚡ Fastest |
| 3B | FP16 | ~8GB | 15 messages | ⚡⚡ Fast |
| 7B | FP16 | ~12GB | 12 messages | ⚡ Good |
| 8B | FP16 | ~15GB | 10 messages | 🎯 Best Quality |
| 8B | 8-bit | ~10GB | 11 messages | 🟡 Balanced |
| 8B | 4-bit | ~8GB | 12 messages | 🗜️ Memory Efficient |
| 13B | 8-bit | ~14GB | 7 messages | 🟡 Large Model, Good Quality |
| 13B | 4-bit | ~10GB | 8 messages | 🚀 Largest Model, Max Efficiency |

**RTX 5060 Ti Advantages:**
- 🚀 No quantization needed - full FP16 precision
- 📈 Larger context windows for better conversations
- ⚡ Flash attention for memory-efficient processing
- 🔥 Optimal performance with CUDA 12.0

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

## WSL2 Setup Guide

### 1. Install WSL2 with Ubuntu 24.04
```bash
# On Windows PowerShell (as Administrator)
wsl --install -d Ubuntu-24.04
```

### 2. Install NVIDIA Drivers
- Install latest NVIDIA drivers on Windows host
- WSL2 will automatically access GPU through Windows drivers
- No need to install drivers inside WSL2

### 3. Install CUDA Toolkit in WSL2
```bash
# Update package list
sudo apt update

# Install CUDA 12.0
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt-get update
sudo apt-get -y install cuda-toolkit-12-0

# Add to PATH
echo 'export PATH=/usr/local/cuda-12.0/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.0/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

## Troubleshooting

### WSL2 + CUDA Issues

If CUDA is not detected in WSL2:

1. **Check Windows NVIDIA drivers**: Ensure latest drivers are installed on Windows host
2. **Verify WSL2 GPU access**: `nvidia-smi` should work in WSL2
3. **Check CUDA toolkit**: `nvcc --version` should show 12.0
4. **Restart WSL2**: `wsl --shutdown` then restart Ubuntu

### Flash Attention Issues

If flash-attention fails to install:

1. **Install build dependencies**:
   ```bash
   sudo apt install build-essential python3-dev
   ```
2. **Try alternative installation**:
   ```bash
   pip install flash-attn --no-build-isolation --no-cache-dir
   ```
3. **Skip flash-attention**: Remove from requirements.txt if needed

### Memory Issues

With 16GB VRAM, memory issues are rare, but if they occur:

1. **Check GPU memory**: `nvidia-smi` to see current usage
2. **Restart chatbot**: Clear GPU memory
3. **Use smaller model**: Switch to 1B or 3B model
4. **Check system RAM**: Ensure sufficient system memory

### Model Loading Issues

If models fail to load:

1. **Check internet connection**: First download requires internet
2. **Verify disk space**: Models can be 5-15GB each
3. **Check HuggingFace cache**: `~/.cache/huggingface/`
4. **Clear cache if corrupted**: `huggingface-cli delete-cache`

## System Requirements

### Hardware
- RTX 5060 Ti 16GB VRAM (or similar high-VRAM GPU)
- 16GB+ system RAM
- 50GB+ free disk space (for model cache)

### Software
- Ubuntu 24.04 in WSL2
- Python 3.12.3 (recommended, 3.10+ required)
- CUDA 12.0 toolkit
- NVIDIA drivers on Windows host (for WSL2)

### Verification Commands
```bash
# Check Python version
python3 --version  # Should show 3.12.3

# Check CUDA
nvcc --version     # Should show release 12.0
nvidia-smi         # Should show RTX 5060 Ti with 16GB

# Check GPU in Python
python3 -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"
```

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

---

## API Service Extensions (Wave 1 & Wave 2)

The repository now also includes an OpenAI-compatible FastAPI inference service with production-focused enhancements:

### Core (Wave 1)
- Chat & text completion endpoints with SSE streaming
- Precise token accounting (prompt vs completion)
- Prometheus metrics (`/metrics`)
- Request ID injection & basic middleware layering
- Conversation & multi-model management

### Hardening (Wave 2)
- Centralized error handling middleware (standard JSON envelope)
- Async model loading + optional bulk preload (`PRELOAD_ALL_MODELS=true`)
- Degraded startup mode surfaced via health/readiness
- Streaming heartbeat & time-to-first-token metrics
- Usage object enriched: `latency_ms`, `tokens_per_second`, `time_to_first_token_ms`, `cost_usd`
- Cost estimation via env: `COST_PER_1K_PROMPT_TOKENS`, `COST_PER_1K_COMPLETION_TOKENS`
- Structured JSON logging (`JSON_LOGS=true`)
- Config validation endpoint: `/admin/config/validate`
- Rate limit exceeded counter: `llm_rate_limit_exceeded_total`
- Access logging middleware with request/context enrichment

### Relevant Environment Variables
| Variable | Purpose | Default |
|----------|---------|---------|
| PRELOAD_ALL_MODELS | Load all configured models at startup | false |
| JSON_LOGS | Enable structured JSON logging | false |
| COST_PER_1K_PROMPT_TOKENS | USD rate for prompt tokens | 0 |
| COST_PER_1K_COMPLETION_TOKENS | USD rate for completion tokens | 0 |

### Additional Metrics
| Metric | Labels | Description |
|--------|--------|-------------|
| llm_chat_latency_seconds | (none) | Chat completion latency histogram |
| llm_chat_time_to_first_token_seconds | (none) | Time to first streamed token |
| llm_chat_tokens_total | type, model | Prompt vs completion token counts (chat) |
| llm_completion_latency_seconds | (none) | Text completion latency |
| llm_completion_tokens_total | type, model | Prompt vs completion token counts (text) |
| llm_rate_limit_exceeded_total | period | Rate limit rejects by window |

### Cost Estimation
Set per‑1000 token rates in USD; the service computes `cost_usd` in the `usage` block for both streaming and non-stream responses.

---