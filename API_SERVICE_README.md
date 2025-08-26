# Simple LLM Service API

## 🎯 Overview

Simple LLM Service là một API service tương thích với OpenAI/NVIDIA NIM, được xây dựng trên FastAPI và Hugging Face Transformers. Service này cung cấp REST API endpoints để tương tác với các mô hình LLM một cách hiệu quả.

## 🏗️ Architecture

### Project Structure
```
simple_chatbot/
├── chatbot.py              # CLI mode entry point
├── nim_service.py          # API service entry point
├── test_api_service.py     # API testing script
├── requirements.txt        # Base dependencies
├── requirements-api.txt    # API-specific dependencies
├── config/
│   └── models.yaml        # Model configurations
├── src/
│   ├── config_manager.py   # Shared configuration manager
│   ├── model_manager.py    # Shared model manager  
│   ├── conversation_manager.py # Shared conversation manager
│   ├── cli/               # CLI Mode
│   │   ├── __init__.py
│   │   └── cli.py
│   └── api/               # API Service Mode
│       ├── __init__.py
│       ├── server.py      # FastAPI server
│       ├── models.py      # Pydantic models
│       ├── middleware/
│       │   ├── __init__.py
│       │   └── logging.py
│       └── routes/
│           ├── __init__.py
│           ├── health.py     # Health checks
│           ├── models.py     # Model management
│           ├── chat.py       # Chat completions
│           └── completions.py # Text completions
└── logs/                  # API service logs
```

### Dual Mode Design
- **CLI Mode**: Interactive chatbot với rich terminal interface
- **API Mode**: REST API service tương thích OpenAI/NIM

## 🚀 Quick Start

### 1. Environment Setup
```bash
# Activate virtual environment
source venv/bin/activate

# Install API dependencies
pip install -r requirements-api.txt
```

### 2. Start CLI Mode
```bash
python chatbot.py                    # Default config
python chatbot.py -m model_name     # Specific model
python chatbot.py -c config.yaml    # Custom config
```

### 3. Start API Service
```bash
python nim_service.py               # Default (127.0.0.1:8000)
python nim_service.py --host 0.0.0.0 --port 8080  # Custom host/port
python nim_service.py --reload      # Development mode
```

### 4. Test API Service
```bash
python test_api_service.py
```

## 📚 API Documentation

### Base URL
```
http://localhost:8000
```

### Available Endpoints

#### Health Check
```bash
GET /health
GET /health/ready  # Kubernetes readiness
GET /health/live   # Kubernetes liveness
```

#### Models Management
```bash
GET /v1/models                    # List available models
GET /v1/models/{model_id}        # Get model info
POST /v1/models/{model_id}/load  # Load specific model
```

#### Chat Completions (OpenAI Compatible)
```bash
POST /v1/chat/completions
```

**Request Example:**
```json
{
  "model": "llama-3.2-3b-instruct",
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello!"}
  ],
  "max_tokens": 100,
  "temperature": 0.7
}
```

#### Text Completions (OpenAI Compatible)
```bash
POST /v1/completions
```

**Request Example:**
```json
{
  "model": "llama-3.2-3b-instruct",
  "prompt": "The future of AI is",
  "max_tokens": 50,
  "temperature": 0.7
}
```

### Interactive Documentation
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## 🔧 Configuration

### Model Configuration (`config/models.yaml`)
```yaml
models:
  llama-3.2-3b-instruct:
    model_id: "meta-llama/Llama-3.2-3B-Instruct"
    display_name: "Llama 3.2 3B Instruct"
    # ... other config
```

### Environment Variables
```bash
# Optional environment variables
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false
```

## 🚀 Features

### Core Features
- ✅ **Dual Mode**: CLI và API service
- ✅ **OpenAI Compatible**: Tương thích với OpenAI API format
- ✅ **Model Management**: Load/unload models dynamically
- ✅ **Flash Attention 2**: Optimized performance
- ✅ **Quantization**: 4-bit, 8-bit support
- ✅ **LoRA Adapters**: Fine-tuned model support

### API Service Features
- ✅ **FastAPI**: High-performance async API
- ✅ **Health Checks**: Kubernetes-ready endpoints
- ✅ **Error Handling**: Robust error responses
- ✅ **Logging**: Comprehensive request/response logging
- ✅ **CORS**: Cross-origin resource sharing
- ✅ **Auto Documentation**: Swagger/ReDoc integration

### Performance Features
- ✅ **Flash Attention 2**: Improved memory efficiency
- ✅ **GPU Optimization**: RTX 5060 Ti optimized
- ✅ **Memory Management**: Automatic device mapping
- ✅ **Model Caching**: Efficient model loading

## 🧪 Testing

### Manual Testing
```bash
# Test health endpoint
curl http://localhost:8000/health

# Test models list
curl http://localhost:8000/v1/models

# Test chat completion
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama-3.2-3b-instruct",
    "messages": [
      {"role": "user", "content": "Hello!"}
    ],
    "max_tokens": 100
  }'
```

### Automated Testing
```bash
python test_api_service.py
```

## 📊 Comparison with NVIDIA NIM

| Feature | NVIDIA NIM | Simple LLM Service |
|---------|------------|-------------------|
| **Cost** | $$$ License fee | ✅ Free & Open Source |
| **Models** | Curated models | ✅ Any Hugging Face model |
| **Customization** | Limited | ✅ Full source control |
| **Hardware** | NVIDIA only | ✅ Any CUDA GPU |
| **Deployment** | Cloud/Enterprise | ✅ Local/Cloud/Edge |
| **API Format** | OpenAI standard | ✅ OpenAI compatible |
| **Setup Time** | Complex | ✅ Minutes to deploy |

## 🛠️ Development

### Adding New Endpoints
1. Create route in `src/api/routes/`
2. Add Pydantic models in `src/api/models.py`
3. Register router in `src/api/server.py`

### Extending Functionality
- Add middleware in `src/api/middleware/`
- Extend core managers in `src/`
- Update configuration in `config/models.yaml`

## 📋 Next Steps (Phase 3+)

### Authentication & Security
- [ ] API key authentication
- [ ] Rate limiting per user
- [ ] Request/response encryption

### Production Features
- [ ] Docker containerization
- [ ] Kubernetes deployment
- [ ] Horizontal scaling
- [ ] Load balancing

### Advanced Features
- [ ] Streaming responses
- [ ] WebSocket support
- [ ] Model switching API
- [ ] Conversation persistence
- [ ] Metrics & monitoring

## 🤝 Usage Examples

### Python Client
```python
import requests

# Chat completion
response = requests.post("http://localhost:8000/v1/chat/completions", json={
    "model": "llama-3.2-3b-instruct",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 100
})
print(response.json()['choices'][0]['message']['content'])
```

### cURL Examples
```bash
# List models
curl http://localhost:8000/v1/models

# Chat completion
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "llama-3.2-3b-instruct", "messages": [{"role": "user", "content": "Hi!"}]}'
```

## 🎉 Success Metrics

Phase 1 & 2 Implementation đã hoàn thành thành công:

- ✅ **Structure**: Clean separation của CLI và API modes
- ✅ **API Server**: FastAPI server với full OpenAI compatibility  
- ✅ **Endpoints**: Health, Models, Chat, Completions
- ✅ **Integration**: Sử dụng lại existing ModelManager và core components
- ✅ **Documentation**: Comprehensive API docs và examples
- ✅ **Testing**: Automated test script

Service đã sẵn sàng để replace NVIDIA NIM trong microservice architectures!
