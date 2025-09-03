# NVIDIA NIM-Compatible Embedding Service

## 🎯 Overview

A NVIDIA NIM-compatible embedding service built to repl```bash
curl -X POST "http://localhost:8009/v1/embeddings" 
  -H "Content-Type: application/json" 
  -d '{
    "model": "e5-large-v2",
    "input": "query: What is machine learning?"
  }'
```cr.io/nim/nvidia/nv-embedqa-e5-v5:1.0.1`. This service follows the [NVIDIA NIM Text Embedding API Reference](https://docs.nvidia.com/nim/nemo-retriever/text-embedding/latest/reference.html) specification while providing a flexible, production-ready alternative.

## 🏗️ Architecture

### Project Structure
```
embedding/
├── README.md
├── requirements.txt           # Dependencies for embedding service
├── docker-compose.yml        # Development deployment
├── Dockerfile               # Production container
├── embedding_service.py     # Main service entry point
├── test_embedding_api.py    # API testing script
├── config/
│   └── models.yaml         # Model configurations
├── src/
│   ├── __init__.py
│   ├── config_manager.py   # Configuration management
│   ├── model_manager.py    # Model loading and management
│   ├── embedding_manager.py # Embedding generation logic
│   ├── models/            # Model family handlers
│   │   ├── __init__.py
│   │   ├── base_model.py  # Base model interface
│   │   ├── e5_model.py    # E5 family models
│   │   ├── gte_model.py   # GTE family models
│   │   ├── nv_embed_model.py # NV-Embed family models
│   │   └── sentence_transformers_model.py # Generic SentenceTransformers
│   ├── api/
│   │   ├── __init__.py
│   │   ├── server.py      # FastAPI server
│   │   ├── models.py      # Pydantic models for API
│   │   ├── middleware/
│   │   │   ├── __init__.py
│   │   │   ├── logging.py
│   │   │   ├── monitoring.py
│   │   │   └── rate_limit.py
│   │   └── routes/
│   │       ├── __init__.py
│   │       ├── health.py     # Health endpoints
│   │       ├── models.py     # Model listing endpoints
│   │       └── embeddings.py # Embedding generation endpoints
│   └── utils/
│       ├── __init__.py
│       ├── preprocessing.py  # Text/image preprocessing
│       ├── postprocessing.py # Embedding postprocessing
│       └── compression.py    # Embedding compression (int8, binary)
└── logs/                    # Service logs
```

## 🚀 Features

### ✅ NVIDIA NIM API Compatibility
- **Full OpenAI compatibility**: `/v1/embeddings`, `/v1/models`, `/v1/health/*`
- **Input type support**: `query` vs `passage` modes for retrieval models
- **Modality support**: `text`, `image`, `text_image` (planned)
- **Model suffix syntax**: Support for `-query` and `-passage` model suffixes
- **Embedding types**: `float`, `int8`, `uint8`, `binary`, `ubinary`
- **Dynamic dimensions**: Matryoshka Representation Learning support

### 🎯 Model Family Support
- **E5 Models**: `e5-large-v2`, `e5-base-v2`, `e5-small-v2` (default family)
- **GTE Models**: `gte-large`, `gte-base`, `gte-small`
- **NV-Embed Models**: Custom NVIDIA embedding models
- **Generic SentenceTransformers**: Any compatible model from Hugging Face

### ⚡ Performance Features
- **GPU acceleration**: CUDA support with memory optimization
- **Dynamic batching**: Configurable batch processing
- **Memory management**: Efficient model loading and unloading
- **Multi-model support**: Runtime model switching

### 🔧 Production Features
- **Health checks**: Kubernetes-ready health endpoints
- **Monitoring**: Prometheus metrics and structured logging
- **Rate limiting**: Per-user/API key rate limiting
- **Authentication**: JWT and API key support
- **Docker deployment**: Production-ready containerization

## 🚀 Quick Start

### 1. Environment Setup
```bash
cd other_services/embedding

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Start Service
```bash
# Development mode
python embedding_service.py --reload

# Production mode
python embedding_service.py --host 0.0.0.0 --port 8001

# Docker deployment
docker-compose up -d --build
```

### 3. Test API
```bash
# Test the service
python test_embedding_api.py

# Manual test
curl -X POST "http://localhost:8001/v1/embeddings" \
  -H "Content-Type: application/json" \
  -d '{
    "input": ["Hello world"],
    "model": "e5-large-v2",
    "input_type": "query"
  }'
```

## 📚 API Reference

### Base URL
```
http://localhost:8001
```

### Health Endpoints
```bash
GET /v1/health/ready    # Kubernetes readiness
GET /v1/health/live     # Kubernetes liveness
```

### Model Management
```bash
GET /v1/models          # List available models
```

### Embeddings Generation
```bash
POST /v1/embeddings     # Generate embeddings (NVIDIA NIM compatible)
```

#### Example Request
```json
{
  "input": ["What is the population of Pittsburgh?"],
  "model": "e5-large-v2",
  "input_type": "query",
  "modality": "text",
  "embedding_type": "float",
  "dimensions": 1024
}
```

#### Example Response
```json
{
  "object": "list",
  "data": [
    {
      "index": 0,
      "embedding": [0.001, -0.017, ...],
      "object": "embedding"
    }
  ],
  "model": "e5-large-v2",
  "usage": {
    "prompt_tokens": 6,
    "total_tokens": 6
  }
}
```

## 🔧 Configuration

### Model Configuration (config/models.yaml)
```yaml
models:
  e5-large-v2:
    model_id: "intfloat/e5-large-v2"
    family: "e5"
    max_seq_length: 512
    embedding_dimension: 1024
    supports_input_type: true
    supports_dimensions: [256, 512, 768, 1024]
    
  gte-large:
    model_id: "thenlper/gte-large"
    family: "gte"
    max_seq_length: 512
    embedding_dimension: 1024
    supports_input_type: false
```

## 🎯 Model Families

### E5 Models
- **Input types**: Requires `query` vs `passage` specification
- **Prefix handling**: Automatic prefix addition for retrieval tasks
- **Performance**: Optimized for retrieval and semantic similarity

### GTE Models  
- **Input types**: No input type differentiation needed
- **Usage**: General-purpose text embeddings
- **Performance**: Balanced performance across tasks

### NV-Embed Models
- **Input types**: Advanced query/passage handling
- **Compression**: Native support for compressed embeddings
- **Performance**: NVIDIA-optimized models

## 🏆 Advantages over NVIDIA NIM

### 💰 Cost Efficiency
- **No licensing fees**: Open source alternative
- **Hardware flexibility**: Run on any CUDA-capable GPU
- **Resource optimization**: Configurable memory usage

### 🔧 Customization
- **Model flexibility**: Support any Hugging Face embedding model
- **Custom preprocessing**: Configurable text preprocessing
- **API extensions**: Easy to add custom endpoints

### 🚀 Performance
- **Local deployment**: No network latency to NVIDIA cloud
- **Batch optimization**: Custom batching strategies
- **Memory efficiency**: Optimized for various GPU memory sizes

## 📈 Scaling & Deployment

### Development
```bash
python embedding_service.py --reload
```

### Production
```bash
# Docker Compose
docker-compose up -d

# Kubernetes
kubectl apply -f k8s/
```

### Load Balancing
```yaml
# Multiple replicas with model sharding
replicas: 3
models_per_replica: 2
```

## 🔍 Monitoring

### Metrics
- Request rate and latency
- Model loading/unloading events  
- GPU memory usage
- Embedding generation performance

### Logs
- Structured JSON logging
- Request/response tracking
- Error handling and debugging

## 🛡️ Security

### Authentication
- JWT token authentication
- API key management
- Role-based access control

### Rate Limiting
- Per-user rate limits
- Model-specific quotas
- Dynamic rate adjustment

---

**Built with the same excellence as the Simple LLM Service architecture** 🚀
