# Simple LLM Service - Production Deployment Guide

## 🎯 Phase 3: Production Features

This guide covers Phase 3 implementation with production-ready features for enterprise deployment.

## 🔐 Authentication & Security

### JWT Authentication
```bash
# Login to get access token
curl -X POST http://localhost:8000/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username": "admin", "password": "admin123"}'

# Use token in requests
curl -H "Authorization: Bearer YOUR_JWT_TOKEN" \
  http://localhost:8000/v1/chat/completions
```

### API Key Authentication
```bash
# Use API key directly
curl -H "Authorization: Bearer sk-simple-llm-demo-key-123" \
  http://localhost:8000/v1/chat/completions
```

### Default Credentials
- **Admin User**: username: `admin`, password: `admin123`
- **Regular User**: username: `user`, password: `user123`  
- **Demo API Key**: `sk-simple-llm-demo-key-123`

⚠️ **Change these in production!**

## ⏱️ Rate Limiting

### Per-User Rate Limits
- **Anonymous**: 10/min, 100/hour, 1000/day
- **Regular Users**: 50/min, 500/hour, 5000/day
- **Admin Users**: 200/min, 2000/hour, 20000/day
- **API Keys**: 100/min, 1000/hour, 10000/day
- **Admin API Keys**: 1000/min, 10000/hour, 100000/day

### Rate Limit Headers
```
X-RateLimit-Per-Minute-Limit: 100
X-RateLimit-Per-Minute-Remaining: 95
X-RateLimit-Per-Minute-Reset: 1629123456
```

## 📊 Monitoring & Metrics

### Health Endpoints
```bash
GET /health              # Basic health check
GET /health/ready        # Kubernetes readiness
GET /health/live         # Kubernetes liveness
GET /health/detailed     # Detailed health (auth required)
```

### Monitoring Endpoints (Admin only)
```bash
GET /admin/status        # Public service status
GET /admin/metrics       # Comprehensive metrics
GET /admin/metrics/system    # System resources
GET /admin/metrics/application  # Application metrics
GET /admin/metrics/requests     # Request statistics
```

### Sample Metrics Response
```json
{
  "system": {
    "cpu": {"usage_percent": 45.2, "count": 8},
    "memory": {"total": 34359738368, "used": 8589934592, "percent": 25.0},
    "gpu": {"name": "RTX 5060 Ti", "memory_usage_percent": 67.3}
  },
  "application": {
    "uptime_seconds": 3600,
    "total_requests": 1250,
    "error_rate": 0.02,
    "requests_per_second": 0.35
  }
}
```

## 🐳 Docker Deployment

### Build & Run
```bash
# Build image
docker build -t simple-llm-service .

# Run with Docker Compose
docker-compose up -d

# Run with custom environment
docker run -d \
  -p 8000:8000 \
  --gpus all \
  -e JWT_SECRET_KEY="your-secret-key" \
  -v ./config:/app/config:ro \
  -v ./logs:/app/logs \
  simple-llm-service
```

### Docker Compose Features
- **GPU Support**: Automatic NVIDIA GPU detection
- **Persistent Storage**: Model cache, logs, conversations
- **Health Checks**: Built-in container health monitoring
- **Optional Services**: Redis, Prometheus, Grafana

### Enable Monitoring Stack
```bash
# Start with monitoring
docker-compose --profile monitoring up -d

# Access services
# API Service: http://localhost:8000
# Prometheus: http://localhost:9090  
# Grafana: http://localhost:3000 (admin/admin)
```

## ☸️ Kubernetes Deployment

### Prerequisites
- Kubernetes cluster with GPU support
- NVIDIA Device Plugin installed
- Ingress controller (nginx recommended)

### Deploy to Kubernetes
```bash
# Create storage
kubectl apply -f k8s/storage.yaml

# Create config and secrets
kubectl apply -f k8s/configmap.yaml

# Deploy application
kubectl apply -f k8s/deployment.yaml

# Check status
kubectl get pods -l app=simple-llm-service
kubectl logs -l app=simple-llm-service

# Port forward for testing
kubectl port-forward service/simple-llm-service 8000:8000
```

### Kubernetes Features
- **Auto-scaling**: HPA support for CPU/memory
- **Health Probes**: Startup, liveness, readiness
- **Resource Limits**: GPU, CPU, memory constraints
- **Persistent Storage**: Model cache and data persistence
- **Ingress**: Load balancing and SSL termination

## 🔧 Environment Configuration

### Environment Variables
```bash
# Server
HOST=0.0.0.0
PORT=8000
LOG_LEVEL=INFO

# Authentication  
JWT_SECRET_KEY=your-very-long-secret-key
ACCESS_TOKEN_EXPIRE_MINUTES=60
ENABLE_AUTHENTICATION=true
REQUIRE_AUTH_FOR_COMPLETIONS=false

# Rate Limiting
ENABLE_RATE_LIMITING=true
REDIS_URL=redis://localhost:6379/0

# Monitoring
ENABLE_METRICS=true
METRICS_RETENTION_HOURS=24

# Performance
MAX_CONCURRENT_REQUESTS=10
REQUEST_TIMEOUT_SECONDS=300
```

### Configuration Files
- `.env` - Environment variables
- `config/models.yaml` - Model configurations
- `docker-compose.yml` - Docker services
- `k8s/` - Kubernetes manifests

## 🧪 Testing Production Features

### Run Phase 3 Tests
```bash
python test_production_features.py
```

### Test Categories
- ✅ **Authentication**: JWT & API key auth
- ✅ **Rate Limiting**: Per-user limits
- ✅ **Monitoring**: Metrics & health checks
- ✅ **Protected Endpoints**: Auth-required endpoints
- ✅ **Health Checks**: All health endpoints

### Manual Testing Examples

#### Authentication
```bash
# Get JWT token
TOKEN=$(curl -s -X POST http://localhost:8000/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username":"admin","password":"admin123"}' \
  | jq -r '.access_token')

# Use token
curl -H "Authorization: Bearer $TOKEN" \
  http://localhost:8000/auth/me
```

#### Rate Limiting
```bash
# Rapid requests to trigger rate limit
for i in {1..20}; do
  curl -w "%{http_code}\n" -o /dev/null -s http://localhost:8000/health
  sleep 0.1
done
```

#### Monitoring
```bash
# Get system metrics (admin required)
curl -H "Authorization: Bearer $TOKEN" \
  http://localhost:8000/admin/metrics/system
```

## 🚀 Production Deployment Checklist

### Security
- [ ] Change default passwords
- [ ] Generate strong JWT secret key
- [ ] Configure HTTPS/TLS
- [ ] Set up firewall rules
- [ ] Enable audit logging

### Performance
- [ ] Configure GPU resources
- [ ] Set appropriate rate limits
- [ ] Optimize model loading
- [ ] Configure caching (Redis)
- [ ] Set up load balancing

### Monitoring
- [ ] Set up Prometheus metrics
- [ ] Configure Grafana dashboards
- [ ] Set up alerting rules
- [ ] Configure log aggregation
- [ ] Set up health checks

### High Availability
- [ ] Multiple replicas
- [ ] Persistent storage
- [ ] Backup strategy
- [ ] Disaster recovery plan
- [ ] Auto-scaling policies

## 📈 Scaling Considerations

### Horizontal Scaling
- **Stateless Design**: Service is stateless except for model loading
- **Load Balancing**: Multiple instances behind load balancer
- **Shared Storage**: Persistent volumes for model cache
- **Session Affinity**: Not required due to stateless design

### Vertical Scaling
- **GPU Memory**: Primary constraint for larger models
- **CPU/RAM**: For request processing and tokenization
- **Storage**: For model cache and conversation history

### Performance Optimization
- **Model Caching**: Shared persistent volumes
- **Connection Pooling**: Redis for rate limiting
- **Async Processing**: FastAPI async capabilities
- **GPU Utilization**: Optimized model loading

## 🔍 Troubleshooting

### Common Issues

#### Authentication Failed
```bash
# Check JWT secret configuration
echo $JWT_SECRET_KEY

# Verify user credentials
curl -X POST http://localhost:8000/auth/login \
  -d '{"username":"admin","password":"admin123"}'
```

#### Rate Limit Issues
```bash
# Check rate limit configuration
curl http://localhost:8000/admin/status

# Monitor rate limit headers
curl -I http://localhost:8000/health
```

#### Model Loading Failed
```bash
# Check GPU availability
nvidia-smi

# Check logs
kubectl logs -l app=simple-llm-service
# or
docker logs simple-llm-service
```

#### High Memory Usage
```bash
# Monitor system metrics
curl -H "Authorization: Bearer $TOKEN" \
  http://localhost:8000/admin/metrics/system
```

### Performance Tuning

#### GPU Optimization
- Use Flash Attention 2 (auto-enabled)
- Configure optimal batch size
- Monitor GPU memory usage
- Consider model quantization

#### API Performance
- Adjust rate limits per use case
- Configure appropriate timeouts
- Monitor response times
- Use caching where appropriate

## 📋 Next Steps

### Phase 4 Enhancements
- [ ] **Streaming Responses**: Real-time response streaming
- [ ] **WebSocket Support**: Interactive connections
- [ ] **Advanced Caching**: Response caching with Redis
- [ ] **Model Switching**: Hot model swapping
- [ ] **Conversation Persistence**: Database integration
- [ ] **Advanced Monitoring**: Custom metrics & dashboards

### Enterprise Features
- [ ] **Multi-tenant**: Organization-based isolation
- [ ] **Advanced Auth**: OAuth2, SAML, LDAP
- [ ] **Audit Logging**: Comprehensive request auditing
- [ ] **Data Governance**: PII detection & handling
- [ ] **Compliance**: SOC2, GDPR compliance features

## 🎉 Phase 3 Complete!

Your Simple LLM Service now includes:

✅ **Production Authentication** - JWT & API key auth  
✅ **Advanced Rate Limiting** - Per-user intelligent limits  
✅ **Comprehensive Monitoring** - Metrics, health checks, alerts  
✅ **Docker Containerization** - Production-ready containers  
✅ **Kubernetes Support** - Enterprise-grade orchestration  
✅ **Security Hardening** - Production security features  

**Ready for enterprise deployment and can fully replace NVIDIA NIM!** 🚀
