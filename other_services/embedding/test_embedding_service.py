"""Test suite for the embedding service."""

import pytest
import asyncio
import httpx
from typing import Dict, Any
import yaml


class TestEmbeddingService:
    """Test cases for the embedding service API."""
    
    BASE_URL = "http://localhost:8009"
    
    @pytest.fixture
    async def client(self):
        """Create an async HTTP client for testing."""
        async with httpx.AsyncClient(base_url=self.BASE_URL) as client:
            yield client
    
    @pytest.mark.asyncio
    async def test_health_endpoints(self, client):
        """Test health check endpoints."""
        
        # Test liveness
        response = await client.get("/health/live")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "alive"
        
        # Test readiness
        response = await client.get("/health/ready")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ready"
        assert "models_loaded" in data
        assert "memory_usage" in data
    
    @pytest.mark.asyncio
    async def test_root_endpoint(self, client):
        """Test root endpoint."""
        response = await client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["service"] == "embedding-service"
        assert "version" in data
    
    @pytest.mark.asyncio
    async def test_list_models(self, client):
        """Test model listing endpoint."""
        response = await client.get("/v1/models")
        assert response.status_code == 200
        data = response.json()
        assert "object" in data
        assert "data" in data
        assert isinstance(data["data"], list)
    
    @pytest.mark.asyncio
    async def test_embedding_generation(self, client):
        """Test embedding generation with different inputs."""
        
        # Test single string input
        request_data = {
            "model": "e5-large-v2",
            "input": "Hello world"
        }
        
        response = await client.post("/v1/embeddings", json=request_data)
        assert response.status_code == 200
        data = response.json()
        
        assert "object" in data
        assert data["object"] == "list"
        assert "data" in data
        assert len(data["data"]) == 1
        assert "embedding" in data["data"][0]
        assert "index" in data["data"][0]
        assert "object" in data["data"][0]
        
        # Test batch input
        request_data = {
            "model": "e5-large-v2",
            "input": ["Hello world", "How are you?", "Goodbye!"]
        }
        
        response = await client.post("/v1/embeddings", json=request_data)
        assert response.status_code == 200
        data = response.json()
        assert len(data["data"]) == 3
        
        # Verify embeddings have correct dimensions
        for item in data["data"]:
            assert isinstance(item["embedding"], list)
            assert len(item["embedding"]) > 0
    
    @pytest.mark.asyncio
    async def test_embedding_with_input_type(self, client):
        """Test embedding generation with input_type parameter."""
        
        # Test query input type
        request_data = {
            "model": "e5-large-v2",
            "input": "What is machine learning?",
            "input_type": "query"
        }
        
        response = await client.post("/v1/embeddings", json=request_data)
        assert response.status_code == 200
        data = response.json()
        assert len(data["data"]) == 1
        
        # Test passage input type
        request_data = {
            "model": "e5-large-v2",
            "input": "Machine learning is a subset of artificial intelligence.",
            "input_type": "passage"
        }
        
        response = await client.post("/v1/embeddings", json=request_data)
        assert response.status_code == 200
        data = response.json()
        assert len(data["data"]) == 1
    
    @pytest.mark.asyncio
    async def test_embedding_with_model_suffix(self, client):
        """Test embedding generation with model name suffix."""
        
        # Test query suffix
        request_data = {
            "model": "e5-large-v2-query",
            "input": "What is machine learning?"
        }
        
        response = await client.post("/v1/embeddings", json=request_data)
        assert response.status_code == 200
        
        # Test passage suffix
        request_data = {
            "model": "e5-large-v2-passage",
            "input": "Machine learning is a subset of artificial intelligence."
        }
        
        response = await client.post("/v1/embeddings", json=request_data)
        assert response.status_code == 200
    
    @pytest.mark.asyncio
    async def test_embedding_types(self, client):
        """Test different embedding types."""
        
        # Test float (default)
        request_data = {
            "model": "e5-large-v2",
            "input": "Test text",
            "embedding_type": "float"
        }
        
        response = await client.post("/v1/embeddings", json=request_data)
        assert response.status_code == 200
        data = response.json()
        embedding = data["data"][0]["embedding"]
        assert all(isinstance(x, float) for x in embedding)
        
        # Test int8
        request_data["embedding_type"] = "int8"
        response = await client.post("/v1/embeddings", json=request_data)
        assert response.status_code == 200
        data = response.json()
        embedding = data["data"][0]["embedding"]
        assert all(isinstance(x, int) for x in embedding)
        assert all(-128 <= x <= 127 for x in embedding)
    
    @pytest.mark.asyncio
    async def test_error_handling(self, client):
        """Test error handling for invalid requests."""
        
        # Test invalid model
        request_data = {
            "model": "invalid-model",
            "input": "Test text"
        }
        
        response = await client.post("/v1/embeddings", json=request_data)
        assert response.status_code == 400
        assert "error" in response.json()
        
        # Test empty input
        request_data = {
            "model": "e5-large-v2",
            "input": ""
        }
        
        response = await client.post("/v1/embeddings", json=request_data)
        assert response.status_code == 400
        
        # Test invalid embedding type
        request_data = {
            "model": "e5-large-v2",
            "input": "Test text",
            "embedding_type": "invalid_type"
        }
        
        response = await client.post("/v1/embeddings", json=request_data)
        assert response.status_code == 422  # Validation error
    
    @pytest.mark.asyncio
    async def test_model_management(self, client):
        """Test model loading and unloading."""
        
        # Get available models
        response = await client.get("/v1/models")
        assert response.status_code == 200
        models = response.json()["data"]
        
        if models:
            model_id = models[0]["id"]
            
            # Test model loading
            response = await client.post(f"/v1/models/{model_id}/load")
            assert response.status_code in [200, 409]  # 200 if loaded, 409 if already loaded
            
            # Test model info
            response = await client.get(f"/v1/models/{model_id}")
            assert response.status_code == 200
            info = response.json()
            assert info["id"] == model_id
            
            # Test model unloading
            response = await client.post(f"/v1/models/{model_id}/unload")
            assert response.status_code in [200, 409]  # 200 if unloaded, 409 if not loaded
    
    @pytest.mark.asyncio
    async def test_embedding_models_endpoint(self, client):
        """Test embedding-specific model information."""
        response = await client.get("/v1/embeddings/models")
        assert response.status_code == 200
        data = response.json()
        assert "object" in data
        assert "data" in data
        
        for model in data["data"]:
            assert "id" in model
            assert "family" in model
            assert "embedding_dimension" in model
    
    def test_configuration_loading(self):
        """Test configuration file loading."""
        try:
            with open("config/models.yaml", "r") as f:
                config = yaml.safe_load(f)
            
            assert "models" in config
            assert "settings" in config
            assert isinstance(config["models"], dict)
            
            # Check that default model exists
            default_model = config["settings"].get("default_model")
            if default_model:
                assert default_model in config["models"]
                
        except FileNotFoundError:
            pytest.skip("Configuration file not found")


class TestEmbeddingServicePerformance:
    """Performance tests for the embedding service."""
    
    BASE_URL = "http://localhost:8009"
    
    @pytest.fixture
    async def client(self):
        """Create an async HTTP client for testing."""
        async with httpx.AsyncClient(base_url=self.BASE_URL, timeout=30.0) as client:
            yield client
    
    @pytest.mark.asyncio
    async def test_batch_performance(self, client):
        """Test performance with different batch sizes."""
        import time
        
        texts = [f"This is test text number {i}" for i in range(100)]
        
        # Test different batch sizes
        batch_sizes = [1, 5, 10, 25, 50]
        
        for batch_size in batch_sizes:
            batch = texts[:batch_size]
            
            start_time = time.time()
            
            request_data = {
                "model": "e5-large-v2",
                "input": batch
            }
            
            response = await client.post("/v1/embeddings", json=request_data)
            assert response.status_code == 200
            
            end_time = time.time()
            duration = end_time - start_time
            
            data = response.json()
            assert len(data["data"]) == batch_size
            
            print(f"Batch size {batch_size}: {duration:.2f}s ({duration/batch_size:.3f}s per item)")
    
    @pytest.mark.asyncio
    async def test_concurrent_requests(self, client):
        """Test handling of concurrent requests."""
        import asyncio
        
        async def make_request():
            request_data = {
                "model": "e5-large-v2",
                "input": "Concurrent test text"
            }
            response = await client.post("/v1/embeddings", json=request_data)
            return response.status_code == 200
        
        # Run 10 concurrent requests
        tasks = [make_request() for _ in range(10)]
        results = await asyncio.gather(*tasks)
        
        # All requests should succeed
        assert all(results)


if __name__ == "__main__":
    # Run basic tests
    pytest.main([__file__, "-v"])
