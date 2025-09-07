"""Multi-model registry with LRU eviction and memory management (Wave 2)."""

import asyncio
import logging
import time
import torch
from collections import OrderedDict
from typing import Dict, Optional, Tuple, Any
from dataclasses import dataclass
from src.model_manager import ModelManager
from src.config_manager import ConfigManager


@dataclass
class ModelInfo:
    """Information about a loaded model."""
    name: str
    model_manager: ModelManager
    last_used: float
    memory_usage_mb: float
    load_time: float


class MemoryTracker:
    """Track GPU memory usage."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def get_gpu_memory_usage(self) -> Tuple[float, float]:
        """Get current GPU memory usage in MB.
        
        Returns:
            Tuple of (used_mb, total_mb)
        """
        if not torch.cuda.is_available():
            return 0.0, 0.0
        
        try:
            used = torch.cuda.memory_allocated() / 1024 / 1024
            total = torch.cuda.get_device_properties(0).total_memory / 1024 / 1024
            return used, total
        except Exception as e:
            self.logger.warning(f"Failed to get GPU memory usage: {e}")
            return 0.0, 0.0
    
    def estimate_model_memory(self, model_name: str, config: Dict[str, Any]) -> float:
        """Estimate memory usage for a model in MB.
        
        Args:
            model_name: Name of the model
            config: Model configuration
            
        Returns:
            Estimated memory usage in MB
        """
        # Simple heuristic based on model size and dtype
        model_id = config.get('model_id', '')
        torch_dtype = config.get('torch_dtype', 'float16')
        
        # Rough estimates based on common model sizes
        size_estimates = {
            '1b': 2000,   # 1B model ~2GB
            '3b': 6000,   # 3B model ~6GB  
            '7b': 14000,  # 7B model ~14GB
            '8b': 16000,  # 8B model ~16GB
            '13b': 26000, # 13B model ~26GB
        }
        
        # Check model ID for size hints
        model_id_lower = model_id.lower()
        for size_key, memory_mb in size_estimates.items():
            if size_key in model_id_lower:
                # Adjust for dtype
                if torch_dtype == 'float32':
                    return memory_mb * 2
                elif torch_dtype == 'float16':
                    return memory_mb
                elif '8bit' in model_name or 'int8' in model_name:
                    return memory_mb * 0.6
                elif '4bit' in model_name or 'int4' in model_name:
                    return memory_mb * 0.4
                
        # Default fallback
        return 4000  # 4GB default estimate


class ModelRegistry:
    """Multi-model registry with LRU eviction and memory budget management."""
    
    def __init__(self, config_manager: ConfigManager, max_models: int = 3, memory_budget_gb: float = 14.0):
        """Initialize model registry.
        
        Args:
            config_manager: Configuration manager
            max_models: Maximum number of models to keep loaded
            memory_budget_gb: Maximum GPU memory budget in GB
        """
        self.config_manager = config_manager
        self.max_models = max_models
        self.memory_budget_mb = memory_budget_gb * 1024
        self.loaded_models: OrderedDict[str, ModelInfo] = OrderedDict()
        self.memory_tracker = MemoryTracker()
        self.logger = logging.getLogger(__name__)
        
        # Locks for thread safety
        self._load_lock = asyncio.Lock()
        
    async def get_model(self, model_name: str) -> Optional[ModelManager]:
        """Get a model, loading if necessary and managing memory.
        
        Args:
            model_name: Name of the model to get
            
        Returns:
            ModelManager instance or None if failed
        """
        async with self._load_lock:
            # Check if model is already loaded
            if model_name in self.loaded_models:
                # Update LRU order
                model_info = self.loaded_models[model_name]
                model_info.last_used = time.time()
                
                # Move to end (most recently used)
                self.loaded_models.move_to_end(model_name)
                
                self.logger.info(f"Model {model_name} retrieved from cache")
                return model_info.model_manager
            
            # Need to load new model
            return await self._load_model(model_name)
    
    async def _load_model(self, model_name: str) -> Optional[ModelManager]:
        """Load a new model with memory management.
        
        Args:
            model_name: Name of the model to load
            
        Returns:
            ModelManager instance or None if failed
        """
        # Check if model exists in config
        available_models = self.config_manager.get_available_models()
        if model_name not in available_models:
            self.logger.error(f"Model {model_name} not found in configuration")
            return None
        
        # Get the full model configuration
        model_config = self.config_manager.get_model_config(model_name)
        
        # Estimate memory requirements
        estimated_memory = self.memory_tracker.estimate_model_memory(model_name, model_config)
        self.logger.info(f"Estimated memory for {model_name}: {estimated_memory:.1f}MB")
        
        # Check if we need to evict models to make space
        await self._ensure_memory_budget(estimated_memory)
        
        # Create new model manager and load
        start_time = time.time()
        model_manager = ModelManager(self.config_manager)
        
        self.logger.info(f"Loading model {model_name}...")
        success = await model_manager.async_load_model(model_name)
        
        if not success:
            self.logger.error(f"Failed to load model {model_name}")
            return None
        
        load_time = time.time() - start_time
        
        # Get actual memory usage after loading
        used_memory, _ = self.memory_tracker.get_gpu_memory_usage()
        
        # Create model info
        model_info = ModelInfo(
            name=model_name,
            model_manager=model_manager,
            last_used=time.time(),
            memory_usage_mb=estimated_memory,  # Use estimate for now
            load_time=load_time
        )
        
        # Add to registry
        self.loaded_models[model_name] = model_info
        
        self.logger.info(f"Model {model_name} loaded successfully in {load_time:.2f}s")
        self.logger.info(f"Registry now has {len(self.loaded_models)} models loaded")
        
        return model_manager
    
    async def _ensure_memory_budget(self, required_memory_mb: float):
        """Ensure there's enough memory budget by evicting LRU models if needed.
        
        Args:
            required_memory_mb: Memory required for new model
        """
        # Calculate current memory usage
        current_usage = sum(info.memory_usage_mb for info in self.loaded_models.values())
        
        # Check if we need to evict
        while (current_usage + required_memory_mb > self.memory_budget_mb or 
               len(self.loaded_models) >= self.max_models):
            
            if not self.loaded_models:
                break
                
            # Evict least recently used model
            lru_name, lru_info = next(iter(self.loaded_models.items()))
            
            self.logger.info(f"Evicting LRU model {lru_name} to free memory")
            await self._evict_model(lru_name)
            
            current_usage = sum(info.memory_usage_mb for info in self.loaded_models.values())
    
    async def _evict_model(self, model_name: str):
        """Evict a model from the registry.
        
        Args:
            model_name: Name of the model to evict
        """
        if model_name not in self.loaded_models:
            return
        
        model_info = self.loaded_models[model_name]
        
        # Unload the model
        model_info.model_manager.unload_model()
        
        # Remove from registry
        del self.loaded_models[model_name]
        
        self.logger.info(f"Model {model_name} evicted from registry")
    
    def get_registry_stats(self) -> Dict[str, Any]:
        """Get statistics about the model registry.
        
        Returns:
            Dictionary with registry statistics
        """
        used_memory, total_memory = self.memory_tracker.get_gpu_memory_usage()
        estimated_usage = sum(info.memory_usage_mb for info in self.loaded_models.values())
        
        return {
            "loaded_models": len(self.loaded_models),
            "max_models": self.max_models,
            "memory_budget_mb": self.memory_budget_mb,
            "estimated_memory_usage_mb": estimated_usage,
            "actual_gpu_usage_mb": used_memory,
            "total_gpu_memory_mb": total_memory,
            "memory_utilization": used_memory / total_memory if total_memory > 0 else 0,
            "models": [
                {
                    "name": info.name,
                    "last_used": info.last_used,
                    "memory_usage_mb": info.memory_usage_mb,
                    "load_time": info.load_time
                }
                for info in self.loaded_models.values()
            ]
        }
    
    async def preload_models(self, model_names: list):
        """Preload multiple models.
        
        Args:
            model_names: List of model names to preload
        """
        for model_name in model_names:
            try:
                await self.get_model(model_name)
            except Exception as e:
                self.logger.error(f"Failed to preload model {model_name}: {e}")
    
    async def clear_registry(self):
        """Clear all models from registry."""
        async with self._load_lock:
            for model_name in list(self.loaded_models.keys()):
                await self._evict_model(model_name)
            
            self.logger.info("Model registry cleared")
