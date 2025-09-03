"""Model manager for the embedding service."""

import logging
import torch
from typing import Dict, Any, Optional, Type
from pathlib import Path

from .config_manager import ConfigManager
from .models import (
    BaseEmbeddingModel,
    E5Model,
    GTEModel,
    SentenceTransformersModel,
    NVEmbedModel
)


class ModelManager:
    """Manages embedding models for the service."""
    
    # Model family to class mapping
    MODEL_FAMILIES = {
        'e5': E5Model,
        'gte': GTEModel,
        'sentence_transformers': SentenceTransformersModel,
        'nv_embed': NVEmbedModel
    }
    
    def __init__(self, config_manager: ConfigManager):
        """Initialize the model manager.
        
        Args:
            config_manager: Configuration manager instance
        """
        self.config_manager = config_manager
        self.current_model: Optional[BaseEmbeddingModel] = None
        self.current_model_name: Optional[str] = None
        self.logger = logging.getLogger(__name__)
        
        # Device management with CUDA error handling
        if torch.cuda.is_available():
            try:
                # Test CUDA availability with a simple operation
                test_tensor = torch.tensor([1.0]).cuda()
                self.device = 'cuda'
                self.logger.info(f"CUDA available and working: {torch.cuda.get_device_name()}")
            except Exception as cuda_error:
                self.logger.warning(f"CUDA available but not working: {cuda_error}")
                self.logger.info("Falling back to CPU mode")
                self.device = 'cpu'
        else:
            self.device = 'cpu'
            self.logger.info("CUDA not available, using CPU for inference")
    
    def load_model(self, model_name: str) -> bool:
        """Load a specific model.
        
        Args:
            model_name: Name of the model to load
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Parse model name to handle -query/-passage suffixes
            base_model_name, _ = self.config_manager.parse_model_name(model_name)
            
            # If same model is already loaded, return success
            if self.current_model_name == base_model_name and self.current_model and self.current_model.is_loaded:
                self.logger.info(f"Model {base_model_name} already loaded")
                return True
            
            # Unload current model if any
            if self.current_model:
                self.unload_model()
            
            # Get model configuration
            model_config = self.config_manager.get_model_config(base_model_name)
            family = model_config['family']
            
            # Get model class
            if family not in self.MODEL_FAMILIES:
                available_families = ', '.join(self.MODEL_FAMILIES.keys())
                raise ValueError(f"Unsupported model family '{family}'. Available: {available_families}")
            
            model_class = self.MODEL_FAMILIES[family]
            
            # Create and load model
            self.logger.info(f"Loading model {base_model_name} (family: {family})")
            self.current_model = model_class(model_config)
            
            if self.current_model.load_model():
                self.current_model_name = base_model_name
                self.logger.info(f"Model {base_model_name} loaded successfully")
                return True
            else:
                self.current_model = None
                self.current_model_name = None
                self.logger.error(f"Failed to load model {base_model_name}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error loading model {model_name}: {str(e)}")
            self.current_model = None
            self.current_model_name = None
            return False
    
    def unload_model(self) -> None:
        """Unload the current model."""
        if self.current_model:
            try:
                self.current_model.unload_model()
                self.logger.info(f"Model {self.current_model_name} unloaded")
            except Exception as e:
                self.logger.error(f"Error unloading model: {str(e)}")
            finally:
                self.current_model = None
                self.current_model_name = None
    
    def get_current_model(self) -> Optional[BaseEmbeddingModel]:
        """Get the current loaded model.
        
        Returns:
            Current model instance or None
        """
        return self.current_model
    
    def get_current_model_name(self) -> Optional[str]:
        """Get the current model name.
        
        Returns:
            Current model name or None
        """
        return self.current_model_name
    
    def is_model_loaded(self, model_name: Optional[str] = None) -> bool:
        """Check if a model is loaded.
        
        Args:
            model_name: Model name to check (None for any model)
            
        Returns:
            True if specified model (or any model) is loaded
        """
        if not self.current_model or not self.current_model.is_loaded:
            return False
            
        if model_name is None:
            return True
            
        # Parse model name to handle suffixes
        base_model_name, _ = self.config_manager.parse_model_name(model_name)
        return self.current_model_name == base_model_name
    
    def get_model_info(self, model_name: Optional[str] = None) -> Dict[str, Any]:
        """Get model information.
        
        Args:
            model_name: Model name (None for current model)
            
        Returns:
            Dictionary with model information
        """
        if model_name is None:
            # Get current model info
            if self.current_model:
                return self.current_model.get_model_info()
            else:
                return {
                    'status': 'No model loaded',
                    'available_models': list(self.config_manager.get_available_models().keys())
                }
        else:
            # Get info for specific model
            try:
                base_model_name, _ = self.config_manager.parse_model_name(model_name)
                model_config = self.config_manager.get_model_config(base_model_name)
                
                info = {
                    'model_id': model_config['model_id'],
                    'display_name': model_config.get('display_name', base_model_name),
                    'family': model_config['family'],
                    'is_loaded': self.is_model_loaded(base_model_name),
                    'max_seq_length': model_config.get('max_seq_length', 512),
                    'embedding_dimension': model_config['embedding_dimension'],
                    'supports_input_type': model_config.get('supports_input_type', False),
                    'supported_embedding_types': model_config.get('supported_embedding_types', ['float']),
                    'supported_modalities': model_config.get('supported_modalities', ['text'])
                }
                
                if self.is_model_loaded(base_model_name) and self.current_model:
                    info['memory_usage'] = self.current_model.get_memory_usage()
                
                return info
                
            except KeyError:
                return {
                    'error': f'Model {model_name} not found',
                    'available_models': list(self.config_manager.get_available_models().keys())
                }
    
    def get_available_models(self) -> Dict[str, str]:
        """Get list of available models.
        
        Returns:
            Dictionary mapping model names to display names
        """
        return self.config_manager.get_available_models()
    
    def validate_model_request(self, model_name: str, input_type: Optional[str] = None) -> tuple[str, Optional[str]]:
        """Validate and normalize a model request.
        
        Args:
            model_name: Requested model name (may include suffixes)
            input_type: Requested input type
            
        Returns:
            Tuple of (normalized_model_name, effective_input_type)
            
        Raises:
            ValueError: If model or parameters are invalid
        """
        # Parse model name to extract base name and suffix
        base_model_name, suffix_input_type = self.config_manager.parse_model_name(model_name)
        
        # Check if base model exists
        if base_model_name not in self.config_manager.get_available_models():
            available = ', '.join(self.config_manager.get_available_models().keys())
            raise ValueError(f"Model '{base_model_name}' not found. Available: {available}")
        
        # Determine effective input type
        effective_input_type = input_type or suffix_input_type
        
        # Validate input_type for this model
        if self.config_manager.supports_input_type(base_model_name):
            if effective_input_type is None:
                raise ValueError(f"Model '{base_model_name}' requires input_type parameter or model suffix (-query/-passage)")
            if effective_input_type not in ['query', 'passage']:
                raise ValueError(f"Invalid input_type '{effective_input_type}'. Must be 'query' or 'passage'")
        else:
            if effective_input_type is not None:
                self.logger.warning(f"Model '{base_model_name}' ignores input_type parameter")
                effective_input_type = None
        
        return base_model_name, effective_input_type
    
    def ensure_model_loaded(self, model_name: str) -> bool:
        """Ensure a specific model is loaded.
        
        Args:
            model_name: Model name to ensure is loaded
            
        Returns:
            True if model is loaded successfully
        """
        base_model_name, _ = self.config_manager.parse_model_name(model_name)
        
        if not self.is_model_loaded(base_model_name):
            return self.load_model(base_model_name)
        
        return True
    
    def get_gpu_memory_info(self) -> Dict[str, float]:
        """Get GPU memory information.
        
        Returns:
            Dictionary with GPU memory stats in GB
        """
        if torch.cuda.is_available():
            return {
                'total': torch.cuda.get_device_properties(0).total_memory / 1024**3,
                'allocated': torch.cuda.memory_allocated() / 1024**3,
                'reserved': torch.cuda.memory_reserved() / 1024**3,
                'free': (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_reserved()) / 1024**3
            }
        else:
            return {'total': 0.0, 'allocated': 0.0, 'reserved': 0.0, 'free': 0.0}
    
    def list_model_families(self) -> Dict[str, Type[BaseEmbeddingModel]]:
        """Get available model families.
        
        Returns:
            Dictionary mapping family names to model classes
        """
        return self.MODEL_FAMILIES.copy()
    
    def __del__(self):
        """Cleanup when manager is destroyed."""
        try:
            self.unload_model()
        except Exception:
            pass  # Ignore errors during cleanup
