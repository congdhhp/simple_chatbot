"""Configuration management for the embedding service."""

import os
import yaml
import logging
from typing import Dict, Any, List, Optional
from pathlib import Path


class ConfigManager:
    """Manages configuration for the embedding service."""
    
    def __init__(self, config_path: str = "config/models.yaml"):
        """Initialize the configuration manager.
        
        Args:
            config_path: Path to the YAML configuration file
        """
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.logger = logging.getLogger(__name__)
        
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file.
        
        Returns:
            Dictionary containing the configuration
            
        Raises:
            FileNotFoundError: If config file doesn't exist
            yaml.YAMLError: If config file is invalid YAML
        """
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
            
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
                
            # Validate configuration structure
            if not isinstance(config, dict):
                raise ValueError("Configuration must be a dictionary")
                
            if 'models' not in config:
                raise ValueError("Configuration must contain 'models' section")
                
            return config
            
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"Invalid YAML in configuration file: {e}")
    
    def get_available_models(self) -> Dict[str, str]:
        """Get list of available models.
        
        Returns:
            Dictionary mapping model names to display names
        """
        models = {}
        for name, config in self.config.get('models', {}).items():
            display_name = config.get('display_name', name)
            models[name] = display_name
        return models
    
    def get_model_config(self, model_name: str) -> Dict[str, Any]:
        """Get configuration for a specific model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Model configuration dictionary
            
        Raises:
            KeyError: If model is not found
        """
        if model_name not in self.config['models']:
            available = list(self.config['models'].keys())
            raise KeyError(f"Model '{model_name}' not found. Available: {available}")
            
        return self.config['models'][model_name].copy()
    
    def get_model_family(self, model_name: str) -> str:
        """Get the family of a model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Model family string
        """
        config = self.get_model_config(model_name)
        return config.get('family', 'unknown')
    
    def supports_input_type(self, model_name: str) -> bool:
        """Check if model supports input_type parameter.
        
        Args:
            model_name: Name of the model
            
        Returns:
            True if model supports input_type
        """
        config = self.get_model_config(model_name)
        return config.get('supports_input_type', False)
    
    def get_supported_embedding_types(self, model_name: str) -> List[str]:
        """Get supported embedding types for a model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            List of supported embedding types
        """
        config = self.get_model_config(model_name)
        return config.get('supported_embedding_types', ['float'])
    
    def get_supported_dimensions(self, model_name: str) -> List[int]:
        """Get supported dimensions for a model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            List of supported dimension sizes
        """
        config = self.get_model_config(model_name)
        return config.get('supports_dimensions', [])
    
    def get_supported_modalities(self, model_name: str) -> List[str]:
        """Get supported modalities for a model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            List of supported modalities
        """
        config = self.get_model_config(model_name)
        return config.get('supported_modalities', ['text'])
    
    def get_embedding_dimension(self, model_name: str) -> int:
        """Get the default embedding dimension for a model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Default embedding dimension
        """
        config = self.get_model_config(model_name)
        return config.get('embedding_dimension', 768)
    
    def get_max_seq_length(self, model_name: str) -> int:
        """Get maximum sequence length for a model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Maximum sequence length
        """
        config = self.get_model_config(model_name)
        return config.get('max_seq_length', 512)
    
    def get_model_settings(self, model_name: str) -> Dict[str, Any]:
        """Get model-specific settings.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Model settings dictionary
        """
        config = self.get_model_config(model_name)
        return config.get('settings', {})
    
    def get_default_model(self) -> str:
        """Get the default model name.
        
        Returns:
            Default model name
        """
        return self.config.get('service', {}).get('default_model', 'e5-large-v2')
    
    def get_service_config(self) -> Dict[str, Any]:
        """Get service-level configuration.
        
        Returns:
            Service configuration dictionary
        """
        return self.config.get('service', {})
    
    def get_max_batch_size(self) -> int:
        """Get maximum batch size for the service.
        
        Returns:
            Maximum batch size
        """
        return self.get_service_config().get('max_batch_size', 64)
    
    def get_max_input_length(self) -> int:
        """Get maximum input length for the service.
        
        Returns:
            Maximum input length in characters
        """
        return self.get_service_config().get('max_input_length', 8192)
    
    def is_dynamic_batching_enabled(self) -> bool:
        """Check if dynamic batching is enabled.
        
        Returns:
            True if dynamic batching is enabled
        """
        return self.get_service_config().get('enable_dynamic_batching', True)
    
    def get_batch_timeout_ms(self) -> int:
        """Get batch timeout in milliseconds.
        
        Returns:
            Batch timeout in milliseconds
        """
        return self.get_service_config().get('batch_timeout_ms', 100)
    
    def parse_model_name(self, model_name: str) -> tuple[str, Optional[str]]:
        """Parse model name to extract base name and input type suffix.
        
        Args:
            model_name: Model name (potentially with -query or -passage suffix)
            
        Returns:
            Tuple of (base_model_name, input_type)
            
        Examples:
            "e5-large-v2" -> ("e5-large-v2", None)
            "e5-large-v2-query" -> ("e5-large-v2", "query")
            "e5-large-v2-passage" -> ("e5-large-v2", "passage")
        """
        if model_name.endswith('-query'):
            return model_name[:-6], 'query'
        elif model_name.endswith('-passage'):
            return model_name[:-8], 'passage'
        else:
            return model_name, None
    
    def reload_config(self) -> None:
        """Reload configuration from file."""
        self.config = self._load_config()
        self.logger.info("Configuration reloaded")
    
    def validate_model_config(self, model_name: str) -> List[str]:
        """Validate model configuration and return any issues.
        
        Args:
            model_name: Name of the model to validate
            
        Returns:
            List of validation issues (empty if valid)
        """
        issues = []
        
        try:
            config = self.get_model_config(model_name)
        except KeyError:
            return [f"Model '{model_name}' not found in configuration"]
        
        # Required fields
        required_fields = ['model_id', 'family', 'embedding_dimension']
        for field in required_fields:
            if field not in config:
                issues.append(f"Missing required field '{field}' for model '{model_name}'")
        
        # Validate family
        valid_families = ['e5', 'gte', 'sentence_transformers', 'nv_embed']
        family = config.get('family')
        if family and family not in valid_families:
            issues.append(f"Invalid family '{family}' for model '{model_name}'. Valid: {valid_families}")
        
        # Validate embedding types
        valid_embedding_types = ['float', 'int8', 'uint8', 'binary', 'ubinary']
        embedding_types = config.get('supported_embedding_types', [])
        for et in embedding_types:
            if et not in valid_embedding_types:
                issues.append(f"Invalid embedding type '{et}' for model '{model_name}'")
        
        # Validate modalities
        valid_modalities = ['text', 'image', 'text_image']
        modalities = config.get('supported_modalities', [])
        for mod in modalities:
            if mod not in valid_modalities:
                issues.append(f"Invalid modality '{mod}' for model '{model_name}'")
        
        return issues
