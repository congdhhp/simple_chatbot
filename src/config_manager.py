"""Configuration management for the CLI chatbot."""

import os
import yaml
from typing import Dict, Any, Optional
from pathlib import Path


class ConfigManager:
    """Manages configuration loading and validation for the chatbot."""
    
    def __init__(self, config_path: str = "config/models.yaml"):
        """Initialize the configuration manager.
        
        Args:
            config_path: Path to the configuration file
        """
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self._validate_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        
        with open(self.config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def _validate_config(self) -> None:
        """Validate the loaded configuration."""
        required_keys = ['models', 'default_model', 'settings']
        for key in required_keys:
            if key not in self.config:
                raise ValueError(f"Missing required configuration key: {key}")
        
        # Validate default model exists
        default_model = self.config['default_model']
        if default_model not in self.config['models']:
            raise ValueError(f"Default model '{default_model}' not found in models configuration")
        
        # Validate each model configuration
        for model_name, model_config in self.config['models'].items():
            required_model_keys = ['model_id', 'display_name', 'generation_config']
            for key in required_model_keys:
                if key not in model_config:
                    raise ValueError(f"Missing required key '{key}' in model '{model_name}'")
    
    def get_model_config(self, model_name: Optional[str] = None) -> Dict[str, Any]:
        """Get configuration for a specific model.
        
        Args:
            model_name: Name of the model. If None, returns default model config.
            
        Returns:
            Model configuration dictionary
        """
        if model_name is None:
            model_name = self.config['default_model']
        
        if model_name not in self.config['models']:
            raise ValueError(f"Model '{model_name}' not found in configuration")
        
        return self.config['models'][model_name]
    
    def get_available_models(self) -> Dict[str, str]:
        """Get list of available models with their display names.
        
        Returns:
            Dictionary mapping model names to display names
        """
        return {
            name: config['display_name'] 
            for name, config in self.config['models'].items()
        }
    
    def get_settings(self) -> Dict[str, Any]:
        """Get global settings."""
        return self.config['settings']
    
    def get_default_model(self) -> str:
        """Get the default model name."""
        return self.config['default_model']
    
    def update_model_config(self, model_name: str, updates: Dict[str, Any]) -> None:
        """Update configuration for a specific model.
        
        Args:
            model_name: Name of the model to update
            updates: Dictionary of updates to apply
        """
        if model_name not in self.config['models']:
            raise ValueError(f"Model '{model_name}' not found in configuration")
        
        self.config['models'][model_name].update(updates)
        self._save_config()
    
    def _save_config(self) -> None:
        """Save current configuration back to file."""
        with open(self.config_path, 'w', encoding='utf-8') as f:
            yaml.dump(self.config, f, default_flow_style=False, indent=2)
