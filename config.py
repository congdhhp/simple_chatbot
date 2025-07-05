"""
Configuration system for the LLM Chatbot
Allows easy switching between different models and settings
"""

import json
import os
from typing import Dict, Any, List
import logging

logger = logging.getLogger(__name__)

class ChatbotConfig:
    """
    Configuration manager for the chatbot
    """
    
    def __init__(self, config_file: str = "chatbot_config.json"):
        """
        Initialize configuration manager
        
        Args:
            config_file (str): Path to configuration file
        """
        self.config_file = config_file
        self.config = self._load_default_config()
        self.load_config()
    
    def _load_default_config(self) -> Dict[str, Any]:
        """
        Load default configuration
        
        Returns:
            dict: Default configuration
        """
        return {
            "current_model": "openai-community/gpt2-large",
            "use_gpu": True,
            "generation_params": {
                "max_new_tokens": 50,
                "temperature": 0.8,
                "top_p": 0.9,
                "do_sample": True
            },
            "available_models": {
                "gpt2-large": {
                    "name": "openai-community/gpt2-large",
                    "description": "GPT-2 Large model (774M parameters)",
                    "type": "causal_lm"
                },
                "gpt2-medium": {
                    "name": "openai-community/gpt2-medium",
                    "description": "GPT-2 Medium model (355M parameters)",
                    "type": "causal_lm"
                },
                "gpt2": {
                    "name": "openai-community/gpt2",
                    "description": "GPT-2 Base model (124M parameters)",
                    "type": "causal_lm"
                },
                "distilgpt2": {
                    "name": "distilgpt2",
                    "description": "DistilGPT-2 (82M parameters, faster)",
                    "type": "causal_lm"
                },
                "llama2-7b": {
                    "name": "meta-llama/Llama-2-7b-chat-hf",
                    "description": "LLaMA 2 7B Chat model (requires access token)",
                    "type": "causal_lm"
                },
                "mistral-7b": {
                    "name": "mistralai/Mistral-7B-Instruct-v0.1",
                    "description": "Mistral 7B Instruct model",
                    "type": "causal_lm"
                },
                "phi-2": {
                    "name": "microsoft/phi-2",
                    "description": "Microsoft Phi-2 (2.7B parameters)",
                    "type": "causal_lm"
                },
                "codegen": {
                    "name": "Salesforce/codegen-350M-mono",
                    "description": "CodeGen model for code generation",
                    "type": "causal_lm"
                }
            },
            "chat_settings": {
                "max_history": 10,
                "show_model_info": True,
                "save_conversations": False,
                "conversation_file": "conversations.json"
            }
        }
    
    def load_config(self) -> bool:
        """
        Load configuration from file
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    loaded_config = json.load(f)
                    # Merge with default config to ensure all keys exist
                    self._merge_config(loaded_config)
                logger.info(f"Configuration loaded from {self.config_file}")
                return True
            else:
                logger.info("No config file found, using default configuration")
                self.save_config()
                return True
        except Exception as e:
            logger.error(f"Error loading config: {str(e)}")
            return False
    
    def save_config(self) -> bool:
        """
        Save current configuration to file
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=2, ensure_ascii=False)
            logger.info(f"Configuration saved to {self.config_file}")
            return True
        except Exception as e:
            logger.error(f"Error saving config: {str(e)}")
            return False
    
    def _merge_config(self, loaded_config: Dict[str, Any]):
        """
        Merge loaded config with default config
        
        Args:
            loaded_config (dict): Configuration loaded from file
        """
        def merge_dicts(default: dict, loaded: dict) -> dict:
            for key, value in loaded.items():
                if key in default and isinstance(default[key], dict) and isinstance(value, dict):
                    merge_dicts(default[key], value)
                else:
                    default[key] = value
            return default
        
        self.config = merge_dicts(self.config, loaded_config)
    
    def get_current_model(self) -> str:
        """
        Get current model name
        
        Returns:
            str: Current model name
        """
        return self.config["current_model"]
    
    def set_current_model(self, model_name: str) -> bool:
        """
        Set current model
        
        Args:
            model_name (str): Model name or alias
            
        Returns:
            bool: True if successful, False otherwise
        """
        # Check if it's an alias
        if model_name in self.config["available_models"]:
            actual_model_name = self.config["available_models"][model_name]["name"]
        else:
            actual_model_name = model_name
        
        self.config["current_model"] = actual_model_name
        return self.save_config()
    
    def add_model(self, alias: str, model_name: str, description: str = "", model_type: str = "causal_lm") -> bool:
        """
        Add a new model to available models
        
        Args:
            alias (str): Short alias for the model
            model_name (str): Full model name or path
            description (str): Model description
            model_type (str): Type of model
            
        Returns:
            bool: True if successful, False otherwise
        """
        self.config["available_models"][alias] = {
            "name": model_name,
            "description": description,
            "type": model_type
        }
        return self.save_config()
    
    def get_available_models(self) -> Dict[str, Dict[str, str]]:
        """
        Get list of available models
        
        Returns:
            dict: Available models
        """
        return self.config["available_models"]
    
    def get_generation_params(self) -> Dict[str, Any]:
        """
        Get generation parameters
        
        Returns:
            dict: Generation parameters
        """
        return self.config["generation_params"]
    
    def update_generation_params(self, **kwargs) -> bool:
        """
        Update generation parameters
        
        Args:
            **kwargs: Parameters to update
            
        Returns:
            bool: True if successful, False otherwise
        """
        self.config["generation_params"].update(kwargs)
        return self.save_config()
    
    def get_chat_settings(self) -> Dict[str, Any]:
        """
        Get chat settings
        
        Returns:
            dict: Chat settings
        """
        return self.config["chat_settings"]
    
    def update_chat_settings(self, **kwargs) -> bool:
        """
        Update chat settings
        
        Args:
            **kwargs: Settings to update
            
        Returns:
            bool: True if successful, False otherwise
        """
        self.config["chat_settings"].update(kwargs)
        return self.save_config()
    
    def use_gpu(self) -> bool:
        """
        Check if GPU should be used
        
        Returns:
            bool: True if GPU should be used
        """
        return self.config.get("use_gpu", True)
    
    def set_use_gpu(self, use_gpu: bool) -> bool:
        """
        Set GPU usage preference
        
        Args:
            use_gpu (bool): Whether to use GPU
            
        Returns:
            bool: True if successful, False otherwise
        """
        self.config["use_gpu"] = use_gpu
        return self.save_config()
