"""Model management for the CLI chatbot."""

import os
import torch
import logging
from typing import Dict, Any, Optional, Tuple
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    GenerationConfig,
    BitsAndBytesConfig
)
try:
    from peft import PeftModel, PeftConfig
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
from config_manager import ConfigManager


class ModelManager:
    """Manages LLM models for the chatbot."""
    
    def __init__(self, config_manager: ConfigManager):
        """Initialize the model manager.
        
        Args:
            config_manager: Configuration manager instance
        """
        self.config_manager = config_manager
        self.current_model = None
        self.current_tokenizer = None
        self.current_model_name = None
        self.generation_config = None
        
        # Setup logging
        self.logger = logging.getLogger(__name__)

        # Cache directory will be set when loading a model
        self.use_hf_cache = True  # Default value, will be updated per model
        self.cache_dir = None
        
        # Check CUDA availability
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.device == "cuda":
            self.logger.info(f"CUDA available. GPU: {torch.cuda.get_device_name()}")
            self.logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            self.logger.warning("CUDA not available. Using CPU.")

    def _is_lora_adapter(self, path: str) -> bool:
        """Check if a path contains a LoRA adapter.

        Args:
            path: Path to check

        Returns:
            True if path contains LoRA adapter files
        """
        from pathlib import Path
        adapter_path = Path(path)

        # Check for LoRA adapter files
        adapter_config = adapter_path / "adapter_config.json"
        adapter_model = adapter_path / "adapter_model.safetensors"

        return adapter_config.exists() and adapter_model.exists()
    
    def load_model(self, model_name: Optional[str] = None) -> bool:
        """Load a model and tokenizer.
        
        Args:
            model_name: Name of the model to load. If None, loads default model.
            
        Returns:
            True if model loaded successfully, False otherwise
        """
        try:
            if model_name is None:
                model_name = self.config_manager.get_default_model()
            
            # Get model configuration and settings
            model_config = self.config_manager.get_model_config(model_name)
            model_settings = self.config_manager.get_settings(model_name)
            model_id = model_config['model_id']

            # Check if this is a LoRA adapter
            is_lora = self._is_lora_adapter(model_id)
            lora_adapter_path = None
            base_model_id = model_id

            if is_lora:
                if not PEFT_AVAILABLE:
                    raise ImportError("peft library is required for LoRA adapters. Install with: pip install peft")

                # Read the adapter config to get the base model
                import json
                from pathlib import Path
                adapter_config_path = Path(model_id) / "adapter_config.json"
                with open(adapter_config_path, 'r') as f:
                    adapter_config = json.load(f)

                base_model_id = adapter_config['base_model_name_or_path']
                lora_adapter_path = model_id
                self.logger.info(f"Detected LoRA adapter. Base model: {base_model_id}, Adapter: {lora_adapter_path}")

            # Check for lora_adapter_path in settings (alternative way to specify LoRA)
            elif 'lora_adapter_path' in model_settings:
                lora_adapter_path = model_settings['lora_adapter_path']
                if self._is_lora_adapter(lora_adapter_path):
                    if not PEFT_AVAILABLE:
                        raise ImportError("peft library is required for LoRA adapters. Install with: pip install peft")

                    # Read the adapter config to get the base model
                    import json
                    from pathlib import Path
                    adapter_config_path = Path(lora_adapter_path) / "adapter_config.json"
                    with open(adapter_config_path, 'r') as f:
                        adapter_config = json.load(f)

                    # Verify base model matches
                    expected_base = adapter_config['base_model_name_or_path']
                    if base_model_id != expected_base:
                        self.logger.warning(f"Base model mismatch. Config: {base_model_id}, Adapter expects: {expected_base}")
                        base_model_id = expected_base

                    self.logger.info(f"Using LoRA adapter from settings. Base model: {base_model_id}, Adapter: {lora_adapter_path}")

            # Setup cache directory based on model settings
            self.use_hf_cache = model_settings.get('use_hf_cache', True)
            if self.use_hf_cache:
                # Use Hugging Face default cache (shared across projects)
                self.cache_dir = None  # Let HF use default cache
                self.logger.info("Using Hugging Face default cache (~/.cache/huggingface)")
            else:
                # Use local cache directory
                from pathlib import Path
                self.cache_dir = Path(model_settings.get('cache_dir', 'model_cache'))
                self.cache_dir.mkdir(exist_ok=True)
                self.logger.info(f"Using local cache directory: {self.cache_dir}")

            self.logger.info(f"Loading model: {model_config['display_name']} ({base_model_id})")
            
            # Setup quantization if needed (for memory efficiency)
            quantization_config = None
            if self.device == "cuda" and torch.cuda.get_device_properties(0).total_memory < 8e9:
                # Use 4-bit quantization for GPUs with less than 8GB memory
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
                self.logger.info("Using 4-bit quantization for memory efficiency")
            
            # Load tokenizer (always from base model)
            self.logger.info("Loading tokenizer...")
            tokenizer_kwargs = {'trust_remote_code': True}
            if not self.use_hf_cache and self.cache_dir:
                tokenizer_kwargs['cache_dir'] = str(self.cache_dir)

            tokenizer = AutoTokenizer.from_pretrained(base_model_id, **tokenizer_kwargs)
            
            # Set pad token if not present
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            # Load base model
            self.logger.info("Loading base model...")
            model_kwargs = {
                'trust_remote_code': True,
                'device_map': 'auto' if self.device == "cuda" else None,
            }

            # Add cache_dir only if not using HF default cache
            if not self.use_hf_cache and self.cache_dir:
                model_kwargs['cache_dir'] = str(self.cache_dir)

            # Add torch dtype if specified
            if 'torch_dtype' in model_config:
                dtype_str = model_config['torch_dtype']
                if dtype_str == 'float16':
                    model_kwargs['torch_dtype'] = torch.float16
                elif dtype_str == 'bfloat16':
                    model_kwargs['torch_dtype'] = torch.bfloat16

            # Add quantization config if available
            if quantization_config is not None:
                model_kwargs['quantization_config'] = quantization_config

            model = AutoModelForCausalLM.from_pretrained(base_model_id, **model_kwargs)

            # Load LoRA adapter if specified
            if lora_adapter_path:
                self.logger.info(f"Loading LoRA adapter from: {lora_adapter_path}")
                model = PeftModel.from_pretrained(model, lora_adapter_path)
                self.logger.info("LoRA adapter loaded successfully")

            # Move to device if not using device_map
            if self.device == "cuda" and 'device_map' not in model_kwargs:
                model = model.to(self.device)
            
            # Setup generation config
            gen_config = model_config.get('generation_config', {})
            self.generation_config = GenerationConfig(**gen_config)
            
            # Store loaded components
            self.current_model = model
            self.current_tokenizer = tokenizer
            self.current_model_name = model_name
            
            self.logger.info(f"Model loaded successfully: {model_config['display_name']}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load model {model_name}: {str(e)}")
            return False
    
    def generate_response(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Generate a response using the current model.
        
        Args:
            prompt: User input prompt
            system_prompt: Optional system prompt override
            
        Returns:
            Generated response text
        """
        if self.current_model is None or self.current_tokenizer is None:
            raise RuntimeError("No model loaded. Call load_model() first.")
        
        try:
            # Get system prompt from config if not provided
            if system_prompt is None:
                model_config = self.config_manager.get_model_config(self.current_model_name)
                system_prompt = model_config.get('system_prompt', '')
            
            # Format the conversation
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            
            # Apply chat template
            formatted_prompt = self.current_tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            
            # Tokenize
            inputs = self.current_tokenizer(
                formatted_prompt, 
                return_tensors="pt", 
                padding=True, 
                truncation=True
            )
            
            # Move inputs to device
            inputs = {k: v.to(self.current_model.device) for k, v in inputs.items()}
            
            # Generate
            with torch.no_grad():
                outputs = self.current_model.generate(
                    **inputs,
                    generation_config=self.generation_config,
                    pad_token_id=self.current_tokenizer.eos_token_id
                )
            
            # Decode response
            response = self.current_tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:], 
                skip_special_tokens=True
            )
            
            return response.strip()
            
        except Exception as e:
            self.logger.error(f"Failed to generate response: {str(e)}")
            return f"Error generating response: {str(e)}"
    
    def get_current_model_info(self) -> Dict[str, Any]:
        """Get information about the currently loaded model.
        
        Returns:
            Dictionary with model information
        """
        if self.current_model_name is None:
            return {"status": "No model loaded"}
        
        model_config = self.config_manager.get_model_config(self.current_model_name)
        return {
            "name": self.current_model_name,
            "display_name": model_config['display_name'],
            "model_id": model_config['model_id'],
            "description": model_config.get('description', ''),
            "device": str(self.current_model.device) if self.current_model else "Unknown",
            "status": "Loaded"
        }
    
    def unload_model(self) -> None:
        """Unload the current model to free memory."""
        if self.current_model is not None:
            del self.current_model
            del self.current_tokenizer
            self.current_model = None
            self.current_tokenizer = None
            self.current_model_name = None
            self.generation_config = None
            
            # Clear CUDA cache if using GPU
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            self.logger.info("Model unloaded successfully")
