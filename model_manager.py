"""
LLM Model Manager - A flexible class for loading and managing different LLM models
Supports GPU acceleration with CUDA and easy model switching
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import logging
from typing import Optional, Dict, Any
import os

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LLMModelManager:
    """
    A flexible model manager that can load different LLM models by name/path
    and automatically utilize GPU if available.
    """
    
    def __init__(self, model_name: str = "openai-community/gpt2-large", use_gpu: bool = True):
        """
        Initialize the model manager
        
        Args:
            model_name (str): HuggingFace model name or local path
            use_gpu (bool): Whether to use GPU if available
        """
        self.model_name = model_name
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.device = "cuda" if self.use_gpu else "cpu"
        
        # Model components
        self.tokenizer = None
        self.model = None
        self.pipeline = None
        
        # Model configuration
        self.max_length = 512
        self.max_new_tokens = 100
        self.temperature = 0.7
        self.top_p = 0.9
        self.do_sample = True
        
        logger.info(f"Initializing LLM Model Manager")
        logger.info(f"Model: {self.model_name}")
        logger.info(f"Device: {self.device}")
        if self.use_gpu:
            logger.info(f"GPU: {torch.cuda.get_device_name()}")
            logger.info(f"CUDA Version: {torch.version.cuda}")
    
    def load_model(self) -> bool:
        """
        Load the specified model and tokenizer
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            logger.info(f"Loading model: {self.model_name}")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            # Add padding token if it doesn't exist
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model with appropriate device mapping
            if self.use_gpu:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float16,  # Use half precision for GPU efficiency
                    device_map="auto",
                    trust_remote_code=True
                )
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float32,
                    trust_remote_code=True
                )
                self.model.to(self.device)
            
            # Create text generation pipeline
            # Don't specify device when using accelerate (device_map="auto")
            if self.use_gpu:
                self.pipeline = pipeline(
                    "text-generation",
                    model=self.model,
                    tokenizer=self.tokenizer,
                    torch_dtype=torch.float16
                )
            else:
                self.pipeline = pipeline(
                    "text-generation",
                    model=self.model,
                    tokenizer=self.tokenizer,
                    device=-1,  # CPU only
                    torch_dtype=torch.float32
                )
            
            logger.info("Model loaded successfully!")
            return True
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return False
    
    def generate_response(self, prompt: str, max_new_tokens: int = 100) -> str:
        """
        Generate a response to the given prompt

        Args:
            prompt (str): Input prompt
            max_new_tokens (int): Maximum number of new tokens to generate

        Returns:
            str: Generated response
        """
        if self.pipeline is None:
            return "Error: Model not loaded. Please load a model first."

        try:
            # Generate response with better parameters to prevent repetition
            outputs = self.pipeline(
                prompt,
                max_new_tokens=max_new_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                do_sample=self.do_sample,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                return_full_text=False,  # Only return the generated part
                clean_up_tokenization_spaces=True,
                repetition_penalty=1.2,  # Stronger penalty for repetition
                no_repeat_ngram_size=2,  # Prevent repeating 2-grams
                num_return_sequences=1   # Only return one sequence
            )

            # Extract the generated text
            response = outputs[0]['generated_text'].strip()

            # Clean up the response
            response = self._clean_response(response, prompt)

            return response

        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return f"Error generating response: {str(e)}"

    def _clean_response(self, response: str, original_prompt: str) -> str:
        """
        Clean up the generated response to remove artifacts and repetition

        Args:
            response (str): Raw generated response
            original_prompt (str): Original input prompt

        Returns:
            str: Cleaned response
        """
        # Remove any remaining prompt text that might have leaked through
        prompt_keywords = ["Human:", "Assistant:", "human:", "assistant:"]
        for keyword in prompt_keywords:
            if keyword in response:
                parts = response.split(keyword)
                if len(parts) > 1:
                    # Take the first part before any role indicators
                    response = parts[0].strip()
                    break

        # Remove numbered assistant responses (Assistant 1:, Assistant 2:, etc.)
        import re
        response = re.sub(r'Assistant \d+:', '', response)
        response = re.sub(r'assistant \d+:', '', response, flags=re.IGNORECASE)

        # Split by common separators and take the first coherent part
        separators = ['\n\n', '. Assistant', '. Human', ' Assistant:', ' Human:']
        for sep in separators:
            if sep in response:
                response = response.split(sep)[0].strip()
                break

        # Clean up multiple spaces and newlines
        response = re.sub(r'\s+', ' ', response).strip()

        # Remove repetitive patterns
        words = response.split()
        if len(words) > 2:
            cleaned_words = []
            prev_word = ""
            repeat_count = 0

            for word in words:
                if word.lower() == prev_word.lower():
                    repeat_count += 1
                    if repeat_count < 2:  # Allow one repetition
                        cleaned_words.append(word)
                else:
                    repeat_count = 0
                    cleaned_words.append(word)
                    prev_word = word

                # Stop if we have enough content
                if len(cleaned_words) > 25:
                    break

            response = ' '.join(cleaned_words)

        # Ensure proper sentence ending
        if response and not response.endswith(('.', '!', '?', ':')):
            # Find the last complete sentence
            last_punct = max(response.rfind('.'), response.rfind('!'), response.rfind('?'))
            if last_punct > len(response) * 0.7:  # If punctuation is near the end
                response = response[:last_punct + 1]
            else:
                response += '.'

        # Final cleanup
        response = response.strip()

        # If response is empty or too short, provide a fallback
        if len(response.strip()) < 5:
            response = "Hello! How can I help you today?"

        return response
    
    def change_model(self, new_model_name: str) -> bool:
        """
        Change to a different model
        
        Args:
            new_model_name (str): New model name or path
            
        Returns:
            bool: True if successful, False otherwise
        """
        logger.info(f"Changing model from {self.model_name} to {new_model_name}")
        
        # Clear current model from memory
        if self.model is not None:
            del self.model
            del self.tokenizer
            del self.pipeline
            if self.use_gpu:
                torch.cuda.empty_cache()
        
        # Update model name and reload
        self.model_name = new_model_name
        return self.load_model()
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the current model
        
        Returns:
            dict: Model information
        """
        info = {
            "model_name": self.model_name,
            "device": self.device,
            "use_gpu": self.use_gpu,
            "model_loaded": self.model is not None,
            "max_length": self.max_length,
            "temperature": self.temperature,
            "top_p": self.top_p
        }
        
        if self.use_gpu and torch.cuda.is_available():
            info.update({
                "gpu_name": torch.cuda.get_device_name(),
                "cuda_version": torch.version.cuda,
                "gpu_memory_allocated": f"{torch.cuda.memory_allocated() / 1024**3:.2f} GB",
                "gpu_memory_reserved": f"{torch.cuda.memory_reserved() / 1024**3:.2f} GB"
            })
        
        return info
    
    def update_generation_params(self, **kwargs):
        """
        Update generation parameters
        
        Args:
            **kwargs: Generation parameters (temperature, top_p, max_length, etc.)
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
                logger.info(f"Updated {key} to {value}")
            else:
                logger.warning(f"Unknown parameter: {key}")
