"""E5 model family implementation."""

import logging
import torch
import numpy as np
from typing import List, Optional, Dict, Any
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel

from .base_model import BaseEmbeddingModel


class E5Model(BaseEmbeddingModel):
    """E5 model family implementation with query/passage support."""
    
    def __init__(self, model_config: Dict[str, Any]):
        """Initialize E5 model.
        
        Args:
            model_config: Model configuration dictionary
        """
        super().__init__(model_config)
        self.logger = logging.getLogger(__name__)
        
        # E5-specific settings
        self.use_sentence_transformers = model_config.get('settings', {}).get('use_sentence_transformers', True)
        self.query_prefix = model_config.get('settings', {}).get('query_prefix', 'query: ')
        self.passage_prefix = model_config.get('settings', {}).get('passage_prefix', 'passage: ')
        
    def load_model(self) -> bool:
        """Load the E5 model and tokenizer.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            self.logger.info(f"Loading E5 model: {self.model_id}")
            
            if self.use_sentence_transformers:
                # Use SentenceTransformers for easier handling
                self.model = SentenceTransformer(
                    self.model_id,
                    device=self.device,
                    trust_remote_code=self.model_config.get('settings', {}).get('trust_remote_code', False)
                )
                
                # Set model to evaluation mode and appropriate dtype
                self.model.eval()
                if self.model_config.get('torch_dtype') == 'float16' and self.device == 'cuda':
                    self.model.half()
                    
            else:
                # Use transformers directly for more control
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
                self.model = AutoModel.from_pretrained(
                    self.model_id,
                    torch_dtype=torch.float16 if self.model_config.get('torch_dtype') == 'float16' else torch.float32,
                    trust_remote_code=self.model_config.get('settings', {}).get('trust_remote_code', False)
                )
                self.model.to(self.device)
                self.model.eval()
            
            self.is_loaded = True
            memory_info = self.get_memory_usage()
            self.logger.info(f"E5 model loaded successfully. Memory usage: {memory_info['allocated']:.2f}GB")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load E5 model {self.model_id}: {str(e)}")
            return False
    
    def unload_model(self) -> None:
        """Unload the E5 model to free memory."""
        try:
            if self.model is not None:
                del self.model
                self.model = None
                
            if hasattr(self, 'tokenizer') and self.tokenizer is not None:
                del self.tokenizer
                self.tokenizer = None
                
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            self.is_loaded = False
            self.logger.info(f"E5 model {self.model_id} unloaded")
            
        except Exception as e:
            self.logger.error(f"Error unloading E5 model: {str(e)}")
    
    def preprocess_text(self, text: str, input_type: Optional[str] = None) -> str:
        """Preprocess text with E5-specific prefixes.
        
        Args:
            text: Input text
            input_type: Type of input ('query' or 'passage')
            
        Returns:
            Preprocessed text with appropriate prefix
        """
        # Clean text
        text = text.strip()
        
        # Add E5-specific prefixes
        if input_type == 'query':
            text = self.query_prefix + text
        elif input_type == 'passage':
            text = self.passage_prefix + text
        elif input_type is None and self.supports_input_type:
            # Default to query prefix if no input_type specified for E5
            text = self.query_prefix + text
            
        return text
    
    def encode_texts(
        self, 
        texts: List[str], 
        input_type: Optional[str] = None,
        normalize: bool = True,
        batch_size: Optional[int] = None
    ) -> np.ndarray:
        """Encode texts into embeddings using E5 model.
        
        Args:
            texts: List of texts to encode
            input_type: Type of input ('query' or 'passage')
            normalize: Whether to normalize embeddings
            batch_size: Batch size for processing
            
        Returns:
            NumPy array of embeddings
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        # Validate input_type
        self.validate_input_type(input_type)
        
        # Preprocess texts with appropriate prefixes
        processed_texts = [self.preprocess_text(text, input_type) for text in texts]
        
        # Get batch size
        if batch_size is None:
            batch_size = self.model_config.get('settings', {}).get('batch_size', 32)
        
        try:
            if self.use_sentence_transformers:
                # Use SentenceTransformers encode method
                embeddings = self.model.encode(
                    processed_texts,
                    batch_size=batch_size,
                    normalize_embeddings=normalize if normalize is not None else self.normalize_embeddings,
                    convert_to_numpy=True,
                    show_progress_bar=len(processed_texts) > 100
                )
            else:
                # Use transformers directly
                embeddings = self._encode_with_transformers(
                    processed_texts, 
                    batch_size=batch_size,
                    normalize=normalize if normalize is not None else self.normalize_embeddings
                )
            
            return embeddings
            
        except Exception as e:
            self.logger.error(f"Error encoding texts with E5 model: {str(e)}")
            raise
    
    def _encode_with_transformers(
        self, 
        texts: List[str], 
        batch_size: int = 32,
        normalize: bool = True
    ) -> np.ndarray:
        """Encode texts using transformers directly.
        
        Args:
            texts: Preprocessed texts to encode
            batch_size: Batch size for processing
            normalize: Whether to normalize embeddings
            
        Returns:
            NumPy array of embeddings
        """
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            # Tokenize
            inputs = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.max_seq_length,
                return_tensors='pt'
            )
            
            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate embeddings
            with torch.no_grad():
                outputs = self.model(**inputs)
                
                # Mean pooling
                token_embeddings = outputs.last_hidden_state
                input_mask_expanded = inputs['attention_mask'].unsqueeze(-1).expand(token_embeddings.size()).float()
                
                embeddings = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                
                # Normalize if requested
                if normalize:
                    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
                
                # Convert to numpy
                batch_embeddings = embeddings.cpu().numpy()
                all_embeddings.append(batch_embeddings)
        
        return np.vstack(all_embeddings)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get E5 model information.
        
        Returns:
            Dictionary with E5 model information
        """
        info = super().get_model_info()
        info.update({
            'family_specific': {
                'query_prefix': self.query_prefix,
                'passage_prefix': self.passage_prefix,
                'use_sentence_transformers': self.use_sentence_transformers
            }
        })
        return info
    
    def _get_effective_batch_size(self, num_texts: int, requested_batch_size: Optional[int] = None) -> int:
        """Get effective batch size based on text count and memory constraints.
        
        Args:
            num_texts: Number of texts to process
            requested_batch_size: Requested batch size
            
        Returns:
            Effective batch size to use
        """
        if requested_batch_size is not None:
            return min(requested_batch_size, num_texts)
        
        # Auto-determine batch size based on model size and available memory
        default_batch_size = self.model_config.get('settings', {}).get('batch_size', 32)
        
        # Reduce batch size for longer sequences
        if self.max_seq_length > 256:
            default_batch_size = max(1, default_batch_size // 2)
        
        return min(default_batch_size, num_texts)
