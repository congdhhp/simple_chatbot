"""Sentence Transformers model family implementation."""

import logging
import torch
import numpy as np
from typing import List, Optional, Dict, Any

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

from .base_model import BaseEmbeddingModel


class SentenceTransformersModel(BaseEmbeddingModel):
    """Generic Sentence Transformers model implementation."""
    
    def __init__(self, model_config: Dict[str, Any]):
        """Initialize Sentence Transformers model.
        
        Args:
            model_config: Model configuration dictionary
        """
        super().__init__(model_config)
        self.logger = logging.getLogger(__name__)
        
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError("sentence-transformers library is required for this model family")
        
    def load_model(self) -> bool:
        """Load the Sentence Transformers model.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            self.logger.info(f"Loading Sentence Transformers model: {self.model_id}")
            
            self.model = SentenceTransformer(
                self.model_id,
                device=self.device,
                trust_remote_code=self.model_config.get('settings', {}).get('trust_remote_code', False)
            )
            
            # Set model to evaluation mode and appropriate dtype
            self.model.eval()
            if self.model_config.get('torch_dtype') == 'float16' and self.device == 'cuda':
                self.model.half()
            
            self.is_loaded = True
            memory_info = self.get_memory_usage()
            self.logger.info(f"Sentence Transformers model loaded successfully. Memory usage: {memory_info['allocated']:.2f}GB")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load Sentence Transformers model {self.model_id}: {str(e)}")
            return False
    
    def unload_model(self) -> None:
        """Unload the Sentence Transformers model to free memory."""
        try:
            if self.model is not None:
                del self.model
                self.model = None
                
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            self.is_loaded = False
            self.logger.info(f"Sentence Transformers model {self.model_id} unloaded")
            
        except Exception as e:
            self.logger.error(f"Error unloading Sentence Transformers model: {str(e)}")
    
    def preprocess_text(self, text: str, input_type: Optional[str] = None) -> str:
        """Preprocess text for Sentence Transformers models.
        
        Args:
            text: Input text
            input_type: Type of input (ignored for most models)
            
        Returns:
            Preprocessed text (cleaned)
        """
        # Most Sentence Transformers models don't use prefixes
        return text.strip()
    
    def encode_texts(
        self, 
        texts: List[str], 
        input_type: Optional[str] = None,
        normalize: bool = True,
        batch_size: Optional[int] = None
    ) -> np.ndarray:
        """Encode texts into embeddings using Sentence Transformers model.
        
        Args:
            texts: List of texts to encode
            input_type: Type of input (ignored for most models)
            normalize: Whether to normalize embeddings
            batch_size: Batch size for processing
            
        Returns:
            NumPy array of embeddings
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        # Most Sentence Transformers models don't support input_type
        if input_type is not None:
            self.logger.warning(f"Sentence Transformers model {self.model_id} ignores input_type parameter")
        
        # Preprocess texts
        processed_texts = [self.preprocess_text(text) for text in texts]
        
        # Get batch size
        if batch_size is None:
            batch_size = self.model_config.get('settings', {}).get('batch_size', 128)
        
        try:
            # Use SentenceTransformers encode method
            embeddings = self.model.encode(
                processed_texts,
                batch_size=batch_size,
                normalize_embeddings=normalize if normalize is not None else self.normalize_embeddings,
                convert_to_numpy=True,
                show_progress_bar=len(processed_texts) > 100
            )
            
            return embeddings
            
        except Exception as e:
            self.logger.error(f"Error encoding texts with Sentence Transformers model: {str(e)}")
            raise
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get Sentence Transformers model information.
        
        Returns:
            Dictionary with model information
        """
        info = super().get_model_info()
        info.update({
            'family_specific': {
                'sentence_transformers_version': self._get_sentence_transformers_version()
            }
        })
        return info
    
    def _get_sentence_transformers_version(self) -> Optional[str]:
        """Get sentence-transformers library version.
        
        Returns:
            Version string or None if not available
        """
        try:
            import sentence_transformers
            return sentence_transformers.__version__
        except (ImportError, AttributeError):
            return None
