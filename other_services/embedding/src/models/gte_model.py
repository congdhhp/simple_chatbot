"""GTE model family implementation."""

import logging
import torch
import numpy as np
from typing import List, Optional, Dict, Any

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

from transformers import AutoTokenizer, AutoModel
from .base_model import BaseEmbeddingModel


class GTEModel(BaseEmbeddingModel):
    """GTE (General Text Embeddings) model family implementation."""
    
    def __init__(self, model_config: Dict[str, Any]):
        """Initialize GTE model.
        
        Args:
            model_config: Model configuration dictionary
        """
        super().__init__(model_config)
        self.logger = logging.getLogger(__name__)
        
        # GTE-specific settings
        self.use_sentence_transformers = (
            model_config.get('settings', {}).get('use_sentence_transformers', True) 
            and SENTENCE_TRANSFORMERS_AVAILABLE
        )
        
    def load_model(self) -> bool:
        """Load the GTE model and tokenizer.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            self.logger.info(f"Loading GTE model: {self.model_id}")
            
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
            self.logger.info(f"GTE model loaded successfully. Memory usage: {memory_info['allocated']:.2f}GB")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load GTE model {self.model_id}: {str(e)}")
            return False
    
    def unload_model(self) -> None:
        """Unload the GTE model to free memory."""
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
            self.logger.info(f"GTE model {self.model_id} unloaded")
            
        except Exception as e:
            self.logger.error(f"Error unloading GTE model: {str(e)}")
    
    def preprocess_text(self, text: str, input_type: Optional[str] = None) -> str:
        """Preprocess text for GTE models.
        
        Args:
            text: Input text
            input_type: Type of input (ignored for GTE models)
            
        Returns:
            Preprocessed text (cleaned)
        """
        # GTE models don't use prefixes or input_type
        return text.strip()
    
    def encode_texts(
        self, 
        texts: List[str], 
        input_type: Optional[str] = None,
        normalize: bool = True,
        batch_size: Optional[int] = None
    ) -> np.ndarray:
        """Encode texts into embeddings using GTE model.
        
        Args:
            texts: List of texts to encode
            input_type: Type of input (ignored for GTE models)
            normalize: Whether to normalize embeddings
            batch_size: Batch size for processing
            
        Returns:
            NumPy array of embeddings
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        # GTE models don't support input_type
        if input_type is not None:
            self.logger.warning(f"GTE model {self.model_id} ignores input_type parameter")
        
        # Preprocess texts (simple cleaning for GTE)
        processed_texts = [self.preprocess_text(text) for text in texts]
        
        # Get batch size
        if batch_size is None:
            batch_size = self.model_config.get('settings', {}).get('batch_size', 64)
        
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
            self.logger.error(f"Error encoding texts with GTE model: {str(e)}")
            raise
    
    def _encode_with_transformers(
        self, 
        texts: List[str], 
        batch_size: int = 64,
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
                
                # Mean pooling for GTE models
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
        """Get GTE model information.
        
        Returns:
            Dictionary with GTE model information
        """
        info = super().get_model_info()
        info.update({
            'family_specific': {
                'use_sentence_transformers': self.use_sentence_transformers,
                'sentence_transformers_available': SENTENCE_TRANSFORMERS_AVAILABLE
            }
        })
        return info
