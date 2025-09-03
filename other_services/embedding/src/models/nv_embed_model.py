"""NV-Embed model family implementation for NVIDIA-style models."""

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


class NVEmbedModel(BaseEmbeddingModel):
    """NV-Embed model family implementation for NVIDIA-style embedding models."""
    
    def __init__(self, model_config: Dict[str, Any]):
        """Initialize NV-Embed model.
        
        Args:
            model_config: Model configuration dictionary
        """
        super().__init__(model_config)
        self.logger = logging.getLogger(__name__)
        
        # NV-Embed specific settings
        self.use_sentence_transformers = (
            model_config.get('settings', {}).get('use_sentence_transformers', True) 
            and SENTENCE_TRANSFORMERS_AVAILABLE
        )
        self.query_prefix = model_config.get('settings', {}).get(
            'query_prefix', 
            'Represent this sentence for searching relevant passages: '
        )
        self.passage_prefix = model_config.get('settings', {}).get(
            'passage_prefix', 
            'Represent this sentence for retrieval: '
        )
        
    def load_model(self) -> bool:
        """Load the NV-Embed model and tokenizer.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            self.logger.info(f"Loading NV-Embed model: {self.model_id}")
            
            if self.use_sentence_transformers:
                # Use SentenceTransformers for easier handling
                self.model = SentenceTransformer(
                    self.model_id,
                    device=self.device,
                    trust_remote_code=self.model_config.get('settings', {}).get('trust_remote_code', True)
                )
                
                # Set model to evaluation mode and appropriate dtype
                self.model.eval()
                if self.model_config.get('torch_dtype') == 'float16' and self.device == 'cuda':
                    self.model.half()
                    
            else:
                # Use transformers directly for more control
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_id,
                    trust_remote_code=self.model_config.get('settings', {}).get('trust_remote_code', True)
                )
                self.model = AutoModel.from_pretrained(
                    self.model_id,
                    torch_dtype=torch.float16 if self.model_config.get('torch_dtype') == 'float16' else torch.float32,
                    trust_remote_code=self.model_config.get('settings', {}).get('trust_remote_code', True)
                )
                self.model.to(self.device)
                self.model.eval()
            
            self.is_loaded = True
            memory_info = self.get_memory_usage()
            self.logger.info(f"NV-Embed model loaded successfully. Memory usage: {memory_info['allocated']:.2f}GB")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load NV-Embed model {self.model_id}: {str(e)}")
            return False
    
    def unload_model(self) -> None:
        """Unload the NV-Embed model to free memory."""
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
            self.logger.info(f"NV-Embed model {self.model_id} unloaded")
            
        except Exception as e:
            self.logger.error(f"Error unloading NV-Embed model: {str(e)}")
    
    def preprocess_text(self, text: str, input_type: Optional[str] = None) -> str:
        """Preprocess text with NV-Embed specific prefixes.
        
        Args:
            text: Input text
            input_type: Type of input ('query' or 'passage')
            
        Returns:
            Preprocessed text with appropriate prefix
        """
        # Clean text
        text = text.strip()
        
        # Add NV-Embed specific prefixes
        if input_type == 'query':
            text = self.query_prefix + text
        elif input_type == 'passage':
            text = self.passage_prefix + text
        elif input_type is None and self.supports_input_type:
            # Default to query prefix if no input_type specified
            text = self.query_prefix + text
            
        return text
    
    def encode_texts(
        self, 
        texts: List[str], 
        input_type: Optional[str] = None,
        normalize: bool = True,
        batch_size: Optional[int] = None
    ) -> np.ndarray:
        """Encode texts into embeddings using NV-Embed model.
        
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
            self.logger.error(f"Error encoding texts with NV-Embed model: {str(e)}")
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
                
                # Use pooler output if available, otherwise use mean pooling
                if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
                    embeddings = outputs.pooler_output
                else:
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
    
    def postprocess_embeddings(
        self, 
        embeddings: np.ndarray, 
        embedding_type: str = 'float',
        dimensions: Optional[int] = None
    ) -> np.ndarray:
        """Postprocess embeddings with NV-Embed optimizations.
        
        Args:
            embeddings: Input embeddings
            embedding_type: Target embedding type
            dimensions: Target dimensions
            
        Returns:
            Processed embeddings
        """
        # NV-Embed models might have specialized compression
        if embedding_type in ['binary', 'ubinary']:
            # Use more sophisticated binary conversion for NV-Embed
            return self._nv_embed_binary_conversion(embeddings, signed=(embedding_type == 'binary'))
        else:
            return super().postprocess_embeddings(embeddings, embedding_type, dimensions)
    
    def _nv_embed_binary_conversion(self, embeddings: np.ndarray, signed: bool = True) -> np.ndarray:
        """NV-Embed specific binary conversion with optimizations.
        
        Args:
            embeddings: Input embeddings
            signed: Whether to use signed binary representation
            
        Returns:
            Binary embeddings
        """
        # More sophisticated binary conversion that preserves more information
        # This could be enhanced with learned thresholds or adaptive quantization
        
        # Use median as threshold instead of zero for better balance
        thresholds = np.median(embeddings, axis=1, keepdims=True)
        binary_embeddings = (embeddings > thresholds).astype(np.uint8)
        
        # Pack bits into bytes
        packed_embeddings = np.packbits(binary_embeddings, axis=1)
        
        if signed:
            return packed_embeddings.astype(np.int8)
        else:
            return packed_embeddings
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get NV-Embed model information.
        
        Returns:
            Dictionary with NV-Embed model information
        """
        info = super().get_model_info()
        info.update({
            'family_specific': {
                'query_prefix': self.query_prefix,
                'passage_prefix': self.passage_prefix,
                'use_sentence_transformers': self.use_sentence_transformers,
                'supports_advanced_compression': True
            }
        })
        return info
