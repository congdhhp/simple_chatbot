"""Base model interface for embedding models."""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union
import torch
import numpy as np


class BaseEmbeddingModel(ABC):
    """Abstract base class for all embedding models."""
    
    def __init__(self, model_config: Dict[str, Any]):
        """Initialize the embedding model.
        
        Args:
            model_config: Model configuration dictionary
        """
        self.model_config = model_config
        self.model_id = model_config['model_id']
        self.model_name = model_config.get('display_name', self.model_id)
        self.family = model_config['family']
        self.device = model_config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        self.max_seq_length = model_config.get('max_seq_length', 512)
        self.embedding_dimension = model_config['embedding_dimension']
        self.normalize_embeddings = model_config.get('normalize_embeddings', True)
        self.supports_input_type = model_config.get('supports_input_type', False)
        self.supported_embedding_types = model_config.get('supported_embedding_types', ['float'])
        self.supported_modalities = model_config.get('supported_modalities', ['text'])
        self.supports_dimensions = model_config.get('supports_dimensions', [])
        
        # Model components (to be initialized by subclasses)
        self.model = None
        self.tokenizer = None
        self.is_loaded = False
        
    @abstractmethod
    def load_model(self) -> bool:
        """Load the model and tokenizer.
        
        Returns:
            True if successful, False otherwise
        """
        pass
    
    @abstractmethod
    def unload_model(self) -> None:
        """Unload the model to free memory."""
        pass
    
    @abstractmethod
    def encode_texts(
        self, 
        texts: List[str], 
        input_type: Optional[str] = None,
        normalize: bool = True,
        batch_size: Optional[int] = None
    ) -> np.ndarray:
        """Encode texts into embeddings.
        
        Args:
            texts: List of texts to encode
            input_type: Type of input ('query' or 'passage') for retrieval models
            normalize: Whether to normalize embeddings
            batch_size: Batch size for processing (None for model default)
            
        Returns:
            NumPy array of embeddings
        """
        pass
    
    def encode_images(
        self, 
        images: List[Union[str, bytes]], 
        input_type: Optional[str] = None,
        normalize: bool = True,
        batch_size: Optional[int] = None
    ) -> np.ndarray:
        """Encode images into embeddings.
        
        Args:
            images: List of image data URLs or bytes
            input_type: Type of input (for compatible models)
            normalize: Whether to normalize embeddings
            batch_size: Batch size for processing
            
        Returns:
            NumPy array of embeddings
            
        Raises:
            NotImplementedError: If model doesn't support images
        """
        if 'image' not in self.supported_modalities:
            raise NotImplementedError(f"Model {self.model_id} doesn't support image modality")
        raise NotImplementedError("Image encoding not implemented for this model family")
    
    def encode_multimodal(
        self,
        inputs: List[Dict[str, Any]],
        input_type: Optional[str] = None,
        normalize: bool = True,
        batch_size: Optional[int] = None
    ) -> np.ndarray:
        """Encode multimodal inputs (text + image) into embeddings.
        
        Args:
            inputs: List of dictionaries with 'text' and/or 'image' keys
            input_type: Type of input (for compatible models)
            normalize: Whether to normalize embeddings
            batch_size: Batch size for processing
            
        Returns:
            NumPy array of embeddings
            
        Raises:
            NotImplementedError: If model doesn't support multimodal inputs
        """
        if 'text_image' not in self.supported_modalities:
            raise NotImplementedError(f"Model {self.model_id} doesn't support multimodal inputs")
        raise NotImplementedError("Multimodal encoding not implemented for this model family")
    
    def preprocess_text(self, text: str, input_type: Optional[str] = None) -> str:
        """Preprocess text before encoding.
        
        Args:
            text: Input text
            input_type: Type of input ('query' or 'passage')
            
        Returns:
            Preprocessed text
        """
        # Base implementation - subclasses can override for model-specific preprocessing
        text = text.strip()
        
        # Add prefixes if supported and specified
        if self.supports_input_type and input_type:
            settings = self.model_config.get('settings', {})
            if input_type == 'query' and 'query_prefix' in settings:
                text = settings['query_prefix'] + text
            elif input_type == 'passage' and 'passage_prefix' in settings:
                text = settings['passage_prefix'] + text
                
        return text
    
    def postprocess_embeddings(
        self, 
        embeddings: np.ndarray, 
        embedding_type: str = 'float',
        dimensions: Optional[int] = None
    ) -> np.ndarray:
        """Postprocess embeddings (compression, dimension reduction).
        
        Args:
            embeddings: Input embeddings
            embedding_type: Target embedding type
            dimensions: Target dimensions (for Matryoshka models)
            
        Returns:
            Processed embeddings
        """
        # Dimension reduction (Matryoshka Representation Learning)
        if dimensions and dimensions < embeddings.shape[1]:
            if dimensions not in self.supports_dimensions:
                available = ', '.join(map(str, self.supports_dimensions))
                raise ValueError(f"Dimension {dimensions} not supported. Available: {available}")
            embeddings = embeddings[:, :dimensions]
        
        # Type conversion
        if embedding_type == 'int8':
            return self._convert_to_int8(embeddings)
        elif embedding_type == 'uint8':
            return self._convert_to_uint8(embeddings)
        elif embedding_type == 'binary':
            return self._convert_to_binary(embeddings, signed=True)
        elif embedding_type == 'ubinary':
            return self._convert_to_binary(embeddings, signed=False)
        else:
            return embeddings.astype(np.float32)
    
    def _convert_to_int8(self, embeddings: np.ndarray) -> np.ndarray:
        """Convert float embeddings to int8."""
        # Simple quantization: scale to [-127, 127] range
        embeddings_normalized = embeddings / np.max(np.abs(embeddings), axis=1, keepdims=True)
        return (embeddings_normalized * 127).astype(np.int8)
    
    def _convert_to_uint8(self, embeddings: np.ndarray) -> np.ndarray:
        """Convert float embeddings to uint8."""
        # Scale to [0, 255] range
        embeddings_min = np.min(embeddings, axis=1, keepdims=True)
        embeddings_max = np.max(embeddings, axis=1, keepdims=True)
        embeddings_scaled = (embeddings - embeddings_min) / (embeddings_max - embeddings_min)
        return (embeddings_scaled * 255).astype(np.uint8)
    
    def _convert_to_binary(self, embeddings: np.ndarray, signed: bool = True) -> np.ndarray:
        """Convert float embeddings to binary."""
        # Convert to binary based on sign (positive = 1, negative = 0)
        binary_embeddings = (embeddings > 0).astype(np.uint8)
        
        # Pack bits into bytes
        packed_embeddings = np.packbits(binary_embeddings, axis=1)
        
        if signed:
            return packed_embeddings.astype(np.int8)
        else:
            return packed_embeddings
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage information.
        
        Returns:
            Dictionary with memory usage stats in GB
        """
        if torch.cuda.is_available() and self.device == 'cuda':
            return {
                'allocated': torch.cuda.memory_allocated() / 1024**3,
                'reserved': torch.cuda.memory_reserved() / 1024**3,
                'max_allocated': torch.cuda.max_memory_allocated() / 1024**3
            }
        else:
            # For CPU, we could use psutil but keeping it simple for now
            return {'allocated': 0.0, 'reserved': 0.0, 'max_allocated': 0.0}
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information.
        
        Returns:
            Dictionary with model information
        """
        return {
            'model_id': self.model_id,
            'model_name': self.model_name,
            'family': self.family,
            'is_loaded': self.is_loaded,
            'device': self.device,
            'max_seq_length': self.max_seq_length,
            'embedding_dimension': self.embedding_dimension,
            'supports_input_type': self.supports_input_type,
            'supported_embedding_types': self.supported_embedding_types,
            'supported_modalities': self.supported_modalities,
            'supports_dimensions': self.supports_dimensions,
            'memory_usage': self.get_memory_usage()
        }
    
    def validate_input_type(self, input_type: Optional[str]) -> None:
        """Validate input_type parameter.
        
        Args:
            input_type: Input type to validate
            
        Raises:
            ValueError: If input_type is invalid for this model
        """
        if self.supports_input_type and input_type is None:
            raise ValueError(f"Model {self.model_id} requires input_type parameter (query or passage)")
        
        if not self.supports_input_type and input_type is not None:
            raise ValueError(f"Model {self.model_id} does not support input_type parameter")
        
        if input_type and input_type not in ['query', 'passage']:
            raise ValueError(f"Invalid input_type '{input_type}'. Must be 'query' or 'passage'")
    
    def validate_embedding_type(self, embedding_type: str) -> None:
        """Validate embedding_type parameter.
        
        Args:
            embedding_type: Embedding type to validate
            
        Raises:
            ValueError: If embedding_type is not supported
        """
        if embedding_type not in self.supported_embedding_types:
            supported = ', '.join(self.supported_embedding_types)
            raise ValueError(f"Embedding type '{embedding_type}' not supported by {self.model_id}. Supported: {supported}")
    
    def validate_modality(self, modality: str) -> None:
        """Validate modality parameter.
        
        Args:
            modality: Modality to validate
            
        Raises:
            ValueError: If modality is not supported
        """
        if modality not in self.supported_modalities:
            supported = ', '.join(self.supported_modalities)
            raise ValueError(f"Modality '{modality}' not supported by {self.model_id}. Supported: {supported}")
    
    def __del__(self):
        """Cleanup when object is destroyed."""
        if hasattr(self, 'is_loaded') and self.is_loaded:
            try:
                self.unload_model()
            except Exception:
                pass  # Ignore errors during cleanup
