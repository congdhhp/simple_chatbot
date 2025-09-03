"""Embedding manager for the embedding service."""

import logging
import asyncio
import time
from typing import List, Dict, Any, Optional, Union
import numpy as np

from .config_manager import ConfigManager
from .model_manager import ModelManager
from .utils import (
    TextPreprocessor, 
    ImagePreprocessor, 
    ModalityDetector,
    EmbeddingPostprocessor,
    EmbeddingCompressor,
    PreprocessingError
)


class EmbeddingManager:
    """Manages embedding generation with preprocessing and postprocessing."""
    
    def __init__(self, config_manager: ConfigManager, model_manager: ModelManager):
        """Initialize the embedding manager.
        
        Args:
            config_manager: Configuration manager instance
            model_manager: Model manager instance
        """
        self.config_manager = config_manager
        self.model_manager = model_manager
        self.logger = logging.getLogger(__name__)
        
        # Initialize processors
        self.text_processor = TextPreprocessor()
        self.image_processor = ImagePreprocessor()
        self.modality_detector = ModalityDetector()
        self.embedding_processor = EmbeddingPostprocessor()
        self.compressor = EmbeddingCompressor()
        
        # Batch processing settings
        self.max_batch_size = config_manager.get_max_batch_size()
        self.enable_dynamic_batching = config_manager.is_dynamic_batching_enabled()
        self.batch_timeout_ms = config_manager.get_batch_timeout_ms()
        
        # Request queue for dynamic batching
        self._request_queue = asyncio.Queue() if self.enable_dynamic_batching else None
        self._batch_processor_task = None
        
    async def start_batch_processor(self):
        """Start the dynamic batch processor."""
        if self.enable_dynamic_batching and self._batch_processor_task is None:
            self._batch_processor_task = asyncio.create_task(self._batch_processor_loop())
            self.logger.info("Dynamic batch processor started")
    
    async def stop_batch_processor(self):
        """Stop the dynamic batch processor."""
        if self._batch_processor_task:
            self._batch_processor_task.cancel()
            try:
                await self._batch_processor_task
            except asyncio.CancelledError:
                pass
            self._batch_processor_task = None
            self.logger.info("Dynamic batch processor stopped")
    
    async def generate_embeddings(
        self,
        inputs: List[Union[str, Dict[str, Any]]],
        model: str,
        input_type: Optional[str] = None,
        modality: Optional[Union[str, List[str]]] = None,
        embedding_type: str = 'float',
        dimensions: Optional[int] = None,
        normalize: bool = True
    ) -> Dict[str, Any]:
        """Generate embeddings for inputs.
        
        Args:
            inputs: List of input data (strings or dictionaries)
            model: Model name to use
            input_type: Input type ('query' or 'passage')
            modality: Modality specification
            embedding_type: Type of embeddings to return
            dimensions: Target dimensions (for supported models)
            normalize: Whether to normalize embeddings
            
        Returns:
            Dictionary with embeddings and metadata
        """
        try:
            start_time = time.time()
            
            # Validate and normalize model request
            normalized_model, effective_input_type = self.model_manager.validate_model_request(
                model, input_type
            )
            
            # Ensure model is loaded
            if not self.model_manager.ensure_model_loaded(normalized_model):
                raise RuntimeError(f"Failed to load model {normalized_model}")
            
            current_model = self.model_manager.get_current_model()
            
            # Validate embedding type
            current_model.validate_embedding_type(embedding_type)
            
            # Preprocess inputs
            processed_inputs = self._preprocess_inputs(inputs, modality)
            
            # Generate embeddings
            embeddings = await self._generate_embeddings_internal(
                processed_inputs, current_model, effective_input_type, normalize
            )
            
            # Postprocess embeddings
            final_embeddings = self._postprocess_embeddings(
                embeddings, embedding_type, dimensions, current_model
            )
            
            generation_time = time.time() - start_time
            
            # Create response
            response = self._create_response(
                final_embeddings, model, generation_time, processed_inputs
            )
            
            self.logger.info(f"Generated {len(inputs)} embeddings in {generation_time:.3f}s")
            return response
            
        except Exception as e:
            self.logger.error(f"Error generating embeddings: {str(e)}")
            raise
    
    def _preprocess_inputs(
        self, 
        inputs: List[Union[str, Dict[str, Any]]], 
        modality: Optional[Union[str, List[str]]] = None
    ) -> List[Dict[str, Any]]:
        """Preprocess input data.
        
        Args:
            inputs: Raw input data
            modality: Modality specification
            
        Returns:
            List of processed input dictionaries
        """
        processed = []
        
        for i, input_data in enumerate(inputs):
            try:
                # Determine modality
                if isinstance(modality, list):
                    input_modality = modality[i] if i < len(modality) else modality[0]
                elif modality:
                    input_modality = modality
                else:
                    # Auto-detect modality
                    input_modality = self.modality_detector.detect_modality(input_data)
                
                # Extract content based on modality
                content = self.modality_detector.extract_content(input_data, input_modality)
                
                # Validate modality support
                current_model = self.model_manager.get_current_model()
                current_model.validate_modality(input_modality)
                
                # Preprocess text if present
                if 'text' in content:
                    # Validate text length
                    max_length = self.config_manager.get_max_input_length()
                    self.text_processor.validate_text(content['text'], max_length)
                    
                    # Clean text
                    content['text'] = self.text_processor.clean_text(content['text'])
                
                # Validate image if present
                if 'image' in content:
                    format_str, image_bytes = self.image_processor.validate_image(content['image'])
                    content['image_format'] = format_str
                    content['image_size'] = len(image_bytes)
                
                processed.append(content)
                
            except PreprocessingError as e:
                raise ValueError(f"Input {i}: {str(e)}")
            except Exception as e:
                raise ValueError(f"Input {i}: Preprocessing failed - {str(e)}")
        
        return processed
    
    async def _generate_embeddings_internal(
        self,
        processed_inputs: List[Dict[str, Any]],
        model: Any,
        input_type: Optional[str],
        normalize: bool
    ) -> np.ndarray:
        """Generate embeddings using the model.
        
        Args:
            processed_inputs: Preprocessed input data
            model: Model instance
            input_type: Input type for the model
            normalize: Whether to normalize embeddings
            
        Returns:
            Generated embeddings
        """
        # Group inputs by modality for efficient processing
        text_inputs = []
        text_indices = []
        
        # Currently only supporting text modality
        # TODO: Add image and multimodal support
        for i, input_data in enumerate(processed_inputs):
            if input_data['modality'] in ['text', 'text_image']:
                if 'text' in input_data:
                    text_inputs.append(input_data['text'])
                    text_indices.append(i)
            elif input_data['modality'] == 'image':
                raise NotImplementedError("Image-only embeddings not yet implemented")
        
        if not text_inputs:
            raise ValueError("No valid text inputs found")
        
        # Generate text embeddings
        if self.enable_dynamic_batching and len(text_inputs) > 1:
            # Use dynamic batching
            embeddings = await self._generate_with_batching(
                text_inputs, model, input_type, normalize
            )
        else:
            # Direct generation
            embeddings = model.encode_texts(
                text_inputs, input_type=input_type, normalize=normalize
            )
        
        # TODO: Handle mixed modality cases
        # For now, we only return text embeddings
        return embeddings
    
    async def _generate_with_batching(
        self,
        texts: List[str],
        model: Any,
        input_type: Optional[str],
        normalize: bool
    ) -> np.ndarray:
        """Generate embeddings with dynamic batching.
        
        Args:
            texts: List of texts
            model: Model instance
            input_type: Input type
            normalize: Whether to normalize
            
        Returns:
            Generated embeddings
        """
        # For now, implement simple batching
        # TODO: Implement proper dynamic batching with queue
        
        all_embeddings = []
        batch_size = min(self.max_batch_size, len(texts))
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_embeddings = model.encode_texts(
                batch_texts, input_type=input_type, normalize=normalize
            )
            all_embeddings.append(batch_embeddings)
        
        return np.vstack(all_embeddings)
    
    def _postprocess_embeddings(
        self,
        embeddings: np.ndarray,
        embedding_type: str,
        dimensions: Optional[int],
        model: Any
    ) -> np.ndarray:
        """Postprocess embeddings.
        
        Args:
            embeddings: Raw embeddings
            embedding_type: Target embedding type
            dimensions: Target dimensions
            model: Model instance
            
        Returns:
            Postprocessed embeddings
        """
        # Apply model-specific postprocessing
        processed_embeddings = model.postprocess_embeddings(
            embeddings, embedding_type, dimensions
        )
        
        return processed_embeddings
    
    def _create_response(
        self,
        embeddings: np.ndarray,
        model: str,
        generation_time: float,
        processed_inputs: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Create API response.
        
        Args:
            embeddings: Final embeddings
            model: Model name
            generation_time: Generation time in seconds
            processed_inputs: Processed input data
            
        Returns:
            API response dictionary
        """
        # Create embedding objects
        data = []
        for i, embedding in enumerate(embeddings):
            # Convert numpy array to list for JSON serialization
            if isinstance(embedding, np.ndarray):
                embedding_list = embedding.tolist()
            else:
                embedding_list = list(embedding)
            
            data.append({
                "index": i,
                "embedding": embedding_list,
                "object": "embedding"
            })
        
        # Calculate token usage (simplified)
        total_tokens = sum(
            len(input_data.get('text', '').split()) 
            for input_data in processed_inputs
        )
        
        response = {
            "object": "list",
            "data": data,
            "model": model,
            "usage": {
                "prompt_tokens": total_tokens,
                "total_tokens": total_tokens
            },
            "metadata": {
                "generation_time": generation_time,
                "embedding_dimension": embeddings.shape[1] if len(embeddings.shape) > 1 else len(embeddings),
                "num_embeddings": len(embeddings)
            }
        }
        
        return response
    
    async def _batch_processor_loop(self):
        """Main loop for dynamic batch processing."""
        while True:
            try:
                # Collect requests for batching
                requests = []
                deadline = time.time() + (self.batch_timeout_ms / 1000.0)
                
                # Collect initial request
                try:
                    request = await asyncio.wait_for(
                        self._request_queue.get(), 
                        timeout=self.batch_timeout_ms / 1000.0
                    )
                    requests.append(request)
                except asyncio.TimeoutError:
                    continue
                
                # Collect additional requests until deadline or batch full
                while len(requests) < self.max_batch_size and time.time() < deadline:
                    try:
                        remaining_time = deadline - time.time()
                        if remaining_time <= 0:
                            break
                        
                        request = await asyncio.wait_for(
                            self._request_queue.get(), 
                            timeout=remaining_time
                        )
                        requests.append(request)
                    except asyncio.TimeoutError:
                        break
                
                # Process batch
                if requests:
                    await self._process_request_batch(requests)
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in batch processor: {str(e)}")
    
    async def _process_request_batch(self, requests: List[Dict[str, Any]]):
        """Process a batch of requests.
        
        Args:
            requests: List of request dictionaries
        """
        # TODO: Implement proper batch processing
        # For now, process requests individually
        for request in requests:
            try:
                # Process individual request
                response = await self.generate_embeddings(**request['params'])
                request['future'].set_result(response)
            except Exception as e:
                request['future'].set_exception(e)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get embedding manager statistics.
        
        Returns:
            Statistics dictionary
        """
        return {
            'max_batch_size': self.max_batch_size,
            'enable_dynamic_batching': self.enable_dynamic_batching,
            'batch_timeout_ms': self.batch_timeout_ms,
            'batch_processor_running': self._batch_processor_task is not None,
            'queue_size': self._request_queue.qsize() if self._request_queue else 0
        }
