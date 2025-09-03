"""Text and image preprocessing utilities."""

import re
import base64
import logging
from typing import List, Dict, Any, Optional, Union
from urllib.parse import urlparse
import io

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


class PreprocessingError(Exception):
    """Exception raised during preprocessing."""
    pass


class TextPreprocessor:
    """Text preprocessing utilities."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Common text cleaning patterns
        self.url_pattern = re.compile(
            r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        )
        self.email_pattern = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
        self.phone_pattern = re.compile(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b')
        self.excessive_whitespace = re.compile(r'\s+')
    
    def clean_text(self, text: str, remove_urls: bool = False, remove_emails: bool = False) -> str:
        """Clean and normalize text.
        
        Args:
            text: Input text
            remove_urls: Whether to remove URLs
            remove_emails: Whether to remove email addresses
            
        Returns:
            Cleaned text
        """
        if not isinstance(text, str):
            text = str(text)
        
        # Remove URLs if requested
        if remove_urls:
            text = self.url_pattern.sub('', text)
        
        # Remove emails if requested  
        if remove_emails:
            text = self.email_pattern.sub('', text)
        
        # Normalize whitespace
        text = self.excessive_whitespace.sub(' ', text)
        
        # Strip leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def truncate_text(self, text: str, max_length: int, strategy: str = 'truncate') -> str:
        """Truncate text to maximum length.
        
        Args:
            text: Input text
            max_length: Maximum length in characters
            strategy: Truncation strategy ('truncate', 'middle', 'sentences')
            
        Returns:
            Truncated text
        """
        if len(text) <= max_length:
            return text
        
        if strategy == 'truncate':
            return text[:max_length]
        elif strategy == 'middle':
            # Keep beginning and end, remove middle
            keep_length = max_length // 2
            return text[:keep_length] + " ... " + text[-keep_length:]
        elif strategy == 'sentences':
            # Try to keep complete sentences
            sentences = text.split('. ')
            result = ""
            for sentence in sentences:
                if len(result + sentence) <= max_length:
                    result += sentence + ". "
                else:
                    break
            return result.strip()
        else:
            raise ValueError(f"Unknown truncation strategy: {strategy}")
    
    def validate_text(self, text: str, max_length: Optional[int] = None) -> bool:
        """Validate text input.
        
        Args:
            text: Text to validate
            max_length: Maximum allowed length
            
        Returns:
            True if valid
            
        Raises:
            PreprocessingError: If text is invalid
        """
        if not isinstance(text, str):
            raise PreprocessingError("Input must be a string")
        
        if not text.strip():
            raise PreprocessingError("Text cannot be empty")
        
        if max_length and len(text) > max_length:
            raise PreprocessingError(f"Text length {len(text)} exceeds maximum {max_length}")
        
        return True


class ImagePreprocessor:
    """Image preprocessing utilities."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        if not PIL_AVAILABLE:
            self.logger.warning("PIL/Pillow not available. Image processing will be limited.")
        
        # Data URL pattern
        self.data_url_pattern = re.compile(r'^data:image/([a-zA-Z0-9]+);base64,(.+)$')
        
        # Supported formats
        self.supported_formats = {'jpeg', 'jpg', 'png', 'webp', 'bmp', 'gif'}
        
        # Size limits (in pixels and bytes)
        self.max_dimension = 8192
        self.max_file_size = 25 * 1024 * 1024  # 25MB
        self.max_decoded_size = 25 * 1024 * 1024  # 25MB decoded
    
    def parse_data_url(self, data_url: str) -> tuple[str, bytes]:
        """Parse a data URL to extract format and image data.
        
        Args:
            data_url: Data URL string
            
        Returns:
            Tuple of (format, image_bytes)
            
        Raises:
            PreprocessingError: If data URL is invalid
        """
        match = self.data_url_pattern.match(data_url)
        if not match:
            raise PreprocessingError("Invalid data URL format")
        
        format_str, base64_data = match.groups()
        
        # Validate format
        format_lower = format_str.lower()
        if format_lower not in self.supported_formats:
            supported = ', '.join(self.supported_formats)
            raise PreprocessingError(f"Unsupported image format '{format_str}'. Supported: {supported}")
        
        # Decode base64
        try:
            image_bytes = base64.b64decode(base64_data)
        except Exception as e:
            raise PreprocessingError(f"Invalid base64 encoding: {e}")
        
        # Check file size
        if len(image_bytes) > self.max_file_size:
            raise PreprocessingError(f"Image size {len(image_bytes)} bytes exceeds maximum {self.max_file_size}")
        
        return format_lower, image_bytes
    
    def validate_image(self, image_data: Union[str, bytes]) -> tuple[str, bytes]:
        """Validate and normalize image data.
        
        Args:
            image_data: Image data URL or bytes
            
        Returns:
            Tuple of (format, validated_image_bytes)
            
        Raises:
            PreprocessingError: If image is invalid
        """
        if isinstance(image_data, str):
            # Parse data URL
            format_str, image_bytes = self.parse_data_url(image_data)
        elif isinstance(image_data, bytes):
            # Detect format from bytes
            format_str = self._detect_image_format(image_data)
            image_bytes = image_data
        else:
            raise PreprocessingError("Image data must be string (data URL) or bytes")
        
        if not PIL_AVAILABLE:
            # Basic validation without PIL
            return format_str, image_bytes
        
        # Validate with PIL
        try:
            with Image.open(io.BytesIO(image_bytes)) as img:
                # Check dimensions
                width, height = img.size
                if width > self.max_dimension or height > self.max_dimension:
                    raise PreprocessingError(
                        f"Image dimensions {width}x{height} exceed maximum {self.max_dimension}x{self.max_dimension}"
                    )
                
                # Check decoded size estimate
                estimated_size = width * height * 3  # RGB estimate
                if estimated_size > self.max_decoded_size:
                    raise PreprocessingError(f"Decoded image size too large: {estimated_size} bytes")
                
                # Verify format matches
                detected_format = img.format.lower() if img.format else 'unknown'
                if detected_format != format_str and not (detected_format == 'jpeg' and format_str == 'jpg'):
                    self.logger.warning(f"Format mismatch: declared {format_str}, detected {detected_format}")
        
        except PreprocessingError:
            raise
        except Exception as e:
            raise PreprocessingError(f"Invalid image data: {e}")
        
        return format_str, image_bytes
    
    def _detect_image_format(self, image_bytes: bytes) -> str:
        """Detect image format from bytes.
        
        Args:
            image_bytes: Image bytes
            
        Returns:
            Detected format string
        """
        # Simple format detection based on magic bytes
        if image_bytes.startswith(b'\xff\xd8\xff'):
            return 'jpeg'
        elif image_bytes.startswith(b'\x89PNG'):
            return 'png'
        elif image_bytes.startswith(b'RIFF') and b'WEBP' in image_bytes[:12]:
            return 'webp'
        elif image_bytes.startswith(b'BM'):
            return 'bmp'
        elif image_bytes.startswith(b'GIF8'):
            return 'gif'
        else:
            return 'unknown'
    
    def extract_image_from_tag(self, html_tag: str) -> Optional[str]:
        """Extract data URL from HTML img tag.
        
        Args:
            html_tag: HTML img tag string
            
        Returns:
            Extracted data URL or None
        """
        # Simple regex to extract src attribute
        src_pattern = re.compile(r'<img[^>]+src=["\']([^"\']+)["\'][^>]*>', re.IGNORECASE)
        match = src_pattern.search(html_tag)
        
        if match:
            src_url = match.group(1)
            if src_url.startswith('data:image/'):
                return src_url
        
        return None


class ModalityDetector:
    """Automatic modality detection for inputs."""
    
    def __init__(self):
        self.text_processor = TextPreprocessor()
        self.image_processor = ImagePreprocessor()
        self.logger = logging.getLogger(__name__)
    
    def detect_modality(self, input_data: Union[str, Dict[str, Any]]) -> str:
        """Detect the modality of input data.
        
        Args:
            input_data: Input data (string or dict)
            
        Returns:
            Detected modality: 'text', 'image', or 'text_image'
        """
        if isinstance(input_data, dict):
            # Structured input
            has_text = 'text' in input_data and input_data['text']
            has_image = 'image' in input_data and input_data['image']
            
            if has_text and has_image:
                return 'text_image'
            elif has_image:
                return 'image'
            elif has_text:
                return 'text'
            else:
                raise PreprocessingError("Input data must contain 'text' or 'image' field")
        
        elif isinstance(input_data, str):
            # String input - detect content type
            input_data = input_data.strip()
            
            # Check for data URL
            if input_data.startswith('data:image/'):
                return 'image'
            
            # Check for HTML img tag with data URL
            img_data_url = self.image_processor.extract_image_from_tag(input_data)
            if img_data_url:
                # Mixed content: text + embedded image
                return 'text_image'
            
            # Default to text
            return 'text'
        
        else:
            raise PreprocessingError("Input must be string or dictionary")
    
    def extract_content(self, input_data: Union[str, Dict[str, Any]], modality: str) -> Dict[str, Any]:
        """Extract content based on detected modality.
        
        Args:
            input_data: Input data
            modality: Detected modality
            
        Returns:
            Dictionary with extracted content
        """
        result = {'modality': modality}
        
        if isinstance(input_data, dict):
            # Structured input
            if 'text' in input_data:
                result['text'] = input_data['text']
            if 'image' in input_data:
                result['image'] = input_data['image']
        
        elif isinstance(input_data, str):
            if modality == 'text':
                result['text'] = input_data
            elif modality == 'image':
                result['image'] = input_data
            elif modality == 'text_image':
                # Extract both text and image from mixed content
                img_data_url = self.image_processor.extract_image_from_tag(input_data)
                if img_data_url:
                    # Remove img tag from text
                    text_content = re.sub(r'<img[^>]*>', '', input_data, flags=re.IGNORECASE)
                    result['text'] = text_content.strip()
                    result['image'] = img_data_url
                else:
                    # Fallback to text
                    result['text'] = input_data
                    result['modality'] = 'text'
        
        return result
