"""Pydantic models for API requests and responses."""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Union
from enum import Enum
import time

class MessageRole(str, Enum):
    """Message roles for chat completions."""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"

class ChatMessage(BaseModel):
    """A single chat message."""
    role: MessageRole
    content: str

class ChatCompletionRequest(BaseModel):
    """Chat completion request model - OpenAI compatible."""
    model: str
    messages: List[ChatMessage]
    max_tokens: Optional[int] = Field(default=1024, ge=1, le=4096)
    temperature: Optional[float] = Field(default=0.7, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(default=0.9, ge=0.0, le=1.0)
    top_k: Optional[int] = Field(default=50, ge=1, le=100)
    frequency_penalty: Optional[float] = Field(default=0.0, ge=-2.0, le=2.0)
    presence_penalty: Optional[float] = Field(default=0.0, ge=-2.0, le=2.0)
    stop: Optional[Union[str, List[str]]] = None
    stream: Optional[bool] = False
    user: Optional[str] = None

class CompletionRequest(BaseModel):
    """Text completion request model - OpenAI compatible."""
    model: str
    prompt: Union[str, List[str]]
    max_tokens: Optional[int] = Field(default=1024, ge=1, le=4096)
    temperature: Optional[float] = Field(default=0.7, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(default=0.9, ge=0.0, le=1.0)
    frequency_penalty: Optional[float] = Field(default=0.0, ge=-2.0, le=2.0)
    presence_penalty: Optional[float] = Field(default=0.0, ge=-2.0, le=2.0)
    stop: Optional[Union[str, List[str]]] = None
    stream: Optional[bool] = False
    user: Optional[str] = None

class ChatChoice(BaseModel):
    """A single chat completion choice."""
    index: int
    message: ChatMessage
    finish_reason: Optional[str] = None

class CompletionChoice(BaseModel):
    """A single text completion choice."""
    index: int
    text: str
    finish_reason: Optional[str] = None

class Usage(BaseModel):
    """Token usage statistics."""
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    # Wave 1 Enhancement: Add metadata for monitoring
    tokens_per_second: Optional[float] = None
    latency_ms: Optional[float] = None

class ChatCompletionResponse(BaseModel):
    """Chat completion response model - OpenAI compatible."""
    id: str
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[ChatChoice]
    usage: Optional[Usage] = None
    # Wave 1 Enhancement: Add system fingerprint for version tracking
    system_fingerprint: Optional[str] = None

class CompletionResponse(BaseModel):
    """Text completion response model - OpenAI compatible."""
    id: str
    object: str = "text_completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[CompletionChoice]
    usage: Optional[Usage] = None
    # Wave 1 Enhancement: Add system fingerprint for version tracking  
    system_fingerprint: Optional[str] = None

class ModelInfo(BaseModel):
    """Model information response."""
    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "simple-llm-service"
    permission: Optional[List[Dict[str, Any]]] = None
    root: Optional[str] = None
    parent: Optional[str] = None

class ModelsResponse(BaseModel):
    """List models response."""
    object: str = "list"
    data: List[ModelInfo]

class ErrorResponse(BaseModel):
    """Error response model."""
    error: Dict[str, Any]

class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    timestamp: int = Field(default_factory=lambda: int(time.time()))
    version: str = "1.0.0"
    models_loaded: Optional[Dict[str, bool]] = None
