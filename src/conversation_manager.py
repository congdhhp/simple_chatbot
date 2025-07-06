"""Conversation management for the CLI chatbot."""

import json
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional
from pathlib import Path

from config_manager import ConfigManager


class ConversationManager:
    """Manages conversation history and context for the chatbot."""
    
    def __init__(self, config_manager: ConfigManager):
        """Initialize the conversation manager.
        
        Args:
            config_manager: Configuration manager instance
        """
        self.config_manager = config_manager
        self.conversation_history: List[Dict[str, Any]] = []
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Get settings
        settings = self.config_manager.get_settings()
        self.max_history = settings.get('max_conversation_history', 10)
        self.save_conversations = settings.get('save_conversations', True)
        
        # Create conversations directory
        self.conversations_dir = Path(settings.get('conversation_dir', 'conversations'))
        if self.save_conversations:
            self.conversations_dir.mkdir(exist_ok=True)
    
    def add_message(self, role: str, content: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Add a message to the conversation history.
        
        Args:
            role: Role of the message sender ('user', 'assistant', 'system')
            content: Content of the message
            metadata: Optional metadata for the message
        """
        message = {
            'role': role,
            'content': content,
            'timestamp': datetime.now().isoformat(),
            'metadata': metadata or {}
        }
        
        self.conversation_history.append(message)
        
        # Trim history if it exceeds max length
        if len(self.conversation_history) > self.max_history * 2:  # *2 for user+assistant pairs
            # Keep system messages and trim user/assistant pairs
            system_messages = [msg for msg in self.conversation_history if msg['role'] == 'system']
            other_messages = [msg for msg in self.conversation_history if msg['role'] != 'system']
            
            # Keep only the most recent conversations
            trimmed_messages = other_messages[-self.max_history * 2:]
            self.conversation_history = system_messages + trimmed_messages
            
            self.logger.info(f"Trimmed conversation history to {len(self.conversation_history)} messages")
    
    def get_conversation_context(self, include_system: bool = True) -> str:
        """Get the conversation context as a formatted string.
        
        Args:
            include_system: Whether to include system messages
            
        Returns:
            Formatted conversation context
        """
        if not self.conversation_history:
            return ""
        
        # Filter messages based on include_system flag
        messages = self.conversation_history
        if not include_system:
            messages = [msg for msg in messages if msg['role'] != 'system']
        
        # Get only user messages for the current context
        # The model manager will handle system prompts separately
        user_messages = [msg for msg in messages if msg['role'] == 'user']
        
        if not user_messages:
            return ""
        
        # Return the most recent user message
        return user_messages[-1]['content']
    
    def get_full_conversation(self) -> List[Dict[str, Any]]:
        """Get the full conversation history.
        
        Returns:
            List of conversation messages
        """
        return self.conversation_history.copy()
    
    def get_conversation_summary(self) -> Dict[str, Any]:
        """Get a summary of the current conversation.
        
        Returns:
            Dictionary with conversation statistics
        """
        total_messages = len(self.conversation_history)
        user_messages = len([msg for msg in self.conversation_history if msg['role'] == 'user'])
        assistant_messages = len([msg for msg in self.conversation_history if msg['role'] == 'assistant'])
        system_messages = len([msg for msg in self.conversation_history if msg['role'] == 'system'])
        
        start_time = None
        end_time = None
        if self.conversation_history:
            start_time = self.conversation_history[0]['timestamp']
            end_time = self.conversation_history[-1]['timestamp']
        
        return {
            'total_messages': total_messages,
            'user_messages': user_messages,
            'assistant_messages': assistant_messages,
            'system_messages': system_messages,
            'start_time': start_time,
            'end_time': end_time
        }
    
    def clear_history(self) -> None:
        """Clear the conversation history."""
        self.conversation_history.clear()
        self.logger.info("Conversation history cleared")
    
    def save_conversation(self, filename: Optional[str] = None) -> bool:
        """Save the current conversation to a file.
        
        Args:
            filename: Optional filename. If None, generates timestamp-based name.
            
        Returns:
            True if saved successfully, False otherwise
        """
        if not self.save_conversations:
            self.logger.warning("Conversation saving is disabled")
            return False
        
        if not self.conversation_history:
            self.logger.warning("No conversation to save")
            return False
        
        try:
            if filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"conversation_{timestamp}.json"
            
            # Ensure .json extension
            if not filename.endswith('.json'):
                filename += '.json'
            
            filepath = self.conversations_dir / filename
            
            # Prepare conversation data
            conversation_data = {
                'metadata': {
                    'created_at': datetime.now().isoformat(),
                    'total_messages': len(self.conversation_history),
                    'summary': self.get_conversation_summary()
                },
                'messages': self.conversation_history
            }
            
            # Save to file
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(conversation_data, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Conversation saved to {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save conversation: {e}")
            return False
    
    def load_conversation(self, filename: str) -> bool:
        """Load a conversation from a file.
        
        Args:
            filename: Name of the file to load
            
        Returns:
            True if loaded successfully, False otherwise
        """
        try:
            # Ensure .json extension
            if not filename.endswith('.json'):
                filename += '.json'
            
            filepath = self.conversations_dir / filename
            
            if not filepath.exists():
                self.logger.error(f"Conversation file not found: {filepath}")
                return False
            
            # Load conversation data
            with open(filepath, 'r', encoding='utf-8') as f:
                conversation_data = json.load(f)
            
            # Validate data structure
            if 'messages' not in conversation_data:
                self.logger.error("Invalid conversation file format")
                return False
            
            # Load messages
            self.conversation_history = conversation_data['messages']
            
            self.logger.info(f"Conversation loaded from {filepath}")
            self.logger.info(f"Loaded {len(self.conversation_history)} messages")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load conversation: {e}")
            return False
    
    def list_saved_conversations(self) -> List[Dict[str, Any]]:
        """List all saved conversations.
        
        Returns:
            List of conversation file information
        """
        if not self.conversations_dir.exists():
            return []
        
        conversations = []
        for filepath in self.conversations_dir.glob('*.json'):
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                conversations.append({
                    'filename': filepath.name,
                    'created_at': data.get('metadata', {}).get('created_at', 'Unknown'),
                    'total_messages': data.get('metadata', {}).get('total_messages', 0),
                    'file_size': filepath.stat().st_size
                })
            except Exception as e:
                self.logger.warning(f"Failed to read conversation file {filepath}: {e}")
        
        # Sort by creation time (newest first)
        conversations.sort(key=lambda x: x['created_at'], reverse=True)
        return conversations
    
    def optimize_context_for_model(self, max_tokens: int = 2048) -> str:
        """Optimize conversation context for model input within token limits.
        
        Args:
            max_tokens: Maximum number of tokens to target
            
        Returns:
            Optimized conversation context
        """
        if not self.conversation_history:
            return ""
        
        # Simple heuristic: assume ~4 characters per token
        max_chars = max_tokens * 4
        
        # Start with the most recent messages and work backwards
        context_messages = []
        total_chars = 0
        
        for message in reversed(self.conversation_history):
            if message['role'] == 'system':
                continue  # Skip system messages for context
            
            message_text = f"{message['role']}: {message['content']}\n"
            message_chars = len(message_text)
            
            if total_chars + message_chars > max_chars and context_messages:
                break
            
            context_messages.insert(0, message_text)
            total_chars += message_chars
        
        return "".join(context_messages).strip()
