"""API routes package."""

from . import health, models, completions, chat, auth, monitoring, registry

__all__ = ['health', 'models', 'completions', 'chat', 'auth', 'monitoring', 'registry']
