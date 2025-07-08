"""
Configuration management for EasilyAI.

This module provides centralized configuration management including:
- API key management
- Model configuration  
- Rate limiting settings
- Cache configuration
- Performance settings
"""

import os
import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Dict, Any

# Try to load dotenv if available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # dotenv not available, environment variables should be set manually
    pass


@dataclass
class ServiceConfig:
    """Configuration for a specific AI service."""
    api_key: Optional[str] = None
    default_model: Optional[str] = None
    rate_limit: int = 60  # requests per minute
    timeout: int = 30  # seconds
    max_retries: int = 3
    
    def __post_init__(self):
        # Load from environment if not provided
        if not self.api_key:
            service_name = self.__class__.__name__.replace("Config", "").upper()
            self.api_key = os.getenv(f"{service_name}_API_KEY")


@dataclass 
class OpenAIConfig(ServiceConfig):
    """OpenAI-specific configuration."""
    default_model: str = "gpt-3.5-turbo"
    default_image_model: str = "dall-e-3"
    default_tts_model: str = "tts-1"
    default_tts_voice: str = "alloy"


@dataclass
class AnthropicConfig(ServiceConfig):
    """Anthropic-specific configuration."""
    default_model: str = "claude-3-haiku-20240307"


@dataclass
class GeminiConfig(ServiceConfig):
    """Google Gemini-specific configuration."""
    default_model: str = "gemini-1.5-flash"


@dataclass
class GrokConfig(ServiceConfig):
    """X.AI Grok-specific configuration."""
    default_model: str = "grok-beta"


@dataclass
class OllamaConfig(ServiceConfig):
    """Ollama-specific configuration."""
    default_model: str = "llama2"
    base_url: str = "http://localhost:11434"


@dataclass
class CacheConfig:
    """Cache configuration."""
    enabled: bool = True
    backend: str = "memory"  # "memory" or "file"
    ttl: int = 3600  # Time to live in seconds
    max_size: int = 1000  # Max items for memory cache
    cache_dir: str = "cache"  # Directory for file cache


@dataclass
class PerformanceConfig:
    """Performance-related configuration."""
    max_workers: int = 5  # For parallel processing
    batch_delay: float = 0.1  # Delay between batch requests
    enable_metrics: bool = True
    enable_cost_tracking: bool = True


@dataclass
class EasilyAIConfig:
    """Main configuration class for EasilyAI."""
    openai: OpenAIConfig = field(default_factory=OpenAIConfig)
    anthropic: AnthropicConfig = field(default_factory=AnthropicConfig)
    gemini: GeminiConfig = field(default_factory=GeminiConfig)
    grok: GrokConfig = field(default_factory=GrokConfig)
    ollama: OllamaConfig = field(default_factory=OllamaConfig)
    
    cache: CacheConfig = field(default_factory=CacheConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    
    default_service: str = "openai"
    log_level: str = "INFO"
    
    @classmethod
    def from_json(cls, file_path: str) -> "EasilyAIConfig":
        """Load configuration from JSON file."""
        with open(file_path, 'r') as f:
            data = json.load(f)
        return cls._from_dict(data)
    
    @classmethod
    def from_env(cls) -> "EasilyAIConfig":
        """Load configuration from environment variables."""
        config = cls()
        
        # Override with environment variables if present
        if default_service := os.getenv("EASILYAI_DEFAULT_SERVICE"):
            config.default_service = default_service
        
        if log_level := os.getenv("EASILYAI_LOG_LEVEL"):
            config.log_level = log_level
            
        # Cache settings from env
        if cache_enabled := os.getenv("EASILYAI_CACHE_ENABLED"):
            config.cache.enabled = cache_enabled.lower() == "true"
            
        if cache_backend := os.getenv("EASILYAI_CACHE_BACKEND"):
            config.cache.backend = cache_backend
            
        if cache_ttl := os.getenv("EASILYAI_CACHE_TTL"):
            config.cache.ttl = int(cache_ttl)
        
        return config
    
    def to_json(self, file_path: str):
        """Save configuration to JSON file."""
        data = self._to_dict()
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def get_service_config(self, service: str) -> ServiceConfig:
        """Get configuration for a specific service."""
        return getattr(self, service.lower(), None)
    
    def get_api_key(self, service: str) -> Optional[str]:
        """Get API key for a specific service."""
        service_config = self.get_service_config(service)
        if service_config:
            return service_config.api_key
        return None
    
    def get_default_model(self, service: str) -> Optional[str]:
        """Get default model for a specific service."""
        service_config = self.get_service_config(service)
        if service_config:
            return service_config.default_model
        return None
    
    def _to_dict(self) -> dict:
        """Convert configuration to dictionary."""
        result = {}
        for field_name, field_value in self.__dict__.items():
            if hasattr(field_value, '__dict__'):
                result[field_name] = {k: v for k, v in field_value.__dict__.items() 
                                    if not k.startswith('_') and v is not None}
            else:
                result[field_name] = field_value
        return result
    
    @classmethod
    def _from_dict(cls, data: dict) -> "EasilyAIConfig":
        """Create configuration from dictionary."""
        config = cls()
        
        for key, value in data.items():
            if hasattr(config, key):
                if isinstance(value, dict) and hasattr(getattr(config, key), '__dict__'):
                    # Update nested config objects
                    for sub_key, sub_value in value.items():
                        setattr(getattr(config, key), sub_key, sub_value)
                else:
                    setattr(config, key, value)
        
        return config


# Global configuration instance
_config = None


def get_config() -> EasilyAIConfig:
    """Get the global configuration instance."""
    global _config
    if _config is None:
        # Try loading from JSON file first
        config_file = Path("easilyai_config.json")
        if config_file.exists():
            _config = EasilyAIConfig.from_json(str(config_file))
        else:
            # Fall back to environment variables
            _config = EasilyAIConfig.from_env()
    return _config


def set_config(config: EasilyAIConfig):
    """Set the global configuration instance."""
    global _config
    _config = config


def reset_config():
    """Reset the global configuration instance."""
    global _config
    _config = None