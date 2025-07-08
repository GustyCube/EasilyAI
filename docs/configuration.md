# Configuration

This guide covers how to configure EasilyAI for different environments, manage API keys securely, and optimize performance settings.

## Environment Configuration

### Environment Variables

The most secure way to manage API keys is through environment variables:

```bash
# .env file
OPENAI_API_KEY=your-openai-key-here
ANTHROPIC_API_KEY=your-anthropic-key-here
GEMINI_API_KEY=your-gemini-key-here
GROK_API_KEY=your-grok-key-here
```

```python
import os
from dotenv import load_dotenv  # pip install python-dotenv
from easilyai import create_app

# Load environment variables
load_dotenv()

# Use environment variables
app = create_app(
    "MyApp",
    "openai",
    os.getenv("OPENAI_API_KEY"),
    "gpt-3.5-turbo"
)
```

### System Environment Variables

Set environment variables at the system level:

```bash
# Linux/macOS
export OPENAI_API_KEY="your-key-here"
export ANTHROPIC_API_KEY="your-key-here"

# Windows
set OPENAI_API_KEY=your-key-here
set ANTHROPIC_API_KEY=your-key-here
```

## Configuration Classes

### Custom Configuration

Create reusable configuration classes:

```python
import os
from dataclasses import dataclass
from typing import Optional

@dataclass
class AIConfig:
    openai_key: Optional[str] = None
    anthropic_key: Optional[str] = None
    gemini_key: Optional[str] = None
    grok_key: Optional[str] = None
    
    def __post_init__(self):
        # Load from environment if not provided
        self.openai_key = self.openai_key or os.getenv("OPENAI_API_KEY")
        self.anthropic_key = self.anthropic_key or os.getenv("ANTHROPIC_API_KEY")
        self.gemini_key = self.gemini_key or os.getenv("GEMINI_API_KEY")
        self.grok_key = self.grok_key or os.getenv("GROK_API_KEY")
    
    def get_key(self, service: str) -> Optional[str]:
        return getattr(self, f"{service}_key", None)

# Usage
config = AIConfig()

from easilyai import create_app

app = create_app("MyApp", "openai", config.get_key("openai"), "gpt-3.5-turbo")
```

### Service-Specific Configuration

```python
from easilyai import create_app

class EasilyAIManager:
    def __init__(self, config: AIConfig):
        self.config = config
        self.apps = {}
    
    def get_app(self, service: str, model: str):
        key = f"{service}_{model}"
        
        if key not in self.apps:
            api_key = self.config.get_key(service)
            if not api_key:
                raise ValueError(f"No API key found for service: {service}")
            
            self.apps[key] = create_app(
                f"{service.title()}App",
                service,
                api_key,
                model
            )
        
        return self.apps[key]
    
    def text_generation(self, prompt: str, service: str = "openai", model: str = "gpt-3.5-turbo"):
        app = self.get_app(service, model)
        return app.request(prompt)
    
    def image_generation(self, prompt: str, **kwargs):
        app = self.get_app("openai", "dall-e-3")
        return app.request(prompt, task_type="generate_image", **kwargs)

# Usage
config = AIConfig()
manager = EasilyAIManager(config)

# Generate text
text = manager.text_generation("Hello, world!")
print(text)

# Generate image
image = manager.image_generation("A beautiful sunset")
print(image)
```

## Model Configuration

### Model Selection Strategies

```python
from easilyai import create_app

class ModelSelector:
    def __init__(self, config: AIConfig):
        self.config = config
        self.model_configs = {
            "fast": {"service": "openai", "model": "gpt-3.5-turbo"},
            "balanced": {"service": "anthropic", "model": "claude-3-haiku-20240307"},
            "powerful": {"service": "openai", "model": "gpt-4"},
            "creative": {"service": "openai", "model": "gpt-4", "temperature": 0.9},
            "precise": {"service": "openai", "model": "gpt-4", "temperature": 0.1}
        }
    
    def get_app(self, strategy: str):
        if strategy not in self.model_configs:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        config = self.model_configs[strategy]
        service = config["service"]
        model = config["model"]
        
        api_key = self.config.get_key(service)
        app = create_app(f"{strategy.title()}App", service, api_key, model)
        
        return app, config
    
    def request(self, prompt: str, strategy: str = "balanced", **kwargs):
        app, config = self.get_app(strategy)
        
        # Merge strategy-specific parameters
        request_params = {**config, **kwargs}
        request_params.pop("service", None)
        request_params.pop("model", None)
        
        return app.request(prompt, **request_params)

# Usage
selector = ModelSelector(AIConfig())

# Use different strategies
fast_response = selector.request("Quick answer: What is 2+2?", strategy="fast")
creative_response = selector.request("Write a creative story", strategy="creative")
precise_response = selector.request("Calculate pi to 5 decimal places", strategy="precise")
```

## Performance Configuration

### Request Optimization

```python
from easilyai import create_app
import time
from functools import wraps

class OptimizedAI:
    def __init__(self, config: AIConfig):
        self.config = config
        self.cache = {}
        self.rate_limits = {
            "openai": {"requests_per_minute": 60, "last_request": 0},
            "anthropic": {"requests_per_minute": 50, "last_request": 0}
        }
    
    def _rate_limit(self, service: str):
        """Simple rate limiting"""
        if service in self.rate_limits:
            limit = self.rate_limits[service]
            now = time.time()
            
            # Calculate minimum time between requests
            min_interval = 60 / limit["requests_per_minute"]
            
            if now - limit["last_request"] < min_interval:
                sleep_time = min_interval - (now - limit["last_request"])
                time.sleep(sleep_time)
            
            self.rate_limits[service]["last_request"] = time.time()
    
    def _cache_key(self, prompt: str, service: str, model: str, **kwargs) -> str:
        """Generate cache key for request"""
        key_parts = [prompt, service, model]
        for k, v in sorted(kwargs.items()):
            key_parts.append(f"{k}={v}")
        return "|".join(key_parts)
    
    def request(self, prompt: str, service: str, model: str, use_cache: bool = True, **kwargs):
        # Check cache first
        if use_cache:
            cache_key = self._cache_key(prompt, service, model, **kwargs)
            if cache_key in self.cache:
                return self.cache[cache_key]
        
        # Apply rate limiting
        self._rate_limit(service)
        
        # Make request
        api_key = self.config.get_key(service)
        app = create_app("OptimizedApp", service, api_key, model)
        
        response = app.request(prompt, **kwargs)
        
        # Cache response
        if use_cache:
            self.cache[cache_key] = response
        
        return response

# Usage
optimized = OptimizedAI(AIConfig())

# First request - will be cached
response1 = optimized.request("What is Python?", "openai", "gpt-3.5-turbo")

# Second identical request - will use cache
response2 = optimized.request("What is Python?", "openai", "gpt-3.5-turbo")
```

### Batch Processing Configuration

```python
from easilyai import create_app
import asyncio
from concurrent.futures import ThreadPoolExecutor
import time

class BatchProcessor:
    def __init__(self, config: AIConfig, max_workers: int = 5):
        self.config = config
        self.max_workers = max_workers
        self.apps = {}
    
    def get_app(self, service: str, model: str):
        key = f"{service}_{model}"
        if key not in self.apps:
            api_key = self.config.get_key(service)
            self.apps[key] = create_app(f"BatchApp_{key}", service, api_key, model)
        return self.apps[key]
    
    def process_single(self, prompt: str, service: str, model: str, **kwargs):
        """Process a single request with error handling"""
        try:
            app = self.get_app(service, model)
            response = app.request(prompt, **kwargs)
            return {"success": True, "response": response, "prompt": prompt}
        except Exception as e:
            return {"success": False, "error": str(e), "prompt": prompt}
    
    def process_batch(self, requests: list, delay: float = 1.0):
        """Process multiple requests with rate limiting"""
        results = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = []
            
            for request in requests:
                future = executor.submit(self.process_single, **request)
                futures.append(future)
                
                # Rate limiting between submissions
                if delay > 0:
                    time.sleep(delay)
            
            # Collect results
            for future in futures:
                results.append(future.result())
        
        return results

# Usage
processor = BatchProcessor(AIConfig(), max_workers=3)

requests = [
    {"prompt": "What is AI?", "service": "openai", "model": "gpt-3.5-turbo"},
    {"prompt": "Explain machine learning", "service": "anthropic", "model": "claude-3-haiku-20240307"},
    {"prompt": "What is deep learning?", "service": "openai", "model": "gpt-3.5-turbo"}
]

results = processor.process_batch(requests, delay=0.5)

for result in results:
    if result["success"]:
        print(f"✓ {result['prompt']}: {result['response'][:100]}...")
    else:
        print(f"✗ {result['prompt']}: {result['error']}")
```

## Logging Configuration

### Basic Logging

```python
import logging
from easilyai import create_app

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('easilyai.log'),
        logging.StreamHandler()
    ]
)

class LoggedAI:
    def __init__(self, config: AIConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def request(self, prompt: str, service: str, model: str, **kwargs):
        self.logger.info(f"Making request to {service} {model}")
        
        start_time = time.time()
        
        try:
            api_key = self.config.get_key(service)
            app = create_app("LoggedApp", service, api_key, model)
            response = app.request(prompt, **kwargs)
            
            end_time = time.time()
            duration = end_time - start_time
            
            self.logger.info(f"Request completed in {duration:.2f}s")
            return response
        
        except Exception as e:
            self.logger.error(f"Request failed: {e}")
            raise

# Usage
logged_ai = LoggedAI(AIConfig())
response = logged_ai.request("Hello!", "openai", "gpt-3.5-turbo")
```

### Advanced Logging with Metrics

```python
import logging
import time
from collections import defaultdict

class MetricsLogger:
    def __init__(self):
        self.metrics = defaultdict(list)
        self.logger = logging.getLogger(__name__)
    
    def log_request(self, service: str, model: str, duration: float, success: bool):
        metric_key = f"{service}_{model}"
        self.metrics[metric_key].append({
            "duration": duration,
            "success": success,
            "timestamp": time.time()
        })
    
    def get_stats(self, service: str, model: str) -> dict:
        metric_key = f"{service}_{model}"
        data = self.metrics[metric_key]
        
        if not data:
            return {"count": 0}
        
        durations = [d["duration"] for d in data]
        successes = [d["success"] for d in data]
        
        return {
            "count": len(data),
            "success_rate": sum(successes) / len(successes) * 100,
            "avg_duration": sum(durations) / len(durations),
            "min_duration": min(durations),
            "max_duration": max(durations)
        }
    
    def print_stats(self):
        for key, data in self.metrics.items():
            service, model = key.split("_", 1)
            stats = self.get_stats(service, model)
            
            print(f"{service} {model}:")
            print(f"  Requests: {stats['count']}")
            if stats['count'] > 0:
                print(f"  Success Rate: {stats['success_rate']:.1f}%")
                print(f"  Avg Duration: {stats['avg_duration']:.2f}s")
                print(f"  Duration Range: {stats['min_duration']:.2f}s - {stats['max_duration']:.2f}s")
            print()

# Usage
metrics = MetricsLogger()

class MonitoredAI:
    def __init__(self, config: AIConfig, metrics_logger: MetricsLogger):
        self.config = config
        self.metrics = metrics_logger
    
    def request(self, prompt: str, service: str, model: str, **kwargs):
        start_time = time.time()
        success = False
        
        try:
            api_key = self.config.get_key(service)
            app = create_app("MonitoredApp", service, api_key, model)
            response = app.request(prompt, **kwargs)
            success = True
            return response
        
        except Exception as e:
            raise
        
        finally:
            duration = time.time() - start_time
            self.metrics.log_request(service, model, duration, success)

# Usage
monitored = MonitoredAI(AIConfig(), metrics)

# Make some requests
monitored.request("Hello!", "openai", "gpt-3.5-turbo")
monitored.request("How are you?", "anthropic", "claude-3-haiku-20240307")

# Print statistics
metrics.print_stats()
```

## Configuration Files

### JSON Configuration

```python
import json
from pathlib import Path

class ConfigManager:
    def __init__(self, config_file: str = "easilyai_config.json"):
        self.config_file = Path(config_file)
        self.config = self.load_config()
    
    def load_config(self) -> dict:
        if self.config_file.exists():
            with open(self.config_file, 'r') as f:
                return json.load(f)
        else:
            return self.default_config()
    
    def default_config(self) -> dict:
        return {
            "default_service": "openai",
            "default_models": {
                "openai": "gpt-3.5-turbo",
                "anthropic": "claude-3-haiku-20240307",
                "gemini": "gemini-1.5-flash"
            },
            "rate_limits": {
                "openai": 60,
                "anthropic": 50
            },
            "timeouts": {
                "openai": 30,
                "anthropic": 30
            },
            "cache_enabled": True,
            "log_level": "INFO"
        }
    
    def save_config(self):
        with open(self.config_file, 'w') as f:
            json.dump(self.config, f, indent=2)
    
    def get(self, key: str, default=None):
        return self.config.get(key, default)
    
    def set(self, key: str, value):
        self.config[key] = value
        self.save_config()

# Usage
config_manager = ConfigManager()

# Get default service
default_service = config_manager.get("default_service")
print(f"Default service: {default_service}")

# Update configuration
config_manager.set("default_service", "anthropic")
```

## Best Practices

1. **Never commit API keys**: Always use environment variables or secure configuration files
2. **Use configuration classes**: Centralize configuration management
3. **Implement rate limiting**: Respect API limits to avoid being blocked
4. **Add logging**: Monitor performance and errors
5. **Cache responses**: Reduce API calls for repeated requests
6. **Handle errors gracefully**: Implement retry logic and fallbacks
7. **Use appropriate models**: Match model capabilities to your use case
8. **Monitor costs**: Track API usage and costs

## Security Considerations

- Store API keys securely (environment variables, key management services)
- Rotate API keys regularly
- Use least privilege principle for API access
- Implement proper error handling to avoid exposing sensitive information
- Consider using API key encryption for stored configurations
- Monitor API usage for unusual patterns

This configuration system provides a robust foundation for managing EasilyAI in production environments while maintaining security and performance best practices.