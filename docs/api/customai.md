# Custom AI Framework API Reference

The Custom AI framework in EasilyAI allows you to integrate your own AI services and models, extending the library's capabilities beyond the built-in providers.

## Core Classes

### CustomAIService

Base class for implementing custom AI services.

```python
class CustomAIService:
    def __init__(api_key: str, model: str)
```

**Parameters:**
- `api_key` (str): API key or authentication token for your service
- `model` (str): Model identifier for your service

**Abstract Methods:**
You must implement these methods in your custom service:

#### generate_text()

```python
def generate_text(prompt: str, **kwargs) -> str
```

**Parameters:**
- `prompt` (str): Input text prompt
- `**kwargs`: Additional parameters specific to your service

**Returns:**
- `str`: Generated text response

#### generate_image() (Optional)

```python
def generate_image(prompt: str, **kwargs) -> str
```

**Parameters:**
- `prompt` (str): Image description or prompt
- `**kwargs`: Additional parameters for image generation

**Returns:**
- `str`: Image URL or file path

#### text_to_speech() (Optional)

```python
def text_to_speech(text: str, **kwargs) -> str
```

**Parameters:**
- `text` (str): Text to convert to speech
- **kwargs**: Additional parameters for TTS

**Returns:**
- `str`: Audio file path or URL

## Registration Functions

### register_custom_ai()

Register a custom AI service with EasilyAI.

```python
def register_custom_ai(service_name: str, service_class: type) -> None
```

**Parameters:**
- `service_name` (str): Unique name for your service
- `service_class` (type): Your custom service class

**Example:**
```python
from easilyai import register_custom_ai
from easilyai.custom_ai import CustomAIService

class MyCustomService(CustomAIService):
    def generate_text(self, prompt, **kwargs):
        # Your implementation here
        return f"Custom response to: {prompt}"

register_custom_ai("mycustom", MyCustomService)
```

### unregister_custom_ai()

Remove a custom AI service from the registry.

```python
def unregister_custom_ai(service_name: str) -> bool
```

**Parameters:**
- `service_name` (str): Name of the service to remove

**Returns:**
- `bool`: True if service was removed, False if not found

### list_custom_ai_services()

Get a list of all registered custom AI services.

```python
def list_custom_ai_services() -> list
```

**Returns:**
- `list`: List of registered custom service names

## Implementation Examples

### Basic Custom Service

```python
from easilyai.custom_ai import CustomAIService
from easilyai import register_custom_ai, create_app

class EchoService(CustomAIService):
    """Simple echo service for testing"""
    
    def __init__(self, api_key, model):
        super().__init__(api_key, model)
        self.model_config = {
            "echo": "Simple echo",
            "reverse": "Reverse text",
            "upper": "Uppercase text"
        }
    
    def generate_text(self, prompt, **kwargs):
        """Generate text based on model type"""
        if self.model == "echo":
            return f"Echo: {prompt}"
        elif self.model == "reverse":
            return f"Reversed: {prompt[::-1]}"
        elif self.model == "upper":
            return f"Upper: {prompt.upper()}"
        else:
            return f"Unknown model {self.model}: {prompt}"
    
    def generate_image(self, prompt, **kwargs):
        """Mock image generation"""
        return f"mock://image/{self.model}/{prompt.replace(' ', '_')}.jpg"
    
    def text_to_speech(self, text, **kwargs):
        """Mock TTS"""
        return f"mock://audio/{self.model}/{len(text)}_chars.mp3"

# Register the service
register_custom_ai("echo", EchoService)

# Use the service
app = create_app("EchoApp", "echo", "no-key-needed", "upper")
response = app.request("hello world")
print(response)  # Output: "Upper: HELLO WORLD"
```

### API-Based Custom Service

```python
import requests
from easilyai.custom_ai import CustomAIService
from easilyai import register_custom_ai

class CustomAPIService(CustomAIService):
    """Custom service that calls an external API"""
    
    def __init__(self, api_key, model):
        super().__init__(api_key, model)
        self.base_url = "https://api.your-service.com"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
    
    def generate_text(self, prompt, **kwargs):
        """Call external API for text generation"""
        payload = {
            "model": self.model,
            "prompt": prompt,
            "max_tokens": kwargs.get("max_tokens", 100),
            "temperature": kwargs.get("temperature", 0.7)
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/generate",
                json=payload,
                headers=self.headers,
                timeout=30
            )
            response.raise_for_status()
            
            data = response.json()
            return data.get("text", "No response")
            
        except requests.RequestException as e:
            raise Exception(f"API request failed: {e}")
    
    def generate_image(self, prompt, **kwargs):
        """Call external API for image generation"""
        payload = {
            "prompt": prompt,
            "size": kwargs.get("size", "512x512"),
            "model": self.model
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/image",
                json=payload,
                headers=self.headers,
                timeout=60
            )
            response.raise_for_status()
            
            data = response.json()
            return data.get("image_url", "No image generated")
            
        except requests.RequestException as e:
            raise Exception(f"Image API request failed: {e}")

# Register the service
register_custom_ai("customapi", CustomAPIService)

# Use the service
app = create_app("CustomAPI", "customapi", "your-api-key", "custom-model-v1")
response = app.request("Generate a story about robots")
```

### Local Model Service

```python
import subprocess
import json
from easilyai.custom_ai import CustomAIService
from easilyai import register_custom_ai

class LocalModelService(CustomAIService):
    """Service for running local models via command line"""
    
    def __init__(self, api_key, model):
        super().__init__(api_key, model)
        self.model_path = api_key  # Use api_key as model path
        self.model_name = model
    
    def generate_text(self, prompt, **kwargs):
        """Run local model via subprocess"""
        try:
            # Prepare command
            cmd = [
                "python", "local_model_runner.py",
                "--model", self.model_path,
                "--prompt", prompt,
                "--max_tokens", str(kwargs.get("max_tokens", 100))
            ]
            
            # Run command
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=kwargs.get("timeout", 60)
            )
            
            if result.returncode != 0:
                raise Exception(f"Model execution failed: {result.stderr}")
            
            # Parse output
            try:
                output = json.loads(result.stdout)
                return output.get("text", "No output")
            except json.JSONDecodeError:
                return result.stdout.strip()
                
        except subprocess.TimeoutExpired:
            raise Exception("Model execution timed out")
        except Exception as e:
            raise Exception(f"Local model error: {e}")

# Register the service
register_custom_ai("localmodel", LocalModelService)

# Use the service (api_key is model path)
app = create_app("Local", "localmodel", "/path/to/model", "my-local-model")
response = app.request("Hello world")
```

### Multi-Model Service

```python
from easilyai.custom_ai import CustomAIService
from easilyai import register_custom_ai

class MultiModelService(CustomAIService):
    """Service that delegates to multiple underlying models"""
    
    def __init__(self, api_key, model):
        super().__init__(api_key, model)
        
        # Initialize different model clients
        self.models = {
            "fast": self._init_fast_model(),
            "quality": self._init_quality_model(),
            "creative": self._init_creative_model()
        }
    
    def _init_fast_model(self):
        """Initialize fast model client"""
        # Your fast model initialization
        return {"type": "fast", "endpoint": "fast-api"}
    
    def _init_quality_model(self):
        """Initialize quality model client"""
        # Your quality model initialization
        return {"type": "quality", "endpoint": "quality-api"}
    
    def _init_creative_model(self):
        """Initialize creative model client"""
        # Your creative model initialization
        return {"type": "creative", "endpoint": "creative-api"}
    
    def generate_text(self, prompt, **kwargs):
        """Route to appropriate model based on configuration"""
        model_config = self.models.get(self.model, self.models["fast"])
        
        # Route based on model type
        if model_config["type"] == "fast":
            return self._generate_fast(prompt, **kwargs)
        elif model_config["type"] == "quality":
            return self._generate_quality(prompt, **kwargs)
        elif model_config["type"] == "creative":
            return self._generate_creative(prompt, **kwargs)
        else:
            raise ValueError(f"Unknown model: {self.model}")
    
    def _generate_fast(self, prompt, **kwargs):
        """Fast generation implementation"""
        return f"Fast response to: {prompt}"
    
    def _generate_quality(self, prompt, **kwargs):
        """Quality generation implementation"""
        return f"High-quality response to: {prompt}"
    
    def _generate_creative(self, prompt, **kwargs):
        """Creative generation implementation"""
        return f"Creative response to: {prompt}"

# Register the service
register_custom_ai("multimodel", MultiModelService)

# Use different models
fast_app = create_app("Fast", "multimodel", "your-key", "fast")
quality_app = create_app("Quality", "multimodel", "your-key", "quality")
creative_app = create_app("Creative", "multimodel", "your-key", "creative")
```

### Caching Custom Service

```python
import hashlib
import json
import time
from pathlib import Path
from easilyai.custom_ai import CustomAIService
from easilyai import register_custom_ai

class CachingCustomService(CustomAIService):
    """Custom service with built-in caching"""
    
    def __init__(self, api_key, model):
        super().__init__(api_key, model)
        self.cache_dir = Path("custom_service_cache")
        self.cache_dir.mkdir(exist_ok=True)
        self.cache_ttl = 3600  # 1 hour
    
    def _get_cache_key(self, prompt, **kwargs):
        """Generate cache key for request"""
        data = {
            "prompt": prompt,
            "model": self.model,
            **kwargs
        }
        data_str = json.dumps(data, sort_keys=True)
        return hashlib.md5(data_str.encode()).hexdigest()
    
    def _get_cached_response(self, cache_key):
        """Get cached response if available and valid"""
        cache_file = self.cache_dir / f"{cache_key}.json"
        
        if cache_file.exists():
            with open(cache_file, 'r') as f:
                cached_data = json.load(f)
            
            # Check if cache is still valid
            if time.time() - cached_data["timestamp"] < self.cache_ttl:
                return cached_data["response"]
        
        return None
    
    def _cache_response(self, cache_key, response):
        """Cache a response"""
        cache_file = self.cache_dir / f"{cache_key}.json"
        
        cache_data = {
            "response": response,
            "timestamp": time.time()
        }
        
        with open(cache_file, 'w') as f:
            json.dump(cache_data, f)
    
    def generate_text(self, prompt, **kwargs):
        """Generate text with caching"""
        cache_key = self._get_cache_key(prompt, **kwargs)
        
        # Check cache first
        cached_response = self._get_cached_response(cache_key)
        if cached_response:
            return cached_response
        
        # Generate new response
        response = self._actual_generate_text(prompt, **kwargs)
        
        # Cache the response
        self._cache_response(cache_key, response)
        
        return response
    
    def _actual_generate_text(self, prompt, **kwargs):
        """Actual text generation implementation"""
        # Your actual implementation here
        return f"Generated response to: {prompt}"

# Register the service
register_custom_ai("caching", CachingCustomService)
```

## Error Handling

### Custom Exceptions

```python
from easilyai.exceptions import EasilyAIException

class CustomServiceException(EasilyAIException):
    """Exception for custom service errors"""
    pass

class RateLimitException(CustomServiceException):
    """Exception for rate limit errors"""
    pass

class ModelNotFoundException(CustomServiceException):
    """Exception for model not found errors"""
    pass

class RobustCustomService(CustomAIService):
    """Custom service with comprehensive error handling"""
    
    def generate_text(self, prompt, **kwargs):
        try:
            # Your implementation here
            response = self._call_external_api(prompt, **kwargs)
            return response
            
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:
                raise RateLimitException("Rate limit exceeded")
            elif e.response.status_code == 404:
                raise ModelNotFoundException(f"Model {self.model} not found")
            else:
                raise CustomServiceException(f"API error: {e}")
        
        except requests.exceptions.RequestException as e:
            raise CustomServiceException(f"Network error: {e}")
        
        except Exception as e:
            raise CustomServiceException(f"Unexpected error: {e}")
```

## Service Configuration

### Configuration Management

```python
import os
import json
from easilyai.custom_ai import CustomAIService

class ConfigurableCustomService(CustomAIService):
    """Custom service with configuration management"""
    
    def __init__(self, api_key, model):
        super().__init__(api_key, model)
        self.config = self._load_config()
    
    def _load_config(self):
        """Load configuration from file or environment"""
        config_file = os.getenv("CUSTOM_SERVICE_CONFIG", "custom_service_config.json")
        
        if os.path.exists(config_file):
            with open(config_file, 'r') as f:
                return json.load(f)
        
        # Default configuration
        return {
            "base_url": os.getenv("CUSTOM_SERVICE_URL", "https://api.example.com"),
            "timeout": int(os.getenv("CUSTOM_SERVICE_TIMEOUT", "30")),
            "max_retries": int(os.getenv("CUSTOM_SERVICE_RETRIES", "3")),
            "models": {
                "default": {"max_tokens": 100, "temperature": 0.7},
                "creative": {"max_tokens": 200, "temperature": 0.9},
                "precise": {"max_tokens": 50, "temperature": 0.1}
            }
        }
    
    def generate_text(self, prompt, **kwargs):
        """Generate text using configuration"""
        model_config = self.config["models"].get(self.model, self.config["models"]["default"])
        
        # Merge model config with kwargs
        params = {**model_config, **kwargs}
        
        # Use configuration for API call
        return self._call_api_with_config(prompt, params)
```

## Testing Custom Services

### Unit Testing

```python
import unittest
from unittest.mock import patch, MagicMock
from easilyai.custom_ai import CustomAIService
from easilyai import register_custom_ai, create_app

class TestCustomService(unittest.TestCase):
    
    def setUp(self):
        """Set up test service"""
        class MockService(CustomAIService):
            def generate_text(self, prompt, **kwargs):
                return f"Mock: {prompt}"
        
        register_custom_ai("test", MockService)
        self.app = create_app("Test", "test", "fake-key", "test-model")
    
    def test_text_generation(self):
        """Test text generation"""
        response = self.app.request("Hello")
        self.assertEqual(response, "Mock: Hello")
    
    @patch('requests.post')
    def test_api_service(self, mock_post):
        """Test API-based service"""
        # Mock API response
        mock_response = MagicMock()
        mock_response.json.return_value = {"text": "API response"}
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response
        
        # Test service
        class APIService(CustomAIService):
            def generate_text(self, prompt, **kwargs):
                import requests
                response = requests.post("https://api.test.com", json={"prompt": prompt})
                response.raise_for_status()
                return response.json()["text"]
        
        register_custom_ai("apitest", APIService)
        app = create_app("APITest", "apitest", "fake-key", "test")
        
        response = app.request("Test prompt")
        self.assertEqual(response, "API response")
        mock_post.assert_called_once()

if __name__ == "__main__":
    unittest.main()
```

## Best Practices

### 1. Implement Proper Error Handling

```python
def generate_text(self, prompt, **kwargs):
    try:
        # Your implementation
        return result
    except SpecificException as e:
        raise CustomServiceException(f"Specific error: {e}")
    except Exception as e:
        raise CustomServiceException(f"Unexpected error: {e}")
```

### 2. Support Standard Parameters

```python
def generate_text(self, prompt, **kwargs):
    # Support common parameters
    temperature = kwargs.get("temperature", 0.7)
    max_tokens = kwargs.get("max_tokens", 100)
    
    # Your implementation using these parameters
```

### 3. Provide Clear Documentation

```python
class MyCustomService(CustomAIService):
    """
    Custom AI service for XYZ provider.
    
    Supported models:
    - model-v1: Fast, general-purpose model
    - model-v2: High-quality, slower model
    
    Parameters:
    - temperature (float): Controls randomness (0.0-1.0)
    - max_tokens (int): Maximum response length
    """
    
    def generate_text(self, prompt, **kwargs):
        """
        Generate text using XYZ API.
        
        Args:
            prompt (str): Input prompt
            **kwargs: Additional parameters
        
        Returns:
            str: Generated text
        
        Raises:
            CustomServiceException: If API call fails
        """
        pass
```

### 4. Implement Caching for Expensive Operations

```python
# Use the CachingCustomService example above
# for services with expensive API calls
```

### 5. Validate Input Parameters

```python
def generate_text(self, prompt, **kwargs):
    # Validate inputs
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("Prompt must be a non-empty string")
    
    temperature = kwargs.get("temperature", 0.7)
    if not 0.0 <= temperature <= 1.0:
        raise ValueError("Temperature must be between 0.0 and 1.0")
    
    # Continue with implementation
```

The Custom AI framework provides powerful extensibility for EasilyAI, allowing you to integrate any AI service or model while maintaining the same simple interface.