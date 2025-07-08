# Custom AI Integration

EasilyAI's extensible architecture allows you to integrate your own AI models and services seamlessly with the unified API. Whether you're working with local models, proprietary APIs, or experimental frameworks, the custom AI system provides a consistent interface.

::: tip 🚀 Fully Implemented Feature
The custom AI integration system is fully implemented and production-ready. You can start using it immediately!
:::

## Core Concepts

### CustomAIService Base Class

All custom AI implementations inherit from the `CustomAIService` base class, which defines the standard interface:

```python
from easilyai.custom_ai import CustomAIService

class CustomAIService:
    def generate_text(self, prompt: str, **kwargs) -> str:
        """Generate text based on a prompt"""
        raise NotImplementedError
    
    def generate_image(self, prompt: str, **kwargs) -> str:
        """Generate image URL/path based on a prompt"""
        raise NotImplementedError
    
    def text_to_speech(self, text: str, **kwargs) -> str:
        """Convert text to speech, return audio file path/URL"""
        raise NotImplementedError
```

### Registration System

Use the global registry to make your custom AI available throughout your application:

```python
from easilyai.custom_ai import register_custom_ai

# Register your custom AI service
register_custom_ai("service_name", YourCustomAIClass)
```

## Implementation Examples

### Basic Custom AI Service

::: code-group

```python [Simple Implementation]
from easilyai.custom_ai import CustomAIService, register_custom_ai
from easilyai import create_app

class MyCustomAI(CustomAIService):
    def __init__(self, api_key=None, model=None):
        self.api_key = api_key
        self.model = model
    
    def generate_text(self, prompt, **kwargs):
        # Your custom text generation logic here
        return f"Custom AI [{self.model}]: {prompt.upper()}"
    
    def generate_image(self, prompt, **kwargs):
        # Your custom image generation logic
        return f"https://example.com/generated_image_{hash(prompt)}.png"
    
    def text_to_speech(self, text, **kwargs):
        # Your custom TTS logic
        return f"/tmp/tts_{hash(text)}.wav"

# Register the service
register_custom_ai("my_custom_ai", MyCustomAI)

# Use with EasilyAI
app = create_app("CustomApp", "my_custom_ai", "your-api-key", "your-model")
response = app.request("text", "Hello from custom AI!")
print(response)
```

```python [Local Model Integration]
import requests
from easilyai.custom_ai import CustomAIService, register_custom_ai

class LocalLlamaAI(CustomAIService):
    def __init__(self, api_key=None, model=None):
        self.base_url = "http://localhost:11434"
        self.model = model or "llama2"
    
    def generate_text(self, prompt, **kwargs):
        response = requests.post(
            f"{self.base_url}/api/generate",
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": False
            }
        )
        return response.json().get("response", "")
    
    def generate_image(self, prompt, **kwargs):
        # Integrate with local Stable Diffusion or similar
        raise NotImplementedError("Image generation not supported")
    
    def text_to_speech(self, text, **kwargs):
        # Integrate with local TTS engine
        raise NotImplementedError("TTS not supported")

register_custom_ai("local_llama", LocalLlamaAI)
```

```python [Hugging Face Integration]
from transformers import pipeline
from easilyai.custom_ai import CustomAIService, register_custom_ai

class HuggingFaceAI(CustomAIService):
    def __init__(self, api_key=None, model=None):
        self.model = model or "gpt2"
        self.generator = pipeline("text-generation", model=self.model)
    
    def generate_text(self, prompt, **kwargs):
        max_length = kwargs.get("max_length", 100)
        result = self.generator(
            prompt, 
            max_length=max_length, 
            num_return_sequences=1
        )
        return result[0]["generated_text"]
    
    def generate_image(self, prompt, **kwargs):
        # Could integrate with diffusers library
        from diffusers import StableDiffusionPipeline
        # Implementation details...
        pass
    
    def text_to_speech(self, text, **kwargs):
        # Could use transformers TTS models
        pass

register_custom_ai("huggingface", HuggingFaceAI)
```

:::

### Advanced Features

#### Error Handling and Validation

```python
from easilyai.custom_ai import CustomAIService
from easilyai.exceptions import EasilyAIError

class RobustCustomAI(CustomAIService):
    def __init__(self, api_key=None, model=None):
        if not api_key:
            raise EasilyAIError("API key is required for RobustCustomAI")
        self.api_key = api_key
        self.model = model
    
    def generate_text(self, prompt, **kwargs):
        try:
            # Validate input
            if not prompt or len(prompt.strip()) == 0:
                raise EasilyAIError("Prompt cannot be empty")
            
            # Your API call logic
            response = self._make_api_call(prompt, **kwargs)
            
            # Validate response
            if not response:
                raise EasilyAIError("Empty response from custom AI service")
            
            return response
            
        except Exception as e:
            raise EasilyAIError(f"Custom AI text generation failed: {str(e)}")
    
    def _make_api_call(self, prompt, **kwargs):
        # Your actual API implementation
        return f"Response for: {prompt}"
```

#### Configuration and Environment Variables

```python
import os
from easilyai.custom_ai import CustomAIService, register_custom_ai

class ConfigurableAI(CustomAIService):
    def __init__(self, api_key=None, model=None):
        # Support multiple configuration sources
        self.api_key = (
            api_key or 
            os.getenv("CUSTOM_AI_API_KEY") or 
            os.getenv("CUSTOM_AI_TOKEN")
        )
        self.model = model or os.getenv("CUSTOM_AI_MODEL", "default-model")
        self.base_url = os.getenv("CUSTOM_AI_BASE_URL", "https://api.example.com")
        self.timeout = int(os.getenv("CUSTOM_AI_TIMEOUT", "30"))
    
    def generate_text(self, prompt, **kwargs):
        # Use configuration in your implementation
        pass

# Register with a more descriptive name
register_custom_ai("configurable_ai", ConfigurableAI)
```

## Integration with EasilyAI Apps

### Using Custom AI in Applications

```python
from easilyai import create_app, create_tts_app
from easilyai.pipeline import EasilyAIPipeline

# Create apps with your custom AI
text_app = create_app("TextApp", "my_custom_ai", "api-key", "model-name")
tts_app = create_tts_app("TTSApp", "my_custom_ai", "api-key")

# Use in pipelines
pipeline = EasilyAIPipeline()
pipeline.add_task("generate_text", "Write a story", service="my_custom_ai")
pipeline.add_task("text_to_speech", "{previous_output}", service="my_custom_ai")

results = pipeline.run()
```

### Testing Custom AI Services

```python
import pytest
from easilyai.custom_ai import CustomAIService, register_custom_ai
from easilyai import create_app

class MockAI(CustomAIService):
    def generate_text(self, prompt, **kwargs):
        return f"Mock response: {prompt}"

def test_custom_ai_integration():
    # Register mock AI for testing
    register_custom_ai("test_ai", MockAI)
    
    # Test app creation
    app = create_app("TestApp", "test_ai", "fake-key", "fake-model")
    
    # Test functionality
    response = app.request("text", "test prompt")
    assert "Mock response: test prompt" in response
    
    # Clean up
    # Note: In real tests, you might want to reset the registry
```

## Best Practices

::: tip 🎯 Implementation Guidelines

1. **Always implement error handling** - Wrap API calls in try-catch blocks
2. **Validate inputs and outputs** - Check for empty prompts and responses
3. **Use type hints** - Make your code more maintainable
4. **Support configuration** - Allow users to configure your service
5. **Test thoroughly** - Write unit tests for your custom implementations
6. **Document your API** - Include docstrings and usage examples

:::

::: warning ⚠️ Common Pitfalls

- **Don't forget to register** your service before using it
- **Handle rate limits** appropriately in your implementation
- **Validate API keys** and other credentials
- **Consider async operations** for better performance
- **Implement proper logging** for debugging

:::

## Real-World Examples

### Enterprise API Integration

```python
class EnterpiseAI(CustomAIService):
    """Integration with enterprise AI platform"""
    
    def __init__(self, api_key=None, model=None):
        self.api_key = api_key
        self.model = model
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {api_key}",
            "User-Agent": "EasilyAI-CustomClient/1.0"
        })
    
    def generate_text(self, prompt, **kwargs):
        response = self.session.post(
            "https://enterprise-ai.company.com/v1/generate",
            json={
                "model": self.model,
                "prompt": prompt,
                "temperature": kwargs.get("temperature", 0.7),
                "max_tokens": kwargs.get("max_tokens", 150)
            },
            timeout=30
        )
        response.raise_for_status()
        return response.json()["generated_text"]
```

### Multi-Model Service

```python
class MultiModelAI(CustomAIService):
    """Custom AI that routes to different models based on task"""
    
    def __init__(self, api_key=None, model=None):
        self.config = {
            "text_model": "gpt-4",
            "code_model": "codex",
            "creative_model": "claude-3"
        }
    
    def generate_text(self, prompt, **kwargs):
        # Route based on prompt characteristics
        if "code" in prompt.lower() or "```" in prompt:
            model = self.config["code_model"]
        elif "creative" in prompt.lower() or "story" in prompt.lower():
            model = self.config["creative_model"]
        else:
            model = self.config["text_model"]
        
        # Delegate to appropriate model
        return self._call_model(model, prompt, **kwargs)
    
    def _call_model(self, model, prompt, **kwargs):
        # Implementation for calling specific models
        pass
```

## Next Steps

- **[Pipeline Integration](./pipelines.md)** - Use custom AI in complex workflows
- **[Error Handling](./errorhandling.md)** - Robust error management
- **[API Reference](./api/customai.md)** - Complete API documentation
- **[Examples](./examples.md)** - More real-world usage examples