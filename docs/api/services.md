# Service Classes API Reference

This page documents the individual service classes that power EasilyAI's functionality.

## Base Service Interface

All service classes implement a common interface for consistency across different AI providers.

### ServiceInterface

```python
class ServiceInterface:
    def __init__(api_key: str, model: str)
    def generate_text(prompt: str, **kwargs) -> str
    def generate_image(prompt: str, **kwargs) -> str  # Optional
    def text_to_speech(text: str, **kwargs) -> str    # Optional
```

## OpenAI Service

### OpenAIService

Provides access to OpenAI's GPT, DALL-E, and TTS models.

```python
class OpenAIService:
    def __init__(api_key: str, model: str)
```

**Supported Models:**
- Text: `gpt-3.5-turbo`, `gpt-4`, `gpt-4-turbo`
- Image: `dall-e-2`, `dall-e-3`
- TTS: `tts-1`, `tts-1-hd`

**Methods:**

#### generate_text()

```python
def generate_text(
    prompt: str,
    temperature: float = 0.7,
    max_tokens: int = None,
    top_p: float = 1.0,
    frequency_penalty: float = 0.0,
    presence_penalty: float = 0.0,
    **kwargs
) -> str
```

**Parameters:**
- `temperature` (float): Controls randomness (0.0 to 2.0)
- `max_tokens` (int): Maximum tokens to generate
- `top_p` (float): Nucleus sampling parameter
- `frequency_penalty` (float): Penalty for frequent tokens
- `presence_penalty` (float): Penalty for new topics

#### generate_image()

```python
def generate_image(
    prompt: str,
    size: str = "1024x1024",
    quality: str = "standard",
    style: str = "vivid",
    n: int = 1,
    **kwargs
) -> str
```

**Parameters:**
- `size` (str): Image dimensions
- `quality` (str): 'standard' or 'hd' (DALL-E 3 only)
- `style` (str): 'natural' or 'vivid' (DALL-E 3 only)
- `n` (int): Number of images (DALL-E 2 only)

#### text_to_speech()

```python
def text_to_speech(
    text: str,
    voice: str = "alloy",
    response_format: str = "mp3",
    speed: float = 1.0,
    **kwargs
) -> str
```

**Parameters:**
- `voice` (str): Voice selection ('alloy', 'echo', 'fable', 'onyx', 'nova', 'shimmer')
- `response_format` (str): Audio format ('mp3', 'opus', 'aac', 'flac')
- `speed` (float): Speech speed (0.25 to 4.0)

## Anthropic Service

### AnthropicService

Provides access to Claude models.

```python
class AnthropicService:
    def __init__(api_key: str, model: str)
```

**Supported Models:**
- `claude-3-haiku-20240307`
- `claude-3-sonnet-20240229`
- `claude-3-opus-20240229`

**Methods:**

#### generate_text()

```python
def generate_text(
    prompt: str,
    temperature: float = 0.7,
    max_tokens: int = 1000,
    top_p: float = 0.9,
    top_k: int = 250,
    **kwargs
) -> str
```

**Parameters:**
- `temperature` (float): Controls randomness (0.0 to 1.0)
- `max_tokens` (int): Maximum tokens to generate
- `top_p` (float): Nucleus sampling parameter
- `top_k` (int): Top-k sampling parameter

## Gemini Service

### GeminiService

Provides access to Google Gemini models.

```python
class GeminiService:
    def __init__(api_key: str, model: str)
```

**Supported Models:**
- `gemini-1.5-flash`
- `gemini-1.5-pro`
- `gemini-1.0-pro`

**Methods:**

#### generate_text()

```python
def generate_text(
    prompt: str,
    temperature: float = 0.7,
    max_output_tokens: int = 1000,
    top_p: float = 0.8,
    top_k: int = 40,
    **kwargs
) -> str
```

**Parameters:**
- `temperature` (float): Controls randomness (0.0 to 1.0)
- `max_output_tokens` (int): Maximum tokens to generate
- `top_p` (float): Nucleus sampling parameter
- `top_k` (int): Top-k sampling parameter

## Grok Service

### GrokService

Provides access to X.AI Grok models.

```python
class GrokService:
    def __init__(api_key: str, model: str)
```

**Supported Models:**
- `grok-beta`

**Methods:**

#### generate_text()

```python
def generate_text(
    prompt: str,
    temperature: float = 0.7,
    max_tokens: int = 1000,
    top_p: float = 0.9,
    **kwargs
) -> str
```

## Ollama Service

### OllamaService

Provides access to local Ollama models.

```python
class OllamaService:
    def __init__(api_key: str, model: str)
```

**Supported Models:**
- `llama2`, `llama2:13b`, `llama2:70b`
- `codellama`, `codellama:13b`, `codellama:34b`
- `mistral`, `neural-chat`
- Any other model available in your local Ollama installation

**Methods:**

#### generate_text()

```python
def generate_text(
    prompt: str,
    temperature: float = 0.8,
    num_predict: int = 128,
    top_p: float = 0.9,
    top_k: int = 40,
    repeat_penalty: float = 1.1,
    **kwargs
) -> str
```

**Parameters:**
- `temperature` (float): Controls randomness (0.0 to 2.0)
- `num_predict` (int): Number of tokens to predict
- `top_p` (float): Nucleus sampling parameter
- `top_k` (int): Top-k sampling parameter
- `repeat_penalty` (float): Penalty for repetition

## Hugging Face Service

### HuggingFaceService

Provides access to Hugging Face models via Inference API.

```python
class HuggingFaceService:
    def __init__(api_key: str, model: str)
```

**Supported Models:**
- Text generation: `gpt2`, `gpt2-medium`, `facebook/opt-350m`
- Classification: `cardiffnlp/twitter-roberta-base-sentiment-latest`
- Question answering: `deepset/roberta-base-squad2`
- And thousands more from the Hugging Face Hub

**Methods:**

#### generate_text()

```python
def generate_text(
    prompt: str,
    max_length: int = 100,
    temperature: float = 1.0,
    top_p: float = 0.9,
    top_k: int = 50,
    repetition_penalty: float = 1.0,
    do_sample: bool = True,
    **kwargs
) -> str
```

**Parameters:**
- `max_length` (int): Maximum length of generated text
- `temperature` (float): Controls randomness
- `top_p` (float): Nucleus sampling parameter
- `top_k` (int): Top-k sampling parameter
- `repetition_penalty` (float): Penalty for repetition
- `do_sample` (bool): Enable sampling

## Service Selection

### get_service()

Internal function to get the appropriate service class.

```python
def get_service(service_name: str) -> type
```

**Parameters:**
- `service_name` (str): Name of the service ('openai', 'anthropic', etc.)

**Returns:**
- `type`: The service class for the specified provider

## Error Handling

All service classes raise `EasilyAIException` for service-specific errors:

```python
from easilyai.exceptions import EasilyAIException

try:
    service = OpenAIService("invalid-key", "gpt-3.5-turbo")
    response = service.generate_text("Hello")
except EasilyAIException as e:
    print(f"Service error: {e}")
```

## Usage Examples

### Direct Service Usage

```python
from easilyai.services.openai_service import OpenAIService

# Create service directly
service = OpenAIService("your-api-key", "gpt-3.5-turbo")

# Generate text
response = service.generate_text(
    "Explain machine learning",
    temperature=0.7,
    max_tokens=200
)

print(response)
```

### Service Comparison

```python
from easilyai.services.openai_service import OpenAIService
from easilyai.services.anthropic_service import AnthropicService

# Create services
openai_service = OpenAIService("openai-key", "gpt-3.5-turbo")
anthropic_service = AnthropicService("anthropic-key", "claude-3-haiku-20240307")

prompt = "What is artificial intelligence?"

# Compare responses
openai_response = openai_service.generate_text(prompt)
anthropic_response = anthropic_service.generate_text(prompt)

print("OpenAI:", openai_response)
print("Anthropic:", anthropic_response)
```

### Custom Service Wrapper

```python
from easilyai.services.openai_service import OpenAIService

class CustomOpenAIWrapper:
    def __init__(self, api_key):
        self.text_service = OpenAIService(api_key, "gpt-4")
        self.image_service = OpenAIService(api_key, "dall-e-3")
    
    def create_content(self, topic):
        # Generate text
        text = self.text_service.generate_text(
            f"Write about {topic}",
            temperature=0.7
        )
        
        # Generate related image
        image_url = self.image_service.generate_image(
            f"Illustration for: {text[:100]}..."
        )
        
        return {"text": text, "image": image_url}

# Usage
wrapper = CustomOpenAIWrapper("your-openai-key")
content = wrapper.create_content("renewable energy")
```

## Service Configuration

### Default Parameters

Each service has default parameters that can be overridden:

```python
# OpenAI defaults
openai_defaults = {
    "temperature": 0.7,
    "max_tokens": None,
    "top_p": 1.0,
    "frequency_penalty": 0.0,
    "presence_penalty": 0.0
}

# Anthropic defaults
anthropic_defaults = {
    "temperature": 0.7,
    "max_tokens": 1000,
    "top_p": 0.9,
    "top_k": 250
}
```

### Model Validation

Services validate model names and parameters:

```python
from easilyai.services.openai_service import OpenAIService

try:
    # Invalid model
    service = OpenAIService("key", "invalid-model")
except ValueError as e:
    print(f"Invalid model: {e}")

try:
    # Invalid parameters
    service = OpenAIService("key", "gpt-3.5-turbo")
    response = service.generate_text("Hello", temperature=3.0)  # Invalid temperature
except ValueError as e:
    print(f"Invalid parameter: {e}")
```

This API reference provides comprehensive documentation for all service classes in EasilyAI, helping developers understand the underlying functionality and customize their AI applications.