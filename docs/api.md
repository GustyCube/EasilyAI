# API Reference

This is the complete API reference for EasilyAI, documenting all classes, methods, and functions.

## Core Classes

### EasyAIApp

The main application class for general AI tasks.

```python
class EasyAIApp:
    def __init__(name: str, service: str, api_key: str, model: str)
```

**Parameters:**
- `name` (str): A unique name for your app instance
- `service` (str): The AI service to use ('openai', 'anthropic', 'gemini', 'grok', 'ollama', 'huggingface')
- `api_key` (str): Your API key for the service
- `model` (str): The specific model to use

**Methods:**

#### request()

Make a request to the AI service.

```python
def request(
    prompt: str,
    task_type: str = "generate_text",
    **kwargs
) -> str:
```

**Parameters:**
- `prompt` (str): The input text or prompt
- `task_type` (str): Type of task ('generate_text', 'generate_image', 'text_to_speech')
- `**kwargs`: Additional parameters specific to the task and service

**Returns:**
- `str`: The response from the AI service

**Example:**
```python
from easilyai import create_app

app = create_app("MyApp", "openai", "your-key", "gpt-3.5-turbo")
response = app.request("Hello, world!")
print(response)
```

### EasyAITTSApp

Specialized application class for text-to-speech tasks.

```python
class EasyAITTSApp:
    def __init__(name: str, service: str, api_key: str, model: str)
```

**Parameters:**
- `name` (str): A unique name for your app instance
- `service` (str): The AI service to use (currently only 'openai' supports TTS)
- `api_key` (str): Your API key for the service
- `model` (str): The TTS model to use

**Methods:**

#### request()

Convert text to speech.

```python
def request(
    text: str,
    voice: str = "alloy",
    output_file: str = None,
    **kwargs
) -> str:
```

**Parameters:**
- `text` (str): The text to convert to speech
- `voice` (str): The voice to use ('alloy', 'echo', 'fable', 'onyx', 'nova', 'shimmer')
- `output_file` (str): Path to save the audio file
- `**kwargs`: Additional parameters for the TTS service

**Returns:**
- `str`: Path to the generated audio file

**Example:**
```python
from easilyai import create_tts_app

tts_app = create_tts_app("TTS", "openai", "your-key", "tts-1")
audio_file = tts_app.request("Hello world", voice="alloy", output_file="hello.mp3")
```

## Factory Functions

### create_app()

Create a general-purpose AI application.

```python
def create_app(
    name: str,
    service: str,
    api_key: str,
    model: str
) -> EasyAIApp:
```

**Parameters:**
- `name` (str): A unique name for your app instance
- `service` (str): The AI service to use
- `api_key` (str): Your API key for the service
- `model` (str): The specific model to use

**Returns:**
- `EasyAIApp`: A configured AI application instance

**Example:**
```python
from easilyai import create_app

app = create_app("MyApp", "openai", "your-api-key", "gpt-3.5-turbo")
```

### create_tts_app()

Create a text-to-speech application.

```python
def create_tts_app(
    name: str,
    service: str,
    api_key: str,
    model: str
) -> EasyAITTSApp:
```

**Parameters:**
- `name` (str): A unique name for your app instance
- `service` (str): The AI service to use (currently only 'openai')
- `api_key` (str): Your API key for the service
- `model` (str): The TTS model to use

**Returns:**
- `EasyAITTSApp`: A configured TTS application instance

**Example:**
```python
from easilyai import create_tts_app

tts_app = create_tts_app("TTS", "openai", "your-api-key", "tts-1")
```

## Pipeline System

### EasilyAIPipeline

Chain multiple AI operations together.

```python
class EasilyAIPipeline:
    def __init__(name: str)
```

**Parameters:**
- `name` (str): A unique name for your pipeline

**Methods:**

#### add_task()

Add a task to the pipeline.

```python
def add_task(
    app: EasyAIApp,
    task_type: str,
    prompt: str
) -> None:
```

**Parameters:**
- `app` (EasyAIApp): The AI application to use for this task
- `task_type` (str): Type of task ('generate_text', 'generate_image', 'text_to_speech')
- `prompt` (str): The prompt for this task (can reference previous results)

**Example:**
```python
from easilyai import create_app
from easilyai.pipeline import EasilyAIPipeline

app = create_app("App", "openai", "your-key", "gpt-3.5-turbo")
pipeline = EasilyAIPipeline("MyPipeline")
pipeline.add_task(app, "generate_text", "Write a story about {previous_result}")
```

#### run()

Execute all tasks in the pipeline.

```python
def run() -> list:
```

**Returns:**
- `list`: Results from all tasks in the pipeline

**Example:**
```python
results = pipeline.run()
print(results[0])  # First task result
print(results[1])  # Second task result
```

## Custom AI Framework

### CustomAIService

Base class for creating custom AI services.

```python
class CustomAIService:
    def __init__(api_key: str, model: str)
```

**Parameters:**
- `api_key` (str): Your API key for the custom service
- `model` (str): The model identifier

**Methods to Override:**

#### generate_text()

Generate text using your custom service.

```python
def generate_text(
    prompt: str,
    **kwargs
) -> str:
```

**Parameters:**
- `prompt` (str): The input prompt
- `**kwargs`: Additional parameters

**Returns:**
- `str`: Generated text

#### generate_image()

Generate images using your custom service.

```python
def generate_image(
    prompt: str,
    **kwargs
) -> str:
```

**Parameters:**
- `prompt` (str): The image description
- `**kwargs`: Additional parameters

**Returns:**
- `str`: Image URL or path

#### text_to_speech()

Convert text to speech using your custom service.

```python
def text_to_speech(
    text: str,
    **kwargs
) -> str:
```

**Parameters:**
- `text` (str): The text to convert
- `**kwargs`: Additional parameters

**Returns:**
- `str`: Audio file path or URL

**Example:**
```python
from easilyai.custom_ai import CustomAIService

class MyCustomService(CustomAIService):
    def __init__(self, api_key, model):
        super().__init__(api_key, model)
    
    def generate_text(self, prompt, **kwargs):
        # Your implementation here
        return f"Response to: {prompt}"
```

### register_custom_ai()

Register a custom AI service.

```python
def register_custom_ai(
    service_name: str,
    service_class: type
) -> None:
```

**Parameters:**
- `service_name` (str): The name to use for your service
- `service_class` (type): Your custom service class

**Example:**
```python
from easilyai import register_custom_ai

register_custom_ai("mycustom", MyCustomService)

# Now you can use it
app = create_app("Custom", "mycustom", "fake-key", "my-model")
```

## Service-Specific Parameters

### OpenAI Parameters

#### Text Generation
- `temperature` (float): Controls randomness (0.0 to 2.0)
- `max_tokens` (int): Maximum tokens to generate
- `top_p` (float): Nucleus sampling parameter
- `frequency_penalty` (float): Penalty for repeated tokens
- `presence_penalty` (float): Penalty for new topics

#### Image Generation
- `size` (str): Image dimensions ('1024x1024', '1792x1024', '1024x1792')
- `quality` (str): Image quality ('standard', 'hd')
- `style` (str): Image style ('natural', 'vivid')
- `n` (int): Number of images to generate (DALL-E 2 only)

#### Text-to-Speech
- `voice` (str): Voice to use ('alloy', 'echo', 'fable', 'onyx', 'nova', 'shimmer')
- `response_format` (str): Audio format ('mp3', 'opus', 'aac', 'flac')
- `speed` (float): Speech speed (0.25 to 4.0)

### Anthropic Parameters

#### Text Generation
- `temperature` (float): Controls randomness (0.0 to 1.0)
- `max_tokens` (int): Maximum tokens to generate
- `top_p` (float): Nucleus sampling parameter
- `top_k` (int): Top-k sampling parameter

### Gemini Parameters

#### Text Generation
- `temperature` (float): Controls randomness (0.0 to 1.0)
- `max_output_tokens` (int): Maximum tokens to generate
- `top_p` (float): Nucleus sampling parameter
- `top_k` (int): Top-k sampling parameter

### Grok Parameters

#### Text Generation
- `temperature` (float): Controls randomness (0.0 to 2.0)
- `max_tokens` (int): Maximum tokens to generate
- `top_p` (float): Nucleus sampling parameter

### Ollama Parameters

#### Text Generation
- `temperature` (float): Controls randomness (0.0 to 2.0)
- `num_predict` (int): Number of tokens to predict
- `top_p` (float): Nucleus sampling parameter
- `top_k` (int): Top-k sampling parameter

## Exception Classes

### EasilyAIException

Base exception class for all EasilyAI errors.

```python
class EasilyAIException(Exception):
    pass
```

**Example:**
```python
from easilyai.exceptions import EasilyAIException

try:
    response = app.request("Hello")
except EasilyAIException as e:
    print(f"EasilyAI error: {e}")
```

### Service-Specific Exceptions

EasilyAI may raise service-specific exceptions that inherit from `EasilyAIException`:

- Authentication errors
- Rate limit errors
- Quota exceeded errors
- Model not found errors
- Invalid parameter errors

## Utility Functions

### get_supported_services()

Get a list of supported AI services.

```python
def get_supported_services() -> list:
```

**Returns:**
- `list`: List of supported service names

**Example:**
```python
from easilyai.utils import get_supported_services

services = get_supported_services()
print(services)  # ['openai', 'anthropic', 'gemini', 'grok', 'ollama', 'huggingface']
```

### get_service_models()

Get available models for a specific service.

```python
def get_service_models(service: str) -> list:
```

**Parameters:**
- `service` (str): The service name

**Returns:**
- `list`: List of available models for the service

**Example:**
```python
from easilyai.utils import get_service_models

models = get_service_models("openai")
print(models)  # ['gpt-3.5-turbo', 'gpt-4', 'dall-e-3', 'tts-1', ...]
```

### validate_api_key()

Validate an API key for a service.

```python
def validate_api_key(service: str, api_key: str) -> bool:
```

**Parameters:**
- `service` (str): The service name
- `api_key` (str): The API key to validate

**Returns:**
- `bool`: True if the API key is valid

**Example:**
```python
from easilyai.utils import validate_api_key

is_valid = validate_api_key("openai", "your-api-key")
if is_valid:
    print("API key is valid")
else:
    print("API key is invalid")
```

## Examples

### Basic Usage

```python
from easilyai import create_app

# Create app
app = create_app("MyApp", "openai", "your-api-key", "gpt-3.5-turbo")

# Generate text
response = app.request("Tell me a joke")
print(response)

# Generate image
image_url = app.request("A cat wearing a hat", task_type="generate_image")
print(image_url)
```

### Pipeline Example

```python
from easilyai import create_app
from easilyai.pipeline import EasilyAIPipeline

# Create apps
text_app = create_app("Text", "openai", "your-key", "gpt-4")
image_app = create_app("Image", "openai", "your-key", "dall-e-3")

# Create pipeline
pipeline = EasilyAIPipeline("ContentPipeline")
pipeline.add_task(text_app, "generate_text", "Write a story about space")
pipeline.add_task(image_app, "generate_image", "Illustration for: {previous_result}")

# Run pipeline
results = pipeline.run()
print("Story:", results[0])
print("Image:", results[1])
```

### Custom Service Example

```python
from easilyai import create_app, register_custom_ai
from easilyai.custom_ai import CustomAIService

class MockService(CustomAIService):
    def generate_text(self, prompt, **kwargs):
        return f"Mock response to: {prompt}"

# Register and use
register_custom_ai("mock", MockService)
app = create_app("Mock", "mock", "fake-key", "mock-model")
response = app.request("Hello!")
print(response)
```

### Error Handling Example

```python
from easilyai import create_app
from easilyai.exceptions import EasilyAIException

try:
    app = create_app("App", "openai", "invalid-key", "gpt-3.5-turbo")
    response = app.request("Hello")
    print(response)
except EasilyAIException as e:
    print(f"EasilyAI error: {e}")
except Exception as e:
    print(f"Other error: {e}")
```

## Type Hints

EasilyAI supports type hints for better IDE support:

```python
from easilyai import create_app, EasyAIApp
from typing import Optional

def create_ai_app(
    name: str,
    service: str,
    api_key: str,
    model: str
) -> EasyAIApp:
    return create_app(name, service, api_key, model)

def safe_request(
    app: EasyAIApp,
    prompt: str,
    task_type: str = "generate_text"
) -> Optional[str]:
    try:
        return app.request(prompt, task_type=task_type)
    except Exception:
        return None
```

This API reference provides comprehensive documentation for all EasilyAI functionality. For more detailed examples and use cases, see the [Examples](/examples) section.