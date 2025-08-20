# EasilyAI

A unified Python library for seamless integration with multiple AI services including OpenAI, Anthropic, Google Gemini, and Groq.

## Features

- **Unified Interface**: Single API for multiple AI providers
- **Easy Integration**: Simple setup with minimal configuration
- **Flexible Usage**: Support for chat completions, streaming, and batch processing
- **Type Safety**: Full type hints for better IDE support
- **Error Handling**: Comprehensive error handling with custom exceptions
- **Extensible**: Easy to add custom AI services

## Quick Example

```python
from easilyai import app_creation

# Create an app with your API key
app = app_creation(apikey="your-api-key", service="openai")

# Generate text
response = app.generate_text("Tell me a joke about programming")
print(response)
```

## Supported Services

| Service | Models | Features |
|---------|--------|----------|
| OpenAI | GPT-4, GPT-3.5-Turbo | Chat, Streaming, Function Calling |
| Anthropic | Claude 3 (Opus, Sonnet, Haiku) | Chat, Streaming, System Messages |
| Google Gemini | Gemini Pro, Gemini Pro Vision | Chat, Multimodal |
| Groq | Mixtral, LLaMA | Fast Inference, Chat |

## Installation

```bash
pip install easilyai
```

For development installation with all dependencies:

```bash
pip install easilyai[dev,test,docs]
```

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](https://github.com/GustyCube/EasilyAI/blob/main/LICENSE) file for details.