# Quick Start Guide

Get up and running with EasilyAI in just a few minutes! This guide will walk you through the basics of using EasilyAI to generate text, images, and speech.

## Prerequisites

- Python 3.7 or higher
- An API key from at least one AI service (OpenAI, Anthropic, Google, etc.)

## Your First AI App

Let's create your first AI application that can generate text:

### 1. Create a Text Generation App

```python
from easilyai import create_app

# Create an app with OpenAI (you can also use 'anthropic', 'gemini', 'grok', or 'ollama')
app = create_app(
    name="MyFirstApp",
    service="openai",
    api_key="your-openai-api-key-here",
    model="gpt-3.5-turbo"
)

# Generate some text
response = app.request("Tell me a short joke about programming")
print(response)
```

### 2. Try Different AI Services

EasilyAI supports multiple AI services. Here's how to use each:

```python
from easilyai import create_app

# OpenAI (GPT models)
openai_app = create_app("OpenAI", "openai", "your-openai-key", "gpt-3.5-turbo")

# Anthropic (Claude models)
claude_app = create_app("Claude", "anthropic", "your-anthropic-key", "claude-3-haiku-20240307")

# Google Gemini
gemini_app = create_app("Gemini", "gemini", "your-gemini-key", "gemini-1.5-flash")

# X.AI Grok
grok_app = create_app("Grok", "grok", "your-grok-key", "grok-beta")

# Ollama (local models)
ollama_app = create_app("Ollama", "ollama", "", "llama2")  # No API key needed for local
```

### 3. Generate Images

If you're using OpenAI, you can also generate images:

```python
from easilyai import create_app

app = create_app("ImageApp", "openai", "your-openai-key", "dall-e-3")

# Generate an image
image_response = app.request(
    "A cute robot learning to code",
    task_type="generate_image",
    size="1024x1024"
)

print(f"Image URL: {image_response}")
```

### 4. Text-to-Speech

Convert text to speech using OpenAI's TTS:

```python
from easilyai import create_tts_app

tts_app = create_tts_app("TTSApp", "openai", "your-openai-key", "tts-1")

# Convert text to speech
audio_response = tts_app.request(
    "Hello! Welcome to EasilyAI!",
    voice="alloy",
    output_file="welcome.mp3"
)

print(f"Audio saved to: {audio_response}")
```

## Next Steps

Now that you've got the basics down, here are some next steps to explore:

1. **[Learn about different AI services →](/services)** - Discover the capabilities of each supported AI service
2. **[Explore pipelines →](/pipelines)** - Chain multiple AI operations together
3. **[Handle errors gracefully →](/errorhandling)** - Learn how to handle API errors and rate limits
4. **[Create custom AI services →](/customai)** - Extend EasilyAI with your own AI providers

## Common Patterns

### Safe API Key Management

Never hardcode your API keys! Use environment variables instead:

```python
import os
from easilyai import create_app

app = create_app(
    name="SecureApp",
    service="openai",
    api_key=os.getenv("OPENAI_API_KEY"),  # Set this in your environment
    model="gpt-3.5-turbo"
)
```

### Basic Error Handling

```python
from easilyai import create_app
from easilyai.exceptions import EasilyAIException

try:
    app = create_app("MyApp", "openai", "your-key", "gpt-3.5-turbo")
    response = app.request("Hello, world!")
    print(response)
except EasilyAIException as e:
    print(f"AI Error: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
```

### Multiple Requests

```python
from easilyai import create_app

app = create_app("ChatApp", "openai", "your-key", "gpt-3.5-turbo")

questions = [
    "What is Python?",
    "How do I install packages?",
    "What are virtual environments?"
]

for question in questions:
    response = app.request(question)
    print(f"Q: {question}")
    print(f"A: {response}\n")
```

## Tips for Beginners

1. **Start Simple**: Begin with basic text generation before moving to advanced features
2. **Test with Different Models**: Each AI service has different strengths - experiment to find what works best for your use case
3. **Read the Error Messages**: EasilyAI provides helpful error messages to guide you
4. **Use Environment Variables**: Keep your API keys secure by using environment variables
5. **Check the Examples**: Look at the `/examples` directory for more code samples

Ready to dive deeper? Check out our [comprehensive guide](/guide) or explore specific [AI services](/services).