# User Guide

Welcome to the comprehensive EasilyAI user guide! This guide will take you through all the features and capabilities of EasilyAI, from basic usage to advanced techniques.

## Getting Started

### What is EasilyAI?

EasilyAI is a Python library that provides a unified interface for multiple AI services. Instead of learning different APIs for each service, you can use one simple interface to work with:

- **OpenAI** (GPT models, DALL-E, TTS)
- **Anthropic** (Claude models)
- **Google Gemini**
- **X.AI Grok**
- **Ollama** (local models)
- **Hugging Face** (various models)

### Core Concepts

#### Apps
Apps are the main interface for interacting with AI services. Each app is configured for a specific service and model:

```python
from easilyai import create_app

# Create an app for text generation
app = create_app("MyApp", "openai", "your-api-key", "gpt-3.5-turbo")

# Use the app to generate text
response = app.request("Hello, world!")
print(response)
```

#### Task Types
EasilyAI supports different types of AI tasks:

- **Text Generation**: Generate text content
- **Image Generation**: Create images from text descriptions
- **Text-to-Speech**: Convert text to audio

```python
# Text generation (default)
text_response = app.request("Tell me a joke")

# Image generation
image_url = app.request("A sunset over mountains", task_type="generate_image")

# Text-to-speech
audio_file = app.request("Hello world", task_type="text_to_speech")
```

#### Services
Services are the different AI providers that EasilyAI supports. Each service has its own strengths:

- **OpenAI**: Best for general-purpose text generation, image creation, and speech
- **Anthropic**: Excellent for reasoning and analysis
- **Gemini**: Google's multimodal AI
- **Grok**: X.AI's conversational AI
- **Ollama**: Run models locally on your machine

## Basic Usage Patterns

### Simple Text Generation

```python
from easilyai import create_app

# Create app
app = create_app("TextBot", "openai", "your-api-key", "gpt-3.5-turbo")

# Generate text
response = app.request("Explain machine learning in simple terms")
print(response)
```

### Working with Multiple Services

```python
from easilyai import create_app

# Create multiple apps
openai_app = create_app("OpenAI", "openai", "your-openai-key", "gpt-3.5-turbo")
claude_app = create_app("Claude", "anthropic", "your-anthropic-key", "claude-3-haiku-20240307")

# Compare responses
prompt = "What are the benefits of renewable energy?"

openai_response = openai_app.request(prompt)
claude_response = claude_app.request(prompt)

print("OpenAI:", openai_response)
print("Claude:", claude_response)
```

### Image Generation

```python
from easilyai import create_app

# Create image generation app
app = create_app("ImageGen", "openai", "your-api-key", "dall-e-3")

# Generate image
image_url = app.request(
    "A cozy coffee shop with warm lighting",
    task_type="generate_image",
    size="1024x1024"
)

print(f"Generated image: {image_url}")
```

### Text-to-Speech

```python
from easilyai import create_tts_app

# Create TTS app
tts_app = create_tts_app("Speaker", "openai", "your-api-key", "tts-1")

# Generate speech
audio_file = tts_app.request(
    "Welcome to EasilyAI! This is a test of text-to-speech functionality.",
    voice="alloy",
    output_file="welcome.mp3"
)

print(f"Audio saved to: {audio_file}")
```

## Advanced Features

### Pipelines

Chain multiple AI operations together:

```python
from easilyai import create_app
from easilyai.pipeline import EasilyAIPipeline

# Create apps for different tasks
text_app = create_app("Writer", "openai", "your-key", "gpt-4")
image_app = create_app("Artist", "openai", "your-key", "dall-e-3")

# Create pipeline
pipeline = EasilyAIPipeline("ContentPipeline")

# Add tasks
pipeline.add_task(text_app, "generate_text", "Write a short story about a magical forest")
pipeline.add_task(image_app, "generate_image", "Create an image for this story: {previous_result}")

# Execute pipeline
results = pipeline.run()

print("Story:", results[0])
print("Image:", results[1])
```

### Custom AI Services

Extend EasilyAI with your own AI services:

```python
from easilyai import create_app, register_custom_ai
from easilyai.custom_ai import CustomAIService

class MyCustomAI(CustomAIService):
    def __init__(self, api_key, model):
        super().__init__(api_key, model)
    
    def generate_text(self, prompt, **kwargs):
        # Your custom implementation
        return f"Custom response to: {prompt}"

# Register your custom service
register_custom_ai("mycustom", MyCustomAI)

# Use your custom service
app = create_app("Custom", "mycustom", "fake-key", "my-model")
response = app.request("Hello!")
print(response)
```

## Working with Different AI Services

### OpenAI

OpenAI provides the most comprehensive feature set:

```python
from easilyai import create_app, create_tts_app

# Text generation
text_app = create_app("GPT", "openai", "your-key", "gpt-3.5-turbo")
text_response = text_app.request("Write a poem about coding")

# Image generation
image_app = create_app("DALLE", "openai", "your-key", "dall-e-3")
image_url = image_app.request("A robot painting a masterpiece", task_type="generate_image")

# Text-to-speech
tts_app = create_tts_app("TTS", "openai", "your-key", "tts-1")
audio_file = tts_app.request("Hello world", voice="alloy", output_file="hello.mp3")
```

### Anthropic (Claude)

Claude excels at reasoning and analysis:

```python
from easilyai import create_app

# Different Claude models
haiku_app = create_app("Haiku", "anthropic", "your-key", "claude-3-haiku-20240307")
sonnet_app = create_app("Sonnet", "anthropic", "your-key", "claude-3-sonnet-20240229")
opus_app = create_app("Opus", "anthropic", "your-key", "claude-3-opus-20240229")

# Use for analysis
analysis_prompt = "Analyze the pros and cons of remote work"
analysis = opus_app.request(analysis_prompt)
print(analysis)
```

### Google Gemini

Gemini offers multimodal capabilities:

```python
from easilyai import create_app

# Gemini models
flash_app = create_app("GeminiFlash", "gemini", "your-key", "gemini-1.5-flash")
pro_app = create_app("GeminiPro", "gemini", "your-key", "gemini-1.5-pro")

# Use for creative tasks
creative_prompt = "Write a creative story about time travel"
story = flash_app.request(creative_prompt)
print(story)
```

### X.AI Grok

Grok provides conversational AI:

```python
from easilyai import create_app

grok_app = create_app("Grok", "grok", "your-key", "grok-beta")

# Use for conversational tasks
conversation = grok_app.request("What's the latest in AI research?")
print(conversation)
```

### Ollama (Local Models)

Run models locally without API keys:

```python
from easilyai import create_app

# No API key needed for local models
ollama_app = create_app("Local", "ollama", "", "llama2")

# Use local model
response = ollama_app.request("What is the capital of France?")
print(response)
```

## Error Handling

### Basic Error Handling

```python
from easilyai import create_app
from easilyai.exceptions import EasilyAIException

try:
    app = create_app("TestApp", "openai", "your-key", "gpt-3.5-turbo")
    response = app.request("Hello world")
    print(response)
except EasilyAIException as e:
    print(f"AI Service Error: {e}")
except Exception as e:
    print(f"General Error: {e}")
```

### Robust Error Handling

```python
def safe_ai_request(prompt, service="openai", model="gpt-3.5-turbo", api_key=None):
    """Make an AI request with comprehensive error handling"""
    try:
        app = create_app("SafeApp", service, api_key, model)
        return app.request(prompt)
    
    except EasilyAIException as e:
        error_msg = str(e).lower()
        
        if "rate limit" in error_msg:
            return "Rate limit exceeded. Please try again later."
        elif "quota" in error_msg:
            return "API quota exceeded. Please check your usage."
        elif "authentication" in error_msg:
            return "Authentication failed. Please check your API key."
        else:
            return f"AI service error: {e}"
    
    except Exception as e:
        return f"Unexpected error: {e}"

# Usage
result = safe_ai_request("Hello!", api_key="your-key")
print(result)
```

## Best Practices

### 1. Secure API Key Management

```python
import os
from easilyai import create_app

# Use environment variables
app = create_app(
    "SecureApp",
    "openai",
    os.getenv("OPENAI_API_KEY"),
    "gpt-3.5-turbo"
)
```

### 2. Choose the Right Model

```python
# For quick responses
quick_app = create_app("Quick", "openai", "your-key", "gpt-3.5-turbo")

# For complex reasoning
complex_app = create_app("Complex", "openai", "your-key", "gpt-4")

# For cost-effective solutions
budget_app = create_app("Budget", "anthropic", "your-key", "claude-3-haiku-20240307")
```

### 3. Optimize Prompts

```python
# Vague prompt
vague_response = app.request("Tell me about AI")

# Specific prompt
specific_response = app.request(
    "Explain artificial intelligence in 3 paragraphs, "
    "covering its definition, main applications, and future prospects"
)
```

### 4. Handle Rate Limits

```python
import time

def request_with_retry(app, prompt, max_retries=3):
    for attempt in range(max_retries):
        try:
            return app.request(prompt)
        except EasilyAIException as e:
            if "rate limit" in str(e).lower() and attempt < max_retries - 1:
                wait_time = 2 ** attempt  # Exponential backoff
                time.sleep(wait_time)
                continue
            else:
                raise e
    return None
```

### 5. Use Appropriate Task Types

```python
# Text generation
text = app.request("Write a story")

# Image generation  
image = app.request("A beautiful landscape", task_type="generate_image")

# Text-to-speech
audio = app.request("Hello world", task_type="text_to_speech")
```

## Common Use Cases

### Content Creation

```python
from easilyai import create_app

writer_app = create_app("Writer", "openai", "your-key", "gpt-4")

# Blog post
blog_post = writer_app.request(
    "Write a 500-word blog post about the benefits of exercise, "
    "with an engaging introduction and practical tips"
)

# Marketing copy
marketing_copy = writer_app.request(
    "Create compelling marketing copy for a new eco-friendly water bottle, "
    "highlighting its sustainability features and health benefits"
)
```

### Data Analysis

```python
from easilyai import create_app

analyst_app = create_app("Analyst", "anthropic", "your-key", "claude-3-opus-20240229")

# Analyze data
analysis = analyst_app.request(
    "Analyze this sales data and provide insights:\n"
    "Q1: $100K, Q2: $150K, Q3: $120K, Q4: $180K\n"
    "What trends do you see and what recommendations do you have?"
)
```

### Code Generation

```python
from easilyai import create_app

coder_app = create_app("Coder", "openai", "your-key", "gpt-4")

# Generate code
code = coder_app.request(
    "Write a Python function that calculates the factorial of a number, "
    "including error handling and documentation"
)
```

### Creative Projects

```python
from easilyai import create_app

creative_app = create_app("Creative", "openai", "your-key", "gpt-4")
artist_app = create_app("Artist", "openai", "your-key", "dall-e-3")

# Generate story
story = creative_app.request("Write a short sci-fi story about AI and humanity")

# Generate accompanying artwork
artwork = artist_app.request(
    "Create a sci-fi artwork showing AI and humans working together",
    task_type="generate_image"
)
```

### Educational Content

```python
from easilyai import create_app

teacher_app = create_app("Teacher", "anthropic", "your-key", "claude-3-sonnet-20240229")

# Create lesson plan
lesson = teacher_app.request(
    "Create a beginner-friendly lesson plan for teaching Python programming, "
    "including objectives, activities, and assessment methods"
)

# Generate quiz questions
quiz = teacher_app.request(
    "Create 5 multiple-choice questions about Python basics, "
    "with answers and explanations"
)
```

## Performance Tips

1. **Use caching** for repeated requests
2. **Choose appropriate models** for your use case
3. **Implement rate limiting** to avoid API throttling
4. **Use batch processing** for multiple requests
5. **Monitor API usage** to manage costs
6. **Optimize prompts** for better results
7. **Handle errors gracefully** with retry logic

## Troubleshooting

### Common Issues

1. **Authentication errors**: Check your API keys
2. **Rate limiting**: Implement delays between requests
3. **Quota exceeded**: Monitor your API usage
4. **Model not found**: Verify model names
5. **Network issues**: Implement retry logic

### Debug Mode

```python
import logging

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)

# Your EasilyAI code here
app = create_app("Debug", "openai", "your-key", "gpt-3.5-turbo")
response = app.request("Hello!")
```

## Next Steps

Now that you understand the basics, explore these advanced topics:

- **[Text Generation](/textgeneration)** - Deep dive into text generation techniques
- **[Image Generation](/imagegeneration)** - Learn about AI image creation
- **[Text-to-Speech](/texttospeech)** - Convert text to audio
- **[Pipelines](/pipelines)** - Chain multiple AI operations
- **[Custom AI Services](/customai)** - Extend EasilyAI with your own services
- **[Configuration](/configuration)** - Advanced configuration options
- **[Performance](/performance)** - Optimize for speed and cost

Happy coding with EasilyAI!