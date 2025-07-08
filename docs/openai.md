# OpenAI

OpenAI provides some of the most popular and capable AI models, including GPT for text generation, DALL-E for image creation, and advanced text-to-speech models. This guide covers how to use OpenAI services through EasilyAI.

## Getting Started

### API Key Setup

1. Sign up at [OpenAI Platform](https://platform.openai.com/)
2. Create an API key in your dashboard
3. Set it as an environment variable or use it directly

```bash
export OPENAI_API_KEY="your-openai-api-key"
```

### Basic Usage

```python
from easilyai import create_app

# Create OpenAI app
app = create_app("OpenAI", "openai", "your-openai-api-key", "gpt-3.5-turbo")

# Generate text
response = app.request("Write a short story about artificial intelligence")
print(response)
```

## Available Models

### GPT Models (Text Generation)

#### GPT-3.5 Turbo

- **Model ID**: `gpt-3.5-turbo`
- **Best for**: Fast, cost-effective text generation
- **Context window**: 16K tokens
- **Strengths**: Speed, cost efficiency, good general performance

```python
gpt35_app = create_app("GPT35", "openai", "your-key", "gpt-3.5-turbo")
response = gpt35_app.request("Explain machine learning in simple terms")
```

#### GPT-4

- **Model ID**: `gpt-4`
- **Best for**: Complex reasoning, high-quality output
- **Context window**: 8K tokens
- **Strengths**: Superior reasoning, accuracy, creativity

```python
gpt4_app = create_app("GPT4", "openai", "your-key", "gpt-4")
complex_analysis = gpt4_app.request("Analyze the economic implications of AI automation")
```

#### GPT-4 Turbo

- **Model ID**: `gpt-4-turbo` or `gpt-4-1106-preview`
- **Best for**: Longer context, improved performance
- **Context window**: 128K tokens
- **Strengths**: Large context, cost-effective GPT-4 alternative

```python
gpt4_turbo_app = create_app("GPT4Turbo", "openai", "your-key", "gpt-4-turbo")
long_analysis = gpt4_turbo_app.request("Analyze this 50-page document...")
```

### DALL-E Models (Image Generation)

#### DALL-E 3

- **Model ID**: `dall-e-3`
- **Best for**: High-quality image generation
- **Resolutions**: 1024×1024, 1792×1024, 1024×1792
- **Strengths**: Superior image quality, better prompt adherence

```python
dalle3_app = create_app("DALLE3", "openai", "your-key", "dall-e-3")
image_url = dalle3_app.request(
    "A serene mountain landscape at sunset",
    task_type="generate_image",
    size="1024x1024",
    quality="hd"
)
```

#### DALL-E 2

- **Model ID**: `dall-e-2`
- **Best for**: Cost-effective image generation, multiple variations
- **Resolutions**: 1024×1024, 512×512, 256×256
- **Strengths**: Lower cost, can generate multiple images

```python
dalle2_app = create_app("DALLE2", "openai", "your-key", "dall-e-2")
image_url = dalle2_app.request(
    "A futuristic city with flying cars",
    task_type="generate_image",
    size="1024x1024",
    n=1
)
```

### TTS Models (Text-to-Speech)

#### TTS-1

- **Model ID**: `tts-1`
- **Best for**: Real-time text-to-speech
- **Voices**: alloy, echo, fable, onyx, nova, shimmer
- **Strengths**: Fast, cost-effective

```python
from easilyai import create_tts_app

tts_app = create_tts_app("TTS", "openai", "your-key", "tts-1")
audio_file = tts_app.request(
    "Hello, this is a test of OpenAI's text-to-speech system",
    voice="alloy",
    output_file="test.mp3"
)
```

#### TTS-1-HD

- **Model ID**: `tts-1-hd`
- **Best for**: High-quality audio generation
- **Voices**: alloy, echo, fable, onyx, nova, shimmer
- **Strengths**: Superior audio quality

```python
tts_hd_app = create_tts_app("TTSHD", "openai", "your-key", "tts-1-hd")
hq_audio = tts_hd_app.request(
    "This is high-quality text-to-speech",
    voice="nova",
    output_file="high_quality.mp3"
)
```

## Text Generation

### Basic Text Generation

```python
from easilyai import create_app

app = create_app("TextGen", "openai", "your-key", "gpt-3.5-turbo")

# Simple generation
response = app.request("Write a haiku about programming")

# With parameters
detailed_response = app.request(
    "Write a technical blog post about Python decorators",
    temperature=0.7,
    max_tokens=1000,
    top_p=0.9
)
```

### Advanced Parameters

```python
# Temperature controls randomness (0.0 to 2.0)
creative_response = app.request(
    "Write a creative story",
    temperature=1.0  # More creative
)

conservative_response = app.request(
    "Explain a scientific concept",
    temperature=0.1  # More focused
)

# Max tokens limits response length
brief_response = app.request(
    "Summarize artificial intelligence",
    max_tokens=100  # Short response
)

# Top-p for nucleus sampling
focused_response = app.request(
    "Analyze this data",
    top_p=0.5  # More focused vocabulary
)

# Frequency and presence penalties
unique_response = app.request(
    "Write about innovation",
    frequency_penalty=0.5,  # Reduce repetition
    presence_penalty=0.3    # Encourage new topics
)
```

### System Messages

Guide the AI's behavior with system messages:

```python
response = app.request(
    "How do I learn Python?",
    system="You are a patient programming tutor. Always provide step-by-step guidance and encourage practice."
)
```

## Image Generation

### Basic Image Generation

```python
from easilyai import create_app

image_app = create_app("ImageGen", "openai", "your-key", "dall-e-3")

# Simple image generation
image_url = image_app.request(
    "A cat wearing a wizard hat",
    task_type="generate_image"
)

print(f"Generated image: {image_url}")
```

### Image Parameters

```python
# Size options
sizes = ["1024x1024", "1792x1024", "1024x1792"]

for size in sizes:
    image_url = image_app.request(
        "A beautiful landscape",
        task_type="generate_image",
        size=size
    )
    print(f"Size {size}: {image_url}")

# Quality settings (DALL-E 3 only)
hd_image = image_app.request(
    "A detailed portrait",
    task_type="generate_image",
    quality="hd"  # "standard" or "hd"
)

# Style settings (DALL-E 3 only)
natural_image = image_app.request(
    "A realistic photo of a cityscape",
    task_type="generate_image",
    style="natural"  # "natural" or "vivid"
)
```

### Multiple Images (DALL-E 2)

```python
dalle2_app = create_app("DALLE2", "openai", "your-key", "dall-e-2")

# Generate multiple variations
for i in range(3):
    image_url = dalle2_app.request(
        "A robot in a garden",
        task_type="generate_image",
        n=1
    )
    print(f"Variation {i+1}: {image_url}")
```

## Text-to-Speech

### Basic TTS

```python
from easilyai import create_tts_app

tts_app = create_tts_app("TTS", "openai", "your-key", "tts-1")

# Basic text-to-speech
audio_file = tts_app.request(
    "Welcome to EasilyAI! This library makes AI simple and accessible.",
    voice="alloy"
)

print(f"Audio saved to: {audio_file}")
```

### Voice Options

```python
voices = ["alloy", "echo", "fable", "onyx", "nova", "shimmer"]

for voice in voices:
    audio_file = tts_app.request(
        f"This is the {voice} voice speaking",
        voice=voice,
        output_file=f"{voice}_sample.mp3"
    )
    print(f"{voice}: {audio_file}")
```

### TTS Parameters

```python
# Speed control
slow_speech = tts_app.request(
    "This is slow speech",
    voice="nova",
    speed=0.5,  # 0.25 to 4.0
    output_file="slow.mp3"
)

fast_speech = tts_app.request(
    "This is fast speech",
    voice="nova",
    speed=2.0,
    output_file="fast.mp3"
)

# Response format
formats = ["mp3", "opus", "aac", "flac"]

for format in formats:
    audio_file = tts_app.request(
        "Testing different audio formats",
        voice="alloy",
        response_format=format,
        output_file=f"test.{format}"
    )
```

## Use Cases

### Content Creation

```python
from easilyai import create_app

writer_app = create_app("Writer", "openai", "your-key", "gpt-4")

# Blog post generation
blog_post = writer_app.request(
    "Write a 1000-word blog post about the benefits of meditation. "
    "Include practical tips and scientific backing. "
    "Make it engaging for beginners."
)

# Marketing copy
ad_copy = writer_app.request(
    "Create compelling ad copy for a new fitness app. "
    "Target audience: busy professionals. "
    "Highlight time efficiency and results."
)

# Social media content
social_posts = writer_app.request(
    "Create 5 engaging social media posts about sustainable living. "
    "Include relevant hashtags and call-to-actions."
)
```

### Code Generation

```python
from easilyai import create_app

coder_app = create_app("Coder", "openai", "your-key", "gpt-4")

# Function generation
function_code = coder_app.request(
    "Write a Python function that implements quicksort algorithm. "
    "Include docstring, type hints, and error handling."
)

# Code explanation
explanation = coder_app.request(
    "Explain this JavaScript code line by line:\n"
    "const fibonacci = n => n <= 1 ? n : fibonacci(n-1) + fibonacci(n-2);"
)

# Code review
review = coder_app.request(
    "Review this Python code for improvements:\n"
    "def process_data(data):\n"
    "    result = []\n"
    "    for item in data:\n"
    "        if item > 0:\n"
    "            result.append(item * 2)\n"
    "    return result"
)
```

### Creative Projects

```python
from easilyai import create_app

creative_app = create_app("Creative", "openai", "your-key", "gpt-4")
artist_app = create_app("Artist", "openai", "your-key", "dall-e-3")

# Story generation
story = creative_app.request(
    "Write a short science fiction story about AI and humanity. "
    "Make it thought-provoking and emotionally engaging."
)

# Character design
character_image = artist_app.request(
    "A futuristic AI character with human-like features, "
    "glowing blue eyes, standing in a high-tech laboratory",
    task_type="generate_image",
    style="vivid"
)

# Concept art
concept_art = artist_app.request(
    "Concept art for a cyberpunk city with neon lights, "
    "flying cars, and towering skyscrapers",
    task_type="generate_image",
    quality="hd"
)
```

### Educational Content

```python
from easilyai import create_app, create_tts_app

educator_app = create_app("Educator", "openai", "your-key", "gpt-4")
narrator_app = create_tts_app("Narrator", "openai", "your-key", "tts-1-hd")

# Lesson content
lesson_content = educator_app.request(
    "Create a lesson plan for teaching 5th graders about the water cycle. "
    "Include objectives, activities, and assessment methods."
)

# Audio narration
audio_lesson = narrator_app.request(
    lesson_content,
    voice="nova",
    output_file="water_cycle_lesson.mp3"
)

# Quiz generation
quiz = educator_app.request(
    "Create a 10-question multiple choice quiz about the water cycle "
    "for 5th grade students. Include answer explanations."
)
```

## Advanced Techniques

### Function Calling

OpenAI models support function calling for structured outputs:

```python
# Note: Function calling requires additional setup
# This is a simplified example of the concept

def get_weather(location):
    # Your weather API call here
    return f"The weather in {location} is sunny and 75°F"

# In practice, you'd define function schemas for the model
response = app.request(
    "What's the weather like in New York?",
    # functions=[weather_function_schema]
)
```

### Chain of Thought Prompting

```python
from easilyai import create_app

reasoning_app = create_app("Reasoning", "openai", "your-key", "gpt-4")

chain_of_thought = reasoning_app.request(
    "Solve this step by step:\n"
    "A store is having a 25% off sale. If a jacket originally costs $80, "
    "and there's an additional 10% tax, what is the final price? "
    "Show your work."
)
```

### Few-Shot Learning

```python
few_shot_prompt = """
Classify these product reviews as positive, negative, or neutral:

Review: "This product is amazing! Highly recommend."
Classification: Positive

Review: "It's okay, nothing special."
Classification: Neutral

Review: "Terrible quality, waste of money."
Classification: Negative

Review: "The best purchase I've made this year!"
Classification:
"""

classification = app.request(few_shot_prompt)
```

## Optimization Tips

### Model Selection

Choose the right model for your use case:

```python
# For speed and cost efficiency
quick_app = create_app("Quick", "openai", "your-key", "gpt-3.5-turbo")

# For quality and complex reasoning
quality_app = create_app("Quality", "openai", "your-key", "gpt-4")

# For long documents
long_context_app = create_app("LongContext", "openai", "your-key", "gpt-4-turbo")
```

### Cost Optimization

```python
# Use shorter prompts when possible
concise_prompt = "Summarize: [content]"

# Limit max_tokens for controlled responses
limited_response = app.request(
    "Explain quantum computing",
    max_tokens=150  # Keep response short
)

# Use GPT-3.5 for simpler tasks
simple_app = create_app("Simple", "openai", "your-key", "gpt-3.5-turbo")
simple_response = simple_app.request("What is 2+2?")
```

### Performance Optimization

```python
# Use lower temperature for consistent results
consistent_response = app.request(
    "List the top 5 programming languages",
    temperature=0.1
)

# Use streaming for long responses (requires additional setup)
# response = app.request_stream("Write a long essay...")
```

## Error Handling

### Common OpenAI Errors

```python
from easilyai import create_app
from easilyai.exceptions import EasilyAIException

app = create_app("OpenAI", "openai", "your-key", "gpt-3.5-turbo")

try:
    response = app.request("Your prompt here")
    print(response)
except EasilyAIException as e:
    error_msg = str(e).lower()
    
    if "api key" in error_msg:
        print("Invalid API key. Check your OpenAI API key.")
    elif "quota" in error_msg:
        print("API quota exceeded. Check your usage limits.")
    elif "rate limit" in error_msg:
        print("Rate limit exceeded. Please wait before making more requests.")
    elif "context length" in error_msg:
        print("Input too long. Try reducing your prompt length.")
    else:
        print(f"OpenAI API error: {e}")
```

### Retry Logic

```python
import time
from easilyai import create_app

def openai_request_with_retry(prompt, max_retries=3):
    app = create_app("OpenAI", "openai", "your-key", "gpt-3.5-turbo")
    
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

## Best Practices

### 1. Choose Appropriate Models

- **GPT-3.5 Turbo**: Fast, cost-effective for simple tasks
- **GPT-4**: Complex reasoning, high-quality output
- **GPT-4 Turbo**: Long context, balanced performance

### 2. Optimize Prompts

```python
# Good: Clear and specific
good_prompt = "Write a 500-word product description for wireless headphones targeting fitness enthusiasts"

# Better: Structured with context
better_prompt = """
Task: Product description
Product: Wireless headphones
Target: Fitness enthusiasts
Length: 500 words
Tone: Energetic and motivational
Include: Features, benefits, use cases
"""
```

### 3. Use Appropriate Parameters

```python
# For factual content
factual_response = app.request(
    "Explain photosynthesis",
    temperature=0.1,
    max_tokens=300
)

# For creative content
creative_response = app.request(
    "Write a poem",
    temperature=0.8,
    max_tokens=200
)
```

### 4. Handle Rate Limits

Implement proper rate limiting and retry logic to handle API limitations gracefully.

### 5. Monitor Costs

Track your token usage and costs, especially with GPT-4 models which are more expensive.

## Comparison with Other Services

### OpenAI vs Claude

- **OpenAI**: Better for creative tasks, image generation, TTS
- **Claude**: Better for reasoning, analysis, long-form content

### OpenAI vs Gemini

- **OpenAI**: More mature ecosystem, better creative capabilities
- **Gemini**: Longer context windows, competitive pricing

### When to Use OpenAI

- Creative writing and content generation
- Image creation and visual content
- Text-to-speech applications
- Code generation and programming tasks
- General-purpose AI applications

OpenAI's comprehensive suite of models makes it an excellent choice for most AI applications, offering the best balance of capability, performance, and ecosystem support.