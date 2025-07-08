# Examples

This page contains practical examples to help you get started with EasilyAI. Each example includes complete, runnable code that you can copy and modify for your own projects.

## Basic Examples

### Simple Text Generation

The most basic use case - generating text with different AI services:

```python
from easilyai import create_app

# Using OpenAI GPT
app = create_app("TextBot", "openai", "your-openai-key", "gpt-3.5-turbo")
response = app.request("Write a haiku about coding")
print(response)

# Using Anthropic Claude
claude_app = create_app("Claude", "anthropic", "your-anthropic-key", "claude-3-haiku-20240307")
response = claude_app.request("Explain quantum computing in simple terms")
print(response)
```

### Image Generation

Generate images using OpenAI's DALL-E:

```python
from easilyai import create_app

app = create_app("ImageBot", "openai", "your-openai-key", "dall-e-3")

# Generate an image
response = app.request(
    "A serene mountain landscape at sunset with a lake reflection",
    task_type="generate_image",
    size="1024x1024",
    quality="standard"
)

print(f"Generated image URL: {response}")
```

### Text-to-Speech

Convert text to speech:

```python
from easilyai import create_tts_app

tts_app = create_tts_app("SpeechBot", "openai", "your-openai-key", "tts-1")

# Generate speech
audio_file = tts_app.request(
    "Welcome to EasilyAI! This library makes AI integration simple and fun.",
    voice="nova",
    output_file="welcome_message.mp3"
)

print(f"Audio saved to: {audio_file}")
```

## Intermediate Examples

### Building a Simple Chatbot

Create a conversational AI that maintains context:

```python
from easilyai import create_app

class SimpleChatbot:
    def __init__(self, service="openai", api_key=None, model="gpt-3.5-turbo"):
        self.app = create_app("Chatbot", service, api_key, model)
        self.conversation_history = []
    
    def chat(self, user_input):
        # Add user input to history
        self.conversation_history.append(f"User: {user_input}")
        
        # Create context from recent history (last 5 exchanges)
        context = "\n".join(self.conversation_history[-10:])
        prompt = f"{context}\nAssistant:"
        
        # Generate response
        response = self.app.request(prompt)
        
        # Add AI response to history
        self.conversation_history.append(f"Assistant: {response}")
        
        return response

# Usage
chatbot = SimpleChatbot(api_key="your-openai-key")

print("Chatbot: Hello! How can I help you today?")
while True:
    user_input = input("You: ")
    if user_input.lower() in ['quit', 'exit', 'bye']:
        break
    
    response = chatbot.chat(user_input)
    print(f"Chatbot: {response}")
```

### Multi-Modal Content Generator

Generate both text and images for a creative project:

```python
from easilyai import create_app
import os

class ContentGenerator:
    def __init__(self, openai_key):
        self.text_app = create_app("TextGen", "openai", openai_key, "gpt-4")
        self.image_app = create_app("ImageGen", "openai", openai_key, "dall-e-3")
    
    def generate_story_with_illustration(self, theme):
        # Generate a short story
        story_prompt = f"Write a short, engaging story about {theme}. Make it vivid and descriptive."
        story = self.text_app.request(story_prompt)
        
        # Generate an image prompt from the story
        image_prompt_request = f"Based on this story, create a detailed visual description for an illustration:\n\n{story}"
        image_description = self.text_app.request(image_prompt_request)
        
        # Generate the image
        image_url = self.image_app.request(
            image_description,
            task_type="generate_image",
            size="1024x1024"
        )
        
        return {
            "story": story,
            "image_url": image_url,
            "image_description": image_description
        }

# Usage
generator = ContentGenerator("your-openai-key")
result = generator.generate_story_with_illustration("a magical forest")

print("Story:")
print(result["story"])
print(f"\nImage URL: {result['image_url']}")
```

### Language Translation Service

Create a translation service using different AI models:

```python
from easilyai import create_app

class TranslationService:
    def __init__(self):
        self.translators = {
            "openai": create_app("OpenAI-Translator", "openai", "your-openai-key", "gpt-3.5-turbo"),
            "claude": create_app("Claude-Translator", "anthropic", "your-anthropic-key", "claude-3-haiku-20240307"),
            "gemini": create_app("Gemini-Translator", "gemini", "your-gemini-key", "gemini-1.5-flash")
        }
    
    def translate(self, text, from_lang, to_lang, service="openai"):
        prompt = f"Translate this text from {from_lang} to {to_lang}. Only return the translation:\n\n{text}"
        
        translator = self.translators.get(service)
        if not translator:
            raise ValueError(f"Service {service} not available")
        
        return translator.request(prompt)
    
    def compare_translations(self, text, from_lang, to_lang):
        """Compare translations from different services"""
        results = {}
        
        for service_name, translator in self.translators.items():
            try:
                translation = self.translate(text, from_lang, to_lang, service_name)
                results[service_name] = translation
            except Exception as e:
                results[service_name] = f"Error: {e}"
        
        return results

# Usage
translator = TranslationService()

text = "Hello, how are you today?"
translations = translator.compare_translations(text, "English", "French")

for service, translation in translations.items():
    print(f"{service}: {translation}")
```

## Advanced Examples

### AI Pipeline for Content Creation

Use pipelines to create a multi-step content creation workflow:

```python
from easilyai import create_app
from easilyai.pipeline import EasilyAIPipeline

# Create apps for different tasks
text_app = create_app("Writer", "openai", "your-openai-key", "gpt-4")
image_app = create_app("Artist", "openai", "your-openai-key", "dall-e-3")
tts_app = create_app("Speaker", "openai", "your-openai-key", "tts-1")

# Create pipeline
pipeline = EasilyAIPipeline("ContentCreator")

# Add tasks to pipeline
pipeline.add_task(text_app, "generate_text", "Write a motivational quote about perseverance")
pipeline.add_task(image_app, "generate_image", "Create an inspiring image to accompany this quote: {previous_result}")
pipeline.add_task(tts_app, "text_to_speech", "Convert this quote to speech: {result_0}")

# Execute pipeline
results = pipeline.run()

print(f"Quote: {results[0]}")
print(f"Image: {results[1]}")
print(f"Audio: {results[2]}")
```

### Smart Document Summarizer

Create a document summarizer with different AI services:

```python
from easilyai import create_app
import os

class DocumentSummarizer:
    def __init__(self):
        self.summarizers = {
            "gpt": create_app("GPT-Summarizer", "openai", os.getenv("OPENAI_API_KEY"), "gpt-3.5-turbo"),
            "claude": create_app("Claude-Summarizer", "anthropic", os.getenv("ANTHROPIC_API_KEY"), "claude-3-haiku-20240307")
        }
    
    def summarize(self, text, service="gpt", length="medium"):
        length_prompts = {
            "short": "Provide a brief 2-3 sentence summary",
            "medium": "Provide a comprehensive summary in 1-2 paragraphs",
            "long": "Provide a detailed summary with key points and analysis"
        }
        
        prompt = f"{length_prompts[length]} of the following text:\n\n{text}"
        
        summarizer = self.summarizers.get(service)
        if not summarizer:
            raise ValueError(f"Service {service} not available")
        
        return summarizer.request(prompt)
    
    def extract_key_points(self, text, service="gpt"):
        prompt = f"Extract the key points from this text as a bulleted list:\n\n{text}"
        
        summarizer = self.summarizers.get(service)
        return summarizer.request(prompt)

# Usage
summarizer = DocumentSummarizer()

# Sample document
document = """
Artificial Intelligence has revolutionized many industries in recent years. 
From healthcare to finance, AI technologies are being implemented to improve 
efficiency and accuracy. Machine learning algorithms can now diagnose diseases, 
predict market trends, and even create art. However, with these advances come 
important ethical considerations about privacy, job displacement, and algorithmic bias.
"""

summary = summarizer.summarize(document, service="gpt", length="medium")
key_points = summarizer.extract_key_points(document, service="claude")

print("Summary:")
print(summary)
print("\nKey Points:")
print(key_points)
```

### Custom AI Service Integration

Create and register a custom AI service:

```python
from easilyai import create_app, register_custom_ai
from easilyai.custom_ai import CustomAIService

class MockAIService(CustomAIService):
    """A mock AI service for testing and development"""
    
    def __init__(self, api_key, model):
        super().__init__(api_key, model)
    
    def generate_text(self, prompt, **kwargs):
        # Mock response based on prompt
        responses = {
            "hello": "Hello! How can I help you today?",
            "joke": "Why don't scientists trust atoms? Because they make up everything!",
            "default": f"This is a mock response to: {prompt}"
        }
        
        # Simple keyword matching
        for key, response in responses.items():
            if key in prompt.lower():
                return response
        
        return responses["default"]
    
    def generate_image(self, prompt, **kwargs):
        return f"Mock image URL for: {prompt}"
    
    def text_to_speech(self, text, **kwargs):
        return f"Mock audio file for: {text}"

# Register the custom service
register_custom_ai("mock", MockAIService)

# Use the custom service
app = create_app("TestApp", "mock", "fake-key", "mock-model")
response = app.request("Tell me a joke")
print(response)  # Output: "Why don't scientists trust atoms? Because they make up everything!"
```

## Error Handling Examples

### Robust Error Handling

```python
from easilyai import create_app
from easilyai.exceptions import EasilyAIException
import time

class RobustAI:
    def __init__(self, service, api_key, model):
        self.app = create_app("RobustApp", service, api_key, model)
        self.max_retries = 3
        self.retry_delay = 1
    
    def request_with_retry(self, prompt, **kwargs):
        for attempt in range(self.max_retries):
            try:
                response = self.app.request(prompt, **kwargs)
                return response
            
            except EasilyAIException as e:
                if "rate limit" in str(e).lower() and attempt < self.max_retries - 1:
                    print(f"Rate limit hit, retrying in {self.retry_delay} seconds...")
                    time.sleep(self.retry_delay)
                    self.retry_delay *= 2  # Exponential backoff
                    continue
                else:
                    raise e
            
            except Exception as e:
                if attempt < self.max_retries - 1:
                    print(f"Unexpected error: {e}. Retrying...")
                    time.sleep(self.retry_delay)
                    continue
                else:
                    raise e
        
        return None

# Usage
robust_ai = RobustAI("openai", "your-openai-key", "gpt-3.5-turbo")

try:
    response = robust_ai.request_with_retry("Tell me about machine learning")
    print(response)
except Exception as e:
    print(f"Failed after all retries: {e}")
```

## Environment Setup Examples

### Using Environment Variables

Create a `.env` file for your API keys:

```bash
# .env file
OPENAI_API_KEY=your-openai-key-here
ANTHROPIC_API_KEY=your-anthropic-key-here
GEMINI_API_KEY=your-gemini-key-here
GROK_API_KEY=your-grok-key-here
```

Python code to use environment variables:

```python
import os
from dotenv import load_dotenv  # pip install python-dotenv
from easilyai import create_app

# Load environment variables from .env file
load_dotenv()

# Create apps using environment variables
openai_app = create_app("OpenAI", "openai", os.getenv("OPENAI_API_KEY"), "gpt-3.5-turbo")
claude_app = create_app("Claude", "anthropic", os.getenv("ANTHROPIC_API_KEY"), "claude-3-haiku-20240307")
gemini_app = create_app("Gemini", "gemini", os.getenv("GEMINI_API_KEY"), "gemini-1.5-flash")

# Test each service
services = [
    ("OpenAI", openai_app),
    ("Claude", claude_app),
    ("Gemini", gemini_app)
]

for name, app in services:
    try:
        response = app.request("What is 2+2?")
        print(f"{name}: {response}")
    except Exception as e:
        print(f"{name}: Error - {e}")
```

These examples should give you a solid foundation for using EasilyAI in your projects. Remember to:

1. Keep your API keys secure using environment variables
2. Handle errors gracefully
3. Start simple and build up complexity
4. Test with different AI services to find what works best for your use case

For more advanced usage, check out our [API Reference](/api) and [Advanced Features](/customai) sections.