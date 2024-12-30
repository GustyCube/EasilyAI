# Custom AI Integration

## Overview
EasyAI allows you to integrate your own AI models and services.

## Registering a Custom AI Service

```python
from easilyai.custom_ai import CustomAIService, register_custom_ai

# Define a custom AI service
class MyCustomAI(CustomAIService):
    def generate_text(self, prompt):
        return f"Custom AI response for: {prompt}"

    def text_to_speech(self, text, **kwargs):
        return f"Custom TTS output: {text}"

# Register the custom AI
register_custom_ai("my_custom_ai", MyCustomAI)

# Use the custom AI
custom_app = easilyai.create_app(name="custom_ai_app", service="my_custom_ai")
print(custom_app.request("Hello from Custom AI!"))
```

Now you are ready to use and expand EasilyAI for your projects! Revisit the [Installation Guide](./installation.md) if needed.