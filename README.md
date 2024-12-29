<div align="center">
  <h1>EasilyAI</h1>
  <p>
    <p align="center">
    <a href="https://pypi.org/project/easilyai"><img src="https://img.shields.io/pypi/v/easilyai.svg" alt="PyPI"></a>
    <a href="tox.ini"><img src="https://img.shields.io/pypi/pyversions/easilyai" alt="Supported Python Versions"></a>
    <a href="https://pypi.org/project/easilyai"><img src="https://img.shields.io/pypi/dm/easilyai" alt="PyPI Downloads"></a>
    <a href="LICENSE"><img src="https://img.shields.io/github/license/GustyCube/EasilyAI" alt="License"></a>
    <a href="https://github.com/GustyCube/EasilAI/actions"><img src="https://github.com/GustyCube/EasilyAI/actions/workflows/python-publish.yml/badge.svg" alt="Build Status"></a>
    <a href="https://github.com/gustycube/EasilyAI/graphs/contributors">
    <img src="https://img.shields.io/github/contributors/gustycube/easilyai.svg" alt="Contributors">
</a>

</p>
  </p>
</div>

**EasilyAI** is a Python library that simplifies AI app development by integrating popular AI services like **OpenAI** and **Ollama**. It provides a clean, unified interface for text generation, image generation, and text-to-speech (TTS) tasks.

---

## Features
- **App Creation**: Simplify initializing AI services like OpenAI and Ollama.
- **Text-to-Speech**: Convert text to speech with OpenAI's TTS API (with voice selection).
- **Custom AI Support**: Integrate and register custom AI models.
- **Unified Request Handling**: Automatically determine task types like text, image, or TTS requests.
- **Pipeline Support**: Chain multiple tasks into a seamless workflow.

---

## Installation

Install the library via pip:

```bash
pip install easilyai
```

---

## Quick Start

### 1. Create an AI App

Create an app for OpenAI:

```python
import easilyai

# Initialize an OpenAI App
app = easilyai.create_app(
    name="my_ai_app",
    service="openai",
    apikey="YOUR_OPENAI_API_KEY",
    model="gpt-4"
)

# Make a request
response = app.request("Tell me a joke about AI.")
print(response)
```

---

### 2. Generate Text-to-Speech

Create a dedicated TTS app and specify the model and voice:

```python
# Initialize a TTS App
tts_app = easilyai.create_tts_app(
    name="my_tts_app",
    service="openai",
    apikey="YOUR_OPENAI_API_KEY",
    model="tts-1"
)

# Convert text to speech
output_file = tts_app.request_tts(
    text="Hello, I am your AI assistant!",
    tts_model="tts-1",
    voice="onyx",
    output_file="hello_ai.mp3"
)
print(f"TTS output saved to: {output_file}")
```

---

### 3. Use Pipelines

Chain multiple tasks (text generation, image generation, and TTS):

```python
# Create a pipeline
pipeline = easilyai.EasilyAIPipeline(app)

# Add tasks
pipeline.add_task("generate_text", "Write a poem about AI and nature.")
pipeline.add_task("generate_image", "A futuristic city with flying cars.")
pipeline.add_task("text_to_speech", "Here is a talking car in a futuristic world!")

# Run the pipeline
results = pipeline.run()

# Print results
for task_result in results:
    print(f"Task: {task_result['task']}\nResult: {task_result['result']}\n")
```

---

### 4. Register Custom AI Services

Integrate your own AI models into EasilyAI:

```python
from easilyai.custom_ai import CustomAIService, register_custom_ai

# Define a custom AI service
class MyCustomAI(CustomAIService):
    def generate_text(self, prompt):
        return f"Custom AI response for: {prompt}"

    def text_to_speech(self, text, **kwargs):
        return f"Custom TTS Output: {text}"

# Register the custom AI
register_custom_ai("my_custom_ai", MyCustomAI)

# Use the custom AI
custom_app = easilyai.create_app(name="custom_app", service="my_custom_ai", model="v1")
print(custom_app.request("What is 2 + 2?"))
```

---

## Supported Services

1. **OpenAI**
   - Text Generation (ChatGPT models like `gpt-4o`)
   - Image Generation (`dall-e-3`)
   - Text-to-Speech (`tts-1`, voices: `onyx`, `alloy`, etc.)

2. **Ollama**
   - Local LLM text generation (e.g., `llama3.1`).

3. **Custom AI**
   - Extend functionality by registering your own AI services.

---

## Error Handling

EasilyAI includes robust error handling with informative, emoji-coded messages.

Examples:
- 🔐 **Missing API Key**: "No API key provided! Add your API key to initialize the service."
- 🚫 **Invalid Request**: "The request is invalid. Please check your inputs."
- 🌐 **Connection Error**: "Unable to connect to the API. Ensure the server is running."
- ⏳ **Rate Limit Exceeded**: "Too many requests! Wait and try again."

---

## Future Features

- Full support for additional TTS providers.
- Model-specific optimizations.
- Enhanced CLI tools for developers.

---

## Contributing

Contributions are welcome! To contribute:
1. Fork the repository.
2. Create a new branch.
3. Submit a pull request with detailed changes.

---

## License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.

---

## Links

- Documentation: [coming soon]
- GitHub Repository: https://github.com/GustyCube/EasilyAI

---

## Contact

For questions, bugs, or feature requests, please reach out to **GustyCube** at **gc@gustycube.xyz**.
