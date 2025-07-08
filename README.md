<div align="center">
  <h1>EasilyAI</h1>
  <p><em>A unified Python library for AI services</em></p>
  
  <p>
    <a href="https://pypi.org/project/easilyai"><img src="https://img.shields.io/pypi/v/easilyai?style=flat-square&color=blue" alt="PyPI Version"></a>
    <a href="https://github.com/GustyCube/EasilyAI/actions/workflows/python-publish.yml"><img src="https://img.shields.io/github/actions/workflow/status/GustyCube/EasilyAI/python-publish.yml?style=flat-square&label=CI%2FCD" alt="CI/CD Status"></a>
    <a href="https://github.com/GustyCube/EasilyAI/actions/workflows/docs.yml"><img src="https://img.shields.io/github/actions/workflow/status/GustyCube/EasilyAI/docs.yml?style=flat-square&label=docs" alt="Docs Status"></a>
    <a href="https://app.deepsource.com/gh/GustyCube/EasilyAI/"><img src="https://app.deepsource.com/gh/GustyCube/EasilyAI.svg/?label=code+coverage&show_trend=true&token=Vidoy6h5_sKpG-0YdVA_ISy_&style=flat-square" alt="Code Coverage"></a>
    <a href="https://pypi.org/project/easilyai"><img src="https://img.shields.io/pypi/dm/easilyai?style=flat-square&color=green" alt="Downloads"></a>
    <a href="https://pypi.org/project/easilyai"><img src="https://img.shields.io/pypi/pyversions/easilyai?style=flat-square" alt="Python Versions"></a>
    <a href="LICENSE"><img src="https://img.shields.io/github/license/GustyCube/EasilyAI?style=flat-square" alt="License"></a>
    <a href="https://github.com/GustyCube/EasilyAI/graphs/contributors"><img src="https://img.shields.io/github/contributors/GustyCube/EasilyAI?style=flat-square" alt="Contributors"></a>
  </p>
</div>

## Overview

**EasilyAI** is a powerful Python library that simplifies AI application development by providing a unified interface for multiple AI services including **OpenAI**, **Anthropic**, **Google Gemini**, **X.AI Grok**, and **Ollama**. Whether you need text generation, image creation, or text-to-speech functionality, EasilyAI offers a consistent API that makes switching between providers effortless.

---

## 🚀 Key Features

- **🔄 Multi-Provider Support**: Seamlessly switch between OpenAI, Anthropic, Google Gemini, X.AI Grok, and Ollama
- **📝 Text Generation**: Advanced language models for chat, completion, and creative writing
- **🎨 Image Generation**: Create stunning visuals with DALL-E and other image models
- **🗣️ Text-to-Speech**: High-quality voice synthesis with multiple voice options
- **🔗 Pipeline System**: Chain multiple AI operations into powerful workflows
- **🛠️ Custom AI Integration**: Easily extend with your own AI services
- **⚡ Unified API**: One consistent interface for all providers and tasks
- **🎯 Auto Task Detection**: Intelligent request routing based on content type

---

## 📦 Installation

```bash
pip install easilyai
```

## 🚀 Quick Start

Get up and running in minutes with these simple examples:

### Basic Text Generation
```python
import easilyai

# Create an app with your preferred provider
app = easilyai.create_app(
    name="my_ai_app",
    service="openai",  # or "anthropic", "gemini", "grok", "ollama"
    apikey="YOUR_API_KEY",
    model="gpt-4"
)

# Generate text
response = app.request("Explain quantum computing in simple terms")
print(response)
```

### Text-to-Speech
```python
# Create a TTS app
tts_app = easilyai.create_tts_app(
    name="my_tts_app",
    service="openai",
    apikey="YOUR_API_KEY",
    model="tts-1"
)

# Convert text to speech
tts_app.request_tts(
    text="Hello from EasilyAI!",
    voice="onyx",
    output_file="greeting.mp3"
)
```

### AI Pipeline
```python
# Chain multiple AI operations
pipeline = easilyai.EasilyAIPipeline(app)
pipeline.add_task("generate_text", "Write a haiku about coding")
pipeline.add_task("generate_image", "A serene coding environment")

results = pipeline.run()
```

## 🛠️ Supported AI Providers

| Provider | Text Generation | Image Generation | Text-to-Speech |
|----------|:---------------:|:----------------:|:--------------:|
| **OpenAI** | ✅ | ✅ | ✅ |
| **Anthropic** | ✅ | ❌ | ❌ |
| **Google Gemini** | ✅ | ❌ | ❌ |
| **X.AI Grok** | ✅ | ❌ | ❌ |
| **Ollama** | ✅ | ❌ | ❌ |
| **Custom AI** | ✅ | ✅ | ✅ |

## 📚 Documentation

For comprehensive guides, API reference, and advanced usage examples, visit our documentation:

**[📖 View Full Documentation →](https://gustycube.github.io/EasilyAI/overview.html)**

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🔗 Links

- **Documentation**: https://gustycube.github.io/EasilyAI/overview.html
- **PyPI Package**: https://pypi.org/project/easilyai
- **GitHub Repository**: https://github.com/GustyCube/EasilyAI
- **Issues & Support**: https://github.com/GustyCube/EasilyAI/issues
