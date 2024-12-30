# App Creation

## Overview
EasilyAI allows you to initialize an AI app quickly and seamlessly using OpenAI or Ollama.

EasilyAI supports many different services. You can view a full list here.  

## Creating an OpenAI App

```python
import easilyai

app = easilyai.create_app(
    name="my_ai_app",
    service="openai",
    apikey="YOUR_API_KEY",
    model="gpt-4o"
)

response = app.request("Tell me a joke about AI.")
print(response)
```

## Creating an Ollama App

For local models using Ollama:

```python
app = easilyai.create_app(
    name="my_ai_app",
    service="ollama",
    model="llama2"
)

response = app.request("What is the future of AI?")
print(response)
```

Learn more about [Text-to-Speech](./texttospeech.md) in EasyAI.