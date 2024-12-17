# AI App Creation

## Overview
EasyAI allows you to initialize an AI app quickly and seamlessly using OpenAI or Ollama.

## Creating an OpenAI App

```python
import easyai

app = easyai.create_app(
    name="my_ai_app",
    service="openai",
    apikey="YOUR_API_KEY",
    model="gpt-4"
)

response = app.request("Tell me a joke about AI.")
print(response)
```
