# Text-to-Speech Guide

## Overview
EasyAI supports OpenAI's Text-to-Speech API for converting text into audio files.

## Generate Speech with OpenAI

```python
# Initialize a TTS App
tts_app = easyai.create_tts_app(
    name="tts_app",
    service="openai",
    apikey="YOUR_API_KEY",
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

## Supported Voices
- `onyx`
- `alloy`
- `echo`

Next, explore [Pipelines](./pipelines.md) for chaining tasks.