# Error Handling

## Overview
EasilyAI includes robust error handling with clear, emoji-coded messages for quick debugging.

### Common Errors
- 🔐 **Missing API Key**: "No API key provided! Add your API key to initialize the service."
- 🚫 **Invalid Request**: "The request is invalid. Please check your inputs."
- 🌐 **Connection Error**: "Unable to connect to the API. Ensure the server is running."
- ⏳ **Rate Limit Exceeded**: "Too many requests! Wait and try again."

## Example

```python
try:
    app = easilyai.create_app(name="example", service="openai")
    app.request("Test request")
except Exception as e:
    print(e)
```
