# Ollama (Local Models)

Ollama allows you to run large language models locally on your machine, providing privacy, control, and no API costs. This guide covers how to use Ollama through EasilyAI.

## Getting Started

### Installation

First, install Ollama on your system:

**macOS:**
```bash
# Download from https://ollama.ai or use Homebrew
brew install ollama
```

**Linux:**
```bash
curl -fsSL https://ollama.ai/install.sh | sh
```

**Windows:**
Download the installer from [ollama.ai](https://ollama.ai)

### Start Ollama Service

```bash
# Start Ollama service
ollama serve
```

### Pull a Model

```bash
# Pull a model (e.g., Llama 2)
ollama pull llama2

# See available models
ollama list
```

### Basic Usage with EasilyAI

```python
from easilyai import create_app

# Create Ollama app (no API key needed)
app = create_app("Ollama", "ollama", "", "llama2")

# Generate text
response = app.request("Explain machine learning in simple terms")
print(response)
```

## Available Models

Ollama supports many open-source models. Here are some popular options:

### Llama 2

- **Model ID**: `llama2` (7B), `llama2:13b`, `llama2:70b`
- **Best for**: General-purpose tasks, instruction following
- **Strengths**: Well-balanced, good performance

```bash
# Pull different sizes
ollama pull llama2         # 7B parameters
ollama pull llama2:13b     # 13B parameters
ollama pull llama2:70b     # 70B parameters (requires more RAM)
```

```python
# Use different Llama 2 variants
llama2_7b = create_app("Llama7B", "ollama", "", "llama2")
llama2_13b = create_app("Llama13B", "ollama", "", "llama2:13b")
```

### Code Llama

- **Model ID**: `codellama`, `codellama:13b`, `codellama:34b`
- **Best for**: Code generation and programming tasks
- **Strengths**: Specialized for coding tasks

```bash
ollama pull codellama
```

```python
code_app = create_app("CodeLlama", "ollama", "", "codellama")
code_response = code_app.request("Write a Python function to calculate fibonacci numbers")
```

### Mistral

- **Model ID**: `mistral`, `mistral:7b`
- **Best for**: Fast, efficient performance
- **Strengths**: Good performance with smaller size

```bash
ollama pull mistral
```

```python
mistral_app = create_app("Mistral", "ollama", "", "mistral")
response = mistral_app.request("What are the benefits of renewable energy?")
```

### Neural Chat

- **Model ID**: `neural-chat`
- **Best for**: Conversational AI
- **Strengths**: Optimized for chat applications

```bash
ollama pull neural-chat
```

```python
chat_app = create_app("NeuralChat", "ollama", "", "neural-chat")
chat_response = chat_app.request("Let's discuss the future of artificial intelligence")
```

### Dolphin Models

- **Model ID**: `dolphin-mistral`, `dolphin-llama2`
- **Best for**: Uncensored, helpful responses
- **Strengths**: Less restricted outputs

```bash
ollama pull dolphin-mistral
```

```python
dolphin_app = create_app("Dolphin", "ollama", "", "dolphin-mistral")
```

## Parameters

### Text Generation Parameters

```python
response = app.request(
    "Write a creative story",
    temperature=0.8,         # Controls randomness (0.0 to 2.0)
    num_predict=500,        # Number of tokens to predict
    top_p=0.9,              # Nucleus sampling
    top_k=40,               # Top-k sampling
    repeat_penalty=1.1      # Penalty for repetition
)
```

### Advanced Parameters

```python
# Fine-tune model behavior
advanced_response = app.request(
    "Explain quantum computing",
    temperature=0.7,
    num_predict=1000,
    top_p=0.95,
    top_k=50,
    repeat_penalty=1.05,
    presence_penalty=0.5,
    frequency_penalty=0.5
)
```

## Use Cases

### Local Development

Perfect for development without API costs:

```python
from easilyai import create_app

dev_app = create_app("LocalDev", "ollama", "", "codellama")

# Code generation
function_code = dev_app.request(
    "Write a Python class for a simple todo list with add, remove, and list methods"
)

# Code review
review = dev_app.request(
    "Review this code for improvements:\n"
    "def bubble_sort(arr):\n"
    "    for i in range(len(arr)):\n"
    "        for j in range(len(arr)-1):\n"
    "            if arr[j] > arr[j+1]:\n"
    "                arr[j], arr[j+1] = arr[j+1], arr[j]\n"
    "    return arr"
)

# Documentation
docs = dev_app.request(
    "Write documentation for this function:\n"
    "def calculate_distance(lat1, lon1, lat2, lon2):\n"
    "    # Implementation here\n"
    "    pass"
)
```

### Private AI Assistant

Run a personal AI assistant locally:

```python
from easilyai import create_app

assistant_app = create_app("LocalAssistant", "ollama", "", "llama2")

# Personal planning
planning = assistant_app.request(
    "Help me plan a productive workday. I have meetings at 10 AM and 3 PM, "
    "and I need to finish a presentation."
)

# Learning assistance
learning = assistant_app.request(
    "I'm learning React. Can you explain the concept of hooks and give me "
    "a simple example?"
)

# Creative writing
creative = assistant_app.request(
    "Help me brainstorm ideas for a short story about time travel"
)
```

### Content Creation

Generate content without sending data to external services:

```python
from easilyai import create_app

content_app = create_app("ContentCreator", "ollama", "", "mistral")

# Blog posts
blog_post = content_app.request(
    "Write a 500-word blog post about the benefits of local AI models"
)

# Social media content
social_content = content_app.request(
    "Create 5 engaging social media posts about sustainable technology"
)

# Email drafts
email_draft = content_app.request(
    "Draft a professional email to introduce our new software product to potential clients"
)
```

### Research and Analysis

Analyze data and documents privately:

```python
from easilyai import create_app

research_app = create_app("LocalResearch", "ollama", "", "llama2:13b")

# Document analysis
analysis = research_app.request(
    "Analyze this market research data and provide insights:\n"
    "[Your private data here]"
)

# Summarization
summary = research_app.request(
    "Summarize the key points from this internal meeting transcript:\n"
    "[Private transcript here]"
)

# Trend analysis
trends = research_app.request(
    "Based on this sales data, what trends do you see?\n"
    "[Private sales data here]"
)
```

## Model Management

### Listing Models

```python
import subprocess

def list_ollama_models():
    """List all downloaded Ollama models"""
    result = subprocess.run(['ollama', 'list'], capture_output=True, text=True)
    return result.stdout

print(list_ollama_models())
```

### Model Information

```python
def get_model_info(model_name):
    """Get information about a specific model"""
    result = subprocess.run(['ollama', 'show', model_name], capture_output=True, text=True)
    return result.stdout

print(get_model_info("llama2"))
```

### Switching Models

```python
from easilyai import create_app

# Switch between models for different tasks
code_model = create_app("Coder", "ollama", "", "codellama")
chat_model = create_app("Chat", "ollama", "", "neural-chat")
general_model = create_app("General", "ollama", "", "llama2")

# Use appropriate model for each task
code_response = code_model.request("Write a sorting algorithm")
chat_response = chat_model.request("Let's discuss AI ethics")
general_response = general_model.request("Explain photosynthesis")
```

## Performance Optimization

### Hardware Considerations

```python
import psutil

def check_system_resources():
    """Check if system can handle larger models"""
    ram_gb = psutil.virtual_memory().total / (1024**3)
    cpu_count = psutil.cpu_count()
    
    print(f"RAM: {ram_gb:.1f} GB")
    print(f"CPU cores: {cpu_count}")
    
    # Model recommendations based on RAM
    if ram_gb >= 32:
        print("Can run: llama2:70b, codellama:34b")
    elif ram_gb >= 16:
        print("Can run: llama2:13b, codellama:13b")
    else:
        print("Recommended: llama2, mistral, codellama")

check_system_resources()
```

### Model Selection by Task

```python
from easilyai import create_app

class LocalAIManager:
    def __init__(self):
        self.models = {
            "code": create_app("Code", "ollama", "", "codellama"),
            "chat": create_app("Chat", "ollama", "", "neural-chat"),
            "general": create_app("General", "ollama", "", "llama2"),
            "fast": create_app("Fast", "ollama", "", "mistral")
        }
    
    def request(self, prompt, task_type="general", **kwargs):
        """Route request to appropriate model"""
        if task_type not in self.models:
            task_type = "general"
        
        return self.models[task_type].request(prompt, **kwargs)

# Usage
ai_manager = LocalAIManager()

# Automatically use best model for task
code_response = ai_manager.request("Write a web scraper", task_type="code")
chat_response = ai_manager.request("How are you today?", task_type="chat")
quick_response = ai_manager.request("What is 2+2?", task_type="fast")
```

### Temperature Control

```python
from easilyai import create_app

app = create_app("TempControl", "ollama", "", "llama2")

# Consistent, factual responses
factual = app.request(
    "What is the capital of Japan?",
    temperature=0.1
)

# Balanced responses
balanced = app.request(
    "Explain machine learning",
    temperature=0.5
)

# Creative responses
creative = app.request(
    "Write a poem about the ocean",
    temperature=0.9
)
```

## Troubleshooting

### Common Issues

```python
from easilyai import create_app
from easilyai.exceptions import EasilyAIException

def safe_ollama_request(prompt, model="llama2"):
    """Make a safe Ollama request with error handling"""
    try:
        app = create_app("Ollama", "ollama", "", model)
        return app.request(prompt)
    
    except EasilyAIException as e:
        error_msg = str(e).lower()
        
        if "connection" in error_msg:
            return "Error: Ollama service not running. Start with 'ollama serve'"
        elif "model" in error_msg:
            return f"Error: Model {model} not found. Pull with 'ollama pull {model}'"
        else:
            return f"Error: {e}"

# Usage
response = safe_ollama_request("Hello!")
print(response)
```

### Model Loading Issues

```python
import subprocess
import time

def ensure_model_available(model_name):
    """Ensure a model is available, pull if necessary"""
    # Check if model exists
    result = subprocess.run(['ollama', 'list'], capture_output=True, text=True)
    
    if model_name not in result.stdout:
        print(f"Model {model_name} not found. Downloading...")
        subprocess.run(['ollama', 'pull', model_name])
        print(f"Model {model_name} downloaded successfully")
    
    return True

# Ensure model is available before use
ensure_model_available("llama2")
app = create_app("Ollama", "ollama", "", "llama2")
```

### Performance Issues

```python
def optimize_ollama_performance():
    """Tips for optimizing Ollama performance"""
    tips = [
        "Use smaller models (7B) for faster responses",
        "Increase system RAM for larger models",
        "Use SSD storage for better model loading",
        "Close other applications to free up resources",
        "Use GPU acceleration if available (CUDA/Metal)"
    ]
    
    for tip in tips:
        print(f"• {tip}")

optimize_ollama_performance()
```

## Batch Processing

### Processing Multiple Requests

```python
from easilyai import create_app
import time

def batch_process_local(prompts, model="llama2"):
    """Process multiple prompts with local model"""
    app = create_app("BatchOllama", "ollama", "", model)
    results = []
    
    for i, prompt in enumerate(prompts):
        print(f"Processing {i+1}/{len(prompts)}")
        
        try:
            response = app.request(prompt)
            results.append({"prompt": prompt, "response": response, "success": True})
        except Exception as e:
            results.append({"prompt": prompt, "error": str(e), "success": False})
        
        # Small delay to prevent overwhelming the system
        time.sleep(0.1)
    
    return results

# Usage
prompts = [
    "What is Python?",
    "Explain machine learning",
    "Write a hello world program"
]

results = batch_process_local(prompts, "codellama")

for result in results:
    if result["success"]:
        print(f"✓ {result['prompt']}: {result['response'][:50]}...")
    else:
        print(f"✗ {result['prompt']}: {result['error']}")
```

## Best Practices

### 1. Choose the Right Model

```python
# For code tasks
code_app = create_app("Code", "ollama", "", "codellama")

# For general conversation
chat_app = create_app("Chat", "ollama", "", "neural-chat")

# For quick responses
fast_app = create_app("Fast", "ollama", "", "mistral")
```

### 2. Manage System Resources

```python
import psutil

def check_resources_before_request():
    """Check system resources before making requests"""
    ram_usage = psutil.virtual_memory().percent
    cpu_usage = psutil.cpu_percent(interval=1)
    
    if ram_usage > 90:
        print("Warning: High RAM usage. Consider using a smaller model.")
    
    if cpu_usage > 90:
        print("Warning: High CPU usage. Wait before making more requests.")
    
    return ram_usage < 90 and cpu_usage < 90

# Check before making requests
if check_resources_before_request():
    response = app.request("Your prompt here")
```

### 3. Use Appropriate Parameters

```python
# For consistent outputs
consistent = app.request("Explain AI", temperature=0.1)

# For creative outputs
creative = app.request("Write a story", temperature=0.8)

# For longer responses
detailed = app.request("Detailed analysis", num_predict=1000)
```

### 4. Privacy and Security

Ollama keeps everything local, but consider:

- Regularly update models for security patches
- Be aware of model capabilities and limitations
- Monitor system resources and performance
- Keep sensitive data processing local

## Advantages of Local Models

### Privacy
- No data sent to external servers
- Complete control over your information
- Compliance with data protection regulations

### Cost
- No API fees or usage limits
- One-time setup cost for hardware
- Unlimited usage once installed

### Control
- Choose specific models for your needs
- Customize parameters without restrictions
- No rate limits or quotas

### Availability
- Works offline
- No dependency on external services
- Consistent availability

## Limitations

### Performance
- Generally slower than cloud APIs
- Limited by local hardware
- Larger models require significant RAM

### Model Selection
- Limited to open-source models
- May not have latest cutting-edge capabilities
- Updates require manual model downloads

### Maintenance
- Requires system administration
- Model management and updates
- Hardware requirements planning

Ollama provides an excellent way to run AI models locally, offering privacy, control, and cost savings for many use cases. It's particularly valuable for development, private data processing, and scenarios where data security is paramount.