# Google Gemini

Google Gemini is a powerful multimodal AI model that excels at understanding and generating text, images, and code. This guide covers how to use Gemini through EasilyAI.

## Getting Started

### API Key Setup

1. Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Create an API key
3. Set it as an environment variable or use it directly

```bash
export GEMINI_API_KEY="your-gemini-api-key"
```

### Basic Usage

```python
from easilyai import create_app

# Create Gemini app
app = create_app("Gemini", "gemini", "your-gemini-api-key", "gemini-1.5-flash")

# Generate text
response = app.request("Explain quantum computing in simple terms")
print(response)
```

## Available Models

### Gemini 1.5 Flash

- **Model ID**: `gemini-1.5-flash`
- **Best for**: Fast responses, general tasks
- **Context window**: 1M tokens
- **Strengths**: Speed, efficiency, cost-effective

```python
flash_app = create_app("GeminiFlash", "gemini", "your-key", "gemini-1.5-flash")
response = flash_app.request("Quick question: What is 2+2?")
```

### Gemini 1.5 Pro

- **Model ID**: `gemini-1.5-pro`
- **Best for**: Complex reasoning, analysis
- **Context window**: 2M tokens
- **Strengths**: Advanced reasoning, multimodal capabilities

```python
pro_app = create_app("GeminiPro", "gemini", "your-key", "gemini-1.5-pro")
analysis = pro_app.request("Analyze the economic impact of renewable energy adoption")
```

### Gemini 1.0 Pro

- **Model ID**: `gemini-1.0-pro`
- **Best for**: General purpose tasks
- **Context window**: 30K tokens
- **Strengths**: Balanced performance

```python
pro_app = create_app("GeminiPro", "gemini", "your-key", "gemini-1.0-pro")
response = pro_app.request("Write a short story about artificial intelligence")
```

## Parameters

### Text Generation Parameters

```python
response = app.request(
    "Write a creative story",
    temperature=0.7,           # Controls randomness (0.0 to 1.0)
    max_output_tokens=1000,    # Maximum tokens to generate
    top_p=0.8,                 # Nucleus sampling
    top_k=40                   # Top-k sampling
)
```

### Safety Settings

Gemini includes built-in safety filters. You can adjust these through the API:

```python
# Note: Safety settings are handled internally by Google's API
response = app.request(
    "Write a safety-conscious story about AI",
    safety_settings={
        "HARM_CATEGORY_HARASSMENT": "BLOCK_MEDIUM_AND_ABOVE",
        "HARM_CATEGORY_HATE_SPEECH": "BLOCK_MEDIUM_AND_ABOVE"
    }
)
```

## Use Cases

### Creative Writing

Gemini excels at creative tasks:

```python
from easilyai import create_app

creative_app = create_app("Creative", "gemini", "your-key", "gemini-1.5-pro")

# Story generation
story = creative_app.request(
    "Write a compelling short story about a time traveler who discovers "
    "they can only travel to moments of great historical significance. "
    "Make it 500 words and include dialogue."
)

# Poetry
poem = creative_app.request(
    "Write a haiku about the relationship between technology and nature"
)

# Screenwriting
screenplay = creative_app.request(
    "Write a short dialogue between two AI systems discussing consciousness"
)
```

### Code Generation

Gemini is excellent for programming tasks:

```python
from easilyai import create_app

coder_app = create_app("Coder", "gemini", "your-key", "gemini-1.5-pro")

# Function generation
function_code = coder_app.request(
    "Write a Python function that implements a binary search algorithm. "
    "Include proper error handling, type hints, and docstrings."
)

# Code explanation
explanation = coder_app.request(
    "Explain this Python code step by step:\n"
    "def fibonacci(n):\n"
    "    if n <= 1:\n"
    "        return n\n"
    "    return fibonacci(n-1) + fibonacci(n-2)"
)

# Code review
review = coder_app.request(
    "Review this code for potential improvements:\n"
    "def calculate_average(numbers):\n"
    "    return sum(numbers) / len(numbers)"
)
```

### Data Analysis

Use Gemini for analyzing data and generating insights:

```python
from easilyai import create_app

analyst_app = create_app("Analyst", "gemini", "your-key", "gemini-1.5-pro")

# Data analysis
analysis = analyst_app.request(
    "Analyze this sales data and provide insights:\n"
    "January: $45,000\n"
    "February: $52,000\n"
    "March: $48,000\n"
    "April: $61,000\n"
    "May: $59,000\n"
    "June: $67,000\n"
    "What trends do you see? What recommendations would you make?"
)

# Research synthesis
research = analyst_app.request(
    "Synthesize the key findings from recent research on climate change impacts "
    "on agriculture. Focus on the most significant threats and adaptation strategies."
)
```

### Educational Content

Create educational materials:

```python
from easilyai import create_app

teacher_app = create_app("Teacher", "gemini", "your-key", "gemini-1.5-flash")

# Lesson plans
lesson = teacher_app.request(
    "Create a lesson plan for teaching 8th graders about photosynthesis. "
    "Include learning objectives, activities, and assessment methods."
)

# Explanations
explanation = teacher_app.request(
    "Explain the concept of machine learning to a 10-year-old using simple "
    "analogies and examples they can relate to."
)

# Quiz generation
quiz = teacher_app.request(
    "Create a 10-question multiple choice quiz about the solar system. "
    "Include answer explanations for each question."
)
```

## Advanced Features

### Long Context Processing

Gemini 1.5 models support very long contexts:

```python
from easilyai import create_app

long_context_app = create_app("LongContext", "gemini", "your-key", "gemini-1.5-pro")

# Process large documents
large_document = """
[Insert very long document here - up to 1M tokens for Flash, 2M for Pro]
"""

summary = long_context_app.request(
    f"Please summarize this document and extract the key points:\n\n{large_document}"
)
```

### Multimodal Capabilities

While EasilyAI focuses on text, Gemini supports multimodal inputs through Google's API:

```python
# Note: Direct multimodal support would require additional implementation
# This is a text-based approach to describe images
image_analysis = app.request(
    "Describe what would be in an image that shows 'a bustling city street at night "
    "with neon signs and reflections on wet pavement'"
)
```

### Chain of Thought Reasoning

Leverage Gemini's reasoning capabilities:

```python
from easilyai import create_app

reasoning_app = create_app("Reasoning", "gemini", "your-key", "gemini-1.5-pro")

# Complex problem solving
solution = reasoning_app.request(
    "Solve this step by step:\n"
    "A company has 100 employees. 60% work in engineering, 25% in sales, "
    "and the rest in administration. If the company grows by 50% and maintains "
    "the same proportions, how many new employees will be needed in each department?"
)

# Logical reasoning
logic = reasoning_app.request(
    "Think through this logical puzzle step by step:\n"
    "All cats are animals. Some animals are pets. Therefore, some cats are pets. "
    "Is this reasoning valid? Explain your analysis."
)
```

## Optimization Tips

### Model Selection

Choose the right model for your use case:

```python
# For quick responses and high throughput
flash_app = create_app("Quick", "gemini", "your-key", "gemini-1.5-flash")

# For complex reasoning and analysis
pro_app = create_app("Complex", "gemini", "your-key", "gemini-1.5-pro")

# For budget-conscious applications
budget_app = create_app("Budget", "gemini", "your-key", "gemini-1.0-pro")
```

### Performance Optimization

```python
from easilyai import create_app

app = create_app("Optimized", "gemini", "your-key", "gemini-1.5-flash")

# Use lower temperature for consistent results
consistent_response = app.request(
    "Summarize this article",
    temperature=0.1
)

# Use higher temperature for creative tasks
creative_response = app.request(
    "Write a creative story",
    temperature=0.8
)

# Limit output tokens for concise responses
brief_response = app.request(
    "Explain quantum computing briefly",
    max_output_tokens=100
)
```

### Prompt Engineering

Optimize your prompts for better results:

```python
from easilyai import create_app

app = create_app("Optimized", "gemini", "your-key", "gemini-1.5-pro")

# Structured prompts work well
structured_prompt = """
Task: Write a product description
Product: Wireless headphones
Requirements:
- Highlight key features
- Target audience: professionals
- Tone: professional yet engaging
- Length: 100-150 words

Please format the response with clear sections for features and benefits.
"""

description = app.request(structured_prompt)
```

## Error Handling

### Common Errors

```python
from easilyai import create_app
from easilyai.exceptions import EasilyAIException

app = create_app("Gemini", "gemini", "your-key", "gemini-1.5-flash")

try:
    response = app.request("Your prompt here")
    print(response)
except EasilyAIException as e:
    error_msg = str(e).lower()
    
    if "api key" in error_msg:
        print("Invalid API key. Please check your Gemini API key.")
    elif "quota" in error_msg:
        print("API quota exceeded. Please check your usage limits.")
    elif "safety" in error_msg:
        print("Content was blocked by safety filters. Try rephrasing your prompt.")
    else:
        print(f"Gemini API error: {e}")
```

### Rate Limiting

```python
import time
from easilyai import create_app

def gemini_request_with_retry(prompt, max_retries=3):
    app = create_app("Gemini", "gemini", "your-key", "gemini-1.5-flash")
    
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

### 1. Choose the Right Model

- **Gemini 1.5 Flash**: Fast responses, high throughput
- **Gemini 1.5 Pro**: Complex reasoning, long context
- **Gemini 1.0 Pro**: Balanced performance

### 2. Optimize Prompts

```python
# Good: Specific and clear
good_prompt = "Write a 200-word product description for wireless earbuds targeting fitness enthusiasts"

# Better: Structured with examples
better_prompt = """
Write a product description for wireless earbuds.
Target audience: Fitness enthusiasts
Key features: Waterproof, 12-hour battery, noise cancellation
Tone: Energetic and motivational
Length: 200 words
Format: Include headline, features list, and call-to-action
"""
```

### 3. Use Safety Features

Gemini includes built-in safety filters that help ensure responsible AI use.

### 4. Monitor Usage

Track your API usage to manage costs:

```python
import time
from collections import defaultdict

class GeminiUsageTracker:
    def __init__(self):
        self.usage_log = defaultdict(int)
    
    def track_request(self, model, tokens_used):
        self.usage_log[model] += tokens_used
    
    def get_usage_summary(self):
        return dict(self.usage_log)
```

## Comparison with Other Services

### Gemini vs GPT-4

- **Gemini**: Longer context window, integrated safety features
- **GPT-4**: More established ecosystem, wider adoption

### Gemini vs Claude

- **Gemini**: Better for code generation, multimodal capabilities
- **Claude**: Better for reasoning, analysis tasks

### When to Use Gemini

- Long document processing
- Code generation and analysis
- Creative writing projects
- Educational content creation
- Research and analysis tasks

Google Gemini offers powerful capabilities through EasilyAI's simple interface. Its long context windows and multimodal capabilities make it excellent for complex, context-heavy tasks.