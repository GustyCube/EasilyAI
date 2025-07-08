# Text Generation

Text generation is the core feature of EasilyAI, allowing you to create human-like text using various AI models. This guide covers everything from basic text generation to advanced techniques.

## Basic Text Generation

### Simple Text Generation

The most straightforward way to generate text:

```python
from easilyai import create_app

# Create an app
app = create_app("TextGenerator", "openai", "your-api-key", "gpt-3.5-turbo")

# Generate text
response = app.request("Write a short story about a robot learning to paint")
print(response)
```

### Choosing the Right Model

Different models have different strengths:

```python
# For general conversations and creative writing
gpt_app = create_app("GPT", "openai", "your-openai-key", "gpt-3.5-turbo")

# For more complex reasoning and analysis
gpt4_app = create_app("GPT4", "openai", "your-openai-key", "gpt-4")

# For fast, efficient responses
claude_app = create_app("Claude", "anthropic", "your-anthropic-key", "claude-3-haiku-20240307")

# For advanced reasoning
claude_opus_app = create_app("ClaudeOpus", "anthropic", "your-anthropic-key", "claude-3-opus-20240229")
```

## Advanced Text Generation

### Prompt Engineering

Craft better prompts for better results:

```python
from easilyai import create_app

app = create_app("PromptExpert", "openai", "your-api-key", "gpt-4")

# Basic prompt
basic_response = app.request("Write about dogs")

# Engineered prompt
engineered_prompt = """
Write a comprehensive guide about dogs that includes:
1. Different breeds and their characteristics
2. Basic care requirements
3. Training tips for beginners
4. Health considerations

Format the response with clear headings and bullet points.
Target audience: First-time dog owners
Tone: Friendly and informative
Length: Approximately 500 words
"""

detailed_response = app.request(engineered_prompt)
print(detailed_response)
```

### System Messages and Context

Use system messages to set behavior and context:

```python
from easilyai import create_app

app = create_app("Assistant", "openai", "your-api-key", "gpt-3.5-turbo")

# Create a specialized assistant
system_message = """
You are a Python programming tutor. Always:
- Provide clear, beginner-friendly explanations
- Include practical code examples
- Suggest best practices
- Encourage learning through experimentation
"""

response = app.request(
    "How do I create a list in Python?",
    system_message=system_message
)
print(response)
```

### Temperature and Creativity Control

Control the randomness and creativity of responses:

```python
from easilyai import create_app

app = create_app("CreativeWriter", "openai", "your-api-key", "gpt-3.5-turbo")

prompt = "Write a short poem about the ocean"

# Conservative (temperature = 0.1)
conservative = app.request(prompt, temperature=0.1)

# Balanced (temperature = 0.7)
balanced = app.request(prompt, temperature=0.7)

# Creative (temperature = 1.0)
creative = app.request(prompt, temperature=1.0)

print("Conservative:")
print(conservative)
print("\nBalanced:")
print(balanced)
print("\nCreative:")
print(creative)
```

## Specialized Text Generation

### Code Generation

Generate code with proper formatting and explanations:

```python
from easilyai import create_app

app = create_app("CodeGen", "openai", "your-api-key", "gpt-4")

code_prompt = """
Create a Python function that:
1. Takes a list of numbers as input
2. Returns the average of the numbers
3. Handles empty lists gracefully
4. Includes proper error handling
5. Has docstring documentation

Provide the complete function with examples of how to use it.
"""

code_response = app.request(code_prompt)
print(code_response)
```

### Content Creation

Generate marketing content, blogs, and articles:

```python
from easilyai import create_app

app = create_app("ContentCreator", "openai", "your-api-key", "gpt-4")

# Blog post generation
blog_prompt = """
Write a blog post about sustainable living with:
- Engaging title
- Introduction hook
- 5 practical tips
- Conclusion with call to action
- SEO-friendly keywords: sustainable, eco-friendly, green living
- Target audience: environmentally conscious millennials
- Tone: Informative yet conversational
- Length: 800-1000 words
"""

blog_post = app.request(blog_prompt)
print(blog_post)
```

### Data Analysis and Insights

Generate insights from data descriptions:

```python
from easilyai import create_app

app = create_app("DataAnalyst", "openai", "your-api-key", "gpt-4")

data_prompt = """
Analyze this sales data summary and provide insights:

Q1 2024 Sales Data:
- Total Revenue: $2.5M (up 15% from Q1 2023)
- Top Product Category: Electronics (40% of sales)
- Customer Acquisition: 1,200 new customers
- Return Rate: 5.2%
- Average Order Value: $85

Provide:
1. Key performance indicators analysis
2. Trends and patterns
3. Areas of concern
4. Recommendations for Q2
"""

analysis = app.request(data_prompt)
print(analysis)
```

## Working with Different AI Services

### OpenAI Models

```python
from easilyai import create_app

# GPT-3.5 Turbo - Fast and cost-effective
gpt35_app = create_app("GPT35", "openai", "your-key", "gpt-3.5-turbo")

# GPT-4 - More capable but slower and more expensive
gpt4_app = create_app("GPT4", "openai", "your-key", "gpt-4")

# GPT-4 Turbo - Balanced performance
gpt4_turbo_app = create_app("GPT4Turbo", "openai", "your-key", "gpt-4-turbo")

prompt = "Explain quantum computing in simple terms"

# Compare responses
print("GPT-3.5:")
print(gpt35_app.request(prompt))
print("\nGPT-4:")
print(gpt4_app.request(prompt))
```

### Anthropic Claude

```python
from easilyai import create_app

# Claude 3 Haiku - Fast and efficient
haiku_app = create_app("Haiku", "anthropic", "your-key", "claude-3-haiku-20240307")

# Claude 3 Sonnet - Balanced performance
sonnet_app = create_app("Sonnet", "anthropic", "your-key", "claude-3-sonnet-20240229")

# Claude 3 Opus - Most capable
opus_app = create_app("Opus", "anthropic", "your-key", "claude-3-opus-20240229")

prompt = "Write a technical explanation of machine learning"

responses = {
    "Haiku": haiku_app.request(prompt),
    "Sonnet": sonnet_app.request(prompt),
    "Opus": opus_app.request(prompt)
}

for model, response in responses.items():
    print(f"{model}:")
    print(response)
    print("-" * 50)
```

### Local Models with Ollama

```python
from easilyai import create_app

# No API key needed for local models
ollama_app = create_app("Local", "ollama", "", "llama2")

# Generate text locally
response = ollama_app.request("Tell me about renewable energy")
print(response)
```

## Best Practices

### 1. Prompt Optimization

```python
from easilyai import create_app

app = create_app("Optimizer", "openai", "your-key", "gpt-3.5-turbo")

# Instead of vague prompts
vague_prompt = "Write about AI"

# Use specific, structured prompts
specific_prompt = """
Write a 300-word explanation of artificial intelligence for high school students.
Include:
- Definition of AI
- Two real-world examples
- One benefit and one concern
- Simple, clear language
"""

better_response = app.request(specific_prompt)
print(better_response)
```

### 2. Context Management

```python
from easilyai import create_app

class ContextualChat:
    def __init__(self, service, api_key, model):
        self.app = create_app("ContextChat", service, api_key, model)
        self.context = []
        self.max_context_length = 4000  # Adjust based on model limits
    
    def chat(self, message):
        # Add user message to context
        self.context.append(f"Human: {message}")
        
        # Create prompt with context
        context_prompt = "\n".join(self.context[-10:])  # Last 10 exchanges
        context_prompt += "\nAssistant:"
        
        # Generate response
        response = self.app.request(context_prompt)
        
        # Add response to context
        self.context.append(f"Assistant: {response}")
        
        return response
    
    def clear_context(self):
        self.context = []

# Usage
chat = ContextualChat("openai", "your-key", "gpt-3.5-turbo")
print(chat.chat("Hello, I'm learning about Python"))
print(chat.chat("What are variables?"))
print(chat.chat("Can you give me an example?"))
```

### 3. Error Handling

```python
from easilyai import create_app
from easilyai.exceptions import EasilyAIException

def safe_text_generation(prompt, fallback_model="gpt-3.5-turbo"):
    models = ["gpt-4", "gpt-3.5-turbo", "claude-3-haiku-20240307"]
    
    for model in models:
        try:
            if model.startswith("gpt"):
                app = create_app("TextGen", "openai", "your-openai-key", model)
            else:
                app = create_app("TextGen", "anthropic", "your-anthropic-key", model)
            
            response = app.request(prompt)
            return response
        
        except EasilyAIException as e:
            print(f"Failed with {model}: {e}")
            continue
    
    return "Sorry, I couldn't generate a response with any available model."

# Usage
result = safe_text_generation("Write a haiku about programming")
print(result)
```

### 4. Batch Processing

```python
from easilyai import create_app
import time

def batch_generate(prompts, service="openai", api_key=None, model="gpt-3.5-turbo"):
    app = create_app("BatchGen", service, api_key, model)
    results = []
    
    for i, prompt in enumerate(prompts):
        try:
            response = app.request(prompt)
            results.append(response)
            print(f"Completed {i+1}/{len(prompts)}")
            
            # Rate limiting - adjust based on your API limits
            time.sleep(1)
            
        except Exception as e:
            print(f"Error with prompt {i+1}: {e}")
            results.append(f"Error: {e}")
    
    return results

# Usage
prompts = [
    "Write a haiku about spring",
    "Explain photosynthesis in one sentence",
    "Name three benefits of exercise"
]

results = batch_generate(prompts, api_key="your-key")
for i, result in enumerate(results):
    print(f"Result {i+1}: {result}")
```

## Performance Tips

1. **Choose the right model**: Use faster models for simple tasks, more powerful models for complex ones
2. **Optimize prompts**: Clear, specific prompts generate better results faster
3. **Manage context**: Keep context relevant and within model limits
4. **Handle rate limits**: Implement retry logic and respect API rate limits
5. **Cache responses**: Cache responses for repeated queries to save API calls

## Common Use Cases

- **Content creation**: Blogs, articles, marketing copy
- **Code generation**: Functions, scripts, documentation
- **Data analysis**: Insights, summaries, reports
- **Creative writing**: Stories, poems, scripts
- **Educational content**: Explanations, tutorials, quizzes
- **Customer service**: Responses, FAQs, support documents
- **Research assistance**: Summaries, analysis, research questions

Next, explore [Image Generation](/imagegeneration) or learn about [Pipelines](/pipelines) for chaining text generation with other AI tasks.