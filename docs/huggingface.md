# Hugging Face

Hugging Face provides access to thousands of open-source models through their Inference API. This guide covers how to use Hugging Face models through EasilyAI.

## Getting Started

### API Key Setup

1. Sign up at [Hugging Face](https://huggingface.co/)
2. Create an API token in your settings
3. Set it as an environment variable or use it directly

```bash
export HUGGINGFACE_API_KEY="your-huggingface-token"
```

### Basic Usage

```python
from easilyai import create_app

# Create Hugging Face app
app = create_app("HuggingFace", "huggingface", "your-hf-token", "gpt2")

# Generate text
response = app.request("The future of artificial intelligence is")
print(response)
```

## Available Models

Hugging Face hosts thousands of models. Here are some popular categories:

### Text Generation Models

#### GPT-2

- **Model ID**: `gpt2`, `gpt2-medium`, `gpt2-large`, `gpt2-xl`
- **Best for**: General text generation, creative writing
- **Strengths**: Fast, well-established, good for experimentation

```python
gpt2_app = create_app("GPT2", "huggingface", "your-token", "gpt2")
response = gpt2_app.request("Once upon a time")
```

#### FLAN-T5

- **Model ID**: `google/flan-t5-small`, `google/flan-t5-base`, `google/flan-t5-large`
- **Best for**: Instruction following, question answering
- **Strengths**: Good at following instructions, versatile

```python
flan_app = create_app("FLAN", "huggingface", "your-token", "google/flan-t5-base")
response = flan_app.request("Translate to French: Hello, how are you?")
```

#### BLOOM

- **Model ID**: `bigscience/bloom-560m`, `bigscience/bloom-1b1`, `bigscience/bloom-3b`
- **Best for**: Multilingual text generation
- **Strengths**: Supports many languages, open-source

```python
bloom_app = create_app("BLOOM", "huggingface", "your-token", "bigscience/bloom-560m")
multilingual_response = bloom_app.request("Explain machine learning in Spanish")
```

#### Code Generation Models

```python
# CodeT5 for code generation
codet5_app = create_app("CodeT5", "huggingface", "your-token", "Salesforce/codet5-base")
code_response = codet5_app.request("def fibonacci(n):")

# CodeBERT for code understanding
codebert_app = create_app("CodeBERT", "huggingface", "your-token", "microsoft/codebert-base")
```

### Text Classification Models

```python
# Sentiment analysis
sentiment_app = create_app("Sentiment", "huggingface", "your-token", "cardiffnlp/twitter-roberta-base-sentiment-latest")

# Emotion detection
emotion_app = create_app("Emotion", "huggingface", "your-token", "j-hartmann/emotion-english-distilroberta-base")

# Text classification
classifier_app = create_app("Classifier", "huggingface", "your-token", "facebook/bart-large-mnli")
```

### Question Answering Models

```python
# BERT for QA
bert_qa_app = create_app("BERTQA", "huggingface", "your-token", "deepset/roberta-base-squad2")

# DistilBERT (faster)
distilbert_qa_app = create_app("DistilBERTQA", "huggingface", "your-token", "distilbert-base-cased-distilled-squad")
```

### Summarization Models

```python
# BART for summarization
bart_sum_app = create_app("BARTSum", "huggingface", "your-token", "facebook/bart-large-cnn")

# T5 for summarization
t5_sum_app = create_app("T5Sum", "huggingface", "your-token", "t5-base")

# Pegasus for summarization
pegasus_app = create_app("Pegasus", "huggingface", "your-token", "google/pegasus-xsum")
```

## Parameters

### Text Generation Parameters

```python
response = app.request(
    "Write a story about robots",
    max_length=200,          # Maximum length of generated text
    temperature=0.7,         # Controls randomness
    top_p=0.9,              # Nucleus sampling
    top_k=50,               # Top-k sampling
    repetition_penalty=1.1,  # Penalty for repetition
    do_sample=True          # Enable sampling
)
```

### Task-Specific Parameters

```python
# For classification tasks
classification_result = classifier_app.request(
    "This movie is amazing!",
    return_all_scores=True   # Return scores for all labels
)

# For question answering
qa_result = qa_app.request(
    "What is the capital of France?",
    context="France is a country in Europe. Its capital is Paris."
)
```

## Use Cases

### Creative Writing

```python
from easilyai import create_app

# Use GPT-2 for creative writing
creative_app = create_app("Creative", "huggingface", "your-token", "gpt2-medium")

# Story generation
story = creative_app.request(
    "In a world where AI and humans coexist, a young programmer discovers",
    max_length=300,
    temperature=0.8
)

# Poetry generation
poem = creative_app.request(
    "Roses are red, violets are blue,",
    max_length=100,
    temperature=0.9
)

# Dialogue generation
dialogue = creative_app.request(
    "Character A: What do you think about artificial intelligence?\nCharacter B:",
    max_length=150,
    temperature=0.7
)
```

### Code Generation

```python
from easilyai import create_app

# Use CodeT5 for code generation
code_app = create_app("Coder", "huggingface", "your-token", "Salesforce/codet5-base")

# Function generation
function_code = code_app.request(
    "def calculate_factorial(n):",
    max_length=150
)

# Code completion
completed_code = code_app.request(
    "# Sort a list of numbers\ndef sort_numbers(numbers):",
    max_length=100
)

# Code documentation
documentation = code_app.request(
    "# Document this function\ndef binary_search(arr, target):",
    max_length=200
)
```

### Text Analysis

```python
from easilyai import create_app

# Sentiment analysis
sentiment_app = create_app("Sentiment", "huggingface", "your-token", "cardiffnlp/twitter-roberta-base-sentiment-latest")

# Analyze customer feedback
feedback_sentiment = sentiment_app.request(
    "This product exceeded my expectations! Great quality and fast shipping."
)

# Emotion detection
emotion_app = create_app("Emotion", "huggingface", "your-token", "j-hartmann/emotion-english-distilroberta-base")

emotion_result = emotion_app.request(
    "I'm so excited about this new opportunity!"
)

# Topic classification
topic_app = create_app("Topics", "huggingface", "your-token", "facebook/bart-large-mnli")

topic_result = topic_app.request(
    "The stock market reached new highs today as technology companies reported strong earnings.",
    candidate_labels=["business", "technology", "politics", "sports"]
)
```

### Question Answering

```python
from easilyai import create_app

qa_app = create_app("QA", "huggingface", "your-token", "deepset/roberta-base-squad2")

# Document-based QA
document = """
Artificial Intelligence (AI) is a branch of computer science that aims to create 
intelligent machines that can perform tasks that typically require human intelligence. 
Machine learning is a subset of AI that enables computers to learn and improve from 
experience without being explicitly programmed.
"""

answer = qa_app.request(
    question="What is machine learning?",
    context=document
)

# FAQ system
faq_context = """
Our company offers a 30-day return policy. Products can be returned within 30 days 
of purchase for a full refund. Shipping costs are non-refundable unless the item 
was defective or damaged.
"""

faq_answer = qa_app.request(
    question="What is your return policy?",
    context=faq_context
)
```

### Summarization

```python
from easilyai import create_app

# Use BART for summarization
summarizer_app = create_app("Summarizer", "huggingface", "your-token", "facebook/bart-large-cnn")

# Article summarization
long_article = """
[Long article text here - news article, research paper, etc.]
"""

summary = summarizer_app.request(
    long_article,
    max_length=130,
    min_length=30,
    do_sample=False
)

# Meeting notes summarization
meeting_notes = """
[Meeting transcript or notes here]
"""

meeting_summary = summarizer_app.request(
    meeting_notes,
    max_length=100,
    min_length=20
)
```

### Translation

```python
from easilyai import create_app

# Use mBART for translation
translator_app = create_app("Translator", "huggingface", "your-token", "facebook/mbart-large-50-many-to-many-mmt")

# English to French
french_translation = translator_app.request(
    "Hello, how are you today?",
    src_lang="en_XX",
    tgt_lang="fr_XX"
)

# Spanish to English
english_translation = translator_app.request(
    "Hola, ¿cómo estás?",
    src_lang="es_XX",
    tgt_lang="en_XX"
)
```

## Model Discovery

### Finding Models

```python
# Use Hugging Face Hub to discover models
from huggingface_hub import HfApi

def find_models_by_task(task="text-generation", limit=10):
    """Find models for a specific task"""
    api = HfApi()
    models = api.list_models(filter=task, limit=limit, sort="downloads")
    
    for model in models:
        print(f"Model: {model.modelId}")
        print(f"Downloads: {model.downloads}")
        print(f"Tags: {model.tags}")
        print("-" * 40)

# Find popular text generation models
find_models_by_task("text-generation")

# Find sentiment analysis models
find_models_by_task("text-classification")
```

### Model Information

```python
from huggingface_hub import HfApi

def get_model_info(model_id):
    """Get detailed information about a model"""
    api = HfApi()
    model_info = api.model_info(model_id)
    
    print(f"Model: {model_info.modelId}")
    print(f"Task: {model_info.pipeline_tag}")
    print(f"Library: {model_info.library_name}")
    print(f"Downloads: {model_info.downloads}")
    print(f"Likes: {model_info.likes}")
    
    return model_info

# Get info about a specific model
get_model_info("gpt2")
```

## Advanced Usage

### Custom Model Pipelines

```python
from easilyai import create_app

class HuggingFaceMultiModel:
    def __init__(self, token):
        self.token = token
        self.models = {
            "generation": create_app("Gen", "huggingface", token, "gpt2"),
            "sentiment": create_app("Sent", "huggingface", token, "cardiffnlp/twitter-roberta-base-sentiment-latest"),
            "summarization": create_app("Sum", "huggingface", token, "facebook/bart-large-cnn"),
            "qa": create_app("QA", "huggingface", token, "deepset/roberta-base-squad2")
        }
    
    def analyze_text(self, text):
        """Comprehensive text analysis"""
        results = {}
        
        # Sentiment analysis
        results["sentiment"] = self.models["sentiment"].request(text)
        
        # Generate continuation
        results["continuation"] = self.models["generation"].request(
            text, max_length=len(text.split()) + 50
        )
        
        # Summarize if text is long
        if len(text.split()) > 50:
            results["summary"] = self.models["summarization"].request(text)
        
        return results
    
    def answer_question(self, question, context):
        """Answer questions based on context"""
        return self.models["qa"].request(question=question, context=context)

# Usage
multi_model = HuggingFaceMultiModel("your-token")

text = "I love using Hugging Face models for my AI projects. They're so versatile and easy to use!"
analysis = multi_model.analyze_text(text)

print("Sentiment:", analysis["sentiment"])
print("Continuation:", analysis["continuation"])
```

### Model Comparison

```python
from easilyai import create_app

def compare_text_generation_models(prompt, token):
    """Compare different text generation models"""
    models = [
        "gpt2",
        "gpt2-medium",
        "distilgpt2",
        "microsoft/DialoGPT-medium"
    ]
    
    results = {}
    
    for model_name in models:
        try:
            app = create_app("Compare", "huggingface", token, model_name)
            response = app.request(prompt, max_length=100)
            results[model_name] = response
        except Exception as e:
            results[model_name] = f"Error: {e}"
    
    return results

# Compare models
prompt = "The future of artificial intelligence will"
comparison = compare_text_generation_models(prompt, "your-token")

for model, response in comparison.items():
    print(f"{model}:")
    print(f"  {response}")
    print("-" * 50)
```

## Optimization Tips

### Model Selection

```python
# For speed: Use smaller models
fast_app = create_app("Fast", "huggingface", "your-token", "distilgpt2")

# For quality: Use larger models
quality_app = create_app("Quality", "huggingface", "your-token", "gpt2-large")

# For specific tasks: Use specialized models
sentiment_app = create_app("Sentiment", "huggingface", "your-token", "cardiffnlp/twitter-roberta-base-sentiment-latest")
```

### Parameter Tuning

```python
from easilyai import create_app

app = create_app("Tuned", "huggingface", "your-token", "gpt2")

# Conservative generation
conservative = app.request(
    "Explain machine learning",
    temperature=0.3,
    top_p=0.8,
    repetition_penalty=1.2
)

# Creative generation
creative = app.request(
    "Write a fantasy story",
    temperature=0.8,
    top_p=0.95,
    repetition_penalty=1.1
)
```

### Batch Processing

```python
def batch_process_hf(texts, model_name, token, task_type="generation"):
    """Process multiple texts with a Hugging Face model"""
    app = create_app("Batch", "huggingface", token, model_name)
    results = []
    
    for text in texts:
        try:
            if task_type == "generation":
                response = app.request(text, max_length=100)
            elif task_type == "sentiment":
                response = app.request(text)
            else:
                response = app.request(text)
            
            results.append({"text": text, "result": response, "success": True})
        except Exception as e:
            results.append({"text": text, "error": str(e), "success": False})
    
    return results

# Batch sentiment analysis
texts = [
    "I love this product!",
    "This is terrible quality.",
    "It's okay, nothing special."
]

batch_results = batch_process_hf(
    texts, 
    "cardiffnlp/twitter-roberta-base-sentiment-latest",
    "your-token",
    "sentiment"
)
```

## Error Handling

### Common Issues

```python
from easilyai import create_app
from easilyai.exceptions import EasilyAIException

def safe_hf_request(prompt, model_name, token):
    """Make a safe Hugging Face request"""
    try:
        app = create_app("HF", "huggingface", token, model_name)
        return app.request(prompt)
    
    except EasilyAIException as e:
        error_msg = str(e).lower()
        
        if "token" in error_msg:
            return "Error: Invalid Hugging Face token"
        elif "model" in error_msg:
            return f"Error: Model {model_name} not found or not accessible"
        elif "rate limit" in error_msg:
            return "Error: Rate limit exceeded. Please wait."
        else:
            return f"Error: {e}"

# Usage
response = safe_hf_request("Hello world", "gpt2", "your-token")
print(response)
```

### Model Loading Issues

```python
def check_model_availability(model_name, token):
    """Check if a model is available"""
    try:
        app = create_app("Test", "huggingface", token, model_name)
        test_response = app.request("test", max_length=10)
        return True
    except Exception:
        return False

# Check before using
if check_model_availability("gpt2", "your-token"):
    app = create_app("Verified", "huggingface", "your-token", "gpt2")
    response = app.request("Hello!")
else:
    print("Model not available")
```

## Best Practices

### 1. Choose Appropriate Models

```python
# For production: Use well-tested models
production_app = create_app("Prod", "huggingface", "your-token", "gpt2")

# For experimentation: Try cutting-edge models
experimental_app = create_app("Exp", "huggingface", "your-token", "microsoft/DialoGPT-large")
```

### 2. Handle Rate Limits

```python
import time

def rate_limited_request(app, prompt, delay=1):
    """Make request with rate limiting"""
    time.sleep(delay)
    return app.request(prompt)
```

### 3. Cache Results

```python
import json
from pathlib import Path

class HFCache:
    def __init__(self, cache_file="hf_cache.json"):
        self.cache_file = Path(cache_file)
        self.cache = self.load_cache()
    
    def load_cache(self):
        if self.cache_file.exists():
            with open(self.cache_file, 'r') as f:
                return json.load(f)
        return {}
    
    def save_cache(self):
        with open(self.cache_file, 'w') as f:
            json.dump(self.cache, f)
    
    def get_or_request(self, app, prompt, **kwargs):
        cache_key = f"{prompt}_{str(kwargs)}"
        
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        response = app.request(prompt, **kwargs)
        self.cache[cache_key] = response
        self.save_cache()
        
        return response
```

### 4. Monitor Performance

```python
import time

def timed_request(app, prompt, **kwargs):
    """Time a request for performance monitoring"""
    start_time = time.time()
    response = app.request(prompt, **kwargs)
    end_time = time.time()
    
    duration = end_time - start_time
    print(f"Request took {duration:.2f} seconds")
    
    return response
```

## Comparison with Other Services

### Hugging Face vs OpenAI

- **Hugging Face**: Open-source models, more variety, often free
- **OpenAI**: Higher quality, more consistent, better support

### Hugging Face vs Anthropic

- **Hugging Face**: More model choices, experimentation-friendly
- **Anthropic**: Better reasoning, more refined outputs

### When to Use Hugging Face

- Experimenting with different models
- Budget-conscious projects
- Open-source requirement
- Specialized tasks with domain-specific models
- Learning and research projects
- Custom model fine-tuning

Hugging Face provides access to a vast ecosystem of open-source models, making it perfect for experimentation, specialized tasks, and cost-effective AI solutions.