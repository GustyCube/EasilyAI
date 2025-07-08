# Performance Tips

This guide provides strategies to optimize EasilyAI performance, reduce latency, manage costs, and improve overall efficiency.

## Model Selection

### Choose the Right Model for Your Task

Different models have different performance characteristics:

```python
from easilyai import create_app

# Fast models for simple tasks
fast_models = {
    "openai": "gpt-3.5-turbo",
    "anthropic": "claude-3-haiku-20240307",
    "gemini": "gemini-1.5-flash"
}

# Powerful models for complex tasks
powerful_models = {
    "openai": "gpt-4",
    "anthropic": "claude-3-opus-20240229",
    "gemini": "gemini-1.5-pro"
}

# Example: Use fast model for simple Q&A
simple_app = create_app("FastApp", "openai", "your-key", "gpt-3.5-turbo")
response = simple_app.request("What is 2+2?")

# Use powerful model for complex analysis
complex_app = create_app("PowerfulApp", "openai", "your-key", "gpt-4")
analysis = complex_app.request("Analyze the economic implications of AI automation")
```

### Model Performance Comparison

```python
import time
from easilyai import create_app

def benchmark_models():
    models = [
        ("openai", "gpt-3.5-turbo"),
        ("openai", "gpt-4"),
        ("anthropic", "claude-3-haiku-20240307"),
        ("anthropic", "claude-3-sonnet-20240229")
    ]
    
    prompt = "Explain quantum computing in simple terms"
    results = []
    
    for service, model in models:
        try:
            app = create_app("BenchmarkApp", service, "your-key", model)
            
            start_time = time.time()
            response = app.request(prompt)
            end_time = time.time()
            
            results.append({
                "service": service,
                "model": model,
                "duration": end_time - start_time,
                "response_length": len(response)
            })
        
        except Exception as e:
            print(f"Error with {service} {model}: {e}")
    
    # Sort by duration
    results.sort(key=lambda x: x["duration"])
    
    print("Performance Benchmark Results:")
    print("-" * 50)
    for result in results:
        print(f"{result['service']} {result['model']}:")
        print(f"  Duration: {result['duration']:.2f}s")
        print(f"  Response Length: {result['response_length']} chars")
        print()

# Run benchmark
benchmark_models()
```

## Caching Strategies

### Response Caching

Implement caching to avoid repeated API calls:

```python
import hashlib
import json
import time
from pathlib import Path

class ResponseCache:
    def __init__(self, cache_dir: str = "cache", ttl: int = 3600):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.ttl = ttl  # Time to live in seconds
    
    def _cache_key(self, prompt: str, service: str, model: str, **kwargs) -> str:
        """Generate a unique cache key"""
        data = {
            "prompt": prompt,
            "service": service,
            "model": model,
            **kwargs
        }
        
        # Create hash of parameters
        data_str = json.dumps(data, sort_keys=True)
        return hashlib.md5(data_str.encode()).hexdigest()
    
    def get(self, prompt: str, service: str, model: str, **kwargs):
        """Get cached response if available and not expired"""
        cache_key = self._cache_key(prompt, service, model, **kwargs)
        cache_file = self.cache_dir / f"{cache_key}.json"
        
        if cache_file.exists():
            with open(cache_file, 'r') as f:
                cached_data = json.load(f)
            
            # Check if cache is still valid
            if time.time() - cached_data["timestamp"] < self.ttl:
                return cached_data["response"]
        
        return None
    
    def set(self, prompt: str, service: str, model: str, response: str, **kwargs):
        """Cache a response"""
        cache_key = self._cache_key(prompt, service, model, **kwargs)
        cache_file = self.cache_dir / f"{cache_key}.json"
        
        cache_data = {
            "response": response,
            "timestamp": time.time(),
            "prompt": prompt,
            "service": service,
            "model": model,
            "kwargs": kwargs
        }
        
        with open(cache_file, 'w') as f:
            json.dump(cache_data, f)

# Usage
from easilyai import create_app

class CachedAI:
    def __init__(self, cache_ttl: int = 3600):
        self.cache = ResponseCache(ttl=cache_ttl)
    
    def request(self, prompt: str, service: str, model: str, api_key: str, **kwargs):
        # Check cache first
        cached_response = self.cache.get(prompt, service, model, **kwargs)
        if cached_response:
            print("Cache hit!")
            return cached_response
        
        # Make API request
        print("Cache miss - making API request")
        app = create_app("CachedApp", service, api_key, model)
        response = app.request(prompt, **kwargs)
        
        # Cache the response
        self.cache.set(prompt, service, model, response, **kwargs)
        
        return response

# Usage
cached_ai = CachedAI(cache_ttl=1800)  # 30 minutes

# First request - will hit API
response1 = cached_ai.request("What is Python?", "openai", "gpt-3.5-turbo", "your-key")

# Second request - will use cache
response2 = cached_ai.request("What is Python?", "openai", "gpt-3.5-turbo", "your-key")
```

### In-Memory Caching

For faster access, use in-memory caching:

```python
from functools import lru_cache
import time

class InMemoryCachedAI:
    def __init__(self, cache_size: int = 1000):
        self.cache_size = cache_size
        self.cache_stats = {"hits": 0, "misses": 0}
    
    @lru_cache(maxsize=1000)
    def _cached_request(self, prompt: str, service: str, model: str, api_key: str, **kwargs):
        """Internal cached request method"""
        from easilyai import create_app
        
        app = create_app("InMemoryApp", service, api_key, model)
        return app.request(prompt, **kwargs)
    
    def request(self, prompt: str, service: str, model: str, api_key: str, **kwargs):
        # Convert kwargs to hashable format for caching
        kwargs_tuple = tuple(sorted(kwargs.items()))
        
        try:
            response = self._cached_request(prompt, service, model, api_key, **kwargs_tuple)
            self.cache_stats["hits"] += 1
            return response
        except TypeError:
            # If caching fails due to unhashable types, make direct request
            from easilyai import create_app
            app = create_app("DirectApp", service, api_key, model)
            response = app.request(prompt, **kwargs)
            self.cache_stats["misses"] += 1
            return response
    
    def get_cache_stats(self):
        total = self.cache_stats["hits"] + self.cache_stats["misses"]
        hit_rate = (self.cache_stats["hits"] / total * 100) if total > 0 else 0
        return {
            "hits": self.cache_stats["hits"],
            "misses": self.cache_stats["misses"],
            "hit_rate": hit_rate
        }

# Usage
cached_ai = InMemoryCachedAI()

# Make several requests
for i in range(5):
    response = cached_ai.request("What is AI?", "openai", "gpt-3.5-turbo", "your-key")
    print(f"Request {i+1}: {response[:50]}...")

# Check cache performance
stats = cached_ai.get_cache_stats()
print(f"Cache hit rate: {stats['hit_rate']:.1f}%")
```

## Batch Processing

### Parallel Processing

Process multiple requests concurrently:

```python
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
from easilyai import create_app

class BatchProcessor:
    def __init__(self, max_workers: int = 5):
        self.max_workers = max_workers
    
    def process_batch_parallel(self, requests: list, delay: float = 0.1):
        """Process requests in parallel with rate limiting"""
        results = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all requests
            future_to_request = {}
            for i, request in enumerate(requests):
                future = executor.submit(self._process_single_request, request)
                future_to_request[future] = i
                
                # Add delay between submissions for rate limiting
                if i > 0:
                    time.sleep(delay)
            
            # Collect results as they complete
            for future in as_completed(future_to_request):
                request_index = future_to_request[future]
                try:
                    result = future.result()
                    results.append((request_index, result))
                except Exception as e:
                    results.append((request_index, {"error": str(e)}))
        
        # Sort results by original order
        results.sort(key=lambda x: x[0])
        return [result for _, result in results]
    
    def _process_single_request(self, request: dict):
        """Process a single request"""
        try:
            app = create_app(
                "BatchApp",
                request["service"],
                request["api_key"],
                request["model"]
            )
            
            response = app.request(request["prompt"], **request.get("kwargs", {}))
            return {
                "success": True,
                "response": response,
                "prompt": request["prompt"]
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "prompt": request["prompt"]
            }

# Usage
processor = BatchProcessor(max_workers=3)

requests = [
    {"prompt": "What is AI?", "service": "openai", "api_key": "your-key", "model": "gpt-3.5-turbo"},
    {"prompt": "What is ML?", "service": "openai", "api_key": "your-key", "model": "gpt-3.5-turbo"},
    {"prompt": "What is DL?", "service": "anthropic", "api_key": "your-key", "model": "claude-3-haiku-20240307"},
]

import time
start_time = time.time()
results = processor.process_batch_parallel(requests)
end_time = time.time()

print(f"Processed {len(requests)} requests in {end_time - start_time:.2f} seconds")

for result in results:
    if result["success"]:
        print(f"✓ {result['prompt']}: {result['response'][:50]}...")
    else:
        print(f"✗ {result['prompt']}: {result['error']}")
```

## Rate Limiting

### Smart Rate Limiting

Implement intelligent rate limiting:

```python
import time
from collections import defaultdict, deque

class RateLimiter:
    def __init__(self):
        self.limits = {
            "openai": {"requests_per_minute": 60, "tokens_per_minute": 90000},
            "anthropic": {"requests_per_minute": 50, "tokens_per_minute": 100000},
            "gemini": {"requests_per_minute": 60, "tokens_per_minute": 120000}
        }
        
        self.request_times = defaultdict(deque)
        self.token_counts = defaultdict(deque)
    
    def can_make_request(self, service: str, estimated_tokens: int = 1000) -> bool:
        """Check if we can make a request without hitting rate limits"""
        if service not in self.limits:
            return True
        
        now = time.time()
        limits = self.limits[service]
        
        # Clean old entries (older than 1 minute)
        minute_ago = now - 60
        
        # Clean request times
        while self.request_times[service] and self.request_times[service][0] < minute_ago:
            self.request_times[service].popleft()
        
        # Clean token counts
        while self.token_counts[service] and self.token_counts[service][0][0] < minute_ago:
            self.token_counts[service].popleft()
        
        # Check request rate limit
        if len(self.request_times[service]) >= limits["requests_per_minute"]:
            return False
        
        # Check token rate limit
        current_tokens = sum(count for _, count in self.token_counts[service])
        if current_tokens + estimated_tokens > limits["tokens_per_minute"]:
            return False
        
        return True
    
    def record_request(self, service: str, tokens_used: int = 1000):
        """Record a request and token usage"""
        now = time.time()
        self.request_times[service].append(now)
        self.token_counts[service].append((now, tokens_used))
    
    def wait_time(self, service: str) -> float:
        """Calculate how long to wait before making another request"""
        if service not in self.limits:
            return 0
        
        if not self.request_times[service]:
            return 0
        
        # Calculate time until oldest request expires
        oldest_request = self.request_times[service][0]
        wait_time = 60 - (time.time() - oldest_request)
        
        return max(0, wait_time)

class RateLimitedAI:
    def __init__(self):
        self.rate_limiter = RateLimiter()
    
    def request(self, prompt: str, service: str, model: str, api_key: str, **kwargs):
        # Estimate tokens (rough approximation)
        estimated_tokens = len(prompt.split()) * 1.3  # Rough token estimation
        
        # Wait if necessary
        if not self.rate_limiter.can_make_request(service, int(estimated_tokens)):
            wait_time = self.rate_limiter.wait_time(service)
            if wait_time > 0:
                print(f"Rate limit approached, waiting {wait_time:.1f} seconds...")
                time.sleep(wait_time)
        
        # Make request
        from easilyai import create_app
        app = create_app("RateLimitedApp", service, api_key, model)
        response = app.request(prompt, **kwargs)
        
        # Record the request
        response_tokens = len(response.split()) * 1.3
        total_tokens = estimated_tokens + response_tokens
        self.rate_limiter.record_request(service, int(total_tokens))
        
        return response

# Usage
rate_limited_ai = RateLimitedAI()

# Make many requests - will automatically rate limit
for i in range(10):
    response = rate_limited_ai.request(
        f"Tell me a fact about number {i}",
        "openai",
        "gpt-3.5-turbo",
        "your-key"
    )
    print(f"Response {i}: {response[:50]}...")
```

## Prompt Optimization

### Efficient Prompting

Optimize prompts for better performance:

```python
class PromptOptimizer:
    def __init__(self):
        self.optimization_rules = [
            self._remove_redundancy,
            self._add_specificity,
            self._optimize_structure,
            self._add_output_format
        ]
    
    def optimize_prompt(self, prompt: str, task_type: str = "general") -> str:
        """Apply optimization rules to a prompt"""
        optimized = prompt
        
        for rule in self.optimization_rules:
            optimized = rule(optimized, task_type)
        
        return optimized
    
    def _remove_redundancy(self, prompt: str, task_type: str) -> str:
        """Remove redundant words and phrases"""
        # Remove common redundant phrases
        redundant_phrases = [
            "please", "can you", "I would like you to", "could you",
            "if possible", "thank you"
        ]
        
        optimized = prompt
        for phrase in redundant_phrases:
            optimized = optimized.replace(phrase, "")
        
        # Clean up extra spaces
        optimized = " ".join(optimized.split())
        
        return optimized
    
    def _add_specificity(self, prompt: str, task_type: str) -> str:
        """Add specific instructions based on task type"""
        if task_type == "code":
            return f"Generate code: {prompt}. Include comments and error handling."
        elif task_type == "creative":
            return f"Creative writing: {prompt}. Be vivid and engaging."
        elif task_type == "analysis":
            return f"Analyze and provide insights: {prompt}. Include specific examples."
        else:
            return prompt
    
    def _optimize_structure(self, prompt: str, task_type: str) -> str:
        """Optimize prompt structure"""
        # Add clear structure for complex prompts
        if len(prompt) > 100:
            return f"Task: {prompt}\n\nFormat your response clearly with numbered points."
        return prompt
    
    def _add_output_format(self, prompt: str, task_type: str) -> str:
        """Add output format specifications"""
        if task_type == "list":
            return f"{prompt}\n\nProvide the answer as a numbered list."
        elif task_type == "brief":
            return f"{prompt}\n\nKeep the response concise (under 100 words)."
        elif task_type == "detailed":
            return f"{prompt}\n\nProvide a comprehensive, detailed response."
        else:
            return prompt

# Usage
optimizer = PromptOptimizer()

# Optimize different types of prompts
original_prompt = "Please can you tell me about Python programming if possible, thank you"
optimized_prompt = optimizer.optimize_prompt(original_prompt, "code")

print(f"Original: {original_prompt}")
print(f"Optimized: {optimized_prompt}")

# Use optimized prompt
from easilyai import create_app
app = create_app("OptimizedApp", "openai", "your-key", "gpt-3.5-turbo")
response = app.request(optimized_prompt)
```

## Memory Management

### Efficient Memory Usage

Manage memory efficiently for large-scale applications:

```python
import gc
import weakref
from typing import Dict, Any

class MemoryEfficientAI:
    def __init__(self):
        self._app_cache = weakref.WeakValueDictionary()
        self.max_cache_size = 100
        self.request_count = 0
    
    def get_app(self, service: str, api_key: str, model: str):
        """Get or create app with weak reference caching"""
        cache_key = f"{service}_{model}"
        
        if cache_key in self._app_cache:
            return self._app_cache[cache_key]
        
        from easilyai import create_app
        app = create_app(f"MemoryApp_{cache_key}", service, api_key, model)
        self._app_cache[cache_key] = app
        
        return app
    
    def request(self, prompt: str, service: str, model: str, api_key: str, **kwargs):
        """Make request with automatic memory management"""
        app = self.get_app(service, api_key, model)
        response = app.request(prompt, **kwargs)
        
        # Periodic garbage collection
        self.request_count += 1
        if self.request_count % 50 == 0:
            self._cleanup_memory()
        
        return response
    
    def _cleanup_memory(self):
        """Perform memory cleanup"""
        # Force garbage collection
        gc.collect()
        
        # Clear weak reference cache if it gets too large
        if len(self._app_cache) > self.max_cache_size:
            self._app_cache.clear()
        
        print(f"Memory cleanup performed. Cache size: {len(self._app_cache)}")

# Usage
memory_efficient_ai = MemoryEfficientAI()

# Make many requests
for i in range(100):
    response = memory_efficient_ai.request(
        f"Question {i}: What is {i} squared?",
        "openai",
        "gpt-3.5-turbo",
        "your-key"
    )
    print(f"Response {i}: {response}")
```

## Performance Monitoring

### Performance Metrics

Track performance metrics:

```python
import time
import statistics
from collections import defaultdict

class PerformanceMonitor:
    def __init__(self):
        self.metrics = defaultdict(list)
        self.start_times = {}
    
    def start_request(self, request_id: str):
        """Start timing a request"""
        self.start_times[request_id] = time.time()
    
    def end_request(self, request_id: str, service: str, model: str, success: bool = True):
        """End timing a request and record metrics"""
        if request_id not in self.start_times:
            return
        
        duration = time.time() - self.start_times[request_id]
        del self.start_times[request_id]
        
        metric_key = f"{service}_{model}"
        self.metrics[metric_key].append({
            "duration": duration,
            "success": success,
            "timestamp": time.time()
        })
    
    def get_performance_stats(self, service: str, model: str) -> Dict[str, Any]:
        """Get performance statistics for a service/model combination"""
        metric_key = f"{service}_{model}"
        data = self.metrics[metric_key]
        
        if not data:
            return {}
        
        durations = [d["duration"] for d in data]
        successes = [d["success"] for d in data]
        
        return {
            "total_requests": len(data),
            "success_rate": sum(successes) / len(successes) * 100,
            "avg_duration": statistics.mean(durations),
            "median_duration": statistics.median(durations),
            "min_duration": min(durations),
            "max_duration": max(durations),
            "std_duration": statistics.stdev(durations) if len(durations) > 1 else 0
        }
    
    def print_performance_report(self):
        """Print a comprehensive performance report"""
        print("Performance Report")
        print("=" * 50)
        
        for metric_key, data in self.metrics.items():
            service, model = metric_key.split("_", 1)
            stats = self.get_performance_stats(service, model)
            
            if stats:
                print(f"\n{service.upper()} {model}:")
                print(f"  Total Requests: {stats['total_requests']}")
                print(f"  Success Rate: {stats['success_rate']:.1f}%")
                print(f"  Average Duration: {stats['avg_duration']:.2f}s")
                print(f"  Median Duration: {stats['median_duration']:.2f}s")
                print(f"  Duration Range: {stats['min_duration']:.2f}s - {stats['max_duration']:.2f}s")
                print(f"  Standard Deviation: {stats['std_duration']:.2f}s")

class MonitoredAI:
    def __init__(self):
        self.monitor = PerformanceMonitor()
        self.request_counter = 0
    
    def request(self, prompt: str, service: str, model: str, api_key: str, **kwargs):
        """Make a monitored request"""
        request_id = f"req_{self.request_counter}"
        self.request_counter += 1
        
        self.monitor.start_request(request_id)
        
        try:
            from easilyai import create_app
            app = create_app("MonitoredApp", service, api_key, model)
            response = app.request(prompt, **kwargs)
            
            self.monitor.end_request(request_id, service, model, success=True)
            return response
        
        except Exception as e:
            self.monitor.end_request(request_id, service, model, success=False)
            raise e
    
    def get_report(self):
        """Get performance report"""
        self.monitor.print_performance_report()

# Usage
monitored_ai = MonitoredAI()

# Make various requests
models = [
    ("openai", "gpt-3.5-turbo"),
    ("openai", "gpt-4"),
    ("anthropic", "claude-3-haiku-20240307")
]

for service, model in models:
    for i in range(5):
        try:
            response = monitored_ai.request(
                f"Tell me about topic {i}",
                service,
                model,
                "your-key"
            )
            print(f"✓ {service} {model}: {response[:50]}...")
        except Exception as e:
            print(f"✗ {service} {model}: {e}")

# Generate performance report
monitored_ai.get_report()
```

## Cost Optimization

### Cost Tracking

Track and optimize API costs:

```python
class CostTracker:
    def __init__(self):
        # Approximate costs per 1K tokens (as of 2024)
        self.costs = {
            "openai": {
                "gpt-3.5-turbo": {"input": 0.001, "output": 0.002},
                "gpt-4": {"input": 0.03, "output": 0.06},
                "gpt-4-turbo": {"input": 0.01, "output": 0.03}
            },
            "anthropic": {
                "claude-3-haiku-20240307": {"input": 0.00025, "output": 0.00125},
                "claude-3-sonnet-20240229": {"input": 0.003, "output": 0.015},
                "claude-3-opus-20240229": {"input": 0.015, "output": 0.075}
            }
        }
        
        self.usage_log = []
    
    def estimate_tokens(self, text: str) -> int:
        """Rough token estimation"""
        return len(text.split()) * 1.3
    
    def calculate_cost(self, service: str, model: str, input_tokens: int, output_tokens: int) -> float:
        """Calculate cost for a request"""
        if service not in self.costs or model not in self.costs[service]:
            return 0.0
        
        model_costs = self.costs[service][model]
        input_cost = (input_tokens / 1000) * model_costs["input"]
        output_cost = (output_tokens / 1000) * model_costs["output"]
        
        return input_cost + output_cost
    
    def log_request(self, service: str, model: str, prompt: str, response: str):
        """Log a request for cost tracking"""
        input_tokens = self.estimate_tokens(prompt)
        output_tokens = self.estimate_tokens(response)
        cost = self.calculate_cost(service, model, input_tokens, output_tokens)
        
        self.usage_log.append({
            "service": service,
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost": cost,
            "timestamp": time.time()
        })
    
    def get_cost_summary(self) -> Dict[str, Any]:
        """Get cost summary"""
        if not self.usage_log:
            return {}
        
        total_cost = sum(entry["cost"] for entry in self.usage_log)
        total_tokens = sum(entry["input_tokens"] + entry["output_tokens"] for entry in self.usage_log)
        
        # Group by service/model
        by_model = defaultdict(lambda: {"requests": 0, "tokens": 0, "cost": 0})
        
        for entry in self.usage_log:
            key = f"{entry['service']}_{entry['model']}"
            by_model[key]["requests"] += 1
            by_model[key]["tokens"] += entry["input_tokens"] + entry["output_tokens"]
            by_model[key]["cost"] += entry["cost"]
        
        return {
            "total_cost": total_cost,
            "total_tokens": total_tokens,
            "total_requests": len(self.usage_log),
            "by_model": dict(by_model)
        }
    
    def print_cost_report(self):
        """Print cost report"""
        summary = self.get_cost_summary()
        
        if not summary:
            print("No usage data available")
            return
        
        print("Cost Report")
        print("=" * 40)
        print(f"Total Cost: ${summary['total_cost']:.4f}")
        print(f"Total Tokens: {summary['total_tokens']:,}")
        print(f"Total Requests: {summary['total_requests']}")
        print(f"Average Cost per Request: ${summary['total_cost']/summary['total_requests']:.4f}")
        print()
        
        print("By Model:")
        for model, data in summary["by_model"].items():
            print(f"  {model}:")
            print(f"    Requests: {data['requests']}")
            print(f"    Tokens: {data['tokens']:,}")
            print(f"    Cost: ${data['cost']:.4f}")
            print(f"    Avg Cost/Request: ${data['cost']/data['requests']:.4f}")
            print()

class CostOptimizedAI:
    def __init__(self, budget_limit: float = 10.0):
        self.cost_tracker = CostTracker()
        self.budget_limit = budget_limit
    
    def request(self, prompt: str, service: str, model: str, api_key: str, **kwargs):
        """Make a cost-tracked request"""
        # Check budget
        current_cost = self.cost_tracker.get_cost_summary().get("total_cost", 0)
        if current_cost >= self.budget_limit:
            raise Exception(f"Budget limit of ${self.budget_limit} exceeded!")
        
        # Make request
        from easilyai import create_app
        app = create_app("CostOptimizedApp", service, api_key, model)
        response = app.request(prompt, **kwargs)
        
        # Log for cost tracking
        self.cost_tracker.log_request(service, model, prompt, response)
        
        return response
    
    def get_remaining_budget(self) -> float:
        """Get remaining budget"""
        current_cost = self.cost_tracker.get_cost_summary().get("total_cost", 0)
        return max(0, self.budget_limit - current_cost)
    
    def suggest_cheaper_alternative(self, service: str, model: str):
        """Suggest cheaper alternatives"""
        alternatives = {
            ("openai", "gpt-4"): ("openai", "gpt-3.5-turbo"),
            ("anthropic", "claude-3-opus-20240229"): ("anthropic", "claude-3-haiku-20240307")
        }
        
        return alternatives.get((service, model), (service, model))

# Usage
cost_optimized_ai = CostOptimizedAI(budget_limit=5.0)

# Make requests with cost tracking
for i in range(10):
    try:
        response = cost_optimized_ai.request(
            f"Tell me about topic {i}",
            "openai",
            "gpt-3.5-turbo",
            "your-key"
        )
        print(f"Request {i}: {response[:50]}...")
        print(f"Remaining budget: ${cost_optimized_ai.get_remaining_budget():.4f}")
    except Exception as e:
        print(f"Error on request {i}: {e}")
        break

# Print cost report
cost_optimized_ai.cost_tracker.print_cost_report()
```

## Best Practices Summary

1. **Choose appropriate models**: Use fast models for simple tasks, powerful models for complex ones
2. **Implement caching**: Avoid repeated API calls for the same requests
3. **Use batch processing**: Process multiple requests concurrently when possible
4. **Apply rate limiting**: Respect API limits to avoid throttling
5. **Optimize prompts**: Use clear, specific prompts to get better results faster
6. **Monitor performance**: Track metrics to identify bottlenecks
7. **Manage costs**: Track usage and optimize for cost efficiency
8. **Handle errors gracefully**: Implement retry logic and fallbacks
9. **Use efficient data structures**: Manage memory effectively for large-scale applications
10. **Regular cleanup**: Perform periodic memory cleanup and cache management

These performance optimization techniques will help you build efficient, scalable, and cost-effective AI applications with EasilyAI.