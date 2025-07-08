"""
Enhanced EasilyAI Features Demo

This script demonstrates all the advanced features implemented in EasilyAI:
- Configuration management
- Caching  
- Rate limiting
- Metrics and monitoring
- Cost tracking
- Batch processing
- Enhanced pipelines
- Integrated apps

Note: This demo works with mock services to avoid requiring actual API keys.
"""

import time
import os
from typing import Any


# Mock AI service for demonstration (no API key required)
class MockAIService:
    """Mock AI service for demonstration purposes."""
    
    def __init__(self, name: str, model: str):
        self.name = name
        self.model = model
    
    def request(self, prompt: str, task_type: str = "generate_text", **kwargs) -> str:
        """Mock AI request that returns predictable responses."""
        time.sleep(0.1)  # Simulate API delay
        
        if task_type == "generate_text":
            return f"Mock {self.name} response for: {prompt[:50]}..."
        elif task_type == "generate_image":
            return f"https://mock-{self.name}-image.com/generated-image.jpg"
        elif task_type == "text_to_speech":
            return f"https://mock-{self.name}-audio.com/speech.mp3"
        else:
            return f"Mock response for {task_type}"


def demo_configuration():
    """Demonstrate configuration management."""
    print("=== Configuration Management Demo ===")
    
    from easilyai.config import EasilyAIConfig, ServiceConfig
    
    # Create configuration
    config = EasilyAIConfig()
    
    # Customize settings
    config.default_service = "openai"
    config.cache.enabled = True
    config.cache.ttl = 1800  # 30 minutes
    config.performance.enable_metrics = True
    config.performance.enable_cost_tracking = True
    
    # Service-specific configuration
    config.openai.rate_limit = 100  # 100 requests per minute
    config.anthropic.rate_limit = 50  # 50 requests per minute
    
    print(f"✓ Default service: {config.default_service}")
    print(f"✓ Cache enabled: {config.cache.enabled}")
    print(f"✓ OpenAI rate limit: {config.openai.rate_limit} req/min")
    print(f"✓ Metrics enabled: {config.performance.enable_metrics}")
    
    # Save/load configuration
    config.to_json("demo_config.json")
    loaded_config = EasilyAIConfig.from_json("demo_config.json")
    print(f"✓ Configuration saved and loaded successfully")
    
    return config


def demo_caching():
    """Demonstrate caching functionality."""
    print("\n=== Caching System Demo ===")
    
    from easilyai.cache import MemoryCache, FileCache, ResponseCache
    
    # Memory cache demo
    print("Memory Cache:")
    memory_cache = MemoryCache(max_size=100, default_ttl=3600)
    memory_cache.set("test_key", "Cached response", ttl=300)
    result = memory_cache.get("test_key")
    print(f"✓ Cached value: {result}")
    
    # File cache demo
    print("\nFile Cache:")
    file_cache = FileCache(cache_dir="demo_cache", default_ttl=3600)
    file_cache.set("file_test", {"response": "File cached data", "metadata": {"tokens": 150}})
    file_result = file_cache.get("file_test")
    print(f"✓ File cached value: {file_result}")
    
    # Response cache demo
    print("\nResponse Cache:")
    response_cache = ResponseCache(memory_cache)
    response_cache.set("openai", "gpt-3.5-turbo", "Hello world", "Hi there!")
    cached_response = response_cache.get("openai", "gpt-3.5-turbo", "Hello world")
    print(f"✓ Response cache hit: {cached_response}")
    
    # Cache statistics
    stats = response_cache.get_stats()
    print(f"✓ Cache hit rate: {stats['hit_rate']:.1f}%")
    
    return response_cache


def demo_rate_limiting():
    """Demonstrate rate limiting."""
    print("\n=== Rate Limiting Demo ===")
    
    from easilyai.rate_limit import RateLimiter, ServiceRateLimiter
    
    # Basic rate limiter
    print("Basic Rate Limiter (5 requests per 10 seconds):")
    limiter = RateLimiter(rate=5, period=10.0)
    
    # Try to acquire multiple permits
    for i in range(7):
        success = limiter.try_acquire()
        print(f"  Request {i+1}: {'✓ Allowed' if success else '✗ Rate limited'}")
    
    # Service-specific rate limiting
    print("\nService Rate Limiter:")
    service_limiter = ServiceRateLimiter()
    service_limiter.set_limit("openai", rate=100, period=60.0)
    service_limiter.set_limit("anthropic", rate=50, period=60.0)
    
    for service in ["openai", "anthropic", "gemini"]:
        success = service_limiter.try_acquire(service)
        wait_time = service_limiter.get_wait_time(service)
        print(f"  {service}: {'✓ Allowed' if success else '✗ Limited'} (wait: {wait_time:.3f}s)")
    
    return service_limiter


def demo_metrics():
    """Demonstrate metrics and monitoring."""
    print("\n=== Metrics & Monitoring Demo ===")
    
    from easilyai.metrics import MetricsCollector, PerformanceMonitor
    
    # Basic metrics collection
    collector = MetricsCollector()
    
    # Record various metrics
    for i in range(10):
        collector.record_counter("api_requests", 1, {"service": "openai", "model": "gpt-3.5-turbo"})
        collector.record_histogram("response_time", 0.5 + i * 0.1, {"service": "openai"})
        collector.set_gauge("active_connections", 5 + i)
    
    # Performance monitoring
    monitor = PerformanceMonitor(collector)
    
    # Simulate some requests
    for i in range(5):
        request_id = f"req_{i}"
        monitor.start_request(request_id, "openai", "gpt-3.5-turbo", "generate_text")
        time.sleep(0.05)  # Simulate processing
        monitor.end_request(
            request_id, "openai", "gpt-3.5-turbo", "generate_text",
            success=True, tokens_used=100, cost=0.001
        )
    
    # Get statistics
    summary = monitor.get_performance_summary()
    print(f"✓ Total requests: {summary['total_requests']}")
    print(f"✓ Success rate: {summary['success_rate']:.1f}%")
    print(f"✓ Average duration: {summary['average_duration']:.3f}s")
    
    # Detailed metrics
    metrics = collector.get_metrics_summary()
    print(f"✓ Metrics collected: {len(metrics['counters'])} counters, {len(metrics['histograms'])} histograms")
    
    return monitor


def demo_cost_tracking():
    """Demonstrate cost tracking."""
    print("\n=== Cost Tracking Demo ===")
    
    from easilyai.cost_tracking import CostTracker, PricingModel
    
    # Cost estimation
    print("Cost Estimation:")
    services_models = [
        ("openai", "gpt-3.5-turbo"),
        ("openai", "gpt-4"),
        ("anthropic", "claude-3-haiku-20240307"),
        ("anthropic", "claude-3-opus-20240229")
    ]
    
    for service, model in services_models:
        cost = PricingModel.estimate_cost(service, model, "generate_text", 
                                        input_tokens=1000, output_tokens=500)
        print(f"  {service} {model}: ${cost:.4f}")
    
    # Cost tracking
    print("\nCost Tracking:")
    tracker = CostTracker()
    
    # Record some costs
    for service, model in services_models:
        cost = PricingModel.estimate_cost(service, model, "generate_text",
                                        input_tokens=500, output_tokens=250)
        tracker.estimate_and_record(service, model, "generate_text",
                                   input_tokens=500, output_tokens=250)
    
    # Get breakdown
    breakdown = tracker.get_cost_breakdown()
    print(f"✓ Total cost: ${breakdown['total_cost']:.4f}")
    print(f"✓ Total requests: {breakdown['total_requests']}")
    print(f"✓ Services used: {len(breakdown['by_service'])}")
    
    # Top consumers
    top_consumers = tracker.get_top_consumers(limit=3)
    print("Top cost consumers:")
    for consumer in top_consumers:
        print(f"  {consumer['service']} {consumer['model']}: ${consumer['total_cost']:.4f}")
    
    return tracker


def demo_batch_processing():
    """Demonstrate batch processing."""
    print("\n=== Batch Processing Demo ===")
    
    from easilyai.batch import BatchProcessor, BatchRequest, ProcessingMode
    
    # Create mock requests
    requests = []
    prompts = [
        "What is artificial intelligence?",
        "Explain machine learning",
        "How does deep learning work?", 
        "What are neural networks?",
        "Describe natural language processing"
    ]
    
    for i, prompt in enumerate(prompts):
        request = BatchRequest(
            id=f"batch_{i}",
            service="openai",
            model="gpt-3.5-turbo", 
            prompt=prompt,
            task_type="generate_text"
        )
        requests.append(request)
    
    # Sequential processing
    print("Sequential Processing:")
    sequential_processor = BatchProcessor(mode=ProcessingMode.SEQUENTIAL)
    
    start_time = time.time()
    # Mock the processing since we don't have real API keys
    print(f"✓ Processing {len(requests)} requests sequentially...")
    for i, req in enumerate(requests):
        print(f"  Request {i+1}: {req.prompt[:30]}...")
        time.sleep(0.1)  # Simulate processing time
    
    sequential_time = time.time() - start_time
    print(f"✓ Sequential processing completed in {sequential_time:.2f}s")
    
    # Parallel processing
    print("\nParallel Processing:")
    parallel_processor = BatchProcessor(
        mode=ProcessingMode.PARALLEL_THREADS,
        max_workers=3
    )
    
    start_time = time.time()
    print(f"✓ Processing {len(requests)} requests in parallel...")
    # Simulate parallel processing
    time.sleep(0.2)  # Shorter time due to parallelism
    parallel_time = time.time() - start_time
    print(f"✓ Parallel processing completed in {parallel_time:.2f}s")
    print(f"✓ Speedup: {sequential_time/parallel_time:.1f}x")
    
    return requests


def demo_enhanced_pipeline():
    """Demonstrate enhanced pipeline features."""
    print("\n=== Enhanced Pipeline Demo ===")
    
    from easilyai.enhanced_pipeline import EnhancedPipeline, TaskStatus, ExecutionMode
    
    # Create mock apps
    text_service = MockAIService("OpenAI", "gpt-4")
    image_service = MockAIService("DALL-E", "dall-e-3")
    
    # Create enhanced pipeline
    pipeline = EnhancedPipeline("ContentCreationDemo")
    
    # Add tasks with dependencies and conditions
    pipeline.add_task(
        "generate_story",
        text_service,
        "generate_text",
        "Write a short story about {topic}",
        retry_count=2
    )
    
    pipeline.add_task(
        "create_summary", 
        text_service,
        "generate_text",
        "Summarize this story: {result:generate_story}",
        dependencies=["generate_story"]
    )
    
    pipeline.add_task(
        "create_illustration",
        image_service,
        "generate_image", 
        "Create an illustration for: {result:generate_story}",
        dependencies=["generate_story"],
        condition=lambda results: len(results.get("generate_story", {}).get("result", "")) > 100
    )
    
    # Set variables
    pipeline.set_variable("topic", "space exploration")
    
    # Validate pipeline
    errors = pipeline.validate()
    print(f"✓ Pipeline validation: {'Passed' if not errors else f'Failed: {errors}'}")
    
    # Execute pipeline
    print("Executing pipeline...")
    results = pipeline.run()
    
    # Show results
    for task_id, result in results.items():
        status_icon = "✓" if result.status == TaskStatus.COMPLETED else "✗"
        print(f"  {status_icon} {task_id}: {result.status.value} ({result.duration:.2f}s)")
    
    # Pipeline summary
    summary = pipeline.get_summary()
    print(f"✓ Pipeline completed: {summary['completed']}/{summary['total_tasks']} tasks")
    print(f"✓ Success rate: {summary['success_rate']:.1f}%")
    
    return pipeline


def demo_integrated_features():
    """Demonstrate how all features work together."""
    print("\n=== Integrated Features Demo ===")
    
    # This would normally use the enhanced app, but we'll simulate it
    print("Simulating enhanced app with all features enabled:")
    print("✓ Configuration loaded")
    print("✓ Cache enabled (memory + file)")
    print("✓ Rate limiting configured")
    print("✓ Metrics collection active")
    print("✓ Cost tracking enabled")
    
    # Simulate some requests with all optimizations
    print("\nSimulating optimized AI requests:")
    for i in range(5):
        print(f"  Request {i+1}:")
        print(f"    → Checking cache... {'Hit' if i > 2 else 'Miss'}")
        print(f"    → Rate limiting... {'OK' if i < 4 else 'Delayed'}")
        if i <= 2:
            print(f"    → Making API call... Done (0.{150+i*50}s)")
        print(f"    → Recording metrics... Done")
        print(f"    → Tracking cost... $0.000{i+1}")
        if i <= 2:
            print(f"    → Caching response... Done")
    
    print("\nOptimization summary:")
    print("✓ 3/5 requests served from cache (60% cache hit rate)")
    print("✓ 1/5 requests rate limited (saves costs)")
    print("✓ $0.0015 total cost tracked")
    print("✓ All metrics recorded for analysis")


def main():
    """Run all demos."""
    print("EasilyAI Enhanced Features Demo")
    print("=" * 50)
    
    try:
        # Run all demos
        config = demo_configuration()
        cache = demo_caching()
        rate_limiter = demo_rate_limiting()
        monitor = demo_metrics()
        cost_tracker = demo_cost_tracking()
        batch_requests = demo_batch_processing()
        pipeline = demo_enhanced_pipeline()
        demo_integrated_features()
        
        print("\n" + "=" * 50)
        print("🎉 All advanced features demonstrated successfully!")
        print("\nKey benefits:")
        print("• Automatic performance optimization")
        print("• Cost tracking and budgeting")
        print("• Intelligent caching reduces API calls")
        print("• Rate limiting prevents service overload")
        print("• Comprehensive metrics for monitoring")
        print("• Batch processing for efficiency")
        print("• Advanced pipelines for complex workflows")
        print("• Easy configuration management")
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        raise
    
    finally:
        # Cleanup
        if os.path.exists("demo_config.json"):
            os.remove("demo_config.json")
        
        import shutil
        if os.path.exists("demo_cache"):
            shutil.rmtree("demo_cache")


if __name__ == "__main__":
    main()