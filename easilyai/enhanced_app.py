"""
Enhanced EasilyAI application with integrated advanced features.

This module provides an enhanced version of EasilyAI that automatically
integrates caching, rate limiting, metrics, cost tracking, and other
advanced features without requiring users to manually configure them.
"""

import time
from typing import Any, Dict, Optional, Union
from .app import EasyAIApp, create_app as _create_app
from .config import get_config, EasilyAIConfig
from .cache import get_cache
from .rate_limit import get_rate_limiter
from .metrics import get_performance_monitor
from .cost_tracking import get_cost_tracker, PricingModel
from .batch import BatchProcessor, BatchRequest
from .enhanced_pipeline import EnhancedPipeline, PipelineTemplate
import logging

logger = logging.getLogger(__name__)


class EasyAIEnhancedApp(EasyAIApp):
    """Enhanced EasilyAI application with automatic optimization features."""
    
    def __init__(self, name: str, service: str, api_key: str, model: str, **kwargs):
        """
        Initialize enhanced EasilyAI app.
        
        Args:
            name: Application name
            service: AI service name
            api_key: API key
            model: Model name
            **kwargs: Additional service-specific parameters
        """
        super().__init__(name, service, api_key, model, **kwargs)
        
        # Get global instances
        self.config = get_config()
        self.cache = get_cache()
        self.rate_limiter = get_rate_limiter()
        self.performance_monitor = get_performance_monitor()
        self.cost_tracker = get_cost_tracker()
        
        # Initialize features based on config
        self._init_features()
    
    def _init_features(self):
        """Initialize advanced features based on configuration."""
        # Set up rate limiting for this service
        service_config = self.config.get_service_config(self.service)
        if service_config:
            self.rate_limiter.set_limit(
                self.service,
                service_config.rate_limit,
                period=60.0
            )
    
    def request(self, prompt: str, task_type: str = "generate_text", **kwargs) -> Any:
        """
        Enhanced request method with automatic optimization.
        
        Args:
            prompt: Input prompt
            task_type: Type of task
            **kwargs: Additional parameters
            
        Returns:
            Response from AI service
        """
        request_id = f"{self.name}_{int(time.time() * 1000)}"
        
        # Start performance monitoring
        self.performance_monitor.start_request(
            request_id, self.service, self.model, task_type
        )
        
        try:
            # Check cache first (if enabled)
            if self.config.cache.enabled:
                cached_response = self.cache.get(
                    self.service, self.model, prompt, task_type=task_type, **kwargs
                )
                if cached_response is not None:
                    logger.debug(f"Cache hit for request {request_id}")
                    
                    # Record metrics for cached request
                    self.performance_monitor.end_request(
                        request_id, self.service, self.model, task_type,
                        success=True, cached=True
                    )
                    
                    return cached_response
            
            # Apply rate limiting
            if not self.rate_limiter.acquire(self.service, timeout=30.0):
                raise Exception(f"Rate limit timeout for {self.service}")
            
            # Estimate cost before making request
            estimated_cost = 0.0
            if self.config.performance.enable_cost_tracking:
                # Simple token estimation (rough)
                estimated_tokens = len(prompt.split()) * 1.3  # Rough estimate
                estimated_cost = PricingModel.estimate_cost(
                    self.service, self.model, task_type,
                    input_tokens=int(estimated_tokens),
                    output_tokens=int(estimated_tokens * 0.5),  # Assume shorter output
                    **kwargs
                )
            
            # Make the actual request
            start_time = time.time()
            response = super().request(prompt, task_type, **kwargs)
            duration = time.time() - start_time
            
            # Cache the response (if enabled)
            if self.config.cache.enabled:
                self.cache.set(
                    self.service, self.model, prompt, response,
                    ttl=self.config.cache.ttl,
                    task_type=task_type, **kwargs
                )
            
            # Record cost
            if self.config.performance.enable_cost_tracking:
                self.cost_tracker.estimate_and_record(
                    self.service, self.model, task_type,
                    input_tokens=int(estimated_tokens),
                    output_tokens=int(estimated_tokens * 0.5),
                    request_id=request_id
                )
            
            # Record performance metrics
            self.performance_monitor.end_request(
                request_id, self.service, self.model, task_type,
                success=True, cached=False
            )
            
            logger.debug(f"Request {request_id} completed in {duration:.2f}s")
            
            return response
            
        except Exception as e:
            # Record failed request
            error_type = type(e).__name__
            self.performance_monitor.end_request(
                request_id, self.service, self.model, task_type,
                success=False, error_type=error_type
            )
            
            logger.error(f"Request {request_id} failed: {e}")
            raise
    
    def get_performance_stats(self) -> dict:
        """Get performance statistics for this app."""
        return self.performance_monitor.get_performance_summary()
    
    def get_cost_summary(self) -> dict:
        """Get cost summary for this app's service/model."""
        return {
            "total_cost": self.cost_tracker.get_total_cost(self.service, self.model),
            "breakdown": self.cost_tracker.get_cost_breakdown()
        }
    
    def create_pipeline(self, name: str = None) -> EnhancedPipeline:
        """Create an enhanced pipeline using this app."""
        pipeline_name = name or f"{self.name}_pipeline"
        pipeline = EnhancedPipeline(pipeline_name)
        return pipeline
    
    def create_batch_processor(self, **kwargs) -> BatchProcessor:
        """Create a batch processor for this app."""
        return BatchProcessor(**kwargs)


class EasyAIManager:
    """Manager for multiple EasilyAI applications with centralized configuration."""
    
    def __init__(self, config: Optional[EasilyAIConfig] = None):
        """
        Initialize EasyAI manager.
        
        Args:
            config: Optional configuration (uses global config if not provided)
        """
        self.config = config or get_config()
        self.apps: Dict[str, EasyAIEnhancedApp] = {}
        
        # Performance tracking
        self.performance_monitor = get_performance_monitor()
        self.cost_tracker = get_cost_tracker()
    
    def get_app(
        self,
        service: str,
        model: Optional[str] = None,
        task_type: str = "generate_text"
    ) -> EasyAIEnhancedApp:
        """
        Get or create an app for the specified service/model.
        
        Args:
            service: AI service name
            model: Model name (uses default if not specified)
            task_type: Task type to optimize for
            
        Returns:
            Enhanced EasilyAI app
        """
        # Get model from config if not specified
        if not model:
            model = self.config.get_default_model(service)
            if not model:
                raise ValueError(f"No default model configured for {service}")
        
        # Create app key
        app_key = f"{service}_{model}"
        
        if app_key not in self.apps:
            # Get API key
            api_key = self.config.get_api_key(service)
            if not api_key:
                raise ValueError(f"No API key configured for {service}")
            
            # Create enhanced app
            app = EasyAIEnhancedApp(
                name=f"{service.title()}App",
                service=service,
                api_key=api_key,
                model=model
            )
            
            self.apps[app_key] = app
        
        return self.apps[app_key]
    
    def text_generation(
        self,
        prompt: str,
        service: Optional[str] = None,
        model: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Generate text using best available service.
        
        Args:
            prompt: Input prompt
            service: Optional service (uses default if not specified)
            model: Optional model
            **kwargs: Additional parameters
            
        Returns:
            Generated text
        """
        service = service or self.config.default_service
        app = self.get_app(service, model, "generate_text")
        return app.request(prompt, "generate_text", **kwargs)
    
    def image_generation(
        self,
        prompt: str,
        service: str = "openai",
        model: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Generate image.
        
        Args:
            prompt: Image description
            service: AI service (default: openai)
            model: Optional model
            **kwargs: Additional parameters
            
        Returns:
            Image URL or data
        """
        app = self.get_app(service, model, "generate_image")
        return app.request(prompt, "generate_image", **kwargs)
    
    def text_to_speech(
        self,
        text: str,
        service: str = "openai",
        model: Optional[str] = None,
        **kwargs
    ) -> Any:
        """
        Convert text to speech.
        
        Args:
            text: Text to convert
            service: AI service (default: openai)
            model: Optional model
            **kwargs: Additional parameters
            
        Returns:
            Audio data or URL
        """
        app = self.get_app(service, model, "text_to_speech")
        return app.request(text, "text_to_speech", **kwargs)
    
    def create_content_pipeline(
        self,
        topic: str,
        text_service: str = "openai",
        image_service: str = "openai"
    ) -> EnhancedPipeline:
        """
        Create a content creation pipeline.
        
        Args:
            topic: Content topic
            text_service: Service for text generation
            image_service: Service for image generation
            
        Returns:
            Configured pipeline
        """
        text_app = self.get_app(text_service)
        image_app = self.get_app(image_service)
        
        pipeline = PipelineTemplate.content_creation_pipeline(text_app, image_app)
        pipeline.set_variable("topic", topic)
        
        return pipeline
    
    def batch_process(
        self,
        requests: List[Dict[str, Any]],
        **processor_kwargs
    ) -> List[Any]:
        """
        Process multiple requests in batch.
        
        Args:
            requests: List of request dictionaries with keys: prompt, service, model, task_type
            **processor_kwargs: Additional BatchProcessor parameters
            
        Returns:
            List of results
        """
        batch_requests = []
        
        for i, req in enumerate(requests):
            # Get app for this request
            app = self.get_app(
                req["service"],
                req.get("model"),
                req.get("task_type", "generate_text")
            )
            
            batch_req = BatchRequest(
                id=f"batch_{i}",
                service=req["service"],
                model=app.model,
                prompt=req["prompt"],
                task_type=req.get("task_type", "generate_text"),
                kwargs=req.get("kwargs", {})
            )
            batch_requests.append(batch_req)
        
        # Create and run batch processor
        processor = BatchProcessor(**processor_kwargs)
        results = processor.process(batch_requests)
        
        return [r.response if r.success else r.error for r in results]
    
    def get_global_stats(self) -> dict:
        """Get global performance and cost statistics."""
        return {
            "performance": self.performance_monitor.get_performance_summary(),
            "costs": self.cost_tracker.get_cost_breakdown(),
            "cache_stats": get_cache().get_stats() if self.config.cache.enabled else None,
            "active_apps": len(self.apps)
        }
    
    def export_metrics(self, file_path: str, format: str = "json"):
        """
        Export metrics to file.
        
        Args:
            file_path: Output file path
            format: Export format ("json" or "prometheus")
        """
        from .metrics import MetricsExporter
        
        exporter = MetricsExporter(get_performance_monitor().collector)
        
        if format == "json":
            exporter.export_json(file_path)
        elif format == "prometheus":
            exporter.export_prometheus(file_path)
        else:
            raise ValueError(f"Unknown export format: {format}")


def create_enhanced_app(
    name: str,
    service: str,
    api_key: str,
    model: str,
    **kwargs
) -> EasyAIEnhancedApp:
    """
    Create an enhanced EasilyAI app with automatic optimization features.
    
    Args:
        name: Application name
        service: AI service name
        api_key: API key
        model: Model name
        **kwargs: Additional parameters
        
    Returns:
        Enhanced EasilyAI app
    """
    return EasyAIEnhancedApp(name, service, api_key, model, **kwargs)


def create_manager(config: Optional[EasilyAIConfig] = None) -> EasyAIManager:
    """
    Create an EasyAI manager for handling multiple apps.
    
    Args:
        config: Optional configuration
        
    Returns:
        EasyAI manager instance
    """
    return EasyAIManager(config)