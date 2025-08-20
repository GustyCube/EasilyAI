# Core functionality - always available
try:
    from easilyai.app import create_app, create_tts_app
    from easilyai.custom_ai import register_custom_ai
    from easilyai.pipeline import EasilyAIPipeline
    _core_available = True
except ImportError:
    _core_available = False

# Advanced features - available when dependencies are met
_enhanced_available = False
_config_available = False
_cache_available = False
_rate_limit_available = False
_metrics_available = False
_cost_tracking_available = False
_batch_available = False

try:
    from easilyai.config import (
        EasilyAIConfig,
        get_config,
        set_config,
        reset_config
    )
    _config_available = True
except ImportError:
    pass

try:
    from easilyai.cache import (
        ResponseCache,
        MemoryCache,
        FileCache,
        get_cache,
        set_cache,
        reset_cache
    )
    _cache_available = True
except ImportError:
    pass

try:
    from easilyai.rate_limit import (
        RateLimiter,
        ServiceRateLimiter,
        get_rate_limiter,
        set_rate_limiter,
        reset_rate_limiter
    )
    _rate_limit_available = True
except ImportError:
    pass

try:
    from easilyai.metrics import (
        MetricsCollector,
        PerformanceMonitor,
        get_metrics_collector,
        get_performance_monitor,
        set_metrics_collector,
        reset_metrics
    )
    _metrics_available = True
except ImportError:
    pass

try:
    from easilyai.cost_tracking import (
        CostTracker,
        PricingModel,
        get_cost_tracker,
        set_cost_tracker,
        reset_cost_tracker
    )
    _cost_tracking_available = True
except ImportError:
    pass

try:
    from easilyai.batch import (
        BatchProcessor,
        BatchRequest,
        ProcessingMode,
        StreamingBatchProcessor
    )
    _batch_available = True
except ImportError:
    pass

try:
    from easilyai.enhanced_pipeline import (
        EnhancedPipeline,
        PipelineTemplate,
        TaskStatus,
        ExecutionMode
    )
    _enhanced_pipeline_available = True
except ImportError:
    _enhanced_pipeline_available = False

try:
    from easilyai.enhanced_app import (
        EasyAIEnhancedApp,
        EasyAIManager,
        create_enhanced_app,
        create_manager
    )
    _enhanced_available = True
except ImportError:
    pass

# Build __all__ dynamically based on available features
__all__ = []

if _core_available:
    __all__.extend([
        "create_app",
        "create_tts_app", 
        "register_custom_ai",
        "EasilyAIPipeline"  # Basic pipeline - use EnhancedPipeline for new projects
    ])

if _enhanced_available:
    __all__.extend([
        "EasyAIEnhancedApp",
        "EasyAIManager", 
        "create_enhanced_app",
        "create_manager"
    ])

if _enhanced_pipeline_available:
    __all__.extend([
        "EnhancedPipeline",  # Recommended pipeline implementation
        "PipelineTemplate",
        "TaskStatus", 
        "ExecutionMode"
    ])

if _config_available:
    __all__.extend([
        "EasilyAIConfig",
        "get_config",
        "set_config",
        "reset_config"
    ])

if _cache_available:
    __all__.extend([
        "ResponseCache",
        "MemoryCache",
        "FileCache",
        "get_cache",
        "set_cache",
        "reset_cache"
    ])

if _rate_limit_available:
    __all__.extend([
        "RateLimiter",
        "ServiceRateLimiter",
        "get_rate_limiter",
        "set_rate_limiter",
        "reset_rate_limiter"
    ])

if _metrics_available:
    __all__.extend([
        "MetricsCollector",
        "PerformanceMonitor",
        "get_metrics_collector",
        "get_performance_monitor",
        "set_metrics_collector",
        "reset_metrics"
    ])

if _cost_tracking_available:
    __all__.extend([
        "CostTracker",
        "PricingModel",
        "get_cost_tracker",
        "set_cost_tracker",
        "reset_cost_tracker"
    ])

if _batch_available:
    __all__.extend([
        "BatchProcessor",
        "BatchRequest", 
        "ProcessingMode",
        "StreamingBatchProcessor"
    ])


def get_available_features():
    """Get a list of available advanced features."""
    features = {
        "core": _core_available,
        "enhanced_app": _enhanced_available,
        "enhanced_pipeline": _enhanced_pipeline_available,
        "configuration": _config_available,
        "caching": _cache_available,
        "rate_limiting": _rate_limit_available,
        "metrics": _metrics_available,
        "cost_tracking": _cost_tracking_available,
        "batch_processing": _batch_available
    }
    return {name: available for name, available in features.items() if available}


def check_dependencies():
    """Check which dependencies are missing for advanced features."""
    missing = []
    
    if not _core_available:
        missing.append("Core dependencies (openai, anthropic, etc.) for basic functionality")
    
    if not _config_available:
        missing.append("python-dotenv for configuration management")
        
    return missing
