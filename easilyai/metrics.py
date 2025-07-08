"""
Metrics and monitoring for EasilyAI.

This module provides comprehensive metrics collection and monitoring capabilities.
"""

import time
import threading
from collections import defaultdict, deque
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import json
import logging

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Types of metrics that can be collected."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"


@dataclass
class MetricPoint:
    """A single metric data point."""
    timestamp: float
    value: float
    labels: Dict[str, str] = field(default_factory=dict)


@dataclass
class RequestMetrics:
    """Metrics for a single request."""
    service: str
    model: str
    task_type: str
    duration: float
    success: bool
    error_type: Optional[str] = None
    tokens_used: Optional[int] = None
    cost: Optional[float] = None
    cached: bool = False
    timestamp: float = field(default_factory=time.time)


class MetricsCollector:
    """Collects and aggregates metrics."""
    
    def __init__(self, max_history: int = 10000):
        """
        Initialize metrics collector.
        
        Args:
            max_history: Maximum number of data points to keep in memory
        """
        self.max_history = max_history
        self._metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_history))
        self._counters: Dict[str, float] = defaultdict(float)
        self._gauges: Dict[str, float] = defaultdict(float)
        self._histograms: Dict[str, List[float]] = defaultdict(list)
        self._lock = threading.Lock()
    
    def record_counter(self, name: str, value: float = 1.0, labels: Optional[Dict[str, str]] = None):
        """Record a counter metric."""
        with self._lock:
            key = self._make_key(name, labels)
            self._counters[key] += value
            self._metrics[key].append(MetricPoint(time.time(), value, labels or {}))
    
    def set_gauge(self, name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """Set a gauge metric."""
        with self._lock:
            key = self._make_key(name, labels)
            self._gauges[key] = value
            self._metrics[key].append(MetricPoint(time.time(), value, labels or {}))
    
    def record_histogram(self, name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """Record a histogram metric."""
        with self._lock:
            key = self._make_key(name, labels)
            self._histograms[key].append(value)
            # Keep only recent values to prevent memory growth
            if len(self._histograms[key]) > self.max_history:
                self._histograms[key] = self._histograms[key][-self.max_history:]
            self._metrics[key].append(MetricPoint(time.time(), value, labels or {}))
    
    def record_timer(self, name: str, duration: float, labels: Optional[Dict[str, str]] = None):
        """Record a timer metric (same as histogram for duration)."""
        self.record_histogram(name, duration, labels)
    
    def get_counter(self, name: str, labels: Optional[Dict[str, str]] = None) -> float:
        """Get current counter value."""
        key = self._make_key(name, labels)
        return self._counters.get(key, 0.0)
    
    def get_gauge(self, name: str, labels: Optional[Dict[str, str]] = None) -> float:
        """Get current gauge value."""
        key = self._make_key(name, labels)
        return self._gauges.get(key, 0.0)
    
    def get_histogram_stats(self, name: str, labels: Optional[Dict[str, str]] = None) -> dict:
        """Get statistics for a histogram."""
        key = self._make_key(name, labels)
        values = self._histograms.get(key, [])
        
        if not values:
            return {"count": 0, "sum": 0, "avg": 0, "min": 0, "max": 0, "p50": 0, "p95": 0, "p99": 0}
        
        sorted_values = sorted(values)
        count = len(sorted_values)
        
        return {
            "count": count,
            "sum": sum(sorted_values),
            "avg": sum(sorted_values) / count,
            "min": sorted_values[0],
            "max": sorted_values[-1],
            "p50": self._percentile(sorted_values, 50),
            "p95": self._percentile(sorted_values, 95),
            "p99": self._percentile(sorted_values, 99)
        }
    
    def get_metrics_summary(self) -> dict:
        """Get a summary of all metrics."""
        with self._lock:
            return {
                "counters": dict(self._counters),
                "gauges": dict(self._gauges),
                "histograms": {
                    name: self.get_histogram_stats(name.split("|")[0], 
                                                  self._parse_labels(name))
                    for name in self._histograms.keys()
                }
            }
    
    def reset(self):
        """Reset all metrics."""
        with self._lock:
            self._metrics.clear()
            self._counters.clear()
            self._gauges.clear()
            self._histograms.clear()
    
    def _make_key(self, name: str, labels: Optional[Dict[str, str]]) -> str:
        """Create a key for the metric including labels."""
        if not labels:
            return name
        
        label_str = "|".join(f"{k}={v}" for k, v in sorted(labels.items()))
        return f"{name}|{label_str}"
    
    def _parse_labels(self, key: str) -> Optional[Dict[str, str]]:
        """Parse labels from a metric key."""
        parts = key.split("|")
        if len(parts) <= 1:
            return None
        
        labels = {}
        for part in parts[1:]:
            if "=" in part:
                k, v = part.split("=", 1)
                labels[k] = v
        
        return labels if labels else None
    
    def _percentile(self, sorted_values: List[float], percentile: float) -> float:
        """Calculate percentile from sorted values."""
        if not sorted_values:
            return 0.0
        
        index = (percentile / 100.0) * (len(sorted_values) - 1)
        if index.is_integer():
            return sorted_values[int(index)]
        
        lower = int(index)
        upper = min(lower + 1, len(sorted_values) - 1)
        weight = index - lower
        
        return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


class PerformanceMonitor:
    """Monitor performance metrics for AI requests."""
    
    def __init__(self, collector: Optional[MetricsCollector] = None):
        """Initialize performance monitor."""
        self.collector = collector or MetricsCollector()
        self._active_requests: Dict[str, float] = {}
        self._lock = threading.Lock()
    
    def start_request(self, request_id: str, service: str, model: str, task_type: str):
        """Start timing a request."""
        with self._lock:
            self._active_requests[request_id] = time.time()
        
        # Record request start
        labels = {"service": service, "model": model, "task_type": task_type}
        self.collector.record_counter("requests_started", 1.0, labels)
        self.collector.set_gauge("active_requests", len(self._active_requests))
    
    def end_request(
        self, 
        request_id: str, 
        service: str, 
        model: str, 
        task_type: str,
        success: bool,
        error_type: Optional[str] = None,
        tokens_used: Optional[int] = None,
        cost: Optional[float] = None,
        cached: bool = False
    ):
        """End timing a request and record metrics."""
        duration = 0.0
        
        with self._lock:
            start_time = self._active_requests.pop(request_id, None)
            if start_time:
                duration = time.time() - start_time
        
        # Create labels
        labels = {"service": service, "model": model, "task_type": task_type}
        
        # Record completion metrics
        self.collector.record_counter("requests_completed", 1.0, labels)
        self.collector.record_timer("request_duration", duration, labels)
        
        if success:
            self.collector.record_counter("requests_successful", 1.0, labels)
        else:
            error_labels = {**labels, "error_type": error_type or "unknown"}
            self.collector.record_counter("requests_failed", 1.0, error_labels)
        
        if cached:
            self.collector.record_counter("cache_hits", 1.0, labels)
        else:
            self.collector.record_counter("cache_misses", 1.0, labels)
        
        if tokens_used is not None:
            self.collector.record_histogram("tokens_used", tokens_used, labels)
        
        if cost is not None:
            self.collector.record_histogram("request_cost", cost, labels)
        
        # Update active requests gauge
        self.collector.set_gauge("active_requests", len(self._active_requests))
        
        # Return metrics object
        return RequestMetrics(
            service=service,
            model=model,
            task_type=task_type,
            duration=duration,
            success=success,
            error_type=error_type,
            tokens_used=tokens_used,
            cost=cost,
            cached=cached
        )
    
    def get_performance_summary(self) -> dict:
        """Get a summary of performance metrics."""
        metrics = self.collector.get_metrics_summary()
        
        # Calculate derived metrics
        summary = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "success_rate": 0.0,
            "cache_hit_rate": 0.0,
            "average_duration": 0.0,
            "active_requests": 0
        }
        
        # Aggregate counters
        for name, value in metrics["counters"].items():
            if "requests_completed" in name:
                summary["total_requests"] += value
            elif "requests_successful" in name:
                summary["successful_requests"] += value
            elif "requests_failed" in name:
                summary["failed_requests"] += value
            elif "cache_hits" in name:
                summary["cache_hits"] += value
            elif "cache_misses" in name:
                summary["cache_misses"] += value
        
        # Get gauges
        for name, value in metrics["gauges"].items():
            if "active_requests" in name:
                summary["active_requests"] = value
        
        # Get duration stats
        duration_stats = None
        for name, stats in metrics["histograms"].items():
            if "request_duration" in name:
                if duration_stats is None:
                    duration_stats = stats
                else:
                    # Combine stats (simple average for now)
                    duration_stats["avg"] = (duration_stats["avg"] + stats["avg"]) / 2
        
        if duration_stats:
            summary["average_duration"] = duration_stats["avg"]
        
        # Calculate rates
        if summary["total_requests"] > 0:
            summary["success_rate"] = (summary["successful_requests"] / summary["total_requests"]) * 100
        
        cache_total = summary["cache_hits"] + summary["cache_misses"]
        if cache_total > 0:
            summary["cache_hit_rate"] = (summary["cache_hits"] / cache_total) * 100
        
        return summary


class MetricsExporter:
    """Export metrics to various formats and destinations."""
    
    def __init__(self, collector: MetricsCollector):
        """Initialize metrics exporter."""
        self.collector = collector
    
    def export_json(self, file_path: str):
        """Export metrics to JSON file."""
        metrics = self.collector.get_metrics_summary()
        with open(file_path, 'w') as f:
            json.dump(metrics, f, indent=2)
    
    def export_prometheus(self, file_path: str):
        """Export metrics in Prometheus format."""
        lines = []
        metrics = self.collector.get_metrics_summary()
        
        # Export counters
        for name, value in metrics["counters"].items():
            metric_name = name.split("|")[0]
            labels = self.collector._parse_labels(name)
            
            lines.append(f"# TYPE {metric_name} counter")
            if labels:
                label_str = ",".join(f'{k}="{v}"' for k, v in labels.items())
                lines.append(f"{metric_name}{{{label_str}}} {value}")
            else:
                lines.append(f"{metric_name} {value}")
        
        # Export gauges
        for name, value in metrics["gauges"].items():
            metric_name = name.split("|")[0]
            labels = self.collector._parse_labels(name)
            
            lines.append(f"# TYPE {metric_name} gauge")
            if labels:
                label_str = ",".join(f'{k}="{v}"' for k, v in labels.items())
                lines.append(f"{metric_name}{{{label_str}}} {value}")
            else:
                lines.append(f"{metric_name} {value}")
        
        # Export histograms
        for name, stats in metrics["histograms"].items():
            metric_name = name.split("|")[0]
            labels = self.collector._parse_labels(name)
            
            lines.append(f"# TYPE {metric_name} histogram")
            label_prefix = ""
            if labels:
                label_str = ",".join(f'{k}="{v}"' for k, v in labels.items())
                label_prefix = f"{{{label_str}}}"
            
            lines.append(f"{metric_name}_count{label_prefix} {stats['count']}")
            lines.append(f"{metric_name}_sum{label_prefix} {stats['sum']}")
            lines.append(f"{metric_name}_bucket{{le=\"+Inf\"{label_prefix[1:] if label_prefix else ''}}} {stats['count']}")
        
        with open(file_path, 'w') as f:
            f.write("\n".join(lines))
    
    def get_dashboard_data(self) -> dict:
        """Get data formatted for dashboard display."""
        summary = PerformanceMonitor(self.collector).get_performance_summary()
        metrics = self.collector.get_metrics_summary()
        
        return {
            "summary": summary,
            "detailed_metrics": metrics,
            "timestamp": time.time()
        }


# Global metrics instances
_collector: Optional[MetricsCollector] = None
_monitor: Optional[PerformanceMonitor] = None


def get_metrics_collector() -> MetricsCollector:
    """Get the global metrics collector."""
    global _collector
    if _collector is None:
        _collector = MetricsCollector()
    return _collector


def get_performance_monitor() -> PerformanceMonitor:
    """Get the global performance monitor."""
    global _monitor
    if _monitor is None:
        _monitor = PerformanceMonitor(get_metrics_collector())
    return _monitor


def set_metrics_collector(collector: MetricsCollector):
    """Set the global metrics collector."""
    global _collector, _monitor
    _collector = collector
    _monitor = None  # Will be recreated with new collector


def reset_metrics():
    """Reset all metrics."""
    global _collector, _monitor
    if _collector:
        _collector.reset()
    _collector = None
    _monitor = None