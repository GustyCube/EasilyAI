"""
Cost tracking for EasilyAI.

This module provides cost tracking and optimization capabilities for AI API usage.
"""

import time
import json
from typing import Dict, Optional, List, Any
from dataclasses import dataclass, field
from pathlib import Path
import threading
from datetime import datetime, timedelta
from collections import defaultdict


@dataclass
class CostEntry:
    """Represents a single cost entry."""
    timestamp: float
    service: str
    model: str
    task_type: str
    tokens_used: Optional[int] = None
    cost: float = 0.0
    request_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class PricingModel:
    """Pricing model for different AI services."""
    
    # Pricing per 1K tokens (approximate, may vary)
    PRICING = {
        "openai": {
            "gpt-3.5-turbo": {"input": 0.0015, "output": 0.002},
            "gpt-4": {"input": 0.03, "output": 0.06},
            "gpt-4-turbo": {"input": 0.01, "output": 0.03},
            "dall-e-2": {"image": 0.02},  # per image
            "dall-e-3": {"image": 0.04},  # per image (standard)
            "tts-1": {"audio": 0.015},  # per 1K characters
            "whisper-1": {"audio": 0.006}  # per minute
        },
        "anthropic": {
            "claude-3-haiku-20240307": {"input": 0.00025, "output": 0.00125},
            "claude-3-sonnet-20240229": {"input": 0.003, "output": 0.015},
            "claude-3-opus-20240229": {"input": 0.015, "output": 0.075}
        },
        "gemini": {
            "gemini-1.5-flash": {"input": 0.000075, "output": 0.0003},
            "gemini-1.5-pro": {"input": 0.0035, "output": 0.0105}
        },
        "grok": {
            "grok-beta": {"input": 0.002, "output": 0.002}  # Estimated
        }
    }
    
    @classmethod
    def estimate_cost(
        cls,
        service: str,
        model: str,
        task_type: str,
        input_tokens: int = 0,
        output_tokens: int = 0,
        **kwargs
    ) -> float:
        """
        Estimate cost for a request.
        
        Args:
            service: AI service name
            model: Model name
            task_type: Type of task
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            **kwargs: Additional parameters (e.g., image_count, audio_duration)
            
        Returns:
            Estimated cost in USD
        """
        if service not in cls.PRICING:
            return 0.0
        
        model_pricing = cls.PRICING[service].get(model, {})
        if not model_pricing:
            return 0.0
        
        cost = 0.0
        
        if task_type == "generate_text":
            # Text generation cost
            input_cost = (input_tokens / 1000) * model_pricing.get("input", 0)
            output_cost = (output_tokens / 1000) * model_pricing.get("output", 0)
            cost = input_cost + output_cost
            
        elif task_type == "generate_image":
            # Image generation cost
            image_count = kwargs.get("image_count", 1)
            cost = image_count * model_pricing.get("image", 0)
            
        elif task_type == "text_to_speech":
            # TTS cost (per 1K characters)
            character_count = kwargs.get("character_count", 0)
            cost = (character_count / 1000) * model_pricing.get("audio", 0)
            
        elif task_type == "speech_to_text":
            # STT cost (per minute)
            duration_minutes = kwargs.get("duration_minutes", 0)
            cost = duration_minutes * model_pricing.get("audio", 0)
        
        return cost
    
    @classmethod
    def get_model_pricing(cls, service: str, model: str) -> Dict[str, float]:
        """Get pricing information for a specific model."""
        return cls.PRICING.get(service, {}).get(model, {})


class CostTracker:
    """Track costs for AI API usage."""
    
    def __init__(self, storage_file: Optional[str] = None):
        """
        Initialize cost tracker.
        
        Args:
            storage_file: Optional file to persist cost data
        """
        self.storage_file = storage_file
        self.entries: List[CostEntry] = []
        self._totals: Dict[str, float] = defaultdict(float)
        self._lock = threading.Lock()
        
        # Load existing data if file exists
        if storage_file and Path(storage_file).exists():
            self.load_from_file()
    
    def record_cost(
        self,
        service: str,
        model: str,
        task_type: str,
        cost: float,
        tokens_used: Optional[int] = None,
        request_id: Optional[str] = None,
        **metadata
    ):
        """
        Record a cost entry.
        
        Args:
            service: AI service name
            model: Model name
            task_type: Type of task
            cost: Cost in USD
            tokens_used: Number of tokens used
            request_id: Optional request identifier
            **metadata: Additional metadata
        """
        with self._lock:
            entry = CostEntry(
                timestamp=time.time(),
                service=service,
                model=model,
                task_type=task_type,
                cost=cost,
                tokens_used=tokens_used,
                request_id=request_id,
                metadata=metadata
            )
            
            self.entries.append(entry)
            
            # Update totals
            service_key = f"{service}_{model}"
            self._totals[service_key] += cost
            self._totals["total"] += cost
            
            # Save to file if configured
            if self.storage_file:
                self._save_to_file()
    
    def estimate_and_record(
        self,
        service: str,
        model: str,
        task_type: str,
        input_tokens: int = 0,
        output_tokens: int = 0,
        request_id: Optional[str] = None,
        **kwargs
    ) -> float:
        """
        Estimate cost and record it.
        
        Returns:
            Estimated cost
        """
        cost = PricingModel.estimate_cost(
            service, model, task_type, input_tokens, output_tokens, **kwargs
        )
        
        total_tokens = input_tokens + output_tokens if input_tokens and output_tokens else None
        
        self.record_cost(
            service=service,
            model=model,
            task_type=task_type,
            cost=cost,
            tokens_used=total_tokens,
            request_id=request_id,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            **kwargs
        )
        
        return cost
    
    def get_total_cost(self, service: Optional[str] = None, model: Optional[str] = None) -> float:
        """
        Get total cost for specified service/model or overall.
        
        Args:
            service: Optional service filter
            model: Optional model filter
            
        Returns:
            Total cost in USD
        """
        if not service and not model:
            return self._totals.get("total", 0.0)
        
        if service and model:
            key = f"{service}_{model}"
            return self._totals.get(key, 0.0)
        
        # Filter by service only
        total = 0.0
        for entry in self.entries:
            if service and entry.service != service:
                continue
            if model and entry.model != model:
                continue
            total += entry.cost
        
        return total
    
    def get_cost_by_period(
        self,
        period: str = "day",
        start_time: Optional[float] = None,
        end_time: Optional[float] = None
    ) -> Dict[str, float]:
        """
        Get costs grouped by time period.
        
        Args:
            period: "hour", "day", "week", or "month"
            start_time: Optional start timestamp
            end_time: Optional end timestamp
            
        Returns:
            Dictionary with period keys and cost values
        """
        if start_time is None:
            start_time = time.time() - (30 * 24 * 3600)  # 30 days ago
        if end_time is None:
            end_time = time.time()
        
        costs_by_period = defaultdict(float)
        
        for entry in self.entries:
            if not (start_time <= entry.timestamp <= end_time):
                continue
            
            # Convert timestamp to period key
            dt = datetime.fromtimestamp(entry.timestamp)
            
            if period == "hour":
                key = dt.strftime("%Y-%m-%d %H:00")
            elif period == "day":
                key = dt.strftime("%Y-%m-%d")
            elif period == "week":
                week_start = dt - timedelta(days=dt.weekday())
                key = week_start.strftime("%Y-%m-%d")
            elif period == "month":
                key = dt.strftime("%Y-%m")
            else:
                key = dt.isoformat()
            
            costs_by_period[key] += entry.cost
        
        return dict(costs_by_period)
    
    def get_cost_breakdown(self) -> Dict[str, Any]:
        """Get detailed cost breakdown."""
        breakdown = {
            "total_cost": self.get_total_cost(),
            "by_service": defaultdict(float),
            "by_model": defaultdict(float),
            "by_task_type": defaultdict(float),
            "total_requests": len(self.entries),
            "total_tokens": 0
        }
        
        for entry in self.entries:
            breakdown["by_service"][entry.service] += entry.cost
            breakdown["by_model"][f"{entry.service}_{entry.model}"] += entry.cost
            breakdown["by_task_type"][entry.task_type] += entry.cost
            
            if entry.tokens_used:
                breakdown["total_tokens"] += entry.tokens_used
        
        # Convert defaultdicts to regular dicts
        breakdown["by_service"] = dict(breakdown["by_service"])
        breakdown["by_model"] = dict(breakdown["by_model"])
        breakdown["by_task_type"] = dict(breakdown["by_task_type"])
        
        return breakdown
    
    def get_top_consumers(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get top cost consumers."""
        model_costs = defaultdict(float)
        model_requests = defaultdict(int)
        
        for entry in self.entries:
            key = f"{entry.service}_{entry.model}"
            model_costs[key] += entry.cost
            model_requests[key] += 1
        
        # Sort by cost
        sorted_models = sorted(
            model_costs.items(),
            key=lambda x: x[1],
            reverse=True
        )[:limit]
        
        result = []
        for model_key, cost in sorted_models:
            service, model = model_key.split("_", 1)
            result.append({
                "service": service,
                "model": model,
                "total_cost": cost,
                "request_count": model_requests[model_key],
                "avg_cost_per_request": cost / model_requests[model_key] if model_requests[model_key] > 0 else 0
            })
        
        return result
    
    def set_budget_alert(self, budget: float, callback: callable):
        """Set up budget alert."""
        current_cost = self.get_total_cost()
        if current_cost >= budget:
            callback(current_cost, budget)
    
    def export_to_csv(self, file_path: str):
        """Export cost data to CSV."""
        import csv
        
        with open(file_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestamp", "datetime", "service", "model", "task_type",
                "cost", "tokens_used", "request_id"
            ])
            
            for entry in self.entries:
                dt = datetime.fromtimestamp(entry.timestamp)
                writer.writerow([
                    entry.timestamp,
                    dt.isoformat(),
                    entry.service,
                    entry.model,
                    entry.task_type,
                    entry.cost,
                    entry.tokens_used or "",
                    entry.request_id or ""
                ])
    
    def load_from_file(self):
        """Load cost data from file."""
        if not self.storage_file or not Path(self.storage_file).exists():
            return
        
        with open(self.storage_file, 'r') as f:
            data = json.load(f)
        
        self.entries = []
        self._totals = defaultdict(float)
        
        for entry_data in data.get("entries", []):
            entry = CostEntry(**entry_data)
            self.entries.append(entry)
            
            # Rebuild totals
            service_key = f"{entry.service}_{entry.model}"
            self._totals[service_key] += entry.cost
            self._totals["total"] += entry.cost
    
    def _save_to_file(self):
        """Save cost data to file."""
        if not self.storage_file:
            return
        
        data = {
            "entries": [
                {
                    "timestamp": entry.timestamp,
                    "service": entry.service,
                    "model": entry.model,
                    "task_type": entry.task_type,
                    "tokens_used": entry.tokens_used,
                    "cost": entry.cost,
                    "request_id": entry.request_id,
                    "metadata": entry.metadata
                }
                for entry in self.entries
            ],
            "totals": dict(self._totals)
        }
        
        # Ensure directory exists
        Path(self.storage_file).parent.mkdir(parents=True, exist_ok=True)
        
        with open(self.storage_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def clear(self):
        """Clear all cost data."""
        with self._lock:
            self.entries.clear()
            self._totals.clear()
            
            if self.storage_file:
                self._save_to_file()


# Global cost tracker instance
_cost_tracker: Optional[CostTracker] = None


def get_cost_tracker() -> CostTracker:
    """Get the global cost tracker instance."""
    global _cost_tracker
    if _cost_tracker is None:
        from .config import get_config
        config = get_config()
        
        storage_file = None
        if config.performance.enable_cost_tracking:
            storage_file = "easilyai_costs.json"
        
        _cost_tracker = CostTracker(storage_file)
    
    return _cost_tracker


def set_cost_tracker(tracker: CostTracker):
    """Set the global cost tracker instance."""
    global _cost_tracker
    _cost_tracker = tracker


def reset_cost_tracker():
    """Reset the global cost tracker instance."""
    global _cost_tracker
    if _cost_tracker:
        _cost_tracker.clear()
    _cost_tracker = None