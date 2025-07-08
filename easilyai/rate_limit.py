"""
Rate limiting for EasilyAI.

This module provides rate limiting functionality to ensure API usage stays within provider limits.
"""

import time
import threading
from collections import deque, defaultdict
from typing import Dict, Optional


class RateLimiter:
    """Token bucket rate limiter implementation."""
    
    def __init__(self, rate: int, period: float = 60.0):
        """
        Initialize rate limiter.
        
        Args:
            rate: Number of requests allowed per period
            period: Time period in seconds (default: 60 for per-minute)
        """
        self.rate = rate
        self.period = period
        self.allowance = rate
        self.last_check = time.time()
        self._lock = threading.Lock()
    
    def acquire(self, timeout: Optional[float] = None) -> bool:
        """
        Acquire permission to make a request.
        
        Args:
            timeout: Maximum time to wait for permission (None = wait forever)
            
        Returns:
            True if permission granted, False if timeout
        """
        start_time = time.time()
        
        while True:
            with self._lock:
                current = time.time()
                time_passed = current - self.last_check
                self.last_check = current
                
                # Replenish tokens based on time passed
                self.allowance += time_passed * (self.rate / self.period)
                if self.allowance > self.rate:
                    self.allowance = self.rate
                
                if self.allowance >= 1.0:
                    self.allowance -= 1.0
                    return True
            
            # Check timeout
            if timeout is not None:
                if time.time() - start_time >= timeout:
                    return False
            
            # Calculate wait time
            wait_time = (1.0 - self.allowance) * (self.period / self.rate)
            wait_time = min(wait_time, 0.1)  # Cap at 100ms to allow timeout checks
            time.sleep(wait_time)
    
    def try_acquire(self) -> bool:
        """Try to acquire permission without waiting."""
        return self.acquire(timeout=0)
    
    def get_wait_time(self) -> float:
        """Get estimated wait time until next request can be made."""
        with self._lock:
            if self.allowance >= 1.0:
                return 0.0
            return (1.0 - self.allowance) * (self.period / self.rate)


class SlidingWindowRateLimiter:
    """Sliding window rate limiter for more accurate rate limiting."""
    
    def __init__(self, rate: int, window: float = 60.0):
        """
        Initialize sliding window rate limiter.
        
        Args:
            rate: Number of requests allowed per window
            window: Window size in seconds
        """
        self.rate = rate
        self.window = window
        self.requests = deque()
        self._lock = threading.Lock()
    
    def acquire(self, timeout: Optional[float] = None) -> bool:
        """Acquire permission to make a request."""
        start_time = time.time()
        
        while True:
            with self._lock:
                current = time.time()
                
                # Remove old requests outside the window
                while self.requests and self.requests[0] <= current - self.window:
                    self.requests.popleft()
                
                # Check if we can make a request
                if len(self.requests) < self.rate:
                    self.requests.append(current)
                    return True
            
            # Check timeout
            if timeout is not None:
                if time.time() - start_time >= timeout:
                    return False
            
            # Wait a bit before retrying
            time.sleep(0.01)
    
    def try_acquire(self) -> bool:
        """Try to acquire permission without waiting."""
        return self.acquire(timeout=0)
    
    def get_current_rate(self) -> int:
        """Get current number of requests in the window."""
        with self._lock:
            current = time.time()
            # Clean old requests
            while self.requests and self.requests[0] <= current - self.window:
                self.requests.popleft()
            return len(self.requests)


class ServiceRateLimiter:
    """Rate limiter for multiple services with different limits."""
    
    def __init__(self):
        self.limiters: Dict[str, RateLimiter] = {}
        self._lock = threading.Lock()
    
    def set_limit(self, service: str, rate: int, period: float = 60.0):
        """Set rate limit for a service."""
        with self._lock:
            self.limiters[service] = RateLimiter(rate, period)
    
    def acquire(self, service: str, timeout: Optional[float] = None) -> bool:
        """Acquire permission for a specific service."""
        limiter = self.limiters.get(service)
        if limiter:
            return limiter.acquire(timeout)
        return True  # No limit set
    
    def try_acquire(self, service: str) -> bool:
        """Try to acquire permission without waiting."""
        limiter = self.limiters.get(service)
        if limiter:
            return limiter.try_acquire()
        return True
    
    def get_wait_time(self, service: str) -> float:
        """Get wait time for a specific service."""
        limiter = self.limiters.get(service)
        if limiter:
            return limiter.get_wait_time()
        return 0.0
    
    def configure_from_config(self, config):
        """Configure rate limits from EasilyAIConfig."""
        for service_name in ["openai", "anthropic", "gemini", "grok"]:
            service_config = getattr(config, service_name, None)
            if service_config and hasattr(service_config, 'rate_limit'):
                self.set_limit(service_name, service_config.rate_limit)


class AdaptiveRateLimiter:
    """Adaptive rate limiter that adjusts based on response times and errors."""
    
    def __init__(self, initial_rate: int, min_rate: int = 10, max_rate: int = 100):
        """
        Initialize adaptive rate limiter.
        
        Args:
            initial_rate: Starting rate limit
            min_rate: Minimum allowed rate
            max_rate: Maximum allowed rate
        """
        self.current_rate = initial_rate
        self.min_rate = min_rate
        self.max_rate = max_rate
        self.limiter = RateLimiter(initial_rate)
        
        # Tracking for adaptation
        self.success_count = 0
        self.error_count = 0
        self.response_times = deque(maxlen=100)
        self.last_adjustment = time.time()
        self.adjustment_interval = 60.0  # Adjust every minute
        
        self._lock = threading.Lock()
    
    def acquire(self, timeout: Optional[float] = None) -> bool:
        """Acquire permission with adaptive rate limiting."""
        self._maybe_adjust_rate()
        return self.limiter.acquire(timeout)
    
    def record_success(self, response_time: float):
        """Record a successful request."""
        with self._lock:
            self.success_count += 1
            self.response_times.append(response_time)
    
    def record_error(self, is_rate_limit_error: bool = False):
        """Record an error."""
        with self._lock:
            self.error_count += 1
            if is_rate_limit_error:
                # Immediately reduce rate on rate limit errors
                self._decrease_rate(factor=0.5)
    
    def _maybe_adjust_rate(self):
        """Adjust rate based on performance metrics."""
        with self._lock:
            current = time.time()
            if current - self.last_adjustment < self.adjustment_interval:
                return
            
            self.last_adjustment = current
            
            # Calculate metrics
            total_requests = self.success_count + self.error_count
            if total_requests == 0:
                return
            
            error_rate = self.error_count / total_requests
            avg_response_time = (sum(self.response_times) / len(self.response_times) 
                               if self.response_times else 0)
            
            # Adjust based on error rate
            if error_rate > 0.1:  # More than 10% errors
                self._decrease_rate(factor=0.8)
            elif error_rate < 0.01 and avg_response_time < 1.0:  # Good performance
                self._increase_rate(factor=1.1)
            
            # Reset counters
            self.success_count = 0
            self.error_count = 0
    
    def _increase_rate(self, factor: float = 1.1):
        """Increase the rate limit."""
        new_rate = min(int(self.current_rate * factor), self.max_rate)
        if new_rate > self.current_rate:
            self.current_rate = new_rate
            self.limiter = RateLimiter(new_rate)
    
    def _decrease_rate(self, factor: float = 0.9):
        """Decrease the rate limit."""
        new_rate = max(int(self.current_rate * factor), self.min_rate)
        if new_rate < self.current_rate:
            self.current_rate = new_rate
            self.limiter = RateLimiter(new_rate)
    
    def get_current_rate(self) -> int:
        """Get the current rate limit."""
        return self.current_rate


# Global rate limiter instance
_rate_limiter: Optional[ServiceRateLimiter] = None


def get_rate_limiter() -> ServiceRateLimiter:
    """Get the global rate limiter instance."""
    global _rate_limiter
    if _rate_limiter is None:
        _rate_limiter = ServiceRateLimiter()
        
        # Configure from config
        from .config import get_config
        config = get_config()
        _rate_limiter.configure_from_config(config)
    
    return _rate_limiter


def set_rate_limiter(limiter: ServiceRateLimiter):
    """Set the global rate limiter instance."""
    global _rate_limiter
    _rate_limiter = limiter


def reset_rate_limiter():
    """Reset the global rate limiter instance."""
    global _rate_limiter
    _rate_limiter = None