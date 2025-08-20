"""
Retry logic and rate limiting utilities for API calls.
"""
import time
import random
from functools import wraps
from typing import Callable, Type, Tuple, Optional, Any
import logging

logger = logging.getLogger(__name__)


class RateLimitError(Exception):
    """Raised when rate limit is exceeded."""
    pass


class RetryableError(Exception):
    """Base class for errors that should trigger a retry."""
    pass


def exponential_backoff(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    jitter: bool = True,
    retryable_exceptions: Tuple[Type[Exception], ...] = (Exception,),
) -> Callable:
    """
    Decorator for implementing exponential backoff retry logic.
    
    Args:
        max_retries: Maximum number of retry attempts
        base_delay: Initial delay in seconds
        max_delay: Maximum delay in seconds
        exponential_base: Base for exponential backoff calculation
        jitter: Whether to add random jitter to delays
        retryable_exceptions: Tuple of exception types that should trigger retries
    
    Returns:
        Decorated function with retry logic
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except retryable_exceptions as e:
                    last_exception = e
                    
                    if attempt == max_retries:
                        logger.error(
                            f"Function {func.__name__} failed after {max_retries} retries. "
                            f"Last error: {str(e)}"
                        )
                        raise e
                    
                    # Calculate delay with exponential backoff
                    delay = min(
                        base_delay * (exponential_base ** attempt),
                        max_delay
                    )
                    
                    # Add jitter to prevent thundering herd
                    if jitter:
                        delay = delay * (0.5 + random.random() * 0.5)
                    
                    logger.warning(
                        f"Function {func.__name__} failed on attempt {attempt + 1}. "
                        f"Retrying in {delay:.2f} seconds. Error: {str(e)}"
                    )
                    
                    time.sleep(delay)
            
            # This should never be reached due to the raise in the except block
            if last_exception:
                raise last_exception
                
        return wrapper
    return decorator


class RateLimiter:
    """
    Token bucket rate limiter implementation.
    """
    
    def __init__(self, rate: float, burst: int = 1):
        """
        Initialize rate limiter.
        
        Args:
            rate: Requests per second allowed
            burst: Maximum burst size (token bucket capacity)
        """
        self.rate = rate
        self.burst = burst
        self.tokens = burst
        self.last_update = time.time()
    
    def acquire(self, tokens: int = 1) -> bool:
        """
        Acquire tokens from the bucket.
        
        Args:
            tokens: Number of tokens to acquire
            
        Returns:
            True if tokens were acquired, False otherwise
        """
        now = time.time()
        
        # Add tokens based on elapsed time
        elapsed = now - self.last_update
        self.tokens = min(self.burst, self.tokens + elapsed * self.rate)
        self.last_update = now
        
        if self.tokens >= tokens:
            self.tokens -= tokens
            return True
        
        return False
    
    def wait_time(self, tokens: int = 1) -> float:
        """
        Calculate how long to wait for tokens to be available.
        
        Args:
            tokens: Number of tokens needed
            
        Returns:
            Wait time in seconds
        """
        if self.tokens >= tokens:
            return 0.0
        
        needed_tokens = tokens - self.tokens
        return needed_tokens / self.rate


def rate_limited(rate: float, burst: int = 1) -> Callable:
    """
    Decorator for rate limiting function calls.
    
    Args:
        rate: Requests per second allowed
        burst: Maximum burst size
    
    Returns:
        Decorated function with rate limiting
    """
    limiter = RateLimiter(rate, burst)
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            if not limiter.acquire():
                wait_time = limiter.wait_time()
                logger.info(
                    f"Rate limit reached for {func.__name__}. "
                    f"Waiting {wait_time:.2f} seconds."
                )
                time.sleep(wait_time)
                limiter.acquire()  # Should succeed after waiting
            
            return func(*args, **kwargs)
        return wrapper
    return decorator


def retry_with_rate_limit(
    max_retries: int = 3,
    base_delay: float = 1.0,
    rate: float = 1.0,
    burst: int = 1,
    retryable_exceptions: Tuple[Type[Exception], ...] = (Exception,),
) -> Callable:
    """
    Combined decorator for retry logic with rate limiting.
    
    Args:
        max_retries: Maximum number of retry attempts
        base_delay: Initial delay in seconds for retries
        rate: Requests per second for rate limiting
        burst: Maximum burst size for rate limiting
        retryable_exceptions: Exception types that should trigger retries
    
    Returns:
        Decorated function with both retry and rate limiting
    """
    def decorator(func: Callable) -> Callable:
        # Apply rate limiting first, then retry logic
        rate_limited_func = rate_limited(rate, burst)(func)
        retry_func = exponential_backoff(
            max_retries=max_retries,
            base_delay=base_delay,
            retryable_exceptions=retryable_exceptions,
        )(rate_limited_func)
        return retry_func
    return decorator


# Common retry configurations for different services
OPENAI_RETRY_CONFIG = {
    "max_retries": 3,
    "base_delay": 1.0,
    "retryable_exceptions": (Exception,),  # Will be replaced with actual OpenAI exceptions
}

ANTHROPIC_RETRY_CONFIG = {
    "max_retries": 3,
    "base_delay": 2.0,
    "retryable_exceptions": (Exception,),  # Will be replaced with actual Anthropic exceptions
}

GEMINI_RETRY_CONFIG = {
    "max_retries": 2,
    "base_delay": 1.5,
    "retryable_exceptions": (Exception,),  # Will be replaced with actual Gemini exceptions
}

GROQ_RETRY_CONFIG = {
    "max_retries": 3,
    "base_delay": 0.5,
    "retryable_exceptions": (Exception,),  # Will be replaced with actual Groq exceptions
}


def create_service_retry_decorator(service_name: str) -> Callable:
    """
    Create a retry decorator configured for a specific service.
    
    Args:
        service_name: Name of the AI service (openai, anthropic, gemini, groq)
    
    Returns:
        Configured retry decorator
    """
    configs = {
        "openai": OPENAI_RETRY_CONFIG,
        "anthropic": ANTHROPIC_RETRY_CONFIG,
        "gemini": GEMINI_RETRY_CONFIG,
        "groq": GROQ_RETRY_CONFIG,
    }
    
    config = configs.get(service_name.lower(), OPENAI_RETRY_CONFIG)
    return exponential_backoff(**config)