"""
Logging configuration and utilities for EasilyAI.
"""
import logging
import sys
from typing import Optional, Dict, Any
from pathlib import Path


class ColoredFormatter(logging.Formatter):
    """Colored logging formatter for console output."""
    
    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[35m', # Magenta
    }
    RESET = '\033[0m'
    
    def format(self, record):
        log_color = self.COLORS.get(record.levelname, self.RESET)
        record.levelname = f"{log_color}{record.levelname}{self.RESET}"
        return super().format(record)


def setup_logging(
    level: str = "INFO",
    format_string: Optional[str] = None,
    log_file: Optional[Path] = None,
    colorize: bool = True,
    include_timestamp: bool = True,
) -> logging.Logger:
    """
    Set up logging configuration for EasilyAI.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format_string: Custom format string for log messages
        log_file: Optional file path to write logs to
        colorize: Whether to use colored output for console logging
        include_timestamp: Whether to include timestamp in log messages
    
    Returns:
        Configured logger instance
    """
    # Convert string level to logging constant
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    
    # Create logger
    logger = logging.getLogger("easilyai")
    logger.setLevel(numeric_level)
    
    # Clear any existing handlers
    logger.handlers.clear()
    
    # Default format
    if format_string is None:
        if include_timestamp:
            format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        else:
            format_string = "%(name)s - %(levelname)s - %(message)s"
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    
    if colorize and hasattr(sys.stdout, 'isatty') and sys.stdout.isatty():
        console_formatter = ColoredFormatter(format_string)
    else:
        console_formatter = logging.Formatter(format_string)
    
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(numeric_level)
        file_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s"
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger with the specified name under the easilyai namespace.
    
    Args:
        name: Logger name (will be prefixed with 'easilyai.')
    
    Returns:
        Logger instance
    """
    return logging.getLogger(f"easilyai.{name}")


def log_api_call(
    logger: logging.Logger,
    service: str,
    method: str,
    model: str,
    tokens_used: Optional[int] = None,
    cost: Optional[float] = None,
    duration: Optional[float] = None,
    **kwargs
) -> None:
    """
    Log API call information in a structured format.
    
    Args:
        logger: Logger instance to use
        service: AI service name (openai, anthropic, etc.)
        method: API method called (chat_complete, generate_text, etc.)
        model: Model used for the call
        tokens_used: Number of tokens consumed
        cost: Estimated cost of the call
        duration: Duration of the call in seconds
        **kwargs: Additional metadata to log
    """
    log_data = {
        "service": service,
        "method": method,
        "model": model,
    }
    
    if tokens_used is not None:
        log_data["tokens_used"] = tokens_used
    
    if cost is not None:
        log_data["cost"] = f"${cost:.4f}"
    
    if duration is not None:
        log_data["duration"] = f"{duration:.3f}s"
    
    # Add any additional metadata
    log_data.update(kwargs)
    
    # Format log message
    message_parts = [f"{k}={v}" for k, v in log_data.items()]
    message = f"API_CALL: {' '.join(message_parts)}"
    
    logger.info(message)


def log_error(
    logger: logging.Logger,
    error: Exception,
    service: str,
    method: str,
    context: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Log error information in a structured format.
    
    Args:
        logger: Logger instance to use
        error: Exception that occurred
        service: AI service name where error occurred
        method: Method where error occurred
        context: Additional context information
    """
    error_data = {
        "error_type": type(error).__name__,
        "error_message": str(error),
        "service": service,
        "method": method,
    }
    
    if context:
        error_data.update(context)
    
    message_parts = [f"{k}={v}" for k, v in error_data.items()]
    message = f"API_ERROR: {' '.join(message_parts)}"
    
    logger.error(message, exc_info=True)


class APICallLogger:
    """Context manager for logging API calls with timing."""
    
    def __init__(
        self,
        logger: logging.Logger,
        service: str,
        method: str,
        model: str,
        **kwargs
    ):
        self.logger = logger
        self.service = service
        self.method = method
        self.model = model
        self.context = kwargs
        self.start_time = None
    
    def __enter__(self):
        import time
        self.start_time = time.time()
        self.logger.debug(
            f"Starting API call: {self.service}.{self.method} with model {self.model}"
        )
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        import time
        duration = time.time() - self.start_time
        
        if exc_type is None:
            # Success
            log_api_call(
                self.logger,
                self.service,
                self.method,
                self.model,
                duration=duration,
                **self.context
            )
        else:
            # Error occurred
            log_error(
                self.logger,
                exc_val,
                self.service,
                self.method,
                context={"duration": f"{duration:.3f}s", **self.context}
            )


# Default logger instance
default_logger = get_logger("main")


def configure_service_logging(service_name: str, level: str = "INFO") -> logging.Logger:
    """
    Configure logging for a specific AI service.
    
    Args:
        service_name: Name of the AI service
        level: Logging level for the service
    
    Returns:
        Configured logger for the service
    """
    logger = get_logger(f"services.{service_name}")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    return logger


# Silence noisy third-party loggers by default
def silence_third_party_loggers():
    """Reduce noise from third-party libraries."""
    noisy_loggers = [
        "urllib3.connectionpool",
        "requests.packages.urllib3.connectionpool",
        "httpx",
        "httpcore",
        "openai._base_client",
        "anthropic._base_client",
    ]
    
    for logger_name in noisy_loggers:
        logging.getLogger(logger_name).setLevel(logging.WARNING)


# Initialize logging on import
silence_third_party_loggers()