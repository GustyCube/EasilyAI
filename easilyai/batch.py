"""
Batch processing for EasilyAI.

This module provides batch and parallel processing capabilities for handling multiple requests efficiently.
"""

import time
import asyncio
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed, Future
from typing import List, Dict, Any, Optional, Callable, Union, Tuple
from dataclasses import dataclass
from enum import Enum
import threading
from queue import Queue, Empty
import logging

logger = logging.getLogger(__name__)


class ProcessingMode(Enum):
    """Processing mode for batch operations."""
    SEQUENTIAL = "sequential"
    PARALLEL_THREADS = "parallel_threads"
    PARALLEL_PROCESSES = "parallel_processes"
    ASYNC = "async"


@dataclass
class BatchRequest:
    """Represents a single request in a batch."""
    id: str
    service: str
    model: str
    prompt: str
    task_type: str = "generate_text"
    kwargs: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.kwargs is None:
            self.kwargs = {}


@dataclass
class BatchResult:
    """Result from a batch request."""
    id: str
    success: bool
    response: Any = None
    error: Optional[str] = None
    duration: float = 0.0
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class BatchProcessor:
    """Process multiple AI requests in batches."""
    
    def __init__(
        self,
        mode: ProcessingMode = ProcessingMode.PARALLEL_THREADS,
        max_workers: int = 5,
        rate_limit_delay: float = 0.1,
        retry_attempts: int = 3,
        timeout: Optional[float] = 30.0
    ):
        """
        Initialize batch processor.
        
        Args:
            mode: Processing mode to use
            max_workers: Maximum number of parallel workers
            rate_limit_delay: Delay between requests (seconds)
            retry_attempts: Number of retry attempts for failed requests
            timeout: Timeout for individual requests
        """
        self.mode = mode
        self.max_workers = max_workers
        self.rate_limit_delay = rate_limit_delay
        self.retry_attempts = retry_attempts
        self.timeout = timeout
        
        # Statistics
        self.stats = {
            "total_processed": 0,
            "successful": 0,
            "failed": 0,
            "total_duration": 0.0
        }
    
    def process(
        self,
        requests: List[BatchRequest],
        callback: Optional[Callable[[BatchResult], None]] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> List[BatchResult]:
        """
        Process a batch of requests.
        
        Args:
            requests: List of batch requests
            callback: Optional callback for each completed request
            progress_callback: Optional callback for progress updates (completed, total)
            
        Returns:
            List of batch results in the same order as requests
        """
        start_time = time.time()
        
        if self.mode == ProcessingMode.SEQUENTIAL:
            results = self._process_sequential(requests, callback, progress_callback)
        elif self.mode == ProcessingMode.PARALLEL_THREADS:
            results = self._process_parallel_threads(requests, callback, progress_callback)
        elif self.mode == ProcessingMode.PARALLEL_PROCESSES:
            results = self._process_parallel_processes(requests, callback, progress_callback)
        elif self.mode == ProcessingMode.ASYNC:
            results = asyncio.run(self._process_async(requests, callback, progress_callback))
        else:
            raise ValueError(f"Unknown processing mode: {self.mode}")
        
        # Update statistics
        self.stats["total_processed"] += len(requests)
        self.stats["successful"] += sum(1 for r in results if r.success)
        self.stats["failed"] += sum(1 for r in results if not r.success)
        self.stats["total_duration"] += time.time() - start_time
        
        return results
    
    def _process_sequential(
        self,
        requests: List[BatchRequest],
        callback: Optional[Callable],
        progress_callback: Optional[Callable]
    ) -> List[BatchResult]:
        """Process requests sequentially."""
        results = []
        
        for i, request in enumerate(requests):
            result = self._process_single_request(request)
            results.append(result)
            
            if callback:
                callback(result)
            if progress_callback:
                progress_callback(i + 1, len(requests))
            
            # Rate limiting delay
            if i < len(requests) - 1:
                time.sleep(self.rate_limit_delay)
        
        return results
    
    def _process_parallel_threads(
        self,
        requests: List[BatchRequest],
        callback: Optional[Callable],
        progress_callback: Optional[Callable]
    ) -> List[BatchResult]:
        """Process requests in parallel using threads."""
        results = [None] * len(requests)
        completed = 0
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks with delays
            futures: Dict[Future, int] = {}
            
            for i, request in enumerate(requests):
                # Add delay between submissions for rate limiting
                if i > 0:
                    time.sleep(self.rate_limit_delay)
                
                future = executor.submit(self._process_single_request, request)
                futures[future] = i
            
            # Process completed tasks
            for future in as_completed(futures):
                index = futures[future]
                result = future.result()
                results[index] = result
                completed += 1
                
                if callback:
                    callback(result)
                if progress_callback:
                    progress_callback(completed, len(requests))
        
        return results
    
    def _process_parallel_processes(
        self,
        requests: List[BatchRequest],
        callback: Optional[Callable],
        progress_callback: Optional[Callable]
    ) -> List[BatchResult]:
        """Process requests in parallel using processes."""
        # Note: This requires requests to be pickleable
        results = [None] * len(requests)
        completed = 0
        
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            futures: Dict[Future, int] = {}
            
            for i, request in enumerate(requests):
                if i > 0:
                    time.sleep(self.rate_limit_delay)
                
                future = executor.submit(self._process_single_request, request)
                futures[future] = i
            
            for future in as_completed(futures):
                index = futures[future]
                result = future.result()
                results[index] = result
                completed += 1
                
                if callback:
                    callback(result)
                if progress_callback:
                    progress_callback(completed, len(requests))
        
        return results
    
    async def _process_async(
        self,
        requests: List[BatchRequest],
        callback: Optional[Callable],
        progress_callback: Optional[Callable]
    ) -> List[BatchResult]:
        """Process requests asynchronously."""
        semaphore = asyncio.Semaphore(self.max_workers)
        results = [None] * len(requests)
        completed = 0
        completed_lock = asyncio.Lock()
        
        async def process_with_semaphore(request: BatchRequest, index: int):
            async with semaphore:
                # Add delay for rate limiting
                if index > 0:
                    await asyncio.sleep(self.rate_limit_delay)
                
                # Process in thread pool to avoid blocking
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    None, self._process_single_request, request
                )
                
                results[index] = result
                
                async with completed_lock:
                    nonlocal completed
                    completed += 1
                    
                    if callback:
                        callback(result)
                    if progress_callback:
                        progress_callback(completed, len(requests))
                
                return result
        
        # Create tasks for all requests
        tasks = [
            process_with_semaphore(request, i)
            for i, request in enumerate(requests)
        ]
        
        # Wait for all tasks to complete
        await asyncio.gather(*tasks)
        
        return results
    
    def _process_single_request(self, request: BatchRequest) -> BatchResult:
        """Process a single request with retries and error handling."""
        start_time = time.time()
        last_error = None
        
        for attempt in range(self.retry_attempts):
            try:
                # Import here to avoid circular imports
                from .app import create_app
                from .cache import get_cache
                from .rate_limit import get_rate_limiter
                
                # Check cache first
                cache = get_cache()
                cached_response = cache.get(
                    request.service,
                    request.model,
                    request.prompt,
                    task_type=request.task_type,
                    **request.kwargs
                )
                
                if cached_response is not None:
                    return BatchResult(
                        id=request.id,
                        success=True,
                        response=cached_response,
                        duration=time.time() - start_time,
                        metadata={"cached": True, "attempt": attempt + 1}
                    )
                
                # Apply rate limiting
                rate_limiter = get_rate_limiter()
                rate_limiter.acquire(request.service, timeout=self.timeout)
                
                # Get API key from config
                from .config import get_config
                config = get_config()
                api_key = config.get_api_key(request.service)
                
                if not api_key:
                    raise ValueError(f"No API key configured for {request.service}")
                
                # Create app and make request
                app = create_app(
                    f"BatchApp_{request.id}",
                    request.service,
                    api_key,
                    request.model
                )
                
                response = app.request(
                    request.prompt,
                    task_type=request.task_type,
                    **request.kwargs
                )
                
                # Cache successful response
                cache.set(
                    request.service,
                    request.model,
                    request.prompt,
                    response,
                    task_type=request.task_type,
                    **request.kwargs
                )
                
                return BatchResult(
                    id=request.id,
                    success=True,
                    response=response,
                    duration=time.time() - start_time,
                    metadata={"attempt": attempt + 1}
                )
                
            except Exception as e:
                last_error = str(e)
                logger.warning(f"Request {request.id} failed on attempt {attempt + 1}: {e}")
                
                # Don't retry on certain errors
                if "API key" in str(e) or "Invalid" in str(e):
                    break
                
                # Wait before retry
                if attempt < self.retry_attempts - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
        
        # All attempts failed
        return BatchResult(
            id=request.id,
            success=False,
            error=last_error,
            duration=time.time() - start_time,
            metadata={"attempts": self.retry_attempts}
        )
    
    def get_stats(self) -> dict:
        """Get processing statistics."""
        if self.stats["total_processed"] > 0:
            success_rate = (self.stats["successful"] / self.stats["total_processed"]) * 100
            avg_duration = self.stats["total_duration"] / self.stats["total_processed"]
        else:
            success_rate = 0.0
            avg_duration = 0.0
        
        return {
            "total_processed": self.stats["total_processed"],
            "successful": self.stats["successful"],
            "failed": self.stats["failed"],
            "success_rate": success_rate,
            "total_duration": self.stats["total_duration"],
            "average_duration": avg_duration,
            "mode": self.mode.value,
            "max_workers": self.max_workers
        }
    
    def reset_stats(self):
        """Reset processing statistics."""
        self.stats = {
            "total_processed": 0,
            "successful": 0,
            "failed": 0,
            "total_duration": 0.0
        }


class StreamingBatchProcessor:
    """Process requests in a streaming fashion as they arrive."""
    
    def __init__(
        self,
        processor: Optional[BatchProcessor] = None,
        buffer_size: int = 10,
        flush_interval: float = 1.0
    ):
        """
        Initialize streaming batch processor.
        
        Args:
            processor: Underlying batch processor
            buffer_size: Number of requests to buffer before processing
            flush_interval: Time to wait before processing partial batch
        """
        self.processor = processor or BatchProcessor()
        self.buffer_size = buffer_size
        self.flush_interval = flush_interval
        
        self._buffer: List[Tuple[BatchRequest, Future]] = []
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._flush_thread = threading.Thread(target=self._flush_worker)
        self._flush_thread.daemon = True
        self._flush_thread.start()
    
    def submit(self, request: BatchRequest) -> Future:
        """
        Submit a request for processing.
        
        Returns:
            Future that will contain the BatchResult
        """
        future = Future()
        
        with self._lock:
            self._buffer.append((request, future))
            
            # Process immediately if buffer is full
            if len(self._buffer) >= self.buffer_size:
                self._process_buffer()
        
        return future
    
    def _flush_worker(self):
        """Background thread that flushes partial batches."""
        while not self._stop_event.is_set():
            time.sleep(self.flush_interval)
            with self._lock:
                if self._buffer:
                    self._process_buffer()
    
    def _process_buffer(self):
        """Process current buffer (must be called with lock held)."""
        if not self._buffer:
            return
        
        # Extract requests and futures
        batch = self._buffer.copy()
        self._buffer.clear()
        
        requests = [req for req, _ in batch]
        futures_map = {req.id: future for req, future in batch}
        
        # Process batch
        def complete_future(result: BatchResult):
            future = futures_map.get(result.id)
            if future:
                future.set_result(result)
        
        # Run processing in separate thread to avoid blocking
        thread = threading.Thread(
            target=lambda: self.processor.process(requests, callback=complete_future)
        )
        thread.start()
    
    def flush(self):
        """Force processing of any buffered requests."""
        with self._lock:
            self._process_buffer()
    
    def stop(self):
        """Stop the streaming processor."""
        self._stop_event.set()
        self.flush()
        self._flush_thread.join(timeout=5.0)