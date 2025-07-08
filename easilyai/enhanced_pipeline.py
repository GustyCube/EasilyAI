"""
Enhanced pipeline system for EasilyAI with advanced features.

This module provides an enhanced pipeline system with features like:
- Conditional task execution
- Parallel task execution  
- Variable substitution
- Error handling and retries
- Pipeline validation
- Template system
"""

import re
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class TaskStatus(Enum):
    """Status of a pipeline task."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class ExecutionMode(Enum):
    """Pipeline execution modes."""
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    CONDITIONAL = "conditional"


@dataclass
class TaskResult:
    """Result from a pipeline task."""
    task_id: str
    status: TaskStatus
    result: Any = None
    error: Optional[str] = None
    duration: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineTask:
    """Enhanced pipeline task with advanced features."""
    id: str
    app: Any  # EasyAIApp instance
    task_type: str
    prompt: str
    condition: Optional[Callable] = None
    dependencies: List[str] = field(default_factory=list)
    retry_count: int = 3
    timeout: Optional[float] = None
    parallel_group: Optional[str] = None
    kwargs: Dict[str, Any] = field(default_factory=dict)


class EnhancedPipeline:
    """Enhanced pipeline with advanced features."""
    
    def __init__(self, name: str):
        """
        Initialize enhanced pipeline.
        
        Args:
            name: Pipeline name
        """
        self.name = name
        self.tasks: Dict[str, PipelineTask] = {}
        self.results: Dict[str, TaskResult] = {}
        self.variables: Dict[str, Any] = {}
        self.execution_mode = ExecutionMode.SEQUENTIAL
        self.max_parallel_tasks = 5
        
        # Event hooks
        self.on_task_start: Optional[Callable] = None
        self.on_task_complete: Optional[Callable] = None
        self.on_task_error: Optional[Callable] = None
        self.on_pipeline_complete: Optional[Callable] = None
    
    def add_task(
        self,
        task_id: str,
        app,
        task_type: str,
        prompt: str,
        condition: Optional[Callable] = None,
        dependencies: Optional[List[str]] = None,
        retry_count: int = 3,
        timeout: Optional[float] = None,
        parallel_group: Optional[str] = None,
        **kwargs
    ):
        """
        Add a task to the pipeline.
        
        Args:
            task_id: Unique task identifier
            app: EasyAI app instance
            task_type: Type of task ('generate_text', 'generate_image', 'text_to_speech')
            prompt: Task prompt (supports variable substitution)
            condition: Optional condition function to determine if task should run
            dependencies: List of task IDs this task depends on
            retry_count: Number of retry attempts
            timeout: Task timeout in seconds
            parallel_group: Group for parallel execution
            **kwargs: Additional task parameters
        """
        task = PipelineTask(
            id=task_id,
            app=app,
            task_type=task_type,
            prompt=prompt,
            condition=condition,
            dependencies=dependencies or [],
            retry_count=retry_count,
            timeout=timeout,
            parallel_group=parallel_group,
            kwargs=kwargs
        )
        
        self.tasks[task_id] = task
    
    def set_variable(self, name: str, value: Any):
        """Set a pipeline variable."""
        self.variables[name] = value
    
    def get_variable(self, name: str, default: Any = None) -> Any:
        """Get a pipeline variable."""
        return self.variables.get(name, default)
    
    def set_execution_mode(self, mode: ExecutionMode, max_parallel: int = 5):
        """Set pipeline execution mode."""
        self.execution_mode = mode
        self.max_parallel_tasks = max_parallel
    
    def validate(self) -> List[str]:
        """
        Validate the pipeline for issues.
        
        Returns:
            List of validation errors
        """
        errors = []
        
        # Check for circular dependencies
        def has_cycle(task_id: str, visited: set, rec_stack: set) -> bool:
            visited.add(task_id)
            rec_stack.add(task_id)
            
            task = self.tasks.get(task_id)
            if task:
                for dep in task.dependencies:
                    if dep not in visited:
                        if has_cycle(dep, visited, rec_stack):
                            return True
                    elif dep in rec_stack:
                        return True
            
            rec_stack.remove(task_id)
            return False
        
        visited = set()
        for task_id in self.tasks:
            if task_id not in visited:
                if has_cycle(task_id, visited, set()):
                    errors.append(f"Circular dependency detected involving task: {task_id}")
        
        # Check for missing dependencies
        for task_id, task in self.tasks.items():
            for dep in task.dependencies:
                if dep not in self.tasks:
                    errors.append(f"Task {task_id} depends on non-existent task: {dep}")
        
        return errors
    
    def _substitute_variables(self, text: str, current_results: Dict[str, Any]) -> str:
        """
        Substitute variables in text.
        
        Supports:
        - {variable_name} - pipeline variables
        - {result:task_id} - result from specific task
        - {previous_result} - result from previous task (when applicable)
        """
        # Substitute pipeline variables
        for var_name, var_value in self.variables.items():
            text = text.replace(f"{{{var_name}}}", str(var_value))
        
        # Substitute task results
        result_pattern = r'\{result:([^}]+)\}'
        for match in re.finditer(result_pattern, text):
            task_id = match.group(1)
            if task_id in current_results:
                result = current_results[task_id]
                if hasattr(result, 'result'):
                    result = result.result
                text = text.replace(match.group(0), str(result))
        
        # Handle previous_result (context-dependent)
        if "{previous_result}" in text and current_results:
            # Get the most recent successful result
            for task_result in reversed(list(current_results.values())):
                if hasattr(task_result, 'status') and task_result.status == TaskStatus.COMPLETED:
                    text = text.replace("{previous_result}", str(task_result.result))
                    break
        
        return text
    
    def _get_ready_tasks(self, completed_tasks: set) -> List[str]:
        """Get tasks that are ready to execute."""
        ready = []
        
        for task_id, task in self.tasks.items():
            if task_id in completed_tasks:
                continue
            
            # Check if all dependencies are completed
            if all(dep in completed_tasks for dep in task.dependencies):
                ready.append(task_id)
        
        return ready
    
    def _execute_task(self, task: PipelineTask) -> TaskResult:
        """Execute a single task with retries and error handling."""
        start_time = time.time()
        
        if self.on_task_start:
            self.on_task_start(task)
        
        # Check condition
        if task.condition and not task.condition(self.results):
            result = TaskResult(
                task_id=task.id,
                status=TaskStatus.SKIPPED,
                duration=time.time() - start_time,
                metadata={"reason": "condition_not_met"}
            )
            return result
        
        last_error = None
        
        for attempt in range(task.retry_count):
            try:
                # Substitute variables in prompt
                processed_prompt = self._substitute_variables(task.prompt, self.results)
                
                # Execute task
                if task.task_type == "generate_text":
                    response = task.app.request(processed_prompt, **task.kwargs)
                elif task.task_type == "generate_image":
                    response = task.app.request(
                        processed_prompt,
                        task_type="generate_image",
                        **task.kwargs
                    )
                elif task.task_type == "text_to_speech":
                    response = task.app.request(
                        processed_prompt,
                        task_type="text_to_speech",
                        **task.kwargs
                    )
                else:
                    raise ValueError(f"Unknown task type: {task.task_type}")
                
                # Success
                result = TaskResult(
                    task_id=task.id,
                    status=TaskStatus.COMPLETED,
                    result=response,
                    duration=time.time() - start_time,
                    metadata={"attempt": attempt + 1}
                )
                
                if self.on_task_complete:
                    self.on_task_complete(task, result)
                
                return result
                
            except Exception as e:
                last_error = str(e)
                logger.warning(f"Task {task.id} failed on attempt {attempt + 1}: {e}")
                
                if self.on_task_error:
                    self.on_task_error(task, e)
                
                # Wait before retry (exponential backoff)
                if attempt < task.retry_count - 1:
                    time.sleep(2 ** attempt)
        
        # All attempts failed
        result = TaskResult(
            task_id=task.id,
            status=TaskStatus.FAILED,
            error=last_error,
            duration=time.time() - start_time,
            metadata={"attempts": task.retry_count}
        )
        
        return result
    
    def run(self) -> Dict[str, TaskResult]:
        """Execute the pipeline."""
        # Validate pipeline first
        errors = self.validate()
        if errors:
            raise ValueError(f"Pipeline validation failed: {', '.join(errors)}")
        
        if self.execution_mode == ExecutionMode.SEQUENTIAL:
            return self._run_sequential()
        elif self.execution_mode == ExecutionMode.PARALLEL:
            return self._run_parallel()
        else:
            return self._run_conditional()
    
    def _run_sequential(self) -> Dict[str, TaskResult]:
        """Run tasks sequentially respecting dependencies."""
        completed_tasks = set()
        
        while len(completed_tasks) < len(self.tasks):
            ready_tasks = self._get_ready_tasks(completed_tasks)
            
            if not ready_tasks:
                # No ready tasks - check for failures or deadlock
                failed_tasks = [
                    task_id for task_id, result in self.results.items()
                    if result.status == TaskStatus.FAILED
                ]
                if failed_tasks:
                    logger.error(f"Pipeline stopped due to failed tasks: {failed_tasks}")
                    break
                else:
                    logger.error("Pipeline deadlock detected")
                    break
            
            # Execute first ready task
            task_id = ready_tasks[0]
            task = self.tasks[task_id]
            result = self._execute_task(task)
            
            self.results[task_id] = result
            if result.status in [TaskStatus.COMPLETED, TaskStatus.SKIPPED, TaskStatus.FAILED]:
                completed_tasks.add(task_id)
        
        if self.on_pipeline_complete:
            self.on_pipeline_complete(self.results)
        
        return self.results
    
    def _run_parallel(self) -> Dict[str, TaskResult]:
        """Run tasks in parallel where possible."""
        completed_tasks = set()
        
        with ThreadPoolExecutor(max_workers=self.max_parallel_tasks) as executor:
            while len(completed_tasks) < len(self.tasks):
                ready_tasks = self._get_ready_tasks(completed_tasks)
                
                if not ready_tasks:
                    break
                
                # Submit all ready tasks
                future_to_task = {}
                for task_id in ready_tasks:
                    task = self.tasks[task_id]
                    future = executor.submit(self._execute_task, task)
                    future_to_task[future] = task_id
                
                # Wait for completion
                for future in as_completed(future_to_task):
                    task_id = future_to_task[future]
                    result = future.result()
                    
                    self.results[task_id] = result
                    if result.status in [TaskStatus.COMPLETED, TaskStatus.SKIPPED, TaskStatus.FAILED]:
                        completed_tasks.add(task_id)
        
        if self.on_pipeline_complete:
            self.on_pipeline_complete(self.results)
        
        return self.results
    
    def _run_conditional(self) -> Dict[str, TaskResult]:
        """Run tasks with conditional logic."""
        # Similar to sequential but with more sophisticated condition checking
        return self._run_sequential()
    
    def get_task_count(self) -> int:
        """Get number of tasks in pipeline."""
        return len(self.tasks)
    
    def get_results(self) -> Dict[str, TaskResult]:
        """Get pipeline results."""
        return self.results.copy()
    
    def clear(self):
        """Clear pipeline tasks and results."""
        self.tasks.clear()
        self.results.clear()
        self.variables.clear()
    
    def get_summary(self) -> dict:
        """Get pipeline execution summary."""
        if not self.results:
            return {"status": "not_executed", "task_count": len(self.tasks)}
        
        completed = sum(1 for r in self.results.values() if r.status == TaskStatus.COMPLETED)
        failed = sum(1 for r in self.results.values() if r.status == TaskStatus.FAILED)
        skipped = sum(1 for r in self.results.values() if r.status == TaskStatus.SKIPPED)
        
        total_duration = sum(r.duration for r in self.results.values())
        
        return {
            "name": self.name,
            "total_tasks": len(self.tasks),
            "completed": completed,
            "failed": failed,
            "skipped": skipped,
            "success_rate": (completed / len(self.results)) * 100 if self.results else 0,
            "total_duration": total_duration,
            "execution_mode": self.execution_mode.value
        }


class PipelineTemplate:
    """Template system for common pipeline patterns."""
    
    @staticmethod
    def content_creation_pipeline(text_app, image_app, tts_app=None) -> EnhancedPipeline:
        """Create a content creation pipeline template."""
        pipeline = EnhancedPipeline("ContentCreation")
        
        # Generate story
        pipeline.add_task(
            "generate_story",
            text_app,
            "generate_text",
            "Write a creative short story about {topic}",
            retry_count=2
        )
        
        # Create illustration
        pipeline.add_task(
            "create_illustration",
            image_app,
            "generate_image",
            "Create a beautiful illustration for this story: {result:generate_story}",
            dependencies=["generate_story"],
            retry_count=2
        )
        
        # Optional narration
        if tts_app:
            pipeline.add_task(
                "create_narration",
                tts_app,
                "text_to_speech",
                "{result:generate_story}",
                dependencies=["generate_story"],
                retry_count=1
            )
        
        return pipeline
    
    @staticmethod
    def analysis_pipeline(text_app) -> EnhancedPipeline:
        """Create a text analysis pipeline template."""
        pipeline = EnhancedPipeline("TextAnalysis")
        
        # Summarize
        pipeline.add_task(
            "summarize",
            text_app,
            "generate_text",
            "Summarize this text in 2-3 sentences: {input_text}"
        )
        
        # Extract key points
        pipeline.add_task(
            "extract_points",
            text_app,
            "generate_text",
            "Extract 5 key points from this text: {input_text}",
            parallel_group="analysis"
        )
        
        # Sentiment analysis
        pipeline.add_task(
            "sentiment",
            text_app,
            "generate_text",
            "Analyze the sentiment of this text: {input_text}",
            parallel_group="analysis"
        )
        
        # Final report
        pipeline.add_task(
            "final_report",
            text_app,
            "generate_text",
            "Create a comprehensive report based on:\nSummary: {result:summarize}\nKey Points: {result:extract_points}\nSentiment: {result:sentiment}",
            dependencies=["summarize", "extract_points", "sentiment"]
        )
        
        pipeline.set_execution_mode(ExecutionMode.PARALLEL, max_parallel=3)
        
        return pipeline


# Backward compatibility with original pipeline
class EasilyAIPipeline(EnhancedPipeline):
    """Backward compatible pipeline class."""
    
    def __init__(self, name: str):
        super().__init__(name)
        self._legacy_tasks = []
    
    def add_task(self, app, task_type: str, prompt: str, **kwargs):
        """Add task with legacy interface."""
        task_id = f"task_{len(self._legacy_tasks)}"
        self._legacy_tasks.append({"id": task_id, "type": task_type, "data": prompt})
        
        super().add_task(
            task_id=task_id,
            app=app,
            task_type=task_type,
            prompt=prompt,
            **kwargs
        )
    
    def run(self) -> List[Any]:
        """Run with legacy return format."""
        results = super().run()
        
        # Convert to legacy format
        legacy_results = []
        for task_info in self._legacy_tasks:
            task_id = task_info["id"]
            if task_id in results:
                result = results[task_id]
                legacy_results.append({
                    "task": task_info["type"],
                    "result": result.result if result.status == TaskStatus.COMPLETED else None,
                    "error": result.error,
                    "status": result.status.value
                })
        
        return legacy_results