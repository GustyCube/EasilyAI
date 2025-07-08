# Pipeline System API Reference

The EasilyAI pipeline system allows you to chain multiple AI operations together, creating complex workflows that combine text generation, image creation, and text-to-speech.

## Core Pipeline Classes

### EasilyAIPipeline

Main pipeline class for orchestrating multiple AI tasks.

```python
class EasilyAIPipeline:
    def __init__(name: str)
```

**Parameters:**
- `name` (str): A unique identifier for the pipeline

**Attributes:**
- `name` (str): Pipeline name
- `tasks` (list): List of pipeline tasks
- `results` (list): Results from executed tasks

## Methods

### add_task()

Add a task to the pipeline.

```python
def add_task(
    app: EasyAIApp,
    task_type: str,
    prompt: str,
    **kwargs
) -> None
```

**Parameters:**
- `app` (EasyAIApp): The AI application to use for this task
- `task_type` (str): Type of task ('generate_text', 'generate_image', 'text_to_speech')
- `prompt` (str): The prompt for this task (supports variable substitution)
- `**kwargs`: Additional parameters for the specific task

**Variable Substitution:**
- `{previous_result}`: Result from the immediately previous task
- `{result_0}`, `{result_1}`, etc.: Results from specific task indices

**Example:**
```python
from easilyai import create_app
from easilyai.pipeline import EasilyAIPipeline

# Create apps
text_app = create_app("Writer", "openai", "your-key", "gpt-4")
image_app = create_app("Artist", "openai", "your-key", "dall-e-3")

# Create pipeline
pipeline = EasilyAIPipeline("ContentCreator")

# Add tasks with variable substitution
pipeline.add_task(text_app, "generate_text", "Write a story about dragons")
pipeline.add_task(image_app, "generate_image", "Create an illustration for: {previous_result}")
```

### run()

Execute all tasks in the pipeline sequentially.

```python
def run() -> list
```

**Returns:**
- `list`: Results from all tasks in order of execution

**Example:**
```python
# Execute pipeline
results = pipeline.run()

print("Story:", results[0])
print("Image URL:", results[1])
```

### clear()

Clear all tasks and results from the pipeline.

```python
def clear() -> None
```

**Example:**
```python
pipeline.clear()
print(f"Tasks: {len(pipeline.tasks)}")  # Output: Tasks: 0
```

### get_task_count()

Get the number of tasks in the pipeline.

```python
def get_task_count() -> int
```

**Returns:**
- `int`: Number of tasks in the pipeline

### get_results()

Get results from executed tasks.

```python
def get_results() -> list
```

**Returns:**
- `list`: Results from all executed tasks

## Pipeline Task Class

### PipelineTask

Internal class representing a single task in the pipeline.

```python
class PipelineTask:
    def __init__(
        app: EasyAIApp,
        task_type: str,
        prompt: str,
        **kwargs
    )
```

**Attributes:**
- `app` (EasyAIApp): The AI application for this task
- `task_type` (str): Type of task
- `prompt` (str): Task prompt
- `kwargs` (dict): Additional parameters

## Advanced Pipeline Features

### Conditional Tasks

Create pipelines with conditional execution:

```python
from easilyai import create_app
from easilyai.pipeline import EasilyAIPipeline

class ConditionalPipeline(EasilyAIPipeline):
    def __init__(self, name: str):
        super().__init__(name)
        self.conditions = []
    
    def add_conditional_task(self, app, task_type, prompt, condition_func, **kwargs):
        """Add a task that only executes if condition is met"""
        task = {
            "app": app,
            "task_type": task_type,
            "prompt": prompt,
            "condition": condition_func,
            "kwargs": kwargs
        }
        self.tasks.append(task)
    
    def run(self):
        """Execute tasks with conditional logic"""
        self.results = []
        
        for i, task in enumerate(self.tasks):
            # Check if this is a conditional task
            if "condition" in task:
                # Evaluate condition based on previous results
                if not task["condition"](self.results):
                    self.results.append(None)  # Skip task
                    continue
            
            # Substitute variables in prompt
            prompt = self._substitute_variables(task["prompt"], i)
            
            # Execute task
            try:
                if hasattr(task, "app"):  # Regular task
                    result = task["app"].request(prompt, task_type=task["task_type"], **task.get("kwargs", {}))
                else:  # Conditional task
                    result = task["app"].request(prompt, task_type=task["task_type"], **task.get("kwargs", {}))
                
                self.results.append(result)
            except Exception as e:
                self.results.append(f"Error: {e}")
        
        return self.results

# Example usage
def should_create_image(results):
    """Only create image if text is longer than 100 characters"""
    if not results:
        return False
    return len(results[-1]) > 100

conditional_pipeline = ConditionalPipeline("ConditionalContent")
conditional_pipeline.add_task(text_app, "generate_text", "Write about AI")
conditional_pipeline.add_conditional_task(
    image_app, 
    "generate_image", 
    "Create image for: {previous_result}",
    should_create_image
)
```

### Parallel Execution

Execute multiple tasks in parallel:

```python
import asyncio
from concurrent.futures import ThreadPoolExecutor

class ParallelPipeline(EasilyAIPipeline):
    def __init__(self, name: str, max_workers: int = 3):
        super().__init__(name)
        self.max_workers = max_workers
    
    def add_parallel_group(self, tasks: list):
        """Add a group of tasks to execute in parallel"""
        self.tasks.append({"type": "parallel", "tasks": tasks})
    
    def run(self):
        """Execute pipeline with parallel task groups"""
        self.results = []
        
        for task_group in self.tasks:
            if isinstance(task_group, dict) and task_group.get("type") == "parallel":
                # Execute parallel group
                parallel_results = self._execute_parallel_group(task_group["tasks"])
                self.results.extend(parallel_results)
            else:
                # Execute single task
                prompt = self._substitute_variables(task_group.prompt, len(self.results))
                result = task_group.app.request(
                    prompt, 
                    task_type=task_group.task_type, 
                    **task_group.kwargs
                )
                self.results.append(result)
        
        return self.results
    
    def _execute_parallel_group(self, tasks):
        """Execute a group of tasks in parallel"""
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = []
            
            for task in tasks:
                prompt = self._substitute_variables(task.prompt, len(self.results))
                future = executor.submit(
                    task.app.request,
                    prompt,
                    task_type=task.task_type,
                    **task.kwargs
                )
                futures.append(future)
            
            # Collect results
            results = []
            for future in futures:
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    results.append(f"Error: {e}")
            
            return results

# Example usage
parallel_pipeline = ParallelPipeline("ParallelContent")

# Add parallel tasks
parallel_tasks = [
    PipelineTask(text_app, "generate_text", "Write about AI"),
    PipelineTask(text_app, "generate_text", "Write about machine learning"),
    PipelineTask(text_app, "generate_text", "Write about data science")
]

parallel_pipeline.add_parallel_group(parallel_tasks)
```

### Error Handling in Pipelines

Robust error handling for pipeline execution:

```python
class RobustPipeline(EasilyAIPipeline):
    def __init__(self, name: str, fail_fast: bool = False):
        super().__init__(name)
        self.fail_fast = fail_fast
        self.errors = []
    
    def run(self):
        """Execute pipeline with robust error handling"""
        self.results = []
        self.errors = []
        
        for i, task in enumerate(self.tasks):
            try:
                # Substitute variables
                prompt = self._substitute_variables(task.prompt, i)
                
                # Execute task with retry logic
                result = self._execute_with_retry(task, prompt)
                self.results.append(result)
                
            except Exception as e:
                error_info = {
                    "task_index": i,
                    "task_type": task.task_type,
                    "error": str(e)
                }
                self.errors.append(error_info)
                
                if self.fail_fast:
                    raise e
                else:
                    # Continue with placeholder result
                    self.results.append(f"[Error in task {i}: {e}]")
        
        return self.results
    
    def _execute_with_retry(self, task, prompt, max_retries=3):
        """Execute task with retry logic"""
        for attempt in range(max_retries):
            try:
                return task.app.request(
                    prompt,
                    task_type=task.task_type,
                    **task.kwargs
                )
            except Exception as e:
                if attempt == max_retries - 1:
                    raise e
                else:
                    # Wait before retry
                    import time
                    time.sleep(2 ** attempt)
    
    def get_error_summary(self):
        """Get summary of errors encountered"""
        return {
            "total_errors": len(self.errors),
            "errors": self.errors,
            "success_rate": (len(self.tasks) - len(self.errors)) / len(self.tasks) * 100
        }

# Example usage
robust_pipeline = RobustPipeline("RobustContent", fail_fast=False)
robust_pipeline.add_task(text_app, "generate_text", "Write about AI")
robust_pipeline.add_task(image_app, "generate_image", "Invalid prompt that might fail")

results = robust_pipeline.run()
error_summary = robust_pipeline.get_error_summary()

print(f"Success rate: {error_summary['success_rate']:.1f}%")
print(f"Errors: {error_summary['total_errors']}")
```

## Pipeline Templates

### Content Creation Pipeline

```python
def create_content_pipeline(text_app, image_app, tts_app, topic):
    """Create a complete content creation pipeline"""
    pipeline = EasilyAIPipeline("ContentCreation")
    
    # Generate article
    pipeline.add_task(
        text_app,
        "generate_text",
        f"Write a comprehensive 500-word article about {topic}"
    )
    
    # Create summary
    pipeline.add_task(
        text_app,
        "generate_text",
        "Create a 2-sentence summary of this article: {previous_result}"
    )
    
    # Generate image
    pipeline.add_task(
        image_app,
        "generate_image",
        "Create a professional illustration for this article: {result_1}"
    )
    
    # Create audio narration
    pipeline.add_task(
        tts_app,
        "text_to_speech",
        "{result_0}",  # Use the full article
        voice="nova",
        output_file=f"{topic.replace(' ', '_')}_narration.mp3"
    )
    
    return pipeline

# Usage
content_pipeline = create_content_pipeline(text_app, image_app, tts_app, "renewable energy")
results = content_pipeline.run()
```

### Analysis Pipeline

```python
def create_analysis_pipeline(analyst_app, visualizer_app, data):
    """Create a data analysis pipeline"""
    pipeline = EasilyAIPipeline("DataAnalysis")
    
    # Initial analysis
    pipeline.add_task(
        analyst_app,
        "generate_text",
        f"Analyze this data and identify key trends: {data}"
    )
    
    # Generate insights
    pipeline.add_task(
        analyst_app,
        "generate_text",
        "Based on this analysis, provide 3 actionable business insights: {previous_result}"
    )
    
    # Create visualization prompt
    pipeline.add_task(
        analyst_app,
        "generate_text",
        "Create a detailed prompt for visualizing this analysis: {result_0}"
    )
    
    # Generate visualization
    pipeline.add_task(
        visualizer_app,
        "generate_image",
        "{previous_result}"
    )
    
    return pipeline
```

### Multi-Language Pipeline

```python
def create_translation_pipeline(translator_app, languages, source_text):
    """Create a multi-language translation pipeline"""
    pipeline = EasilyAIPipeline("MultiLanguageTranslation")
    
    # Add source text as first result
    pipeline.results = [source_text]
    
    # Add translation tasks for each language
    for lang in languages:
        pipeline.add_task(
            translator_app,
            "generate_text",
            f"Translate this text to {lang}: {source_text}"
        )
    
    return pipeline
```

## Pipeline Utilities

### Variable Substitution

Internal method for replacing variables in prompts:

```python
def _substitute_variables(self, prompt: str, current_index: int) -> str
```

**Supported Variables:**
- `{previous_result}`: Result from previous task
- `{result_N}`: Result from task at index N
- `{current_index}`: Current task index
- `{task_count}`: Total number of tasks

### Pipeline Validation

Validate pipeline before execution:

```python
def validate_pipeline(pipeline: EasilyAIPipeline) -> dict:
    """Validate pipeline configuration"""
    issues = []
    
    if not pipeline.tasks:
        issues.append("Pipeline has no tasks")
    
    for i, task in enumerate(pipeline.tasks):
        # Check for circular references
        if "{result_" + str(i) + "}" in task.prompt:
            issues.append(f"Task {i} references its own result")
        
        # Check for forward references
        import re
        forward_refs = re.findall(r'\{result_(\d+)\}', task.prompt)
        for ref in forward_refs:
            if int(ref) >= i:
                issues.append(f"Task {i} references future result {ref}")
    
    return {
        "valid": len(issues) == 0,
        "issues": issues
    }

# Usage
validation = validate_pipeline(pipeline)
if not validation["valid"]:
    print("Pipeline issues:")
    for issue in validation["issues"]:
        print(f"  - {issue}")
```

## Best Practices

### 1. Design for Failure

```python
# Use robust pipelines for production
robust_pipeline = RobustPipeline("Production", fail_fast=False)

# Always check for errors
results = robust_pipeline.run()
if robust_pipeline.errors:
    print("Pipeline completed with errors")
```

### 2. Optimize Task Order

```python
# Place fast tasks first, expensive tasks last
pipeline.add_task(fast_app, "generate_text", "Quick summary")
pipeline.add_task(expensive_app, "generate_image", "Detailed illustration")
```

### 3. Use Meaningful Variable Names

```python
# Good: Clear variable usage
pipeline.add_task(image_app, "generate_image", "Illustration for story: {result_0}")

# Avoid: Unclear references
pipeline.add_task(image_app, "generate_image", "Make image: {previous_result}")
```

### 4. Validate Inputs

```python
def safe_pipeline_run(pipeline):
    """Safely run a pipeline with validation"""
    validation = validate_pipeline(pipeline)
    
    if not validation["valid"]:
        raise ValueError(f"Invalid pipeline: {validation['issues']}")
    
    return pipeline.run()
```

The pipeline system provides powerful capabilities for creating complex AI workflows while maintaining flexibility and robustness.