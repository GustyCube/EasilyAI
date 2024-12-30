# Pipeline Guide

## Overview
Pipelines in EasilyAI allow you to chain multiple tasks (e.g., text generation, image generation, and TTS) into a workflow.

## Example Pipeline

```python
# Create a pipeline
pipeline = easyai.EasilyAIPipeline(app)

# Add tasks
pipeline.add_task("generate_text", "Write a poem about AI and nature.")
pipeline.add_task("generate_image", "A futuristic city skyline.")
pipeline.add_task("text_to_speech", "Here is a futuristic AI-powered city!")

# Run the pipeline
results = pipeline.run()

# Print results
for task_result in results:
    print(f"Task: {task_result['task']}\nResult: {task_result['result']}\n")
```

Discover how to extend EasilyAI with [Custom AI Models](./customai.md).
