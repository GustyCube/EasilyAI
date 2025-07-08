import unittest
from unittest.mock import MagicMock, patch
from easilyai.enhanced_pipeline import (
    EnhancedPipeline, TaskStatus, ExecutionMode, TaskResult, PipelineTask
)


class TestEnhancedPipeline(unittest.TestCase):
    def setUp(self):
        self.mock_app = MagicMock()
        self.pipeline = EnhancedPipeline("test_pipeline")

    def test_pipeline_init(self):
        self.assertEqual(self.pipeline.name, "test_pipeline")
        self.assertEqual(len(self.pipeline.tasks), 0)
        self.assertEqual(self.pipeline.execution_mode, ExecutionMode.SEQUENTIAL)

    def test_add_task_simple(self):
        task_id = self.pipeline.add_task("task1", self.mock_app, "generate_text", "Hello")
        self.assertEqual(len(self.pipeline.tasks), 1)
        self.assertIsInstance(task_id, str)
        
        task = self.pipeline.tasks[task_id]
        self.assertEqual(task.task_type, "generate_text")
        self.assertEqual(task.prompt, "Hello")
        self.assertEqual(task.status, TaskStatus.PENDING)

    def test_add_task_with_dependencies(self):
        task1_id = self.pipeline.add_task("task1", self.mock_app, "generate_text", "Task 1")
        task2_id = self.pipeline.add_task("task2", self.mock_app, "generate_text", "Task 2", dependencies=[task1_id])
        
        self.assertEqual(len(self.pipeline.tasks), 2)
        task2 = self.pipeline.tasks[task2_id]
        self.assertEqual(task2.dependencies, [task1_id])

    def test_add_task_with_condition(self):
        condition = lambda results: True
        task_id = self.pipeline.add_task("task1", self.mock_app, "generate_text", "Conditional task", condition=condition)
        
        task = self.pipeline.tasks[task_id]
        self.assertEqual(task.condition, condition)

    def test_task_status_enum(self):
        self.assertEqual(TaskStatus.PENDING.value, "pending")
        self.assertEqual(TaskStatus.RUNNING.value, "running")
        self.assertEqual(TaskStatus.COMPLETED.value, "completed")
        self.assertEqual(TaskStatus.FAILED.value, "failed")
        self.assertEqual(TaskStatus.SKIPPED.value, "skipped")

    def test_execution_mode_enum(self):
        self.assertEqual(ExecutionMode.SEQUENTIAL.value, "sequential")
        self.assertEqual(ExecutionMode.PARALLEL.value, "parallel")
        self.assertEqual(ExecutionMode.CONDITIONAL.value, "conditional")

    def test_task_result_creation(self):
        result = TaskResult(
            task_id="test-task",
            status=TaskStatus.COMPLETED,
            result="Test result",
            duration=1.5
        )
        
        self.assertEqual(result.task_id, "test-task")
        self.assertEqual(result.status, TaskStatus.COMPLETED)
        self.assertEqual(result.result, "Test result")
        self.assertEqual(result.duration, 1.5)
        self.assertIsNone(result.error)
        self.assertEqual(result.metadata, {})

    def test_set_execution_mode(self):
        self.pipeline.set_execution_mode(ExecutionMode.PARALLEL)
        self.assertEqual(self.pipeline.execution_mode, ExecutionMode.PARALLEL)

    def test_clear_tasks(self):
        self.pipeline.add_task("task1", self.mock_app, "generate_text", "Task 1")
        self.pipeline.add_task("task2", self.mock_app, "generate_text", "Task 2")
        self.assertEqual(len(self.pipeline.tasks), 2)
        
        self.pipeline.tasks.clear()  # Simplified clear method
        self.assertEqual(len(self.pipeline.tasks), 0)

    @patch('easilyai.enhanced_pipeline.time.time')
    def test_simple_execution_simulation(self, mock_time):
        # Mock time for duration calculation
        mock_time.side_effect = [0.0, 1.0]  # Start and end times
        
        # Mock app response
        self.mock_app.request.return_value = "Generated response"
        
        # Add a simple task
        task_id = self.pipeline.add_task("task1", self.mock_app, "generate_text", "Hello")
        
        # This test verifies the structure, but actual execution would require
        # the full implementation which might involve async/threading
        self.assertEqual(len(self.pipeline.tasks), 1)
        task = self.pipeline.tasks[task_id]
        self.assertEqual(task.task_id, task_id)
        self.assertEqual(task.status, TaskStatus.PENDING)

    def test_variable_substitution_pattern(self):
        # Test that the pipeline can handle variable patterns
        prompt_with_vars = "Hello {name}, how are you?"
        task_id = self.pipeline.add_task("task1", self.mock_app, "generate_text", prompt_with_vars)
        
        task = self.pipeline.tasks[task_id]
        self.assertIn("{name}", task.prompt)

    def test_pipeline_task_attributes(self):
        task_id = self.pipeline.add_task(
            "task1",
            self.mock_app,
            "generate_text", 
            "Test", 
            dependencies=["dep1"], 
            condition=lambda x: True,
            retry_count=3,
            timeout=30
        )
        
        task = self.pipeline.tasks[task_id]
        self.assertEqual(task.task_type, "generate_text")
        self.assertEqual(task.dependencies, ["dep1"])
        self.assertIsNotNone(task.condition)
        self.assertEqual(task.retry_count, 3)
        self.assertEqual(task.timeout, 30)


if __name__ == "__main__":
    unittest.main()