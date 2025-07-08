import unittest
from unittest.mock import MagicMock, patch
from easilyai.pipeline import EasilyAIPipeline


class TestEasilyAIPipeline(unittest.TestCase):
    def setUp(self):
        self.mock_app = MagicMock()
        self.pipeline = EasilyAIPipeline(self.mock_app)

    def test_add_task_simple(self):
        self.pipeline.add_task("generate_text", "Hello, world!")
        self.assertEqual(len(self.pipeline.tasks), 1)
        self.assertEqual(self.pipeline.tasks[0]["type"], "generate_text")
        self.assertEqual(self.pipeline.tasks[0]["data"], "Hello, world!")

    def test_add_task_with_kwargs(self):
        self.pipeline.add_task("generate_text", "Hello", max_tokens=100, temperature=0.7)
        self.assertEqual(len(self.pipeline.tasks), 1)
        task = self.pipeline.tasks[0]
        self.assertEqual(task["type"], "generate_text")
        self.assertEqual(task["data"]["data"], "Hello")
        self.assertEqual(task["data"]["max_tokens"], 100)
        self.assertEqual(task["data"]["temperature"], 0.7)

    def test_add_multiple_tasks(self):
        self.pipeline.add_task("generate_text", "Task 1")
        self.pipeline.add_task("generate_image", "Task 2")
        self.pipeline.add_task("text_to_speech", "Task 3")
        self.assertEqual(len(self.pipeline.tasks), 3)

    @patch('builtins.print')
    def test_run_generate_text_task(self, mock_print):
        self.mock_app.request.return_value = "Generated text response"
        self.pipeline.add_task("generate_text", "Test prompt")
        
        results = self.pipeline.run()
        
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["task"], "generate_text")
        self.assertEqual(results[0]["result"], "Generated text response")
        self.mock_app.request.assert_called_once_with("generate_text", "Test prompt")

    @patch('builtins.print')
    def test_run_generate_image_task(self, mock_print):
        self.mock_app.request.return_value = "Generated image path"
        self.pipeline.add_task("generate_image", "A beautiful sunset")
        
        results = self.pipeline.run()
        
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["task"], "generate_image")
        self.assertEqual(results[0]["result"], "Generated image path")
        self.mock_app.request.assert_called_once_with("Generate an image: A beautiful sunset")

    @patch('builtins.print')
    def test_run_text_to_speech_task(self, mock_print):
        self.mock_app.client.text_to_speech.return_value = "audio_file.mp3"
        self.pipeline.add_task("text_to_speech", "Hello world")
        
        results = self.pipeline.run()
        
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["task"], "text_to_speech")
        self.assertEqual(results[0]["result"], "audio_file.mp3")
        self.mock_app.client.text_to_speech.assert_called_once_with("Hello world")

    @patch('builtins.print')
    def test_run_multiple_tasks(self, mock_print):
        self.mock_app.request.return_value = "Response"
        self.mock_app.client.text_to_speech.return_value = "audio.mp3"
        
        self.pipeline.add_task("generate_text", "Text task")
        self.pipeline.add_task("text_to_speech", "Speech task")
        
        results = self.pipeline.run()
        
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["task"], "generate_text")
        self.assertEqual(results[1]["task"], "text_to_speech")

    @patch('builtins.print')
    def test_run_unknown_task_type(self, mock_print):
        self.pipeline.add_task("unknown_task", "Test data")
        
        with self.assertRaises(ValueError) as context:
            self.pipeline.run()
        
        self.assertIn("Unknown task type: unknown_task", str(context.exception))


if __name__ == "__main__":
    unittest.main()