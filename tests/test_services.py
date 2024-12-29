import unittest
from unittest.mock import patch
from easilyai.services.openai_service import OpenAIService
from easilyai.services.ollama_service import OllamaService

class TestOpenAIService(unittest.TestCase):
    """Unit tests for OpenAIService."""

    def setUp(self):
        self.service = OpenAIService(apikey="fake_api_key", model="gpt-4")

    def test_openai_init(self):
        self.assertEqual(self.service.model, "gpt-4")

    @patch.object(OpenAIService, "generate_text", return_value="Mocked response")
    def test_text_generation(self, mock_generate_text):
        result = self.service.generate_text("Test prompt")
        self.assertEqual(result, "Mocked response")

    def test_text_generation_failure(self):
        with self.assertRaises(Exception):
            self.service.generate_text("Test prompt")  # Without patching, it should fail with a real API call


class TestOllamaService(unittest.TestCase):
    """Unit tests for OllamaService."""

    def setUp(self):
        self.service = OllamaService(model="llama2")

    def test_ollama_init(self):
        self.assertEqual(self.service.model, "llama2")

    @patch.object(OllamaService, "generate_text", return_value="Mocked response")
    def test_text_generation(self, mock_generate_text):
        result = self.service.generate_text("Test prompt")
        self.assertEqual(result, "Mocked response")

    def test_text_generation_failure(self):
        with self.assertRaises(Exception):
            self.service.generate_text("Test prompt")  # Without patching, it should fail


if __name__ == "__main__":
    unittest.main()
