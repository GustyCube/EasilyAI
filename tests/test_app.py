import unittest
from unittest.mock import patch
from easilyai.services.openai_service import OpenAIService
from easilyai.services.ollama_service import OllamaService
from easilyai.app import create_app

class BaseServiceTest(unittest.TestCase):
    """Base class for testing AI services."""
    def setUp(self):
        self.service = None  # This will be set in child classes

    def test_service_initialization(self):
        self.assertIsNotNone(self.service)
        self.assertIsInstance(self.service.model, str)

    def test_text_generation(self):
        with patch.object(self.service, "generate_text", return_value="Mocked response"):
            result = self.service.generate_text("Test prompt")
            self.assertEqual(result, "Mocked response")

class TestOpenAIService(BaseServiceTest):
    def setUp(self):
        self.service = OpenAIService(apikey="fake_api_key", model="gpt-4")

class TestOllamaService(BaseServiceTest):
    def setUp(self):
        self.service = OllamaService(model="llama2")

class TestEasyAIApp(unittest.TestCase):
    """Tests for the EasyAI app creation and functionality."""
    def test_openai_app_creation(self):
        app = create_app(name="TestApp", service="openai", apikey="fake_api_key", model="gpt-4")
        self.assertEqual(app.name, "TestApp")
        self.assertEqual(app.service, "openai")

    def test_ollama_app_creation(self):
        app = create_app(name="TestApp", service="ollama", model="llama2")
        self.assertEqual(app.name, "TestApp")
        self.assertEqual(app.service, "ollama")

    def test_invalid_service(self):
        with self.assertRaises(ValueError):
            create_app(name="TestApp", service="invalid_service")

    def test_request_not_implemented(self):
        app = create_app(name="TestApp", service="ollama", model="llama2")
        with self.assertRaises(NotImplementedError):
            app.client.generate_image("Create an image")

if __name__ == "__main__":
    unittest.main()
