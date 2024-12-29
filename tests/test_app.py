import unittest
from easilyai.app import create_app
from easilyai.exceptions import ValueError, MissingAPIKeyError

class TestEasyAIApp(unittest.TestCase):
    """Tests for the EasyAI app creation and functionality."""

    def test_openai_app_creation(self):
        app = create_app(name="TestOpenAIApp", service="openai", apikey="fake_api_key", model="gpt-4")
        self.assertEqual(app.name, "TestOpenAIApp")
        self.assertEqual(app.service, "openai")

    def test_ollama_app_creation(self):
        app = create_app(name="TestOllamaApp", service="ollama", model="llama2")
        self.assertEqual(app.name, "TestOllamaApp")
        self.assertEqual(app.service, "ollama")

    def test_anthropic_app_creation(self):
        app = create_app(name="TestAnthropicApp", service="anthropic", apikey="fake_api_key", model="claude-2")
        self.assertEqual(app.name, "TestAnthropicApp")
        self.assertEqual(app.service, "anthropic")

    def test_gemini_app_creation(self):
        app = create_app(name="TestGeminiApp", service="gemini", apikey="fake_api_key", model="gemini-1")
        self.assertEqual(app.name, "TestGeminiApp")
        self.assertEqual(app.service, "gemini")

    def test_grok_app_creation(self):
        app = create_app(name="TestGrokApp", service="grok", apikey="fake_api_key", model="grok-v1")
        self.assertEqual(app.name, "TestGrokApp")
        self.assertEqual(app.service, "grok")

    def test_invalid_service(self):
        with self.assertRaises(ValueError):
            create_app(name="TestApp", service="invalid_service")

    def test_missing_api_key(self):
        with self.assertRaises(MissingAPIKeyError):
            create_app(name="TestApp", service="anthropic", apikey=None)

    def test_request_not_implemented(self):
        app = create_app(name="TestApp", service="ollama", model="llama2")
        with self.assertRaises(NotImplementedError):
            app.client.generate_image("Create an image")


if __name__ == "__main__":
    unittest.main()
