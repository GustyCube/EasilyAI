import unittest
from easilyai.app import create_app


class TestAppCreation(unittest.TestCase):
    def test_openai_app_creation(self):
        app = create_app(name="TestOpenAIApp", service="openai", apikey="fake_api_key", model="gpt-4")
        self.assertEqual(app.name, "TestOpenAIApp")
        self.assertEqual(app.service, "openai")
        self.assertEqual(app.client.model, "gpt-4")

    def test_ollama_app_creation(self):
        app = create_app(name="TestOllamaApp", service="ollama", model="ollama-test-model")
        self.assertEqual(app.name, "TestOllamaApp")
        self.assertEqual(app.service, "ollama")
        self.assertEqual(app.client.model, "ollama-test-model")

    def test_anthropic_app_creation(self):
        app = create_app(name="TestAnthropicApp", service="anthropic", apikey="fake_api_key", model="claude-3")
        self.assertEqual(app.name, "TestAnthropicApp")
        self.assertEqual(app.service, "anthropic")
        self.assertEqual(app.client.model, "claude-3")

    def test_gemini_app_creation(self):
        app = create_app(name="TestGeminiApp", service="gemini", apikey="fake_api_key", model="gemini-1")
        self.assertEqual(app.name, "TestGeminiApp")
        self.assertEqual(app.service, "gemini")
        self.assertEqual(app.client.model.model_name, "models/gemini-1")  


    def test_grok_app_creation(self):
        app = create_app(name="TestGrokApp", service="grok", apikey="fake_api_key", model="grok-v1")
        self.assertEqual(app.name, "TestGrokApp")
        self.assertEqual(app.service, "grok")
        self.assertEqual(app.client.model, "grok-v1")

if __name__ == "__main__":
    unittest.main()
