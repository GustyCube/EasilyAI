import unittest
from unittest.mock import patch
from easilyai.services.anthropic_service import AnthropicService
from easilyai.services.gemini_service import GeminiService
from easilyai.services.grok_service import GrokService
from easilyai.exceptions import MissingAPIKeyError, ServerError

class TestAnthropicService(unittest.TestCase):
    def setUp(self):
        self.service = AnthropicService(apikey="fake_api_key", model="claude-2")

    def test_anthropic_init(self):
        self.assertEqual(self.service.model, "claude-2")

    @patch("easilyai.services.anthropic_service.AnthropicService.client.messages.create")
    def test_text_generation(self, mock_create):
        mock_create.return_value = type("Response", (), {"content": "Mocked response"})
        result = self.service.generate_text("Test prompt")
        self.assertEqual(result, "Mocked response")


class TestGeminiService(unittest.TestCase):
    def setUp(self):
        self.service = GeminiService(apikey="fake_api_key", model="gemini-1")

    def test_gemini_init(self):
        self.assertEqual(self.service.model.name, "gemini-1")

    @patch("easilyai.services.gemini_service.googleai.GenerativeModel.generate_content")
    def test_text_generation(self, mock_generate_content):
        mock_generate_content.return_value = type("Response", (), {"text": "Mocked response"})
        result = self.service.generate_text("Test prompt")
        self.assertEqual(result, "Mocked response")

    def test_missing_api_key(self):
        with self.assertRaises(MissingAPIKeyError):
            GeminiService(apikey=None, model="gemini-1")


class TestGrokService(unittest.TestCase):
    def setUp(self):
        self.service = GrokService(apikey="fake_api_key", model="grok-v1")

    def test_grok_init(self):
        self.assertEqual(self.service.model, "grok-v1")

    @patch("easilyai.services.grok_service.OpenAI.chat.completions.create")
    def test_text_generation(self, mock_completions_create):
        mock_completions_create.return_value = type(
            "Response", (), {"choices": [{"message": {"content": "Mocked response"}}]}
        )
        result = self.service.generate_text("Test prompt")
        self.assertEqual(result, "Mocked response")

    def test_missing_api_key(self):
        with self.assertRaises(MissingAPIKeyError):
            GrokService(apikey=None, model="grok-v1")


if __name__ == "__main__":
    unittest.main()
