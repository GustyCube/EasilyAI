import unittest
from unittest.mock import patch
from easilyai.services.gemini_service import GeminiService
from easilyai.exceptions import MissingAPIKeyError, ServerError

class TestGeminiService(unittest.TestCase):
    def setUp(self):
        self.service = GeminiService(apikey="fake_api_key", model="gemini-1")

    def test_missing_api_key(self):
        with self.assertRaises(MissingAPIKeyError):
            GeminiService(apikey=None, model="gemini-1")

    @patch("google.generativeai.GenerativeModel.generate_content")
    def test_generate_text_success(self, mock_generate):
        mock_generate.return_value = MockResponse("Mocked Gemini response")
        response = self.service.generate_text("Test prompt")
        self.assertEqual(response, "Mocked Gemini response")

    @patch("google.generativeai.GenerativeModel.generate_content")
    def test_generate_text_server_error(self, mock_generate):
        mock_generate.side_effect = Exception("Server error")
        with self.assertRaises(ServerError):
            self.service.generate_text("Error prompt")

class MockResponse:
    def __init__(self, text):
        self.text = text

if __name__ == "__main__":
    unittest.main()
