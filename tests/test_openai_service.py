import unittest
from unittest.mock import patch
from easilyai.services.openai_service import OpenAIService
from easilyai.exceptions import MissingAPIKeyError, AuthenticationError


class TestOpenAIService(unittest.TestCase):
    def setUp(self):
        self.apikey = "fake_api_key"
        self.model = "gpt-4"
        self.service = OpenAIService(apikey=self.apikey, model=self.model)

    @patch("openai.ChatCompletion.create")
    def test_generate_text_success(self, mock_create):
        mock_create.return_value = {
            "choices": [{"message": {"content": "Mocked OpenAI response"}}]
        }
        response = self.service.generate_text("Test prompt")
        self.assertEqual(response, "Mocked OpenAI response")


if __name__ == "__main__":
    unittest.main()
