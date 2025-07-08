import unittest
from unittest.mock import patch
from easilyai.services.openai_service import OpenAIService
from easilyai.exceptions import MissingAPIKeyError, AuthenticationError


class TestOpenAIService(unittest.TestCase):
    def setUp(self):
        self.apikey = "fake_api_key"
        self.model = "gpt-4"
        self.service = OpenAIService(apikey=self.apikey, model=self.model)

    @patch.object(OpenAIService, '__init__', lambda x, y, z: None)
    @patch('openai.OpenAI')
    def test_generate_text_success(self, mock_openai_class):
        mock_client = mock_openai_class.return_value
        mock_response = type('Response', (), {})()
        mock_choice = type('Choice', (), {})()
        mock_message = type('Message', (), {'content': 'Mocked OpenAI response'})()
        mock_choice.message = mock_message
        mock_response.choices = [mock_choice]
        
        mock_client.chat.completions.create.return_value = mock_response
        
        service = OpenAIService("fake_key", "gpt-4")
        service.client = mock_client
        service.model = "gpt-4"
        
        response = service.generate_text("Test prompt")
        self.assertEqual(response, "Mocked OpenAI response")


if __name__ == "__main__":
    unittest.main()
