import unittest
from unittest.mock import patch, MagicMock
from easilyai.services.grok_service import GrokService
from easilyai.exceptions import MissingAPIKeyError, AuthenticationError, ServerError


class TestGrokService(unittest.TestCase):
    def setUp(self):
        self.apikey = "fake_api_key"
        self.model = "grok-beta"
        
    def test_missing_api_key(self):
        with self.assertRaises(MissingAPIKeyError):
            GrokService(apikey=None, model=self.model)

    @patch.object(GrokService, '__init__', lambda x, y, z: None)
    @patch('openai.OpenAI')
    def test_generate_text_success(self, mock_openai_class):
        mock_client = mock_openai_class.return_value
        mock_response = MagicMock()
        mock_choice = MagicMock()
        mock_message = MagicMock()
        mock_message.content = "Mocked Grok response"
        mock_choice.message = mock_message
        mock_response.choices = [mock_choice]
        
        mock_client.chat.completions.create.return_value = mock_response
        
        service = GrokService("fake_key", "grok-beta")
        service.client = mock_client
        service.model = "grok-beta"
        
        response = service.generate_text("Explain quantum physics")
        self.assertEqual(response, "Mocked Grok response")

    @patch.object(GrokService, '__init__', lambda x, y, z: None)
    @patch('openai.OpenAI')
    def test_generate_text_with_image(self, mock_openai_class):
        mock_client = mock_openai_class.return_value
        mock_response = MagicMock()
        mock_choice = MagicMock()
        mock_message = MagicMock()
        mock_message.content = "Mocked response with image"
        mock_choice.message = mock_message
        mock_response.choices = [mock_choice]
        
        mock_client.chat.completions.create.return_value = mock_response
        
        service = GrokService("fake_key", "grok-beta")
        service.client = mock_client
        service.model = "grok-beta"
        
        # Mock encode_image to return the URL unchanged (simulating URL instead of local file)
        with patch.object(service, 'encode_image', return_value="http://example.com/image.jpg"):
            response = service.generate_text("What's in this image?", "http://example.com/image.jpg")
            self.assertEqual(response, "Mocked response with image")

    @patch.object(GrokService, '__init__', lambda x, y, z: None)
    @patch('openai.OpenAI')
    def test_generate_text_server_error(self, mock_openai_class):
        mock_client = mock_openai_class.return_value
        mock_client.chat.completions.create.side_effect = Exception("Server error")
        
        service = GrokService("fake_key", "grok-beta")
        service.client = mock_client
        service.model = "grok-beta"
        
        with self.assertRaises(ServerError):
            service.generate_text("Test prompt")


if __name__ == "__main__":
    unittest.main()