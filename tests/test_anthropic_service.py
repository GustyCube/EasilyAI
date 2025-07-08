import unittest
from unittest.mock import patch, MagicMock
from easilyai.services.anthropic_service import AnthropicService
from easilyai.exceptions import MissingAPIKeyError, AuthenticationError, ServerError


class TestAnthropicService(unittest.TestCase):
    def setUp(self):
        self.apikey = "fake_api_key"
        self.model = "claude-3-sonnet-20240229"
        
    def test_missing_api_key(self):
        with self.assertRaises(MissingAPIKeyError):
            AnthropicService(apikey=None, model=self.model)

    @patch.object(AnthropicService, '__init__', lambda x, y, z, **kwargs: None)
    @patch('anthropic.Anthropic')
    def test_generate_text_success(self, mock_anthropic_class):
        mock_client = mock_anthropic_class.return_value
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="Mocked Anthropic response")]
        mock_client.messages.create.return_value = mock_response
        
        service = AnthropicService("fake_key", "claude-3-sonnet-20240229")
        service.client = mock_client
        service.model = "claude-3-sonnet-20240229"
        service.max_tokens = 1024
        
        response = service.generate_text("Test prompt")
        self.assertEqual(response, "Mocked Anthropic response")

    @patch.object(AnthropicService, '__init__', lambda x, y, z, **kwargs: None)
    @patch('anthropic.Anthropic')
    def test_generate_text_with_image(self, mock_anthropic_class):
        mock_client = mock_anthropic_class.return_value
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="Mocked response with image")]
        mock_client.messages.create.return_value = mock_response
        
        service = AnthropicService("fake_key", "claude-3-sonnet-20240229")
        service.client = mock_client
        service.model = "claude-3-sonnet-20240229"
        service.max_tokens = 1024
        
        # Mock prepare_image to return None (simulating URL instead of local file)
        with patch.object(service, 'prepare_image', return_value=None):
            response = service.generate_text("Describe this image", "http://example.com/image.jpg")
            self.assertEqual(response, "Mocked response with image")

    @patch.object(AnthropicService, '__init__', lambda x, y, z, **kwargs: None)
    @patch('anthropic.Anthropic')
    def test_generate_text_authentication_error(self, mock_anthropic_class):
        mock_client = mock_anthropic_class.return_value
        mock_client.messages.create.side_effect = Exception("Authentication failed")
        
        service = AnthropicService("fake_key", "claude-3-sonnet-20240229")
        service.client = mock_client
        service.model = "claude-3-sonnet-20240229"
        service.max_tokens = 1024
        
        with self.assertRaises(ServerError):
            service.generate_text("Test prompt")


if __name__ == "__main__":
    unittest.main()