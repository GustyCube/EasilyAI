import unittest
from unittest.mock import patch, MagicMock
from easilyai.services.huggingface_service import HuggingFaceService
from easilyai.exceptions import MissingAPIKeyError, AuthenticationError, ServerError


class TestHuggingFaceService(unittest.TestCase):
    def setUp(self):
        self.apikey = "fake_api_key"
        self.model = "gpt2"
        self.service = HuggingFaceService(apikey=self.apikey, model=self.model)
        
    def test_missing_api_key(self):
        with self.assertRaises(MissingAPIKeyError):
            HuggingFaceService(apikey=None, model=self.model)

    @patch('requests.post')
    def test_generate_text_success(self, mock_post):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = [{"generated_text": "Mocked HuggingFace response"}]
        mock_post.return_value = mock_response
        
        response = self.service.generate_text("Test prompt")
        self.assertEqual(response, "Mocked HuggingFace response")

    @patch('requests.post')
    def test_generate_text_with_parameters(self, mock_post):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = [{"generated_text": "Custom response"}]
        mock_post.return_value = mock_response
        
        response = self.service.generate_text("Test prompt", max_length=200, temperature=0.5)
        self.assertEqual(response, "Custom response")

    @patch('requests.post')
    def test_generate_text_authentication_error(self, mock_post):
        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_post.return_value = mock_response
        
        with self.assertRaises(ServerError):  # HuggingFace wraps AuthError in ServerError
            self.service.generate_text("Test prompt")

    @patch('requests.post')
    def test_generate_text_server_error(self, mock_post):
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_post.return_value = mock_response
        
        with self.assertRaises(ServerError):
            self.service.generate_text("Test prompt")

    @patch('requests.post')
    def test_generate_text_empty_response(self, mock_post):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = []
        mock_post.return_value = mock_response
        
        response = self.service.generate_text("Test prompt")
        self.assertEqual(response, "[]")  # HuggingFace returns str(result) for empty lists


if __name__ == "__main__":
    unittest.main()