import unittest
from unittest.mock import patch
from requests.exceptions import ConnectionError
from easilyai.services.ollama_service import OllamaService
from easilyai.exceptions import APIConnectionError

class TestOllamaService(unittest.TestCase):
    def setUp(self):
        self.service = OllamaService(model="llama2")

    @patch("requests.post")
    def test_generate_text_success(self, mock_post):
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = {"response": "Mocked Ollama response"}
        response = self.service.generate_text("Test prompt")
        self.assertEqual(response, "Mocked Ollama response")

    @patch("requests.post")
    def test_generate_text_connection_error(self, mock_post):
        mock_post.side_effect = ConnectionError
        with self.assertRaises(APIConnectionError):
            self.service.generate_text("Test prompt")

if __name__ == "__main__":
    unittest.main()
