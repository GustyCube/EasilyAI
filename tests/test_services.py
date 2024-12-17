import unittest
from easilyai.services.openai_service import OpenAIService
from easilyai.services.ollama_service import OllamaService

class TestOpenAIService(unittest.TestCase):
    def test_openai_init(self):
        service = OpenAIService(apikey="fake_api_key", model="gpt-4")
        self.assertEqual(service.model, "gpt-4")

    def test_text_generation(self):
        service = OpenAIService(apikey="fake_api_key", model="gpt-4")
        with self.assertRaises(Exception):
            service.generate_text("Test prompt")  # Without a real API key

class TestOllamaService(unittest.TestCase):
    def test_ollama_init(self):
        service = OllamaService(model="llama2")
        self.assertEqual(service.model, "llama2")
    
    def test_text_generation(self):
        service = OllamaService(model="llama2")
        with self.assertRaises(Exception):
            service.generate_text("Test prompt")  # Ollama might not be running locally

if __name__ == "__main__":
    unittest.main()
