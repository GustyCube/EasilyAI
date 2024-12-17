import unittest
from easyai.app import create_app

class TestEasyAIApp(unittest.TestCase):
    def test_openai_app_creation(self):
        app = create_app(name="TestApp", service="openai", apikey="fake_api_key", model="gpt-4")
        self.assertEqual(app.name, "TestApp")
        self.assertEqual(app.service, "openai")
    
    def test_invalid_service(self):
        with self.assertRaises(ValueError):
            create_app(name="TestApp", service="invalid_service")
    
    def test_request_placeholder(self):
        app = create_app(name="TestApp", service="ollama", model="llama2")
        with self.assertRaises(NotImplementedError):
            app.client.generate_image("Create an image")
    
if __name__ == "__main__":
    unittest.main()
