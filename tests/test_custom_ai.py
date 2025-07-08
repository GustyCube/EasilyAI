import unittest
from easilyai.custom_ai import CustomAIService, register_custom_ai, _registered_custom_ais


class TestCustomAI(unittest.TestCase):
    def setUp(self):
        # Clear registered custom AIs before each test
        _registered_custom_ais.clear()

    def test_custom_ai_service_init(self):
        service = CustomAIService(model="test-model", apikey="test-key")
        self.assertEqual(service.model, "test-model")
        self.assertEqual(service.apikey, "test-key")

    def test_custom_ai_service_init_without_apikey(self):
        service = CustomAIService(model="test-model")
        self.assertEqual(service.model, "test-model")
        self.assertIsNone(service.apikey)

    def test_custom_ai_service_not_implemented_methods(self):
        service = CustomAIService(model="test-model")
        
        with self.assertRaises(NotImplementedError):
            service.generate_text("test prompt")
        
        with self.assertRaises(NotImplementedError):
            service.generate_image("test prompt")
        
        with self.assertRaises(NotImplementedError):
            service.text_to_speech("test text")

    def test_register_valid_custom_ai(self):
        class ValidCustomAI(CustomAIService):
            def generate_text(self, prompt):
                return f"Generated: {prompt}"
        
        register_custom_ai("valid_ai", ValidCustomAI)
        self.assertIn("valid_ai", _registered_custom_ais)
        self.assertEqual(_registered_custom_ais["valid_ai"], ValidCustomAI)

    def test_register_invalid_custom_ai(self):
        class InvalidCustomAI:
            pass
        
        with self.assertRaises(TypeError) as context:
            register_custom_ai("invalid_ai", InvalidCustomAI)
        
        self.assertIn("Custom service must inherit from CustomAIService", str(context.exception))
        self.assertNotIn("invalid_ai", _registered_custom_ais)

    def test_register_multiple_custom_ais(self):
        class CustomAI1(CustomAIService):
            def generate_text(self, prompt):
                return "AI1 response"
        
        class CustomAI2(CustomAIService):
            def generate_text(self, prompt):
                return "AI2 response"
        
        register_custom_ai("ai1", CustomAI1)
        register_custom_ai("ai2", CustomAI2)
        
        self.assertEqual(len(_registered_custom_ais), 2)
        self.assertIn("ai1", _registered_custom_ais)
        self.assertIn("ai2", _registered_custom_ais)

    def test_custom_ai_implementation_example(self):
        class MockCustomAI(CustomAIService):
            def generate_text(self, prompt):
                return f"Mock response to: {prompt}"
            
            def generate_image(self, prompt):
                return f"Mock image for: {prompt}"
            
            def text_to_speech(self, text):
                return f"Mock audio for: {text}"
        
        service = MockCustomAI(model="mock-model", apikey="mock-key")
        
        self.assertEqual(service.generate_text("Hello"), "Mock response to: Hello")
        self.assertEqual(service.generate_image("Cat"), "Mock image for: Cat")
        self.assertEqual(service.text_to_speech("Speech"), "Mock audio for: Speech")


if __name__ == "__main__":
    unittest.main()