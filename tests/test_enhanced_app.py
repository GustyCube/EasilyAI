import unittest
from unittest.mock import MagicMock, patch


class TestEnhancedApp(unittest.TestCase):

    def test_enhanced_app_imports(self):
        # Test that enhanced app modules can be imported
        try:
            from easilyai.enhanced_app import create_enhanced_app
            self.assertTrue(callable(create_enhanced_app))
        except ImportError:
            self.fail("Could not import enhanced_app module")

    def test_enhanced_app_function_signature(self):
        # Test the function exists with correct parameters
        from easilyai.enhanced_app import create_enhanced_app
        import inspect
        
        sig = inspect.signature(create_enhanced_app)
        params = list(sig.parameters.keys())
        
        # Check required parameters exist
        self.assertIn('name', params)
        self.assertIn('service', params)
        self.assertIn('api_key', params)
        self.assertIn('model', params)


if __name__ == "__main__":
    unittest.main()