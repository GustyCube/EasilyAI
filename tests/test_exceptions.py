import unittest
from easilyai.exceptions import (
    EasilyAIError, AuthenticationError, RateLimitError, InvalidRequestError,
    APIConnectionError, NotFoundError, ServerError, MissingAPIKeyError,
    UnsupportedServiceError, Color
)


class TestExceptions(unittest.TestCase):
    
    def test_base_exception(self):
        error = EasilyAIError("Base error message")
        self.assertIsInstance(error, Exception)
        self.assertEqual(str(error), "Base error message")

    def test_authentication_error_default_message(self):
        error = AuthenticationError()
        self.assertIn("Authentication failed!", str(error))
        self.assertIn(Color.RED, str(error))
        self.assertIn(Color.RESET, str(error))
        self.assertIn("🔑", str(error))

    def test_authentication_error_custom_message(self):
        custom_message = "Invalid API key"
        error = AuthenticationError(custom_message)
        self.assertIn(custom_message, str(error))
        self.assertIn(Color.RED, str(error))

    def test_rate_limit_error_default_message(self):
        error = RateLimitError()
        self.assertIn("API rate limit exceeded!", str(error))
        self.assertIn(Color.YELLOW, str(error))
        self.assertIn("⏳", str(error))

    def test_rate_limit_error_custom_message(self):
        custom_message = "Too many requests"
        error = RateLimitError(custom_message)
        self.assertIn(custom_message, str(error))

    def test_invalid_request_error(self):
        error = InvalidRequestError("Bad request format")
        self.assertIn("Bad request format", str(error))
        self.assertIn(Color.RED, str(error))
        self.assertIn("🚫", str(error))

    def test_api_connection_error(self):
        error = APIConnectionError("Network timeout")
        self.assertIn("Network timeout", str(error))
        self.assertIn(Color.CYAN, str(error))
        self.assertIn("🌐", str(error))

    def test_not_found_error(self):
        error = NotFoundError("Model not found")
        self.assertIn("Model not found", str(error))
        self.assertIn(Color.YELLOW, str(error))
        self.assertIn("🔍", str(error))

    def test_server_error(self):
        error = ServerError("Internal server error")
        self.assertIn("Internal server error", str(error))
        self.assertIn(Color.RED, str(error))
        self.assertIn("💥", str(error))

    def test_missing_api_key_error(self):
        error = MissingAPIKeyError("Please provide API key")
        self.assertIn("Please provide API key", str(error))
        self.assertIn(Color.RED, str(error))
        self.assertIn("🔐", str(error))

    def test_unsupported_service_error(self):
        service_name = "unknown_service"
        error = UnsupportedServiceError(service_name)
        self.assertIn(service_name, str(error))
        self.assertIn(Color.BLUE, str(error))
        self.assertIn("❌", str(error))
        self.assertIn("Unsupported service", str(error))

    def test_color_constants(self):
        self.assertEqual(Color.RESET, "\033[0m")
        self.assertEqual(Color.RED, "\033[91m")
        self.assertEqual(Color.GREEN, "\033[92m")
        self.assertEqual(Color.YELLOW, "\033[93m")
        self.assertEqual(Color.BLUE, "\033[94m")
        self.assertEqual(Color.CYAN, "\033[96m")

    def test_exception_inheritance(self):
        # Test that all custom exceptions inherit from EasilyAIError
        self.assertTrue(issubclass(AuthenticationError, EasilyAIError))
        self.assertTrue(issubclass(RateLimitError, EasilyAIError))
        self.assertTrue(issubclass(InvalidRequestError, EasilyAIError))
        self.assertTrue(issubclass(APIConnectionError, EasilyAIError))
        self.assertTrue(issubclass(NotFoundError, EasilyAIError))
        self.assertTrue(issubclass(ServerError, EasilyAIError))
        self.assertTrue(issubclass(MissingAPIKeyError, EasilyAIError))
        self.assertTrue(issubclass(UnsupportedServiceError, EasilyAIError))

    def test_exception_can_be_caught(self):
        # Test that exceptions can be caught properly
        with self.assertRaises(EasilyAIError):
            raise AuthenticationError("Test error")
        
        with self.assertRaises(AuthenticationError):
            raise AuthenticationError("Test error")

    def test_all_exceptions_have_colored_output(self):
        # Test that all exception messages contain ANSI reset code
        exceptions = [
            AuthenticationError(),
            RateLimitError(),
            InvalidRequestError(),
            APIConnectionError(),
            NotFoundError(),
            ServerError(),
            MissingAPIKeyError(),
            UnsupportedServiceError("test")
        ]
        
        for exception in exceptions:
            self.assertIn(Color.RESET, str(exception))


if __name__ == "__main__":
    unittest.main()