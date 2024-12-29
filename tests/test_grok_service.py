# import unittest
# from unittest.mock import patch
# from easilyai.services.grok_service import GrokService
# from openai import BadRequestError
# from easilyai.exceptions import InvalidRequestError as EasyAIInvalidRequestError, ServerError

# class TestGrokService(unittest.TestCase):
#     def setUp(self):
#         self.service = GrokService(apikey="fake_api_key", model="grok-v1")

#     @patch("openai.ChatCompletion.create")
#     def test_generate_text_success(self, mock_create):
#         mock_create.return_value = {
#             "choices": [{"message": {"content": "Mocked Grok response"}}]
#         }
#         response = self.service.generate_text("Explain Grok")
#         self.assertEqual(response, "Mocked Grok response")

# if __name__ == "__main__":
#     unittest.main()
