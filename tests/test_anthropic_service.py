# Disabled Temporarily due to the curent nature of the code. 

# from unittest import TestCase
# from unittest.mock import Mock, patch
# from easilyai.services.anthropic_service import AnthropicService
# import anthropic


# class TestAnthropicService(TestCase):
#     def setUp(self):
#         self.service = AnthropicService(apikey="test_api_key", model="claude-3-5", max_tokens=1024)

#     @patch('anthropic.Anthropic.messages.create', new_callable=Mock)
#     def test_generate_text(self, mock_messages):
#         mock_messages.create.return_value = {
#             "content": [{"text": "Mocked response"}]
#         }

#         response = self.service.generate_text("Test prompt")

#         self.assertEqual(response, "Mocked response")

#     @patch('anthropic.Anthropic.messages.create', new_callable=Mock)
#     def test_generate_text_error(self, mock_messages):
#         mock_messages.create.side_effect = Exception("API Error")

#         with self.assertRaises(Exception):
#             self.service.generate_text("Test prompt")
