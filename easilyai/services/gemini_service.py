import google.generativeai as googleai
from google.api_core import exceptions as google_exceptions
from easilyai.exceptions import (
    AuthenticationError, RateLimitError, InvalidRequestError,
    APIConnectionError, NotFoundError, ServerError, MissingAPIKeyError
)

class GeminiService:
    def __init__(self, apikey, model):
        if not apikey:
            raise MissingAPIKeyError(
                "Gemini API key is missing! Please provide your API key when initializing the service. "
                "Refer to the EasilyAI documentation for more information."
            )
        googleai.configure(api_key=apikey)
        # Ensure only the last part of the model name is used
        self.model_name = model.split("/")[-1]  # Extracts "gemini-1" even if input is "models/gemini-1"
        print(self.model_name)
        self.full_model_name = model  # Full name (e.g., "models/gemini-1")
        self.model = googleai.GenerativeModel(self.full_model_name)

    def generate_text(self, prompt):
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except google_exceptions.Unauthenticated:
            raise AuthenticationError(
                "Authentication failed! Please check your Gemini API key and ensure it's correct. "
                "Refer to the EasilyAI documentation for more information."
            )
        except google_exceptions.ResourceExhausted:
            raise RateLimitError(
                "Rate limit exceeded! You've made too many requests in a short period. "
                "Please wait and try again later. Refer to the EasilyAI documentation for more information."
            )
        except google_exceptions.InvalidArgument as e:
            raise InvalidRequestError(
                f"Invalid request! {str(e)}. Please check your request parameters. "
                "Refer to the EasilyAI documentation for more information."
            )
        except google_exceptions.DeadlineExceeded:
            raise APIConnectionError(
                "Request timeout! The request took too long to complete. "
                "Please try again later. Refer to the EasilyAI documentation for more information."
            )
        except google_exceptions.NotFound:
            raise NotFoundError(
                "Model not found! Please check your model name and ensure it's correct. "
                "Refer to the EasilyAI documentation for more information."
            )
        except google_exceptions.GoogleAPIError as e:
            raise ServerError(
                f"An error occurred on Google's side: {str(e)}. Please try again later. "
                "Refer to the EasilyAI documentation for more information."
            )
        except Exception as e:
            raise ServerError(
                f"An unexpected error occurred: {str(e)}. Please try again later. "
                "Refer to the EasilyAI documentation for more information."
            )
