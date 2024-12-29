import google.generativeai as googleai
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
        except Exception as e:
            raise ServerError(
                f"Unknown error occurred! Please try again later or look at the EasilyAI Docs. Error: {e}"
            )
