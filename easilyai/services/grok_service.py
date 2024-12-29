import os
import base64
from openai import OpenAI
from easilyai.exceptions import (
    AuthenticationError,
    RateLimitError,
    InvalidRequestError,
    APIConnectionError,
    NotFoundError,
    ServerError,
    MissingAPIKeyError,
)

class GrokService:
    def __init__(self, apikey, model):
        if not apikey:
            raise MissingAPIKeyError(
                "Grok API key is missing! Please provide your API key when initializing the service. "
                "Refer to the EasyAI documentation for more information."
            )
        self.model = model
        self.client = OpenAI(
            api_key=apikey,
            base_url="https://api.x.ai/v1",
        )

    def encode_image(self, img_url):
        """Encodes an image file into Base64 format if it's a local file."""
        if os.path.exists(img_url):  # Check if it's a local file
            with open(img_url, "rb") as f:
                encoded_string = base64.b64encode(f.read()).decode("utf-8")
            return f"data:image/jpeg;base64,{encoded_string}"
        return img_url  # Assume it's already a URL if the file doesn't exist locally

    def generate_text(self, prompt, img_url=None):
        """Generates text using Grok's chat completion API."""
        try:
            # Prepare messages
            messages = [{"role": "user", "content": prompt}]
            
            if img_url:
                encoded_img = self.encode_image(img_url)
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": encoded_img, "detail": "high"}},
                            {"type": "text", "text": prompt},
                        ],
                    }
                ]

            # Send request to Grok
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.7,  # Adjust based on your needs
            )

            # Return response content
            return response.choices[0].message.content

        except Exception as e:
            self.handle_exception(e)

    @staticmethod
    def handle_exception(exception):
        """Handles known exceptions and raises custom errors."""
        if isinstance(exception, AuthenticationError):
            raise AuthenticationError(
                "Authentication failed! Please check your Grok API key."
            )
        elif isinstance(exception, RateLimitError):
            raise RateLimitError(
                "Rate limit exceeded! You've made too many requests. Please wait and try again later."
            )
        elif isinstance(exception, InvalidRequestError):
            raise InvalidRequestError(
                f"Invalid request! {str(exception)}. Please check your request parameters."
            )
        elif isinstance(exception, APIConnectionError):
            raise APIConnectionError(
                "Connection error! Unable to connect to Grok's API. Please check your internet connection."
            )
        else:
            raise ServerError(
                f"An unknown error occurred while using Grok's API: {str(exception)}"
            )
