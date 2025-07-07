import os
import base64
import openai
from openai import OpenAI
from easilyai.exceptions import (
    AuthenticationError, RateLimitError, InvalidRequestError,
    APIConnectionError, NotFoundError, ServerError, MissingAPIKeyError
)

class OpenAIService:
    def __init__(self, apikey, model):
        if not apikey:
            raise MissingAPIKeyError(
                "OpenAI API key is missing! Please provide your API key when initializing the service. "
                "Refer to the EasyAI documentation for more information."
            )
        openai.api_key = apikey
        self.client = OpenAI(api_key=apikey)
        self.model = model

    def encode_image(self, img_url):
        """Encodes an image file into Base64 format if it's a local file."""
        if os.path.exists(img_url):  # Check if it's a local file
            with open(img_url, "rb") as f:
                encoded_string = base64.b64encode(f.read()).decode("utf-8")
            return f"data:image/jpeg;base64,{encoded_string}"
        return img_url  # Assume it's already a URL if the file doesn't exist locally

    def generate_text(self, prompt, img_url=None):
        try:
            # Prepare messages for vision or text-only
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
            else:
                messages = [{"role": "user", "content": prompt}]
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages
            )
            return response.choices[0].message.content
        except openai.AuthenticationError:
            raise AuthenticationError(
                "Authentication failed! Please check your OpenAI API key and ensure it's correct. "
                "Refer to the EasyAI documentation for more information."
            )
        except openai.RateLimitError:
            raise RateLimitError(
                "Rate limit exceeded! You've made too many requests in a short period. "
                "Please wait and try again later. Refer to the EasyAI documentation for more information."
            )
        except openai.BadRequestError as e:
            raise InvalidRequestError(
                f"Invalid request! {str(e)}. Please check your request parameters. "
                "Refer to the EasyAI documentation for more information."
            )
        except openai.APIConnectionError:
            raise APIConnectionError(
                "Connection error! Unable to connect to OpenAI's API. "
                "Please check your internet connection and try again. "
                "Refer to the EasyAI documentation for more information."
            )
        except openai.OpenAIError as e:
            raise ServerError(
                f"An error occurred on OpenAI's side: {str(e)}. Please try again later. "
                "Refer to the EasyAI documentation for more information."
            )

    def generate_image(self, prompt, model="dall-e-3", size="1024x1024", quality="standard", n=1):
        try:
            response = self.client.images.generate(
                model=model,
                prompt=prompt,
                size=size,
                quality=quality,
                n=n
            )
            return response.data[0].url
        except openai.AuthenticationError:
            raise AuthenticationError(
                "Authentication failed! Please check your OpenAI API key and ensure it's correct. "
                "Refer to the EasyAI documentation for more information."
            )
        except openai.RateLimitError:
            raise RateLimitError(
                "Rate limit exceeded! You've made too many requests in a short period. "
                "Please wait and try again later. Refer to the EasyAI documentation for more information."
            )
        except openai.BadRequestError as e:
            raise InvalidRequestError(
                f"Invalid request! {str(e)}. Please check your request parameters. "
                "Refer to the EasyAI documentation for more information."
            )
        except openai.APIConnectionError:
            raise APIConnectionError(
                "Connection error! Unable to connect to OpenAI's API. "
                "Please check your internet connection and try again. "
                "Refer to the EasyAI documentation for more information."
            )
        except openai.OpenAIError as e:
            raise ServerError(
                f"An error occurred on OpenAI's side: {str(e)}. Please try again later. "
                "Refer to the EasyAI documentation for more information."
            )
