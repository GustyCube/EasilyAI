import openai
from openai import OpenAI
from easilyai.exceptions import (
    AuthenticationError, RateLimitError, InvalidRequestError,
    APIConnectionError, NotFoundError, ServerError, MissingAPIKeyError
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

    def generate_text(self, prompt, img_url=None):
        try:
            content = []
            if img_url:
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": img_url,
                        "detail": "high"
                    }
                })
            content.append({
                "type": "text",
                "text": prompt
            })

            response = self.client.chat.completions.create(
                model = self.model,
                messages = [{
                    "role": "user",
                    "content": content
                }]
            )
            return response.choices[0].message.content
        
        except openai.error.AuthenticationError:
            raise AuthenticationError(
                "Authentication failed! Please check your OpenAI API key and ensure it's correct. "
                "Refer to the EasyAI documentation for more information."
            )
        except openai.error.RateLimitError:
            raise RateLimitError(
                "Rate limit exceeded! You've made too many requests in a short period. "
                "Please wait and try again later. Refer to the EasyAI documentation for more information."
            )
        except openai.error.InvalidRequestError as e:
            raise InvalidRequestError(
                f"Invalid request! {str(e)}. Please check your request parameters. "
                "Refer to the EasyAI documentation for more information."
            )
        except openai.error.APIConnectionError:
            raise APIConnectionError(
                "Connection error! Unable to connect to OpenAI's API. "
                "Please check your internet connection and try again. "
                "Refer to the EasyAI documentation for more information."
            )
        except openai.error.OpenAIError as e:
            raise ServerError(
                f"An error occurred on OpenAI's side: {str(e)}. Please try again later. "
                "Refer to the EasyAI documentation for more information."
            )
