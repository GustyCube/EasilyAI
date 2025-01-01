import anthropic
from easilyai.exceptions import (
    AuthenticationError, RateLimitError, InvalidRequestError,
    APIConnectionError, ServerError, MissingAPIKeyError
)

class AnthropicService:
    def __init__(self, apikey, model, max_tokens=1024):
        if not apikey:
            raise MissingAPIKeyError(
                "Anthropic API key is missing! Please provide your API key when initializing the service. "
                "Refer to the EasyAI documentation for more information."
            )
        self.apikey = apikey
        self.model = model
        self.max_tokens = max_tokens
        self.client = anthropic.Anthropic(api_key=apikey)

    def generate_text(self, prompt):
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                messages=[{"role": "user", "content": prompt}],
            )
            return response["content"][0]["text"]
        except anthropic.errors.AuthenticationError:
            raise AuthenticationError("Invalid API key. Please check your Anthropic API key.")
        except anthropic.errors.RateLimitError:
            raise RateLimitError("Rate limit exceeded. Please wait and try again later.")
        except anthropic.errors.InvalidRequestError as e:
            raise InvalidRequestError(f"Invalid request: {str(e)}. Check your parameters.")
        except anthropic.errors.APIConnectionError:
            raise APIConnectionError("Unable to connect to Anthropic API. Check your network.")
        except Exception as e:
            raise ServerError(f"An unexpected error occurred: {str(e)}. Please try again later.")
