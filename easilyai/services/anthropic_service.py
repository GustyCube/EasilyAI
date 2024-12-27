import anthropic

from easilyai.exceptions import (
    AuthenticationError, RateLimitError, InvalidRequestError,
    APIConnectionError, NotFoundError, ServerError, MissingAPIKeyError
)

class AnthropicService:
    def __init__(self, apikey, model, max_tokens = 1024):
        self.apikey = apikey
        self.model = model
        self.max_tokens = max_tokens
        self.client = anthropic.Anthropic(apikey)

    def generate_text(self, prompt):
        try:
            response = self.client.messages.create(max_tokens = self.max_tokens,
                                                   messages = [{"role": "user", "content": prompt}],
                                                   model = self.model)
            return response.content
        except Exception as e:
            raise ServerError(
                f"Unknown error occurred! Please try again later or look at the EasilyAi Docs. Error: {e}"
            )
