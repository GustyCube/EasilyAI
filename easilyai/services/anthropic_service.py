import os
import base64
import anthropic
from easilyai.exceptions import (
    AuthenticationError, RateLimitError, InvalidRequestError,
    APIConnectionError, NotFoundError, ServerError, MissingAPIKeyError
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
        self.client = anthropic.Anthropic(api_key=apikey)  # Correct initialization

    def prepare_image(self, img_url):
        """Prepare image for Claude API - handles both local files and URLs."""
        if os.path.exists(img_url):  # Local file
            with open(img_url, "rb") as f:
                image_data = base64.b64encode(f.read()).decode("utf-8")
            
            # Determine media type based on file extension
            if img_url.lower().endswith('.png'):
                media_type = 'image/png'
            elif img_url.lower().endswith('.jpg') or img_url.lower().endswith('.jpeg'):
                media_type = 'image/jpeg'
            elif img_url.lower().endswith('.gif'):
                media_type = 'image/gif'
            elif img_url.lower().endswith('.webp'):
                media_type = 'image/webp'
            else:
                media_type = 'image/jpeg'  # Default fallback
            
            return {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": media_type,
                    "data": image_data
                }
            }
        else:
            # Handle URLs - Claude doesn't support URLs directly, so we'd need to fetch and encode
            # For now, return None to indicate URL handling isn't supported
            return None

    def generate_text(self, prompt, img_url=None):
        try:
            if img_url:
                # Vision mode - include image with prompt
                image_block = self.prepare_image(img_url)
                if image_block:
                    content = [
                        {"type": "text", "text": prompt},
                        image_block
                    ]
                else:
                    # If image preparation failed (e.g., URL), fall back to text-only
                    content = prompt
            else:
                # Text-only mode
                content = prompt
            
            response = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                messages=[{"role": "user", "content": content}],
            )
            # Extract the text content
            return response.content[0].text
        except anthropic.AuthenticationError:
            raise AuthenticationError("Invalid API key. Please check your Anthropic API key.")
        except anthropic.RateLimitError:
            raise RateLimitError("Rate limit exceeded. Please wait and try again later.")
        except anthropic.BadRequestError as e:
            raise InvalidRequestError(f"Invalid request: {str(e)}. Check your parameters.")
        except anthropic.APIConnectionError:
            raise APIConnectionError("Unable to connect to Anthropic API. Check your network.")
        except Exception as e:
            raise ServerError(
                f"An unexpected error occurred: {str(e)}. Please try again later."
            )
