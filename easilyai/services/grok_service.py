import os
from openai import OpenAI
from easilyai.exceptions import MissingAPIKeyError, ServerError


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

    def generate_text(self, prompt):
        try:
            response = self.client.chat.completions.create(
                model="grok-2-1212",
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
            )
            return response.choices[0].message.content
        except Exception as e:
            raise ServerError(
                f"Unknown error occurred! Please try again later or look at the EasilyAi Docs. Error: {e}"
            )
