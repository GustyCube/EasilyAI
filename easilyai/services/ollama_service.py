import requests
from easilyai.exceptions import (
    APIConnectionError, InvalidRequestError, NotFoundError,
    ServerError, MissingAPIKeyError, NotImplementedError
)

class OllamaService:
    def __init__(self, model):
        self.model = model
        self.base_url = "http://localhost:11434/api"

    def generate_text(self, prompt):
        payload = {"model": self.model, "prompt": prompt, "stream": False}
        try:
            response = requests.post(f"{self.base_url}/generate", json=payload)
            response.raise_for_status()
            return response.json().get("response", "").strip()
        except requests.exceptions.ConnectionError:
            raise APIConnectionError(
                "Connection error! Unable to connect to Ollama's API. "
                "Please ensure Ollama is running locally and accessible. "
                "Refer to the EasyAI documentation for more information."
            )
        except requests.exceptions.HTTPError as e:
            if response.status_code == 400:
                raise InvalidRequestError(
                    f"Invalid request! {str(e)}. Please check your request parameters. "
                    "Refer to the EasyAI documentation for more information."
                )
            elif response.status_code == 404:
                raise NotFoundError(
                    f"Resource not found! {str(e)}. The requested endpoint does not exist. "
                    "Refer to the EasyAI documentation for more information."
                )
            elif response.status_code >= 500:
                raise ServerError(
                    f"Server error! {str(e)}. Ollama's server encountered an error. "
                    "Please try again later. Refer to the EasyAI documentation for more information."
                )
            else:
                raise ServerError(
                    f"An unexpected error occurred: {str(e)}. Please try again later. "
                    "Refer to the EasyAI documentation for more information."
                )
        except requests.exceptions.RequestException as e:
            raise ServerError(
                f"An error occurred: {str(e)}. Please try again later. "
                "Refer to the EasyAI documentation for more information."
            )

    def text_to_speech(self, text, **kwargs):
        raise NotImplementedError(
            "Text-to-Speech (TTS) is not supported for Ollama models at this time. "
            "Refer to the EasyAI documentation for more information."
        )
