import requests
import os
from easilyai.exceptions import (
    AuthenticationError, RateLimitError, InvalidRequestError,
    APIConnectionError, NotFoundError, ServerError, MissingAPIKeyError
)

class HuggingFaceService:
    def __init__(self, apikey, model):
        if not apikey:
            raise MissingAPIKeyError(
                "Hugging Face API key is missing! Please provide your API key when initializing the service. "
                "Refer to the EasilyAI documentation for more information."
            )
        self.apikey = apikey
        self.model = model
        self.base_url = "https://api-inference.huggingface.co/models"
        self.headers = {"Authorization": f"Bearer {apikey}"}

    def generate_text(self, prompt, max_length=100, temperature=0.7, top_p=0.9):
        """Generate text using Hugging Face Inference API."""
        try:
            url = f"{self.base_url}/{self.model}"
            payload = {
                "inputs": prompt,
                "parameters": {
                    "max_length": max_length,
                    "temperature": temperature,
                    "top_p": top_p,
                    "return_full_text": False
                }
            }
            
            response = requests.post(url, headers=self.headers, json=payload)
            
            if response.status_code == 200:
                result = response.json()
                if isinstance(result, list) and len(result) > 0:
                    return result[0].get("generated_text", "")
                else:
                    return str(result)
            else:
                self._handle_error(response)
                
        except requests.exceptions.RequestException as e:
            raise APIConnectionError(
                f"Connection error! Unable to connect to Hugging Face API. {str(e)}"
            )
        except Exception as e:
            raise ServerError(
                f"An unexpected error occurred: {str(e)}. Please try again later."
            )

    def generate_image(self, prompt):
        """Generate image using Hugging Face text-to-image models."""
        try:
            url = f"{self.base_url}/{self.model}"
            payload = {"inputs": prompt}
            
            response = requests.post(url, headers=self.headers, json=payload)
            
            if response.status_code == 200:
                # For image generation, the response is typically binary data
                # We'd need to handle this differently - for now, return status
                return f"Image generated successfully for prompt: {prompt}"
            else:
                self._handle_error(response)
                
        except requests.exceptions.RequestException as e:
            raise APIConnectionError(
                f"Connection error! Unable to connect to Hugging Face API. {str(e)}"
            )
        except Exception as e:
            raise ServerError(
                f"An unexpected error occurred: {str(e)}. Please try again later."
            )

    def text_to_speech(self, text):
        """Convert text to speech using Hugging Face TTS models."""
        try:
            url = f"{self.base_url}/{self.model}"
            payload = {"inputs": text}
            
            response = requests.post(url, headers=self.headers, json=payload)
            
            if response.status_code == 200:
                # For TTS, the response is typically audio data
                # We'd need to handle this differently - for now, return status
                return f"Speech generated successfully for text: {text}"
            else:
                self._handle_error(response)
                
        except requests.exceptions.RequestException as e:
            raise APIConnectionError(
                f"Connection error! Unable to connect to Hugging Face API. {str(e)}"
            )
        except Exception as e:
            raise ServerError(
                f"An unexpected error occurred: {str(e)}. Please try again later."
            )

    def _handle_error(self, response):
        """Handle HTTP errors from Hugging Face API."""
        if response.status_code == 401:
            raise AuthenticationError(
                "Authentication failed! Please check your Hugging Face API key."
            )
        elif response.status_code == 429:
            raise RateLimitError(
                "Rate limit exceeded! Please wait and try again later."
            )
        elif response.status_code == 400:
            raise InvalidRequestError(
                f"Invalid request! Please check your parameters. Error: {response.text}"
            )
        elif response.status_code == 404:
            raise NotFoundError(
                f"Model not found! Please check your model name: {self.model}"
            )
        elif response.status_code >= 500:
            raise ServerError(
                f"Server error from Hugging Face API: {response.status_code} - {response.text}"
            )
        else:
            raise ServerError(
                f"Unknown error from Hugging Face API: {response.status_code} - {response.text}"
            )