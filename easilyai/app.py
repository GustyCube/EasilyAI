from easilyai.services.openai_service import OpenAIService
from easilyai.services.ollama_service import OllamaService
from easilyai.services.gemini_service import GeminiService
from easilyai.services.grok_service import GrokService
from easilyai.services.anthropic_service import AnthropicService
from easilyai.custom_ai import CustomAIService
from easilyai.exceptions import UnsupportedServiceError, NotImplementedError

_registered_custom_ais = {}

class EasyAIApp:
    def __init__(self, name, service, apikey=None, model=None, max_tokens = None):
        self.name = name
        self.service = service
        self.model = model
        self.client = None

        if service == "openai":
            self.client = OpenAIService(apikey, model)
        elif service == "ollama":
            self.client = OllamaService(model)
        elif service == "gemini":
            self.client = GeminiService(apikey, model)
        elif service == "grok":
            self.client = GrokService(apikey, model)
        elif service == "anthropic":
            if max_tokens:
                self.client = AnthropicService(apikey,  model, max_tokens)
            else:
                self.client = AnthropicService(apikey, model)
        elif service in _registered_custom_ais:
            self.client = _registered_custom_ais[service](model, apikey)
        else:
            raise UnsupportedServiceError(
                f"Unsupported service '{service}'! Use 'openai', 'ollama', or a registered custom service. "
                "Refer to the EasyAI documentation for more information."
            )
    
    def request(self, task_type, task):
        # Instead of checking if the task contains "image" or "speech", we should 
        # check if the task_type is "generate_image" or "text_to_speech"
        if task_type == "generate_text":
            # If the task is a dictionary, it contains both prompt and image_url
            # Currently, should only work with Grok, but need to add support for other services
            if isinstance(task, dict):
                prompt = task["data"]
                img_url = task.get("image_url")
                return self.client.generate_text(prompt, img_url)
            else:
                # If the task is a string, it contains only the prompt, works for all services
                return self.client.generate_text(task)
        elif task_type == "generate_image":
            return self.client.generate_image(task)
        elif task_type == "text_to_speech":
            return self.client.text_to_speech(task)
        else:
            raise ValueError(f"Unsupported task type: {task_type}")

class EasyAITTSApp:
    def __init__(self, name, service, apikey=None, model=None):
        self.name = name
        self.service = service
        self.model = model
        self.client = None

        if service == "openai":
            self.client = OpenAIService(apikey, model)
        elif service in _registered_custom_ais:
            self.client = _registered_custom_ais[service](model, apikey)
        else:
            raise ValueError("Unsupported service for TTS. Use 'openai' or a registered custom service.")
    
    def request_tts(self, text, tts_model="tts-1", voice="onyx", output_file="output.mp3"):
        """
        Convert text to speech using the selected service.
        """
        if hasattr(self.client, "text_to_speech"):
            return self.client.text_to_speech(text, tts_model=tts_model, voice=voice, output_file=output_file)
        else:
            raise NotImplementedError("TTS is not supported for this service.")


def create_app(name, service, apikey=None, model=None):
    return EasyAIApp(name, service, apikey, model)

def create_tts_app(name, service, apikey=None, model=None):
    return EasyAITTSApp(name, service, apikey, model)
