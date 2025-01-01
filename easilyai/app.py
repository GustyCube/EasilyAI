from easilyai.services.openai_service import OpenAIService
from easilyai.services.ollama_service import OllamaService
from easilyai.services.gemini_service import GeminiService
from easilyai.services.grok_service import GrokService
from easilyai.services.anthropic_service import AnthropicService
from easilyai.custom_ai import CustomAIService
from easilyai.exceptions import UnsupportedServiceError, NotImplementedError

_registered_custom_ais = {}

class EasyAIApp:
    def __init__(self, name, service, apikey=None, model=None, max_tokens=None):
        self.name = name
        self.service = service
        self.model = model
        self.client = self._initialize_client(service, apikey, model, max_tokens)

    def _initialize_client(self, service, apikey, model, max_tokens):
        service_clients = {
            "openai": OpenAIService(apikey, model),
            "ollama": OllamaService(model),
            "gemini": GeminiService(apikey, model),
            "grok": GrokService(apikey, model),
            "anthropic": AnthropicService(apikey, model, max_tokens) if max_tokens else AnthropicService(apikey, model)
        }
        if service in service_clients:
            return service_clients[service]
        elif service in _registered_custom_ais:
            return _registered_custom_ais[service](model, apikey)
        else:
            raise UnsupportedServiceError(
                f"Unsupported service '{service}'! Use 'openai', 'ollama', or a registered custom service. "
                "Refer to the EasyAI documentation for more information."
            )

    def request(self, task_type, task):
        task_methods = {
            "generate_text": self._generate_text,
            "generate_image": self.client.generate_image,
            "text_to_speech": self.client.text_to_speech
        }
        if task_type in task_methods:
            return task_methods[task_type](task)
        else:
            raise ValueError(f"Unsupported task type: {task_type}")

    def _generate_text(self, task):
        if isinstance(task, dict):
            return self.client.generate_text(task["data"], task.get("image_url"))
        return self.client.generate_text(task)


class EasyAITTSApp:
    def __init__(self, name, service, apikey=None, model=None):
        self.name = name
        self.service = service
        self.model = model
        self.client = self._initialize_client(service, apikey, model)

    def _initialize_client(self, service, apikey, model):
        if service == "openai":
            return OpenAIService(apikey, model)
        elif service in _registered_custom_ais:
            return _registered_custom_ais[service](model, apikey)
        else:
            raise ValueError("Unsupported service for TTS. Use 'openai' or a registered custom service.")

    def request_tts(self, text, tts_model="tts-1", voice="onyx", output_file="output.mp3"):
        if hasattr(self.client, "text_to_speech"):
            return self.client.text_to_speech(text, tts_model=tts_model, voice=voice, output_file=output_file)
        else:
            raise NotImplementedError("TTS is not supported for this service.")


def create_app(name, service, apikey=None, model=None):
    return EasyAIApp(name, service, apikey, model)

def create_tts_app(name, service, apikey=None, model=None):
    return EasyAITTSApp(name, service, apikey, model)
