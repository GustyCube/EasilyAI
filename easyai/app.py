from easyai.services.openai_service import OpenAIService
from easyai.services.ollama_service import OllamaService
from easyai.custom_ai import CustomAIService
from easyai.exceptions import UnsupportedServiceError, NotImplementedError

_registered_custom_ais = {}

class EasyAIApp:
    def __init__(self, name, service, apikey=None, model=None):
        self.name = name
        self.service = service
        self.model = model
        self.client = None

        if service == "openai":
            self.client = OpenAIService(apikey, model)
        elif service == "ollama":
            self.client = OllamaService(model)
        elif service in _registered_custom_ais:
            self.client = _registered_custom_ais[service](model, apikey)
        else:
            raise UnsupportedServiceError(
                f"Unsupported service '{service}'! Use 'openai', 'ollama', or a registered custom service. "
                "Refer to the Easy ::contentReference[oaicite:0]{index=0}")
    
    def request(self, task):
        if "image" in task.lower():
            return self.client.generate_image(task)
        elif "speech" in task.lower() or "convert text to speech" in task.lower():
            return self.client.text_to_speech(task)
        else:
            return self.client.generate_text(task)

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
