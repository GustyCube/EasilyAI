class CustomAIService:
    def __init__(self, model, apikey=None):
        self.model = model
        self.apikey = apikey

    def generate_text(self, prompt):
        raise NotImplementedError("Custom AI services must implement 'generate_text'.")

    def generate_image(self, prompt):
        raise NotImplementedError("Custom AI services must implement 'generate_image'.")

    def text_to_speech(self, text):
        raise NotImplementedError("Custom AI services must implement 'text_to_speech'.")

_registered_custom_ais = {}

def register_custom_ai(name, custom_service_class):
    """
    Register a custom AI service.

    :param name: Name of the custom service.
    :param custom_service_class: Class inheriting from CustomAIService.
    """
    if not issubclass(custom_service_class, CustomAIService):
        raise TypeError("Custom service must inherit from CustomAIService.")
    _registered_custom_ais[name] = custom_service_class
