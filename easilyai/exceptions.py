import sys

class EasilyAIError(Exception):
    """Base class for all EasyAI exceptions."""
    pass

# ANSI Color Codes
class Color:
    RESET = "\033[0m"
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"

class AuthenticationError(EasilyAIError):
    def __init__(self, message="Authentication failed!"):
        super().__init__(f"{Color.RED}🔑 {message} {Color.RESET}")

class RateLimitError(EasilyAIError):
    def __init__(self, message="API rate limit exceeded! Please slow down."):
        super().__init__(f"{Color.YELLOW}⏳ {message} {Color.RESET}")

class InvalidRequestError(EasilyAIError):
    def __init__(self, message="Invalid request!"):
        super().__init__(f"{Color.RED}🚫 {message} {Color.RESET}")

class APIConnectionError(EasilyAIError):
    def __init__(self, message="Unable to connect to the API."):
        super().__init__(f"{Color.CYAN}🌐 {message} {Color.RESET}")

class NotFoundError(EasilyAIError):
    def __init__(self, message="The requested resource was not found!"):
        super().__init__(f"{Color.YELLOW}🔍 {message} {Color.RESET}")

class ServerError(EasilyAIError):
    def __init__(self, message="Server encountered an error!"):
        super().__init__(f"{Color.RED}💥 {message} {Color.RESET}")

class MissingAPIKeyError(EasilyAIError):
    def __init__(self, message="No API key provided!"):
        super().__init__(f"{Color.RED}🔐 {message} {Color.RESET}")

class UnsupportedServiceError(EasilyAIError):
    def __init__(self, service_name):
        super().__init__(
            f"{Color.BLUE}❌ Unsupported service '{service_name}'! Use 'openai', 'ollama', or a custom registered service. "
            f"Refer to the EasyAI documentation for more information.{Color.RESET}"
        )

class NotImplementedError(EasilyAIError):
    def __init__(self, feature="This feature"):
        message = f"{Color.CYAN}🛠️ {feature} is not yet implemented! Stay tuned for future updates.{Color.RESET}"
        print(message)
        sys.exit(1)
