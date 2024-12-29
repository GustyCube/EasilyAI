import os

def save_to_file(data, filename="output.txt"):
    """
    Save text data to a file.
    """
    with open(filename, "w") as file:
        file.write(data)
    return f"Data saved to {os.path.abspath(filename)}"

def validate_service(service_name):
    """
    Validate the service name input.
    """
    valid_services = ["openai", "ollama", "grok", "gemini", "claude"]
    if service_name not in valid_services:
        raise ValueError(f"Invalid service: {service_name}. Supported: {valid_services}")

def print_banner(app_name):
    """
    Display a simple banner when the app is created.
    """
    banner = f"""
    *************************************
            {app_name} Initialized
    *************************************
    """
    print(banner)
