"""
Pytest configuration and shared fixtures for all tests.
"""
import os
import sys
from pathlib import Path
from unittest.mock import Mock, MagicMock
import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture
def mock_api_key():
    """Provide a mock API key for testing."""
    return "test-api-key-123"


@pytest.fixture
def mock_openai_client():
    """Mock OpenAI client for testing."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_choice = MagicMock()
    mock_message = MagicMock()
    mock_message.content = "Mocked OpenAI response"
    mock_choice.message = mock_message
    mock_response.choices = [mock_choice]
    mock_client.chat.completions.create.return_value = mock_response
    return mock_client


@pytest.fixture
def mock_anthropic_client():
    """Mock Anthropic client for testing."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.content = [MagicMock(text="Mocked Anthropic response")]
    mock_client.messages.create.return_value = mock_response
    return mock_client


@pytest.fixture
def mock_gemini_client():
    """Mock Google Gemini client for testing."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.text = "Mocked Gemini response"
    mock_client.generate_content.return_value = mock_response
    return mock_client


@pytest.fixture
def mock_groq_client():
    """Mock Groq client for testing."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_choice = MagicMock()
    mock_message = MagicMock()
    mock_message.content = "Mocked Groq response"
    mock_choice.message = mock_message
    mock_response.choices = [mock_choice]
    mock_client.chat.completions.create.return_value = mock_response
    return mock_client


@pytest.fixture
def sample_messages():
    """Sample chat messages for testing."""
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello, how are you?"},
        {"role": "assistant", "content": "I'm doing well, thank you!"},
        {"role": "user", "content": "What's the weather like?"}
    ]


@pytest.fixture
def sample_prompt():
    """Sample prompt for testing."""
    return "Tell me a short story about a robot."


@pytest.fixture
def mock_env_vars(monkeypatch):
    """Mock environment variables for testing."""
    env_vars = {
        "OPENAI_API_KEY": "test-openai-key",
        "ANTHROPIC_API_KEY": "test-anthropic-key",
        "GEMINI_API_KEY": "test-gemini-key",
        "GROQ_API_KEY": "test-groq-key",
    }
    for key, value in env_vars.items():
        monkeypatch.setenv(key, value)
    return env_vars


@pytest.fixture(autouse=True)
def reset_environment(monkeypatch):
    """Reset environment variables after each test."""
    # Store original environment
    original_env = dict(os.environ)
    yield
    # Restore original environment
    os.environ.clear()
    os.environ.update(original_env)


@pytest.fixture
def temp_config_file(tmp_path):
    """Create a temporary configuration file for testing."""
    config_file = tmp_path / "config.json"
    config_data = {
        "api_keys": {
            "openai": "test-key-1",
            "anthropic": "test-key-2",
            "gemini": "test-key-3",
            "groq": "test-key-4"
        },
        "default_models": {
            "openai": "gpt-4",
            "anthropic": "claude-3-opus",
            "gemini": "gemini-pro",
            "groq": "mixtral-8x7b"
        }
    }
    import json
    config_file.write_text(json.dumps(config_data))
    return config_file


@pytest.fixture
def mock_response_stream():
    """Mock streaming response for testing."""
    def stream_generator():
        responses = ["Hello", " ", "world", "!"]
        for response in responses:
            mock_chunk = MagicMock()
            mock_choice = MagicMock()
            mock_delta = MagicMock()
            mock_delta.content = response
            mock_choice.delta = mock_delta
            mock_chunk.choices = [mock_choice]
            yield mock_chunk
    return stream_generator()


class MockException(Exception):
    """Mock exception for testing error handling."""
    pass


@pytest.fixture
def mock_rate_limit_error():
    """Mock rate limit error for testing."""
    return MockException("Rate limit exceeded")


@pytest.fixture
def mock_auth_error():
    """Mock authentication error for testing."""
    return MockException("Invalid API key")


@pytest.fixture
def mock_timeout_error():
    """Mock timeout error for testing."""
    return MockException("Request timeout")


# Markers for test categorization
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "unit: mark test as a unit test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers", "requires_api_key: mark test as requiring API keys"
    )