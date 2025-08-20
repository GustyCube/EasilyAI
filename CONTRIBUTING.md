# Contributing to EasilyAI

Thank you for your interest in contributing to EasilyAI! We welcome contributions from the community and are grateful for any help you can provide.

## Code of Conduct

Please note that this project is released with a [Code of Conduct](CODE_OF_CONDUCT.md). By participating in this project you agree to abide by its terms.

## How to Contribute

### Reporting Bugs

Before creating bug reports, please check existing issues to avoid duplicates. When you create a bug report, include as many details as possible:

- **Use a clear and descriptive title**
- **Describe the exact steps to reproduce the problem**
- **Provide specific examples to demonstrate the steps**
- **Describe the behavior you observed and what you expected**
- **Include screenshots if relevant**
- **Include your environment details** (OS, Python version, package versions)

### Suggesting Enhancements

Enhancement suggestions are tracked as GitHub issues. When creating an enhancement suggestion:

- **Use a clear and descriptive title**
- **Provide a detailed description of the suggested enhancement**
- **Provide specific examples to demonstrate the enhancement**
- **Describe the current behavior and the expected behavior**
- **Explain why this enhancement would be useful**

### Pull Requests

1. **Fork the repository** and create your branch from `main`
2. **Set up your development environment**:
   ```bash
   git clone https://github.com/yourusername/EasilyAI.git
   cd EasilyAI
   make dev-setup
   ```

3. **Make your changes**:
   - Write clear, commented code
   - Follow the existing code style
   - Add or update tests as needed
   - Update documentation as needed

4. **Run quality checks**:
   ```bash
   # Format your code
   make format
   
   # Run linting
   make lint
   
   # Run tests
   make test
   
   # Run all checks
   make check
   ```

5. **Commit your changes**:
   - Use clear and meaningful commit messages
   - Follow conventional commit format when possible:
     - `feat:` for new features
     - `fix:` for bug fixes
     - `docs:` for documentation changes
     - `test:` for test changes
     - `refactor:` for code refactoring
     - `chore:` for maintenance tasks

6. **Push to your fork** and submit a pull request

7. **Pull Request Guidelines**:
   - Provide a clear description of the problem and solution
   - Include the relevant issue number if applicable
   - Make sure all tests pass
   - Update documentation as needed
   - Request review from maintainers

## Development Setup

### Prerequisites

- Python 3.8 or higher
- pip
- git

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/GustyCube/EasilyAI.git
   cd EasilyAI
   ```

2. Install development dependencies:
   ```bash
   make dev-setup
   ```

   Or manually:
   ```bash
   pip install -e ".[dev,test,docs]"
   pre-commit install
   ```

### Running Tests

```bash
# Run all tests
make test

# Run unit tests only
make test-unit

# Run with coverage
make test-cov

# Run specific test file
pytest tests/test_openai_service.py

# Run specific test
pytest tests/test_openai_service.py::TestOpenAIService::test_generate_text
```

### Code Quality

We use several tools to maintain code quality:

- **black**: Code formatting
- **isort**: Import sorting
- **flake8**: Linting
- **mypy**: Type checking
- **bandit**: Security checking

Run all quality checks:
```bash
make check
```

### Documentation

Build documentation locally:
```bash
make docs
```

Serve documentation locally:
```bash
make serve-docs
```

Documentation will be available at http://localhost:8000

## Project Structure

```
EasilyAI/
├── easilyai/          # Main package
│   ├── services/      # AI service implementations
│   ├── utils/         # Utility functions
│   └── exceptions.py  # Custom exceptions
├── tests/             # Test suite
├── docs/              # Documentation
├── examples/          # Example scripts
└── .github/           # GitHub workflows
```

## Adding a New AI Service

To add support for a new AI service:

1. Create a new service file in `easilyai/services/`
2. Implement the base service interface
3. Add tests in `tests/`
4. Update documentation
5. Add example usage in `examples/`

Example service implementation:

```python
from easilyai.services.base import BaseService

class NewAIService(BaseService):
    def __init__(self, api_key: str, model: str):
        self.api_key = api_key
        self.model = model
        # Initialize client
    
    def generate_text(self, prompt: str, **kwargs):
        # Implement text generation
        pass
    
    def chat_complete(self, messages: list, **kwargs):
        # Implement chat completion
        pass
```

## Release Process

Releases are automated through GitHub Actions when a tag is pushed:

```bash
# Create a new tag
git tag -a v1.0.0 -m "Release version 1.0.0"

# Push the tag
git push origin v1.0.0
```

## Getting Help

If you need help, you can:

- Open an issue on GitHub
- Check existing documentation
- Look at existing code examples
- Ask questions in discussions

## Recognition

Contributors will be recognized in:
- The project's README
- Release notes
- GitHub contributors page

Thank you for contributing to EasilyAI!