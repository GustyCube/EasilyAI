from setuptools import setup, find_packages
from pathlib import Path
this_directory = Path(__file__).parent

long_description = (this_directory / "README.md").read_text()

setup(
    name="EasilyAI",
    version="0.1.9",
    description="A library that simplifies the usage of AI!",
    author="GustyCube",
    author_email="gc@gustycube.xyz",
    url="https://github.com/GustyCube/EasilyAI",
    packages=find_packages(),
    install_requires=[
        "openai>=1.0.0",
        "requests>=2.0.0",
        "google-generativeai>=0.8.3",
        "anthropic>=0.42.0"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
    long_description=long_description,
    long_description_content_type='text/markdown'
)
