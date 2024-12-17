from setuptools import setup, find_packages

setup(
    name="easyai",
    version="0.1.0",
    description="A library that simplifies the usage of AI!",
    author="GustyCube",
    author_email="gc@gustycube.xyz",
    url="https://github.com/GustyCube/EasyAI",
    packages=find_packages(),
    install_requires=[
        "openai>=1.0.0",
        "requests>=2.0.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
)
