from setuptools import setup, find_packages

setup(
    name="weaviate-middleware",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "weaviate-client>=3.26.0",
        "numpy>=1.24.0",
        "pytest>=7.3.1",
        "pytest-asyncio>=0.21.0"
    ],
    extras_require={
        "embeddings": [
            "sentence-transformers>=2.2.0",
            "openai>=1.3.0",
            "cohere>=4.32"
        ]
    }
) 