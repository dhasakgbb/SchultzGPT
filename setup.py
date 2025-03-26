#!/usr/bin/env python
"""
Setup script for SchultzGPT.
"""

from setuptools import setup, find_packages

setup(
    name="schultzgpt",
    version="2.0.0",
    description="A terminal-based AI persona chatbot with OpenAI Retrieval API memory",
    author="Damian Schultz",
    author_email="damianschultz@gmail.com",
    packages=find_packages(),
    install_requires=[
        "openai>=1.0.0",
        "python-dotenv>=0.19.0",
        "tqdm>=4.65.0",
        "tabulate>=0.9.0",
        "psutil>=5.9.0",
        "tiktoken>=0.5.0"
    ],
    python_requires=">=3.8",
    entry_points={
        "console_scripts": [
            "schultzgpt=src.main:main",
        ],
    },
) 