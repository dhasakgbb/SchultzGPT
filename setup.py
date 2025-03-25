#!/usr/bin/env python
"""
Setup script for SchultzGPT.
"""

from setuptools import setup, find_packages

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="schultzgpt",
    version="0.1.0",
    description="A terminal-based AI persona chatbot with vector store memory",
    author="Damian",
    package_dir={"": "src"},
    packages=["config", "controllers", "models", "services", "ui"],
    install_requires=requirements,
    python_requires=">=3.8",
    entry_points={
        "console_scripts": [
            "schultzgpt=run:main",
        ],
    },
) 