"""Setup script for Counsel of Agents."""

from setuptools import setup, find_packages

setup(
    name="counsel-of-agents",
    version="0.1.0",
    description="Multi-agent orchestration system with specialized agents",
    author="Your Name",
    packages=find_packages(),
    install_requires=[
        "transformers>=4.35.0",
        "torch>=2.0.0",
        "accelerate>=0.24.0",
        "sentencepiece>=0.1.99",
        "bitsandbytes>=0.41.0",
        "rich>=13.0.0",
    ],
    entry_points={
        "console_scripts": [
            "agent=main:main",
        ],
    },
    python_requires=">=3.8",
)
