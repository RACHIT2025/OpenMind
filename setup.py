from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="openmind",
    version="0.1.0",
    author="OpenMind Team",
    description="An open-source LLM training and serving framework",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/RACHIT2025/openmind",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.1.0",
        "transformers>=4.35.0",
        "datasets>=2.14.0",
        "tokenizers>=0.14.0",
        "accelerate>=0.24.0",
        "fastapi>=0.104.0",
        "uvicorn>=0.24.0",
        "pydantic>=2.5.0",
        "numpy>=1.24.0",
        "tqdm>=4.66.0",
        "pyyaml>=6.0.1",
        "typer>=0.9.0",
        "rich>=13.7.0",
        "regex>=2023.10.3",
    ],
    entry_points={
        "console_scripts": [
            "openmind=cli.main:app",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
