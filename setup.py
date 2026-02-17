"""
LLMCompiler installation script.

Recommended installation (from parent directory):
    cd /scratch1/zyu273/research-pilot
    pip install -e ./llmcompiler

Or install from the current directory:
    cd llmcompiler
    pip install -e .
"""

from setuptools import setup
from pathlib import Path

# Read README
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""

setup(
    name="llmcompiler",
    version="0.1.0",
    description="A framework for selectively invoking LLMs and distilling repeated workloads into smaller models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Liang Zhao",
    author_email="liang.zhao@emory.edu",
    url="https://github.com/emory-llmcompiler/llmcompiler",
    license="Apache 2.0",
    # Package name is llmcompiler; package contents are in this directory
    packages=["llmcompiler"],
    package_dir={"llmcompiler": "."},
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.9.0",
        "transformers>=4.20.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
            "mypy>=1.0.0",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    keywords=[
        "llm",
        "distillation",
        "knowledge-distillation",
        "online-learning",
        "adaptive-inference",
        "machine-learning",
    ],
    include_package_data=True,
    zip_safe=False,
)
