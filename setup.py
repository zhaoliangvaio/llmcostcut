"""
LLMCostCut installation script.

Install in development mode (recommended):
    pip install -e .

From repository root:
    cd /path/to/llmcostcut
    pip install -e .
"""
from setuptools import setup, find_packages
from pathlib import Path

readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""

setup(
    name="llmcostcut",
    version="1.0.4",
    description="A framework for selectively invoking LLMs and distilling repeated workloads into smaller models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Liang Zhao",
    author_email="liang.zhao@emory.edu",
    url="https://github.com/zhaoliangvaio/llmcostcut",
    license="Apache 2.0",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
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
