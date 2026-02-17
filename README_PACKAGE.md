# LLMCompiler Python Package

LLMCompiler is configured as a standalone Python package and is ready to install and use.

## Quick Install

### Option 1: Install from parent directory (recommended)

```bash
cd /scratch1/zyu273/research-pilot
pip install -e ./llmcompiler
```

### Option 2: Install from the `llmcompiler` directory

```bash
cd llmcompiler
pip install -e .
```

### Option 3: Install development dependencies

```bash
pip install -e ./llmcompiler[dev]
```

## Verify Installation

```python
from llmcompiler import monitor
print("✅ LLMCompiler installed successfully!")
```

## Usage Example

```python
from llmcompiler import monitor

# Define your LLM function
def my_llm(text, task_id2classes, **kwargs):
    # Your LLM call logic
    return {"sentiment": "positive"}

# Run
results, fallback = monitor(
    task_id2classes={"sentiment": ["positive", "negative"]},
    text="I love this!",
    llm_fn=my_llm
)
```

For more examples, see `example_usage.py` and `QUICKSTART.md`.

## Package Structure

```
llmcompiler/
├── __init__.py          # Package initialization, exports monitor
├── monitor.py           # Core API
├── task.py              # Task abstraction
├── models.py            # Model definitions
├── buffers.py           # Replay buffer
├── correctness.py       # Correctness predictor
├── trainer.py           # Training functions
├── registry.py          # Task registry
├── defaults.py          # Default configuration
├── setup.py             # Installation script
├── pyproject.toml       # Modern Python package config
├── requirements.txt     # Dependency list
├── README.md            # Project overview
├── QUICKSTART.md        # Quick start guide
└── INSTALL.md           # Detailed install guide
```

## Dependencies

- Python >= 3.8
- PyTorch >= 1.9.0
- Transformers >= 4.20.0

## Build and Publish (Optional)

To publish to PyPI:

```bash
# Install build tools
pip install build twine

# Build package
python -m build

# Upload to PyPI (requires configuration)
twine upload dist/*
```

## Uninstall

```bash
pip uninstall llmcompiler
```

