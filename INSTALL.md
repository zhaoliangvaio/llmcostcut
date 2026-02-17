# LLMCompiler Installation Guide

## Installation Options

### Option 1: Install from local directory (development mode)

```bash
cd llmcompiler
pip install -e .
```

Use editable mode (`-e`) for development (recommended).

### Option 2: Install from local directory (standard mode)

```bash
cd llmcompiler
pip install .
```

### Option 3: Install from parent directory

If you are in the `research-pilot` directory:

```bash
pip install -e ./llmcompiler
```

### Option 4: Install development dependencies

```bash
pip install -e .[dev]
```

Or:

```bash
pip install -e ".[dev]"
```

## Verify Installation

After installation, verify with:

```python
from llmcompiler import monitor
print("LLMCompiler installed successfully!")
```

## Requirements

- Python >= 3.8
- PyTorch >= 1.9.0
- Transformers >= 4.20.0

## Uninstall

```bash
pip uninstall llmcompiler
```

## FAQ

### Issue 1: Module not found

If you see `ModuleNotFoundError`, make sure:
1. The package is installed correctly.
2. You are using the correct Python environment.
3. Try editable install: `pip install -e .`.

### Issue 2: Dependency conflicts

If dependency versions conflict:
1. Check your PyTorch and Transformers versions.
2. Consider using a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   # or
   venv\Scripts\activate  # Windows
   pip install -e .
   ```

### Issue 3: CUDA-related issues

If using GPU, install a matching PyTorch build:
```bash
# CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

