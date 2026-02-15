# LLMCompiler 安装指南

## 安装方式

### 方式 1: 从本地目录安装（开发模式）

```bash
cd llmcompiler
pip install -e .
```

或者使用 `-e` 标志进行可编辑安装（推荐用于开发）：

```bash
pip install -e .
```

### 方式 2: 从本地目录安装（普通模式）

```bash
cd llmcompiler
pip install .
```

### 方式 3: 从父目录安装

如果你在 `research-pilot` 目录下：

```bash
pip install -e ./llmcompiler
```

### 方式 4: 安装开发依赖

```bash
pip install -e .[dev]
```

或者：

```bash
pip install -e ".[dev]"
```

## 验证安装

安装完成后，可以验证是否成功：

```python
from llmcompiler import monitor
print("LLMCompiler 安装成功！")
```

## 依赖要求

- Python >= 3.8
- PyTorch >= 1.9.0
- Transformers >= 4.20.0

## 卸载

```bash
pip uninstall llmcompiler
```

## 常见问题

### 问题 1: 找不到模块

如果遇到 `ModuleNotFoundError`，确保：
1. 已正确安装包
2. 使用的 Python 环境正确
3. 尝试使用 `pip install -e .` 进行可编辑安装

### 问题 2: 依赖冲突

如果遇到依赖版本冲突：
1. 检查 PyTorch 和 Transformers 版本
2. 考虑使用虚拟环境：
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   # 或
   venv\Scripts\activate  # Windows
   pip install -e .
   ```

### 问题 3: CUDA 相关问题

如果使用 GPU，确保安装了对应版本的 PyTorch：
```bash
# CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

