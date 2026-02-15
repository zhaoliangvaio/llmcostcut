# LLMCompiler Python 包

LLMCompiler 已经配置为独立的 Python 包，可以安装和使用。

## 快速安装

### 方式 1: 从父目录安装（推荐）

```bash
cd /scratch1/zyu273/research-pilot
pip install -e ./llmcompiler
```

### 方式 2: 从 llmcompiler 目录安装

```bash
cd llmcompiler
pip install -e .
```

### 方式 3: 安装开发依赖

```bash
pip install -e ./llmcompiler[dev]
```

## 验证安装

```python
from llmcompiler import monitor
print("✅ LLMCompiler 安装成功！")
```

## 使用示例

```python
from llmcompiler import monitor

# 定义你的 LLM 函数
def my_llm(text, task_id2classes, **kwargs):
    # 你的 LLM 调用逻辑
    return {"sentiment": "positive"}

# 使用
results, fallback = monitor(
    task_id2classes={"sentiment": ["positive", "negative"]},
    text="I love this!",
    llm_fn=my_llm
)
```

更多示例请查看 `example_usage.py` 和 `QUICKSTART.md`。

## 包结构

```
llmcompiler/
├── __init__.py          # 包初始化，导出 monitor
├── monitor.py           # 核心 API
├── task.py              # 任务抽象
├── models.py            # 模型定义
├── buffers.py           # 回放缓冲区
├── correctness.py       # 正确性预测器
├── trainer.py           # 训练函数
├── registry.py          # 任务注册表
├── defaults.py          # 默认配置
├── setup.py             # 安装脚本
├── pyproject.toml       # 现代 Python 包配置
├── requirements.txt     # 依赖列表
├── README.md            # 项目说明
├── QUICKSTART.md        # 快速开始指南
└── INSTALL.md           # 详细安装指南
```

## 依赖

- Python >= 3.8
- PyTorch >= 1.9.0
- Transformers >= 4.20.0

## 构建和发布（可选）

如果你想将包发布到 PyPI：

```bash
# 安装构建工具
pip install build twine

# 构建包
python -m build

# 发布到 PyPI（需要配置）
twine upload dist/*
```

## 卸载

```bash
pip uninstall llmcompiler
```

