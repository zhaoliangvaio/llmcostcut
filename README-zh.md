<p align="center">
  <img src="assets/llmcompiler_logo.png" alt="LLMCompiler Logo" width="600"/>
</p>

# LLMCompiler：面向大语言模型的智能编译器

**优化。高效。强大。**

[![GitHub](https://img.shields.io/badge/GitHub-Project-181717?logo=github)](https://github.com/zhaoliangvaio/llmcompiler)
[![arXiv](https://img.shields.io/badge/arXiv-2602.03006-red?logo=arxiv)](https://arxiv.org/abs/2602.03006)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyPI](https://img.shields.io/pypi/v/llmcompiler?color=blue)](https://pypi.org/project/llmcompiler/)
<!-- [![PyPI Downloads](https://img.shields.io/pypi/dm/llmcompiler?color=green)](https://pypi.org/project/llmcompiler/) -->

[![中文](https://img.shields.io/badge/中文-README--zh.md-green)](README-zh.md)
[![English](https://img.shields.io/badge/English-README.md-blue)](README.md)

---

LLMCompiler 是一个用于**按需调用 LLM** 并将**重复工作负载蒸馏**到更小的学生模型中的框架。仅在必要时使用教师 LLM —— 随着学生模型的成长，LLM 调用趋近于零，同时保持准确率。

> 💰 **节省 API 成本。** 随时间推移将 LLM 调用降至趋近于零。由学生模型处理绝大多数查询，仅在不确定时使用教师模型，您可在保持约 95% 准确率的同时显著降低推理费用。

## 🎯 核心特性

| 🎯 **正确性感知回退** | 📚 **在线蒸馏** | 🔄 **回放缓冲训练** |
|----------------------------------|---------------------------|-------------------------------|
| 仅当学生模型不确定（低于置信度阈值）时使用教师 LLM | 持续从回退样本上的教师标签更新学生模型 | 存储回退（文本、标签）对，定期重训学生模型与正确性预测器 |

---

## 目录

- 🎯 [核心特性](#核心特性)
- 📦 [安装](#安装)
  - 📋 [依赖要求](#依赖要求)
- 🚀 [快速开始](#快速开始)
- ⚙️ [工作原理](#工作原理)
- 📊 [实验结果](#实验结果)
  - 📉 [回退机制](#回退机制)
  - 📈 [准确率](#准确率)
  - 💰 [成本降低](#成本降低)
- 📁 [项目结构](#项目结构)
- 📚 [API 参考](#api-参考)
  - 🔧 [monitor()](#monitor---主要参数)
  - ⏳ [wait_for_pending_training()](#wait_for_pending_training)
- 📐 [概念图](#概念图)
- 📖 [引用](#引用)
- 🙏 [致谢](#致谢)

---

## 📦 安装

**从源码安装**（推荐用于开发）

```bash
git clone https://github.com/emory-llmcompiler/llmcompiler.git
cd llmcompiler
pip install -e .
```

**从 PyPI 安装**

```bash
pip install llmcompiler
```

### 📋 依赖要求

- Python >= 3.8
- PyTorch >= 1.9.0
- Transformers >= 4.20.0

可选（用于内置 OpenAI 教师和示例）：

- `openai>=1.0.0`
- `python-dotenv>=1.0.0`
- `pydantic>=2.0.0`

使用内置 OpenAI 教师时，请设置 API 密钥：

```bash
export OPENAI_API_KEY="your-key-here"
```

或将 `OPENAI_API_KEY=your-key-here` 放入项目根目录的 `.env` 文件中（若已安装 `python-dotenv`，将自动加载）。



## 🚀 快速开始

```python
from llmcompiler.monitor import monitor, wait_for_pending_training

# 定义任务与类别
task_id2classes = {"topic": ["Positive", "Negative"]}

# 单条文本
results, fallback = monitor(
    task_id2classes,
    "I love this product!", # 在此填入您的输入文本
    mode="online",  # 或 "offline"
    p_threshold=0.8,
)

print(results)   # {"topic": "Positive"}
print(fallback) # True 表示使用了 LLM，False 表示学生模型预测

# 使用 online 模式时，退出前请等待后台训练完成
wait_for_pending_training()
```

参见 **[examples/example.py](examples/example.py)** 获取完整的 AG-News 演示，包括 GCP 分类器与概念级标签。

---

## ⚙️ 工作原理

1. **学生预测** — 使用小模型对输入进行分类。
2. **置信度** — 估计预测的可靠性。
3. **回退** — 若置信度 ≥ `p_threshold`，返回学生输出；否则调用教师 LLM。
4. **在线学习** — 将（文本、教师标签）存入回放缓冲，定期重训学生模型与正确性预测器。对于 GCP，可选子模块重训可细化选定的概念节点。

---

## 📊 实验结果

### 💰 成本降低

![Cost Curve](cost_curve.png)

随着学生模型处理更多查询，回退到教师 LLM 的频率降低，推理成本随时间下降。

### 📉 回退机制

![Fallback](fallback.png)

随着学生模型能力提升，回退比例（教师/LLM 利用率）随迭代次数降低。初期系统依赖教师保证正确性；随着在线蒸馏推进，学生模型处理更多查询，约在第 100 次迭代时 LLM 调用趋近于零。

### 📈 准确率

![Accuracy](acc.png)

训练期间系统准确率接近教师基线（100%）。尽管 LLM 回退次数下降，准确率在初始阶段后稳定在约 95%，表明蒸馏后的学生模型在降低推理成本的同时保持了质量。



---

## 📁 项目结构

```
llmcompiler/
├── src/
│   └── llmcompiler/
│       __init__.py       # 导出 monitor
│       buffers.py       # 回放缓冲（RingBuffer，均衡采样）
│       correctness.py   # 正确性预测器
│       defaults.py      # 编码器、分词器、优化器默认配置
│       models.py        # 分类头（DeepMLP、GCP）
│       monitor.py       # 核心 monitor() API 及训练编排
│       registry.py      # 每任务注册表
│       selector.py      # 主动学习 / 离线样本选择
│       task.py          # 任务抽象
│       trainer.py       # 训练及 GCP 子模块重训
├── examples/
│   example.py           # AG-News 与 GCP 分类器（概念 DAG + 在线/离线）
├── test_installation.py # 快速安装检查
├── setup.py
├── pyproject.toml
├── requirements.txt
├── README.md
└── LICENSE
```

---

## 📚 API 参考

### 🔧 `monitor(...)` — 主要参数

#### 必填

- **`task_id2classes`** (`dict[str, list[str]]`): 任务 ID → 允许的类别标签列表。
  ```python
  {"task1": ["class1", "class2"], "task2": ["A", "B", "C"]}
  ```
- **`text`** (`str | list[str] | tuple[str, ...]`): 输入文本。单字符串为单样本，list/tuple 为批量。
- **`mode`** (str): **必填。** `"online"` 或 `"offline"`。
  - `"online"`: 推理期间持续更新学生模型。
  - `"offline"`: 学生模型冻结；通过 `offline_select_method` 和 `offline_select_budget` 预先选择样本。

#### 可选参数

| 参数 | 默认值 | 说明 |
|-----------|---------|-------------|
| `llm_fn` | 内置 OpenAI | 教师函数。自定义签名：`llm_fn(texts, task_id2classes, **kwargs)`。 |
| `p_threshold` | `0.8` | 信任学生模型的最小置信度；低于此值则使用 LLM。 |
| `classifier_type` | `"deep_mlp"` | 学生分类头：`"deep_mlp"` 或 `"gcp"`。 |
| `classifier_kwargs` | `None` | 架构特定 kwargs（见下文）。 |
| `encoder` | `distilbert-base-uncased` | 编码器模型实例。 |
| `device` | auto | 设备（如 `"cuda:0"`、`"cpu"`）。 |
| `llm_kwargs` | `{}` | 传递给 `llm_fn`。对于默认 OpenAI 教师，可包含 GCP 概念标签的 `concept_info`。 |

#### 仅离线模式参数

| 参数 | 说明 |
|-----------|-------------|
| `offline_select_method` | `mode="offline"` 时**必填**。策略（如 `"random"`、`"uncertainty"`）。 |
| `offline_select_budget` | `mode="offline"` 时**必填**。发送给教师的最大样本数。 |

#### 返回值

返回 2 元组 `(results, fallback)`：

| 输入 | `results` | `fallback` |
|-------|------------|------------|
| 单个 `str` | `dict[str, str]` | `bool` |
| list/tuple str | `list[dict[str, str]]` | `list[bool]` |

### ⏳ `wait_for_pending_training()`

使用 `mode="online"` 时，训练在后台线程中运行。在退出或评估学生模型之前，请调用：

```python
from llmcompiler.monitor import wait_for_pending_training
wait_for_pending_training()
```

---

## 📐 概念图

**概念图（Graph of Concepts，GCP）** 是我们论文中提出的推理感知蒸馏架构。它将教师 LLM 的推理过程建模为概念节点的**有向无环图（DAG）**。每个节点维护一个概念嵌入和节点特定的预测器；信息通过可学习的转移函数从父节点传播到子节点，汇节点产生最终任务预测。通过将学生模型建模为概念图，GCP 在实现高效蒸馏的同时保留可解释的中间推理，并支持对选定概念节点的可选子模块重训。

---

## 📖 引用

如在研究中使用本框架，请引用：

```bibtex
@article{yu2026distilling,
  title     = {Distilling {LLM} Reasoning into Graph of Concept Predictors},
  author    = {Ziyang Yu and Liang Zhao},
  journal   = {arXiv preprint arXiv:2602.03006},
  year      = {2026},
  url       = {https://arxiv.org/abs/2602.03006},
  doi       = {10.48550/arXiv.2602.03006},
}
```

---

## 🙏 致谢

本框架由埃默里大学 Prof. Liang Zhao 领导的团队开发。感谢合作者与学生的反馈。设计借鉴了在线学习、知识蒸馏和自适应推理等领域的工作。
