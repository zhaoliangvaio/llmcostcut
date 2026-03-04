<p align="center">
  <img src="assets/llmcostcut_logo.png" alt="LLMCostCut Logo" width="600"/>
</p>

# LLMCostCut：大语言模型的智能编译器

**优化。高效。强大。**

[![GitHub](https://img.shields.io/badge/GitHub-Project-181717?logo=github)](https://github.com/zhaoliangvaio/llmcostcut)
[![arXiv](https://img.shields.io/badge/arXiv-2602.03006-red?logo=arxiv)](https://arxiv.org/abs/2602.03006)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyPI](https://img.shields.io/pypi/v/llmcostcut?color=blue)](https://pypi.org/project/llmcostcut/)
<!-- [![PyPI Downloads](https://img.shields.io/pypi/dm/llmcostcut?color=green)](https://pypi.org/project/llmcostcut/) -->

[![中文](https://img.shields.io/badge/中文-README--zh.md-green)](README-zh.md)
[![English](https://img.shields.io/badge/English-README.md-blue)](README.md)

---

LLMCostCut 是一个用于**选择性调用 LLM** 并将**重复工作负载蒸馏**到更小学生模型的框架。仅在需要时使用教师 LLM——随着学生模型的学习，LLM 调用逐渐趋近于零，同时保持准确率。该框架适用于多种工业领域，包括**法律**（如案件结果预测、合同条款分类）、**医疗健康**（如放射报告异常分类、临床笔记编码）、**金融**（如情感分析、风险评估）等。

> 💰 **节省 API 成本。** 随着时间推移将 LLM 调用降至趋近于零。学生模型处理大部分查询，仅在不确定时使用教师模型，在保持约 95% 准确率的同时显著降低推理费用。

## 🎯 核心特性

| 📈 **准确率高达 95%** | 💰 **10× 成本降低** | ⚡ **1000× 加速** |
|---------------------------|---------------------------|----------------------|
| 在将 LLM 调用降至趋近于零的同时保持接近教师的预测质量 | 学生模型处理大部分查询；仅在不确定时使用教师——推理费用大幅下降 | 小型学生模型推理 vs LLM API 调用——数量级的速度提升 |

---

## 目录

- 🎯 [核心特性](#核心特性)
- 📦 [安装](#安装)
  - 📋 [环境要求](#环境要求)
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
git clone https://github.com/emory-llmcostcut/llmcostcut.git
cd llmcostcut
pip install -e .
```

**从 PyPI 安装**

```bash
pip install llmcostcut
```

### 📋 环境要求

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

或将 `OPENAI_API_KEY=your-key-here` 写入项目根目录的 `.env` 文件中（若已安装 `python-dotenv` 会自动加载）。



## 🚀 快速开始

不要每次都直接调用 LLM，而是使用 **`monitor`** 作为智能路由器：它首先尝试小型学生模型；仅当学生模型不确定（置信度低于 `p_threshold`）时才回退到教师 LLM。随着时间推移，学生模型不断学习，LLM 调用逐渐趋近于零。

```python
from llmcostcut.monitor import monitor, wait_for_pending_training

# 定义任务和类别（例如新闻主题：World, Sports, Business, Sci/Tech）
task_id2classes = {"topic": ["World", "Sports", "Business", "Sci/Tech"]}

# monitor 在学生模型（小模型）和 LLM 之间自动切换
results, fallback = monitor(
    task_id2classes,
    "The Nobel physicist's quantum startup now supplies chips to military satellites.",  # 跨 World, Sci/Tech, Business
    mode="online",           # 或 "offline"
    p_threshold=0.8,         # 低于此值：使用 LLM；高于：使用学生模型
)
# 使用 online 模式时，退出前等待后台训练完成
wait_for_pending_training()
```

完整 AG-News 演示（含 GCP 分类器和概念级标签）请参见 **[examples/example.py](examples/example.py)**。

---

## ⚙️ 工作原理

1. **学生预测** — 使用小模型对输入进行分类。
2. **置信度** — 估计预测的可靠性。
3. **回退** — 若置信度 ≥ `p_threshold`，返回学生模型输出；否则调用教师 LLM。
4. **在线学习** — 将（文本，教师标签）存入回放缓冲区，并定期重新训练学生模型和正确性预测器。对于 GCP，可选的子模块重训练会细化选定的概念节点。

---

## 📊 实验结果

### 💰 成本降低

![Cost Curve](cost_curve.png)

随着学生模型处理更多查询且回退到教师 LLM 的频率降低，推理成本随时间下降。

### 📉 回退机制

![Fallback](fallback.png)

回退比例（教师/LLM 利用率）随迭代次数下降，学生模型能力逐渐增强。初期系统依赖教师保证正确性；随着在线蒸馏的进行，学生模型处理更多查询，约在第 100 次迭代时 LLM 调用趋近于零。

### 📈 准确率

![Accuracy](acc.png)

训练期间系统准确率保持接近教师基线（100%）。尽管 LLM 回退减少，准确率在初始阶段后稳定在约 95%，表明蒸馏后的学生模型在降低推理成本的同时保持了预测质量。

**AGNews 基准对比**

![AGNews Benchmark](benchmark_compare.png)

在 AGNews 上，我们的蒸馏学生模型达到了**与 LLM 基线大致相当**的性能：GCP 达到 81.4%，MLP 达到 79.9%，接近教师 LLM（4o-mini）的 83.4%。这证明了该框架在实现高效推理的同时保持了预测质量。



---

## 📁 项目结构

```
llmcostcut/
├── src/
│   └── llmcostcut/
│       __init__.py       # 导出 monitor
│       buffers.py       # 回放缓冲区（RingBuffer、平衡采样）
│       correctness.py   # 正确性预测器
│       defaults.py      # 编码器、分词器、优化器默认配置
│       models.py        # 分类器头（DeepMLP、GCP）
│       monitor.py       # 核心 monitor() API 与训练编排
│       registry.py      # 每任务注册表
│       selector.py      # 主动学习 / 离线样本选择
│       task.py          # 任务抽象
│       trainer.py       # 训练与 GCP 子模块重训练
├── examples/
│   example.py           # AG-News 与 GCP 分类器（概念 DAG + 在线/离线）
├── setup.py
├── pyproject.toml
├── requirements.txt
├── README.md
└── LICENSE
```

---

## 📚 API 参考

### 🔧 `monitor(...)` — 主要参数

#### 必填参数

- **`task_id2classes`** (`dict[str, list[str]]`): 任务 ID → 允许的类别标签列表。
  ```python
  {"task1": ["class1", "class2"], "task2": ["A", "B", "C"]}
  ```
- **`text`** (`str | list[str] | tuple[str, ...]`): 输入文本。单个字符串表示单样本，列表/元组表示批量。
- **`mode`** (str): **必填。** `"online"` 或 `"offline"`。
  - `"online"`: 推理期间更新学生模型。
  - `"offline"`: 学生模型冻结；通过 `offline_select_method` 和 `offline_select_budget` 预先选择样本。

#### 可选参数

| 参数 | 默认值 | 说明 |
|-----------|---------|-------------|
| `llm_fn` | 内置 OpenAI | 教师函数。自定义签名：`llm_fn(texts, task_id2classes, **kwargs)`。 |
| `p_threshold` | `0.8` | 信任学生模型的最小置信度；低于此值则使用 LLM。 |
| `classifier_type` | `"deep_mlp"` | 学生模型头：`"deep_mlp"` 或 `"gcp"`。 |
| `classifier_kwargs` | `None` | 架构相关 kwargs（见下文）。 |
| `encoder` | `distilbert-base-uncased` | 编码器模型实例。 |
| `device` | 自动 | 设备（如 `"cuda:0"`、`"cpu"`）。 |
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
from llmcostcut.monitor import wait_for_pending_training
wait_for_pending_training()
```

---

## 📐 概念图

**概念图（GCP）** 是我们论文中引入的一种具有推理感知的蒸馏架构。它将教师 LLM 的推理过程建模为概念节点的**有向无环图（DAG）**。每个节点维护一个概念嵌入和节点特定的预测器；信息通过可学习的转移从父节点传播到子节点，汇节点产生最终的任务预测。通过将学生模型结构化为概念图，GCP 在实现高效蒸馏的同时保留了可解释的中间推理，并支持对选定概念节点进行可选的子模块重训练。

---

## 📖 引用

如果您在研究中使用了本框架，请引用：

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

本框架由 Emory University 的 Liang Zhao 教授团队开发。感谢合作者和学生的反馈。设计借鉴了在线学习、知识蒸馏和自适应推理等领域的工作。
