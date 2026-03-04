<p align="center">
  <img src="assets/llmcostcut_logo.png" alt="LLMCostCut Logo" width="600"/>
</p>

# LLMCostCut：大语言模型的智能成本削减器

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

随着大语言模型（LLM）越来越普及，使用 LLM 的成本已成为主要关注点。LLMCostCut 是一种判别式工作负载方案，可在保持准确性的同时降低 LLM 使用成本。


## 应用场景

| ⚖️ **法律** | 🏥 **医疗** | 💼 **金融** |
|--------------|-------------------|----------------|
| - 案件结果预测<br>- 合同条款分类 | - 放射报告异常分类<br>- 临床记录编码 | - 情感分析<br>- 风险评估 |


## 🎯 核心特性

| 📈 **准确率高达 95%** | 💰 **约 10× 成本削减** | ⚡ **近 1000× 加速** | 🛠️ **易于使用** |
|---------------------------|---------------------------|----------------------|-------------------|
| 在将 LLM 调用降至接近零的同时保持接近教师模型的质量 | 学生模型处理大部分查询；仅在不确定时调用教师模型 — 推理费用大幅下降 | 小规模学生模型（10 万参数）推理 vs LLM（1T 参数）API 调用 — 数量级级加速 | 最少代码改动 — 几行代码即可运行 |

---

## 目录

- 🎯 [核心特性](#核心特性)
- 📦 [安装](#安装)
  - 📋 [依赖要求](#依赖要求)
- 🚀 [快速开始](#快速开始)
- ⚙️ [工作原理](#工作原理)
- 📊 [实验结果](#实验结果)
  - 📈 [准确率](#准确率)
  - 💰 [成本削减](#成本削减)
  - 📊 [基准对比](#基准对比)
- 📁 [项目结构](#项目结构)
- 📚 [API 参考](#api-参考)
  - 🔧 [monitor()](#monitor---主要参数)
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

### 📋 依赖要求

- Python >= 3.8
- PyTorch >= 1.9.0
- Transformers >= 4.20.0

可选（用于内置 OpenAI 教师模型和示例）：

- `openai>=1.0.0`
- `python-dotenv>=1.0.0`
- `pydantic>=2.0.0`

使用内置 OpenAI 教师模型时，请设置 API 密钥：

```bash
export OPENAI_API_KEY="your-key-here"
```

或将 `OPENAI_API_KEY=your-key-here` 写入项目根目录的 `.env` 文件中（若已安装 `python-dotenv` 会自动加载）。



## 🚀 快速开始

不要每次都直接调用 LLM，而是使用 **`monitor`** 作为智能路由器：它首先尝试小规模学生模型；仅当学生模型不确定（置信度低于 `p_threshold`）时才回退到教师 LLM。随着时间推移，学生模型不断学习，LLM 调用逐渐趋近于零。

```python
"""
简单的 LLMCostCut 示例：在 AG-News 上降低 LLM API 调用成本。

LLMCostCut 即时训练小型学生模型（DistilBERT + MLP）。
仅当学生模型置信度不足时才调用教师 LLM。
随着学生模型学习，LLM 调用率下降 — 削减 API 成本。

用法：
export OPENAI_API_KEY=sk-...
python examples/simple_cost_reduction.py
"""

import random
from datasets import load_dataset
from llmcostcut.monitor import monitor

# --- 基准：AG-News（4 类主题分类）---
ds = load_dataset("ag_news", split="train[:500]")  # 500 个样本用于快速演示
LABEL_MAP = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}
samples = [(row["text"], LABEL_MAP[row["label"]]) for row in ds]
random.shuffle(samples)

TASK = {"topic": ["World", "Sports", "Business", "Sci/Tech"]}

# --- 可选：接入你自己的 LLM 函数（如 Claude、GPT-4）---
# 签名：llm_fn(texts, task_id2classes) -> list[dict[task_id, label]]
# 若 llm_fn=None 则使用内置 OpenAI 教师（读取 OPENAI_API_KEY）。
def my_llm_fn(texts, task_id2classes, **kwargs):
    import openai
    client = openai.OpenAI()
    results = []
    for text in texts:
        classes = task_id2classes["topic"]
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{
                "role": "user",
                "content": f"Classify the news into one of {classes}.\nText: {text[:300]}\nReply with only the label."
            }],
            max_tokens=10,
        )
        label = resp.choices[0].message.content.strip()
        label = label if label in classes else classes[0]  # 回退
        results.append({"topic": label})
    return results

# --- 通过 monitor() 流式处理样本；观察 LLM 调用率下降 ---
llm_calls, correct, total = 0, 0, len(samples)

for i, (text, true_label) in enumerate(samples):
    pred, used_llm = monitor(TASK, text, llm_fn=my_llm_fn, mode="online")

    if used_llm:
        llm_calls += 1
    if pred.get("topic") == true_label:
        correct += 1

    # 每 100 个样本打印进度
    if (i + 1) % 100 == 0:
        print(f"[{i+1:>4}/{total}] LLM 调用率: {llm_calls/(i+1)*100:.1f}% "
              f"准确率: {correct/(i+1)*100:.1f}%")

monitor.close()
print(f"\n最终 — 总 LLM 调用: {llm_calls}/{total} ({llm_calls/total*100:.1f}%)")
print(f"最终 — 准确率: {correct}/{total} ({correct/total*100:.1f}%)")
```

完整 AG-News 演示（含 GCP 分类器和概念级标签）请参见 **[examples/example.py](examples/example.py)**。

---

## ⚙️ 工作原理

![LLMCostCut 架构](diagram.png)

---

## 📊 实验结果

### 💰 成本削减

![成本曲线](cost_curve.png)

随着学生模型处理更多查询、回退到教师 LLM 的频率降低，推理成本随时间下降。

### 📈 准确率

![准确率](acc.png)

训练期间系统准确率保持接近教师基线（100%）。尽管 LLM 回退率下降，准确率在初始阶段后稳定在约 95%，表明蒸馏后的学生模型在削减推理成本的同时保持了质量。

### 📊 基准对比

![AGNews 基准](benchmark_compare.png)

在 AGNews、SemEval 和 Amazon 上，我们的蒸馏学生模型**超越了教师 LLM 基线**：GCP 分别达到 97.6%、96.5% 和 97.8%，而 MLP 达到 95.8%、93.6% 和 95.7%。这证明该框架在实现高效推理的同时保持了预测质量。



---

## 📁 项目结构

```
llmcostcut/
├── src/
│   └── llmcostcut/
│       __init__.py       # 导出 monitor
│       buffers.py        # 回放缓冲区（RingBuffer、平衡采样）
│       correctness.py    # 正确性预测器
│       defaults.py       # 编码器、分词器、优化器默认配置
│       models.py         # 分类头（DeepMLP、GCP）
│       monitor.py        # 核心 monitor() API 及训练编排
│       registry.py       # 每任务注册表
│       selector.py       # 主动学习 / 离线样本选择
│       task.py           # 任务抽象
│       trainer.py        # 训练及 GCP 子模块重训练
├── examples/
│   example.py            # AG-News 与 GCP 分类器（概念 DAG + 在线/离线）
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

- **`task_id2classes`** (`dict[str, list[str]]`)：任务 ID → 允许的类别标签列表。
  ```python
  {"task1": ["class1", "class2"], "task2": ["A", "B", "C"]}
  ```
- **`text`** (`str | list[str] | tuple[str, ...]`)：输入文本。单字符串表示单样本，列表/元组表示批量。
- **`mode`** (str)：**必填。** `"online"` 或 `"offline"`。
  - `"online"`：推理过程中更新学生模型。
  - `"offline"`：学生模型冻结；样本通过 `offline_select_method` 和 `offline_select_budget` 预先选择。

#### 可选参数

| 参数 | 默认值 | 说明 |
|-----------|---------|-------------|
| `llm_fn` | 内置 OpenAI | 教师函数。自定义签名：`llm_fn(texts, task_id2classes, **kwargs)`。 |
| `p_threshold` | `0.8` | 信任学生模型的最小置信度；低于此值则使用 LLM。 |
| `classifier_type` | `"deep_mlp"` | 学生分类头：`"deep_mlp"` 或 `"gcp"`。 |
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

二元组 `(results, fallback)`：

| 输入 | `results` | `fallback` |
|-------|------------|------------|
| 单个 `str` | `dict[str, str]` | `bool` |
| list/tuple str | `list[dict[str, str]]` | `list[bool]` |

<!-- ### ⏳ `monitor.close()` / `monitor.start()`

使用 `mode="online"` 时，训练在后台线程中运行。在退出或评估学生模型之前，需显式调用 `monitor.close()` 或使用上下文管理器：

```python
from llmcostcut.monitor import monitor

# 显式关闭
monitor.close()

# 或使用上下文管理器（退出时自动关闭）
with monitor.start():
    results, fallback = monitor(task_id2classes, text, mode="online")
``` -->

---


## 📖 引用

如在研究中使用了本框架，请引用：

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
