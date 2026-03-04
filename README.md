<p align="center">
  <img src="assets/llmcostcut_logo.png" alt="LLMCostCut Logo" width="600"/>
</p>

# LLMCostCut: The Intelligent Compiler for Large Language Models

**Optimized. Efficient. Powerful.**

[![GitHub](https://img.shields.io/badge/GitHub-Project-181717?logo=github)](https://github.com/zhaoliangvaio/llmcostcut)
[![arXiv](https://img.shields.io/badge/arXiv-2602.03006-red?logo=arxiv)](https://arxiv.org/abs/2602.03006)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyPI](https://img.shields.io/pypi/v/llmcostcut?color=blue)](https://pypi.org/project/llmcostcut/)
<!-- [![PyPI Downloads](https://img.shields.io/pypi/dm/llmcostcut?color=green)](https://pypi.org/project/llmcostcut/) -->

[![中文](https://img.shields.io/badge/中文-README--zh.md-green)](README-zh.md)
[![English](https://img.shields.io/badge/English-README.md-blue)](README.md)

---

LLMCostCut is a framework for **selectively invoking LLMs** and **distilling repeated workloads** into smaller student models. Use a teacher LLM only when needed — as the student learns, LLM calls drop toward zero while preserving accuracy. The framework is applicable across diverse industrial domains, including **Legal** (e.g., Case Outcome Prediction, Contract Clause Classification), **Healthcare** (e.g., Radiology Report Abnormality Classification, Clinical Note Coding), **Finance** (e.g., Sentiment Analysis, Risk Assessment), and more.

<!-- > 💰 **Save on API costs.** Reduce LLM calls toward zero over time. With the student handling most queries and the teacher only used when uncertain, you cut inference bills significantly while maintaining ~95% accuracy. -->

## 🎯 Key Features

| 📈 **Accuracy up to 95%** | 💰 **10× Cost Reduction** | ⚡ **1000× Speedup** |
|---------------------------|---------------------------|----------------------|
| Maintain near-teacher quality while cutting LLM calls toward zero | Student handles most queries; teacher only when uncertain — inference bills drop dramatically | Small student model inference vs. LLM API calls — orders of magnitude faster |

---

## Table of Contents

- 🎯 [Key Features](#key-features)
- 📦 [Installation](#installation)
  - 📋 [Requirements](#requirements)
- 🚀 [Quick Start](#quick-start)
- ⚙️ [How It Works](#how-it-works)
- 📊 [Experimental Results](#experimental-results)
  - 📉 [Fallback Mechanism](#fallback-mechanism)
  - 📈 [Accuracy](#accuracy)
  - 💰 [Cost Reduction](#cost-reduction)
- 📁 [Project Structure](#project-structure)
- 📚 [API Reference](#api-reference)
  - 🔧 [monitor()](#monitor---main-parameters)
  - ⏳ [wait_for_pending_training()](#wait_for_pending_training)
- 📐 [Graph of Concepts](#graph-of-concepts)
- 📖 [Citation](#citation)
- 🙏 [Acknowledgements](#acknowledgements)

---

## 📦 Installation

**Install from source** (recommended for development)

```bash
git clone https://github.com/emory-llmcostcut/llmcostcut.git
cd llmcostcut
pip install -e .
```

**Install from PyPI**

```bash
pip install llmcostcut
```

### 📋 Requirements

- Python >= 3.8
- PyTorch >= 1.9.0
- Transformers >= 4.20.0

Optional (for built-in OpenAI teacher and examples):

- `openai>=1.0.0`
- `python-dotenv>=1.0.0`
- `pydantic>=2.0.0`

When using the built-in OpenAI teacher, set your API key:

```bash
export OPENAI_API_KEY="your-key-here"
```

Or put `OPENAI_API_KEY=your-key-here` in a `.env` file in the project root (loaded automatically if `python-dotenv` is installed).



## 🚀 Quick Start

Instead of calling the LLM directly every time, use **`monitor`** as a smart router: it first tries the small student model; only when the student is uncertain (confidence below `p_threshold`) does it fall back to the teacher LLM. Over time, the student learns and LLM calls drop toward zero.

```python
from llmcostcut.monitor import monitor, wait_for_pending_training

# Define your task and classes (e.g. news topic: World, Sports, Business, Sci/Tech)
task_id2classes = {"topic": ["World", "Sports", "Business", "Sci/Tech"]}

# monitor switches between student (small model) and LLM automatically
results, fallback = monitor(
    task_id2classes,
    "The Nobel physicist's quantum startup now supplies chips to military satellites.",  # Crosses World, Sci/Tech, Business
    mode="online",           # or "offline"
    p_threshold=0.8,         # below this: use LLM; above: use student
)
# When using online mode, wait for background training before exit
wait_for_pending_training()
```

See **[examples/example.py](examples/example.py)** for a full AG-News demo with GCP classifier and concept-level labels.

---

## ⚙️ How It Works

1. **Student prediction** — Classify input with the small model.
2. **Confidence** — Estimate reliability of the prediction.
3. **Fallback** — If confidence ≥ `p_threshold`, return student output; otherwise call the teacher LLM.
4. **Online learning** — Store (text, teacher labels) in a replay buffer and periodically retrain the student and correctness predictor. For GCP, optional sub-module retraining refines selected concept nodes.

---

## 📊 Experimental Results

### 💰 Cost Reduction

![Cost Curve](cost_curve.png)

The inference cost decreases over time as the student model handles more queries and the fallback to the teacher LLM becomes less frequent.

### 📉 Fallback Mechanism

![Fallback](fallback.png)

The fallback ratio (teacher/LLM utilization) decreases over iterations as the student becomes more capable. Early on, the system relies on the teacher for correctness; as online distillation progresses, the student handles more queries, reducing LLM calls toward zero by around iteration 100.

### 📈 Accuracy

![Accuracy](acc.png)

System accuracy stays close to the teacher baseline (100%) during training. Despite the drop in LLM fallback, accuracy stabilizes around 95% after the initial phase, showing that the distilled student preserves quality while cutting inference cost.

**AGNews Benchmark Comparison**

![AGNews Benchmark](benchmark_compare.png)

On AGNews, our distilled student models achieve performance **approximately comparable to the LLM baseline**: GCP reaches 81.4% and MLP 79.9%, close to the teacher LLM (4o-mini) at 83.4%. This demonstrates that the framework preserves prediction quality while enabling efficient inference.



---

## 📁 Project Structure

```
llmcostcut/
├── src/
│   └── llmcostcut/
│       __init__.py       # Exports monitor
│       buffers.py       # Replay buffer (RingBuffer, balanced sampling)
│       correctness.py   # Correctness predictor
│       defaults.py      # Encoder, tokenizer, optimizer defaults
│       models.py        # Classifier heads (DeepMLP, GCP)
│       monitor.py       # Core monitor() API and training orchestration
│       registry.py      # Per-task registry
│       selector.py      # Active-learning / offline sample selection
│       task.py          # Task abstraction
│       trainer.py       # Training and GCP submodule retraining
├── examples/
│   example.py           # AG-News with GCP classifier (concept DAG + online/offline)
├── setup.py
├── pyproject.toml
├── requirements.txt
├── README.md
└── LICENSE
```

---

## 📚 API Reference

### 🔧 `monitor(...)` — main parameters

#### Required

- **`task_id2classes`** (`dict[str, list[str]]`): Task ID → list of allowed class labels.
  ```python
  {"task1": ["class1", "class2"], "task2": ["A", "B", "C"]}
  ```
- **`text`** (`str | list[str] | tuple[str, ...]`): Input text(s). Single string for one sample, list/tuple for batch.
- **`mode`** (str): **Required.** `"online"` or `"offline"`.
  - `"online"`: student is updated during inference.
  - `"offline"`: student is frozen; samples are selected up-front via `offline_select_method` and `offline_select_budget`.

#### Optional parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `llm_fn` | built-in OpenAI | Teacher function. Custom signature: `llm_fn(texts, task_id2classes, **kwargs)`. |
| `p_threshold` | `0.8` | Min confidence to trust the student; below this, use LLM. |
| `classifier_type` | `"deep_mlp"` | Student head: `"deep_mlp"` or `"gcp"`. |
| `classifier_kwargs` | `None` | Architecture-specific kwargs (see below). |
| `encoder` | `distilbert-base-uncased` | Encoder model instance. |
| `device` | auto | Device (e.g. `"cuda:0"`, `"cpu"`). |
| `llm_kwargs` | `{}` | Passed to `llm_fn`. For default OpenAI teacher, can include `concept_info` for GCP concept labels. |

#### Offline-only parameters

| Parameter | Description |
|-----------|-------------|
| `offline_select_method` | **Required** when `mode="offline"`. Strategy (e.g. `"random"`, `"uncertainty"`). |
| `offline_select_budget` | **Required** when `mode="offline"`. Max number of samples to send to the teacher. |

#### Return value

A 2-tuple `(results, fallback)`:

| Input | `results` | `fallback` |
|-------|------------|------------|
| Single `str` | `dict[str, str]` | `bool` |
| list/tuple str | `list[dict[str, str]]` | `list[bool]` |

### ⏳ `wait_for_pending_training()`

When using `mode="online"`, training runs in a background thread. Before exit or before evaluating the student, call:

```python
from llmcostcut.monitor import wait_for_pending_training
wait_for_pending_training()
```

---

## 📐 Graph of Concepts

The **Graph of Concepts (GCP)** is a reasoning-aware distillation architecture introduced in our paper. It mirrors the teacher LLM's reasoning process as a **Directed Acyclic Graph (DAG)** of concept nodes. Each node maintains a concept embedding and a node-specific predictor; information propagates from parent nodes to children through learnable transitions, and sink nodes produce the final task prediction. By structuring the student as a concept graph, GCP preserves interpretable intermediate reasoning while enabling efficient distillation and optional sub-module retraining for selected concept nodes.

---

## 📖 Citation

If you use this framework in your research, please cite:

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

## 🙏 Acknowledgements

This framework was developed by the team led by Prof. Liang Zhao at Emory University. We thank collaborators and students for feedback. The design draws on work in online learning, knowledge distillation, and adaptive inference.
