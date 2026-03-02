<p align="center">
  <img src="assets/llmcompiler_logo.png" alt="LLMCompiler Logo" width="600"/>
</p>

# LLMCompiler: The Intelligent Compiler for Large Language Models

**Optimized. Efficient. Powerful.**

[![GitHub](https://img.shields.io/badge/GitHub-Project-181717?logo=github)](https://github.com/zhaoliangvaio/llmcompiler)
[![arXiv](https://img.shields.io/badge/arXiv-2602.03006-red?logo=arxiv)](https://arxiv.org/abs/2602.03006)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyPI](https://img.shields.io/pypi/v/llmcompiler?color=blue)](https://pypi.org/project/llmcompiler/)
[![PyPI Downloads](https://img.shields.io/pypi/dm/llmcompiler?color=green)](https://pypi.org/project/llmcompiler/)

[![中文](https://img.shields.io/badge/中文-README--zh.md-green)](README-zh.md)
[![English](https://img.shields.io/badge/English-README.md-blue)](README.md)

---

LLMCompiler is a framework for **selectively invoking LLMs** and **distilling repeated workloads** into smaller student models. Use a teacher LLM only when needed — as the student learns, LLM calls drop toward zero while preserving accuracy.

> 💰 **Save on API costs.** Reduce LLM calls toward zero over time. With the student handling most queries and the teacher only used when uncertain, you cut inference bills significantly while maintaining ~95% accuracy.

## Key Features

| 🎯 **Correctness-Aware Fallback** | 📚 **Online Distillation** | 🔄 **Replay-Buffer Training** |
|----------------------------------|---------------------------|-------------------------------|
| Use teacher LLM only when student is uncertain (below confidence threshold) | Continuously update the student from teacher labels on fallback samples | Store fallback (text, label) pairs and periodically retrain the student and correctness predictor |

---

## Installation

**Install from source** (recommended for development)

```bash
git clone https://github.com/emory-llmcompiler/llmcompiler.git
cd llmcompiler
pip install -e .
```

**Install from PyPI** (when available)

```bash
pip install llmcompiler
```

### Requirements

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



## Quick Start

```python
from llmcompiler.monitor import monitor, wait_for_pending_training

# Define your task and classes
task_id2classes = {"topic": ["Positive", "Negative"]}

# Single text
results, fallback = monitor(
    task_id2classes,
    "I love this product!", # Your input text here
    mode="online",  # or "offline"
    p_threshold=0.8,
)

print(results)   # {"topic": "Positive"}
print(fallback) # True if LLM was used, False if student predicted

# When using online mode, wait for background training before exit
wait_for_pending_training()
```

See **[examples/example.py](examples/example.py)** for a full AG-News demo with GCP classifier and concept-level labels.

---

## How It Works

1. **Student prediction** — Classify input with the small model.
2. **Confidence** — Estimate reliability of the prediction.
3. **Fallback** — If confidence ≥ `p_threshold`, return student output; otherwise call the teacher LLM.
4. **Online learning** — Store (text, teacher labels) in a replay buffer and periodically retrain the student and correctness predictor. For GCP, optional sub-module retraining refines selected concept nodes.

---

## Experimental Results

### Fallback Mechanism

![Fallback](fallback.png)

The fallback ratio (teacher/LLM utilization) decreases over iterations as the student becomes more capable. Early on, the system relies on the teacher for correctness; as online distillation progresses, the student handles more queries, reducing LLM calls toward zero by around iteration 100.

### Accuracy

![Accuracy](acc.png)

System accuracy stays close to the teacher baseline (100%) during training. Despite the drop in LLM fallback, accuracy stabilizes around 95% after the initial phase, showing that the distilled student preserves quality while cutting inference cost.

---

## Project Structure

```
llmcompiler/
├── src/
│   └── llmcompiler/
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
├── test_installation.py # Quick install check
├── setup.py
├── pyproject.toml
├── requirements.txt
├── README.md
└── LICENSE
```

---

## API Reference

### `monitor(...)` — main parameters

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

### `wait_for_pending_training()`

When using `mode="online"`, training runs in a background thread. Before exit or before evaluating the student, call:

```python
from llmcompiler.monitor import wait_for_pending_training
wait_for_pending_training()
```

---

## Citation

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

## Acknowledgements

This framework was developed by the team led by Prof. Liang Zhao at Emory University. We thank collaborators and students for feedback. The design draws on work in online learning, knowledge distillation, and adaptive inference.
