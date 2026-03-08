
<p align="center">
  <img src="assets/llmcostcut_logo.png" alt="LLMCostCut Logo" width="600"/>
</p>

# LLMCostCut: The Intelligent Cost Cutter for Large Language Models

**Optimized. Efficient. Powerful.**

[![GitHub](https://img.shields.io/badge/GitHub-Project-black?logo=github)](https://github.com/zhaoliangvaio/llmcostcut)
[![arXiv](https://img.shields.io/badge/arXiv-2602.03006-b31b1b?logo=arxiv)](https://arxiv.org/abs/2602.03006)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8-blue?logo=python)](https://www.python.org/)
[![PyPI](https://img.shields.io/badge/PyPI-v1.0.0-orange?logo=pypi)](https://pypi.org/project/llmcostcut/)


---

As LLM becomes more and more popular, the cost of using LLM is becoming a major concern. LLMCostCut is a discriminative workload for LLM to reduce the cost of using LLM while maintaining the accuracy.

## Table of Contents

- 📋 [Applications](#applications)
- 🎯 [Key Features](#key-features)
- 📦 [Installation](#installation)
  - 📋 [Requirements](#requirements)
- 🚀 [Quick Start](#quick-start)
- ⚙️ [How It Works](#how-it-works)
- 📊 [Experimental Results](#experimental-results)
  - 📈 [Accuracy](#accuracy)
  - 💰 [Cost Reduction](#cost-reduction)
  - 📊 [Benchmarks Comparison](#benchmarks-comparison)
- 📁 [Project Structure](#project-structure)
- 📚 [API Reference](#api-reference)
  - 🔧 [monitor()](#monitor---main-parameters)
- 📖 [Citation](#citation)
- 🙏 [Acknowledgements](#acknowledgements)


## 📋 Applications
Reducing Cost of LLM Reasoning in Discriminative Workloads

<p align="center">
  <img src="assets/table.png" alt="Applications" width="900"/>
</p>

<!-- | ⚖️ Legal | 🏥 Healthcare | 💼 Finance | ✨ More |
|---|---|---|---|
| -Case Outcome Prediction<br>-Contract Clause Classification<br>-Legal Reasoning / Multi-hop QA<br><div align="center">...</div> | -Radiology Report Abnormality Classification<br>-Clinical Note Coding<br>-Medical NER / Drug-Drug Interaction<br><div align="center">...</div> | -Sentiment Analysis<br>-Risk Assessment<br>-Fraud Detection / Financial QA<br><div align="center">...</div> | -Science<br>-Materials<br><div align="center">...</div> | -->

## 🎯 Key Features


| 📈 **Accuracy up to 95%**                                         | 💰 **Around 10× Cost Reduction**                                                              | ⚡ **Nearly 1000× Speedup**                                                                                                     | 🛠️ **Easy to Use**                                  |
| ----------------------------------------------------------------- | --------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------ | ---------------------------------------------------- |
| Maintain near-teacher quality while cutting LLM calls toward zero | Student handles most queries; teacher only when uncertain — inference bills drop dramatically | Local small student model (~100k-2M params) inference vs. LLM (~1T params) API calls — orders of magnitude faster in latency | Minimal code changes — Runnable in few lines of code |


---

## 📦 Installation

**Install from source** (recommended for development)

```bash
git clone https://github.com/zhaoliangvaio/llmcostcut.git
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

Instead of calling the LLM directly every time, use `**monitor**` as a smart router: it first tries the small student model; only when the student is uncertain (confidence below `p_threshold`) does it fall back to the teacher LLM. Over time, the student learns and LLM calls drop toward zero.

```python
from datasets import load_dataset
from llmcostcut.monitor import monitor
from openai import OpenAI

def classify_with_llm(texts, task):
    labels, client, results = task["topic_classification"], OpenAI(), []
    for text in texts:
        r = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role":"user","content":f"Classify into {labels}. Text: {text[:300]}"}],
            max_tokens=10,
        )
        results.append({"topic_classification": r.choices[0].message.content.strip()})
    return results

TASK = {"topic_classification": ["World","Sports","Business","Sci/Tech"]}

for example in load_dataset("ag_news", split="train[:100]"):
    output, used_llm_or_not = monitor(TASK, example["text"], llm_fn=classify_with_llm, mode="online")

monitor.close()
```

See **[examples/example.py](examples/example.py)** for a full AG-News demo with GCP classifier and concept-level labels.

---

## ⚙️ How It Works

![LLMCostCut](assets/LLMCostCut.gif)

---

## 📊 Experimental Results

### 💰 Cost Reduction

![Cost Curve](assets/cost_curve.png)

The inference cost decreases over time as the student model handles more queries and the fallback to the teacher LLM becomes less frequent.

### 📈 Accuracy

![Accuracy](assets/acc.png)

System accuracy stays close to the teacher baseline (100%) during training. Despite the drop in LLM fallback, accuracy stabilizes around 95% after the initial phase, showing that the distilled student preserves quality while cutting inference cost.

### 📊 Benchmarks Comparison in Accuracy


| Dataset                                       | LLMCost with Multilayer Perceptron (MLP) as small model | LLMCost with Graph of Concepts (GCP) as small model |
| --------------------------------------------- | ------------------------------------------------------- | --------------------------------------------------- |
| Supreme Court Judgment Prediction Dataset     | 95.8 %                                                   | 97.6 %                                               |
| MIMIC-CXR Dataset                             | 93.6 %                                                   | 96.5 %                                               |
| American Express - Default Prediction Dataset | 95.7 %                                                   | 97.8 %                                               |


Across [Supreme Court Judgment Prediction Dataset](https://www.kaggle.com/datasets/deepcontractor/supreme-court-judgment-prediction), [MIMIC-CXR Dataset](https://physionet.org/content/mimic-cxr/2.1.0/), and [American Express - Default Prediction Dataset](https://www.kaggle.com/competitions/amex-default-prediction), our student models **approximately comparable to the teacher LLM baseline** in *relative accuracy*: GCP (proposed in [Distilling LLM Reasoning into Graph of Concept Predictors](https://arxiv.org/abs/2602.03006)) reaches 97.6%, 96.5%, and 97.8% respectively, while MLP achieves 95.8%, 93.6%, and 95.7%. This demonstrates that the framework preserves prediction quality while enabling efficient inference.

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

- `**tasks**` (`dict[str, list[str]]`): Task ID → list of allowed class labels.
  ```python
  {"task1": ["class1", "class2"], "task2": ["A", "B", "C"]}
  ```
- `**text**` (`str | list[str] | tuple[str, ...]`): Input text(s). Single string for one sample, list/tuple for batch.
- `**mode**` (str): **Required.** `"online"` or `"offline"`.
  - `"online"`: student is updated during inference.
  - `"offline"`: student is frozen; samples are selected up-front via `offline_select_method` and `offline_select_budget`.

#### Optional parameters


| Parameter           | Default                   | Description                                                                                        |
| ------------------- | ------------------------- | -------------------------------------------------------------------------------------------------- |
| `llm_fn`            | built-in OpenAI           | Teacher function. Custom signature: `llm_fn(texts, tasks, **kwargs)`.                              |
| `p_threshold`       | `0.8`                     | Min confidence to trust the student; below this, use LLM.                                          |
| `classifier_type`   | `"deep_mlp"`              | Student head: `"deep_mlp"` or `"gcp"`.                                                             |
| `classifier_kwargs` | `None`                    | Architecture-specific kwargs (see below).                                                          |
| `encoder`           | `distilbert-base-uncased` | Encoder model instance.                                                                            |
| `device`            | auto                      | Device (e.g. `"cuda:0"`, `"cpu"`).                                                                 |
| `llm_kwargs`        | `{}`                      | Passed to `llm_fn`. For default OpenAI teacher, can include `concept_info` for GCP concept labels. |


#### Offline-only parameters


| Parameter               | Description                                                                       |
| ----------------------- | --------------------------------------------------------------------------------- |
| `offline_select_method` | **Required** when `mode="offline"`. Strategy (e.g. `"random"`, `"uncertainty"`).  |
| `offline_select_budget` | **Required** when `mode="offline"`. Max number of samples to send to the teacher. |


#### Return value

A 2-tuple `(results, used_llm)`:


| Input          | `results`              | `used_llm`   |
| -------------- | ---------------------- | ------------ |
| Single `str`   | `dict[str, str]`       | `bool`       |
| list/tuple str | `list[dict[str, str]]` | `list[bool]` |


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
