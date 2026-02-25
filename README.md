# LLMCompiler

A framework for selectively invoking LLMs and distilling repeated workloads into smaller student models.

This library provides:

- **Correctness-aware LLM fallback** — use a teacher LLM only when the student is uncertain (below a confidence threshold).
- **Online distillation** — continuously update the student from teacher labels on fallback samples.
- **Replay-buffer–based incremental training** — store fallback (text, label) pairs and periodically retrain the student and correctness predictor.

Associated research: [Distilling LLM Reasoning into Graph of Concept Predictors](https://arxiv.org/abs/2602.03006).


## Experimental Results

### Fallback Mechanism
![Fallback](fallback.png)

The fallback ratio (teacher/LLM utilization) decreases over iterations as the student becomes more capable. Early on, the system relies on the teacher for correctness; as online distillation progresses, the student handles more queries, reducing LLM calls toward zero by around iteration 100.

### Accuracy
![Accuracy](acc.png)

System accuracy stays close to the teacher baseline (100%) during training. Despite the drop in LLM fallback, accuracy stabilizes around 95% after the initial phase, showing that the distilled student preserves quality while cutting inference cost.


## Installation

From the repository root:

```bash
pip install -e .
```

### Requirements

- Python >= 3.8
- PyTorch >= 1.9.0
- Transformers >= 4.20.0

Optional (for the built-in OpenAI teacher and examples such as `agnews_gcp.py`):

- `openai>=1.0.0`
- `python-dotenv>=1.0.0`
- `pydantic>=2.0.0`

When using the built-in OpenAI teacher, set your API key so the framework can call the LLM. Either put `OPENAI_API_KEY=your-key-here` in a `.env` file in the project root (loaded automatically if `python-dotenv` is installed), or export it in your environment: `export OPENAI_API_KEY=your-key-here`.

### Verify installation

```python
from llmcompiler import monitor
print("LLMCompiler installed successfully!")
```

Or run the installation check script from the repo root:

```bash
python test_installation.py
```


## Project structure

```
llmcompiler/
├── src/
│   └── llmcompiler/
│       __init__.py       # Exports monitor
│       buffers.py        # Replay buffer (RingBuffer, balanced sampling)
│       correctness.py    # Correctness predictor
│       defaults.py       # Encoder, tokenizer, optimizer defaults
│       models.py         # Classifier heads (DeepMLP, GCP)
│       monitor.py        # Core monitor() API and training orchestration
│       registry.py       # Per-task registry
│       selector.py       # Active-learning / offline sample selection
│       task.py           # Task abstraction
│       trainer.py        # Training and GCP submodule retraining
├── examples/
│   agnews_gcp.py         # AG-News with GCP classifier (concept DAG + online/offline)
├── test_installation.py  # Quick install check
├── setup.py
├── pyproject.toml
├── requirements.txt
├── README.md
└── LICENSE
```

Install in development mode with `pip install -e .`. If present, run tests with `pytest tests/` from the repo root (see `pyproject.toml` for pytest config).


## Quick start

See the **[examples/](examples/)** directory for runnable usage:


- **[examples/agnews_gcp.py](examples/agnews_gcp.py)** — AG-News with GCP classifier and concept-level labels (online/offline).


## API reference

### `monitor(...)` — main parameters

#### Required

- **`task_id2classes`** (`dict[str, list[str]]`): Task ID → list of allowed class labels.
  ```python
  {"task1": ["class1", "class2"], "task2": ["A", "B", "C"]}
  ```
- **`text`** (`str | list[str] | tuple[str, ...]`): Input text(s). Single string for one sample, list/tuple for batch.
- **`llm_fn`** (callable, optional): Teacher function. Signature: `llm_fn(texts, task_id2classes, **kwargs)`. Return one of:
  - `list[dict[task_id, label]]` — one dict per input (preferred)
  - `dict[task_id, list[label]]` — one list per task
  - `dict[task_id, label]` — single-sample form only when batch size is 1  
  If omitted, the built-in OpenAI teacher is used (requires `OPENAI_API_KEY`; supports GCP concept labels via `llm_kwargs["concept_info"]`).
- **`mode`** (str): **Required.** `"online"` or `"offline"`.
  - `"online"`: student is updated during inference.
  - `"offline"`: student is frozen; samples are selected up-front via `offline_select_method` and `offline_select_budget`.

#### Offline-only parameters

Used only when `mode="offline"`.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `offline_select_method` | — | **Required** when `mode="offline"`. Strategy (e.g. `"random"`, `"uncertainty"`). |
| `offline_select_budget` | — | **Required** when `mode="offline"`. Max number of samples to send to the teacher. |
| `offline_select_seed` | `42` | Random seed for selection. |
| `offline_select_probs` | `None` | Per-sample predicted probabilities (uncertainty selection). |
| `offline_select_embeddings` | `None` | Per-sample embeddings (diversity selection). |
| `offline_select_mc_probs` | `None` | Monte-Carlo dropout probs (BALD-style). |
| `offline_select_already_selected` | `None` | Indices already selected (excluded). |
| `offline_select_pool_size` | from `text` | Pool size; inferred from `text` length if not set. |

#### Other optional parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `p_threshold` | `0.8` | Min confidence to trust the student; below this, use LLM. |
| `classifier_type` | `"deep_mlp"` | Student head: `"deep_mlp"` or `"gcp"`. Only applied when the task is first seen. |
| `classifier_kwargs` | `None` | Architecture-specific kwargs (see below). |
| `encoder` | `distilbert-base-uncased` | Encoder model instance. |
| `tokenizer` | auto | Tokenizer (inferred if not set). |
| `hidden_size` | auto | Encoder hidden size. |
| `device` | auto | Device (e.g. `"cuda:0"`, `"cpu"`). |
| `optimizer` | auto | Optimizer for the student. |
| `scheduler` | `None` | LR scheduler for training. |
| `llm_kwargs` | `{}` | Passed to `llm_fn`. For default OpenAI teacher, can include `concept_info` for GCP concept labels. |
| `submodule_retrain_top_k` | `None` | Number of GCP concept nodes for sub-module retraining (§3.4 of the paper). Only for `classifier_type="gcp"`. |

#### `classifier_kwargs` by architecture

- **`"deep_mlp"`** — MLP with residual connections and LayerNorm (default).

  | Key         | Type | Default | Description                    |
  |-------------|------|---------|--------------------------------|
  | `num_layers`| int  | `3`     | Hidden layers before output.  |
  | `dropout`   | float| `0.1`   | Dropout after hidden layers.  |

- **`"gcp"`** — Graph of Concept Predictors (DAG; see [paper](https://arxiv.org/abs/2602.03006)).

  | Key          | Type | Default | Description |
  |--------------|------|---------|-------------|
  | `edges`      | list[tuple[int,int]] | *(required)* | DAG edges `(parent, child)`; nodes 0..N-1. |
  | `concept_dim`| int  | `256`   | Feature dim per concept node. |
  | `use_resnet` | bool | `False` | Residual + LayerNorm per concept. |
  | `dropout`    | float| `0.1`   | Dropout in sub-modules. |

  Example:
  ```python
  classifier_type="gcp",
  classifier_kwargs={"edges": [(0, 1), (1, 2), (2, 3)], "concept_dim": 128}
  ```

#### Return value

A 2-tuple `(results, fallback)`:

| Input          | `results`              | `fallback`   |
|----------------|------------------------|-------------|
| Single `str`   | `dict[str, str]`       | `bool`      |
| list/tuple str | `list[dict[str, str]]` | `list[bool]`|

### `wait_for_pending_training()`

When using `mode="online"`, training runs in a background thread. Before exit or before evaluating the student, call:

```python
from llmcompiler.monitor import wait_for_pending_training
wait_for_pending_training()
```


## How it works

1. **Student prediction** — Classify input with the small model.
2. **Confidence** — Estimate reliability of the prediction.
3. **Fallback** — If confidence ≥ `p_threshold`, return student output; otherwise call the teacher LLM.
4. **Online learning** — Store (text, teacher labels) in a replay buffer and periodically retrain the student and correctness predictor. For GCP, optional sub-module retraining refines selected concept nodes.


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


## Acknowledgements

This framework was developed by the LLMCompiler team at Emory University. We thank collaborators and students for feedback. The design draws on work in online learning, knowledge distillation, and adaptive inference.
