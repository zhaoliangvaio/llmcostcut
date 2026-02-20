# llmcompiler

A framework for selectively invoking LLMs and distilling repeated workloads into smaller models.

This library provides infrastructure for:
- Correctness-aware LLM fallback
- Online distillation
- Replay-buffer–based incremental training

Used in research work: [Distilling LLM Reasoning into Graph of Concept Predictors](https://arxiv.org/abs/2602.03006)


## Experimental Results

### Fallback Mechanism
![Fallback](fallback.png)

The fallback ratio (teacher/LLM utilization) decreases over iterations as the student model becomes more capable. Early in training, the system relies entirely on the teacher LLM for correctness; as online distillation progresses, the student handles an increasing fraction of queries independently, reducing LLM calls to near zero by iteration 100.

### Accuracy Results
![Accuracy](acc.png)

Overall system accuracy remains close to the teacher's baseline (100%) throughout training. Despite the sharp reduction in LLM fallback, accuracy stabilizes around 95% after the initial adaptation phase, demonstrating that the distilled student model preserves predictive quality while significantly cutting inference cost.


## Installation

```bash
pip install -e .
```

### Dependencies

- Python >= 3.8
- PyTorch >= 1.9.0
- Transformers >= 4.20.0

### Verify Installation

```python
from llmcompiler import monitor
print("LLMCompiler installed successfully!")
```


## Quick Start

```python
from llmcompiler import monitor

# 1. Define your teacher LLM function
#    texts is a list[str] (batch of fallback samples)
def my_llm(texts, task_id2classes, **kwargs):
    # Your LLM call logic
    # Return format: list[{task_id: predicted_label}]  (one dict per input)
    return [
        {"sentiment": "positive", "topic": "technology"}
        for _ in texts
    ]

# 2. Define tasks and class labels
task_id2classes = {
    "sentiment": ["positive", "negative", "neutral"],
    "topic": ["sports", "politics", "technology"]
}

# 3. Run classification with monitor (mode is required)
results, fallback = monitor(
    task_id2classes=task_id2classes,
    text="I love this new phone!",
    llm_fn=my_llm,
    mode="online",       # required: "online" or "offline"
    p_threshold=0.8,
)

print(f"Predictions: {results}")
print(f"LLM fallback used: {fallback}")
```

For a more complete walkthrough, see [`example_usage.py`](example_usage.py) and [`QUICKSTART.md`](QUICKSTART.md).


## API Reference

### `monitor(...)` — Core Parameters

#### Required

- `task_id2classes` (`dict[str, list[str]]`): Mapping from task IDs to allowed class label lists.
  ```python
  {"task1": ["class1", "class2"], "task2": ["A", "B", "C"]}
  ```
- `text` (`str | list[str] | tuple[str, ...]`): Input text(s) to classify. Pass a single string for single-sample inference or a list/tuple for batch inference.
- `llm_fn` (callable): Teacher LLM function called on fallback samples.
  - Signature: `llm_fn(texts: list[str], task_id2classes, **kwargs)`
  - Return (one of):
    - `list[dict[task_id, label]]` — one dict per input (preferred)
    - `dict[task_id, list[label]]` — one list per task
    - `dict[task_id, label]` — legacy single-sample form (only when batch size is 1)
- `mode` (`str`): **Required keyword argument.** Inference mode. Must be `"online"` or `"offline"`.
  - `"online"` — student model is updated continuously during inference.
  - `"offline"` — student model is frozen; samples are selected up-front via `offline_select_method` / `offline_select_budget`.

#### Offline-Mode Parameters

These parameters are only used when `mode="offline"`.

| Parameter | Default | Description |
|---|---|---|
| `offline_select_method` | — | **Required** when `mode="offline"`. Sample-selection strategy (e.g. `"random"`, `"uncertainty"`). |
| `offline_select_budget` | — | **Required** when `mode="offline"`. Maximum number of samples to send to the teacher. |
| `offline_select_seed` | `42` | Random seed for offline sample selection. |
| `offline_select_probs` | `None` | Per-sample predicted probability array for uncertainty-based selection. |
| `offline_select_embeddings` | `None` | Per-sample embedding matrix for diversity-based selection. |
| `offline_select_mc_probs` | `None` | Monte-Carlo dropout probability matrix for BALD-style selection. |
| `offline_select_already_selected` | `None` | Indices already selected in previous rounds (excluded from new selection). |
| `offline_select_pool_size` | auto-derived | Total pool size. Inferred from `text` length when not provided. |

#### General Optional Parameters

| Parameter | Default | Description |
|---|---|---|
| `p_threshold` | `0.8` | Minimum confidence to trust the student prediction; falls back to LLM below this value. |
| `classifier_type` | `"mlp"` | Architecture of the student classification head. Options: `"mlp"`, `"linear"`, `"deep_mlp"`, `"cnn"`, `"gnn"`, `"gcp"`. Only takes effect the first time a task is seen. |
| `classifier_kwargs` | `None` | Architecture-specific keyword arguments (see table below). |
| `encoder` | `distilbert-base-uncased` | Custom encoder model instance. |
| `tokenizer` | auto | Custom tokenizer instance. |
| `hidden_size` | auto | Encoder hidden size; inferred automatically when `encoder` is not provided. |
| `device` | auto-detected | Compute device (`torch.device` or string). |
| `optimizer` | auto | Custom optimizer for the student classifier. |
| `scheduler` | `None` | Learning-rate scheduler passed to the training loop. |
| `llm_kwargs` | `{}` | Extra keyword arguments forwarded to `llm_fn`. |

##### `classifier_kwargs` by Architecture

**`"mlp"`** — 2-layer MLP with GELU activation and dropout *(default, no configurable kwargs)*

**`"linear"`** — Single linear layer (fastest, no hidden layer)

| Key | Type | Default | Description |
|---|---|---|---|
| `dropout` | float | `0.1` | Dropout probability applied before the linear layer. |

**`"deep_mlp"`** — Configurable-depth MLP with residual connections and LayerNorm

| Key | Type | Default | Description |
|---|---|---|---|
| `num_layers` | int | `3` | Number of hidden linear layers before the final projection (minimum 1). |
| `dropout` | float | `0.1` | Dropout probability applied after every hidden layer. |

```python
classifier_type="deep_mlp",
classifier_kwargs={"num_layers": 4, "dropout": 0.05}
```

**`"cnn"`** — Multi-scale 1-D CNN with parallel filter branches and global max-pooling

| Key | Type | Default | Description |
|---|---|---|---|
| `num_filters` | int | `128` | Number of output filters per kernel size. |
| `kernel_sizes` | tuple[int, ...] | `(3, 5, 7)` | Kernel widths for each parallel convolution branch. |
| `dropout` | float | `0.1` | Dropout probability. |

```python
classifier_type="cnn",
classifier_kwargs={"num_filters": 256, "kernel_sizes": (3, 5, 7, 9)}
```

**`"gnn"`** — GNN-inspired head using virtual-graph message passing over partitioned embeddings

| Key | Type | Default | Description |
|---|---|---|---|
| `num_nodes` | int | `12` | Number of virtual graph nodes. `hidden_size` must be divisible by `num_nodes`. |
| `num_layers` | int | `2` | Number of message-passing rounds. |
| `dropout` | float | `0.1` | Dropout probability. |

```python
classifier_type="gnn",
classifier_kwargs={"num_nodes": 8, "num_layers": 3}
```

**`"gcp"`** — Graph of Concept Predictors (DAG-structured reasoning head; see [paper](https://arxiv.org/abs/2602.03006))

| Key | Type | Default | Description |
|---|---|---|---|
| `edges` | list[tuple[int, int]] | *(required)* | Directed edges `(parent, child)` defining the concept DAG. Node indices must be contiguous starting at 0. The graph must be a valid DAG with no isolated nodes. |
| `concept_dim` | int | `256` | Feature dimension used for every concept node embedding. |
| `use_resnet` | bool | `False` | If `True`, each concept node's transform adds a residual skip connection with LayerNorm. |
| `dropout` | float | `0.1` | Dropout probability applied in all sub-modules. |

```python
# Linear chain: 0 → 1 → 2 → 3
classifier_type="gcp",
classifier_kwargs={"edges": [(0, 1), (1, 2), (2, 3)], "concept_dim": 128}

# Branching DAG: two roots merge, then project to output
# 0 → 2, 1 → 2, 2 → 3
classifier_type="gcp",
classifier_kwargs={"edges": [(0, 2), (1, 2), (2, 3)], "use_resnet": True}
```

#### Return Values

Returns a 2-tuple `(results, fallback)`. The exact types depend on whether the input was a single string or a batch:

| Input type | `results` type | `fallback` type |
|---|---|---|
| Single `str` | `dict[str, str]` — `{task_id: predicted_label}` | `bool` |
| `list[str]` / `tuple[str, ...]` | `list[dict[str, str]]` — one dict per input | `list[bool]` |


## How It Works

1. **Student prediction** — classify input with the small model
2. **Confidence estimation** — estimate prediction reliability
3. **Smart fallback**
   - If confidence ≥ `p_threshold`: return student output directly
   - If confidence < `p_threshold`: call the LLM teacher for labels
4. **Online learning** — store teacher labels in replay buffer and periodically retrain the student and correctness predictor


## Package Structure

```
llmcompiler/
├── __init__.py          # Package initialization, exports monitor
├── monitor.py           # Core API
├── task.py              # Task abstraction
├── models.py            # Model definitions
├── buffers.py           # Replay buffer
├── correctness.py       # Correctness predictor
├── trainer.py           # Training functions
├── registry.py          # Task registry
├── defaults.py          # Default configuration
├── setup.py             # Installation script
├── pyproject.toml       # Modern Python package config
└── requirements.txt     # Dependency list
```

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

This framework was developed by the LLMCompiler team at Emory University.

We thank our collaborators and students for discussions and feedback.
Portions of this system were inspired by work in online learning,
knowledge distillation, and adaptive inference.
