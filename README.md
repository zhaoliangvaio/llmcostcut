# llmcompiler

A framework for selectively invoking LLMs and distilling repeated workloads into smaller models.

This library provides infrastructure for:
- Correctness-aware LLM fallback
- Online distillation
- Replay-buffer–based incremental training

Used in research work: [Distilling LLM Reasoning into Graph of Concept Predictors](https://arxiv.org/abs/2602.03006)


## Figures

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
def my_llm(text, task_id2classes, **kwargs):
    # Your LLM call logic
    # Return format: {task_id: predicted_label}
    return {
        "sentiment": "positive",
        "topic": "technology"
    }

# 2. Define tasks and class labels
task_id2classes = {
    "sentiment": ["positive", "negative", "neutral"],
    "topic": ["sports", "politics", "technology"]
}

# 3. Run classification with monitor
results, fallback = monitor(
    task_id2classes=task_id2classes,
    text="I love this new phone!",
    llm_fn=my_llm,
    p_threshold=0.8  # Confidence threshold
)

print(f"Predictions: {results}")
print(f"LLM fallback used: {fallback}")
```

For a more complete walkthrough, see [`example_usage.py`](example_usage.py) and [`QUICKSTART.md`](QUICKSTART.md).


## API Reference

### `monitor(...)` — Core Parameters

#### Required

- `task_id2classes` (dict): Mapping from task IDs to class label lists
  ```python
  {"task1": ["class1", "class2"], "task2": ["A", "B", "C"]}
  ```
- `text` (str): Text to classify
- `llm_fn` (callable): Your LLM function
  - Signature: `llm_fn(text, task_id2classes, **kwargs) -> dict`
  - Return: `{task_id: predicted_label}`

#### Optional

| Parameter | Default | Description |
|---|---|---|
| `p_threshold` | `0.8` | Confidence threshold; falls back to LLM below this value |
| `encoder` | `distilbert-base-uncased` | Custom encoder model |
| `tokenizer` | — | Custom tokenizer |
| `device` | auto-detected | Compute device |
| `hidden_size` | — | Encoder hidden size |
| `optimizer` | — | Custom optimizer |
| `scheduler` | — | Learning rate scheduler |
| `llm_kwargs` | — | Extra arguments passed to `llm_fn` |

#### Return Values

- `results` (dict): Predictions in `{task_id: predicted_label}` format
- `fallback` (bool): `True` if the LLM teacher was called, `False` if the student model handled it


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

## Acknowledgements

This framework was developed by the LLMCompiler team at Emory University.

We thank our collaborators and students for discussions and feedback.
Portions of this system were inspired by work in online learning,
knowledge distillation, and adaptive inference.
