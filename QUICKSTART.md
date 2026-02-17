# LLMCompiler Quick Start

## Simplest Usage

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

## Core Parameters

### Required

- `task_id2classes` (dict): Mapping from task IDs to class label lists
  ```python
  {
      "task1": ["class1", "class2", "class3"],
      "task2": ["A", "B", "C"]
  }
  ```

- `text` (str): Text to classify

- `llm_fn` (callable): Your LLM function
  - Signature: `llm_fn(text, task_id2classes, **kwargs) -> dict`
  - Return: `{task_id: predicted_label}`

### Optional

- `p_threshold` (float, default `0.8`): Confidence threshold
  - Falls back to LLM below this value
  - Higher values trigger LLM more often

- `encoder`: Custom encoder (default: `distilbert-base-uncased`)
- `tokenizer`: Custom tokenizer
- `device`: Compute device (auto-detected by default)
- `hidden_size`: Encoder hidden size
- `optimizer`: Custom optimizer
- `scheduler`: Learning rate scheduler
- `llm_kwargs`: Extra arguments passed to `llm_fn`

## Workflow

1. **Student prediction**: classify input with the small model
2. **Confidence estimation**: estimate prediction reliability
3. **Smart fallback**:
   - If confidence >= `p_threshold`: return student output directly
   - If confidence < `p_threshold`: call LLM teacher for labels
4. **Online learning**:
   - Store teacher labels in replay buffer
   - Periodically train student and correctness predictor

## Return Values

- `results` (dict): Predictions in `{task_id: predicted_label}` format
- `fallback` (bool): Whether LLM fallback was used (`True`) or not (`False`)

## Full Example

See `example_usage.py` for a more complete walkthrough.

