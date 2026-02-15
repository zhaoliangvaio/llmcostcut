# LLMCompiler 快速开始指南

## 最简单的使用方式

```python
from llmcompiler import monitor

# 1. 定义你的 LLM 函数（教师模型）
def my_llm(text, task_id2classes, **kwargs):
    # 你的 LLM 调用逻辑
    # 返回格式: {task_id: predicted_label}
    return {
        "sentiment": "positive",
        "topic": "technology"
    }

# 2. 定义任务和类别
task_id2classes = {
    "sentiment": ["positive", "negative", "neutral"],
    "topic": ["sports", "politics", "technology"]
}

# 3. 使用 monitor 进行分类
results, fallback = monitor(
    task_id2classes=task_id2classes,
    text="I love this new phone!",
    llm_fn=my_llm,
    p_threshold=0.8  # 置信度阈值
)

print(f"预测结果: {results}")
print(f"是否调用了 LLM: {fallback}")
```

## 核心参数说明

### 必需参数

- `task_id2classes` (dict): 任务ID到类别列表的映射
  ```python
  {
      "task1": ["class1", "class2", "class3"],
      "task2": ["A", "B", "C"]
  }
  ```

- `text` (str): 要分类的文本

- `llm_fn` (callable): 你的 LLM 函数
  - 函数签名: `llm_fn(text, task_id2classes, **kwargs) -> dict`
  - 返回值: `{task_id: predicted_label}`

### 可选参数

- `p_threshold` (float, 默认 0.8): 置信度阈值
  - 低于此值时回退到 LLM
  - 值越高，越频繁使用 LLM

- `encoder`: 自定义编码器（默认使用 distilbert-base-uncased）
- `tokenizer`: 自定义分词器
- `device`: 计算设备（默认自动检测）
- `hidden_size`: 编码器隐藏层大小
- `optimizer`: 自定义优化器
- `scheduler`: 学习率调度器
- `llm_kwargs`: 传递给 LLM 函数的额外参数

## 工作流程

1. **学生模型预测**: 使用小模型对输入进行分类
2. **置信度评估**: 评估预测的置信度
3. **智能回退**: 
   - 如果置信度 ≥ `p_threshold`: 直接返回学生模型结果
   - 如果置信度 < `p_threshold`: 调用 LLM 获取标注
4. **在线学习**: 
   - 将 LLM 标注存入回放缓冲区
   - 定期训练学生模型和正确性预测器

## 返回值

- `results` (dict): `{task_id: predicted_label}` 格式的预测结果
- `fallback` (bool): 是否调用了 LLM（True=调用了，False=使用学生模型）

## 完整示例

查看 `example_usage.py` 获取更多详细示例。

