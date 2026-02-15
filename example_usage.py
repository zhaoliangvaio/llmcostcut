"""
LLMCompiler 使用示例

展示如何将 LLMCompiler 框架接入到你的代码中。
"""
import torch
from llmcompiler.monitor import monitor


# ============================================
# 步骤 1: 定义你的 LLM 教师函数
# ============================================
def your_llm_teacher(text, task_id2classes, **kwargs):
    """
    你的 LLM 调用函数（教师模型）
    
    Args:
        text: 输入文本
        task_id2classes: 任务ID到类别列表的映射
        **kwargs: 其他参数（如 temperature, max_tokens 等）
    
    Returns:
        dict: {task_id: predicted_label} 格式的字典
    """
    # 示例：使用 OpenAI API
    # import openai
    # response = openai.ChatCompletion.create(
    #     model="gpt-3.5-turbo",
    #     messages=[{"role": "user", "content": text}],
    #     **kwargs
    # )
    # result = response.choices[0].message.content
    
    # 示例：使用本地模型
    # from transformers import pipeline
    # classifier = pipeline("text-classification", model="your-model")
    # result = classifier(text)
    
    # 这里是一个简单的示例实现
    results = {}
    for task_id, classes in task_id2classes.items():
        # 你的 LLM 推理逻辑
        # 这里只是示例，你需要替换为实际的 LLM 调用
        results[task_id] = classes[0]  # 示例：返回第一个类别
    
    return results


# ============================================
# 步骤 2: 配置任务和类别
# ============================================
def setup_tasks():
    """
    定义你要分类的任务和对应的类别
    """
    task_id2classes = {
        "sentiment": ["positive", "negative", "neutral"],
        "topic": ["sports", "politics", "technology", "entertainment"],
        # 可以添加更多任务
        # "intent": ["question", "command", "statement"],
    }
    return task_id2classes


# ============================================
# 步骤 3: 基本使用示例
# ============================================
def basic_usage_example():
    """基本使用示例"""
    # 配置任务
    task_id2classes = setup_tasks()
    
    # 输入文本
    text = "I love this new smartphone! It's amazing."
    
    # 调用 monitor 函数
    results, fallback = monitor(
        task_id2classes=task_id2classes,
        text=text,
        llm_fn=your_llm_teacher,
        llm_kwargs=None,  # 传递给 llm_fn 的额外参数
        p_threshold=0.8,  # 置信度阈值，低于此值会回退到 LLM
    )
    
    print(f"预测结果: {results}")
    print(f"是否使用了 LLM: {fallback}")
    
    return results, fallback


# ============================================
# 步骤 4: 自定义编码器和设备
# ============================================
def advanced_usage_example():
    """高级使用示例：自定义编码器和优化器"""
    from transformers import AutoModel, AutoTokenizer
    from torch.optim import AdamW
    
    # 自定义编码器（可选，默认使用 distilbert-base-uncased）
    encoder = AutoModel.from_pretrained("bert-base-uncased")
    for p in encoder.parameters():
        p.requires_grad = False  # 冻结编码器
    
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    hidden_size = 768  # BERT 的隐藏层大小
    
    # 自定义设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder.to(device)
    
    # 配置任务
    task_id2classes = setup_tasks()
    
    # 自定义优化器（可选）
    # 注意：optimizer 会在 monitor 内部为每个任务创建，这里只是示例
    # 如果需要自定义，可以在 monitor 调用后手动设置
    
    # 调用 monitor
    text = "The stock market crashed today."
    results, fallback = monitor(
        task_id2classes=task_id2classes,
        text=text,
        llm_fn=your_llm_teacher,
        encoder=encoder,
        tokenizer=tokenizer,
        device=device,
        hidden_size=hidden_size,
        p_threshold=0.75,  # 更低的阈值，更频繁地使用 LLM
    )
    
    return results, fallback


# ============================================
# 步骤 5: 批量处理示例
# ============================================
def batch_processing_example():
    """批量处理多个文本"""
    task_id2classes = setup_tasks()
    
    texts = [
        "This movie is fantastic!",
        "The weather is terrible today.",
        "I need to buy groceries.",
    ]
    
    all_results = []
    llm_call_count = 0
    
    for text in texts:
        results, fallback = monitor(
            task_id2classes=task_id2classes,
            text=text,
            llm_fn=your_llm_teacher,
            p_threshold=0.8,
        )
        all_results.append(results)
        if fallback:
            llm_call_count += 1
    
    print(f"处理了 {len(texts)} 个文本")
    print(f"LLM 调用次数: {llm_call_count}")
    print(f"LLM 调用率: {llm_call_count / len(texts) * 100:.2f}%")
    
    return all_results


# ============================================
# 步骤 6: 与现有代码集成
# ============================================
class YourApplication:
    """示例：将 LLMCompiler 集成到你的应用中"""
    
    def __init__(self):
        self.task_id2classes = setup_tasks()
        self.llm_call_count = 0
        self.total_calls = 0
    
    def classify(self, text, **llm_kwargs):
        """
        分类接口，自动使用学生模型或回退到 LLM
        """
        self.total_calls += 1
        
        results, fallback = monitor(
            task_id2classes=self.task_id2classes,
            text=text,
            llm_fn=your_llm_teacher,
            llm_kwargs=llm_kwargs,
            p_threshold=0.8,
        )
        
        if fallback:
            self.llm_call_count += 1
        
        return results
    
    def get_statistics(self):
        """获取统计信息"""
        return {
            "total_calls": self.total_calls,
            "llm_calls": self.llm_call_count,
            "student_model_usage_rate": (1 - self.llm_call_count / max(self.total_calls, 1)) * 100,
        }


# ============================================
# 主函数：运行示例
# ============================================
if __name__ == "__main__":
    print("=" * 60)
    print("LLMCompiler 使用示例")
    print("=" * 60)
    
    print("\n1. 基本使用示例:")
    basic_usage_example()
    
    print("\n2. 批量处理示例:")
    batch_processing_example()
    
    print("\n3. 应用集成示例:")
    app = YourApplication()
    app.classify("I love this product!")
    app.classify("This is terrible.")
    stats = app.get_statistics()
    print(f"统计信息: {stats}")

