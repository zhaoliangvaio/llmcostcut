"""
LLMCompiler usage examples.

Demonstrates how to integrate the LLMCompiler framework into your codebase.
"""
import torch
from llmcompiler.monitor import monitor


# ============================================
# Step 1: Define your teacher LLM function
# ============================================
def your_llm_teacher(text, task_id2classes, **kwargs):
    """
    Your LLM call function (teacher model).
    
    Args:
        text: Input text.
        task_id2classes: Mapping from task IDs to class lists.
        **kwargs: Additional parameters (for example temperature, max_tokens).
    
    Returns:
        dict: A dictionary in the format {task_id: predicted_label}.
    """
    # Example: use OpenAI API
    # import openai
    # response = openai.ChatCompletion.create(
    #     model="gpt-3.5-turbo",
    #     messages=[{"role": "user", "content": text}],
    #     **kwargs
    # )
    # result = response.choices[0].message.content
    
    # Example: use a local model
    # from transformers import pipeline
    # classifier = pipeline("text-classification", model="your-model")
    # result = classifier(text)
    
    # This is a simple placeholder implementation
    results = {}
    for task_id, classes in task_id2classes.items():
        # Your LLM inference logic
        # This is only an example; replace with your real LLM call
        results[task_id] = classes[0]  # Example: return the first class
    
    return results


# ============================================
# Step 2: Configure tasks and classes
# ============================================
def setup_tasks():
    """
    Define the tasks and corresponding classes to classify.
    """
    task_id2classes = {
        "sentiment": ["positive", "negative", "neutral"],
        "topic": ["sports", "politics", "technology", "entertainment"],
        # You can add more tasks
        # "intent": ["question", "command", "statement"],
    }
    return task_id2classes


# ============================================
# Step 3: Basic usage example
# ============================================
def basic_usage_example():
    """Basic usage example."""
    # Configure tasks
    task_id2classes = setup_tasks()
    
    # Input text
    text = "I love this new smartphone! It's amazing."
    
    # Call monitor
    results, fallback = monitor(
        task_id2classes=task_id2classes,
        text=text,
        llm_fn=your_llm_teacher,
        llm_kwargs=None,  # Extra parameters passed to llm_fn
        p_threshold=0.8,  # Confidence threshold; below this value it falls back to LLM
    )
    
    print(f"Prediction results: {results}")
    print(f"Used LLM fallback: {fallback}")
    
    return results, fallback


# ============================================
# Step 4: Customize encoder and device
# ============================================
def advanced_usage_example():
    """Advanced usage example: customize encoder and optimizer."""
    from transformers import AutoModel, AutoTokenizer
    from torch.optim import AdamW
    
    # Custom encoder (optional; defaults to distilbert-base-uncased)
    encoder = AutoModel.from_pretrained("bert-base-uncased")
    for p in encoder.parameters():
        p.requires_grad = False  # Freeze encoder
    
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    hidden_size = 768  # BERT hidden size
    
    # Custom device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder.to(device)
    
    # Configure tasks
    task_id2classes = setup_tasks()
    
    # Custom optimizer (optional)
    # Note: optimizer is created per task inside monitor; this is only an example
    # If needed, you can customize it after calling monitor
    
    # Call monitor
    text = "The stock market crashed today."
    results, fallback = monitor(
        task_id2classes=task_id2classes,
        text=text,
        llm_fn=your_llm_teacher,
        encoder=encoder,
        tokenizer=tokenizer,
        device=device,
        hidden_size=hidden_size,
        p_threshold=0.75,  # Lower threshold, more frequent LLM fallback
    )
    
    return results, fallback


# ============================================
# Step 5: Batch processing example
# ============================================
def batch_processing_example():
    """Process multiple texts in batch."""
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
    
    print(f"Processed {len(texts)} texts")
    print(f"LLM calls: {llm_call_count}")
    print(f"LLM call rate: {llm_call_count / len(texts) * 100:.2f}%")
    
    return all_results


# ============================================
# Step 6: Integrate with existing code
# ============================================
class YourApplication:
    """Example: integrate LLMCompiler into your application."""
    
    def __init__(self):
        self.task_id2classes = setup_tasks()
        self.llm_call_count = 0
        self.total_calls = 0
    
    def classify(self, text, **llm_kwargs):
        """
        Classification interface that automatically uses student model
        or falls back to the LLM.
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
        """Get runtime statistics."""
        return {
            "total_calls": self.total_calls,
            "llm_calls": self.llm_call_count,
            "student_model_usage_rate": (1 - self.llm_call_count / max(self.total_calls, 1)) * 100,
        }


# ============================================
# Main entry: run examples
# ============================================
if __name__ == "__main__":
    print("=" * 60)
    print("LLMCompiler Usage Examples")
    print("=" * 60)
    
    print("\n1. Basic usage example:")
    basic_usage_example()
    
    print("\n2. Batch processing example:")
    batch_processing_example()
    
    print("\n3. Application integration example:")
    app = YourApplication()
    app.classify("I love this product!")
    app.classify("This is terrible.")
    stats = app.get_statistics()
    print(f"Statistics: {stats}")

