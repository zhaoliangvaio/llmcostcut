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
        mode="online",
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
        mode="online",
        p_threshold=0.75,  # Lower threshold, more frequent LLM fallback
    )
    
    return results, fallback


# ============================================
# Step 4b: All classifier type examples
# ============================================
def all_classifier_types_example():
    """Demonstrate every available classifier_type.

    All six architectures accept the same ``monitor()`` interface.
    The head is created once on the first call and reused afterwards.
    """
    task_id2classes = setup_tasks()
    text = "Scientists announced a major breakthrough in renewable energy."

    # ── 1. mlp (default) ──────────────────────────────────────────────────────
    # 2-layer MLP with GELU activation and dropout.  Good all-round baseline.
    results_mlp, _ = monitor(
        task_id2classes=task_id2classes,
        text=text,
        llm_fn=your_llm_teacher,
        mode="online",
        classifier_type="mlp",
    )
    print(f"[mlp]      {results_mlp}")

    # ── 2. linear ─────────────────────────────────────────────────────────────
    # Single linear layer – fastest inference, best when encoder is fine-tuned
    # end-to-end or the task is linearly separable.
    results_linear, _ = monitor(
        task_id2classes=task_id2classes,
        text=text,
        llm_fn=your_llm_teacher,
        mode="online",
        classifier_type="linear",
    )
    print(f"[linear]   {results_linear}")

    # ── 3. deep_mlp ───────────────────────────────────────────────────────────
    # Configurable-depth MLP with residual connections and LayerNorm.
    # Use num_layers to control depth; num_layers=1 degrades to a plain MLP.
    results_deep, _ = monitor(
        task_id2classes=task_id2classes,
        text=text,
        llm_fn=your_llm_teacher,
        mode="online",
        classifier_type="deep_mlp",
        classifier_kwargs={"num_layers": 4, "dropout": 0.05},
    )
    print(f"[deep_mlp] {results_deep}")

    # ── 4. cnn ────────────────────────────────────────────────────────────────
    # Multi-scale 1-D CNN: parallel conv filters of different widths, global
    # max-pool, then a linear projection.
    results_cnn, _ = monitor(
        task_id2classes=task_id2classes,
        text=text,
        llm_fn=your_llm_teacher,
        mode="online",
        classifier_type="cnn",
        classifier_kwargs={"num_filters": 256, "kernel_sizes": (3, 5, 7, 9)},
    )
    print(f"[cnn]      {results_cnn}")

    # ── 5. gnn ────────────────────────────────────────────────────────────────
    # GNN-inspired head: partitions the CLS embedding into virtual graph nodes
    # and runs attention-weighted message passing.
    # Constraint: hidden_size (768 for distilbert) must be divisible by num_nodes.
    results_gnn, _ = monitor(
        task_id2classes=task_id2classes,
        text=text,
        llm_fn=your_llm_teacher,
        mode="online",
        classifier_type="gnn",
        classifier_kwargs={"num_nodes": 8, "num_layers": 3},
    )
    print(f"[gnn]      {results_gnn}")

    # ── 6. gcp ────────────────────────────────────────────────────────────────
    # Graph of Concept Predictors: a DAG-structured head where each node
    # represents a reasoning concept.  The topology is supplied as a list of
    # directed (parent, child) edge pairs.
    #
    # Example graph: a linear chain  0 → 1 → 2 → 3
    #   node 0: root (receives CLS embedding)
    #   node 3: sink (contributes to the final prediction)
    #
    # During training, every node's concept predictor is trained simultaneously
    # alongside the final head (concept-level supervision via forward_with_concepts).
    results_gcp, _ = monitor(
        task_id2classes=task_id2classes,
        text=text,
        llm_fn=your_llm_teacher,
        mode="online",
        classifier_type="gcp",
        classifier_kwargs={
            "edges": [(0, 1), (1, 2), (2, 3)],
            "concept_dim": 256,
            "use_resnet": True,
            "dropout": 0.1,
        },
    )
    print(f"[gcp]      {results_gcp}")

    return {
        "mlp": results_mlp,
        "linear": results_linear,
        "deep_mlp": results_deep,
        "cnn": results_cnn,
        "gnn": results_gnn,
        "gcp": results_gcp,
    }


def gcp_branching_dag_example():
    """GCPClassifier with a branching DAG: two root nodes merge at a shared node.

    Graph topology::

        0 ──→ 2 ──→ 3
        1 ──→ 2

    Nodes 0 and 1 are roots (no parents); node 2 receives both.
    Node 3 is the sink whose embedding is projected to the final label.
    """
    task_id2classes = {"intent": ["question", "command", "statement", "greeting"]}

    def mock_teacher(texts, task_id2classes, **kwargs):
        return [{"intent": "question"} for _ in texts]

    text = "Can you help me find the nearest coffee shop?"
    result, fallback = monitor(
        task_id2classes=task_id2classes,
        text=text,
        llm_fn=mock_teacher,
        mode="online",
        classifier_type="gcp",
        classifier_kwargs={
            "edges": [(0, 2), (1, 2), (2, 3)],
            "concept_dim": 128,
            "use_resnet": False,
        },
    )
    print(f"[gcp branching DAG] intent={result['intent']}  fallback={fallback}")
    return result, fallback


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
            mode="online",
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
            mode="online",
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

    print("\n4. All classifier types example:")
    all_classifier_types_example()    print("\n5. GCP branching DAG example:")
    gcp_branching_dag_example()
