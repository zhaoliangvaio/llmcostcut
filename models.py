"""
LLMCompiler: Adaptive LLM-to-Small-Model Distillation Framework

Copyright (c) 2024–2025
Liang Zhao and Ziyang Yu
Emory University

This file is part of the LLMCompiler framework.
Released under the Apache 2.0 License (see LICENSE).

If you use this code in academic work, please cite:
<Paper citation to appear>

Contact:
Liang Zhao (liang.zhao@emory.edu)
"""
import torch
import torch.nn as nn
from typing import List
class Classifier(nn.Module):
    """Lightweight MLP classification head over frozen-encoder CLS embeddings."""
    def __init__(self, hidden_size=768, num_labels=4):
        super().__init__()
        self.dropout = nn.Dropout(0.1)  # Dropout layer for regularization
        self.dense = nn.Linear(hidden_size, hidden_size)  # Fully connected layer
        self.activation = nn.GELU()  # GELU activation
        self.classifier = nn.Linear(hidden_size, num_labels)  # Final classification layer
    
    def forward(self, x):
        """Forward pass: classify encoder features."""
        x = self.dropout(x)
        x = self.dense(x)
        x = self.activation(x)
        x = self.dropout(x)
        return self.classifier(x)

@torch.no_grad()
def annotate_with_classifier(encoder, classifier, tokenizer, texts: List[str], device) -> List[int]:
    """Fallback path: use current model predictions as pseudo labels when LLM is unavailable."""
    classifier.eval()
    enc = tokenizer(
        texts,
        truncation=True,
        max_length=128,
        padding=True,
        return_tensors="pt",
    )
    enc = {k: v.to(device) for k, v in enc.items()}
    outputs = encoder(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"]).last_hidden_state[:, 0]
    logits = classifier(outputs)
    return logits, outputs