import torch
import torch.nn as nn
from typing import List
class Classifier(nn.Module):
    """轻量级 MLP 分类头，基于冻结编码器的 CLS 嵌入"""
    def __init__(self, hidden_size=768, num_labels=4):
        super().__init__()
        self.dropout = nn.Dropout(0.1)  # Dropout 层用于正则化
        self.dense = nn.Linear(hidden_size, hidden_size)  # 全连接层
        self.activation = nn.GELU()  # GELU 激活函数
        self.classifier = nn.Linear(hidden_size, num_labels)  # 最终分类层
    
    def forward(self, x):
        """前向传播：对编码器特征进行分类"""
        x = self.dropout(x)
        x = self.dense(x)
        x = self.activation(x)
        x = self.dropout(x)
        return self.classifier(x)

@torch.no_grad()
def annotate_with_classifier(encoder, classifier, tokenizer, texts: List[str], device) -> List[int]:
    """回退方案：当 LLM 不可用时，使用当前模型预测作为伪标签"""
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