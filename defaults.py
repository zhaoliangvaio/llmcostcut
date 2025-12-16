"""
LLMCompiler: Adaptive LLM-to-Small-Model Distillation Framework

Copyright (c) 2024–2025
Liang Zhao and collaborators
Emory University

This file is part of the LLMCompiler framework.
Released under the Apache 2.0 License (see LICENSE).

If you use this code in academic work, please cite:
<Paper citation to appear>

Contact:
Liang Zhao (liang.zhao@emory.edu)
"""
import torch
from transformers import AutoTokenizer, AutoModel, AutoConfig
from torch.optim import AdamW

_DEFAULTS = {}

def get_device(device=None):
    return device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_encoder(model_name="distilbert-base-uncased", device=None):
    key = ("encoder", model_name)
    if key not in _DEFAULTS:
        config = AutoConfig.from_pretrained(model_name)
        encoder = AutoModel.from_pretrained(model_name, config=config)
        for p in encoder.parameters():
            p.requires_grad = False
        encoder.to(get_device(device))
        _DEFAULTS[key] = (encoder, config.hidden_size)
    return _DEFAULTS[key]

def get_tokenizer(model_name="distilbert-base-uncased"):
    key = ("tokenizer", model_name)
    if key not in _DEFAULTS:
        _DEFAULTS[key] = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    return _DEFAULTS[key]

def get_optimizer(classifier, lr=5e-4, weight_decay=1e-3):
    return AdamW(classifier.parameters(), lr=lr, weight_decay=weight_decay)
