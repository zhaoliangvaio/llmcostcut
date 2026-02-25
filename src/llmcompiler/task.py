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

"""
task.py

Defines the Task abstraction.

A Task encapsulates:
- One classification objective
- Its student classifier
- Its correctness predictor
- Its replay buffer
- Training state counters

Tasks are managed by TaskRegistry and consumed by monitor().
"""
import hashlib
from .buffers import ReplayBufferManager
from .models import build_classifier
from .correctness import OnlineCorrectnessPredictor

class Task:
    def __init__(
        self,
        task_id,
        classes,
        encoder,
        tokenizer,
        device,
        hidden_size,
        classifier_type="deep_mlp",
        classifier_kwargs=None,
    ):
        self.task_id = task_id
        self.classes = classes
        self.num_labels = len(classes)
        self.device = device
        self.step = 0                 # how many samples processed
        self.last_train_step = 0
        self.num_labeled = 0
        self.last_train_labeled = 0
        self.num_train_rounds = 0
        self.last_correctness_train_step = 0
        self.last_correctness_train_labeled = 0
        self.num_correctness_train_rounds = 0
        # Sub-module retraining state (GCP only; §3.4 arXiv:2602.03006)
        self.last_submodule_retrain_step = 0
        self.num_submodule_retrain_rounds = 0
        self.optimizer = None
        # === student classifier ===
        self.classifier = build_classifier(
            classifier_type=classifier_type,
            hidden_size=hidden_size,
            num_labels=self.num_labels,
            **(classifier_kwargs or {}),
        ).to(device)

        # === correctness predictor ===
        self.correctness = OnlineCorrectnessPredictor(
            encoding_dim=hidden_size,
            num_classes=self.num_labels,
            device=device
        )

        # === replay buffer ===
        self.buffers = ReplayBufferManager()

        # === shared encoder/tokenizer ===
        self.encoder = encoder
        self.tokenizer = tokenizer
        self.label2id = {label: i for i, label in enumerate(classes)}
        self.id2label = {i: label for label, i in self.label2id.items()}

    def workflow_id(self):
        return hashlib.sha256(self.task_id.encode()).hexdigest()

