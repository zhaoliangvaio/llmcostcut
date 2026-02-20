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
from .task import Task

class TaskRegistry:
    def __init__(self):
        self.tasks = {}

    def get_or_create(self, task_id, classes, **kwargs):
        if task_id not in self.tasks:
            self.tasks[task_id] = Task(task_id, classes, **kwargs)
        return self.tasks[task_id]
