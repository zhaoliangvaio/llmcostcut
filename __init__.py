"""
LLMCompiler: Adaptive LLM-to-Small-Model Distillation Framework

主要 API:
    monitor: 核心函数，执行自适应推理和自动蒸馏
"""

from .monitor import monitor

__all__ = ['monitor']

