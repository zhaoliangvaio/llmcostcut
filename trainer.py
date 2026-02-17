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
def train_one_round_buff(classifier, loader, device, optimizer, scheduler=None,
                         steps_per_round=200, max_grad_norm=1.0, train_tag=None, log_fn=None):
    """Online incremental training: run fixed number of SGD steps each round"""

    classifier.train()
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    ### >>> UPDATED: remove optimizer redefinition – keep existing optimizer persistent
    # no_decay = ["bias", "LayerNorm.weight"]
    # optimizer_grouped_parameters = [...]
    # (removed)
    ### <<< UPDATED

    step = 0
    running_loss = 0.0
    loader_iter = iter(loader)

    while step < steps_per_round:

        try:
            batch = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            batch = next(loader_iter)

        batch = {k: v.to(device) for k, v in batch.items() if k != 'text'}

        logits = classifier(batch['encoding'])
        loss = criterion(logits, batch['label'])
        running_loss += float(loss.item())

        loss.backward()
        nn.utils.clip_grad_norm_(classifier.parameters(), max_grad_norm)

        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        ### >>> UPDATED
        if scheduler is not None:
            scheduler.step()
        ### <<< UPDATED

        step += 1

    avg_loss = running_loss / max(1, step)
    prefix = f"[Train:{train_tag}]" if train_tag else "[Train]"
    msg = f"{prefix} steps={step} avg_loss={avg_loss:.6f}"
    print(msg)
    if callable(log_fn):
        log_fn(msg)