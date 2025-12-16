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

"""
monitor.py

Core orchestration logic for LLMCompiler.

Responsibilities:
- Route inputs through task-specific student models
- Predict student correctness
- Decide when to fall back to teacher LLM
- Trigger online training for student and correctness models
- Maintain per-task lifecycle and training policy

This file defines the main public API: `monitor(...)`.
"""
import math
import torch
from torch.utils.data import DataLoader

from .registry import TaskRegistry
from .models import annotate_with_classifier
from .trainer import train_one_round_buff
from .defaults import (
    get_device,
    get_encoder,
    get_tokenizer,
    get_optimizer,
)

# -----------------------------
# Training policy (tunable)
# -----------------------------
NEW_LABELS_TRIGGER = 10        # like args.num_labeled_per_iteration
MAX_TRAIN_ROUNDS   = 500
REPLAY_SAMPLE_SIZE = 10_000
TRAIN_BATCH_SIZE   = 32
CORRECTNESS_BATCH  = 64
CORRECTNESS_STEPS  = 3

def _should_train(task):
    return (
        task.num_labeled - task.last_train_labeled >= NEW_LABELS_TRIGGER
        and task.num_train_rounds < MAX_TRAIN_ROUNDS
    )

def _compute_steps(num_labeled):
    # distilled from your code: steps ∝ log(L + 10)
    return int(80 * math.log(num_labeled + 10))

def _should_fallback(p_corrects,threshold):
    if min(p_corrects.values())<threshold:
        return True
    return False

_registry = TaskRegistry()

def monitor(
    task_id2classes,
    text,
    llm_fn,
    llm_kwargs=None,
    *,
    encoder=None,
    tokenizer=None,
    device=None,
    hidden_size=None,
    optimizer=None,
    scheduler=None,
    p_threshold=0.8,
):
    """
    Execute adaptive inference with automatic LLM distillation.

    Args:
        task_id2classes (dict[str, list[str]]):
            Mapping from task name to allowed class labels.
        text (str):
            Input text to be classified.
        llm_fn (callable):
            Teacher function. Must return dict[task_id -> label].
        llm_kwargs (dict, optional):
            Extra arguments forwarded to llm_fn.



            
        p_threshold (float):
            Minimum confidence to trust student prediction.

    Returns:
        results (dict[str, str]):
            Predicted label per task.
        fallback (bool):
            Whether teacher LLM was invoked.

    Notes:
        - Student models are updated online.
        - Teacher calls are minimized automatically.
        - No gradients flow through the teacher.
    """
    TRAIN_CLASSIFIER_EVERY = 50      # t steps
    TRAIN_CORRECTNESS_EVERY = 20     # t steps
    results = {}
    device = get_device(device)

    if encoder is None or hidden_size is None:
        encoder, hidden_size = get_encoder(device=device)

    if tokenizer is None:
        tokenizer = get_tokenizer()
    
    # preds = {_task: None for _task in task_id2classes.keys()}
    # p_corrects = {_task: None for _task in task_id2classes.keys()}
    caches = {_task: {'task': None,'classes':None,
            'logits':None,'embeddings':None, 'pred':None,
            'p_correct':None} for _task in task_id2classes.keys()}
    for task_id, classes in task_id2classes.items():
        task = _registry.get_or_create(
            task_id,
            classes,
            encoder=encoder,
            tokenizer=tokenizer,
            device=device,
            hidden_size=hidden_size
        )
        if task.optimizer is None:
            task.optimizer = optimizer or get_optimizer(task.classifier)

        # ---- encode text ----
        logits, embeddings = annotate_with_classifier(
            encoder,
            task.classifier,
            tokenizer,
            [text],
            device
        )

        pred = logits.argmax(dim=-1).item()

        # ---- correctness ----
        p_correct = task.correctness.predict_p_correct(
            encoding=embeddings[0],
            student_logits=logits[0],
            input_len=len(text)
        )
        caches[task_id]['pred'] = task.id2label[pred]
        caches[task_id]['p_correct'] = p_correct
        caches[task_id]['task'] = task
        caches[task_id]['classes'] = classes
        caches[task_id]['logits'] = logits
        caches[task_id]['embeddings'] = embeddings
        # preds[task_id] = task.id2label[pred]
        # p_corrects[task_id] = p_correct
        # if p_correct >= p_threshold:
        #     results[task_id] = pred
        #     continue

        # ---- teacher fallback ----
    fallback = _should_fallback({_task_id: content['p_correct'] for _task_id, content in caches.items()},p_threshold)
    if not fallback:
        for task_id in task_id2classes.keys():
            results[task_id] = caches[task_id]['pred']
    else:
        llm_kwargs = llm_kwargs or {}
        teacher_out = llm_fn(text, task_id2classes, **llm_kwargs)

        for k in teacher_out:
            if teacher_out[k] == None:
                raise(f"the Task {k} has returned any label")
        for task_id, classes in task_id2classes.items():
            # label = teacher_out[task_id]
            task = caches[task_id]['task']
            pred = task.label2id[caches[task_id]['pred']]
            p_correct = caches[task_id]['p_correct']
            classes = caches[task_id]['classes']
            logits = caches[task_id]['logits']
            embeddings = caches[task_id]['embeddings']
            label_str = teacher_out[task_id]
            label_id = task.label2id[label_str]

            task.num_labeled += 1

            # ---- update correctness ----
            task.correctness.add_training_example(
                encoding=embeddings[0],
                student_logits=logits[0],
                input_len=len(text),
                teacher_label=label_id,
                student_label=pred
            )

            # ---- store in replay buffer ----
            task.buffers.add_sample(
                workflow_id=task.workflow_id(),
                text=text,
                encoding=embeddings[0],
                label=label_id,
                student_pred=pred,
                confidence=p_correct
            )
            results[task_id] = label_str

            # ---- incremental training ----
            
            
            if _should_train(task):
                buf = task.buffers.get_buffer(task.workflow_id())
                if buf.size > 0:
                    steps = _compute_steps(task.num_labeled)
                    data = task.buffers.sample_for_training(task.workflow_id(),REPLAY_SAMPLE_SIZE)
                    loader = torch.utils.data.DataLoader(data, batch_size=TRAIN_BATCH_SIZE, shuffle=True)

                    train_one_round_buff(
                        task.classifier,
                        loader,
                        device,
                        task.optimizer,
                        scheduler,
                        steps_per_round=50
                    )
                    task.last_train_labeled = task.num_labeled
                    task.num_train_rounds += 1

            # ---- train correctness predictor ----
            if task.step % TRAIN_CORRECTNESS_EVERY == 0:
                task.correctness.train_step(batch=CORRECTNESS_BATCH, steps=CORRECTNESS_STEPS)

    return results, fallback
