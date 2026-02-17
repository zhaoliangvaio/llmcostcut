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
import time
import threading
from pathlib import Path
from typing import Sequence
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
TRAIN_CLASSIFIER_EVERY = 20    # trigger classifier when processed samples cross each interval
TRAIN_CORRECTNESS_EVERY = 20   # trigger correctness when processed samples cross each interval
CORRECTNESS_BATCH  = 64
CORRECTNESS_STEPS  = 3

def _crossed_sample_interval(current_step, last_train_step, interval):
    if interval <= 0:
        return False
    return (current_step // interval) > (last_train_step // interval)

def _should_train(task):
    return (
        _crossed_sample_interval(task.step, task.last_train_step, TRAIN_CLASSIFIER_EVERY)
        and task.num_train_rounds < MAX_TRAIN_ROUNDS
    )

def _should_train_correctness(task):
    return (
        _crossed_sample_interval(task.step, task.last_correctness_train_step, TRAIN_CORRECTNESS_EVERY)
        and task.num_correctness_train_rounds < MAX_TRAIN_ROUNDS
    )

def _compute_steps(num_labeled):
    # distilled from your code: steps ∝ log(L + 10)
    return int(80 * math.log(num_labeled + 10))

def _should_fallback(p_corrects,threshold):
    if min(p_corrects.values())<threshold:
        return True
    return False

_registry = TaskRegistry()
_classifier_locks = {}
_correctness_locks = {}
_training_locks_guard = threading.Lock()
MONITOR_TIMING_LOG = True
MONITOR_TIMING_STDOUT = False
_MONITOR_TIMING_FILE = Path(__file__).resolve().parents[1] / "outputs" / "time_consumption" / "monitor_timing.log"
_MONITOR_TIMING_LOCK = threading.Lock()

def _get_task_lock(lock_map, task_id):
    with _training_locks_guard:
        if task_id not in lock_map:
            lock_map[task_id] = threading.Lock()
        return lock_map[task_id]

def _run_in_background_with_lock(lock_map, task_id, name, fn, *args, **kwargs):
    def _target():
        task_lock = _get_task_lock(lock_map, task_id)
        with task_lock:
            try:
                fn(*args, **kwargs)
            except Exception as err:
                print(f"[monitor] background {name} training failed for {task_id}: {err}")

    threading.Thread(target=_target, daemon=True).start()

def _log_monitor_timing(msg):
    if not MONITOR_TIMING_LOG:
        return
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"{ts} {msg}\n"
    with _MONITOR_TIMING_LOCK:
        try:
            _MONITOR_TIMING_FILE.parent.mkdir(parents=True, exist_ok=True)
            with _MONITOR_TIMING_FILE.open("a", encoding="utf-8") as f:
                f.write(line)
        except Exception as e:
            print(f"[monitor_timing] failed to write log file: {e}")
            print(line, end="")
    if MONITOR_TIMING_STDOUT:
        print(line, end="")

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
        text (str | list[str] | tuple[str, ...]):
            Input text(s) to be classified.
        llm_fn (callable):
            Teacher function. Must return dict[task_id -> label].
        llm_kwargs (dict, optional):
            Extra arguments forwarded to llm_fn.



            
        p_threshold (float):
            Minimum confidence to trust student prediction.

    Returns:
        results (dict[str, str] | list[dict[str, str]]):
            Predicted label per task. Returns a list when batch input is used.
        fallback (bool | list[bool]):
            Whether teacher LLM was invoked. Returns a list when batch input is used.

    Notes:
        - Student models are updated online.
        - Teacher calls are minimized automatically.
        - No gradients flow through the teacher.
    """
    monitor_t0 = time.perf_counter()
    task_timing = {}
    is_single_input = isinstance(text, str)
    if is_single_input:
        texts = [text]
    elif isinstance(text, Sequence):
        texts = list(text)
        if any(not isinstance(t, str) for t in texts):
            raise TypeError("When 'text' is a sequence, every item must be a string.")
    else:
        raise TypeError("'text' must be a string or a sequence of strings.")

    batch_size = len(texts)
    if batch_size == 0:
        return [], []

    all_results = [{} for _ in range(batch_size)]
    t0 = time.perf_counter()
    device = get_device(device)
    setup_device_ms = (time.perf_counter() - t0) * 1000.0

    t0 = time.perf_counter()
    if encoder is None or hidden_size is None:
        encoder, hidden_size = get_encoder(device=device)
    setup_encoder_ms = (time.perf_counter() - t0) * 1000.0

    t0 = time.perf_counter()
    if tokenizer is None:
        tokenizer = get_tokenizer()
    setup_tokenizer_ms = (time.perf_counter() - t0) * 1000.0
    
    # preds = {_task: None for _task in task_id2classes.keys()}
    # p_corrects = {_task: None for _task in task_id2classes.keys()}
    caches = {
        _task: {
            'task': None,
            'classes': None,
            'logits': None,
            'embeddings': None,
            'preds': [None for _ in range(batch_size)],
            'p_corrects': [None for _ in range(batch_size)],
        }
        for _task in task_id2classes.keys()
    }
    for task_id, classes in task_id2classes.items():
        task_metrics = {}
        t0 = time.perf_counter()
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
        task.step += batch_size
        task_metrics["task_prepare_ms"] = (time.perf_counter() - t0) * 1000.0

        # ---- encode text ----
        t0 = time.perf_counter()
        logits, embeddings = annotate_with_classifier(
            encoder,
            task.classifier,
            tokenizer,
            texts,
            device
        )
        task_metrics["encode_ms"] = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        pred_ids = logits.argmax(dim=-1).tolist()
        task_metrics["argmax_ms"] = (time.perf_counter() - t0) * 1000.0

        correctness_total_ms = 0.0
        cache_write_total_ms = 0.0
        for i, sample_text in enumerate(texts):
            # ---- correctness ----
            t0 = time.perf_counter()
            p_correct = task.correctness.predict_p_correct(
                encoding=embeddings[i],
                student_logits=logits[i],
                input_len=len(sample_text)
            )
            correctness_total_ms += (time.perf_counter() - t0) * 1000.0

            t0 = time.perf_counter()
            caches[task_id]['preds'][i] = task.id2label[pred_ids[i]]
            caches[task_id]['p_corrects'][i] = p_correct
            cache_write_total_ms += (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        caches[task_id]['task'] = task
        caches[task_id]['classes'] = classes
        caches[task_id]['logits'] = logits
        caches[task_id]['embeddings'] = embeddings
        cache_write_total_ms += (time.perf_counter() - t0) * 1000.0

        task_metrics["correctness_predict_ms"] = correctness_total_ms
        task_metrics["cache_write_ms"] = cache_write_total_ms
        task_timing[task_id] = task_metrics

    fallback_flags = []
    fallback_decision_ms = 0.0
    non_fallback_copy_ms = 0.0
    teacher_call_ms = 0.0
    teacher_validate_ms = 0.0
    llm_kwargs = llm_kwargs or {}

    def _acc_task_metric(_task_id, _metric, _delta):
        task_timing[_task_id][_metric] = task_timing[_task_id].get(_metric, 0.0) + _delta

    scheduled_classifier_tasks = set()
    scheduled_correctness_tasks = set()

    fallback_indices = []
    for i, sample_text in enumerate(texts):
        t0 = time.perf_counter()
        p_correct_map = {_task_id: content['p_corrects'][i] for _task_id, content in caches.items()}
        print(f"p_correct_map:{p_correct_map}")
        sample_fallback = _should_fallback(p_correct_map, p_threshold)
        fallback_flags.append(sample_fallback)
        fallback_decision_ms += (time.perf_counter() - t0) * 1000.0

        if not sample_fallback:
            t0 = time.perf_counter()
            for task_id in task_id2classes.keys():
                all_results[i][task_id] = caches[task_id]['preds'][i]
            non_fallback_copy_ms += (time.perf_counter() - t0) * 1000.0
            continue
        fallback_indices.append(i)

    fallback_teacher_outs = []
    if fallback_indices:
        fallback_texts = [texts[i] for i in fallback_indices]

        t0 = time.perf_counter()
        teacher_out_batch = llm_fn(fallback_texts, task_id2classes, **llm_kwargs)
        teacher_call_ms += (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        if isinstance(teacher_out_batch, list):
            fallback_teacher_outs = teacher_out_batch
        elif isinstance(teacher_out_batch, dict):
            # Support two return formats:
            # 1) When len(fallback_indices) == 1, legacy API returns a single dict[task_id -> label]
            # 2) New batch API returns dict[task_id -> list[label]]
            if len(fallback_indices) == 1 and all(not isinstance(v, (list, tuple)) for v in teacher_out_batch.values()):
                fallback_teacher_outs = [teacher_out_batch]
            else:
                fallback_teacher_outs = []
                for pos in range(len(fallback_indices)):
                    item = {task_id: teacher_out_batch[task_id][pos] for task_id in task_id2classes.keys()}
                    fallback_teacher_outs.append(item)
        else:
            raise TypeError(
                "llm_fn batch output must be list[dict] or dict[task_id -> list[label]] "
                "(or dict[task_id -> label] when fallback batch size is 1)."
            )

        if len(fallback_teacher_outs) != len(fallback_indices):
            raise ValueError(
                f"llm_fn batch output size mismatch: expected {len(fallback_indices)}, got {len(fallback_teacher_outs)}"
            )
        for out in fallback_teacher_outs:
            for task_id in task_id2classes.keys():
                if task_id not in out:
                    raise ValueError(f"llm_fn output missing task_id: {task_id}")
                if out[task_id] is None:
                    raise ValueError(f"the Task {task_id} has returned an empty label")
        teacher_validate_ms += (time.perf_counter() - t0) * 1000.0

    for local_idx, i in enumerate(fallback_indices):
        sample_text = texts[i]
        teacher_out = fallback_teacher_outs[local_idx]
        for task_id, classes in task_id2classes.items():
            teacher_task_t0 = time.perf_counter()
            task = caches[task_id]['task']
            pred = task.label2id[caches[task_id]['preds'][i]]
            p_correct = caches[task_id]['p_corrects'][i]
            classes = caches[task_id]['classes']
            logits = caches[task_id]['logits']
            embeddings = caches[task_id]['embeddings']
            label_str = teacher_out[task_id]
            label_id = task.label2id[label_str]

            t0 = time.perf_counter()
            task.num_labeled += 1
            _acc_task_metric(task_id, "label_count_inc_ms", (time.perf_counter() - t0) * 1000.0)

            # ---- update correctness ----
            t0 = time.perf_counter()
            task.correctness.add_training_example(
                encoding=embeddings[i],
                student_logits=logits[i],
                input_len=len(sample_text),
                teacher_label=label_id,
                student_label=pred
            )
            _acc_task_metric(task_id, "correctness_add_example_ms", (time.perf_counter() - t0) * 1000.0)

            # ---- store in replay buffer ----
            t0 = time.perf_counter()
            task.buffers.add_sample(
                workflow_id=task.workflow_id(),
                text=sample_text,
                encoding=embeddings[i],
                label=label_id,
                student_pred=pred,
                confidence=p_correct
            )
            all_results[i][task_id] = label_str
            _acc_task_metric(task_id, "replay_add_ms", (time.perf_counter() - t0) * 1000.0)

            # ---- incremental training ----
            t0 = time.perf_counter()
            # Use label-based trigger only; step modulo can miss when batch sizes vary.
            if _should_train(task):
            # if True:
                buf = task.buffers.get_buffer(task.workflow_id())
                if buf.size > 0 and task.task_id not in scheduled_classifier_tasks:
                    steps = _compute_steps(task.num_labeled)
                    clipped_steps = max(5, min(50, steps))
                    train_round = task.num_train_rounds + 1
                    schedule_msg = (
                        f"[monitor] schedule classifier training "
                        f"task={task.task_id} round={train_round} step={task.step} "
                        f"num_labeled={task.num_labeled} "
                        f"new_samples_since_last={task.step - task.last_train_step} "
                        f"buffer_size={buf.size} steps_per_round={clipped_steps}"
                    )
                    print(schedule_msg)
                    _log_monitor_timing(
                        "[monitor_train_schedule] "
                        f"task_id={task.task_id} round={train_round} step={task.step} "
                        f"num_labeled={task.num_labeled} "
                        f"new_samples_since_last={task.step - task.last_train_step} "
                        f"buffer_size={buf.size} steps_per_round={clipped_steps}"
                    )
                    data = task.buffers.sample_for_training(task.workflow_id(),REPLAY_SAMPLE_SIZE)
                    loader = DataLoader(
                        data,
                        batch_size=TRAIN_BATCH_SIZE,
                        shuffle=True,
                        pin_memory=(getattr(device, "type", "cpu") == "cuda"),
                    )
                    _run_in_background_with_lock(
                        _classifier_locks,
                        task.task_id,
                        "classifier",
                        train_one_round_buff,
                        task.classifier,
                        loader,
                        device,
                        task.optimizer,
                        scheduler,
                        clipped_steps,
                        train_tag=f"{task.task_id}:r{train_round}",
                        log_fn=_log_monitor_timing,
                    )
                    scheduled_classifier_tasks.add(task.task_id)
                    task.last_train_step = task.step
                    task.last_train_labeled = task.num_labeled
                    task.num_train_rounds += 1
            _acc_task_metric(task_id, "classifier_train_schedule_ms", (time.perf_counter() - t0) * 1000.0)

            # ---- train correctness predictor ----
            t0 = time.perf_counter()
            if _should_train_correctness(task) and task.task_id not in scheduled_correctness_tasks:
            # if task.task_id not in scheduled_correctness_tasks:
                correctness_round = task.num_correctness_train_rounds + 1
                _log_monitor_timing(
                    "[monitor_correctness_schedule] "
                    f"task_id={task.task_id} round={correctness_round} step={task.step} "
                    f"num_labeled={task.num_labeled} "
                    f"new_samples_since_last={task.step - task.last_correctness_train_step} "
                    f"batch={CORRECTNESS_BATCH} steps={CORRECTNESS_STEPS}"
                )
                _run_in_background_with_lock(
                    _correctness_locks,
                    task.task_id,
                    "correctness",
                    task.correctness.train_step,
                    batch=CORRECTNESS_BATCH,
                    steps=CORRECTNESS_STEPS
                )
                scheduled_correctness_tasks.add(task.task_id)
                task.last_correctness_train_step = task.step
                task.last_correctness_train_labeled = task.num_labeled
                task.num_correctness_train_rounds += 1
            _acc_task_metric(task_id, "correctness_train_schedule_ms", (time.perf_counter() - t0) * 1000.0)
            _acc_task_metric(task_id, "teacher_task_total_ms", (time.perf_counter() - teacher_task_t0) * 1000.0)

    total_ms = (time.perf_counter() - monitor_t0) * 1000.0
    fallback_count = sum(1 for x in fallback_flags if x)
    _log_monitor_timing(
        "[monitor_timing] "
        f"total_ms={total_ms:.3f} "
        f"setup_device_ms={setup_device_ms:.3f} "
        f"setup_encoder_ms={setup_encoder_ms:.3f} "
        f"setup_tokenizer_ms={setup_tokenizer_ms:.3f} "
        f"fallback_decision_ms={fallback_decision_ms:.3f} "
        f"teacher_call_ms={teacher_call_ms:.3f} "
        f"teacher_validate_ms={teacher_validate_ms:.3f} "
        f"non_fallback_copy_ms={non_fallback_copy_ms:.3f} "
        f"fallback_count={fallback_count} "
        f"batch_size={batch_size}"
    )
    for task_id, metrics in task_timing.items():
        _log_monitor_timing(
            "[monitor_timing_task] "
            f"task_id={task_id} "
            + " ".join(f"{k}={v:.3f}" for k, v in metrics.items())
        )

    if is_single_input:
        return all_results[0], fallback_flags[0]
    return all_results, fallback_flags
