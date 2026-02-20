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
import json
import math
import time
import threading
from pathlib import Path
from typing import Callable, Optional, Sequence
import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate

from .registry import TaskRegistry
from .models import annotate_with_classifier, CLASSIFIER_REGISTRY, GCPClassifier
from .trainer import train_one_round_buff, submodule_retrain
from .models import GCPClassifier
from .defaults import (
    get_device,
    get_encoder,
    get_tokenizer,
    get_optimizer,
)
from .selector import ActiveLearningSelector

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
# Sub-module retraining (§3.4 arXiv:2602.03006) — GCPClassifier only
SUBMODULE_RETRAIN_EVERY = 40   # trigger sub-module retrain every N samples (≥ TRAIN_CLASSIFIER_EVERY)
SUBMODULE_RETRAIN_STEPS = 30   # gradient steps per sub-module retrain round
SUBMODULE_RETRAIN_TOP_K = 1    # number of concept nodes to select (Theorem 3.3)

def _default_llm_fn(texts, task_id2classes, **kwargs):
    """Default teacher that calls the OpenAI Chat Completions API.

    Used automatically when the caller omits ``llm_fn`` in :func:`monitor`.
    Reads ``OPENAI_API_KEY`` from the environment (standard OpenAI SDK
    convention).

    The model is prompted to return a JSON object with one key per task and
    the predicted label as its value.  Labels are validated against
    ``task_id2classes``; any unrecognised label falls back to the first class.

    **GCP concept-label support** – pass ``concept_info`` (via ``llm_kwargs``)
    to also generate per-node concept labels that ``classifier_type="gcp"``
    consumes.  Each node description is used as a sub-question in the prompt;
    the model returns extra ``"{task_id}__node_{i}"`` keys alongside the
    normal task keys (same label vocabulary as the parent task).

    Args:
        texts (list[str]): Batch of input strings to classify.
        task_id2classes (dict[str, list[str]]): Mapping from task id to its
            allowed class labels.
        **kwargs: Most keys are forwarded directly to
            ``openai.OpenAI().chat.completions.create()``.  The following
            keys are consumed by this function and **not** forwarded:

            * ``model`` (str): OpenAI model name.  Defaults to
              ``"gpt-4o-mini"``.
            * ``concept_info`` (dict[str, list[str]] | None): Mapping from
              task id to a list of per-node concept descriptions.  Each
              description is a short string explaining what the node should
              predict (e.g. ``"Broad domain of the article"``).  The number
              of descriptions determines how many ``__node_`` keys are
              generated.  When ``None`` (default) no concept keys are
              produced.  Example::

                  concept_info = {
                      "topic": [
                          "Broad domain of the article",
                          "Primary sector the article covers",
                          "Most specific applicable category",
                      ]
                  }

    Returns:
        list[dict[str, str]]: One dict per input text.  Each dict contains:

        * One ``task_id → label`` entry per task in ``task_id2classes``.
        * When ``concept_info`` is supplied, additional
          ``"{task_id}__node_{i}" → label`` entries for every node
          description provided (same label vocabulary as the parent task).

    Raises:
        ImportError: If the ``openai`` package is not installed.
    """
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise ImportError(
            "The default teacher requires the 'openai' package. "
            "Install it with:  pip install openai"
        ) from exc

    model = kwargs.pop("model", "gpt-4o-mini")
    # concept_info: dict[task_id -> list[node_description_str]] | None
    concept_info: Optional[dict] = kwargs.pop("concept_info", None)
    client = OpenAI()

    # ------------------------------------------------------------------ #
    # Build system prompt                                                  #
    # ------------------------------------------------------------------ #
    task_lines = "\n".join(
        f'  - "{task_id}": one of [{", ".join(repr(c) for c in classes)}]'
        for task_id, classes in task_id2classes.items()
    )

    concept_lines = ""
    if concept_info:
        node_entries = []
        for task_id, node_descs in concept_info.items():
            if task_id not in task_id2classes:
                continue
            classes = task_id2classes[task_id]
            label_opts = ", ".join(repr(c) for c in classes)
            for node_idx, desc in enumerate(node_descs):
                key = f"{task_id}__node_{node_idx}"
                node_entries.append(
                    f'  - "{key}": {desc}  → one of [{label_opts}]'
                )
        if node_entries:
            concept_lines = (
                "\nAdditionally, predict the following concept keys "
                "(same label vocabulary as the parent task):\n"
                + "\n".join(node_entries)
            )

    system_prompt = (
        "You are a text classification assistant.\n"
        "For the given input text, return a JSON object with exactly one key "
        "per item listed below and the predicted label (exactly as written) "
        "as the value.\n"
        f"Tasks and allowed labels:\n{task_lines}"
        f"{concept_lines}\n"
        "Respond with ONLY a valid JSON object – no markdown, no explanation."
    )

    # ------------------------------------------------------------------ #
    # Call the API once per text and parse results                         #
    # ------------------------------------------------------------------ #
    results = []
    for text in texts:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text},
            ],
            response_format={"type": "json_object"},
            **kwargs,
        )
        raw = response.choices[0].message.content or "{}"
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            parsed = {}

        row = {}
        # Main task labels
        for task_id, classes in task_id2classes.items():
            label = parsed.get(task_id, classes[0])
            if label not in classes:
                label = classes[0]
            row[task_id] = label

        # Per-node concept labels (only when concept_info was supplied)
        if concept_info:
            for task_id, node_descs in concept_info.items():
                if task_id not in task_id2classes:
                    continue
                classes = task_id2classes[task_id]
                for node_idx in range(len(node_descs)):
                    key = f"{task_id}__node_{node_idx}"
                    label = parsed.get(key, row.get(task_id, classes[0]))
                    if label not in classes:
                        label = row.get(task_id, classes[0])
                    row[key] = label

        results.append(row)

    return results


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

def _should_submodule_retrain(task):
    """Return True when sub-module retraining should fire for a GCP task."""
    return (
        isinstance(task.classifier, GCPClassifier)
        and _crossed_sample_interval(task.step, task.last_submodule_retrain_step, SUBMODULE_RETRAIN_EVERY)
        and task.num_submodule_retrain_rounds < MAX_TRAIN_ROUNDS
        and task.num_labeled >= SUBMODULE_RETRAIN_TOP_K  # need at least one labeled sample
    )

def _compute_steps(num_labeled):
    # distilled from your code: steps ∝ log(L + 10)
    return int(80 * math.log(num_labeled + 10))

def _should_fallback(p_corrects,threshold):
    if min(p_corrects.values())<threshold:
        return True
    return False

def _collate_buffer(batch: list) -> dict:
    """Custom collate for replay-buffer dicts.

    Handles two special cases that the default PyTorch collate cannot:

    * ``text`` values are plain Python strings – returned as a list, not
      stacked into a tensor.
    * ``concept_labels`` may be ``None`` for entries that were stored before
      a ``__node_`` key was present in the teacher's output.  When *any* entry lacks a valid
      concept_labels tensor the key is dropped from the collated batch so
      that ``trainer.py`` can detect its absence and fall back gracefully.
    """
    # Decide whether to include concept_labels in this batch
    has_concept = all(
        item.get("concept_labels") is not None for item in batch
    )

    texts = [item["text"] for item in batch]
    # Build per-key lists, excluding text (handled above) and, conditionally,
    # concept_labels.
    numeric_batch = []
    for item in batch:
        row = {k: v for k, v in item.items()
               if k != "text"
               and (k != "concept_labels" or has_concept)
               and v is not None}
        numeric_batch.append(row)

    collated = default_collate(numeric_batch)
    collated["text"] = texts
    return collated


_registry = TaskRegistry()
_classifier_locks = {}
_correctness_locks = {}
_submodule_retrain_locks = {}
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
    llm_fn=None,
    llm_kwargs=None,
    *,
    mode,
    offline_select_method=None,
    offline_select_budget=None,
    offline_select_seed=42,
    offline_select_probs=None,
    offline_select_embeddings=None,
    offline_select_mc_probs=None,
    offline_select_already_selected=None,
    offline_select_pool_size=None,
    encoder=None,
    tokenizer=None,
    device=None,
    hidden_size=None,
    optimizer=None,
    scheduler=None,
    p_threshold=0.8,
    classifier_type="mlp",
    classifier_kwargs=None,
    submodule_retrain_top_k=None,
):
    """
    Execute adaptive inference with automatic LLM distillation.

    Args:
        task_id2classes (dict[str, list[str]]):
            Mapping from task name to allowed class labels.
        text (str | list[str] | tuple[str, ...]):
            Input text(s) to be classified.
        llm_fn (callable, optional):
            Teacher function. Must accept ``(texts, task_id2classes, **kwargs)``
            and return ``list[dict[task_id -> label]]`` or
            ``dict[task_id -> list[label]]``.  When omitted, defaults to
            :func:`_default_llm_fn` which always returns the first class label
            for every task (useful for smoke-tests; not suitable for production).
        llm_kwargs (dict, optional):
            Extra arguments forwarded to llm_fn.
        mode (str):
            **Required.** Inference mode. Must be one of ``"online"`` or ``"offline"``.
            - ``"online"``: student model is updated continuously during inference.
            - ``"offline"``: student model is frozen; a fixed subset of samples is
              selected up-front using ``offline_select_method`` and
              ``offline_select_budget``.
        offline_select_method (str, optional):
            Sample-selection strategy used in offline mode (e.g. ``"random"``,
            ``"uncertainty"``). Required when ``mode="offline"``.
        offline_select_budget (int, optional):
            Maximum number of samples to send to the teacher in offline mode.
            Required when ``mode="offline"``.
        offline_select_seed (int, optional):
            Random seed for offline sample selection. Defaults to ``42``.
        p_threshold (float):
            Minimum confidence to trust student prediction.
        classifier_type (str):
            Architecture of the student classification head.  Must be one of
            the keys registered in ``CLASSIFIER_REGISTRY``:

            * ``"mlp"``      – 2-layer MLP with dropout and GELU *(default)*.
            * ``"linear"``   – Single linear layer (fastest, no hidden layer).
            * ``"deep_mlp"`` – Configurable-depth MLP with residual connections.
            * ``"cnn"``      – Multi-scale 1-D CNN with global max-pooling.
            * ``"gnn"``      – GNN-inspired head using virtual-graph message passing.
            * ``"gcp"``      – Graph of Concept Predictors (DAG-structured head);
              requires ``classifier_kwargs={"edges": [(0,1),(1,2),...]}``.

            Defaults to ``"mlp"``.
        classifier_kwargs (dict, optional):
            Architecture-specific keyword arguments forwarded to the selected
            classifier class.  For example::
        submodule_retrain_top_k (int, optional):
            Number of concept nodes to select for targeted sub-module
            retraining (§3.4 of arXiv:2602.03006).  Only applicable when
            ``classifier_type="gcp"``.  When ``None`` (default), falls back
            to the module-level :data:`SUBMODULE_RETRAIN_TOP_K` constant.

                # Deeper MLP with less dropout
                classifier_type="deep_mlp",
                classifier_kwargs={"num_layers": 4, "dropout": 0.05}

                # CNN with larger filters
                classifier_type="cnn",
                classifier_kwargs={"num_filters": 256, "kernel_sizes": (3, 5, 7, 9)}

                # GNN with 8 virtual nodes and 3 message-passing rounds
                classifier_type="gnn",
                classifier_kwargs={"num_nodes": 8, "num_layers": 3}

    **GCP concept-label convention** (``classifier_type="gcp"`` only):

        ``llm_fn`` may return per-node concept labels alongside the final task
        label by including keys of the form ``f"{task_id}__node_{i}"`` in each
        output dict, where ``i`` is the 0-based node index in topological order.
        Values must be strings drawn from the same label vocabulary as the
        final task.  When a key is missing for a node, the final task label is
        used as a fallback for that node.

        Example – teacher for a 4-node AG-News GCP chain::

            def teacher(texts, task_id2classes, **kwargs):
                results = []
                for text in texts:
                    topic = lookup[text]      # e.g. "Sports"
                    results.append({
                        "topic":          topic,
                        "topic__node_0":  domain_concept(topic),   # e.g. "World"
                        "topic__node_1":  sector_concept(topic),   # e.g. "Business"
                        "topic__node_2":  specificity_concept(topic),
                        # node_3 omitted → falls back to final "topic" label
                    })
                return results

        The ``__node_`` keys are stripped before validation so they do not
        interfere with ``task_id2classes`` key checks.

    Returns:
        results (dict[str, str] | list[dict[str, str]]):
            Predicted label per task. Returns a list when batch input is used.
        fallback (bool | list[bool]):
            Whether teacher LLM was invoked. Returns a list when batch input is used.

    Notes:
        - Teacher calls are minimized automatically.
        - No gradients flow through the teacher.
        - ``classifier_type`` and ``classifier_kwargs`` only take effect the
          *first time* a task is seen; subsequent calls reuse the already-created
          ``Task`` object from the registry.
    """
    if llm_fn is None:
        llm_fn = _default_llm_fn

    _VALID_CLASSIFIERS = sorted(CLASSIFIER_REGISTRY)
    if classifier_type.lower() not in CLASSIFIER_REGISTRY:
        raise ValueError(
            f"'classifier_type' must be one of {_VALID_CLASSIFIERS}, "
            f"got {classifier_type!r}."
        )
    _VALID_MODES = {"online", "offline"}
    if mode not in _VALID_MODES:
        raise ValueError(
            f"'mode' must be one of {sorted(_VALID_MODES)}, got {mode!r}."
        )
    # Bug-fix 1: capture is_single_input BEFORE offline may convert text to a
    # list, so the return type stays consistent regardless of mode.
    is_single_input = isinstance(text, str)

    if mode == "offline":
        if offline_select_method is None:
            raise ValueError(
                "'offline_select_method' is required when mode='offline'."
            )
        if offline_select_budget is None:
            raise ValueError(
                "'offline_select_budget' is required when mode='offline'."
            )

        # Bug-fix 3: auto-derive pool_size from the input when the caller did
        # not supply it explicitly.  This makes "random" work out of the box.
        _effective_pool_size = offline_select_pool_size
        if _effective_pool_size is None:
            if isinstance(text, str):
                _effective_pool_size = 1
            else:
                _effective_pool_size = len(text)

        selected_indices = ActiveLearningSelector.select(
            method=offline_select_method,
            budget=offline_select_budget,
            probs=offline_select_probs,
            embeddings=offline_select_embeddings,
            mc_probs=offline_select_mc_probs,
            already_selected=offline_select_already_selected,
            pool_size=_effective_pool_size,
            seed=offline_select_seed,
        )
        if isinstance(text, str):
            text = [text]
        else:
            text = [text[i] for i in selected_indices]

    monitor_t0 = time.perf_counter()
    task_timing = {}
    # Bug-fix 5: offline mode may have converted a single string to a list
    # already; guard against wrapping it a second time into a nested list.
    if is_single_input and isinstance(text, str):
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
            hidden_size=hidden_size,
            classifier_type=classifier_type,
            classifier_kwargs=classifier_kwargs,
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
    scheduled_submodule_retrain_tasks = set()
    # Accumulate per-task scoring-batch data from the current fallback batch
    # (encoding / label pairs for counterfactual impact scoring; GCP only).
    _smr_enc: dict = {}   # task_id -> list[Tensor]
    _smr_lbl: dict = {}   # task_id -> list[int]

    fallback_indices = []
    for i, sample_text in enumerate(texts):
        t0 = time.perf_counter()
        p_correct_map = {_task_id: content['p_corrects'][i] for _task_id, content in caches.items()}
        # print(f"p_correct_map:{p_correct_map}")
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
            # Extract per-node concept labels from teacher_out when the teacher
            # returned them under keys "{task_id}__node_{i}" (GCP convention).
            # Falls back to the final label for any node whose key is absent.
            concept_labels = None
            if hasattr(task.classifier, 'num_nodes'):   # GCPClassifier only
                node_label_ids = []
                for node_idx in range(task.classifier.num_nodes):
                    node_key = f"{task_id}__node_{node_idx}"
                    if node_key in teacher_out:
                        node_str = teacher_out[node_key]
                        node_label_ids.append(
                            task.label2id.get(node_str, label_id)
                        )
                    else:
                        node_label_ids.append(label_id)
                concept_labels = node_label_ids
            task.buffers.add_sample(
                workflow_id=task.workflow_id(),
                text=sample_text,
                encoding=embeddings[i],
                label=label_id,
                concept_labels=concept_labels,
                student_pred=pred,
                confidence=p_correct,
            )
            all_results[i][task_id] = label_str
            _acc_task_metric(task_id, "replay_add_ms", (time.perf_counter() - t0) * 1000.0)

            # ---- accumulate scoring-batch data for sub-module retraining (GCP only) ----
            if isinstance(task.classifier, GCPClassifier):
                _smr_enc.setdefault(task_id, []).append(embeddings[i].detach().cpu())
                _smr_lbl.setdefault(task_id, []).append(label_id)

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
                    # schedule_msg = (
                    #     f"[monitor] schedule classifier training "
                    #     f"task={task.task_id} round={train_round} step={task.step} "
                    #     f"num_labeled={task.num_labeled} "
                    #     f"new_samples_since_last={task.step - task.last_train_step} "
                    #     f"buffer_size={buf.size} steps_per_round={clipped_steps}"
                    # )
                    # print(schedule_msg)
                    # _log_monitor_timing(
                    #     "[monitor_train_schedule] "
                    #     f"task_id={task.task_id} round={train_round} step={task.step} "
                    #     f"num_labeled={task.num_labeled} "
                    #     f"new_samples_since_last={task.step - task.last_train_step} "
                    #     f"buffer_size={buf.size} steps_per_round={clipped_steps}"
                    # )
                    data = task.buffers.sample_for_training(task.workflow_id(), REPLAY_SAMPLE_SIZE, num_labels=task.num_labels)
                    loader = DataLoader(
                        data,
                        batch_size=TRAIN_BATCH_SIZE,
                        shuffle=True,
                        pin_memory=(getattr(device, "type", "cpu") == "cuda"),
                        collate_fn=_collate_buffer,
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
                # _log_monitor_timing(
                #     "[monitor_correctness_schedule] "
                #     f"task_id={task.task_id} round={correctness_round} step={task.step} "
                #     f"num_labeled={task.num_labeled} "
                #     f"new_samples_since_last={task.step - task.last_correctness_train_step} "
                #     f"batch={CORRECTNESS_BATCH} steps={CORRECTNESS_STEPS}"
                # )
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

            # ---- sub-module retraining (GCP only; §3.4 arXiv:2602.03006) ----
            t0 = time.perf_counter()
            if (
                _should_submodule_retrain(task)
                and task.task_id not in scheduled_submodule_retrain_tasks
                and task_id in _smr_enc
            ):
                smr_round = task.num_submodule_retrain_rounds + 1
                # Build scoring batch from fallback samples collected this call.
                scoring_batch = {
                    "encoding": torch.stack(_smr_enc[task_id]),   # [N, hidden_size]
                    "label":    torch.tensor(_smr_lbl[task_id], dtype=torch.long),
                }
                _smr_top_k = (
                    submodule_retrain_top_k
                    if submodule_retrain_top_k is not None
                    else SUBMODULE_RETRAIN_TOP_K
                )
                buf_smr = task.buffers.get_buffer(task.workflow_id())
                if buf_smr.size > 0:
                    data_smr = task.buffers.sample_for_training(
                        task.workflow_id(), REPLAY_SAMPLE_SIZE, num_labels=task.num_labels
                    )
                    loader_smr = DataLoader(
                        data_smr,
                        batch_size=TRAIN_BATCH_SIZE,
                        shuffle=True,
                        pin_memory=(getattr(device, "type", "cpu") == "cuda"),
                        collate_fn=_collate_buffer,
                    )
                    _run_in_background_with_lock(
                        _submodule_retrain_locks,
                        task.task_id,
                        "submodule_retrain",
                        submodule_retrain,
                        task.classifier,
                        loader_smr,
                        device,
                        task.optimizer,
                        scoring_batch,
                        _smr_top_k,
                        SUBMODULE_RETRAIN_STEPS,
                        1.0,   # max_grad_norm
                        0.5,   # concept_loss_weight
                        f"{task.task_id}:smr{smr_round}",
                        _log_monitor_timing,
                    )
                    scheduled_submodule_retrain_tasks.add(task.task_id)
                    task.last_submodule_retrain_step = task.step
                    task.num_submodule_retrain_rounds += 1
            _acc_task_metric(task_id, "submodule_retrain_schedule_ms", (time.perf_counter() - t0) * 1000.0)
            _acc_task_metric(task_id, "teacher_task_total_ms", (time.perf_counter() - teacher_task_t0) * 1000.0)

    total_ms = (time.perf_counter() - monitor_t0) * 1000.0
    fallback_count = sum(1 for x in fallback_flags if x)
    # _log_monitor_timing(
    #     "[monitor_timing] "
    #     f"total_ms={total_ms:.3f} "
    #     f"setup_device_ms={setup_device_ms:.3f} "
    #     f"setup_encoder_ms={setup_encoder_ms:.3f} "
    #     f"setup_tokenizer_ms={setup_tokenizer_ms:.3f} "
    #     f"fallback_decision_ms={fallback_decision_ms:.3f} "
    #     f"teacher_call_ms={teacher_call_ms:.3f} "
    #     f"teacher_validate_ms={teacher_validate_ms:.3f} "
    #     f"non_fallback_copy_ms={non_fallback_copy_ms:.3f} "
    #     f"fallback_count={fallback_count} "
    #     f"batch_size={batch_size}"
    # )
    # for task_id, metrics in task_timing.items():
        # _log_monitor_timing(
        #     "[monitor_timing_task] "
        #     f"task_id={task_id} "
        #     + " ".join(f"{k}={v:.3f}" for k, v in metrics.items())
        # )

    if is_single_input:
        return all_results[0], fallback_flags[0]
    return all_results, fallback_flags
