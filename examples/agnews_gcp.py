"""
example_agnews_gcp.py

End-to-end example: LLMCompiler with GCP classifier on the AG-News dataset,
with full concept-label supervision at every DAG node.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
What are "concept labels"?
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
GCPClassifier uses a DAG of intermediate concept nodes before making the
final prediction.  Each node has its own linear head (concept_predictors),
so it can be trained with a dedicated label rather than sharing the final
task label.

For the AG-News 4-class topic task we design a 4-node linear chain:

    Node 0 – "domain"     : Is the text about  (A) Human Activity
                                            or  (B) Knowledge / Systems?
              Binary label derived from:  World→A, Sports→A, Business→A,
                                          Sci/Tech→B

    Node 1 – "sector"     : Is the (human-activity) text about
                            (A) Competitive events  (Sports / Business)
                            or (B) Governance / World affairs?
              For Sci/Tech texts use the closest proxy (Business-like=A).

    Node 2 – "specificity": Is the text about a (A) specific event/outcome
                            or (B) a broader trend / policy / development?

    Node 3 (sink) – "topic": Final 4-class topic label (World / Sports /
                              Business / Sci/Tech).

The teacher is asked once per sample and returns all four concept labels
simultaneously.  During training, trainer.py calls forward_with_concepts()
and computes a weighted cross-entropy loss at every node:

    L = L_final  +  w * (L_node0 + L_node1 + L_node2)

This multi-level supervision is the key advantage of GCP over plain MLP/CNN:
earlier nodes receive gradient signal that captures coarser reasoning steps,
making the student more sample-efficient.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
What this script demonstrates
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  1. load_agnews()            – load the real AG-News corpus
  2. build_concept_teacher()  – teacher that returns per-node concept labels
  3. inspect_concept_nodes()  – call forward_with_concepts() directly and
                                visualise per-node prediction confidence
  4. run_online()             – online distillation with concept supervision;
                                shows fallback-rate learning curve
  5. run_offline()            – offline budget labelling
  6. run_multitask()          – topic + sentiment with concept supervision
  7. compare_topologies()     – linear chain vs branching DAG vs deep chain

Run:
    python test/example_agnews_gcp.py
    python test/example_agnews_gcp.py --n_train 500 --n_test 200
"""

import argparse
import csv
import os
import random
import time
import urllib.request
from collections import defaultdict
from pathlib import Path

import torch

from llmcompiler.monitor import monitor, _registry
from llmcompiler.defaults import get_device, get_encoder, get_tokenizer

# ---------------------------------------------------------------------------
# AG-News constants
# ---------------------------------------------------------------------------

TOPIC_CLASSES     = ["World", "Sports", "Business", "Sci/Tech"]
SENTIMENT_CLASSES = ["positive", "negative"]

TOPIC_TASK     = {"topic": TOPIC_CLASSES}
SENTIMENT_TASK = {"sentiment": SENTIMENT_CLASSES}
MULTI_TASK     = {**TOPIC_TASK, **SENTIMENT_TASK}

# ── Concept definitions (one per DAG node in the 4-node linear chain) ──────
#
# Each concept is a coarse binary question whose answer can be derived
# deterministically from the ground-truth topic label.  In a real deployment
# these would be answered by the LLM teacher via chain-of-thought prompting.

CONCEPT_DESCRIPTIONS = {
    0: "domain     │ Human-Activity (World/Sports/Business) vs Knowledge/Systems (Sci/Tech)",
    1: "sector     │ Competitive (Sports/Business/Sci/Tech) vs Governance/World-affairs (World)",
    2: "specificity│ Specific event/outcome (Sports) vs Trend/Policy/Development (others)",
    3: "topic      │ Final 4-class label: World / Sports / Business / Sci/Tech",
}

# Concept-label mappings: topic → [label_at_node0, label_at_node1,
#                                   label_at_node2, label_at_node3]
# All labels are expressed in the TOPIC_CLASSES vocabulary so that a single
# num_labels=4 head can be reused at every node.
# We encode coarser concepts as a *subset* prediction that maps:
#   node0 domain:       {Sci/Tech} → "Sci/Tech",  others → "World"
#   node1 sector:       {World}    → "World",      others → "Business"
#   node2 specificity:  {Sports}   → "Sports",     others → "World"
#   node3 final topic:  exact label

_CONCEPT_LABEL_MAP = {
    #  topic        node0        node1        node2        node3
    "World":     ["World",    "World",    "World",    "World"],
    "Sports":    ["World",    "Business", "Sports",   "Sports"],
    "Business":  ["World",    "Business", "World",    "Business"],
    "Sci/Tech":  ["Sci/Tech", "Business", "World",    "Sci/Tech"],
}


def concept_labels_for(topic: str) -> list:
    """Return [label_node0, label_node1, label_node2, label_node3] for a topic."""
    return _CONCEPT_LABEL_MAP[topic]


# ---------------------------------------------------------------------------
# GCP DAG configurations
# ---------------------------------------------------------------------------

GCP_LINEAR_CHAIN = {                        # 0 → 1 → 2 → 3
    "edges":       [(0, 1), (1, 2), (2, 3)],
    "concept_dim": 256,
    "use_resnet":  True,
    "dropout":     0.1,
}

GCP_BRANCHING_DAG = {                       # 0 ──→ 2 ──→ 3
    "edges":       [(0, 2), (1, 2), (2, 3)],#  1 ──→ 2
    "concept_dim": 256,
    "use_resnet":  True,
    "dropout":     0.1,
}

GCP_DEEP_CHAIN = {                          # 0 → 1 → 2 → 3 → 4 → 5
    "edges":       [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)],
    "concept_dim": 192,
    "use_resnet":  True,
    "dropout":     0.05,
}

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

_CACHE_DIR = Path(os.environ.get("AGNEWS_CACHE_DIR",
                                  Path.home() / ".cache" / "agnews"))
_CSV_URLS = {
    "train": ("https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras"
              "/master/data/ag_news_csv/train.csv"),
    "test":  ("https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras"
              "/master/data/ag_news_csv/test.csv"),
}
_CSV_INT2STR = {1: "World", 2: "Sports", 3: "Business", 4: "Sci/Tech"}


def load_agnews(split: str = "train", max_samples: int = None,
                seed: int = 42) -> list:
    """Return ``[(text, topic_label), ...]`` from the AG-News corpus.

    Tries HuggingFace ``datasets`` first; falls back to downloading the
    original CSV with ``urllib`` (no extra dependency).
    """
    data = _try_hf(split) or _download_csv(split)
    if max_samples and max_samples < len(data):
        random.Random(seed).shuffle(data)
        data = data[:max_samples]
    return data


def _try_hf(split: str) -> list:
    try:
        from datasets import load_dataset
        hf_map = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}
        ds = load_dataset("ag_news", split=split)
        return [(r["text"], hf_map[r["label"]]) for r in ds]
    except Exception:
        return []


def _download_csv(split: str) -> list:
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = _CACHE_DIR / f"{split}.csv"
    if not cache_file.exists():
        url = _CSV_URLS[split]
        print(f"[agnews] Downloading {split} split …")
        urllib.request.urlretrieve(url, cache_file)
        print(f"[agnews] Saved → {cache_file}")
    data = []
    with cache_file.open(encoding="utf-8") as fh:
        for row in csv.reader(fh):
            if len(row) < 3:
                continue
            label_str = _CSV_INT2STR.get(int(row[0]))
            text = (row[1].strip() + " " + row[2].strip()).strip()
            if label_str and text:
                data.append((text, label_str))
    return data

# ---------------------------------------------------------------------------
# Teacher LLM (mock – backed by ground-truth labels + concept derivation)
# ---------------------------------------------------------------------------

def build_concept_teacher(train_data: list):
    """Build a batch-aware mock teacher that returns per-node concept labels.

    The teacher returns a dict of task_id → label for each text.  Here we
    add four synthetic concept tasks (node0–node3) on top of the main topic
    task so that every DAG node receives its own supervision signal.

    In a real deployment replace the lookup logic below with an LLM call
    that answers a chain-of-thought prompt for each concept question.

    Returned format (list of dicts, one per input text):
        [
          {
            "topic":   "Sports",
            "node0":   "World",      # domain concept label
            "node1":   "Business",   # sector concept label
            "node2":   "Sports",     # specificity concept label
            # node3 == topic, no separate key needed
          },
          ...
        ]
    """
    lookup = {text: label for text, label in train_data}

    def _sentiment(text: str) -> str:
        positive_kw = {
            "win", "victory", "record", "growth", "profit", "launch",
            "breakthrough", "rise", "advance", "gain", "success",
            "award", "achieve", "improve", "surge",
        }
        return "positive" if set(text.lower().split()) & positive_kw else "negative"

    def teacher(texts, task_id2classes, **kwargs):
        """Batch teacher with the same signature as llm_fn.

        Returns per-sample dicts that include:
          - one key per task_id  (required final labels)
          - ``"{task_id}__node_{i}"`` keys for GCP concept supervision
            (same vocabulary as the task, one per DAG node in topo order)

        monitor() reads the ``__node_`` keys automatically when
        classifier_type="gcp"; they are ignored for all other architectures.
        """
        results = []
        for text in texts:
            topic = lookup.get(text, TOPIC_CLASSES[0])
            cls   = concept_labels_for(topic)   # list[str], len = num_nodes (4)
            row   = {}
            for task_id in task_id2classes:
                if task_id == "topic":
                    row["topic"] = topic
                    # Concept labels for each DAG node – same interface as llm_fn
                    for node_idx, node_label in enumerate(cls):
                        row[f"topic__node_{node_idx}"] = node_label
                elif task_id == "sentiment":
                    row["sentiment"] = _sentiment(text)
                else:
                    row[task_id] = task_id2classes[task_id][0]
            results.append(row)
        return results

    return teacher

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def reset():
    _registry.tasks.clear()


def accuracy(predictions: list, ground_truth: list, task_id: str) -> float:
    correct = sum(1 for p, g in zip(predictions, ground_truth)
                  if p.get(task_id) == g)
    return correct / len(predictions) if predictions else 0.0


def eval_on_testset(test_data: list, task_id: str = "topic",
                    batch_size: int = 64) -> float:
    """Evaluate the current student classifier on the test set.

    Calls the GCP head directly (no monitor(), no teacher, no step-counter
    side-effects) so evaluation is completely isolated from training.

    Args:
        test_data:  List of (text, label_str) tuples.
        task_id:    Which registered task to evaluate.
        batch_size: Tokenisation batch size.

    Returns:
        Accuracy in [0, 1], or 0.0 if the task is not yet registered.
    """
    if task_id not in _registry.tasks:
        return 0.0

    task      = _registry.tasks[task_id]
    device    = task.device
    tokenizer = task.tokenizer
    encoder   = task.encoder
    classifier = task.classifier

    classifier.eval()
    encoder.eval()

    correct = 0
    total   = 0

    texts  = [t for t, _ in test_data]
    labels = [l for _, l in test_data]

    for start in range(0, len(texts), batch_size):
        batch_texts  = texts[start : start + batch_size]
        batch_labels = labels[start : start + batch_size]

        enc = tokenizer(
            batch_texts,
            truncation=True, max_length=128,
            padding=True, return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}

        with torch.no_grad():
            embeddings = encoder(
                input_ids      = enc["input_ids"],
                attention_mask = enc["attention_mask"],
            ).last_hidden_state[:, 0]
            logits = classifier(embeddings)

        pred_ids = logits.argmax(dim=-1).tolist()
        for pred_id, true_lbl in zip(pred_ids, batch_labels):
            if task.id2label[pred_id] == true_lbl:
                correct += 1
            total += 1

    return correct / total if total else 0.0


def print_header(title: str):
    bar = "─" * 64
    print(f"\n┌{bar}┐")
    print(f"│  {title:<62}│")
    print(f"└{bar}┘")


def print_stats(label: str, total: int, fallbacks: int,
                elapsed_s: float, acc: float = None):
    rate = fallbacks / total * 100 if total else 0.0
    msg  = (f"  {label:<34} samples={total:>5}  fallbacks={fallbacks:>5}"
            f"  LLM-rate={rate:5.1f}%  time={elapsed_s:.1f}s")
    if acc is not None:
        msg += f"  acc={acc*100:.1f}%"
    print(msg)


# ---------------------------------------------------------------------------
# Demo 0: Inspect concept-node predictions via forward_with_concepts()
# ---------------------------------------------------------------------------

def inspect_concept_nodes(train_data: list, n_samples: int = 8):
    """Directly call GCPClassifier.forward_with_concepts() on a few samples
    and print the per-node top-1 prediction and softmax confidence.

    This demo does NOT use monitor() – it accesses the classifier directly
    to illustrate what each concept node has learned to predict.
    """
    print_header("Demo 0 – Concept-Node Inspection (forward_with_concepts)")

    # ── Build a fresh GCP classifier ─────────────────────────────────────
    from llmcompiler.models import GCPClassifier
    device    = get_device()
    encoder, hidden_size = get_encoder(device=device)
    tokenizer = get_tokenizer()

    num_labels = len(TOPIC_CLASSES)
    gcp = GCPClassifier(
        hidden_size = hidden_size,
        num_labels  = num_labels,
        **GCP_LINEAR_CHAIN,
    ).to(device)
    gcp.eval()

    # Brief fine-tuning on a small labelled set so concept nodes are
    # more than random – use the monitor() API for convenience then
    # extract the trained classifier from the registry.
    teacher = build_concept_teacher(train_data)
    reset()
    # The teacher already returns "topic__node_i" keys; monitor() picks them
    # up automatically – no separate concept_label_fn needed.
    for text, _ in train_data[:200]:
        monitor(
            TOPIC_TASK, text, teacher,
            mode="online", p_threshold=0.9,
            classifier_type="gcp",
            classifier_kwargs=GCP_LINEAR_CHAIN,
        )
    trained_gcp = _registry.tasks["topic"].classifier
    trained_gcp.eval()

    # ── Encode a handful of samples and call forward_with_concepts() ─────
    samples = train_data[:n_samples]
    texts   = [t for t, _ in samples]
    true_labels = [l for _, l in samples]

    enc = tokenizer(
        texts,
        truncation=True, max_length=128,
        padding=True, return_tensors="pt",
    )
    enc = {k: v.to(device) for k, v in enc.items()}

    with torch.no_grad():
        embeddings = encoder(
            input_ids      = enc["input_ids"],
            attention_mask = enc["attention_mask"],
        ).last_hidden_state[:, 0]           # [B, hidden_size]

        final_logits, concept_logits = trained_gcp.forward_with_concepts(embeddings)

    # concept_logits: list of [B, num_labels], one per node in topo order
    # For a 4-node chain the topo order is [0, 1, 2, 3].
    softmax = torch.nn.functional.softmax

    print(f"\n  Showing {n_samples} samples  "
          f"(4-node chain: node0=domain, node1=sector, "
          f"node2=specificity, node3=final topic)\n")

    col_w = 16
    header_parts = ["  True label   "] + [
        f"Node {i} ({['dom','sec','spc','top'][i]})"[:col_w].ljust(col_w)
        for i in range(len(concept_logits))
    ] + ["Final pred".ljust(col_w)]
    print("  " + " │ ".join(header_parts))
    print("  " + "─" * (len(header_parts) * (col_w + 3)))

    for b in range(len(samples)):
        text_short   = texts[b][:40].ljust(40)
        true_lbl     = true_labels[b].ljust(10)
        concept_cols = []
        for node_idx, clogits in enumerate(concept_logits):
            probs     = softmax(clogits[b], dim=-1)
            pred_id   = probs.argmax().item()
            pred_lbl  = TOPIC_CLASSES[pred_id]
            conf      = probs[pred_id].item()
            concept_cols.append(f"{pred_lbl:<10} {conf*100:4.0f}%")

        final_probs  = softmax(final_logits[b], dim=-1)
        final_pred   = TOPIC_CLASSES[final_probs.argmax().item()]
        final_conf   = final_probs.max().item()
        final_col    = f"{final_pred:<10} {final_conf*100:4.0f}%"

        row_parts = [true_lbl.ljust(col_w)] + [
            c[:col_w].ljust(col_w) for c in concept_cols
        ] + [final_col[:col_w].ljust(col_w)]
        print("  " + " │ ".join(row_parts))

    # ── Concept label agreement with CONCEPT_LABEL_MAP ───────────────────
    print("\n  Per-node prediction accuracy vs derived concept labels "
          "(un-trained weights → random; after 200 warmup samples):")
    for node_idx, clogits in enumerate(concept_logits):
        probs      = softmax(clogits, dim=-1)               # [B, C]
        pred_ids   = probs.argmax(dim=-1).tolist()
        expected   = [
            TOPIC_CLASSES.index(concept_labels_for(l)[node_idx])
            for l in true_labels
        ]
        acc = sum(p == e for p, e in zip(pred_ids, expected)) / len(samples)
        desc = CONCEPT_DESCRIPTIONS.get(node_idx, f"node {node_idx}")
        print(f"    Node {node_idx} ({desc.split('│')[0].strip():<14}) "
              f"acc = {acc*100:.0f}%")

    reset()


# ---------------------------------------------------------------------------
# Demo 1: Online distillation with concept supervision
# ---------------------------------------------------------------------------

def run_online(train_data: list, test_data: list,
               p_threshold: float = 0.8,
               n_train: int = 1000, n_test: int = 200):
    """Stream n_train samples through monitor() with concept supervision.

    The teacher returns concept labels for nodes 0–2 alongside the final
    topic label.  trainer.train_one_round_buff() automatically calls
    forward_with_concepts() for GCP, computing:

        L = L_topic  +  0.5 * (L_node0 + L_node1 + L_node2 + L_node3)

    After training, evaluate accuracy on the held-out test split.
    """
    print_header("Demo 1 – Online Distillation with Concept Supervision")
    reset()

    teacher   = build_concept_teacher(train_data)
    train_sub = train_data[:n_train]
    test_sub  = test_data[:n_test]

    print(f"\n  Concept label flow (teacher → student nodes):")
    for node_idx, desc in CONCEPT_DESCRIPTIONS.items():
        print(f"    {desc}")

    print(f"\n  [train] {len(train_sub)} samples  p_threshold={p_threshold}")

    # Track fallback rate in 10 windows to show the learning curve
    window     = max(1, len(train_sub) // 10)
    win_counts = defaultdict(int)
    win_fall   = defaultdict(int)
    fallbacks  = 0
    t0         = time.perf_counter()

    for i, (text, _) in enumerate(train_sub):
        _, fb = monitor(
            TOPIC_TASK, text, teacher,
            mode="online",
            p_threshold=p_threshold,
            classifier_type="gcp",
            classifier_kwargs=GCP_LINEAR_CHAIN,
        )
        if fb:
            fallbacks += 1
        w = i // window
        win_counts[w] += 1
        win_fall[w]   += int(fb)

    train_elapsed = time.perf_counter() - t0
    print_stats("training pass", len(train_sub), fallbacks, train_elapsed)

    print("\n  Fallback rate per decile (↓ = student more confident):")
    for w in sorted(win_counts):
        rate = win_fall[w] / win_counts[w] * 100
        bar  = "█" * int(rate / 5)
        print(f"    decile {w+1:>2}  {rate:5.1f}%  {bar}")

    # ── Evaluate: student only (p_threshold=0 → never call teacher) ──────
    print(f"\n  [eval]  {len(test_sub)} test samples  (student only)")
    preds  = []
    labels = [l for _, l in test_sub]
    t0     = time.perf_counter()

    for text, _ in test_sub:
        result, _ = monitor(
            TOPIC_TASK, text, teacher,
            mode="online", p_threshold=0.0,
            classifier_type="gcp",
            classifier_kwargs=GCP_LINEAR_CHAIN,
        )
        preds.append(result)

    acc = accuracy(preds, labels, "topic")
    print_stats("evaluation", len(test_sub), 0, time.perf_counter() - t0, acc)

    # Per-class breakdown
    class_ok  = defaultdict(int)
    class_tot = defaultdict(int)
    for pred, gt in zip(preds, labels):
        class_tot[gt] += 1
        class_ok[gt]  += int(pred.get("topic") == gt)
    print("\n  Per-class accuracy:")
    for cls in TOPIC_CLASSES:
        n = class_tot[cls]
        c = class_ok[cls]
        bar = "█" * (c * 20 // max(n, 1))
        print(f"    {cls:<12}  {c:>4}/{n:<4}  {c/n*100:5.1f}%  {bar}" if n
              else f"    {cls:<12}  –")

    return acc


# ---------------------------------------------------------------------------
# Demo 2: Offline mode – iterative active-learning rounds
#
# Each round:
#   1. Pass the *remaining* unlabelled pool to monitor(mode="offline").
#   2. monitor() selects `budget_per_round` samples (random or uncertainty).
#   3. The teacher labels those samples; the student learns from them.
#   4. The selected texts are removed from the pool.
#   5. Evaluate student accuracy on a held-out test set.
#
# This mirrors a realistic annotation pipeline where you query the oracle
# in fixed-size batches and retrain between rounds.
# ---------------------------------------------------------------------------

def run_offline(train_data: list, test_data: list,
                budget_per_round: int = 100, n_rounds: int = 5,
                n_test: int = 200, seed: int = 42):
    """Iterative offline labelling with per-round test-set evaluation.

    Each round:
      1. Pass the *remaining* unlabelled pool to monitor(mode="offline").
      2. monitor() selects ``budget_per_round`` samples via random selection.
      3. The teacher labels those samples (with GCP concept labels); the
         student head is trained in the background.
      4. Selected texts are removed from the pool.
      5. **Evaluate the student directly on the full held-out test set**
         by calling the classifier head without going through monitor(),
         so evaluation has zero side-effects on training state.

    Args:
        train_data:       Unlabelled pool – list of (text, topic_label) tuples.
        test_data:        Held-out evaluation corpus (never used for training).
        budget_per_round: Samples labelled per active-learning round.
        n_rounds:         Number of rounds to run.
        n_test:           How many test samples to evaluate on each round.
        seed:             Base random seed (incremented each round).
    """
    print_header(
        f"Demo 2 – Offline Iterative  "
        f"(budget/round={budget_per_round}, rounds={n_rounds}, "
        f"test_set={n_test})"
    )
    reset()

    # Build teacher from train + test so concept labels are correct for
    # any text the teacher might be asked about during offline rounds.
    teacher  = build_concept_teacher(train_data + test_data)
    pool     = [t for t, _ in train_data]   # mutable unlabelled pool
    test_sub = test_data[:n_test]           # fixed held-out evaluation set

    total_labelled   = 0
    all_label_counts: dict = defaultdict(int)
    t_start = time.perf_counter()

    print(f"\n  Train pool: {len(pool):,}  |  Test set: {len(test_sub):,}\n")
    print(f"  {'Round':<6} {'Labelled':>9} {'Pool left':>10} "
          f"{'Round(s)':>9} {'Test acc':>10}")
    print("  " + "─" * 50)

    for rnd in range(1, n_rounds + 1):
        if not pool:
            print("  Pool exhausted – stopping early.")
            break

        t0 = time.perf_counter()

        # ── Offline selection + labelling ─────────────────────────────────
        results, flags = monitor(
            TOPIC_TASK, pool, teacher,
            mode="offline",
            offline_select_method="random",
            offline_select_budget=budget_per_round,
            offline_select_seed=seed + rnd,
            classifier_type="gcp",
            classifier_kwargs=GCP_LINEAR_CHAIN,
        )
        round_elapsed   = time.perf_counter() - t0
        total_labelled += len(results)
        for r in results:
            all_label_counts[r.get("topic", "?")] += 1

        # Replicate the same shuffle used by the random selector to know
        # exactly which texts were selected, then remove them from the pool.
        rng     = random.Random(seed + rnd)
        indices = list(range(len(pool)))
        rng.shuffle(indices)
        selected = {pool[i] for i in indices[:budget_per_round]}
        pool     = [t for t in pool if t not in selected]

        # Wait briefly so background training triggered by the offline round
        # has a chance to complete before we evaluate.
        time.sleep(0.5)

        # ── Evaluate on test set (direct classifier call, no monitor()) ───
        # eval_on_testset() bypasses monitor() entirely: no step-counter
        # increment, no training trigger, no teacher call.
        acc = eval_on_testset(test_sub, task_id="topic")

        print(f"  {rnd:<6} {total_labelled:>9} {len(pool):>10} "
              f"{round_elapsed:>9.1f} {acc*100:>9.1f}%")

    total_elapsed = time.perf_counter() - t_start
    print(f"\n  Total wall time : {total_elapsed:.1f}s")
    print(f"  Total labelled  : {total_labelled}")

    # ── Final per-class label distribution ───────────────────────────────
    max_n = max(all_label_counts.values(), default=1)
    print("\n  Label distribution across all labelled training samples:")
    for cls in TOPIC_CLASSES:
        n   = all_label_counts.get(cls, 0)
        bar = "█" * (n * 30 // max_n)
        print(f"    {cls:<12}  {n:>5}  {bar}")

    # ── Final per-class test accuracy ─────────────────────────────────────
    if "topic" in _registry.tasks:
        task      = _registry.tasks["topic"]
        class_ok  = defaultdict(int)
        class_tot = defaultdict(int)
        for text, true_lbl in test_sub:
            enc = task.tokenizer(
                [text], truncation=True, max_length=128,
                padding=True, return_tensors="pt",
            )
            enc = {k: v.to(task.device) for k, v in enc.items()}
            with torch.no_grad():
                emb    = task.encoder(**enc).last_hidden_state[:, 0]
                logits = task.classifier(emb)
            pred = task.id2label[logits.argmax(dim=-1).item()]
            class_tot[true_lbl] += 1
            class_ok[true_lbl]  += int(pred == true_lbl)
        print("\n  Per-class accuracy on test set (final round):")
        for cls in TOPIC_CLASSES:
            n = class_tot[cls]
            c = class_ok[cls]
            bar = "█" * (c * 20 // max(n, 1))
            print(f"    {cls:<12}  {c:>4}/{n:<4}  "
                  f"{c/n*100:5.1f}%  {bar}" if n else
                  f"    {cls:<12}  –")


# ---------------------------------------------------------------------------
# Demo 3: Multi-task (topic + sentiment) with concept supervision
# ---------------------------------------------------------------------------

def run_multitask(train_data: list, n_samples: int = 300):
    """Classify each sample for topic (with concept labels) AND sentiment."""
    print_header("Demo 3 – Multi-task: topic (GCP + concepts) + sentiment")
    reset()

    teacher   = build_concept_teacher(train_data)
    data      = train_data[:n_samples]
    fallbacks = 0
    topic_ok  = 0
    t0        = time.perf_counter()

    for text, true_topic in data:
        result, fb = monitor(
            MULTI_TASK, text, teacher,
            mode="online", p_threshold=0.8,
            classifier_type="gcp",
            classifier_kwargs=GCP_LINEAR_CHAIN,
        )
        if fb:
            fallbacks += 1
        if result.get("topic") == true_topic:
            topic_ok += 1

    acc = topic_ok / len(data)
    print_stats("multi-task online", len(data), fallbacks,
                time.perf_counter() - t0, acc)
    print("\n  Note: both 'topic' (GCP with concept supervision) and "
          "'sentiment' classifiers are trained simultaneously.")


# ---------------------------------------------------------------------------
# Demo 4: Compare GCP topologies
# ---------------------------------------------------------------------------

def compare_topologies(train_data: list, n_samples: int = 200):
    """Compare fallback rates and timing across three DAG topologies."""
    print_header("Demo 4 – GCP Topology Comparison")
    teacher = build_concept_teacher(train_data)
    data    = train_data[:n_samples]

    configs = {
        "linear chain  (0→1→2→3)":     GCP_LINEAR_CHAIN,
        "branching DAG (0,1→2→3)":     GCP_BRANCHING_DAG,
        "deep chain    (0→1→…→5)":     GCP_DEEP_CHAIN,
    }

    print(f"\n  n_samples={n_samples}  p_threshold=0.8\n")
    for name, kwargs in configs.items():
        reset()
        fb_count = 0
        t0 = time.perf_counter()
        for text, _ in data:
            _, fb = monitor(
                TOPIC_TASK, text, teacher,
                mode="online", p_threshold=0.8,
                classifier_type="gcp",
                classifier_kwargs=kwargs,
            )
            if fb:
                fb_count += 1
        print_stats(name, len(data), fb_count, time.perf_counter() - t0)


# ---------------------------------------------------------------------------
# Demo 5: Concept-label ablation
#         Compare GCP (with concept supervision) vs plain MLP
# ---------------------------------------------------------------------------

def ablation_concept_vs_mlp(train_data: list, test_data: list,
                             n_train: int = 500, n_test: int = 200):
    """Train both a GCP head (concept supervision) and a plain MLP head on the
    same stream and compare their final accuracy and fallback rates.
    """
    print_header("Demo 5 – Ablation: GCP with Concept Labels vs plain MLP")

    teacher   = build_concept_teacher(train_data)
    train_sub = train_data[:n_train]
    test_sub  = test_data[:n_test]

    models = {
        "GCP (concept supervision)": ("gcp", GCP_LINEAR_CHAIN),
        "MLP (no concept labels)":   ("mlp", None),
    }

    print(f"\n  Training samples: {n_train}   Test samples: {n_test}"
          f"   p_threshold=0.8\n")

    results_table = {}
    for label, (ctype, ckwargs) in models.items():
        reset()
        fb   = 0
        t0   = time.perf_counter()
        for text, _ in train_sub:
            _, f = monitor(
                TOPIC_TASK, text, teacher,
                mode="online", p_threshold=0.8,
                classifier_type=ctype,
                classifier_kwargs=ckwargs,
            )
            if f:
                fb += 1
        train_t = time.perf_counter() - t0

        # Evaluate student (p_threshold=0 → never call teacher)
        preds  = []
        for text, _ in test_sub:
            result, _ = monitor(
                TOPIC_TASK, text, teacher,
                mode="online", p_threshold=0.0,
                classifier_type=ctype,
                classifier_kwargs=ckwargs,
            )
            preds.append(result)
        gt  = [l for _, l in test_sub]
        acc = accuracy(preds, gt, "topic")
        results_table[label] = (fb, train_t, acc)
        print_stats(label, len(train_sub), fb, train_t, acc)

    # Summary
    gcp_acc = results_table["GCP (concept supervision)"][2]
    mlp_acc = results_table["MLP (no concept labels)"][2]
    delta   = (gcp_acc - mlp_acc) * 100
    sign    = "+" if delta >= 0 else ""
    print(f"\n  Accuracy delta  GCP – MLP = {sign}{delta:.1f}pp")
    print("  (positive = concept supervision helps; "
          "result varies with sample size and randomness)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="LLMCompiler + GCP with concept labels on AG-News"
    )
    parser.add_argument("--n_train",   type=int,   default=1000)
    parser.add_argument("--n_test",    type=int,   default=200)
    parser.add_argument("--budget",    type=int,   default=100,
                        help="Samples labelled per offline round (default 100)")
    parser.add_argument("--n_rounds",  type=int,   default=5,
                        help="Number of offline active-learning rounds (default 5)")
    parser.add_argument("--threshold", type=float, default=0.8)
    parser.add_argument("--seed",      type=int,   default=42)
    parser.add_argument("--skip_inspect", action="store_true",
                        help="Skip Demo 0 (encoder forward pass) to save time")
    args = parser.parse_args()

    print("Loading AG-News …")
    t0         = time.perf_counter()
    train_data = load_agnews("train", seed=args.seed)
    test_data  = load_agnews("test",  seed=args.seed)
    print(f"  train={len(train_data):,}  test={len(test_data):,}  "
          f"({time.perf_counter()-t0:.1f}s)\n")

    print("Concept label definitions used throughout this example:")
    for node_idx, desc in CONCEPT_DESCRIPTIONS.items():
        print(f"  Node {node_idx}: {desc}")

    # # Demo 0 – inspect forward_with_concepts() directly
    # if not args.skip_inspect:
    #     inspect_concept_nodes(train_data, n_samples=8)

    # Demo 1 – online distillation with concept supervision
    run_online(
        train_data, test_data,
        p_threshold = args.threshold,
        n_train     = args.n_train,
        n_test      = args.n_test,
    )

    # Demo 2 – iterative offline labelling
    run_offline(
        train_data, test_data,
        budget_per_round = args.budget,
        n_rounds         = args.n_rounds,
        n_test           = args.n_test,
        seed             = args.seed,
    )

if __name__ == "__main__":
    main()
