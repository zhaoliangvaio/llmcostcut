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
from typing import Dict, List
from .models import GCPClassifier


def train_one_round_buff(
    classifier,
    loader,
    device,
    optimizer,
    scheduler=None,
    steps_per_round=200,
    max_grad_norm=1.0,
    train_tag=None,
    log_fn=None,
    concept_loss_weight=0.5,
):
    """Online incremental training: run fixed number of SGD steps each round.

    For :class:`~llmcompiler.models.GCPClassifier`, concept-level supervision
    is automatically applied via :meth:`forward_with_concepts`.  Each concept
    node's predictor is trained with the same ground-truth label as the final
    task head, weighted by *concept_loss_weight*.

    Args:
        classifier: Student classification head (any architecture).
        loader: DataLoader over the replay buffer.
        device: Torch device to move batches onto.
        optimizer: Persistent optimizer (caller is responsible for creation).
        scheduler: Optional LR scheduler; ``step()`` is called every gradient
            update when provided.
        steps_per_round (int): Number of gradient updates to perform.
        max_grad_norm (float): Gradient clipping norm.
        train_tag (str, optional): Short label used in log messages.
        log_fn (callable, optional): Extra logging sink (receives the summary
            string produced at the end of the round).
        concept_loss_weight (float): Weight applied to each concept-node loss
            when training a :class:`~llmcompiler.models.GCPClassifier`.
            Ignored for all other architectures.  Defaults to ``0.5``.
    """
    classifier.train()
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    is_gcp = isinstance(classifier, GCPClassifier)

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
        labels = batch['label']

        if is_gcp:
            final_logits, concept_logits = classifier.forward_with_concepts(batch['encoding'])
            loss = criterion(final_logits, labels)
            # concept_labels shape: [B, num_nodes] when stored by monitor();
            # None (key absent) when no concept_label_fn was registered – fall
            # back to the final task label for every node (original behaviour).
            concept_label_batch = batch.get('concept_labels')  # [B, num_nodes] | None
            for node_idx, clogits in enumerate(concept_logits):
                if concept_label_batch is not None and node_idx < concept_label_batch.shape[1]:
                    node_labels = concept_label_batch[:, node_idx].long().to(clogits.device)
                else:
                    node_labels = labels
                loss = loss + concept_loss_weight * criterion(clogits, node_labels)
        else:
            logits = classifier(batch['encoding'])
            loss = criterion(logits, labels)

        running_loss += float(loss.item())

        loss.backward()
        nn.utils.clip_grad_norm_(classifier.parameters(), max_grad_norm)

        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        if scheduler is not None:
            scheduler.step()

        step += 1

    avg_loss = running_loss / max(1, step)
    prefix = f"[Train:{train_tag}]" if train_tag else "[Train]"
    msg = f"{prefix} steps={step} avg_loss={avg_loss:.6f}"
    print(msg)
    if callable(log_fn):
        log_fn(msg)


# ---------------------------------------------------------------------------
# Sub-module retraining  (§3.4 of arXiv:2602.03006)
# ---------------------------------------------------------------------------

def _iter_node_params(classifier: GCPClassifier, node_idx: int):
    """Yield all trainable parameters that belong to a single GCP concept node.

    Covers the node's input projection (root nodes) or transition function
    (non-root nodes), as well as its per-node concept predictor.
    """
    key = str(node_idx)
    if key in classifier.input_projections:
        yield from classifier.input_projections[key].parameters()
    if key in classifier.transition_fns:
        yield from classifier.transition_fns[key].parameters()
    yield from classifier.concept_predictors[node_idx].parameters()


@torch.no_grad()
def compute_node_counterfactual_scores(
    classifier: GCPClassifier,
    encoding: torch.Tensor,
    labels: torch.Tensor,
    criterion: nn.Module,
) -> List[float]:
    """Compute the counterfactual impact score Δ̂_i for every concept node.

    For each node *i*, its embedding is zeroed out and the final-task loss is
    re-evaluated on the provided mini-batch.  Downstream nodes that depend on
    node *i* are re-propagated using the ablated (zero) embedding, so the
    effect propagates through the entire subgraph rooted at *i*.

    The score is defined as::

        Δ̂_i = L_ablated_i − L_full

    A **large positive** Δ̂_i indicates that node *i* is a high-impact
    predictor (removing it substantially increases the final loss).  A near-zero
    or negative score means the node has little influence on the current batch.

    The model is temporarily switched to eval mode so that dropout is
    deterministic; the original training/eval state is restored on return.

    Args:
        classifier: A :class:`~llmcompiler.models.GCPClassifier` instance.
        encoding:   CLS embeddings ``[B, hidden_size]`` on the correct device.
        labels:     Ground-truth label indices ``[B]``.
        criterion:  Loss function used to score the final task logits.

    Returns:
        List of float Δ̂_i values, one per node in topological order.
    """
    was_training = classifier.training
    classifier.eval()

    device = encoding.device
    B = encoding.shape[0]
    # dropout is disabled in eval mode, so this is an identity op
    x = classifier.dropout_layer(encoding)

    # ── Full forward pass ────────────────────────────────────────────────
    node_emb_full: Dict[int, torch.Tensor] = {}
    for j in classifier._topo_order:
        if not classifier._parents[j]:
            node_emb_full[j] = classifier.input_projections[str(j)](x)
        else:
            parent_cat = torch.cat(
                [node_emb_full[p] for p in classifier._parents[j]], dim=-1
            )
            node_emb_full[j] = classifier.transition_fns[str(j)](parent_cat)

    sink_cat = torch.cat([node_emb_full[s] for s in classifier._sink_nodes], dim=-1)
    loss_full = criterion(classifier.final_classifier(sink_cat), labels).item()

    # ── One ablation pass per node ───────────────────────────────────────
    scores: List[float] = []
    for ablate_j in classifier._topo_order:
        # BFS: collect ablate_j and all of its descendants
        affected: set = {ablate_j}
        frontier = [ablate_j]
        while frontier:
            n = frontier.pop()
            for child in classifier._children[n]:
                if child not in affected:
                    affected.add(child)
                    frontier.append(child)

        # Re-propagate: unaffected nodes reuse full embeddings
        node_emb_abl: Dict[int, torch.Tensor] = {}
        for j in classifier._topo_order:
            if j == ablate_j:
                node_emb_abl[j] = torch.zeros(B, classifier.concept_dim, device=device)
            elif j not in affected:
                node_emb_abl[j] = node_emb_full[j]
            else:
                # Descendant: re-compute from (possibly ablated) parents
                parent_cat = torch.cat(
                    [node_emb_abl[p] for p in classifier._parents[j]], dim=-1
                )
                node_emb_abl[j] = classifier.transition_fns[str(j)](parent_cat)

        sink_cat_abl = torch.cat(
            [node_emb_abl[s] for s in classifier._sink_nodes], dim=-1
        )
        loss_abl = criterion(classifier.final_classifier(sink_cat_abl), labels).item()
        scores.append(loss_abl - loss_full)

    classifier.train(was_training)
    return scores


def submodule_retrain(
    classifier,
    loader,
    device,
    optimizer,
    scoring_batch: dict,
    top_k: int = 1,
    steps: int = 50,
    max_grad_norm: float = 1.0,
    concept_loss_weight: float = 0.5,
    train_tag=None,
    log_fn=None,
) -> List[int]:
    """Targeted sub-module retraining for :class:`~llmcompiler.models.GCPClassifier`.

    Implements Algorithm 1 from §3.4 of *Distilling LLM Reasoning into Graph
    of Concept Predictors* (Yu & Zhao, 2026, arXiv:2602.03006):

    1. **Score** — run counterfactual ablations on *scoring_batch* to obtain
       the impact score Δ̂_i for each concept node.
    2. **Select** — choose the top-K nodes with the largest Δ̂_i (most
       influential; see Theorem 3.3 in the paper).
    3. **Freeze** — set ``requires_grad=False`` for all parameters except
       those belonging to the selected nodes and the shared
       ``final_classifier``.
    4. **Retrain** — run *steps* gradient updates, computing the final task
       loss plus the concept-level losses for selected nodes only.
    5. **Restore** — re-enable gradients for all parameters.

    Args:
        classifier: :class:`~llmcompiler.models.GCPClassifier` instance.
        loader: DataLoader over the replay buffer (reused from full training).
        device: Torch device.
        optimizer: Persistent optimizer shared with regular training rounds.
        scoring_batch (dict): A collated batch with keys ``'encoding'`` and
            ``'label'`` (optionally ``'concept_labels'``) used to compute
            counterfactual scores.  Values should be CPU tensors; they are
            moved to *device* internally.
        top_k (int): Number of concept nodes to select for retraining.
            Defaults to ``1``.
        steps (int): Number of gradient update steps.  Defaults to ``50``.
        max_grad_norm (float): Gradient-clipping norm.  Defaults to ``1.0``.
        concept_loss_weight (float): Weight for per-node concept losses.
            Defaults to ``0.5``.
        train_tag (str, optional): Short label appended to log lines.
        log_fn (callable, optional): Extra logging sink.

    Returns:
        List[int]: Sorted list of selected node indices.

    Raises:
        TypeError: When *classifier* is not a
            :class:`~llmcompiler.models.GCPClassifier`.
    """
    if not isinstance(classifier, GCPClassifier):
        raise TypeError(
            "submodule_retrain is only applicable to GCPClassifier; "
            f"got {type(classifier).__name__}."
        )

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    prefix = f"[SubRetrain:{train_tag}]" if train_tag else "[SubRetrain]"

    # ── Step 1: Counterfactual impact scores ─────────────────────────────
    enc_sc = scoring_batch["encoding"].to(device)
    lbl_sc = scoring_batch["label"].to(device)
    scores = compute_node_counterfactual_scores(classifier, enc_sc, lbl_sc, criterion)

    # ── Step 2: Select top-K nodes (Theorem 3.3) ─────────────────────────
    topo_order = classifier._topo_order
    top_k_actual = min(top_k, len(topo_order))
    ranked = sorted(zip(topo_order, scores), key=lambda ns: ns[1], reverse=True)
    selected_nodes = {n for n, _ in ranked[:top_k_actual]}
    selected_list = sorted(selected_nodes)

    score_summary = {n: round(s, 4) for n, s in zip(topo_order, scores)}
    print(f"{prefix} scores={score_summary} selected={selected_list}")

    # ── Step 3: Freeze all; unfreeze selected nodes + final_classifier ───
    for p in classifier.parameters():
        p.requires_grad_(False)
    for node_idx in selected_nodes:
        for p in _iter_node_params(classifier, node_idx):
            p.requires_grad_(True)
    # The final_classifier aggregates sink-node embeddings; keep it active so
    # the task-loss gradient can flow back through the updated concept nodes.
    for p in classifier.final_classifier.parameters():
        p.requires_grad_(True)

    # ── Step 4: Targeted gradient updates ────────────────────────────────
    classifier.train()
    topo_pos = {j: i for i, j in enumerate(topo_order)}  # node → list position

    step = 0
    running_loss = 0.0
    loader_iter = iter(loader)

    while step < steps:
        try:
            batch = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            batch = next(loader_iter)

        batch = {k: v.to(device) for k, v in batch.items() if k != "text"}
        b_labels = batch["label"]
        concept_label_batch = batch.get("concept_labels")

        final_logits, concept_logits = classifier.forward_with_concepts(batch["encoding"])

        # Final task loss (gradients flow through all concept node embeddings)
        loss = criterion(final_logits, b_labels)

        # Concept-level loss only for the selected nodes
        for node_idx in selected_nodes:
            pos = topo_pos[node_idx]
            clogits = concept_logits[pos]
            if concept_label_batch is not None and pos < concept_label_batch.shape[1]:
                node_labels = concept_label_batch[:, pos].long().to(clogits.device)
            else:
                node_labels = b_labels
            loss = loss + concept_loss_weight * criterion(clogits, node_labels)

        running_loss += float(loss.item())
        loss.backward()

        active_params = [p for p in classifier.parameters() if p.requires_grad]
        if active_params:
            nn.utils.clip_grad_norm_(active_params, max_grad_norm)

        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        step += 1

    # ── Step 5: Restore all parameters to trainable ──────────────────────
    for p in classifier.parameters():
        p.requires_grad_(True)

    avg_loss = running_loss / max(1, step)
    msg = f"{prefix} steps={step} avg_loss={avg_loss:.6f} selected={selected_list}"
    print(msg)
    if callable(log_fn):
        log_fn(msg)

    return selected_list