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
from typing import Dict, List, Optional, Sequence, Tuple, Type


# ---------------------------------------------------------------------------
# Classification head architectures
# ---------------------------------------------------------------------------

class DeepMLPClassifier(nn.Module):
    """Configurable deep MLP classification head with residual connections.

    Args:
        hidden_size (int): Input / intermediate feature dimension.
        num_labels (int): Number of output classes.
        num_layers (int): Total number of hidden linear layers before the
            final projection.  Minimum ``1``.
        dropout (float): Dropout probability applied after every hidden layer.
    """

    def __init__(
        self,
        hidden_size: int = 768,
        num_labels: int = 4,
        num_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        num_layers = max(1, num_layers)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

        self.hidden_layers = nn.ModuleList(
            [nn.Linear(hidden_size, hidden_size) for _ in range(num_layers)]
        )
        self.layer_norms = nn.ModuleList(
            [nn.LayerNorm(hidden_size) for _ in range(num_layers)]
        )
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for linear, norm in zip(self.hidden_layers, self.layer_norms):
            residual = x
            x = self.dropout(x)
            x = self.activation(linear(x))
            x = norm(x + residual)  # residual connection
        x = self.dropout(x)
        return self.classifier(x)


class _ConceptNode(nn.Module):
    """Single concept node: linear transform + LayerNorm with optional residual skip.

    Used internally by :class:`GCPClassifier` for both root-node input
    projections and non-root transition functions.

    When ``use_resnet=True`` a residual path is added::

        h = LayerNorm( drop(act(W·x))  +  shortcut(x) )

    where ``shortcut`` is an identity when ``in_dim == out_dim``, and a
    bias-free linear projection otherwise.
    """

    def __init__(self, in_dim: int, out_dim: int, dropout: float, use_resnet: bool):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(out_dim)
        self.use_resnet = use_resnet
        # Projection shortcut needed only when dimensions differ
        self.shortcut: Optional[nn.Linear] = (
            nn.Linear(in_dim, out_dim, bias=False)
            if (use_resnet and in_dim != out_dim)
            else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.drop(self.act(self.linear(x)))
        if self.use_resnet:
            res = self.shortcut(x) if self.shortcut is not None else x
            h = h + res
        return self.norm(h)


class GCPClassifier(nn.Module):
    """Graph of Concept Predictors (GCP) classification head.

    Implements the reasoning-aware distillation architecture from:
        *Distilling LLM Reasoning into Graph of Concept Predictors*
        (Yu & Zhao, 2026, arXiv:2602.03006)

    The teacher's reasoning process is mirrored as a **Directed Acyclic Graph
    (DAG)** of concept nodes.  Each node maintains a continuous concept
    embedding and a node-specific concept predictor.  Information propagates
    from parent nodes to child nodes through learnable transition functions;
    sink node(s) are concatenated and projected to produce the final task
    prediction.

    Concept propagation equations (see §3.2 of the paper):

    * Root nodes (no parents):
        ``h₀ = f₀(x)`` — project the input CLS embedding.
    * Non-root nodes:
        ``hⱼ = fⱼ(concat({hᵢ | i ∈ Pa(j)}))`` — transform concatenated
        parent embeddings through a node-specific MLP.

    The graph topology must be supplied explicitly via ``edges``; no default
    topology is assumed.  The provided graph is validated on construction:
    it must be a connected DAG (no cycles, no isolated nodes).

    Args:
        hidden_size (int): Dimension of the input CLS embedding.
        num_labels (int): Number of output classes for the final task
            prediction.
        concept_dim (int): Feature dimension used for every concept node
            embedding.
        edges (list[tuple[int, int]]): Directed edges ``(parent, child)``
            defining the concept DAG.  Node indices must be contiguous
            integers starting at 0.  The graph must be a valid DAG with no
            isolated nodes (every node must appear in at least one edge).
        use_resnet (bool): If ``True``, each concept node's transform adds a
            residual (skip) connection::

                hⱼ = LayerNorm( drop(act(Wⱼ · x))  +  shortcut(x) )

            The shortcut is an identity when the input and output dimensions
            match (single-parent non-root nodes), or a bias-free linear
            projection otherwise (root nodes and multi-parent merges).
            Defaults to ``False``.
        dropout (float): Dropout probability applied in all sub-modules.

    Raises:
        ValueError: If ``edges`` is empty, contains a cycle, or leaves any
            node isolated (connected to no edges).

    Example — linear chain, standard::

        # 0 → 1 → 2 → 3
        head = GCPClassifier(hidden_size=768, num_labels=4,
                             edges=[(0, 1), (1, 2), (2, 3)])

    Example — linear chain with ResNet skip connections::

        head = GCPClassifier(hidden_size=768, num_labels=4,
                             edges=[(0, 1), (1, 2), (2, 3)],
                             use_resnet=True)

    Example — branching DAG (two roots merge, then project to output)::

        # 0 → 2, 1 → 2, 2 → 3
        head = GCPClassifier(hidden_size=768, num_labels=4,
                             edges=[(0, 2), (1, 2), (2, 3)])
    """

    def __init__(
        self,
        hidden_size: int = 768,
        num_labels: int = 4,
        concept_dim: int = 256,
        edges: List[Tuple[int, int]] = [],
        use_resnet: bool = False,
        dropout: float = 0.1,
    ):
        super().__init__()

        # ── Validate and build DAG topology ──────────────────────────────
        edges = list(edges)
        if not edges:
            raise ValueError(
                "GCPClassifier requires a non-empty 'edges' list defining the "
                "concept DAG.  Provide directed edges as (parent, child) pairs "
                "with contiguous node indices starting at 0."
            )

        num_nodes = max(max(p, c) for p, c in edges) + 1

        # Check for isolated nodes: every node index in [0, num_nodes) must
        # appear in at least one edge (as parent or child).
        connected = set()
        for p, c in edges:
            connected.add(p)
            connected.add(c)
        isolated = sorted(set(range(num_nodes)) - connected)
        if isolated:
            raise ValueError(
                f"GCPClassifier: node(s) {isolated} are isolated — they do not "
                "appear in any edge.  Every node must be reachable from the "
                "graph structure."
            )

        self.num_nodes = num_nodes
        self.concept_dim = concept_dim
        self.use_resnet = use_resnet
        self.dropout_layer = nn.Dropout(dropout)

        # parents[j] = sorted list of parent node indices for node j
        parents: Dict[int, List[int]] = {j: [] for j in range(num_nodes)}
        children: Dict[int, List[int]] = {j: [] for j in range(num_nodes)}
        for p, c in edges:
            parents[c].append(p)
            children[p].append(c)
        # Store as plain Python dicts (not parameters) so they survive pickling
        self._parents: Dict[int, List[int]] = {j: sorted(v) for j, v in parents.items()}
        self._children: Dict[int, List[int]] = {j: sorted(v) for j, v in children.items()}

        # Topological order via Kahn's algorithm (raises on cycle)
        self._topo_order: List[int] = self._build_topo_order(num_nodes, parents, children)

        # Sink nodes: nodes with no outgoing edges → contribute to final prediction
        self._sink_nodes: List[int] = sorted(
            j for j in range(num_nodes) if not children[j]
        )

        # ── Learnable modules ─────────────────────────────────────────────
        # Input projections for root nodes (no parents).
        # use_resnet: shortcut projects hidden_size → concept_dim when they differ.
        self.input_projections = nn.ModuleDict()
        for j in range(num_nodes):
            if not self._parents[j]:
                self.input_projections[str(j)] = _ConceptNode(
                    in_dim=hidden_size,
                    out_dim=concept_dim,
                    dropout=dropout,
                    use_resnet=use_resnet,
                )

        # Node-specific transition functions for non-root nodes.
        # input dim = concept_dim × |Pa(j)|
        # use_resnet: shortcut projects that concatenated dim → concept_dim.
        self.transition_fns = nn.ModuleDict()
        for j in range(num_nodes):
            if self._parents[j]:
                in_dim = concept_dim * len(self._parents[j])
                self.transition_fns[str(j)] = _ConceptNode(
                    in_dim=in_dim,
                    out_dim=concept_dim,
                    dropout=dropout,
                    use_resnet=use_resnet,
                )

        # Per-node concept predictors (used for intermediate supervision)
        self.concept_predictors = nn.ModuleList(
            [nn.Linear(concept_dim, num_labels) for _ in range(num_nodes)]
        )

        # Final aggregation: concat sink embeddings → task logits
        self.final_classifier = nn.Linear(concept_dim * len(self._sink_nodes), num_labels)

    # ── Private helpers ───────────────────────────────────────────────────

    @staticmethod
    def _build_topo_order(
        num_nodes: int,
        parents: Dict[int, List[int]],
        children: Dict[int, List[int]],
    ) -> List[int]:
        """Return a topological ordering of the DAG via Kahn's algorithm."""
        in_deg = [len(parents[j]) for j in range(num_nodes)]
        queue = [j for j in range(num_nodes) if in_deg[j] == 0]
        order: List[int] = []
        while queue:
            node = queue.pop(0)
            order.append(node)
            for child in children[node]:
                in_deg[child] -= 1
                if in_deg[child] == 0:
                    queue.append(child)
        if len(order) != num_nodes:
            raise ValueError(
                "GCPClassifier: the provided edges contain a cycle; "
                "only directed acyclic graphs (DAGs) are supported."
            )
        return order

    def _propagate(self, x: torch.Tensor) -> Dict[int, torch.Tensor]:
        """Run the concept propagation forward pass.

        Args:
            x (torch.Tensor): Dropout-applied CLS embeddings ``[B, hidden_size]``.

        Returns:
            dict mapping node index → concept embedding ``[B, concept_dim]``.
        """
        node_emb: Dict[int, torch.Tensor] = {}
        for j in self._topo_order:
            if not self._parents[j]:
                node_emb[j] = self.input_projections[str(j)](x)
            else:
                parent_cat = torch.cat([node_emb[p] for p in self._parents[j]], dim=-1)
                node_emb[j] = self.transition_fns[str(j)](parent_cat)
        return node_emb

    # ── Public API ────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: propagate through the concept DAG and return task logits.

        Args:
            x (torch.Tensor): CLS embeddings, shape ``[batch, hidden_size]``.

        Returns:
            torch.Tensor: Task logits, shape ``[batch, num_labels]``.
        """
        node_emb = self._propagate(self.dropout_layer(x))
        sink_cat = torch.cat([node_emb[s] for s in self._sink_nodes], dim=-1)
        return self.final_classifier(sink_cat)

    def forward_with_concepts(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Forward pass returning both final logits and per-node concept logits.

        Useful for concept-level supervision during training (distilling
        intermediate reasoning signals from the LLM teacher) and for
        diagnostics / interpretability.

        Args:
            x (torch.Tensor): CLS embeddings, shape ``[batch, hidden_size]``.

        Returns:
            tuple:
                - **final_logits** (*torch.Tensor*): shape ``[batch, num_labels]``.
                - **concept_logits** (*list[torch.Tensor]*): one
                  ``[batch, num_labels]`` tensor per concept node, ordered
                  according to the topological traversal of the DAG.
        """
        node_emb = self._propagate(self.dropout_layer(x))
        # For sink nodes, node_emb[j] is used in both concept_predictors and sink_cat.
        # Using the same tensor in two branches can trigger in-place modification
        # during backward (e.g. in LayerNorm/Dropout), causing autograd version errors.
        # Clone for one branch to avoid "modified by an inplace operation".
        concept_logits = [
            self.concept_predictors[j](
                node_emb[j].clone() if j in self._sink_nodes else node_emb[j]
            )
            for j in self._topo_order
        ]
        sink_cat = torch.cat([node_emb[s] for s in self._sink_nodes], dim=-1)
        final_logits = self.final_classifier(sink_cat)
        return final_logits, concept_logits


# ---------------------------------------------------------------------------
# Classifier factory
# ---------------------------------------------------------------------------

#: Registry mapping lowercase type names to classifier classes.
CLASSIFIER_REGISTRY: Dict[str, Type[nn.Module]] = {
    "deep_mlp": DeepMLPClassifier,
    "gcp": GCPClassifier,
}


def build_classifier(
    classifier_type: str = "deep_mlp",
    hidden_size: int = 768,
    num_labels: int = 4,
    **classifier_kwargs,
) -> nn.Module:
    """Instantiate a classification head by name.

    Args:
        classifier_type (str):
            Architecture to use.  One of ``"deep_mlp"`` (default) or ``"gcp"``.
        hidden_size (int):
            Encoder hidden dimension.  Must match the dimension of the CLS
            embedding produced by the shared encoder.
        num_labels (int):
            Number of output classes.
        **classifier_kwargs:
            Architecture-specific keyword arguments forwarded verbatim to
            the selected classifier class.  For example:

            * ``"deep_mlp"`` – ``num_layers=4``, ``dropout=0.2``
            * ``"gcp"``      – ``concept_dim=256``, ``use_resnet=False``,
              ``edges=[(0,1),(1,2),(2,3)]`` (required)

    Returns:
        nn.Module: Freshly initialised (un-trained) classification head.

    Raises:
        ValueError: If ``classifier_type`` is not in the registry.

    Example::

        head = build_classifier("deep_mlp", hidden_size=768, num_labels=3,
                                num_layers=4, dropout=0.1)
    """
    key = classifier_type.lower()
    if key not in CLASSIFIER_REGISTRY:
        raise ValueError(
            f"Unknown classifier_type {classifier_type!r}. "
            f"Available options: {sorted(CLASSIFIER_REGISTRY)}"
        )
    return CLASSIFIER_REGISTRY[key](
        hidden_size=hidden_size, num_labels=num_labels, **classifier_kwargs
    )

@torch.no_grad()
def annotate_with_classifier(encoder, classifier, tokenizer, texts: List[str], device) -> List[int]:
    """Fallback path: use current model predictions as pseudo labels when LLM is unavailable."""
    classifier.eval()
    enc = tokenizer(
        texts,
        truncation=True,
        max_length=128,
        padding=True,
        return_tensors="pt",
    )
    enc = {k: v.to(device) for k, v in enc.items()}
    outputs = encoder(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"]).last_hidden_state[:, 0]
    logits = classifier(outputs)
    return logits, outputs