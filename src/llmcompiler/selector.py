"""
Active learning sample selection utilities.

This module provides common query strategies that work with PyTorch tensors.
All methods return indices of selected samples, ranked from high priority to low.
"""

from __future__ import annotations

from typing import Iterable, List, Optional, Sequence

import torch


def _normalize_budget(budget: int, pool_size: int) -> int:
    if budget <= 0:
        return 0
    return min(int(budget), int(pool_size))


def _to_2d_probs(probs: torch.Tensor) -> torch.Tensor:
    if not torch.is_tensor(probs):
        probs = torch.as_tensor(probs)
    probs = probs.float()
    if probs.ndim != 2:
        raise ValueError("`probs` must have shape [N, C].")
    return probs


def _to_3d_mc_probs(mc_probs: torch.Tensor) -> torch.Tensor:
    if not torch.is_tensor(mc_probs):
        mc_probs = torch.as_tensor(mc_probs)
    mc_probs = mc_probs.float()
    if mc_probs.ndim != 3:
        raise ValueError("`mc_probs` must have shape [T, N, C].")
    return mc_probs


def _topk_indices(scores: torch.Tensor, budget: int, largest: bool = True) -> List[int]:
    budget = _normalize_budget(budget, scores.numel())
    if budget == 0:
        return []
    values, indices = torch.topk(scores, k=budget, largest=largest)
    _ = values
    return indices.cpu().tolist()


class ActiveLearningSelector:
    """Collection of common active learning query strategies."""

    @staticmethod
    def random_sampling(
        pool_size: int, budget: int, seed: Optional[int] = None
    ) -> List[int]:
        """Uniform random sampling over pool indices [0, pool_size)."""
        budget = _normalize_budget(budget, pool_size)
        if budget == 0:
            return []
        g = torch.Generator()
        if seed is not None:
            g.manual_seed(seed)
        perm = torch.randperm(pool_size, generator=g)[:budget]
        return perm.cpu().tolist()

    @staticmethod
    def least_confidence(probs: torch.Tensor, budget: int) -> List[int]:
        """
        Uncertainty sampling by least confidence.
        score = 1 - max_c p(y=c|x)
        """
        probs = _to_2d_probs(probs)
        uncertainty = 1.0 - probs.max(dim=1).values
        return _topk_indices(uncertainty, budget, largest=True)

    @staticmethod
    def margin_sampling(probs: torch.Tensor, budget: int) -> List[int]:
        """
        Uncertainty sampling by smallest margin between top-2 classes.
        Smaller margin means more uncertain.
        """
        probs = _to_2d_probs(probs)
        if probs.size(1) < 2:
            raise ValueError("`margin_sampling` requires class count C >= 2.")
        top2 = torch.topk(probs, k=2, dim=1).values
        margin = top2[:, 0] - top2[:, 1]
        return _topk_indices(margin, budget, largest=False)

    @staticmethod
    def entropy_sampling(probs: torch.Tensor, budget: int, eps: float = 1e-12) -> List[int]:
        """
        Uncertainty sampling by predictive entropy.
        score = -sum_c p_c * log(p_c)
        """
        probs = _to_2d_probs(probs).clamp_min(eps)
        entropy = -(probs * probs.log()).sum(dim=1)
        return _topk_indices(entropy, budget, largest=True)

    @staticmethod
    def bald_sampling(mc_probs: torch.Tensor, budget: int, eps: float = 1e-12) -> List[int]:
        """
        BALD for MC Dropout / Bayesian approximation.

        Args:
            mc_probs: [T, N, C], T stochastic forward passes.
        """
        mc_probs = _to_3d_mc_probs(mc_probs).clamp_min(eps)
        mean_probs = mc_probs.mean(dim=0)  # [N, C]

        predictive_entropy = -(mean_probs * mean_probs.log()).sum(dim=1)  # [N]
        expected_entropy = -(mc_probs * mc_probs.log()).sum(dim=2).mean(dim=0)  # [N]
        mutual_info = predictive_entropy - expected_entropy

        return _topk_indices(mutual_info, budget, largest=True)

    @staticmethod
    def kcenter_greedy(
        embeddings: torch.Tensor,
        budget: int,
        already_selected: Optional[Sequence[int]] = None,
    ) -> List[int]:
        """
        Diversity-based core-set selection using k-center greedy.

        Args:
            embeddings: [N, D] feature vectors.
            budget: number of new points to select.
            already_selected: optional indices that are already labeled.
        """
        if not torch.is_tensor(embeddings):
            embeddings = torch.as_tensor(embeddings)
        embeddings = embeddings.float()
        if embeddings.ndim != 2:
            raise ValueError("`embeddings` must have shape [N, D].")

        n = embeddings.size(0)
        budget = _normalize_budget(budget, n)
        if budget == 0:
            return []

        selected_mask = torch.zeros(n, dtype=torch.bool)
        if already_selected is not None:
            idx_tensor = torch.as_tensor(list(already_selected), dtype=torch.long)
            idx_tensor = idx_tensor[(idx_tensor >= 0) & (idx_tensor < n)]
            selected_mask[idx_tensor] = True

        if selected_mask.all():
            return []

        min_dist = torch.full((n,), float("inf"), dtype=embeddings.dtype)
        if selected_mask.any():
            labeled = embeddings[selected_mask]
            dist_to_labeled = torch.cdist(embeddings, labeled, p=2)
            min_dist = dist_to_labeled.min(dim=1).values
        else:
            first = torch.randint(0, n, (1,)).item()
            selected_mask[first] = True
            min_dist = torch.cdist(embeddings, embeddings[first : first + 1], p=2).squeeze(1)

        new_selected: List[int] = []
        while len(new_selected) < budget:
            candidates = torch.where(~selected_mask)[0]
            if candidates.numel() == 0:
                break
            pick_local = torch.argmax(min_dist[candidates]).item()
            pick = candidates[pick_local].item()

            selected_mask[pick] = True
            new_selected.append(pick)

            dist_to_pick = torch.cdist(embeddings, embeddings[pick : pick + 1], p=2).squeeze(1)
            min_dist = torch.minimum(min_dist, dist_to_pick)

        return new_selected

    @staticmethod
    def select(
        method: str,
        budget: int,
        probs: Optional[torch.Tensor] = None,
        embeddings: Optional[torch.Tensor] = None,
        mc_probs: Optional[torch.Tensor] = None,
        already_selected: Optional[Sequence[int]] = None,
        pool_size: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> List[int]:
        """
        Unified selection entry point.

        Supported methods:
        - random
        - least_confidence
        - margin
        - entropy
        - bald
        - kcenter
        """
        method = method.lower().strip()

        if method == "random":
            if pool_size is None:
                if probs is not None:
                    pool_size = int(_to_2d_probs(probs).size(0))
                elif embeddings is not None:
                    pool_size = int(torch.as_tensor(embeddings).size(0))
                else:
                    raise ValueError("`random` requires `pool_size`, `probs`, or `embeddings`.")
            return ActiveLearningSelector.random_sampling(pool_size, budget, seed=seed)

        if method == "least_confidence":
            if probs is None:
                raise ValueError("`least_confidence` requires `probs`.")
            return ActiveLearningSelector.least_confidence(probs, budget)

        if method == "margin":
            if probs is None:
                raise ValueError("`margin` requires `probs`.")
            return ActiveLearningSelector.margin_sampling(probs, budget)

        if method == "entropy":
            if probs is None:
                raise ValueError("`entropy` requires `probs`.")
            return ActiveLearningSelector.entropy_sampling(probs, budget)

        if method == "bald":
            if mc_probs is None:
                raise ValueError("`bald` requires `mc_probs` with shape [T, N, C].")
            return ActiveLearningSelector.bald_sampling(mc_probs, budget)

        if method in {"kcenter"}:
            if embeddings is None:
                raise ValueError("`kcenter` requires `embeddings`.")
            return ActiveLearningSelector.kcenter_greedy(
                embeddings=embeddings,
                budget=budget,
                already_selected=already_selected,
            )

        raise ValueError(
            f"Unsupported method `{method}`. "
            "Use one of: random, least_confidence, margin, entropy, bald, kcenter."
        )


__all__ = ["ActiveLearningSelector"]
