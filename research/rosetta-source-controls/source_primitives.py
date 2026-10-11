"""Draft primitives for the separate full-bank source-overlap retrieval study.

These are not an execution entry point or a frozen protocol. A future supervisor
must bind source/runtime/cache hashes, canonical identities, whole-source masks,
fixed strata and the complete schedule before real fitting. No masks are sampled
here. Upstream refers to fdfad84ea488f29702a61a455be71622c4ff5d49.

Training retains the authors' stored-precision reduction and independent bank
permutations. Evaluation retains their FP32 similarities, independent shuffles,
FP64 ranking, and full-gallery FOSCTTM denominator. Strata select queries only.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor

from unpaired_rosetta.embeddings import load_rows, num_rows
from unpaired_rosetta.evaluation import Aligner, independent_split, midranks
from unpaired_rosetta.randomness import Randomness
from unpaired_rosetta.wasserstein_procrustes import WassersteinProcrustes


@dataclass
class TrainingArm:
    samples_x: Tensor
    samples_y: Tensor
    indices_x: Tensor
    indices_y: Tensor
    randomness: Randomness


def canonical_mask(
    rows: Tensor,
    population_size: int,
    source_keys: Sequence[str] | None = None,
) -> Tensor:
    """Validate increasing original row indices; optionally enforce whole sources.

    The caller supplies already-canonical source keys (COCO aliases included).
    Without keys this checks row order only, not source membership or aliasing.
    """
    if rows.device.type != "cpu" or rows.dtype != torch.int64 or rows.ndim != 1:
        raise ValueError("Mask must be a one-dimensional CPU int64 tensor")
    if not rows.numel() or rows[0] < 0 or rows[-1] >= population_size:
        raise ValueError("Mask must be nonempty and within the original population")
    if rows.numel() > 1 and not bool(torch.all(rows[1:] > rows[:-1])):
        raise ValueError("Mask must be unique and in canonical increasing row order")
    if source_keys is not None:
        if len(source_keys) != population_size or any(not isinstance(k, str) or not k for k in source_keys):
            raise ValueError("A canonical source key is required for every original row")
        chosen = set(rows.tolist())
        retained_keys = {source_keys[i] for i in chosen}
        if any(k in retained_keys and i not in chosen for i, k in enumerate(source_keys)):
            raise ValueError("Mask splits a canonical source group")
    return rows.clone()


def load_training_arm(
    path_x: Path,
    path_y: Path,
    canonical_y_rows: Tensor,
    seed: int,
    *,
    source_keys: Sequence[str] | None = None,
) -> TrainingArm:
    """Keep all X; mask Y in file order, then consume the original full shuffles.

    In arm O (all Y rows), this is exactly the cross-dataset loader's sequence.
    Both means will be learned from these selected banks by the original fit.
    Equal seeds across differently sized arms need not give later equal draws.
    """
    nx, ny = num_rows(path_x), num_rows(path_y)
    if nx < 1:
        raise ValueError("The full image bank must be nonempty")
    selected = canonical_mask(canonical_y_rows, ny, source_keys)
    randomness = Randomness(seed)
    indices_x, positions_y = independent_split(nx, len(selected), randomness, None)
    indices_y = selected[positions_y]
    # Do not cast stored BF16 before load_rows performs the original reduction.
    samples_x, samples_y = load_rows(path_x, indices_x), load_rows(path_y, indices_y)
    if not torch.isfinite(samples_x).all() or not torch.isfinite(samples_y).all():
        raise ValueError("Selected training rows must reduce to finite embeddings")
    return TrainingArm(samples_x, samples_y, indices_x, indices_y, randomness)


def fit_original(arm: TrainingArm) -> WassersteinProcrustes:
    """Fit unchanged upstream defaults, 4 workers/threads, no paired counterparts.

    This consumes the same Randomness object advanced by load_training_arm.
    The caller must not reuse an arm for a new attempt after this consumption.
    Original C=30, S=30, b=10000, R=100 remain constructor defaults; smaller
    available text banks are passed whole, without padding or an image cap.
    """
    aligner = WassersteinProcrustes(num_workers=4, num_threads=4)
    aligner.fit(arm.samples_x, arm.samples_y, None, None, arm.randomness)
    return aligner


def summarize_ranks(ranks: Tensor, num_higher: Tensor, tie_count: Tensor, gallery_size: int) -> dict:
    """Descriptive query metrics against an unchanged gallery; ties include true Y.

    Recall@k is expected credit under uniform ordering within the tie block:
    clamp(k - num_higher, 0, tie_count) / tie_count. Median averages the two
    central ranks for even query counts. Empty strata have null metrics.
    """
    if gallery_size < 2:
        raise ValueError("FOSCTTM requires at least two gallery items")
    if ranks.ndim != 1 or ranks.shape != num_higher.shape or ranks.shape != tie_count.shape:
        raise ValueError("Rank and count vectors must have the same one-dimensional shape")
    if ranks.dtype != torch.float64 or num_higher.dtype != torch.int64 or tie_count.dtype != torch.int64:
        raise ValueError("Ranks must be float64 and counts int64")
    if any(t.device.type != "cpu" for t in (ranks, num_higher, tie_count)):
        raise ValueError("Rank records must be on CPU")
    if not torch.isfinite(ranks).all() or (num_higher < 0).any() or (tie_count < 1).any():
        raise ValueError("Invalid ranks or tie counts")
    if (num_higher + tie_count > gallery_size).any():
        raise ValueError("Tie block extends outside the full gallery")
    expected = num_higher.double() + 0.5 * (tie_count.double() - 1) + 1
    if not torch.equal(ranks, expected):
        raise ValueError("Ranks disagree with strictly-higher and tied counts")
    n = len(ranks)
    result = {"num_queries": n, "gallery_size": gallery_size, "foscttm_denominator": gallery_size - 1}
    if not n:
        result.update({name: None for name in ("foscttm", "mean_rank", "median_rank", "recall_at_1", "recall_at_5", "recall_at_10")})
        return result
    ordered = torch.sort(ranks).values
    result.update({
        "foscttm": ((ranks - 1) / (gallery_size - 1)).mean().item(),
        "mean_rank": ranks.mean().item(),
        "median_rank": ((ordered[(n - 1) // 2] + ordered[n // 2]) / 2).item(),
    })
    for k in (1, 5, 10):
        credit = ((k - num_higher).double() / tie_count.double()).clamp(0, 1)
        result[f"recall_at_{k}"] = credit.mean().item()
    return result


@torch.inference_mode()
def retrieval_with_ties(
    aligner: Aligner,
    val_x: Tensor,
    val_y: Tensor,
    randomness: Randomness,
    *,
    canonical_rows: Tensor | None = None,
    strata: Mapping[str, Tensor] | None = None,
    block_size: int = 4096,
) -> dict:
    """Score the full gallery once; return canonical file-order ranks and counts.

    canonical_rows maps each incoming paired validation row to its original file
    row, and must be a full permutation, not a subset. Every stratum is a fixed
    boolean mask in that canonical order. This function consumes only the two
    evaluator shuffles; evaluate_full_gallery also consumes the original earlier
    validation subset draw from a fresh, independent Randomness(seed).
    """
    n = val_x.shape[0]
    if val_x.ndim != 2 or val_y.ndim != 2 or n < 2 or val_y.shape[0] != n:
        raise ValueError("Validation must contain complete row-paired matrices with at least two items")
    if block_size < 1 or isinstance(block_size, bool):
        raise ValueError("Block size must be positive")
    if any(t.device.type != "cpu" or t.dtype != torch.float32 or not torch.isfinite(t).all() for t in (val_x, val_y)):
        raise ValueError("Validation must use finite CPU FP32 embeddings after the authors' reduction")
    canonical_rows = torch.arange(n) if canonical_rows is None else canonical_rows
    if canonical_rows.dtype != torch.int64 or canonical_rows.device.type != "cpu" or canonical_rows.shape != (n,):
        raise ValueError("Canonical row mapping must be a full CPU int64 permutation")
    if not torch.equal(torch.sort(canonical_rows).values, torch.arange(n)):
        raise ValueError("Evaluation must retain every canonical gallery row exactly once")
    strata = {} if strata is None else dict(strata)
    for name, mask in strata.items():
        if not isinstance(name, str) or not name or mask.dtype != torch.bool or mask.device.type != "cpu" or mask.shape != (n,):
            raise ValueError("Each named fixed stratum must be a canonical full-length CPU boolean mask")

    order_x, order_y = randomness.permutation(n), randomness.permutation(n)
    shuffled_x, shuffled_y = val_x[order_x], val_y[order_y]
    partner = torch.argsort(order_y)[order_x]
    ranks = torch.empty(n, dtype=torch.float64)
    higher = torch.empty(n, dtype=torch.int64)
    ties = torch.empty(n, dtype=torch.int64)
    for start in range(0, n, block_size):
        stop = min(start + block_size, n)
        raw_scores = aligner.similarity(shuffled_x[start:stop], shuffled_y)
        if raw_scores.dtype != torch.float32 or raw_scores.shape != (stop - start, n):
            raise ValueError("Original similarity must return an FP32 full-gallery score block")
        scores = raw_scores.double().cpu()
        if not torch.isfinite(scores).all():
            raise ValueError("Nonfinite similarity scores")
        true_scores = scores[torch.arange(stop - start), partner[start:stop]]
        rows = order_x[start:stop]
        ranks[rows] = midranks(scores, true_scores)
        higher[rows] = (scores > true_scores[:, None]).sum(dim=1)
        ties[rows] = (scores == true_scores[:, None]).sum(dim=1)

    # Preserve upstream's incoming row-order reduction for bitwise comparison.
    overall = summarize_ranks(ranks, higher, ties, n)
    inverse = torch.argsort(canonical_rows)
    ranks, higher, ties = ranks[inverse], higher[inverse], ties[inverse]
    return {
        "metrics": {k: overall[k] for k in ("foscttm", "mean_rank")},
        "overall": overall,
        "strata": {name: summarize_ranks(ranks[mask], higher[mask], ties[mask], n) for name, mask in strata.items()},
        "ranks": ranks,
        "num_higher": higher,
        "tie_count": ties,
        "query_order": canonical_rows[order_x],
        "key_order": canonical_rows[order_y],
        "validation_order": canonical_rows.clone(),
        "rank_order": "canonical_original_file_row",
        "tie_policy": "expected_credit_under_uniform_order_within_exact_score_tie_block",
        "overall_reduction_order": "original_randomized_validation_order",
    }


def evaluate_full_gallery(
    aligner: Aligner,
    path_x: Path,
    path_y: Path,
    seed: int,
    *,
    strata: Mapping[str, Tensor] | None = None,
    block_size: int = 4096,
) -> dict:
    """Original fresh evaluation RNG, full subset draw, reduction, then shuffles."""
    n = num_rows(path_x)
    if n != num_rows(path_y):
        raise ValueError("Validation banks must have the same complete paired population")
    randomness = Randomness(seed)
    indices = randomness.subset(n, None)
    x, y = load_rows(path_x, indices), load_rows(path_y, indices)
    return retrieval_with_ties(aligner, x, y, randomness, canonical_rows=indices, strata=strata, block_size=block_size)
