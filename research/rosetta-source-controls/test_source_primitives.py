"""Artificial fixtures only: no published embeddings, masks, fits or scores.

Compatibility checks call the untouched authors' loaders/evaluator. The fit test
uses a spy, not an optimization. These tests do not authorize a real run.
"""

from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from experiments.alignment import AlignmentRun, evaluate, load_training_data, load_validation_data
from unpaired_rosetta.embeddings import load_rows, reduce_texts
from unpaired_rosetta.evaluation import retrieval_metrics
from unpaired_rosetta.randomness import Randomness
from unpaired_rosetta.wasserstein_procrustes import WassersteinProcrustes

import source_primitives as source


class TableAligner:
    def __init__(self, table):
        self.table, self.calls = table, []

    def similarity(self, x, y):
        self.calls.append((x.clone(), y.clone()))
        return self.table[x[:, 0].long()][:, y[:, 0].long()]


class DotAligner:
    def similarity(self, x, y):
        return x @ y.T


def save_banks(tmp_path, *, slots=1):
    rng = torch.Generator().manual_seed(41)
    raw_x = torch.randn(17, 4, generator=rng).bfloat16()
    raw_y = torch.randn(11, slots, 3, generator=rng).bfloat16()
    if slots > 1:
        raw_y[::2, -1] = float("nan")
    x, y = tmp_path / "x.pt", tmp_path / "y.pt"
    torch.save(raw_x, x)
    torch.save(raw_y, y)
    return x, y, raw_x, raw_y


def assert_rng_equal(a, b):
    assert torch.equal(a.torch.get_state(), b.torch.get_state())
    assert a.kmeans_seed() == b.kmeans_seed()
    assert a.python.random() == b.python.random()
    assert a.solver_seed() == b.solver_seed()


@pytest.mark.parametrize("seed", [734796314, 576165995, 2197670066, 839703249, 2584932063])
@pytest.mark.parametrize("slots", [1, 3])
def test_original_arm_matches_untouched_cross_dataset_loader_and_rng(tmp_path, seed, slots):
    x, y, raw_x, raw_y = save_banks(tmp_path, slots=slots)
    run = AlignmentRun("coco_train2014", "synthetic-x", "synthetic-y", seed, dataset_y="StanfordParagraphCaptioning")
    original_rng = Randomness(seed)
    with patch("experiments.alignment.embedding_path", side_effect=[x, y]):
        expected_x, expected_y, px, py = load_training_data(run, original_rng)
    arm = source.load_training_arm(x, y, torch.arange(len(raw_y)), seed)
    assert px is py is None
    assert torch.equal(arm.samples_x, expected_x)
    assert torch.equal(arm.samples_y, expected_y)
    assert arm.samples_x.dtype == arm.samples_y.dtype == torch.float32
    assert torch.equal(arm.samples_y, reduce_texts(raw_y[arm.indices_y]).float())
    # This fixture detects an accidental early cast to FP32, not just shape drift.
    assert not torch.equal(arm.samples_y, reduce_texts(raw_y[arm.indices_y].float()))
    assert_rng_equal(arm.randomness, original_rng)


def test_mask_before_permutation_keeps_all_images_and_selected_rows(tmp_path):
    x, y, raw_x, raw_y = save_banks(tmp_path)
    selected = torch.tensor([0, 1, 5, 7, 10])
    arm = source.load_training_arm(x, y, selected, 31)
    expected_rng = Randomness(31)
    expected_x = expected_rng.permutation(len(raw_x))
    expected_y = selected[expected_rng.permutation(len(selected))]
    assert torch.equal(arm.indices_x, expected_x)
    assert torch.equal(arm.indices_y, expected_y)
    assert torch.equal(arm.samples_y, load_rows(y, expected_y))
    assert len(arm.samples_x) == 17 and len(arm.samples_y) == 5
    assert_rng_equal(arm.randomness, expected_rng)


@pytest.mark.parametrize("rows", [[], [1, 0], [0, 0], [-1, 1], [0, 11]])
def test_noncanonical_masks_rejected(rows):
    with pytest.raises(ValueError):
        source.canonical_mask(torch.tensor(rows, dtype=torch.int64), 11)


def test_canonical_alias_group_must_be_retained_whole():
    # Distinct VG IDs can have the same COCO key; validate the canonical key.
    keys = ["coco:112040", "vg:2", "coco:112040", "vg:4"]
    with pytest.raises(ValueError, match="splits"):
        source.canonical_mask(torch.tensor([0, 1]), 4, keys)
    assert source.canonical_mask(torch.tensor([0, 2, 3]), 4, keys).tolist() == [0, 2, 3]


def test_fit_passes_advanced_rng_and_original_defaults_without_fitting(tmp_path):
    x, y, _, raw_y = save_banks(tmp_path)
    arm = source.load_training_arm(x, y, torch.arange(len(raw_y)), 29)
    before = arm.randomness.torch.get_state().clone()
    assert not torch.equal(before, Randomness(29).torch.get_state())
    original = WassersteinProcrustes(num_workers=4, num_threads=4)
    assert (original.num_clusters, original.num_restarts, original.batch_size, original.num_iterations) == (30, 30, 10000, 100)
    with patch.object(source, "WassersteinProcrustes") as factory:
        returned = source.fit_original(arm)
        factory.assert_called_once_with(num_workers=4, num_threads=4)
        args = factory.return_value.fit.call_args.args
        assert args[0] is arm.samples_x and args[1] is arm.samples_y
        assert args[2] is args[3] is None and args[4] is arm.randomness
        assert returned is factory.return_value
    assert torch.equal(before, arm.randomness.torch.get_state())


@pytest.mark.parametrize("block", [1, 3, 4096])
@pytest.mark.parametrize("seed", [0, 17])
def test_ties_match_original_evaluator_and_hand_computed_counts(block, seed):
    table = torch.tensor([[3, 3, 0, 1], [5, 4, 4, 4], [1, 4, 2, 2], [0, 3, 2, 1]], dtype=torch.float32)
    ids = torch.arange(4).float()[:, None]
    ra, rb = Randomness(seed), Randomness(seed)
    indices = ra.subset(4, None)
    assert torch.equal(indices, rb.subset(4, None))
    a, b = TableAligner(table), TableAligner(table)
    original = retrieval_metrics(a, ids[indices], ids[indices], ra, block_size=block)
    out = source.retrieval_with_ties(b, ids[indices], ids[indices], rb, canonical_rows=indices, block_size=block)
    assert out["metrics"] == original == {"foscttm": 0.5, "mean_rank": 2.5}
    assert out["ranks"].tolist() == [1.5, 3.0, 2.5, 3.0]
    assert out["num_higher"].tolist() == [0, 1, 1, 2]
    assert out["tie_count"].tolist() == [2, 3, 2, 1]
    assert out["overall"]["median_rank"] == 2.75
    assert out["overall"]["recall_at_1"] == 0.125
    assert out["overall"]["recall_at_5"] == out["overall"]["recall_at_10"] == 1
    for call_a, call_b in zip(a.calls, b.calls):
        assert all(torch.equal(x, y) for x, y in zip(call_a, call_b))
    assert_rng_equal(ra, rb)


def test_all_ties_recall_uses_expected_credit_not_a_midrank_threshold():
    n = 12
    ids = torch.arange(n).float()[:, None]
    out = source.retrieval_with_ties(TableAligner(torch.zeros(n, n)), ids, ids, Randomness(57))
    assert out["ranks"].tolist() == [6.5] * n
    assert out["tie_count"].tolist() == [12] * n
    assert out["overall"]["foscttm"] == 0.5
    for k in (1, 5, 10):
        assert out["overall"][f"recall_at_{k}"] == pytest.approx(k / n)
    assert not torch.equal(out["query_order"], out["key_order"])


def test_fixed_strata_use_full_gallery_denominator_and_preserve_exposed_queries():
    table = torch.tensor([[3, 3, 0, 1], [5, 4, 4, 4], [1, 4, 2, 2], [0, 3, 2, 1]], dtype=torch.float32)
    ids = torch.arange(4).float()[:, None]
    strata = {"never_known_exposed": torch.tensor([True, False, True, False]), "known_original_exposure": torch.tensor([False, True, False, True]), "empty": torch.zeros(4, dtype=torch.bool)}
    order = torch.tensor([2, 0, 3, 1])
    out = source.retrieval_with_ties(TableAligner(table), ids[order], ids[order], Randomness(35), canonical_rows=order, strata=strata)
    primary, exposed = out["strata"]["never_known_exposed"], out["strata"]["known_original_exposure"]
    assert primary["num_queries"] == exposed["num_queries"] == 2
    assert primary["gallery_size"] == exposed["gallery_size"] == 4
    assert primary["foscttm_denominator"] == 3
    assert primary["foscttm"] == pytest.approx(1 / 3)
    assert exposed["foscttm"] == pytest.approx(2 / 3)
    assert primary["median_rank"] == 2 and exposed["median_rank"] == 3
    assert primary["recall_at_1"] == 0.25 and exposed["recall_at_1"] == 0
    assert out["strata"]["empty"]["num_queries"] == 0
    assert out["strata"]["empty"]["foscttm"] is None


def test_rectangular_known_map_has_perfect_ranks_without_any_fit():
    x, y = torch.eye(6), torch.cat([torch.eye(6), torch.zeros(6, 3)], dim=1)
    class PlantedMap:
        def similarity(self, x, y):
            return torch.cat([x, torch.zeros(len(x), 3)], dim=1) @ y.T
    order = torch.tensor([4, 2, 0, 1, 5, 3])
    out = source.retrieval_with_ties(PlantedMap(), x[order], y[order], Randomness(12), canonical_rows=order, block_size=2)
    assert out["metrics"] == {"foscttm": 0.0, "mean_rank": 1.0}
    assert out["tie_count"].tolist() == [1] * 6
    assert out["overall"]["recall_at_1"] == 1


def test_full_evaluation_matches_untouched_loader_and_evaluate_in_bf16(tmp_path):
    rng = torch.Generator().manual_seed(19)
    x, y = tmp_path / "vx.pt", tmp_path / "vy.pt"
    torch.save(torch.randn(11, 5, generator=rng).bfloat16(), x)
    raw_y = torch.randn(11, 7, 5, generator=rng).bfloat16()
    raw_y[::2, 2:] = float("nan")
    torch.save(raw_y, y)
    run = AlignmentRun("unused_train", "synthetic-x", "synthetic-y", 17, dataset_y="StanfordParagraphCaptioning")
    with patch("experiments.alignment.embedding_path", side_effect=[x, y]):
        original = evaluate(DotAligner(), run)
    actual = source.evaluate_full_gallery(DotAligner(), x, y, 17)
    assert actual["metrics"] == {k: original[k] for k in ("foscttm", "mean_rank")}
    with patch("experiments.alignment.embedding_path", side_effect=[x, y]):
        original_x, original_y, indices = load_validation_data(run, Randomness(17))
    assert torch.equal(actual["validation_order"], indices)
    assert torch.equal(original_y, reduce_texts(raw_y[indices]).float())


def test_invalid_gallery_mapping_strata_and_nonfinite_scores_fail():
    ids = torch.arange(4).float()[:, None]
    aligner = TableAligner(torch.eye(4))
    with pytest.raises(ValueError, match="every canonical"):
        source.retrieval_with_ties(aligner, ids, ids, Randomness(0), canonical_rows=torch.tensor([0, 1, 1, 3]))
    with pytest.raises(ValueError, match="stratum"):
        source.retrieval_with_ties(aligner, ids, ids, Randomness(0), strata={"bad": torch.ones(2, dtype=torch.bool)})
    with pytest.raises(ValueError, match="Nonfinite"):
        source.retrieval_with_ties(TableAligner(torch.full((4, 4), float("nan"))), ids, ids, Randomness(0))
    with pytest.raises(ValueError, match="FP32"):
        source.retrieval_with_ties(TableAligner(torch.eye(4).double()), ids, ids, Randomness(0))
