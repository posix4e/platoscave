"""Synthetic-only checks against the untouched authors evaluator and known answers.

No released benchmark embeddings, user photographs or pretrained encoders are
loaded. Passing these tests does not freeze or validate an extraction protocol.
"""
import hashlib
import json

import pytest
import torch

from unpaired_rosetta.baselines.linear import LinearMap
from unpaired_rosetta.baselines.orthogonal import OrthogonalMap
from unpaired_rosetta.evaluation import retrieval_metrics
from unpaired_rosetta.randomness import Randomness
from control_primitives import fit_paired_controls, linear_pinv_diagnostics, retrieval_with_ranks, training_partner_permutation


class TableAligner:
    def __init__(self, table):
        self.table = table
        self.calls = []

    def similarity(self, x, y):
        self.calls.append((x.clone(), y.clone()))
        return self.table[x[:, 0].long()][:, y[:, 0].long()]


class MapAligner:
    def __init__(self, weight):
        self.weight = weight

    def similarity(self, x, y):
        return (x @ self.weight) @ y.T


def assert_original_agrees(aligner, x, y, seed, block):
    original_rng, extra_rng = Randomness(seed), Randomness(seed)
    # Simulate the validation subset draw occurring before retrieval_metrics.
    original_rng.subset(91, 17)
    extra_rng.subset(91, 17)
    original = retrieval_metrics(aligner, x, y, original_rng, block_size=block)
    extra = retrieval_with_ranks(aligner, x, y, extra_rng, block_size=block)
    assert extra['metrics'] == original
    assert torch.equal(original_rng.torch.get_state(), extra_rng.torch.get_state())
    return extra


def test_partner_permutation_is_repeatable_without_consuming_global_or_original_rng():
    global_before = torch.get_rng_state().clone()
    author_rng = Randomness(17)
    author_before = author_rng.torch.get_state().clone()
    a = training_partner_permutation('synthetic-protocol', 17, 100)
    b = training_partner_permutation('synthetic-protocol', 17, 100)
    assert torch.equal(a.indices, b.indices) and a.metadata == b.metadata
    assert torch.equal(torch.get_rng_state(), global_before)
    assert torch.equal(author_rng.torch.get_state(), author_before)
    assert sorted(a.indices.tolist()) == list(range(100))
    assert a.metadata['fixed_point_count'] == sum(i == j for i, j in enumerate(a.indices.tolist()))
    payload = json.dumps(a.indices.tolist(), separators=(',', ':')).encode()
    assert a.metadata['permutation_sha256'] == hashlib.sha256(payload).hexdigest()
    assert a.metadata['seed_derivation_sha256'] != training_partner_permutation('other-protocol', 17, 100).metadata['seed_derivation_sha256']
    assert a.metadata['seed_derivation_sha256'] != training_partner_permutation('synthetic-protocol', 18, 100).metadata['seed_derivation_sha256']


def test_uniform_shuffle_allows_fixed_points_instead_of_forcing_a_derangement():
    # For two items, an identity draw is direct evidence that fixed points survive.
    draws = [training_partner_permutation('fixed-point-fixture', seed, 2) for seed in range(16)]
    assert {p.metadata['fixed_point_count'] for p in draws} == {0, 2}
    assert all(not p.metadata['forced_derangement'] for p in draws)


@pytest.mark.parametrize('block', [1, 3, 4096])
@pytest.mark.parametrize('seed', [0, 17])
def test_hand_computable_ties_preserve_authors_metric_and_per_query_ranks(block, seed):
    scores = torch.tensor([[3, 3, 0, 1], [5, 4, 4, 4], [1, 4, 2, 2], [0, 3, 2, 1]], dtype=torch.float32)
    # Ranks by inspection: one tied leader; one better + two tied others;
    # one better + one tied other; two strictly better.
    expected = torch.tensor([1.5, 3.0, 2.5, 3.0], dtype=torch.float64)
    ids = torch.arange(4).float()[:, None]
    out = assert_original_agrees(TableAligner(scores), ids, ids, seed, block)
    assert torch.equal(out['ranks'], expected)
    assert out['metrics'] == {'foscttm': 0.5, 'mean_rank': 2.5}


def test_all_ties_are_exactly_chance_and_both_independent_shuffle_orders_match():
    x = torch.arange(9).float()[:, None]
    a, b = TableAligner(torch.zeros(9, 9)), TableAligner(torch.zeros(9, 9))
    rng_a, rng_b = Randomness(124), Randomness(124)
    original = retrieval_metrics(a, x, x, rng_a, block_size=4)
    recorded = retrieval_with_ranks(b, x, x, rng_b, block_size=4)
    assert original == recorded['metrics'] == {'foscttm': 0.5, 'mean_rank': 5.0}
    assert len(a.calls) == len(b.calls) == 3
    for actual, expected in zip(b.calls, a.calls):
        assert all(torch.equal(u, v) for u, v in zip(actual, expected))
    assert not torch.equal(recorded['query_order'], recorded['key_order'])
    assert torch.equal(recorded['ranks'], torch.full((9,), 5.0, dtype=torch.float64))


def test_rectangular_planted_map_and_changed_incoming_row_order():
    g = torch.Generator().manual_seed(61)
    x = torch.nn.functional.normalize(torch.randn(31, 4, generator=g), dim=1)
    w = torch.cat((torch.eye(4), torch.zeros(4, 3)), dim=1)
    y = x @ w
    out = assert_original_agrees(MapAligner(w), x, y, 22, 7)
    assert out['metrics']['foscttm'] == 0 and torch.equal(out['ranks'], torch.ones(31, dtype=torch.float64))
    order = torch.randperm(31, generator=g)
    reordered = assert_original_agrees(MapAligner(w), x[order], y[order], 71, 8)
    assert torch.equal(reordered['ranks'], out['ranks'][order])


def planted_training_banks():
    g = torch.Generator().manual_seed(37)
    # Each disjoint centering half has antipodal samples and exactly zero mean.
    first, second = torch.randn(512, 16, generator=g), torch.randn(512, 16, generator=g)
    x = torch.cat((first, -first, second, -second))
    w = torch.cat((torch.eye(16), torch.zeros(16, 5)), dim=1)
    y = x @ w
    ids_x, ids_y, ids = torch.arange(1024), torch.arange(1024, 2048), torch.arange(2048)
    val_x = torch.randn(80, 16, generator=g)
    return x[ids_x], y[ids_y], x, y, ids_x, ids_y, ids, val_x, val_x @ w


@pytest.fixture(scope='module')
def controls():
    cx, cy, x, y, ix, iy, ids, vx, vy = planted_training_banks()
    models, metadata = fit_paired_controls(cx, cy, x, y, centering_ids_x=ix, centering_ids_y=iy,
                                          paired_ids=ids, protocol_id='planted-synthetic-map', split_seed=37)
    return models, metadata, (cx, cy, x, y, ix, iy, ids, vx, vy)


@pytest.mark.parametrize('family', ['orthogonal', 'linear'])
def test_planted_map_recovered_and_shuffled_partners_fail_on_unseen_rows(controls, family):
    models, metadata, (_, _, _, _, _, _, _, x, y) = controls
    correct = assert_original_agrees(models[family + '_paired'], x, y, 92, 19)
    shuffled = assert_original_agrees(models[family + '_shuffled'], x, y, 92, 19)
    assert correct['metrics']['foscttm'] == 0
    assert shuffled['metrics']['foscttm'] > 0.2
    assert metadata['additional_x_counterparts'] == metadata['additional_y_counterparts'] == 1024
    assert metadata['paired_training_items'] == 2048


def test_controls_preserve_original_centering_and_exact_same_shuffle(controls):
    models, metadata, (cx, cy, x, y, _, _, _, _, _) = controls
    p = training_partner_permutation('planted-synthetic-map', 37, len(x))
    assert metadata['partner_permutation'] == p.metadata
    for family, cls in [('orthogonal', OrthogonalMap), ('linear', LinearMap)]:
        for suffix, target in [('paired', y), ('shuffled', y[p.indices])]:
            direct = cls().fit(cx, cy, x, target)
            wrapped = models[family + '_' + suffix]
            assert torch.equal(wrapped.mean_x, cx.mean(0, keepdim=True))
            assert torch.equal(wrapped.mean_y, cy.mean(0, keepdim=True))
            assert torch.equal(wrapped.weight, direct.weight)
    diagnostic = metadata['linear_pinv_diagnostic']
    assert diagnostic['effective_rank_by_default_cutoff'] == 16
    assert diagnostic['applies_to'] == ['linear_paired', 'linear_shuffled']
    assert diagnostic['diagnostic_svd_count'] == 1
    assert not diagnostic['fit_modified'] and not diagnostic['fit_internal_spectrum_captured']


@pytest.mark.parametrize('weak_cutoff_multiple, expected_rank', [(0.25, 2), (4.0, 3)])
def test_near_singular_design_reports_default_cutoff_without_changing_authors_fit(weak_cutoff_multiple, expected_rank):
    # Orthogonal sign columns have known singular values. Every row has the same
    # norm, so the authors' row normalization preserves their singular ratios.
    # Both disjoint centering banks have exact zero means. The weak direction
    # is mathematically nonzero on both sides of the numerical cutoff.
    signs = torch.tensor([[1., 1., 1.], [1., -1., -1.], [-1., 1., -1.], [-1., -1., 1.]])
    default_rtol = 8 * 2 ** -23
    scales = torch.tensor([1., 1e-3, weak_cutoff_multiple * default_rtol])
    x = (signs * scales).repeat(2, 1)
    ids, ix, iy = torch.arange(8), torch.arange(4), torch.arange(4, 8)
    models, metadata = fit_paired_controls(x[ix], x[iy], x, x, centering_ids_x=ix,
                                          centering_ids_y=iy, paired_ids=ids,
                                          protocol_id='near-singular-synthetic-design', split_seed=11)
    diagnostic = metadata['linear_pinv_diagnostic']
    expected_singular = (8 ** 0.5) * scales.double() / scales.double().norm()
    torch.testing.assert_close(torch.tensor(diagnostic['singular_values'], dtype=torch.float64),
                               expected_singular, rtol=1e-5, atol=1e-12)
    assert diagnostic['effective_rtol'] == default_rtol
    assert diagnostic['effective_atol'] == 0 and diagnostic['cutoff_dtype'] == 'torch.float64'
    assert diagnostic['effective_rank_by_default_cutoff'] == expected_rank
    assert diagnostic['maximum_possible_rank'] == 3
    assert diagnostic['discarded_singular_values'] == 3 - expected_rank
    assert diagnostic['absolute_cutoff'] == diagnostic['sigma_max'] * default_rtol
    assert diagnostic['nearest_cutoff_absolute_margin'] > 0
    if expected_rank == 2:
        assert diagnostic['largest_discarded_singular_value'] < diagnostic['absolute_cutoff']
        # The original solver drops this nonzero direction rather than returning
        # an unrestricted identity fit. Check the fitted map, not only metadata.
        assert abs(models['linear_paired'].weight[2, 2].item()) < 1e-5
    else:
        assert diagnostic['largest_discarded_singular_value'] is None
        assert models['linear_paired'].weight[2, 2].item() == pytest.approx(1., abs=1e-5)
    direct = LinearMap().fit(x[ix], x[iy], x, x)
    assert torch.equal(models['linear_paired'].weight, direct.weight)
    assert diagnostic['smallest_retained_singular_value'] > diagnostic['absolute_cutoff']
    json.dumps(diagnostic, allow_nan=False)


def test_zero_design_reports_zero_rank_without_infinite_or_missing_cutoff_metadata():
    diagnostic = linear_pinv_diagnostics(torch.ones(8, 3), torch.ones(1, 3))
    assert diagnostic['effective_rank_by_default_cutoff'] == 0
    assert diagnostic['absolute_cutoff'] == diagnostic['sigma_max'] == 0
    assert diagnostic['nearest_cutoff_absolute_margin'] == 0
    assert diagnostic['smallest_retained_singular_value'] is None
    assert diagnostic['nearest_cutoff_margin_over_sigma_max'] is None
    assert diagnostic['singular_values'] == [0., 0., 0.]
    json.dumps(diagnostic, allow_nan=False)


def test_controls_reject_overlapping_ids_and_changed_centering_values():
    cx, cy, x, y, ix, iy, ids, _, _ = planted_training_banks()
    args = dict(centering_ids_x=ix, centering_ids_y=iy, paired_ids=ids, protocol_id='guards', split_seed=3)
    with pytest.raises(ValueError, match='disjoint'):
        fit_paired_controls(cx, cy, x, y, **{**args, 'centering_ids_y': ix})
    with pytest.raises(ValueError, match='original full-bank rows'):
        fit_paired_controls(cx + 1, cy, x, y, **args)


def test_obvious_wrong_map_is_worse_than_chance_by_hand():
    x = torch.tensor([[1., 0.], [0., 1.], [-1., 0.], [0., -1.]])
    out = assert_original_agrees(MapAligner(-torch.eye(2)), x, x, 3, 3)
    assert out['metrics'] == {'foscttm': 1.0, 'mean_rank': 4.0}
    assert torch.equal(out['ranks'], torch.full((4,), 4., dtype=torch.float64))


def test_rank_recorder_rejects_nonfinite_input_and_double_similarity():
    x = torch.eye(3)
    with pytest.raises(ValueError, match='nonfinite'):
        retrieval_with_ranks(MapAligner(torch.eye(3)), x * float('nan'), x, Randomness(1))
    class DoubleScores:
        def similarity(self, queries, keys):
            return (queries @ keys.T).double()
    with pytest.raises(ValueError, match='float32 similarities'):
        retrieval_with_ranks(DoubleScores(), x, x, Randomness(1))
