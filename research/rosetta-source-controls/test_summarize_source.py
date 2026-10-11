"""Artificial orders, ranks and lifecycle records; no corpus reads or fits."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import summarize_source as s
import source_masks as masks
import source_primitives as primitives

arm, r = s.arm, s.r


@pytest.fixture
def fixture(tmp_path):
    rows = masks.SourceRows(('coco:10', 'coco:20', 'coco:25'),
        ('coco:10', 'coco:30', 'vg:1', 'vg:2', 'vg:3', 'vg:4', 'vg:5', 'vg:6'),
        ('coco:30', 'coco:40', 'coco:50', 'coco:60', 'coco:70'), (100, 103, 1, 2, 3, 4, 5, 6))
    protocol = {'selection_seeds': {'Rv': [1000, 1001], 'Rp': [2000, 2001]}}
    built = masks.build_masks(rows, validation_seeds=(1000, 1001), pairing_seeds=(2000, 2001))
    selected = dict(zip(s.mask_names(protocol), built))
    strata = {}
    for name, indices in masks.fixed_strata(rows).items():
        strata[name] = torch.zeros(5, dtype=torch.bool)
        strata[name][list(indices)] = True
    api = SimpleNamespace(torch=torch, masks=masks, primitives=primitives)
    data = SimpleNamespace(rows=rows, strata=strata, api=api, protocol=protocol)
    binding = {'schema_version': 1, 'protocol_id': arm.PROTOCOL_ID, 'protocol_sha256': 'a'*64,
               'masks_sha256': 'b'*64, 'mask': 'O', 'seed': arm.SEEDS[0]}
    return SimpleNamespace(tmp=tmp_path, data=data, masks=selected, protocol=protocol, binding=binding)


def make_attempt(f, *, counter=1, status='complete', retry_review=None):
    mask = f.masks['O']; binding = f.binding
    out = f.tmp / f'attempt-{counter:02d}'
    attempt = arm.Attempt(out, f.tmp / f'private-{counter:02d}', binding)
    if retry_review is not None:
        r.write_json_new(out / 'retry-review.json', {**binding, **retry_review})
    expected = s.expected_orders(f.data.rows, mask, binding['seed'], f.data.api)
    r.save_tensor_new(out / 'training_order.pt', expected['training'], torch)
    r.save_tensor_new(out / 'map.pt', {'mean_x': torch.zeros(1, 2), 'mean_y': torch.zeros(1, 2), 'weight': torch.eye(2)}, torch)
    higher = torch.tensor([0, 0, 2, 3, 4]); ties = torch.tensor([2, 2, 1, 2, 1])
    ranks = higher.double() + (ties.double()-1)/2 + 1
    r.save_tensor_new(out / 'ranks.pt', {**expected['evaluation'], 'ranks': ranks, 'num_higher': higher, 'tie_count': ties}, torch)
    order = expected['evaluation']['validation_order']
    overall = primitives.summarize_ranks(ranks[order], higher[order], ties[order], 5)
    metrics = {'original_overall': {k: overall[k] for k in ('foscttm', 'mean_rank')}, 'overall': overall,
        'fixed_strata': {name: primitives.summarize_ranks(ranks[which], higher[which], ties[which], 5) for name, which in f.data.strata.items()},
        'rank_order': 'canonical_original_file_row', 'tie_policy': arm.METRIC_RULES['recall_ties'],
        'overall_reduction_order': 'original_randomized_validation_order'}
    r.write_json_new(out / 'metrics.json', metrics)
    result = {'fit_status': 'complete', 'evaluation_status': 'complete', 'classification_rerun': False,
        'counts': {'train_x': 3, 'train_y': 8, 'validation': 5}, 'configured_batch_cap': 10000,
        'refinement_batch_dimensions': {'x': 3, 'y': 8}, 'mask_sha256': mask.mask_sha256,
        'composition': masks.composition(f.data.rows, mask),
        'artifact_hashes': {name: r.digest(out/name) for name in ('training_order.pt', 'map.pt', 'ranks.pt', 'metrics.json')}}
    if status == 'complete':
        attempt.finish(result, r.BoundFiles())
    elif status == 'failed':
        attempt.stage = 'final_input_recheck'; attempt.fail(RuntimeError('artificial external failure'))
    else:
        attempt.journal.close()
    return out, expected


def inspect(f, out, expected):
    return s.inspect_attempt(out, f'O/seed-{f.binding["seed"]}/{out.name}', f.binding, expected,
                             f.data, f.masks['O'], r.BoundFiles())


def refresh_hashes(path):
    result = r.read_json(path/'result.json')
    result['artifact_hashes'] = {name: r.digest(path/name) for name in result['artifact_hashes']}
    (path/'result.json').write_text(json.dumps(result)+'\n')
    marker = r.read_json(path/'COMPLETE.json')
    marker['files'] = {name: r.digest(path/name) for name in marker['files']}
    (path/'COMPLETE.json').write_text(json.dumps(marker)+'\n')


def full_observations(protocol):
    result = {}
    for mi, mask in enumerate(s.mask_names(protocol)):
        for si, seed in enumerate(arm.SEEDS):
            result[s.key(mask, seed)] = {'status': 'available', 'scores': {
                view: {metric: mi + si*.1 + vi*.01 + ki*.001 for ki, metric in enumerate(s.METRICS)}
                for vi, view in enumerate(('original_overall', 'never_known_spc_overlap', 'original_spc_overlap', 'all'))}}
    return result


def test_all_declared_masks_seeds_contrasts_and_conditional_spread(fixture):
    observations = full_observations(fixture.protocol)
    report = s.descriptive(observations, fixture.protocol, list(fixture.data.strata))
    assert len(report['per_mask_five_seed']) == 7*4
    assert len(report['within_seed_contrasts']) == 3*4
    score = report['per_mask_five_seed']['O/all']
    assert score['mean']['foscttm'] == pytest.approx(.23)
    assert score['sample_sd']['foscttm'] == pytest.approx(.025**.5)
    spread = report['within_seed_random_mask_spread']['Rv/all'][str(arm.SEEDS[0])]
    assert spread['required'] == ['Rv-1000', 'Rv-1001'] and spread['mean']['foscttm'] == pytest.approx(3.53)
    assert spread['sample_sd']['foscttm'] == pytest.approx(.5**.5)
    assert spread['minimum']['foscttm'] == pytest.approx(3.03) and spread['maximum']['foscttm'] == pytest.approx(4.03)
    effects = report['within_seed_contrasts']
    assert effects['all/C_minus_mean_Rv']['mean']['foscttm'] == pytest.approx(-2.5)
    assert effects['all/D_minus_mean_Rp']['mean']['foscttm'] == pytest.approx(-3.5)
    assert effects['all/C_minus_O']['mean']['foscttm'] == pytest.approx(1)
    assert effects['all/D_minus_C']['mean']['foscttm'] == pytest.approx(1)


def test_missing_random_mask_or_seed_never_averages_successful_subset(fixture):
    observations = full_observations(fixture.protocol)
    observations[s.key('Rv-1001', arm.SEEDS[2])] = {'status': 'unavailable', 'scores': None}
    report = s.descriptive(observations, fixture.protocol, list(fixture.data.strata))
    assert report['within_seed_random_mask_spread']['Rv/all'][str(arm.SEEDS[2])]['mean'] is None
    effect = report['within_seed_contrasts']['all/C_minus_mean_Rv']
    assert effect['mean'] is None and effect['sample_sd'] is None and len(effect['available']) == 4
    assert effect['individual'][str(arm.SEEDS[2])]['missing_masks'] == ['Rv-1001']
    assert report['within_seed_contrasts']['all/D_minus_mean_Rp']['status'] == 'available'
    # Explicit single-mask budgets have a mean but no estimable mask sample SD.
    p = copy.deepcopy(fixture.protocol); p['selection_seeds']['Rv'] = [1000]
    one = s.descriptive(observations, p, list(fixture.data.strata))['within_seed_random_mask_spread']['Rv/all'][str(arm.SEEDS[0])]
    assert one['status'] == 'available' and one['sample_sd'] is None


def test_inventory_enumerates_full_schedule_and_unknown_entries(fixture):
    root = fixture.tmp/'cells'
    target = root/'Rv-1000'/f'seed-{arm.SEEDS[0]}'/'attempt-01'; target.mkdir(parents=True)
    tree, findings = s.inventory(root, s.mask_names(fixture.protocol))
    assert len(tree) == 35 and not findings and len(tree[s.key('Rv-1000', arm.SEEDS[0])]) == 1
    (target.parent/'attempt-00').mkdir(); (root/'unlisted').mkdir()
    _, findings = s.inventory(root, s.mask_names(fixture.protocol))
    assert {entry['code'] for entry in findings} == {'unknown_attempt_entry', 'unknown_mask_entry'}


def test_exact_original_orders_ties_median_and_fixed_full_gallery_metrics(fixture):
    out, expected = make_attempt(fixture)
    record = inspect(fixture, out, expected)
    assert record['eligible'] and record['training_order_verified'] and not record['findings']
    metrics = record['recomputed_rank_metrics']['recomputed']
    assert metrics['overall']['foscttm'] == .525 and metrics['overall']['median_rank'] == 3
    assert metrics['overall']['recall_at_1'] == .2
    primary = metrics['fixed_strata']['never_known_spc_overlap']
    assert primary['num_queries'] == 4 and primary['gallery_size'] == 5 and primary['foscttm_denominator'] == 4
    assert primary['median_rank'] == 3.75 and primary['recall_at_1'] == .125
    assert s.selection([record])['status'] == 'available'
    r.no_local_paths(record)


@pytest.mark.parametrize('mutation', ['metric', 'ties', 'evaluation_order', 'training_order', 'terminal_hash'])
def test_corrupt_ranks_orders_metrics_or_artifact_hash_cannot_contribute(fixture, mutation):
    out, expected = make_attempt(fixture)
    if mutation == 'metric':
        metrics = r.read_json(out/'metrics.json'); metrics['fixed_strata']['all']['foscttm'] += .01
        (out/'metrics.json').write_text(json.dumps(metrics)+'\n'); refresh_hashes(out)
    elif mutation in ('ties', 'evaluation_order'):
        record = torch.load(out/'ranks.pt', weights_only=True)
        if mutation == 'ties': record['tie_count'][0] = 1
        else: record['query_order'] = record['query_order'].roll(1)
        torch.save(record, out/'ranks.pt'); refresh_hashes(out)
    elif mutation == 'training_order':
        record = torch.load(out/'training_order.pt', weights_only=True)
        record['indices_y'] = record['indices_y'].roll(1)
        torch.save(record, out/'training_order.pt'); refresh_hashes(out)
    else:
        (out/'map.pt').write_bytes(b'changed map')
    record = inspect(fixture, out, expected)
    assert not record['eligible'] and record['findings']
    if mutation == 'metric':
        assert record['recomputed_rank_metrics']['status'] == 'metric_mismatch'
        assert record['recomputed_rank_metrics']['recomputed']['overall']['foscttm'] == .525


@pytest.mark.parametrize('status', ['failed', 'aborted'])
def test_failed_partial_observations_are_retained_but_never_aggregated(fixture, status):
    out, expected = make_attempt(fixture, status=status)
    record = inspect(fixture, out, expected)
    assert not record['eligible'] and record['reported_metrics'] and record['recomputed_rank_metrics']['status'] == 'verified'
    assert record['input_hashes']['ranks.pt'] and s.selection([record])['scores'] is None


def test_conflicting_terminal_markers_and_multiple_completions_are_unavailable(fixture):
    out, expected = make_attempt(fixture)
    original = inspect(fixture, out, expected)
    repeated = copy.deepcopy(original); repeated['attempt'] = repeated['attempt'].replace('attempt-01', 'attempt-02')
    assert s.selection([original, repeated])['status'] == 'multiple_complete_attempts'
    (out/'FAILED.json').write_text(json.dumps({**fixture.binding, 'status': 'failed', 'files': {}})+'\n')
    both = inspect(fixture, out, expected)
    assert not both['eligible'] and {'code': 'both_terminal_markers'} in both['findings']
    assert s.selection([both, repeated])['status'] == 'multiple_complete_attempts'


def test_late_failed_marker_during_inspection_cannot_leave_eligible_complete(fixture, monkeypatch):
    out, expected = make_attempt(fixture)
    original_check = s.rank_check
    def late_failure(*args):
        result = original_check(*args)
        (out/'FAILED.json').write_text(json.dumps({**fixture.binding, 'status': 'failed', 'files': {}})+'\n')
        return result
    monkeypatch.setattr(s, 'rank_check', late_failure)
    record = inspect(fixture, out, expected)
    assert not record['eligible']
    assert {'code': 'artifact_names_changed_during_inspection'} in record['findings']


def test_stable_partial_failure_marker_invalidates_complete_and_retry_predecessor(fixture):
    out, expected = make_attempt(fixture)
    (out/'FAILED.json.partial').write_text('{"status":')
    record = inspect(fixture, out, expected)
    assert not record['eligible'] and record['terminal_partial_markers'] == ['FAILED.json.partial']
    assert record['input_hashes']['FAILED.json.partial'] == r.digest(out/'FAILED.json.partial')
    # An interrupted marker also cannot silently become a permissible predecessor.
    record['terminal_markers'] = ['FAILED.json']
    record['input_hashes']['FAILED.json'] = 'd'*64
    later = copy.deepcopy(record); later['attempt'] = later['attempt'].replace('attempt-01', 'attempt-02')
    later.update(retry_review_binding_verified=True, retry_review={'previous_attempt': 'attempt-01', 'previous_artifacts': {'FAILED.json': 'd'*64}})
    assert s.selection([record, later])['retry_history_findings']


def test_only_documented_source_locked_retry_after_failed_or_aborted_attempt(fixture):
    first, orders = make_attempt(fixture, status='failed')
    original = inspect(fixture, first, orders)
    review = {'previous_attempt': 'attempt-01', 'previous_artifacts': {'FAILED.json': r.digest(first/'FAILED.json')},
              'reviewed': True, 'source_or_numerics_changed': False, 'reason_kind': 'external_execution_failure',
              'reason': 'Artificial external interruption after metric recording.'}
    second, orders = make_attempt(fixture, counter=2, retry_review=review)
    retried = inspect(fixture, second, orders)
    assert retried['retry_review_binding_verified'] and retried['eligible']
    chosen = s.selection([retried, original])
    assert chosen['status'] == 'available' and chosen['selected_attempt'].endswith('attempt-02')
    for change in ('reason_kind', 'predecessor_hash', 'missing_review', 'counter_gap'):
        bad = copy.deepcopy(retried)
        if change == 'reason_kind': bad['retry_review_binding_verified'] = False
        if change == 'predecessor_hash': bad['retry_review']['previous_artifacts']['FAILED.json'] = 'c'*64
        if change == 'missing_review': bad['retry_review'] = None
        if change == 'counter_gap': bad['attempt'] = bad['attempt'].replace('attempt-02', 'attempt-03')
        assert s.selection([original, bad])['status'] == 'retry_history_unavailable'


def test_draft_protocol_is_rejected_before_result_inventory(tmp_path, monkeypatch):
    protocol = tmp_path/'protocol.json'; protocol.write_text(json.dumps({'status': 'draft'}))
    bindings = tmp_path/'bindings.json'
    bindings.write_text(json.dumps({'runner_bindings': str(tmp_path/'private.json'), 'attempts_root': str(tmp_path/'cells')}))
    calls = []
    monkeypatch.setattr(s, 'inventory', lambda *a: calls.append(a))
    args = SimpleNamespace(protocol=protocol, protocol_sha256=r.digest(protocol), masks=tmp_path/'masks.json',
                           masks_sha256='a'*64, bindings=bindings, out=tmp_path/'summary.json')
    with pytest.raises(ValueError, match='Unfrozen or changed protocol'):
        s.summarize(args)
    assert not calls and not args.out.exists()
