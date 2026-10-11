"""Synthetic interface checks; no corpus features or real Rosetta fits are used."""
from pathlib import Path
from types import SimpleNamespace
import json
import sys

import pytest
import torch

import run_cell as runner
import unpaired_rosetta
from experiments.alignment import AlignmentRun, load_training_data, load_validation_data
from unpaired_rosetta.embeddings import reduce_texts
from unpaired_rosetta.evaluation import retrieval_metrics
from unpaired_rosetta.randomness import Randomness


@pytest.fixture(scope='module')
def api():
    return runner.load_api(Path(unpaired_rosetta.__file__).resolve().parent.parent)


def protocol_fixture():
    return {'schema_version': 1, 'protocol_id': runner.PROTOCOL_ID,
            'status': 'frozen_before_model_control_fits', 'upstream_commit': runner.UPSTREAM,
            'seeds': runner.SEEDS, 'vision_models': runner.VISION, 'text_models': runner.TEXT,
            'methods': runner.METHODS, 'models': runner.MODEL_SPECS, 'counts': runner.COUNTS,
            'algorithm_defaults': runner.ALGORITHM, 'new_rosetta_fits': 60,
            'code_sha256': {name: '0' * 64 for name in runner.CODE_NAMES},
            'locks': {name: '0' * 64 for name in runner.LOCK_NAMES},
            'execution': {'num_workers': 4, 'num_threads': 4, 'device': 'cpu',
                          'evaluation_block_size': 4096, 'torch_threads': 16, 'torch_interop_threads': 16},
            'baseline_replay': {'vision': 'dinov2', 'text': 'mpnet', 'absolute_tolerance': 1e-10,
                                'require_all_five_before_expansion': True}}


def test_protocol_rejects_drafts_smaller_problems_changed_seeds_and_missing_hashes():
    good = protocol_fixture()
    runner.check_protocol(good)
    bad_values = [('status', 'draft'), ('seeds', runner.SEEDS[:1]), ('new_rosetta_fits', 55),
                  ('counts', {**runner.COUNTS, 'validation': 5}), ('algorithm_defaults', {**runner.ALGORITHM, 'batch_size': 256}),
                  ('code_sha256', {name: None for name in runner.CODE_NAMES}),
                  ('execution', {**good['execution'], 'torch_threads': 4})]
    for key, value in bad_values:
        with pytest.raises(ValueError):
            runner.check_protocol({**good, key: value})
    with pytest.raises(ValueError, match='Machine-local'):
        runner.check_protocol({**good, 'private_path': '/home/private/features.pt'})


@pytest.fixture
def synthetic_data(tmp_path, monkeypatch, api):
    generator = torch.Generator().manual_seed(432)
    paths = {}
    for split, n in [('train2014', 17), ('val2014', 11)]:
        x = torch.randn(n, 4, generator=generator).bfloat16()
        y = torch.randn(n, 7, 6, generator=generator).bfloat16()
        y[:, 3:] = float('nan')
        for model, modality, author_name, values in [('dinov2', 'vision', 'dinov2_vit-b14@224_mean', x),
                                                     ('mpnet', 'language', 'mpnet', y)]:
            path = tmp_path / 'embeddings' / ('coco_' + split) / modality / (author_name + '.pt')
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(values, path)
            paths[f'{model}/{split}'] = path
    rows = {'train2014': {'image_ids': list(range(100, 117))},
            'val2014': {'image_ids': list(range(200, 211))}}
    monkeypatch.setenv('UNPAIRED_ROSETTA_ROOT', str(tmp_path))
    data = runner.prepare_data(paths, 'dinov2', 'mpnet', runner.SEEDS[0], rows, api)
    return data, paths, rows


def test_split_loading_rng_and_bfloat16_reduction_match_original_interfaces(synthetic_data):
    data, paths, _ = synthetic_data
    run = AlignmentRun('coco_train2014', 'dinov2_vit-b14@224_mean', 'mpnet', runner.SEEDS[0])
    rng = Randomness(run.seed)
    x, y, px, py = load_training_data(run, rng)
    assert px is py is None
    assert torch.equal(data.cx, x) and torch.equal(data.cy, y)
    assert not set(data.ix.tolist()) & set(data.iy.tolist())
    assert sorted(data.ix.tolist() + data.iy.tolist()) == list(range(17))
    assert torch.equal(data.train_rng.torch.get_state(), rng.torch.get_state())
    vx, vy, indices = load_validation_data(run, Randomness(run.seed))
    assert torch.equal(data.vx, vx) and torch.equal(data.vy, vy)
    assert torch.equal(data.validation_indices, indices)
    saved = torch.load(paths['mpnet/train2014'], weights_only=True)
    assert torch.equal(data.full_y, reduce_texts(saved).float())
    assert not torch.equal(data.full_y, reduce_texts(saved.float()))


class RectangularModel:
    def __init__(self):
        self.weight = torch.cat((torch.eye(4), torch.zeros(4, 2)), dim=1)
        self.mean_x, self.mean_y = torch.zeros(1, 4), torch.zeros(1, 6)

    def similarity(self, x, y):
        return (x @ self.weight) @ y.T


def test_full_gallery_rank_recorder_matches_original_evaluator_and_id_order(synthetic_data, api):
    data, _, rows = synthetic_data
    result = runner.evaluate_ranks(RectangularModel(), data, runner.SEEDS[0], api)
    rng = Randomness(runner.SEEDS[0])
    indices = rng.subset(11, None)
    original = retrieval_metrics(RectangularModel(), data.vx, data.vy, rng, block_size=4096)
    assert result['metrics'] == original and len(result['ranks']) == 11
    assert data.validation_ids.tolist() == [rows['val2014']['image_ids'][i] for i in indices]


def test_rosetta_receives_original_advanced_rng_defaults_and_no_pairs(synthetic_data, api):
    data, _, _ = synthetic_data
    calls = {}
    class Spy:
        def __init__(self, **kwargs):
            calls['constructor'] = kwargs

        def fit(self, x, y, px, py, randomness):
            calls['fit'] = (x, y, px, py, randomness)
            return self
    runner.fit_rosetta(data, SimpleNamespace(**{**api.__dict__, 'WassersteinProcrustes': Spy}))
    assert calls['constructor'] == {'num_workers': 4, 'num_threads': 4}
    assert calls['fit'][0] is data.cx and calls['fit'][1] is data.cy
    assert calls['fit'][2] is calls['fit'][3] is None
    assert calls['fit'][4] is data.train_rng


def test_one_failed_fit_is_retained_and_does_not_suppress_other_four_methods(tmp_path, synthetic_data, api):
    data, _, _ = synthetic_data
    class Failure:
        def __init__(self, **kwargs):
            pass

        def fit(self, *args):
            raise RuntimeError('deliberate synthetic failure')
    synthetic_api = SimpleNamespace(**{**api.__dict__, 'WassersteinProcrustes': Failure})
    attempt = runner.Attempt(tmp_path / 'attempt', tmp_path / 'private', {'fixture': 'synthetic'})
    results = runner.run_methods(attempt, data, runner.SEEDS[0], None, synthetic_api)
    assert results['rosetta']['status'] == 'failed'
    assert all(results[method]['status'] == 'complete' for method in runner.METHODS[1:])
    assert not attempt.finish({}, runner.BoundFiles())
    assert (attempt.out / 'FAILED.json').exists() and not (attempt.out / 'COMPLETE.json').exists()
    for method in runner.METHODS[1:]:
        saved = torch.load(attempt.out / method / 'ranks.pt', weights_only=True)
        assert len(saved['ranks']) == 11
        assert torch.equal(saved['validation_image_ids'], data.validation_ids)
        assert results[method]['metrics']['num_validation_samples'] == 11
    assert results['linear_paired']['linear_pinv_diagnostic'] == results['linear_shuffled']['linear_pinv_diagnostic']
    assert results['orthogonal_shuffled']['partner_permutation'] == results['linear_shuffled']['partner_permutation']
    assert (attempt.private / 'rosetta.traceback.txt').exists()
    with pytest.raises(FileExistsError):
        runner.Attempt(attempt.out, tmp_path / 'another-private', {})


def test_changed_bound_file_blocks_completion_and_baseline_mismatch_is_preserved(tmp_path):
    path = tmp_path / 'locked.json'
    runner.write_json_new(path, {'value': 1})
    bound = runner.BoundFiles()
    bound.add(path, runner.digest(path))
    path.write_text('{"value": 2}')
    with pytest.raises(ValueError, match='changed'):
        bound.recheck()
    reference = {'sha256': 'a' * 64, 'metrics': {'foscttm': 0.1, 'mean_rank': 4051.3}}
    result = runner.replay_comparison({'foscttm': 0.2, 'mean_rank': 8101.6}, reference)
    assert not result['agrees'] and not result['original_result_replaced']
    assert result['metrics']['foscttm']['signed_error'] == 0.1
    with pytest.raises(ValueError, match='All five'):
        runner.require_baseline_replays({}, 'a' * 64, 'b' * 64, {}, bound)


@pytest.mark.parametrize('fail_diagnostic', [False, True])
def test_diagnostic_failure_retains_valid_metrics_and_success_marker_requires_every_stage(tmp_path, synthetic_data, api, fail_diagnostic):
    data, _, _ = synthetic_data
    class SyntheticRosetta:
        def __init__(self, **kwargs):
            pass

        def fit(self, *args):
            return RectangularModel()
    diagnostic_calls = []
    def diagnostic(x, mean):
        diagnostic_calls.append(1)
        if fail_diagnostic:
            raise RuntimeError('deliberate diagnostic failure after valid fit and evaluation')
        return api.controls.linear_pinv_diagnostics(x, mean)
    controls = SimpleNamespace(**{**vars(api.controls), 'linear_pinv_diagnostics': diagnostic})
    synthetic_api = SimpleNamespace(**{**api.__dict__, 'WassersteinProcrustes': SyntheticRosetta, 'controls': controls})
    attempt = runner.Attempt(tmp_path / 'attempt', tmp_path / 'private', {'fixture': 'synthetic'})
    results = runner.run_methods(attempt, data, runner.SEEDS[0], None, synthetic_api)
    assert diagnostic_calls == [1]
    for method in ('linear_paired', 'linear_shuffled'):
        result = results[method]
        assert result['fit_status'] == result['evaluation_status'] == 'complete'
        assert result['metrics']['num_validation_samples'] == 11
        assert torch.isfinite(torch.tensor(result['metrics']['foscttm']))
        assert result['status'] == ('failed' if fail_diagnostic else 'complete')
        if fail_diagnostic:
            assert result['failure_stage'] == 'shared_post_fit_svd_diagnostic'
    assert attempt.finish({}, runner.BoundFiles()) is (not fail_diagnostic)
    assert (attempt.out / 'COMPLETE.json').exists() is (not fail_diagnostic)
    assert (attempt.out / 'FAILED.json').exists() is fail_diagnostic
    assert not list(attempt.out.glob('*.partial'))
    if not fail_diagnostic:
        marker = runner.read_json(attempt.out / 'COMPLETE.json')
        for path, expected in marker['files'].items():
            assert runner.digest(attempt.out / path) == expected


def test_validation_rejects_partial_caption_padding(tmp_path, api):
    path = tmp_path / 'bad.pt'
    values = torch.ones(2, 7, 3)
    values[:, 2:] = float('nan')
    values[0, 2, 0] = 0  # Missing caption slot must be entirely NaN.
    torch.save(values, path)
    with pytest.raises(ValueError, match='padding'):
        runner.validate_tensor(path, {'modality': 'language', 'dtype': 'float32'}, [2, 7, 3], [2, 2], api)


def test_baseline_replay_gate_rejects_initial_and_late_conflicting_or_partial_markers(tmp_path):
    paths, references = {}, {}
    for seed in runner.SEEDS:
        root=tmp_path/str(seed); root.mkdir(); paths[str(seed)]=str(root)
        binding={'status':'complete','protocol_sha256':'a'*64,'feature_lock_sha256':'b'*64,
                 'vision':'dinov2','text':'mpnet','seed':seed}
        metrics={'foscttm':.1,'mean_rank':4051.3,'num_validation_samples':runner.COUNTS['validation']}
        references[seed]={'sha256':'c'*64,'metrics':metrics}
        runner.write_json_new(root/'cell.json',binding)
        for method in runner.METHODS:
            folder=root/method; folder.mkdir()
            for name in ('map.pt','ranks.pt'): (folder/name).write_bytes(b'artificial bytes')
            result={**binding,'method':method,'metrics':metrics,
                    'map_sha256':runner.digest(folder/'map.pt'),'ranks_sha256':runner.digest(folder/'ranks.pt')}
            if method=='rosetta': result['baseline_replay']=runner.replay_comparison(metrics,references[seed])
            runner.write_json_new(folder/'result.json',result)
        runner.write_json_new(root/'COMPLETE.json',{**binding,'files':{str(p.relative_to(root)):runner.digest(p) for p in root.rglob('*') if p.is_file()}})
    for name in ('FAILED.json','COMPLETE.json.partial','FAILED.json.partial'):
        bound=runner.BoundFiles()
        runner.require_baseline_replays(paths,'a'*64,'b'*64,references,bound)
        marker=Path(paths[str(runner.SEEDS[0])])/name; marker.write_text('{"interrupted":')
        with pytest.raises(ValueError,match='Conflicting or partial'):
            bound.recheck()
        with pytest.raises(ValueError,match='Conflicting or partial'):
            runner.require_baseline_replays(paths,'a'*64,'b'*64,references,runner.BoundFiles())
        marker.unlink()


def reviewed_record(binding):
    return {**binding,'previous_attempt':'attempt-01','previous_artifacts':{'FAILED.json':'d'*64},
            'reviewed':True,'source_or_numerics_changed':False,'reason_kind':'external_execution_failure',
            'reason':'Artificial reviewed interruption; this test never fits a model.'}


def test_retry_review_is_copied_and_bound_before_model_preflight(tmp_path,monkeypatch):
    out=tmp_path/'attempt-02'; private=tmp_path/'private-02'
    binding={'schema_version':1,'protocol_id':runner.PROTOCOL_ID,'protocol_sha256':'a'*64,'feature_lock_sha256':'b'*64,
             'vision':'dinov2','text':'mpnet','seed':runner.SEEDS[0]}
    review=tmp_path/'approved.json'; review.write_text(json.dumps(reviewed_record(binding),indent=4)+'\n')
    digest=runner.digest(review); calls=[]
    def stop(args,bound):
        calls.append(True)
        assert (out/'retry-review.json').read_bytes()==review.read_bytes()
        assert bound.files[(out/'retry-review.json').resolve()]==digest
        raise ValueError('artificial preflight stop')
    monkeypatch.setattr(runner,'locked_inputs',stop)
    monkeypatch.setattr(sys,'argv',['run_cell.py','--protocol',str(tmp_path/'p'),'--protocol-sha256','a'*64,
        '--feature-lock',str(tmp_path/'f'),'--feature-lock-sha256','b'*64,'--bindings',str(tmp_path/'b'),
        '--out',str(out),'--private-log-dir',str(private),'--vision','dinov2','--text','mpnet','--seed',str(runner.SEEDS[0]),
        '--retry-review',str(review),'--retry-review-sha256',digest])
    with pytest.raises(ValueError,match='artificial preflight stop'): runner.main()
    assert calls and (out/'FAILED.json').is_file()
    assert (out/'retry-review.json').read_bytes()==review.read_bytes()


@pytest.mark.parametrize('mutation',['pair','digest','binding','unreviewed','changed_numerics','reason','counter','predecessor_hash'])
def test_retry_hook_rejects_unreviewed_or_wrongly_bound_records(tmp_path,mutation):
    binding={'schema_version':1,'protocol_id':runner.PROTOCOL_ID,'protocol_sha256':'a'*64,'feature_lock_sha256':'b'*64,
             'vision':'dinov2','text':'mpnet','seed':runner.SEEDS[0]}
    record=reviewed_record(binding)
    if mutation=='binding': record['protocol_sha256']='c'*64
    if mutation=='unreviewed': record['reviewed']=False
    if mutation=='changed_numerics': record['source_or_numerics_changed']=True
    if mutation=='reason': record['reason_kind']='unfavorable_score'
    if mutation=='counter': record['previous_attempt']='attempt-03'
    if mutation=='predecessor_hash': record['previous_artifacts']['FAILED.json']='bad'
    path=tmp_path/'approved.json'; runner.write_json_new(path,record)
    args=SimpleNamespace(retry_review=path,retry_review_sha256=None if mutation=='pair' else '0'*64 if mutation=='digest' else runner.digest(path))
    attempt=runner.Attempt(tmp_path/'attempt-02',tmp_path/'private',binding)
    try:
        with pytest.raises(ValueError): runner.install_retry_review(args,attempt,runner.BoundFiles())
        assert not (attempt.out/'retry-review.json').exists()
    finally: attempt.journal.close()
