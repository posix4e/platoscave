"""Artificial ranks and attempt files only; no corpus features or fitting.

Run in the original baseline runtime so rank reductions and original Randomness
can be checked. Small counts below are test fixtures, never production options.
"""
import copy
import json
from pathlib import Path
import pickle
from types import SimpleNamespace

import pytest
import torch

import summarize_panel as s
from unpaired_rosetta.evaluation import disjoint_split
from unpaired_rosetta.randomness import Randomness

r=s.r


@pytest.fixture
def synthetic(tmp_path,monkeypatch):
    monkeypatch.setattr(r,'COUNTS',{'train':7,'fit_x':3,'fit_y':4,'validation':5})
    api=SimpleNamespace(torch=torch,Randomness=Randomness,disjoint_split=disjoint_split)
    rows={'train2014':{'image_ids':list(range(10,17))},'val2014':{'image_ids':list(range(30,35))}}
    seed=r.SEEDS[0]; expected=s.expected_orders(seed,rows,api)
    binding={'schema_version':1,'protocol_id':r.PROTOCOL_ID,'protocol_sha256':'a'*64,'feature_lock_sha256':'b'*64,
             'vision':'dinov2','text':'mpnet','seed':seed}
    return tmp_path,api,expected,binding


def make_attempt(synthetic,failed_method=None,reference=None,counter=1,retry_review=None):
    tmp,api,expected,binding=synthetic
    path=tmp/f'attempt-{counter:02d}'; attempt=r.Attempt(path,tmp/f'private-log-{counter:02d}',binding)
    if retry_review is not None: r.write_json_new(path/'retry-review.json',{**binding,**retry_review})
    r.save_tensor_new(path/'split.pt',expected['split'],torch)
    ranks=torch.tensor([1.,1.5,3.,4.,5.],dtype=torch.float64)
    metrics={'foscttm':((ranks-1)/4).mean().item(),'mean_rank':ranks.mean().item(),'num_validation_samples':5}
    for method in r.METHODS:
        def operation(folder,method=method):
            r.save_tensor_new(folder/'map.pt',{'weight':torch.eye(2)},torch)
            r.save_tensor_new(folder/'ranks.pt',{**expected['rank_order'],'ranks':ranks},torch)
            result={'status':'complete','fit_status':'complete','evaluation_status':'complete','metrics':metrics,
                    'map_sha256':r.digest(folder/'map.pt'),'ranks_sha256':r.digest(folder/'ranks.pt')}
            if method=='rosetta' and reference is not None: result['baseline_replay']=r.replay_comparison(metrics,reference)
            if method==failed_method:
                result.update(status='failed',diagnostic_status='failed',failure_stage='shared_post_fit_svd_diagnostic')
            return result
        attempt.method(method,operation)
    detail={'counts':r.COUNTS,'algorithm_defaults':r.ALGORITHM,'models':{m:r.MODEL_SPECS[m] for m in ('dinov2','mpnet')},
            'split_sha256':r.digest(path/'split.pt'),'baseline_replay_agrees':reference is None or r.replay_comparison(metrics,reference)['agrees']}
    attempt.finish(detail,r.BoundFiles())
    return path,metrics


def inspect(synthetic,path,reference=None):
    _,api,expected,binding=synthetic
    return s.inspect_attempt(path,'dinov2--mpnet/seed-'+str(binding['seed'])+'/'+path.name,binding,expected,api,r.BoundFiles(),reference)


def refresh_terminal(path):
    marker=path/('COMPLETE.json' if (path/'COMPLETE.json').exists() else 'FAILED.json')
    data=r.read_json(marker); data['files']={name:r.digest(path/name) for name in data['files']}
    marker.write_text(json.dumps(data)+'\n')


def complete_observations():
    observations={}
    for vi,v in enumerate(r.VISION):
        for ti,t in enumerate(r.TEXT):
            for si,seed in enumerate(r.SEEDS):
                for mi,method in enumerate(r.METHODS):
                    value=.1+vi*.01+ti*.02+vi*ti*.003+si*.0001+mi*.005
                    observations[s.slot_key(v,t,seed,method)]={'status':'available','metrics':{'foscttm':value,'mean_rank':1+40503*value}}
    return observations


def test_all_prespecified_contrasts_and_sample_sd():
    obs=complete_observations(); report=s.descriptive(obs)
    assert len(report['cell_scores'])==60 and len(report['contrasts'])==25+30+48
    bridge=report['contrasts']['primary/text/glove/rosetta']
    assert bridge['mean']['foscttm']==pytest.approx(.02)
    interaction=report['contrasts']['interaction/moco/elmo/rosetta']
    assert interaction['mean']['foscttm']==pytest.approx(2*2*.003)
    within=report['contrasts']['within/mae/glove/rosetta_minus_linear_paired']
    assert within['mean']['foscttm']==pytest.approx(-.015)
    scores=report['cell_scores']['dinov2/mpnet/rosetta']
    assert scores['mean']['foscttm']==pytest.approx(.1002)
    assert scores['sample_sd']['foscttm']==pytest.approx((2.5e-8)**.5)
    assert scores['ddof']==1


def test_missing_seed_never_averages_a_successful_subset():
    obs=complete_observations(); key=s.slot_key('dinov2','mpnet',r.SEEDS[2],'rosetta')
    obs[key]={'status':'unavailable','metrics':None}; report=s.descriptive(obs)
    score=report['cell_scores']['dinov2/mpnet/rosetta']
    assert score['status']=='unavailable' and score['mean'] is None and score['sample_sd'] is None
    assert len(score['available_seeds'])==4
    assert report['contrasts']['primary/text/glove/rosetta']['mean'] is None
    assert report['contrasts']['interaction/moco/elmo/rosetta']['mean'] is None
    assert report['cell_scores']['mae/elmo/linear_paired']['status']=='available'


def test_tree_inventory_retains_all_planned_keys_and_flags_unknown(tmp_path):
    p=tmp_path/'dinov2--mpnet'/f'seed-{r.SEEDS[0]}'/'attempt-01'; p.mkdir(parents=True)
    q=p.parent/'attempt-02'; q.mkdir()
    tree,findings=s.inventory(tmp_path)
    assert len(tree)==60 and not findings and len(tree[s.cell_key('dinov2','mpnet',r.SEEDS[0])])==2
    (tmp_path/'unknown').mkdir(); (p.parent/'attempt-00').mkdir()
    _,findings=s.inventory(tmp_path)
    assert {x['code'] for x in findings}=={'unknown_cell_entry','unknown_attempt_entry'}


def test_duplicate_completions_are_ambiguous_and_baseline_gate_blocks_expansion():
    fake={'attempt':'fixture/attempt-01','terminal_marker':'COMPLETE.json','terminal_manifest_verified':True,'findings':[],
          'input_hashes':{'COMPLETE.json':'c'*64},
          'methods':{m:{'eligible':True,'reported_metrics':{'foscttm':.2,'mean_rank':8101.6},
          'rank_check':{'canonical_ranks_by_coco_id_sha256':'a'*64}} for m in r.METHODS}}
    attempts={s.cell_key(*cell):[copy.deepcopy(fake)] for cell in s.cells()}
    assert s.baseline_gate(attempts,[])['status']=='satisfied'
    key=s.cell_key('mae','elmo',r.SEEDS[0]); repeated=copy.deepcopy(fake); repeated['attempt']='fixture/attempt-02'; attempts[key].append(repeated)
    obs=s.observations(attempts,[])
    assert obs[s.slot_key('mae','elmo',r.SEEDS[0],'rosetta')]['status']=='ambiguous_multiple_complete_cells'
    attempts[s.cell_key('dinov2','mpnet',r.SEEDS[0])]=[]
    obs=s.observations(attempts,[])
    assert obs[s.slot_key('dino','glove',r.SEEDS[0],'rosetta')]['status']=='baseline_replay_gate_unavailable'


def test_saved_ranks_exact_metrics_ties_and_id_order(synthetic):
    path,metrics=make_attempt(synthetic); record=inspect(synthetic,path)
    assert not record['findings'] and all(x['eligible'] for x in record['methods'].values())
    checked=record['methods']['rosetta']['rank_check']
    assert checked['recomputed_metrics']=={k:metrics[k] for k in s.METRICS}
    assert checked['midrank_step']==.5
    r.no_local_paths(record)


def test_finalized_failed_cell_retains_unaffected_methods_and_diagnostic_metrics(synthetic):
    path,_=make_attempt(synthetic,failed_method='linear_paired'); record=inspect(synthetic,path)
    assert record['terminal_marker']=='FAILED.json' and record['terminal_manifest_verified']
    assert record['methods']['rosetta']['eligible']
    failed=record['methods']['linear_paired']
    assert not failed['eligible'] and failed['rank_check']['status']=='verified' and failed['reported_metrics']


def test_aborted_marker_does_not_promote_unfinished_results(synthetic):
    path,_=make_attempt(synthetic); (path/'COMPLETE.json').unlink()
    binding=synthetic[3]; (path/'FAILED.json').write_text(json.dumps({**binding,'status':'failed'})+'\n')
    record=inspect(synthetic,path)
    assert not record['terminal_manifest_verified'] and not any(x['eligible'] for x in record['methods'].values())
    assert record['methods']['rosetta']['reported_metrics']


@pytest.mark.parametrize('change',['metric','id_order','terminal_hash'])
def test_rank_identity_metric_and_terminal_hash_mismatches(synthetic,change):
    path,_=make_attempt(synthetic); result_path=path/'rosetta'/'result.json'; result=r.read_json(result_path)
    if change=='metric':
        result['metrics']['foscttm']+=.001; result_path.write_text(json.dumps(result)+'\n'); refresh_terminal(path)
    elif change=='id_order':
        ranks_path=path/'rosetta'/'ranks.pt'; ranks=torch.load(ranks_path,weights_only=True)
        ranks['validation_image_ids']=ranks['validation_image_ids'].roll(1); torch.save(ranks,ranks_path)
        result['ranks_sha256']=r.digest(ranks_path); result_path.write_text(json.dumps(result)+'\n'); refresh_terminal(path)
    else: result_path.write_text(json.dumps(result)+' \n')
    record=inspect(synthetic,path)
    assert not record['methods']['rosetta']['eligible']
    assert record['findings'] or record['methods']['rosetta']['findings']
    if change=='metric': assert record['methods']['rosetta']['rank_check']['status']=='metric_mismatch'


def test_numerically_complete_baseline_replay_mismatch_remains_a_gate_failure(synthetic):
    reference={'sha256':'c'*64,'metrics':{'foscttm':.1,'mean_rank':1.4}}
    path,_=make_attempt(synthetic,reference=reference); record=inspect(synthetic,path,reference)
    assert record['terminal_marker']=='FAILED.json' and not record['methods']['rosetta']['eligible']
    assert record['methods']['rosetta']['rank_check']['status']=='verified'
    assert not record['methods']['rosetta']['baseline_replay']['agrees']


def test_late_failed_marker_during_inspection_cannot_leave_eligible_complete(synthetic,monkeypatch):
    path,_=make_attempt(synthetic)
    original_check=s.rank_check
    def late_failure(*args):
        result=original_check(*args)
        if not (path/'FAILED.json').exists():
            (path/'FAILED.json').write_text(json.dumps({**synthetic[3],'status':'failed'})+'\n')
        return result
    monkeypatch.setattr(s,'rank_check',late_failure)
    record=inspect(synthetic,path)
    assert not any(entry['eligible'] for entry in record['methods'].values())
    assert {'code':'artifact_names_changed_during_inspection'} in record['findings']


def test_stable_partial_failure_marker_invalidates_complete_and_retry_predecessor(synthetic):
    path,_=make_attempt(synthetic)
    (path/'FAILED.json.partial').write_text('{"status":')
    record=inspect(synthetic,path)
    assert not any(entry['eligible'] for entry in record['methods'].values())
    assert record['terminal_partial_markers']==['FAILED.json.partial']
    assert record['input_hashes']['FAILED.json.partial']==r.digest(path/'FAILED.json.partial')
    record['terminal_marker']='FAILED.json'; record['input_hashes']['FAILED.json']='d'*64
    later=copy.deepcopy(record); later['attempt']=later['attempt'].replace('attempt-01','attempt-02')
    later.update(retry_review_binding_verified=True,retry_review={'previous_attempt':'attempt-01','previous_artifacts':{'FAILED.json':'d'*64}})
    assert s.cell_selection([record,later])['retry_history_findings']


def test_documented_retry_uses_earliest_matching_methods_and_retains_reason(synthetic):
    first,_=make_attempt(synthetic,failed_method='linear_paired')
    original=inspect(synthetic,first)
    review={'previous_attempt':'attempt-01','reason_kind':'external_execution_failure',
            'reason':'Artificial documented resource interruption during diagnostic.', 'reviewed':True,
            'source_or_numerics_changed':False,'previous_artifacts':{'FAILED.json':r.digest(first/'FAILED.json')}}
    second,_=make_attempt(synthetic,counter=2,retry_review=review)
    retried=inspect(synthetic,second)
    selected=s.cell_selection([retried,original])
    assert not selected['retry_history_findings']
    assert selected['methods']['rosetta']['selected_attempt'].endswith('attempt-01')
    assert selected['methods']['linear_paired']['selected_attempt'].endswith('attempt-02')
    assert selected['methods']['rosetta']['repeat_agreement'] is True
    assert retried['retry_review']['reason']==review['reason']
    assert retried['retry_review_binding_verified']
    # Matching aggregate metrics cannot hide changed individual query ranks.
    changed=copy.deepcopy(retried)
    changed['methods']['rosetta']['rank_check']['canonical_ranks_by_coco_id_sha256']='d'*64
    assert s.cell_selection([original,changed])['methods']['rosetta']['status']=='conflicting_repeated_results'
    undocumented=copy.deepcopy(retried); undocumented['retry_review']=None
    assert s.cell_selection([original,undocumented])['methods']['rosetta']['status']=='retry_history_unavailable'


@pytest.mark.parametrize('artifact',['split.pt','rosetta/ranks.pt'])
@pytest.mark.parametrize('payload',[b'',b'not a torch archive'])
def test_interrupted_tensor_is_retained_and_does_not_hide_reviewed_retry(synthetic,artifact,payload):
    first,metrics=make_attempt(synthetic)
    (first/'COMPLETE.json').unlink()  # Artificial interruption: no final manifest.
    tensor_path=first/artifact
    tensor_path.write_bytes(payload)
    if artifact.endswith('ranks.pt'):
        result_path=first/'rosetta'/'result.json'
        result=r.read_json(result_path); result['ranks_sha256']=r.digest(tensor_path)
        result_path.write_text(json.dumps(result)+'\n')
    with pytest.raises((EOFError,pickle.UnpicklingError)):
        torch.load(tensor_path,weights_only=True,map_location='cpu')
    original=inspect(synthetic,first)
    assert not any(entry['eligible'] for entry in original['methods'].values())
    assert original['methods']['rosetta']['reported_metrics']==metrics
    if artifact=='split.pt':
        assert {'code':'split_verification_failed'} in original['findings']
    else:
        assert 'rank_map_or_baseline_verification_failed' in original['methods']['rosetta']['findings']
    review={'previous_attempt':'attempt-01','reason_kind':'external_execution_failure',
            'reason':'Artificial interrupted tensor write; original corrupt bytes retained.',
            'reviewed':True,'source_or_numerics_changed':False,
            'previous_artifacts':{'started.json':r.digest(first/'started.json')}}
    second,_=make_attempt(synthetic,counter=2,retry_review=review)
    selected=s.cell_selection([original,inspect(synthetic,second)])
    assert not selected['retry_history_findings']
    assert all(entry['selected_attempt'].endswith('attempt-02') for entry in selected['methods'].values())
    assert tensor_path.read_bytes()==payload
