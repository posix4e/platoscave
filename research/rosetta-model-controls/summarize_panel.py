#!/usr/bin/env python3
"""Describe the fixed panel, retaining every attempt and all unavailable entries.

Run in the original locked Pixi runtime. Private --bindings is path-only JSON:
{runner_bindings: <run_cell bindings JSON>, attempts_root: <cells directory>}.
The exact tree is cells/{vision}--{text}/seed-{seed}/attempt-{01..99}. All planned
60 cell/seed keys and 300 method slots are reported, even before attempts exist.
Unknown cell, seed, or attempt entries are findings and disable aggregation.

A terminal FAILED cell with a complete verified file manifest may contribute an
unaffected complete method. Aborted cells and diagnostic-failed methods retain
observed metrics but cannot contribute complete observations. A documented retry
may reuse the earliest eligible method only when all later eligible repetitions
have identical canonical ranks and metrics. Multiple fully COMPLETE cells remain
ambiguous. Every retry/reason is retained, and all five seeds are
required for every mean/sample-SD and contrast. There is no ranking, threshold,
significance test, query-level interval, refitting or rescoring of similarities.

Ranks are independently checked against scalar metrics and original RNG/ID order.
This verifies saved-rank consistency, not a second scoring of the fitted maps.
The output contains shared validation order/ID arrays and bound rank-artifact
hashes, so the per-query ranks retain their original mapping and can be sorted by
COCO ID without changing the metric reduction. Raw paths/tracebacks stay private.
A future outer release must bind this helper's hash before results are inspected.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import math
import os
from pathlib import Path
import pickle
import re
import statistics
from types import SimpleNamespace

import run_cell as r

ROOT = Path(__file__).resolve().parent
r.require(Path(r.__file__).resolve() == ROOT / 'run_cell.py', 'Use runner from this release')
METRICS = ('foscttm', 'mean_rank')
WITHIN = [('orthogonal_pairing', 'orthogonal_paired', 'orthogonal_shuffled'),
          ('linear_pairing', 'linear_paired', 'linear_shuffled'),
          ('rosetta_minus_orthogonal_paired', 'rosetta', 'orthogonal_paired'),
          ('rosetta_minus_linear_paired', 'rosetta', 'linear_paired')]


def cells():
    return [(v, t, seed) for v in r.VISION for t in r.TEXT for seed in r.SEEDS]


def cell_key(v, t, seed):
    return f'{v}/{t}/{seed}'


def slot_key(v, t, seed, method):
    return f'{cell_key(v,t,seed)}/{method}'


def inventory(root):
    """Inspect the declared experiment tree; never accept a curated attempt list."""
    root = Path(root)
    result = {cell_key(*cell): [] for cell in cells()}
    findings = []
    if not root.exists():
        return result, [{'code': 'attempts_root_missing'}]
    r.require(root.is_dir(), 'Attempts root must be a directory')
    known = {f'{v}--{t}': (v,t) for v in r.VISION for t in r.TEXT}
    for folder in sorted(root.iterdir()):
        if not folder.is_dir() or folder.name not in known:
            findings.append({'code': 'unknown_cell_entry', 'relative_path': folder.name}); continue
        v,t = known[folder.name]
        for seed_dir in sorted(folder.iterdir()):
            expected = {f'seed-{seed}': seed for seed in r.SEEDS}
            if not seed_dir.is_dir() or seed_dir.name not in expected:
                findings.append({'code': 'unknown_seed_entry', 'relative_path': str(seed_dir.relative_to(root))}); continue
            seed = expected[seed_dir.name]
            for attempt in sorted(seed_dir.iterdir()):
                if not attempt.is_dir() or not re.fullmatch(r'attempt-(0[1-9]|[1-9][0-9])', attempt.name):
                    findings.append({'code': 'unknown_attempt_entry', 'relative_path': str(attempt.relative_to(root))}); continue
                r.require(attempt.resolve().is_relative_to(root.resolve()), 'Attempt escapes experiment tree')
                result[cell_key(v,t,seed)].append(attempt)
    return result, findings


def artifact_names(path):
    return sorted(str(item.relative_to(path)) for item in path.rglob('*') if item.is_file())


def expected_orders(seed, rows, api):
    torch = api.torch
    ix,iy = api.disjoint_split(len(rows['train2014']['image_ids']), api.Randomness(seed), None)
    rng = api.Randomness(seed)
    indices = rng.subset(len(rows['val2014']['image_ids']), None)
    query,keys = rng.permutation(len(indices)), rng.permutation(len(indices))
    train_ids = torch.tensor(rows['train2014']['image_ids'], dtype=torch.int64)
    validation_ids = torch.tensor(rows['val2014']['image_ids'], dtype=torch.int64)[indices]
    return {'split': {'training_image_ids':train_ids, 'indices_x':ix, 'indices_y':iy,
                      'validation_indices':indices, 'validation_image_ids':validation_ids},
            'rank_order': {'validation_image_ids':validation_ids, 'validation_indices':indices,
                           'query_order':query, 'key_order':keys, 'partner_in_shuffled_keys':torch.argsort(keys)[query]}}


def check_tensors(actual, expected, torch):
    r.require(set(actual) == set(expected), 'Tensor record fields differ')
    for key,value in expected.items():
        candidate = actual[key]
        r.require(isinstance(candidate,torch.Tensor) and candidate.device.type=='cpu' and candidate.dtype==value.dtype and
                  candidate.shape==value.shape and torch.equal(candidate,value), 'Tensor ID/order differs')


def rank_check(path, metrics, expected, api):
    torch = api.torch
    record = torch.load(path, weights_only=True, map_location='cpu')
    r.require(isinstance(record,dict) and set(record)==set(expected)|{'ranks'}, 'Rank record fields differ')
    check_tensors({k:v for k,v in record.items() if k!='ranks'}, expected, torch)
    ranks = record['ranks']; n = len(expected['validation_image_ids'])
    r.require(isinstance(ranks,torch.Tensor) and ranks.device.type=='cpu' and ranks.dtype==torch.float64 and
              ranks.shape==(n,) and bool(torch.isfinite(ranks).all()) and bool(((ranks>=1)&(ranks<=n)).all()) and
              bool(torch.equal(ranks*2,torch.round(ranks*2))), 'Invalid full-gallery midranks')
    recomputed = {'foscttm':((ranks-1)/(n-1)).mean().item(), 'mean_rank':ranks.mean().item()}
    gallery_matches=type(metrics.get('num_validation_samples')) is int and metrics['num_validation_samples']==n
    comparisons={}
    for key,value in recomputed.items():
        reported=metrics.get(key)
        finite=type(reported) in (int,float) and math.isfinite(reported)
        comparisons[key]={'reported':reported if finite else None,'recomputed':value,
                          'signed_error':reported-value if finite else None,'exactly_equal':finite and reported==value}
    matches=gallery_matches and all(x['exactly_equal'] for x in comparisons.values())
    return {'status':'verified' if matches else 'metric_mismatch', 'recomputed_metrics':recomputed,
            'comparisons':comparisons,'gallery_count_matches':gallery_matches,'metric_equality':'exact in locked runtime',
            'mean_rank_relation_residual':recomputed['mean_rank']-(1+(n-1)*recomputed['foscttm']),
            'validation_count':n, 'ranks_dtype':'float64', 'midrank_step':0.5,
            'canonical_ranks_by_coco_id_sha256':r.json_hash(ranks[torch.argsort(expected['validation_image_ids'])].tolist()),
            'order_ids_checked_against_original_rng':True}


def public_scalars(record):
    """Retain reported scalar metrics even when a later validation rejects them."""
    return {str(k):v for k,v in record.items() if type(v) in (int,float) and math.isfinite(v)} if isinstance(record,dict) else {}


def inspect_attempt(path, relative, binding, expected, api, bound, reference):
    issues = []; hashes = {}; parsed = {}
    initial_files=artifact_names(path)
    def read(name):
        target=path/name
        if not target.is_file(): return None
        hashes[name]=r.digest(target); bound.add(target,hashes[name])
        try:
            value=r.read_json(target); r.require(isinstance(value,dict),'Expected JSON object'); return value
        except (ValueError,TypeError): issues.append({'code':'invalid_json','file':name}); return None
    def identity(value):
        return isinstance(value,dict) and all(value.get(k)==v for k,v in binding.items())
    start=read('started.json'); complete=read('COMPLETE.json'); failed=read('FAILED.json'); cell=read('cell.json'); retry=read('retry-review.json')
    partials=[name for name in ('COMPLETE.json.partial','FAILED.json.partial') if (path/name).exists()]
    for name in partials:
        if (path/name).is_file():
            hashes[name]=r.digest(path/name); bound.add(path/name,hashes[name])
    if partials: issues.append({'code':'partial_terminal_marker','files':partials})
    if not identity(start): issues.append({'code':'missing_or_wrong_start_binding'})
    terminal = complete if complete is not None else failed
    terminal_name = 'COMPLETE.json' if complete is not None else 'FAILED.json' if failed is not None else None
    finalized = terminal is not None and not (complete is not None and failed is not None)
    if complete is not None and failed is not None: issues.append({'code':'both_terminal_markers'})
    manifest = terminal.get('files') if isinstance(terminal,dict) else None
    finalized = finalized and identity(terminal) and isinstance(manifest,dict)
    if finalized:
        for name,expected_hash in manifest.items():
            try:
                candidate=(path/name).resolve(strict=True)
                r.require(not Path(name).is_absolute() and candidate.is_relative_to(path.resolve()), 'Manifest escape')
                bound.add(candidate,expected_hash); hashes[name]=expected_hash
            except (OSError,ValueError,TypeError):
                issues.append({'code':'terminal_file_hash_failed','file':name}); finalized=False
        required={'started.json','journal.jsonl','cell.json','split.pt'} | {f'{m}/result.json' for m in r.METHODS}
        if not required <= set(manifest): issues.append({'code':'terminal_manifest_incomplete'}); finalized=False
        if not identity(cell) or not isinstance(cell.get('methods'),dict) or set(cell['methods'])!=set(r.METHODS):
            issues.append({'code':'missing_or_wrong_cell_binding'}); finalized=False
        elif cell.get('counts')!=r.COUNTS or cell.get('algorithm_defaults')!=r.ALGORITHM:
            issues.append({'code':'cell_contract_differs'}); finalized=False
        if terminal.get('status') != ('complete' if terminal_name=='COMPLETE.json' else 'failed') or (isinstance(cell,dict) and cell.get('status')!=terminal.get('status')):
            issues.append({'code':'terminal_status_differs'}); finalized=False
    if not finalized and not issues: issues.append({'code':'no_final_file_manifest'})
    split_ok=False
    if (path/'split.pt').is_file():
        hashes['split.pt']=r.digest(path/'split.pt'); bound.add(path/'split.pt',hashes['split.pt'])
        try:
            check_tensors(api.torch.load(path/'split.pt',weights_only=True,map_location='cpu'),expected['split'],api.torch)
            r.require(not finalized or cell.get('split_sha256')==hashes['split.pt'], 'Split hash differs')
            split_ok=True
        except (OSError,ValueError,TypeError,RuntimeError,KeyError,EOFError,pickle.UnpicklingError): issues.append({'code':'split_verification_failed'})
    outcomes={}
    for method in r.METHODS:
        name=f'{method}/result.json'; result=read(name)
        entry={'status':'missing' if result is None else result.get('status','invalid'), 'eligible':False,
               'reported_metrics':public_scalars(result.get('metrics')) if isinstance(result,dict) else {}, 'findings':[]}
        outcomes[method]=entry
        if result is None: continue
        if not identity(result) or result.get('method')!=method:
            entry['findings'].append('result_binding_differs'); continue
        for key in ('fit_status','evaluation_status','diagnostic_status','failure_stage','error_type','fit_seconds',
                    'fit_and_evaluation_seconds','paired_training_items','additional_x_counterparts','additional_y_counterparts',
                    'partner_permutation','linear_pinv_diagnostic','baseline_replay'):
            if key in result:
                try: r.no_local_paths(result[key]); entry[key]=result[key]
                except ValueError: entry['findings'].append('unsafe_result_field_'+key)
        verified=False
        if entry['reported_metrics'] and (path/method/'ranks.pt').is_file():
            try:
                for artifact in ('map.pt','ranks.pt'):
                    target=path/method/artifact; expected_hash=result[artifact.replace('.pt','_sha256')]
                    bound.add(target,expected_hash); hashes[f'{method}/{artifact}']=expected_hash
                    if finalized: r.require(manifest.get(f'{method}/{artifact}')==expected_hash,'Artifact not bound by terminal marker')
                entry['rank_check']=rank_check(path/method/'ranks.pt',result['metrics'],expected['rank_order'],api)
                entry['ranks_sha256']=result['ranks_sha256']
                r.require(entry['rank_check']['status']=='verified','Saved-rank metrics disagree')
                verified=True
                if method=='rosetta' and reference is not None:
                    comparison=r.replay_comparison(result['metrics'],reference)
                    r.require(result.get('baseline_replay')==comparison and comparison['agrees'],'Baseline replay mismatch')
            except (OSError,ValueError,TypeError,RuntimeError,KeyError,EOFError,pickle.UnpicklingError):
                entry['findings'].append('rank_map_or_baseline_verification_failed'); verified=False
        if finalized and cell['methods'].get(method)!=result.get('status'):
            entry['findings'].append('cell_method_status_differs'); verified=False
        entry['eligible']=bool(finalized and split_ok and not issues and not entry['findings'] and verified and result.get('status')=='complete')
    if artifact_names(path)!=initial_files:
        issues.append({'code':'artifact_names_changed_during_inspection'})
        for entry in outcomes.values(): entry['eligible']=False
    if complete is not None and not all(entry['eligible'] for entry in outcomes.values()):
        issues.append({'code':'complete_cell_has_unavailable_method'})
        for entry in outcomes.values(): entry['eligible']=False
    retry_verified=bool(isinstance(retry,dict) and identity(retry) and retry.get('reviewed') is True and
       retry.get('source_or_numerics_changed') is False and retry.get('reason_kind') in ('external_execution_failure','input_failure') and
       isinstance(retry.get('reason'),str) and retry['reason'].strip() and
       (not finalized or manifest.get('retry-review.json')==hashes.get('retry-review.json')))
    if retry is not None:
        try: r.no_local_paths(retry)
        except ValueError: retry=None; retry_verified=False
    return {'attempt':relative,'retry_review':retry,'retry_review_binding_verified':retry_verified,'terminal_marker':terminal_name,'terminal_partial_markers':partials,'terminal_manifest_verified':bool(finalized),
            'cell_status':terminal.get('status') if isinstance(terminal,dict) else 'incomplete',
            'split_verified':split_ok,'input_hashes':hashes,'artifact_names':initial_files,'findings':issues,'methods':outcomes}


def cell_selection(attempts):
    """Earliest-by-counter, with documented retry chains and exact repeat checks."""
    attempts=sorted(attempts,key=lambda a:Path(a['attempt']).name)
    complete=sum(a['terminal_marker']=='COMPLETE.json' and a['terminal_manifest_verified'] and not a['findings'] and
                 all(m['eligible'] for m in a['methods'].values()) for a in attempts)
    retry_issues=[]
    if attempts and Path(attempts[0]['attempt']).name!='attempt-01': retry_issues.append('missing_first_attempt')
    for index,current in enumerate(attempts[1:],1):
        previous=attempts[index-1]; review=current.get('retry_review'); previous_name=Path(previous['attempt']).name
        terminal='FAILED.json' if previous['terminal_marker']=='FAILED.json' else 'started.json'
        expected_hash=previous['input_hashes'].get(terminal)
        valid=(Path(current['attempt']).name==f'attempt-{index+1:02d}' and previous['terminal_marker']!='COMPLETE.json' and
               not previous.get('terminal_partial_markers') and
               current.get('retry_review_binding_verified') and isinstance(review,dict) and
               review.get('previous_attempt')==previous_name and expected_hash is not None and
               review.get('previous_artifacts',{}).get(terminal)==expected_hash)
        if not valid: retry_issues.append({'attempt':current['attempt'],'code':'undocumented_or_invalid_retry_chain'})
    selections={}
    for method in r.METHODS:
        eligible=[a for a in attempts if a['methods'][method]['eligible']]
        status='available' if eligible else 'unavailable'
        if complete>1: status='ambiguous_multiple_complete_cells'
        elif retry_issues: status='retry_history_unavailable'
        repeats_agree=True
        if len(eligible)>1:
            first=eligible[0]['methods'][method]
            for attempt in eligible[1:]:
                other=attempt['methods'][method]
                if (first['rank_check']['canonical_ranks_by_coco_id_sha256']!=other['rank_check']['canonical_ranks_by_coco_id_sha256'] or
                    any(first['reported_metrics'][metric]!=other['reported_metrics'][metric] for metric in METRICS)):
                    repeats_agree=False
            if not repeats_agree and status=='available': status='conflicting_repeated_results'
        chosen=eligible[0] if status=='available' else None
        selections[method]={'status':status,'eligible_completions':len(eligible),'attempt_count':len(attempts),
            'selected_attempt':chosen['attempt'] if chosen else None,
            'selection_rule':'earliest eligible attempt counter; every later eligible repetition must agree',
            'repeat_agreement':repeats_agree if len(eligible)>1 else None,
            'metrics':{metric:chosen['methods'][method]['reported_metrics'][metric] for metric in METRICS} if chosen else None}
    return {'fully_complete_cell_attempts':complete,'retry_history_findings':retry_issues,'methods':selections}


def baseline_gate(attempts, findings):
    selected={str(seed):cell_selection(attempts[cell_key('dinov2','mpnet',seed)]) for seed in r.SEEDS}
    valid=not findings and all(v['fully_complete_cell_attempts']==1 and not v['retry_history_findings'] and
                             all(m['status']=='available' for m in v['methods'].values()) for v in selected.values())
    return {'status':'satisfied' if valid else 'unavailable',
            'eligible_reference_cell_completions_by_seed':{s:v['fully_complete_cell_attempts'] for s,v in selected.items()},
            'rule':'Exactly one verified COMPLETE reference cell per seed; documented retries agree in canonical ranks and metrics; raw baseline metrics agree within 1e-10.'}


def observations(attempts, inventory_findings):
    output={}; gate=baseline_gate(attempts,inventory_findings)
    for v,t,seed in cells():
        selected=cell_selection(attempts[cell_key(v,t,seed)])
        for method,entry in selected['methods'].items():
            entry={**entry,'retry_history_findings':selected['retry_history_findings']}
            if inventory_findings: entry.update(status='invalid_inventory',metrics=None,selected_attempt=None)
            elif entry['status']=='available' and (v,t)!=('dinov2','mpnet') and gate['status']!='satisfied':
                entry.update(status='baseline_replay_gate_unavailable',metrics=None,selected_attempt=None)
            output[slot_key(v,t,seed,method)]=entry
    return output


def five_seed(values):
    available=[seed for seed in r.SEEDS if values[str(seed)]['status']=='available']
    complete=len(available)==len(r.SEEDS)==5
    return {'status':'available' if complete else 'unavailable','required_seeds':r.SEEDS,'available_seeds':available,
            'per_seed':values,'mean':{m:statistics.mean(values[str(s)]['metrics'][m] for s in r.SEEDS) for m in METRICS} if complete else None,
            'sample_sd':{m:statistics.stdev(values[str(s)]['metrics'][m] for s in r.SEEDS) for m in METRICS} if complete else None,'ddof':1}


def descriptive(obs):
    scores={f'{v}/{t}/{method}':five_seed({str(s):obs[slot_key(v,t,s,method)] for s in r.SEEDS})
            for v in r.VISION for t in r.TEXT for method in r.METHODS}
    contrasts={}
    def contrast(name,terms):
        values={}
        for seed in r.SEEDS:
            operands=[(coefficient,slot_key(v,t,seed,method)) for coefficient,v,t,method in terms]
            missing=[key for coefficient,key in operands if obs[key]['status']!='available']
            values[str(seed)]={'status':'unavailable' if missing else 'available','missing_entries':missing,
               'metrics':None if missing else {metric:math.fsum(coefficient*obs[key]['metrics'][metric] for coefficient,key in operands) for metric in METRICS}}
        contrasts[name]={'terms':[{'coefficient':a,'vision':v,'text':t,'method':m} for a,v,t,m in terms],**five_seed(values)}
    for method in r.METHODS:
        for t in r.TEXT[1:]:
            contrast(f'primary/text/{t}/{method}',[(1,'dinov2',t,method),(-1,'dinov2','mpnet',method)])
        for v in r.VISION[1:]:
            contrast(f'primary/vision/{v}/{method}',[(1,v,'mpnet',method),(-1,'dinov2','mpnet',method)])
            for t in r.TEXT[1:]:
                contrast(f'interaction/{v}/{t}/{method}',[(1,v,t,method),(-1,v,'mpnet',method),(-1,'dinov2',t,method),(1,'dinov2','mpnet',method)])
    for v in r.VISION:
        for t in r.TEXT:
            for label,left,right in WITHIN:
                contrast(f'within/{v}/{t}/{label}',[(1,v,t,left),(-1,v,t,right)])
    return {'metrics':list(METRICS),'direction':'Positive differences mean worse retrieval; mean_rank is algebraically dependent on FOSCTTM.',
            'cell_scores':scores,'contrasts':contrasts,'inference':'Descriptive only; no significance tests, rankings, thresholds or confidence intervals.'}


def summarize(args):
    r.require(not args.out.exists() and not args.out.is_symlink(),'Output exists; use a new summary path')
    binding=r.read_json(args.bindings)
    r.require(set(binding)=={'runner_bindings','attempts_root'} and all(isinstance(v,str) and Path(v).is_absolute() for v in binding.values()),'Use private absolute path bindings')
    bound=r.BoundFiles(); bound.add(__file__,r.digest(__file__)); bound.add(args.bindings,r.digest(args.bindings))
    protocol,api,rows,paths,references=r.locked_inputs(SimpleNamespace(protocol=args.protocol,protocol_sha256=args.protocol_sha256,
       feature_lock=args.feature_lock,feature_lock_sha256=args.feature_lock_sha256,bindings=Path(binding['runner_bindings']),vision='dinov2',text='mpnet'),bound)
    bound.add(binding['runner_bindings'],r.digest(binding['runner_bindings']))
    tree,findings=inventory(binding['attempts_root']); attempts={}; orders={s:expected_orders(s,rows,api) for s in r.SEEDS}
    for v,t,seed in cells():
        expected_binding={'schema_version':1,'protocol_id':r.PROTOCOL_ID,'protocol_sha256':args.protocol_sha256,
                          'feature_lock_sha256':args.feature_lock_sha256,'vision':v,'text':t,'seed':seed}
        attempts[cell_key(v,t,seed)]=[inspect_attempt(path,str(path.relative_to(binding['attempts_root'])),expected_binding,
             orders[seed],api,bound,references[seed] if (v,t)==('dinov2','mpnet') else None) for path in tree[cell_key(v,t,seed)]]
    obs=observations(attempts,findings)
    mappings={str(seed):{**{name:tensor.tolist() for name,tensor in orders[seed]['rank_order'].items()},
       'sort_positions_by_image_id':api.torch.argsort(orders[seed]['rank_order']['validation_image_ids']).tolist(),
       'training_split_hashes':{name:r.json_hash(tensor.tolist()) for name,tensor in orders[seed]['split'].items() if name.startswith('indices_')}} for seed in r.SEEDS}
    result={'schema_version':1,'protocol_id':r.PROTOCOL_ID,'recorded_at_utc':datetime.now(timezone.utc).isoformat(),
       'status':'complete' if all(v['status']=='available' for v in obs.values()) else 'incomplete',
       'input_hashes':{'protocol':args.protocol_sha256,'feature_lock':args.feature_lock_sha256,'summarizer':r.digest(__file__),
                       'private_bindings':r.digest(args.bindings),'runner_bindings':r.digest(binding['runner_bindings']),'prespecified_reporting':r.json_hash(protocol['research_design']['reporting'])},
       'planned':{'cells':12,'seeds':5,'method_slots':300},'inventory_findings':findings,'baseline_replay_gate':baseline_gate(attempts,findings),'attempts':attempts,'observations':obs,
       'validation_order_and_coco_id_mapping':mappings,'descriptive':descriptive(obs),
       'limitations':['Rank checks recompute saved-rank metrics and original RNG/ID order; they do not rescore fitted maps.',
          'All attempt histories and retry reviews are retained. Earliest eligible methods are used only for documented execution/input retries whose repetitions agree; multiple fully COMPLETE cells or conflicting repeated results remain unavailable.',
          'Seeds share fixed data and pretrained encoders; these summaries establish no pretraining-cleanliness claim.']}
    bound.recheck()
    # Check both directory inventory and late artifacts after the byte rechecks.
    new_tree,new_findings=inventory(binding['attempts_root'])
    if new_tree!=tree or new_findings!=findings: findings.append({'code':'inventory_changed_during_summary'})
    for name,paths in tree.items():
        for path,attempt in zip(paths,attempts[name]):
            if artifact_names(path)!=attempt['artifact_names']:
                findings.append({'code':'attempt_artifacts_changed_during_summary','attempt':attempt['attempt']})
    if findings:
        result['status']='incomplete'
        result['observations']=observations(attempts,findings)
        result['baseline_replay_gate']=baseline_gate(attempts,findings)
        result['descriptive']=descriptive(result['observations'])
    r.no_local_paths(result)
    r.publish_marker(args.out,result)
    fd=os.open(args.out.parent,os.O_RDONLY)
    try: os.fsync(fd)
    finally: os.close(fd)
    return {'status':result['status'],'summary_sha256':r.digest(args.out),'planned_method_slots':300,
            'available_method_slots':sum(v['status']=='available' for v in result['observations'].values()),'real_fitting_performed':False}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('protocol','feature-lock','bindings','out'): parser.add_argument('--'+name,type=Path,required=True)
    for name in ('protocol-sha256','feature-lock-sha256'): parser.add_argument('--'+name,required=True)
    args=parser.parse_args()
    import json
    print(json.dumps(summarize(args),indent=2,allow_nan=False))


if __name__=='__main__': main()
