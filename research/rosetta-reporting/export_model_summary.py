#!/usr/bin/env python3
"""Prepare a compact model-panel projection; no metric arithmetic or publication."""
from __future__ import annotations

import argparse
from datetime import datetime
import json
import math
import os
from pathlib import Path
import re
import sys

# Reviewed metadata validators only. Never call this module's tensor loader/export.
import export_rank_tables as ranks

need = ranks.need
digest = ranks.digest
object_hash = ranks.object_hash
METRICS = ('foscttm', 'mean_rank')
STATUS = {'complete', 'failed', 'missing', 'invalid', 'not_attempted', 'incomplete'}
SELECTION_RULE = 'earliest eligible attempt counter; every later eligible repetition must agree'
REPORTING_FIELDS = ('aggregates', 'primary_contrasts', 'crossing_interactions',
                    'within_cell_controls', 'inference', 'dependence', 'missing')
DESCRIPTIVE_DIRECTION = 'Positive differences mean worse retrieval; mean_rank is algebraically dependent on FOSCTTM.'
DESCRIPTIVE_INFERENCE = 'Descriptive only; no significance tests, rankings, thresholds or confidence intervals.'


def hash_value(value):
    need(isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value), 'Invalid SHA-256')
    return value


def boolean(value):
    need(type(value) is bool, 'Expected boolean')
    return value


def count(value):
    need(type(value) is int and value >= 0, 'Expected nonnegative count')
    return value


def metrics(value, *, partial=False):
    allowed = set(METRICS) | ({'num_validation_samples'} if partial else set())
    need(isinstance(value, dict) and (set(value) <= allowed if partial else set(value) == allowed), 'Metric fields differ')
    need(all(type(x) in (int, float) and math.isfinite(x) for x in value.values()), 'Nonfinite/nonscalar metric')
    return {k: value[k] for k in value}  # Preserve original parsed scalar types/values.


def disposition(value):
    need(value in STATUS, 'Unknown status')
    return value


def omission(value):
    """Bind omitted detail without exposing arbitrary strings or nested fields."""
    return {'present': value is not None,
            'entries': len(value) if isinstance(value, (dict, list, str)) else None,
            'canonical_json_sha256': object_hash(value)}


def allowed_attempt(path, vision, text, seed):
    need(isinstance(path, str) and re.fullmatch(re.escape(f'{vision}--{text}/seed-{seed}/') +
         r'attempt-(0[1-9]|[1-9][0-9])', path), 'Unsafe/incorrect attempt identity')
    return path


def project_attempt(item, vision, text, seed):
    attempt = allowed_attempt(item['attempt'], vision, text, seed)
    terminal = item.get('terminal_marker')
    need(terminal in (None, 'COMPLETE.json', 'FAILED.json'), 'Unknown terminal marker')
    partials = item['terminal_partial_markers']
    need(isinstance(partials, list) and set(partials) <= {'COMPLETE.json.partial', 'FAILED.json.partial'}, 'Unknown partial marker')
    need(isinstance(item['findings'], list) and isinstance(item['artifact_names'], list), 'History detail shape differs')
    allowed_files = {'started.json', 'journal.jsonl', 'cell.json', 'split.pt', 'COMPLETE.json', 'FAILED.json',
                     'COMPLETE.json.partial', 'FAILED.json.partial', 'retry-review.json'} | {
        f'{m}/{f}' for m in ranks.METHODS for f in ('map.pt', 'ranks.pt', 'result.json')}
    inputs = item['input_hashes']; need(isinstance(inputs, dict), 'Attempt hash inventory differs')
    files = {name: hash_value(value) for name, value in inputs.items() if name in allowed_files}
    need(set(item['methods']) == set(ranks.METHODS), 'Attempt method coverage differs')
    methods = {}
    for name in ranks.METHODS:
        method = item['methods'][name]
        need(isinstance(method.get('findings'), list), 'Method findings shape differs')
        projected = {'status': disposition(method['status']), 'eligible': boolean(method['eligible']),
            'reported_metrics': metrics(method['reported_metrics'], partial=True),
            'findings': omission(method['findings']), 'full_reader_method_record': omission(method)}
        for field in ('fit_status', 'evaluation_status', 'diagnostic_status'):
            if field in method:
                projected[field] = disposition(method[field])
        if 'ranks_sha256' in method:
            projected['ranks_sha256'] = hash_value(method['ranks_sha256'])
        # Diagnostic arrays, error text/types and other unrestricted details remain bound, not copied.
        methods[name] = projected
    review = item.get('retry_review')
    retry = {'binding_verified': boolean(item['retry_review_binding_verified']), 'record': omission(review)}
    if review is not None:
        need(isinstance(review, dict), 'Retry review must be an object')
        for field in ('reviewed', 'source_or_numerics_changed'):
            retry[field] = boolean(review[field])
        reason = review.get('reason_kind')
        need(reason in ('external_execution_failure', 'input_failure'), 'Unknown retry reason category')
        previous = review.get('previous_attempt')
        need(isinstance(previous, str) and re.fullmatch(r'attempt-(0[1-9]|[1-9][0-9])', previous), 'Invalid retry predecessor')
        retry.update(reason_kind=reason, previous_attempt=previous,
                     reason_text=omission(review.get('reason')), previous_artifacts=omission(review.get('previous_artifacts')))
    return {'attempt': attempt, 'cell_status': disposition(item['cell_status']), 'terminal_marker': terminal,
        'terminal_partial_markers': list(partials), 'terminal_manifest_verified': boolean(item['terminal_manifest_verified']),
        'split_verified': boolean(item['split_verified']), 'findings': omission(item['findings']),
        'artifact_hashes': files, 'full_artifact_hash_inventory': omission(inputs),
        'artifact_name_inventory': omission(item['artifact_names']), 'retry_review': retry,
        'methods': methods, 'full_reader_attempt_record': omission(item)}


def model_contrast_terms():
    """Declared identities/operands only, with no metric arithmetic."""
    result = {}
    def add(key, terms):
        result[key] = [dict(coefficient=c, vision=v, text=t, method=m) for c, v, t, m in terms]
    for m in ranks.METHODS:
        for t in ranks.TEXT[1:]:
            add(f'primary/text/{t}/{m}', [(1,'dinov2',t,m),(-1,'dinov2','mpnet',m)])
        for v in ranks.VISION[1:]:
            add(f'primary/vision/{v}/{m}', [(1,v,'mpnet',m),(-1,'dinov2','mpnet',m)])
            for t in ranks.TEXT[1:]:
                add(f'interaction/{v}/{t}/{m}', [(1,v,t,m),(-1,v,'mpnet',m),(-1,'dinov2',t,m),(1,'dinov2','mpnet',m)])
    for v in ranks.VISION:
        for t in ranks.TEXT:
            for label, left, right in (
                ('orthogonal_pairing','orthogonal_paired','orthogonal_shuffled'),
                ('linear_pairing','linear_paired','linear_shuffled'),
                ('rosetta_minus_orthogonal_paired','rosetta','orthogonal_paired'),
                ('rosetta_minus_linear_paired','rosetta','linear_paired')):
                add(f'within/{v}/{t}/{label}', [(1,v,t,left),(-1,v,t,right)])
    return result


def project_series(group, name, *, terms=None, observations=None):
    need(group['status'] == 'available' and type(group['ddof']) is int and group['ddof'] == 1 and
         group['required_seeds'] == group['available_seeds'] == list(ranks.SEEDS), 'Incomplete five-seed series')
    need(set(group['per_seed']) == {str(s) for s in ranks.SEEDS}, 'Series seed coverage differs')
    result = {'status': group['status'], 'required_seeds': list(group['required_seeds']),
        'available_seeds': list(group['available_seeds']), 'ddof': group['ddof'],
        'mean': metrics(group['mean']), 'sample_sd': metrics(group['sample_sd']), 'per_seed': {}}
    if terms is not None:
        need(group.get('terms') == terms, 'Contrast operands differ')
        # Reconstruct only the four allowlisted identity fields, retaining their values.
        result['terms'] = [{k: term[k] for k in ('coefficient','vision','text','method')} for term in group['terms']]
    for seed in ranks.SEEDS:
        item = group['per_seed'][str(seed)]; need(item['status'] == 'available', 'Missing series observation')
        projected = {'status': item['status'], 'metrics': metrics(item['metrics'])}
        if terms is not None:
            need(item.get('missing_entries') == [], 'Missing contrast operands')
            projected['missing_entries'] = []
        else:
            v, t, method = name.split('/')
            need(item == observations[f'{v}/{t}/{seed}/{method}'], 'Cell-score observation differs from selected observation')
        result['per_seed'][str(seed)] = projected
    return result


def expected_audit_fields(observations, descriptive):
    individual = {f'{key}/{metric}': value['metrics'][metric]
                  for key, value in observations.items() for metric in METRICS}
    aggregates = {}
    for category in ('cell_scores', 'contrasts'):
        for key, group in descriptive[category].items():
            for seed, value in group['per_seed'].items():
                for metric in METRICS:
                    aggregates[f'{category}/{key}/individual/{seed}/{metric}'] = value['metrics'][metric]
            for statistic in ('mean', 'sample_sd'):
                for metric in METRICS:
                    aggregates[f'{category}/{key}/{statistic}/{metric}'] = group[statistic][metric]
    return {'rank_metrics': individual, 'aggregates_from_saved_scalars': aggregates,
            'aggregates_from_reconstructed_metrics': aggregates}


def check_audit(report, args, summary, ticket_data, stage, selected, fields):
    need(report.get('schema_version') == 1 and report.get('study') == 'model' and
         report.get('status') == 'comparison_generated' and report.get('scientific_acceptance') == 'not_assessed' and
         report.get('numeric_tolerance') is None, 'Completed independent model comparison required')
    need(report['coverage'] == {'scheduled_cells':60,'method_observations':300,'five_seed_series':60,'contrast_series':103}, 'Audit coverage differs')
    inputs = report['inputs']
    expected = {'summary_sha256':args.summary_sha256,'acceptance_sha256':args.acceptance_sha256,
        'study_receipt_sha256':args.stage_receipt_sha256,'protocol_sha256':args.protocol_sha256,
        'ticket_sha256':digest(ticket_data),'ticket_canonical_sha256':stage['ticket_sha256'],
        'masks_sha256':None,'audit_code_sha256':digest(Path(__file__).with_name('audit_rank_summaries.py').read_bytes()),
        'accepted_reader_input_hashes':summary['input_hashes']}
    need(all(inputs.get(k) == value for k,value in expected.items()), 'Independent audit evidence binding differs')
    hash_value(inputs['canonical_coco_ids_sha256'])
    evidence = report['selected_artifacts']; need(isinstance(evidence,list) and len(evidence)==300, 'Audit selection coverage differs')
    indexed = {entry['identity']: entry for entry in evidence}
    need(len(indexed)==300 and set(indexed)==set(selected), 'Duplicate/missing audit selections')
    for identity, value in selected.items():
        need(indexed[identity] == {'identity':identity,'selected_attempt':value['selected_attempt'],
             'terminal_marker':value['terminal_marker'],'ranks_sha256':value['ranks_sha256']}, 'Audit selected artifact differs')
    need(set(report['comparisons'])==set(fields)==set(report['comparison_counts']), 'Audit comparison sections differ')
    counts = {}
    for section, expected_values in fields.items():
        rows = report['comparisons'][section]
        need(isinstance(rows,list) and len(rows)==len(expected_values), 'Audit scalar coverage differs')
        indexed = {row['field']:row for row in rows}
        need(len(indexed)==len(rows) and set(indexed)==set(expected_values), 'Audit comparison field coverage differs')
        for name, frozen in expected_values.items():
            row=indexed[name]
            need(type(row['frozen']) is type(frozen) and row['frozen']==frozen, 'Audit frozen scalar differs')
            boolean(row['binary64_exactly_equal']); boolean(row['exactly_equal_at_reported_precision'])
            for field in ('independent','signed_difference_frozen_minus_independent','absolute_difference'):
                need(type(row[field]) in (int,float) and math.isfinite(row[field]), 'Invalid audit numeric comparison')
        expected_count = {'scalars':len(rows), 'binary64_differences':sum(not row['binary64_exactly_equal'] for row in rows),
            'nonzero_exact_or_decimal_differences':sum(not row['exactly_equal_at_reported_precision'] for row in rows)}
        need(report['comparison_counts'][section]==expected_count, 'Audit comparison counts differ')
        counts[section]=expected_count
    return counts


def check_manifest(manifest, args, summary, selected):
    need(manifest.get('schema_version')==1 and manifest.get('status')=='complete' and
         manifest.get('protocol_id')==summary['protocol_id'] and manifest.get('summary_sha256')==args.summary_sha256 and
         manifest.get('accepted_reader_receipt_sha256')==args.acceptance_sha256 and
         manifest.get('exporter_sha256')==digest(Path(ranks.__file__).read_bytes()) and
         manifest.get('reader_input_hashes')==summary['input_hashes'], 'Rank manifest evidence binding differs')
    need(manifest.get('partition_count')==300 and manifest.get('rows_per_partition')==ranks.ROWS and
         manifest.get('columns')==list(ranks.HEADER) and len(manifest['files'])==300, 'Rank manifest coverage differs')
    seen=set(); output={}
    for item in manifest['files']:
        identity=f"{item['vision']}/{item['text']}/{item['seed']}/{item['method']}"
        need(identity in selected and identity not in seen, 'Duplicate/unexpected rank partition');seen.add(identity)
        value=selected[identity];v,t,seed,method=identity.split('/')
        need(item['file']==f'{v}--{t}/seed-{seed}/{method}.csv.gz' and
             item['selected_attempt']==value['selected_attempt'] and item['cell_terminal']==value['terminal_marker'] and
             item['rank_artifact']==f"{value['selected_attempt']}/{method}/ranks.pt" and
             item['rank_artifact_sha256']==value['ranks_sha256'] and
             item['canonical_ranks_by_coco_id_sha256']==value['canonical_ranks_sha256'] and
             item['rows']==ranks.ROWS and count(item['bytes'])>0, 'Rank partition metadata differs')
        output[identity]={'file':item['file'],'sha256':hash_value(item['sha256']),'bytes':item['bytes'],'rows':item['rows']}
    return output


def export_summary(args):
    paths = {name:Path(getattr(args,name)) for name in ('summary','acceptance','stage_receipt','protocol','audit','rank_manifest')}
    data={name:ranks.pinned_bytes(path,getattr(args,name+'_sha256')) for name,path in paths.items()}
    values={name:ranks.read_json(raw) for name,raw in data.items()}
    summary,stage,protocol=values['summary'],values['stage_receipt'],values['protocol']
    ranks.validate_summary(summary)
    need(args.protocol_sha256==ranks.PINNED['protocol'] and protocol['protocol_id']==summary['protocol_id'] and
         protocol['seeds']==list(ranks.SEEDS) and protocol['counts']['validation']==ranks.ROWS and
         object_hash(protocol['research_design']['reporting'])==summary['input_hashes']['prespecified_reporting'], 'Frozen protocol differs')
    need(paths['summary'].name=='summary.json' and paths['acceptance'].name=='accepted.json' and
         paths['summary'].resolve().parent==paths['acceptance'].resolve().parent and
         paths['stage_receipt'].name=='model-stage-complete.json', 'Use the matching model-stage reader evidence')
    need(values['acceptance']=={'summary_sha256':args.summary_sha256}, 'Reader acceptance differs')
    need(set(stage)=={'summary_path','summary_sha256','verification','ticket_sha256'} and stage['summary_sha256']==args.summary_sha256 and
         Path(stage['summary_path']).name=='summary.json' and Path(stage['summary_path']).parent==Path(stage['verification']), 'Model-stage receipt binding differs')
    ticket_path=paths['acceptance'].parent/'ticket.json'; ticket_data=ticket_path.read_bytes();ticket=ranks.read_json(ticket_data)
    need(object_hash(ticket)==stage['ticket_sha256'] and ticket.get('kind')=='verification' and ticket.get('study')=='model' and
         ticket.get('preflight') is False and ticket.get('final_audit') is False and ticket.get('out')==stage['summary_path'], 'Reader ticket type/binding differs')
    need(ticket.get('runner_bindings_sha256')==summary['input_hashes']['runner_bindings'] and
         ticket.get('summary_bindings_sha256')==summary['input_hashes']['private_bindings'], 'Reader input bindings differ')
    out=Path(args.out).absolute();resolved=out.resolve()
    need(not out.exists() and not out.is_symlink() and
         not any(resolved.is_relative_to(p.resolve().parent) for p in [*paths.values(),ticket_path]), 'Use a fresh output outside input evidence directories')
    stamp=summary['recorded_at_utc'];need(isinstance(stamp,str), 'Missing reader timestamp')
    need(datetime.fromisoformat(stamp.replace('Z','+00:00')).utcoffset() is not None, 'Reader timestamp requires timezone')
    for mapping in summary['validation_order_and_coco_id_mapping'].values():
        ranks.validate_mapping(mapping)
    observations={};selected={};history={}
    for v in ranks.VISION:
        for t in ranks.TEXT:
            for seed in ranks.SEEDS:
                cell=f'{v}/{t}/{seed}';attempts=summary['attempts'][cell]
                need(attempts and len({x['attempt'] for x in attempts})==len(attempts), 'Empty/duplicate attempt history')
                complete=[x for x in attempts if x['terminal_marker']=='COMPLETE.json']
                need(len(complete)==1 and complete[0]['attempt']==max(x['attempt'] for x in attempts) and
                     complete[0]['terminal_manifest_verified'] is True and complete[0]['split_verified'] is True and
                     complete[0]['findings']==[] and all(m['eligible'] for m in complete[0]['methods'].values()), 'Full model-stage cell completion absent')
                history[cell]=[project_attempt(x,v,t,seed) for x in attempts]
                for method in ranks.METHODS:
                    identity=f'{cell}/{method}';o=summary['observations'][identity]
                    path,attempt,value,rank_sha=ranks.selected_method(summary,v,t,seed,method)
                    need(count(o['attempt_count'])==len(attempts) and count(o['eligible_completions'])==sum(x['methods'][method]['eligible'] is True for x in attempts) and
                         o['selection_rule']==SELECTION_RULE, 'Observation history counts/rule differ')
                    observations[identity]={'status':o['status'],'metrics':metrics(o['metrics']),
                        'selected_attempt':path,'attempt_count':o['attempt_count'],'eligible_completions':o['eligible_completions'],
                        'repeat_agreement':o['repeat_agreement'],'selection_rule':o['selection_rule'],'retry_history_findings':[],
                        'cell_terminal':attempt['terminal_marker'],'ranks_sha256':hash_value(rank_sha)}
                    selected[identity]={'selected_attempt':path,'terminal_marker':attempt['terminal_marker'],'ranks_sha256':rank_sha,
                        'canonical_ranks_sha256':hash_value(value['rank_check']['canonical_ranks_by_coco_id_sha256'])}
    descriptive=summary['descriptive'];terms=model_contrast_terms()
    score_ids={f'{v}/{t}/{m}' for v in ranks.VISION for t in ranks.TEXT for m in ranks.METHODS}
    need(descriptive['metrics']==list(METRICS) and set(descriptive['cell_scores'])==score_ids and
         set(descriptive['contrasts'])==set(terms), 'Missing/unexpected scalar or contrast series')
    need(descriptive.get('direction')==DESCRIPTIVE_DIRECTION and
         descriptive.get('inference')==DESCRIPTIVE_INFERENCE, 'Descriptive direction/inference differs')
    projected={'metrics':list(METRICS),'direction':descriptive['direction'],
        'inference':descriptive['inference'],'cell_scores':{},'contrasts':{}}
    for name,group in descriptive['cell_scores'].items():
        projected['cell_scores'][name]=project_series(group,name,observations=summary['observations'])
    for name,group in descriptive['contrasts'].items():
        projected['contrasts'][name]=project_series(group,name,terms=terms[name])
    # Only copy pinned protocol prose, not arbitrary strings from reader history.
    design=protocol['research_design'];reporting=design['reporting']
    scope={k:reporting[k] for k in REPORTING_FIELDS}
    scope.update(stage=design['stage'],prior_information=design['prior_information'],
        interpretation_unit=design['interpretation_unit'],
        paired_control_extra_information=design['paired_controls']['extra_information'],
        score_direction=design['evaluation']['direction'],mean_rank_relation=design['evaluation']['mean_rank_relation'])
    need(all(isinstance(v,str) and not any(p in v for p in ('/home/','/Users/','/tmp/','/var/','file://')) for v in scope.values()), 'Unsafe protocol scope')
    fields=expected_audit_fields(observations,projected)
    audit_counts=check_audit(values['audit'],args,summary,ticket_data,stage,selected,fields)
    partitions=check_manifest(values['rank_manifest'],args,summary,selected)
    result={'schema_version':1,'status':'projection_prepared','scientific_acceptance':'not_assessed','publication_authorized':False,
        'publication_review_reference':None,'protocol_id':summary['protocol_id'],'reader_recorded_at_utc':stamp,
        'inputs':{**{name+'_sha256':getattr(args,name+'_sha256') for name in paths},
            'ticket_sha256':digest(ticket_data),'ticket_canonical_sha256':stage['ticket_sha256'],
            'projection_code_sha256':digest(Path(__file__).read_bytes()),
            'metadata_validator_sha256':digest(Path(ranks.__file__).read_bytes()),
            'reader_input_hashes':{k:hash_value(v) for k,v in summary['input_hashes'].items()}},
        'coverage':{'model_pairings':12,'cell_seed_jobs':60,'method_observations':300,'five_seed_score_series':60,'contrast_series':103},
        'baseline_replay_gate':{'status':'satisfied','required_seeds':list(ranks.SEEDS),
            'scope':'Rosetta retrieval replay against our preserved baseline; classification was not rerun. This does not establish agreement with author values.'},
        'scope':scope,'observations':observations,'descriptive':projected,'attempt_history':history,
        'history_counts':{'attempts':sum(len(x) for x in history.values()),
            'failed_terminal_attempts':sum(x['terminal_marker']=='FAILED.json' for group in history.values() for x in group),
            'unterminated_attempts':sum(x['terminal_marker'] is None for group in history.values() for x in group),
            'attempts_with_findings':sum(x['findings']['entries']>0 for group in history.values() for x in group)},
        'independent_audit':{'status':'comparison_generated','scientific_acceptance':'not_assessed','numeric_tolerance':None,'comparison_counts':audit_counts},
        'rank_partitions':partitions,
        'limitations':['Frozen scalar values are copied without metric arithmetic, rounding, ranking or significance claims.',
            'Lower FOSCTTM and mean rank mean better retrieval; positive signed contrasts mean worse retrieval.',
            'The five seeds share fixed data and pretrained encoders; they are not five independent datasets or encoder-training runs.',
            'Evidence hashes and controller acceptance provenance must be independently trusted. No rank tensors, maps, CSV contents or processes are read.',
            'Omitted history details are retained by their summary binding and canonical hashes/counts, not erased or interpreted as absent failures.',
            'Preparation is not scientific or publication approval. Final independent review remains required and must be linked when results are published.',
            'This implementation has synthetic-only validation until a separately documented real-output review.']}
    payload=(json.dumps(result,indent=2,allow_nan=False)+'\n').encode()
    out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('xb') as handle:
        handle.write(payload);handle.flush();os.fsync(handle.fileno())
    return {'status':'projection_prepared','sha256':digest(payload),'bytes':len(payload),
            'scientific_acceptance':'not_assessed','publication_authorized':False,'coverage':result['coverage']}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('summary','acceptance','stage-receipt','protocol','audit','rank-manifest'):
        parser.add_argument('--'+name,type=Path,required=True)
        parser.add_argument('--'+name+'-sha256',required=True)
    parser.add_argument('--out',type=Path,required=True)
    try:
        print(json.dumps(export_summary(parser.parse_args()),allow_nan=False))
    except (ValueError,KeyError,TypeError,OSError) as error:
        print(f'Input rejected: {error}',file=sys.stderr);return 2
    return 0


if __name__=='__main__':
    raise SystemExit(main())
