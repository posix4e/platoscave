#!/usr/bin/env python3
"""Copy complete accepted source-study scalars into a bounded public projection."""
from __future__ import annotations

import argparse
from datetime import datetime
from itertools import combinations
import json
import math
import os
from pathlib import Path
import re
import sys

# Reporting metadata helpers only; no scientific module or tensor loader is called.
import export_model_summary as metadata

need, digest, object_hash = metadata.need, metadata.digest, metadata.object_hash
hash_value, boolean, count, omission = metadata.hash_value, metadata.boolean, metadata.count, metadata.omission
controller_ticket_hash = metadata.controller_ticket_hash
SEEDS = (734796314, 576165995, 2197670066, 839703249, 2584932063)
ROWS = 40504
STRATA = {'never_known_spc_overlap': 37169, 'original_spc_overlap': 3335, 'all': ROWS}
VIEWS = ('original_overall', *STRATA)
METRICS = ('foscttm', 'mean_rank', 'median_rank', 'recall_at_1', 'recall_at_5', 'recall_at_10')
STATS = ('mean', 'sample_sd', 'minimum', 'maximum')
PINS = {'protocol': '75076af19cdeeefd3d839e9eb2db9a33af2c47e17a7a0600339763b95b90eeae',
        'masks': '1dd8f1fa7fd33207ecfe359643009352251e59820a1c12e936c75650e5f16ab6',
        'summarizer': '1c59ca05f15df720bb66160934a725837cc9953481f71781c39a76ad567dcb85',
        'runner': '6dbf7c3132a75607d5d7912759a88feba3ef943deeb89f3c6d81afcc045648af'}
PREPARATION_PINS = {'compositions': 'a501577aec0fc80fbdd796ce1c672d05c8aea72ccde7a2b3b68f1e6a64e29b9b',
    'intersections': 'b8d34bb2f9dd133a6b27b993a6c8f3eaa530cb69e9c77882bbe3d4d6d0a0f9c3'}
PREPARATION_RELEASE = 'fc65f7a87b5956d97074e850cc7dc31b3f3df9316398d7fbdf2ec0a0f3205d31'
DIRECTION = 'Positive rank/FOSCTTM differences and negative Recall differences mean worse retrieval.'
INFERENCE = 'Descriptive conditional variation only. Masks share sources; no independent-corpus, significance, equivalence or power claim.'
MEMBERSHIP = ('known_coco_train', 'known_coco_validation', 'mapped_coco_outside_reference', 'no_coco_mapping')
VARIANTS = {'O': 'original', 'C': 'validation_clean', 'D': 'fully_dedup',
            'Rv': 'random_drop_validation_size_matched', 'Rp': 'random_drop_size_matched'}
COVERAGE = {'scheduled_fits': 115, 'masks': 23, 'five_seed_series': 92,
            'random_mask_spread_series': 40, 'contrast_series': 12}
COMPOSITION_SCOPE = 'Known metadata identities only; no image-byte, near-duplicate or pretraining-cleanliness claim.'


def metrics(value):
    need(isinstance(value, dict) and set(value) == set(METRICS), 'Metric fields differ')
    need(all(type(v) in (int, float) and math.isfinite(v) for v in value.values()), 'Nonfinite/nonscalar metric')
    return dict(value)


def safe_reported(value):
    """Retain safe observed scalars even in an earlier failed or invalid attempt."""
    if not isinstance(value, dict):
        return {'full_record': omission(value), 'views': {}}
    views = {'original_overall': value.get('overall')}
    fixed = value.get('fixed_strata')
    views.update({v: fixed.get(v) for v in STRATA} if isinstance(fixed, dict) else {})
    return {'full_record': omission(value), 'views': {v: {m: x for m, x in item.items()
        if m in (*METRICS, 'num_queries', 'gallery_size', 'foscttm_denominator') and
        type(x) in (int, float) and math.isfinite(x)} for v, item in views.items() if isinstance(item, dict)}}


def allowed_attempt(value, mask, seed):
    need(isinstance(value, str) and re.fullmatch(re.escape(f'{mask}/seed-{seed}/') +
         r'attempt-(0[1-9]|[1-9][0-9])', value), 'Unsafe/incorrect attempt identity')
    return value


def project_history(item, mask, seed):
    attempt = allowed_attempt(item['attempt'], mask, seed)
    markers, partials = item['terminal_markers'], item['terminal_partial_markers']
    need(isinstance(markers, list) and len(markers) == len(set(markers)) and
         set(markers) <= {'COMPLETE.json', 'FAILED.json'}, 'Unknown terminal marker')
    need(isinstance(partials, list) and set(partials) <= {'COMPLETE.json.partial', 'FAILED.json.partial'}, 'Unknown partial marker')
    need(isinstance(item['findings'], list) and isinstance(item['artifact_names'], list), 'History detail shape differs')
    files = {'started.json', 'journal.jsonl', 'COMPLETE.json', 'FAILED.json', 'COMPLETE.json.partial',
             'FAILED.json.partial', 'training_order.pt', 'map.pt', 'ranks.pt', 'metrics.json',
             'result.json', 'failure.json', 'retry-review.json'}
    hashes = item['input_hashes']; need(isinstance(hashes, dict), 'Artifact hash inventory differs')
    review = item.get('retry_review')
    retry = {'binding_verified': boolean(item['retry_review_binding_verified']), 'record': omission(review)}
    if review is not None:
        need(isinstance(review, dict), 'Retry review must be an object')
        # Only categories and predecessor identities are copied, never unrestricted prose.
        reason, previous = review.get('reason_kind'), review.get('previous_attempt')
        if reason in ('external_execution_failure', 'input_failure'):
            retry['reason_kind'] = reason
        if isinstance(previous, str) and re.fullmatch(r'attempt-(0[1-9]|[1-9][0-9])', previous):
            retry['previous_attempt'] = previous
        for key in ('reviewed', 'source_or_numerics_changed'):
            if type(review.get(key)) is bool:
                retry[key] = review[key]
        retry['reason_text'] = omission(review.get('reason'))
        retry['previous_artifacts'] = omission(review.get('previous_artifacts'))
    return {'attempt': attempt, 'terminal_markers': list(markers), 'terminal_partial_markers': list(partials),
        'terminal_manifest_verified': boolean(item['terminal_manifest_verified']), 'eligible': boolean(item['eligible']),
        'training_order_verified': boolean(item['training_order_verified']), 'findings': omission(item['findings']),
        'artifact_hashes': {k: hash_value(v) for k, v in hashes.items() if k in files},
        'full_artifact_hash_inventory': omission(hashes), 'artifact_name_inventory': omission(item['artifact_names']),
        'reported_metrics': safe_reported(item.get('reported_metrics')), 'failure': omission(item.get('failure')),
        'retry_review': retry, 'full_reader_attempt_record': omission(item)}


def mask_metadata(values, args, protocol, masks):
    bundle = values['masks']; names = masks
    need(bundle.get('schema_version') == 1 and bundle.get('protocol_id') == protocol['protocol_id'] and
         bundle.get('protocol_sha256') == args.protocol_sha256 and
         bundle.get('row_identity_sha256') == protocol['row_identity_sha256'] and set(bundle['masks']) == set(names), 'Mask bundle identity differs')
    need(set(bundle['strata']) == set(STRATA), 'Fixed stratum coverage differs')
    for view, n in STRATA.items():
        rows = bundle['strata'][view]
        need(isinstance(rows, list) and len(rows) == n and all(type(i) is int for i in rows) and
             rows == sorted(set(rows)) and all(0 <= i < ROWS for i in rows), 'Fixed stratum indices differ')
    a, b = (set(bundle['strata'][v]) for v in ('never_known_spc_overlap', 'original_spc_overlap'))
    need(not a & b and a | b == set(range(ROWS)) and bundle['strata']['all'] == list(range(ROWS)), 'Strata do not partition gallery')
    for name in PREPARATION_PINS:
        document = values[name]
        expected = {'schema_version': 1, 'operation': 'prepare_all_source_masks', 'protocol_id': protocol['protocol_id'],
            'protocol_sha256': args.protocol_sha256, 'masks_sha256': args.masks_sha256,
            'release_lock_sha256': PREPARATION_RELEASE, 'row_identity_sha256': protocol['row_identity_sha256']}
        need(all(document.get(k) == v for k, v in expected.items()), 'Preparation metadata binding differs')
    compositions = values['compositions']['masks']; need(set(compositions) == set(names), 'Composition coverage differs')
    projected = {}
    for name in names:
        mask, item = bundle['masks'][name], compositions[name]
        family = name.split('-')[0]; selection_seed = int(name.split('-')[1]) if '-' in name else None
        need(mask['variant'] == item['variant'] == VARIANTS[family] and
             mask['selection_seed'] == item['selection_seed'] == selection_seed and
             mask['mask_sha256'] == item['mask_sha256'] and
             mask['row_identity_sha256'] == item['row_identity_sha256'] == protocol['row_identity_sha256'], 'Mask/composition identity differs')
        rows = mask['train_y_indices']
        need(isinstance(rows, list) and all(type(i) is int for i in rows) and rows == sorted(set(rows)) and
             all(0 <= i < protocol['counts']['original_y'] for i in rows) and
             len(rows) == item['paragraph_rows'] == protocol['arm_counts'][family] and
             item['training_image_rows'] == protocol['counts']['train_x'], 'Mask/composition row count differs')
        need(item['scope'] == COMPOSITION_SCOPE, 'Composition scope differs')
        projected[name] = {k: item[k] for k in ('variant', 'selection_seed', 'scope')}
        projected[name].update({k: hash_value(item[k]) for k in ('mask_sha256', 'row_identity_sha256')})
        for field in ('paragraph_rows', 'source_groups', 'duplicate_source_rows', 'training_image_rows'):
            projected[name][field] = count(item[field])
        for field in ('source_membership_groups', 'source_membership_rows'):
            need(set(item[field]) == set(MEMBERSHIP), 'Membership categories differ')
            projected[name][field] = {k: count(item[field][k]) for k in MEMBERSHIP}
        histogram = item['group_size_histogram']
        need(isinstance(histogram, dict) and all(re.fullmatch(r'[1-9][0-9]*', k) for k in histogram), 'Histogram identity differs')
        projected[name]['group_size_histogram'] = {k: count(v) for k, v in histogram.items()}
    pairs = values['intersections']['pairs']; expected_pairs = list(combinations(names, 2))
    need(isinstance(pairs, list) and len(pairs) == len(expected_pairs) == 253, 'Intersection coverage differs')
    intersections = []
    for item, (left, right) in zip(pairs, expected_pairs):
        need(item['left'] == left and item['right'] == right and
             item['left_mask_sha256'] == projected[left]['mask_sha256'] and
             item['right_mask_sha256'] == projected[right]['mask_sha256'], 'Intersection mask identity/order differs')
        result = {k: item[k] for k in ('left', 'right', 'left_mask_sha256', 'right_mask_sha256')}
        for key in ('shared_rows', 'union_rows', 'shared_source_groups', 'union_source_groups',
                    'shared_known_training_sources', 'shared_known_validation_sources'):
            result[key] = count(item[key])
        intersections.append(result)
    return {'masks': projected, 'pairwise_intersections': intersections,
        'fixed_strata': {v: {'num_queries': n, 'canonical_indices_sha256': object_hash(bundle['strata'][v])} for v, n in STRATA.items()},
        'gallery_size': ROWS, 'foscttm_denominator': ROWS - 1, 'row_identity_sha256': hash_value(protocol['row_identity_sha256'])}


def project_group(group, required, *, operands=None, expected_individual=None):
    need(group['status'] == 'available' and type(group['ddof']) is int and group['ddof'] == 1 and
         group['required'] == group['available'] == required and
         set(group['individual']) == {str(x) for x in required}, 'Incomplete series/operand coverage')
    output = {'status': 'available', 'ddof': group['ddof'], 'required': list(required), 'available': list(required),
              **{k: metrics(group[k]) for k in STATS}, 'individual': {}}
    if operands is not None:
        need(group.get('operands') == operands, 'Contrast operands differ')
        output['operands'] = list(group['operands'])
    for identity in required:
        item = group['individual'][str(identity)]
        need(item['status'] == 'available', 'Unavailable individual')
        output['individual'][str(identity)] = {'status': 'available', 'metrics': metrics(item['metrics'])}
        if operands is not None:
            need(item.get('missing_masks') == [], 'Missing contrast operand')
            output['individual'][str(identity)]['missing_masks'] = []
        else:
            expected = expected_individual[str(identity)]
            need(all(type(item['metrics'][m]) is type(expected[m]) and item['metrics'][m] == expected[m] for m in METRICS), 'Series individual differs from observation')
    return output


def project_descriptive(summary, names, families):
    desc = summary['descriptive']; observations = summary['observations']
    need(desc['metrics'] == list(METRICS) and desc['direction'] == DIRECTION and desc['inference'] == INFERENCE, 'Metric direction/inference differs')
    contrasts = {f'{v}/{label}': [left, *(families[right] if right in families else [right])]
        for v in STRATA for label, left, right in (('C_minus_mean_Rv', 'C', 'Rv'), ('D_minus_mean_Rp', 'D', 'Rp'), ('C_minus_O', 'C', 'O'), ('D_minus_C', 'D', 'C'))}
    need(set(desc['per_mask_five_seed']) == {f'{m}/{v}' for m in names for v in VIEWS} and
         set(desc['within_seed_random_mask_spread']) == {f'{f}/{v}' for f in families for v in VIEWS} and
         set(desc['within_seed_contrasts']) == set(contrasts), 'Descriptive series coverage differs')
    output = {'metrics': list(METRICS), 'direction': desc['direction'], 'inference': desc['inference'],
              'per_mask_five_seed': {}, 'within_seed_random_mask_spread': {}, 'within_seed_contrasts': {}}
    for name, group in desc['per_mask_five_seed'].items():
        mask, view = name.split('/')
        output['per_mask_five_seed'][name] = project_group(group, list(SEEDS),
            expected_individual={str(s): observations[f'{mask}/{s}']['scores'][view] for s in SEEDS})
    for name, groups in desc['within_seed_random_mask_spread'].items():
        family, view = name.split('/')
        need(set(groups) == {str(s) for s in SEEDS}, 'Random spread seed coverage differs')
        output['within_seed_random_mask_spread'][name] = {str(s): project_group(groups[str(s)], families[family],
            expected_individual={m: observations[f'{m}/{s}']['scores'][view] for m in families[family]}) for s in SEEDS}
    for name, group in desc['within_seed_contrasts'].items():
        output['within_seed_contrasts'][name] = project_group(group, list(SEEDS), operands=contrasts[name])
    return output


def check_audit(report, args, summary, ticket_data, stage, selected, descriptive, canonical):
    need(report.get('schema_version') == 1 and report.get('study') == 'source' and
         report.get('status') == 'comparison_generated' and report.get('scientific_acceptance') == 'not_assessed' and
         report.get('numeric_tolerance') is None and report.get('coverage') == COVERAGE, 'Complete independent source comparison required')
    expected = {'summary_sha256': args.summary_sha256, 'acceptance_sha256': args.acceptance_sha256,
        'study_receipt_sha256': args.stage_receipt_sha256, 'protocol_sha256': args.protocol_sha256,
        'ticket_sha256': digest(ticket_data), 'ticket_canonical_sha256': stage['ticket_sha256'], 'masks_sha256': args.masks_sha256,
        'audit_code_sha256': digest(Path(__file__).with_name('audit_rank_summaries.py').read_bytes()),
        'accepted_reader_input_hashes': summary['input_hashes'], 'canonical_coco_ids_sha256': canonical}
    need(all(report['inputs'].get(k) == v for k, v in expected.items()), 'Independent audit evidence binding differs')
    evidence = report['selected_artifacts']; need(isinstance(evidence, list) and len(evidence) == 115, 'Audit selection coverage differs')
    indexed = {x['identity']: x for x in evidence}
    need(len(indexed) == len(evidence) and indexed == selected, 'Audit selected artifact differs')
    individual = {f'{identity}/{view}/{m}': o['scores'][view][m] for identity, o in summary['observations'].items() for view in VIEWS for m in METRICS}
    aggregates = {}
    for category in ('per_mask_five_seed', 'within_seed_random_mask_spread', 'within_seed_contrasts'):
        groups = descriptive[category]
        if category == 'within_seed_random_mask_spread':
            groups = {f'{name}/{seed}': group for name, series in groups.items() for seed, group in series.items()}
        for name, group in groups.items():
            prefix = f'{category}/{name}'
            for identity, item in group['individual'].items():
                aggregates.update({f'{prefix}/individual/{identity}/{m}': item['metrics'][m] for m in METRICS})
            for stat in STATS:
                aggregates.update({f'{prefix}/{stat}/{m}': group[stat][m] for m in METRICS})
    fields = {'rank_metrics': individual, 'aggregates_from_saved_scalars': aggregates, 'aggregates_from_reconstructed_metrics': aggregates}
    need(set(report['comparisons']) == set(report['comparison_counts']) == set(fields), 'Audit section coverage differs')
    counts = {}
    for section, expected_values in fields.items():
        rows = report['comparisons'][section]
        need(isinstance(rows, list) and len(rows) == len(expected_values), 'Audit scalar coverage differs')
        indexed = {row['field']: row for row in rows}
        need(len(indexed) == len(rows) and set(indexed) == set(expected_values), 'Audit scalar identity coverage differs')
        for name, frozen in expected_values.items():
            row = indexed[name]
            need(type(row['frozen']) is type(frozen) and row['frozen'] == frozen, 'Audit frozen scalar differs')
            boolean(row['binary64_exactly_equal']); boolean(row['exactly_equal_at_reported_precision'])
            for field in ('independent', 'signed_difference_frozen_minus_independent', 'absolute_difference'):
                need(type(row[field]) in (int, float) and math.isfinite(row[field]), 'Invalid audit comparison')
        expected_count = {'scalars': len(rows), 'binary64_differences': sum(not x['binary64_exactly_equal'] for x in rows),
            'nonzero_exact_or_decimal_differences': sum(not x['exactly_equal_at_reported_precision'] for x in rows)}
        need(report['comparison_counts'][section] == expected_count, 'Audit comparison counts differ')
        counts[section] = expected_count
    return counts


def export_summary(args):
    names_in = ('summary', 'acceptance', 'stage_receipt', 'protocol', 'masks', 'compositions', 'intersections', 'audit')
    paths = {k: Path(getattr(args, k)) for k in names_in}
    values = {k: metadata.ranks.read_json(metadata.ranks.pinned_bytes(p, getattr(args, k + '_sha256'))) for k, p in paths.items()}
    summary, protocol, stage = (values[k] for k in ('summary', 'protocol', 'stage_receipt'))
    need(args.protocol_sha256 == PINS['protocol'] and args.masks_sha256 == PINS['masks'] and
         all(getattr(args, k + '_sha256') == v for k, v in PREPARATION_PINS.items()), 'Unsupported frozen inputs')
    need(summary.get('schema_version') == 1 and summary.get('status') == 'complete' and summary.get('inventory_findings') == [] and
         summary.get('protocol_id') == protocol.get('protocol_id') == 'rosetta-source-controls-v1' and
         protocol['seeds'] == list(SEEDS) and protocol['counts']['validation'] == ROWS and
         summary.get('planned_masks') == 23 and summary.get('planned_fits') == 115, 'Complete frozen source study required')
    hashes = summary['input_hashes']
    need(set(hashes) == set(PINS) | {'prespecified_reporting', 'private_bindings', 'runner_bindings'} and
         all(hash_value(v) for v in hashes.values()) and all(hashes[k] == v for k, v in PINS.items()) and
         hashes['prespecified_reporting'] == object_hash(protocol['research_design']['reporting']), 'Frozen reader bindings differ')
    need(paths['summary'].name == 'summary.json' and paths['acceptance'].name == 'accepted.json' and
         paths['summary'].resolve().parent == paths['acceptance'].resolve().parent and
         paths['stage_receipt'].name == 'source-stage-complete.json', 'Matching source-stage reader evidence required')
    need(values['acceptance'] == {'summary_sha256': args.summary_sha256}, 'Reader acceptance differs')
    need(set(stage) == {'summary_path', 'summary_sha256', 'verification', 'ticket_sha256'} and stage['summary_sha256'] == args.summary_sha256 and
         Path(stage['summary_path']).name == 'summary.json' and Path(stage['summary_path']).parent == Path(stage['verification']), 'Stage receipt binding differs')
    ticket_path = paths['acceptance'].parent / 'ticket.json'; ticket_data = ticket_path.read_bytes(); ticket = metadata.ranks.read_json(ticket_data)
    need(controller_ticket_hash(ticket) == stage['ticket_sha256'] and ticket.get('kind') == 'verification' and
         ticket.get('study') == 'source' and ticket.get('preflight') is False and ticket.get('final_audit') is False and
         ticket.get('out') == stage['summary_path'] and ticket.get('runner_bindings_sha256') == hashes['runner_bindings'] and
         ticket.get('summary_bindings_sha256') == hashes['private_bindings'], 'Reader ticket binding/type differs')
    out = Path(args.out).absolute()
    need(not out.exists() and not out.is_symlink() and not any(out.resolve().is_relative_to(p.resolve().parent)
         for p in [*paths.values(), ticket_path]), 'Use a fresh output outside input directories')
    stamp = summary['recorded_at_utc']; need(isinstance(stamp, str) and datetime.fromisoformat(stamp.replace('Z', '+00:00')).utcoffset() is not None, 'Timestamp needs timezone')
    families = {f: [f'{f}-{s}' for s in protocol['selection_seeds'][f]] for f in ('Rv', 'Rp')}
    masks = ['O', 'C', 'D', *families['Rv'], *families['Rp']]
    need(len(masks) == len(set(masks)) == 23 and all(len(v) == 10 for v in families.values()), 'Mask plan differs')
    prepared = mask_metadata(values, args, protocol, masks)
    identities = [f'{mask}/{seed}' for mask in masks for seed in SEEDS]
    need(set(summary['observations']) == set(summary['attempts']) == set(identities) and len(summary['schedule']) == 115, 'Fit identity coverage differs')
    mappings = summary['validation_order_and_coco_ids']; need(set(mappings) == {str(s) for s in SEEDS}, 'Mapping seed coverage differs')
    canonical = None
    for mapping in mappings.values():
        ids = mapping['canonical_coco_image_ids']
        need(len(ids) == ROWS and all(type(i) is int and i >= 0 for i in ids) and ids == sorted(set(ids)), 'Canonical COCO identities differ')
        current = object_hash(ids); need(canonical in (None, current), 'Canonical identities differ between seeds'); canonical = current
        for field in ('validation_order', 'query_order', 'key_order'):
            order = mapping[field]
            need(isinstance(order, list) and all(type(i) is int for i in order) and sorted(order) == list(range(ROWS)), 'Invalid mapping permutation')
    observations, history, selected = {}, {}, {}
    for index, identity in enumerate(identities):
        mask, seed_text = identity.split('/'); seed = int(seed_text); schedule = summary['schedule'][index]
        need(schedule['index'] == index and schedule['mask'] == mask and schedule['seed'] == seed and
             schedule['mask_sha256'] == prepared['masks'][mask]['mask_sha256'], 'Schedule binding/order differs')
        orders = schedule['training_order_sha256']; need(set(orders) == {'indices_x', 'indices_y', 'canonical_mask_y'} and all(hash_value(v) for v in orders.values()), 'Training order hash coverage differs')
        attempts = summary['attempts'][identity]; need(isinstance(attempts, list) and attempts, 'Empty attempt history')
        history[identity] = [project_history(x, mask, seed) for x in attempts]
        need([x['attempt'] for x in attempts] == [f'{mask}/seed-{seed}/attempt-{i+1:02d}' for i in range(len(attempts))], 'Attempt counters/order differ')
        complete = [x for x in attempts if 'COMPLETE.json' in x['terminal_markers']]
        item = attempts[-1]
        need(len(complete) == 1 and complete[0] is item and item['terminal_markers'] == ['COMPLETE.json'] and
             item['terminal_partial_markers'] == [] and item['terminal_manifest_verified'] is True and item['eligible'] is True and
             item['training_order_verified'] is True and item['findings'] == [], 'Full source fit completion absent')
        for i, current in enumerate(attempts[1:], 1):
            previous = attempts[i-1]; review = current['retry_review']; marker = 'FAILED.json' if 'FAILED.json' in previous['terminal_markers'] else 'started.json'
            need(not previous['terminal_partial_markers'] and current['retry_review_binding_verified'] is True and isinstance(review, dict) and
                 review.get('previous_attempt') == f'attempt-{i:02d}' and review.get('reviewed') is True and
                 review.get('source_or_numerics_changed') is False and review.get('reason_kind') in ('external_execution_failure', 'input_failure') and
                 isinstance(review.get('previous_artifacts'), dict) and previous['input_hashes'].get(marker) is not None and
                 review['previous_artifacts'].get(marker) == previous['input_hashes'][marker], 'Retry history binding differs')
        o = summary['observations'][identity]
        need(o['status'] == 'available' and o['retry_history_findings'] == [] and o['complete_markers'] == o['verified_complete_attempts'] == 1 and
             count(o['attempt_count']) == len(attempts) and o['selected_attempt'] == item['attempt'] and
             sum(x['eligible'] is True for x in attempts) == 1, 'Selected source completion/history differs')
        check = item['recomputed_rank_metrics']; reported = item['reported_metrics']
        need(check['status'] == 'verified' and check['differences'] == [] and check['original_evaluation_order_and_coco_ids_checked'] is True and
             reported == check['recomputed'] and reported['rank_order'] == 'canonical_original_file_row' and
             reported['overall_reduction_order'] == 'original_randomized_validation_order' and
             reported['tie_policy'] == 'expected_credit_under_uniform_order_within_exact_score_tie_block', 'Accepted rank checks differ')
        hash_value(check['canonical_ranks_higher_ties_sha256'])
        result = item['result']
        need(all(result.get(k) == v for k, v in {'schema_version': 1, 'protocol_id': protocol['protocol_id'], 'protocol_sha256': args.protocol_sha256,
             'masks_sha256': args.masks_sha256, 'mask': mask, 'seed': seed, 'status': 'complete', 'fit_status': 'complete',
             'evaluation_status': 'complete', 'classification_rerun': False, 'mask_sha256': prepared['masks'][mask]['mask_sha256']}.items()), 'Completion result identity/status differs')
        need(result['composition'] == values['compositions']['masks'][mask], 'Fit composition differs from preparation')
        artifacts = result['artifact_hashes']; need(set(artifacts) == {'training_order.pt', 'map.pt', 'ranks.pt', 'metrics.json'} and
             all(hash_value(v) == item['input_hashes'].get(k) for k, v in artifacts.items()) and
             all(hash_value(item['input_hashes'][k]) for k in ('COMPLETE.json', 'result.json')), 'Completed artifact bindings differ')
        need(set(o['scores']) == set(VIEWS) and set(reported['fixed_strata']) == set(STRATA), 'Score view coverage differs')
        scores = {}
        for view in VIEWS:
            value = reported['overall'] if view == 'original_overall' else reported['fixed_strata'][view]
            expected_count = ROWS if view == 'original_overall' else STRATA[view]
            need(set(value) == set(METRICS) | {'num_queries', 'gallery_size', 'foscttm_denominator'} and
                 value['num_queries'] == expected_count and value['gallery_size'] == ROWS and value['foscttm_denominator'] == ROWS-1, 'Metric query/gallery counts differ')
            scores[view] = metrics(o['scores'][view])
            need(all(type(scores[view][m]) is type(value[m]) and scores[view][m] == value[m] for m in METRICS), 'Selected saved metric differs')
        need(reported['original_overall'] == {m: reported['overall'][m] for m in ('foscttm', 'mean_rank')}, 'Original overall binding differs')
        observations[identity] = {'status': 'available', 'scores': scores, 'selected_attempt': item['attempt'], 'attempt_count': o['attempt_count'],
            'complete_markers': o['complete_markers'], 'verified_complete_attempts': o['verified_complete_attempts'], 'retry_history_findings': [],
            'ranks_sha256': artifacts['ranks.pt'], 'mask_sha256': schedule['mask_sha256'], 'training_order_sha256': dict(orders)}
        selected[identity] = {'identity': identity, 'selected_attempt': item['attempt'], 'terminal_marker': 'COMPLETE.json', 'ranks_sha256': artifacts['ranks.pt']}
    descriptive = project_descriptive(summary, masks, families)
    audit_counts = check_audit(values['audit'], args, summary, ticket_data, stage, selected, descriptive, canonical)
    design = protocol['research_design']
    # These fields come only from the exactly pinned public protocol, never reader history.
    scope = {k: design[k] for k in ('stage', 'arms', 'source_identity', 'random_removal', 'fitting', 'evaluation', 'reporting', 'interpretation_limits')}
    result = {'schema_version': 1, 'status': 'projection_prepared', 'scientific_acceptance': 'not_assessed', 'publication_authorized': False,
        'publication_review_reference': None, 'protocol_id': summary['protocol_id'], 'reader_recorded_at_utc': stamp,
        'inputs': {**{k + '_sha256': getattr(args, k + '_sha256') for k in names_in}, 'ticket_sha256': digest(ticket_data),
            'ticket_canonical_sha256': stage['ticket_sha256'], 'projection_code_sha256': digest(Path(__file__).read_bytes()),
            'metadata_helper_sha256': digest(Path(metadata.__file__).read_bytes()),
            'json_validator_sha256': digest(Path(metadata.ranks.__file__).read_bytes()), 'reader_input_hashes': dict(hashes)},
        'coverage': {**COVERAGE, 'metrics': 6, 'score_views': 4, 'contrast_strata': 3, 'pairwise_mask_intersections': 253},
        'scope': scope, 'mask_provenance': prepared, 'observations': observations, 'descriptive': descriptive, 'attempt_history': history,
        'history_counts': {'attempts': sum(len(x) for x in history.values()),
            'failed_terminal_attempts': sum('FAILED.json' in x['terminal_markers'] for group in history.values() for x in group),
            'unterminated_attempts': sum(not x['terminal_markers'] for group in history.values() for x in group),
            'attempts_with_findings': sum(x['findings']['entries'] > 0 for group in history.values() for x in group)},
        'independent_audit': {'status': 'comparison_generated', 'scientific_acceptance': 'not_assessed', 'numeric_tolerance': None, 'comparison_counts': audit_counts},
        'limitations': ['Frozen scalar values are copied exactly as parsed, without metric arithmetic, rounding, ranking or significance claims.',
            'Original overall is a fourth score view, not an additional contrast stratum; all views retain the full gallery.',
            'Positive rank/FOSCTTM differences and negative Recall differences mean worse retrieval.',
            'Masks share sources and the five algorithm seeds share data and frozen encoders; these are not independent corpora or encoder training runs.',
            'Removing fitting paragraphs cannot remove encoder pretraining exposure; exclusion changes semantics and centering, so effects do not isolate memorization.',
            'Metadata identity is not image-byte or perceptual deduplication; sources without COCO mappings remain unknown.',
            'Hashes and controller/audit provenance require independent trust and review. No rank tensor, fitted map or process is read.',
            'Omitted unrestricted history and arrays remain bound by full-summary and record hashes; omissions do not mean failures were absent.',
            'Preparation is not scientific or publication approval. Final independent review must be linked before public interpretation.',
            'This implementation has synthetic-only validation until separately documented real-output review.']}
    payload = (json.dumps(result, indent=2, allow_nan=False) + '\n').encode()
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('xb') as handle:
        handle.write(payload); handle.flush(); os.fsync(handle.fileno())
    return {'status': 'projection_prepared', 'sha256': digest(payload), 'bytes': len(payload),
        'scientific_acceptance': 'not_assessed', 'publication_authorized': False, 'coverage': result['coverage']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('summary', 'acceptance', 'stage-receipt', 'protocol', 'masks', 'compositions', 'intersections', 'audit'):
        parser.add_argument('--' + name, type=Path, required=True); parser.add_argument('--' + name + '-sha256', required=True)
    parser.add_argument('--out', type=Path, required=True)
    try:
        print(json.dumps(export_summary(parser.parse_args()), allow_nan=False))
    except (ValueError, KeyError, TypeError, OSError) as error:
        print(f'Input rejected: {error}', file=sys.stderr); return 2
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
