#!/usr/bin/env python3
"""Describe every declared source-mask fit from saved canonical ranks.

Requires final protocol and mask SHA256s, the original locked runtime, and private
path-only bindings {runner_bindings: ABSOLUTE_JSON, attempts_root: ABSOLUTE_CELLS}.
The tree is cells/{mask}/seed-{seed}/attempt-{01..99}. The complete schedule comes
from the explicit protocol lists; missing fits and every failed attempt remain.
No fit, feature extraction, map rescoring, automatic retry or score selection.

Only one verified COMPLETE attempt may contribute a mask/seed. Reviewed retries
must be contiguous, follow a failed/aborted attempt, and bind its FAILED.json (or
started.json if no marker exists) in retry-review.json inside the new attempt.
The review uses the normal source binding plus previous_attempt, previous_artifacts,
reviewed:true, source_or_numerics_changed:false, a nonempty reason and reason_kind
external_execution_failure|input_failure. Its hash must be in the terminal manifest.
Multiple COMPLETE markers, undocumented histories, or changed ranks/metrics remain
unavailable. Failed observations are reported but never averaged into scores.

Every five-seed mean/sample SD requires all five. Random-mask means and spreads
require every declared mask, conditional on the same algorithm seed. No p-values,
equivalence claim, query-level interval or independence assumption is made.
A future outer release must bind this helper before real results are inspected.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import pickle
import re
import statistics
from types import SimpleNamespace

RUNNER_SHA256 = 'eba2a216f4d95cc34804ef8a585a4713964dc9888df326eb28b389f2bd71df58'
ROOT = Path(__file__).resolve().parent
if hashlib.sha256((ROOT / 'run_source_arm.py').read_bytes()).hexdigest() != RUNNER_SHA256:
    raise ValueError('Pinned source runner differs')
import run_source_arm as arm

r = arm.r
r.require(Path(arm.__file__).resolve() == ROOT / 'run_source_arm.py', 'Use runner from this release')
METRICS = ('foscttm', 'mean_rank', 'median_rank', 'recall_at_1', 'recall_at_5', 'recall_at_10')


def mask_names(protocol):
    return ['O', 'C', 'D'] + [f'{family}-{seed}' for family in ('Rv', 'Rp')
                            for seed in protocol['selection_seeds'][family]]


def key(mask, seed):
    return f'{mask}/{seed}'


def inventory(root, names):
    root = Path(root)
    tree = {key(mask, seed): [] for mask in names for seed in arm.SEEDS}
    findings = []
    if not root.exists():
        return tree, [{'code': 'attempts_root_missing'}]
    r.require(root.is_dir(), 'Attempts root must be a directory')
    seeds = {f'seed-{seed}': seed for seed in arm.SEEDS}
    for folder in sorted(root.iterdir()):
        if not folder.is_dir() or folder.name not in names:
            findings.append({'code': 'unknown_mask_entry', 'relative_path': folder.name}); continue
        for seed_dir in sorted(folder.iterdir()):
            if not seed_dir.is_dir() or seed_dir.name not in seeds:
                findings.append({'code': 'unknown_seed_entry', 'relative_path': str(seed_dir.relative_to(root))}); continue
            for attempt in sorted(seed_dir.iterdir()):
                if not attempt.is_dir() or not re.fullmatch(r'attempt-(0[1-9]|[1-9][0-9])', attempt.name):
                    findings.append({'code': 'unknown_attempt_entry', 'relative_path': str(attempt.relative_to(root))}); continue
                r.require(attempt.resolve().is_relative_to(root.resolve()), 'Attempt escapes declared tree')
                tree[key(folder.name, seeds[seed_dir.name])].append(attempt)
    return tree, findings


def artifact_names(path):
    return sorted(str(item.relative_to(path)) for item in path.rglob('*') if item.is_file())


def expected_orders(rows, mask, seed, api):
    torch, primitives = api.torch, api.primitives
    selected = torch.tensor(mask.train_y_indices, dtype=torch.int64)
    ix, positions_y = primitives.independent_split(len(rows.train_x), len(selected), primitives.Randomness(seed), None)
    rng = primitives.Randomness(seed)
    validation = rng.subset(len(rows.validation), None)
    query, keys = validation[rng.permutation(len(validation))], validation[rng.permutation(len(validation))]
    return {'training': {'indices_x': ix, 'indices_y': selected[positions_y], 'canonical_mask_y': selected},
            'evaluation': {'validation_order': validation, 'query_order': query, 'key_order': keys,
                'canonical_coco_image_ids': torch.tensor([int(k.split(':')[1]) for k in rows.validation], dtype=torch.int64)}}


def check_tensors(actual, expected, torch):
    r.require(isinstance(actual, dict) and set(actual) == set(expected), 'Tensor fields differ')
    for name, value in expected.items():
        candidate = actual[name]
        r.require(isinstance(candidate, torch.Tensor) and candidate.device.type == 'cpu' and
                  candidate.dtype == value.dtype and candidate.shape == value.shape and torch.equal(candidate, value),
                  'Tensor row identity or original RNG order differs')


def metric_differences(reported, expected, prefix=''):
    differences = []
    if isinstance(expected, dict):
        if not isinstance(reported, dict):
            return [{'field': prefix, 'reported': reported, 'recomputed': expected}]
        for name in sorted(set(reported) | set(expected)):
            label = f'{prefix}/{name}' if prefix else name
            if name not in reported or name not in expected:
                differences.append({'field': label, 'reported': reported.get(name), 'recomputed': expected.get(name)})
            else:
                differences.extend(metric_differences(reported[name], expected[name], label))
    elif type(reported) is not type(expected) or reported != expected:
        value = {'field': prefix, 'reported': reported, 'recomputed': expected}
        if type(reported) in (float, int) and type(expected) in (float, int):
            value['signed_error'] = reported - expected
        differences.append(value)
    return differences


def rank_check(path, reported, expected, strata, api):
    torch = api.torch
    record = torch.load(path, weights_only=True, map_location='cpu')
    values = ('ranks', 'num_higher', 'tie_count')
    r.require(isinstance(record, dict) and set(record) == set(expected) | set(values), 'Rank artifact fields differ')
    check_tensors({k: record[k] for k in expected}, expected, torch)
    n = len(expected['canonical_coco_image_ids'])
    r.require(all(isinstance(record[k], torch.Tensor) and record[k].shape == (n,) for k in values), 'Full gallery ranks required')
    ranks, higher, ties = (record[k] for k in values)
    order = expected['validation_order']
    # This also validates dtype, finiteness, exact tie-count/midrank consistency and bounds.
    overall = api.primitives.summarize_ranks(ranks[order], higher[order], ties[order], n)
    fixed = {name: api.primitives.summarize_ranks(ranks[mask], higher[mask], ties[mask], n) for name, mask in strata.items()}
    recomputed = {'original_overall': {k: overall[k] for k in ('foscttm', 'mean_rank')},
        'overall': overall, 'fixed_strata': fixed, 'rank_order': 'canonical_original_file_row',
        'tie_policy': arm.METRIC_RULES['recall_ties'], 'overall_reduction_order': 'original_randomized_validation_order'}
    differences = metric_differences(reported, recomputed)
    return {'status': 'verified' if not differences else 'metric_mismatch', 'recomputed': recomputed,
            'differences': differences, 'equality': 'exact in locked baseline runtime',
            'canonical_ranks_higher_ties_sha256': r.json_hash({k: record[k].tolist() for k in values}),
            'original_evaluation_order_and_coco_ids_checked': True}


def inspect_attempt(path, relative, binding, expected, data, mask, bound):
    findings, hashes = [], {}
    initial_files = artifact_names(path)
    def observe(name):
        target = path / name
        if not target.is_file():
            return None
        hashes[name] = r.digest(target); bound.add(target, hashes[name])
        return target
    def read(name):
        target = observe(name)
        if target is None:
            return None
        try:
            value = r.read_json(target)
            r.require(isinstance(value, dict), 'Expected JSON object'); r.no_local_paths(value)
            json.dumps(value, allow_nan=False)
            return value
        except (ValueError, TypeError):
            findings.append({'code': 'invalid_or_unsafe_json', 'file': name}); return None
    def identity(value):
        return isinstance(value, dict) and all(value.get(k) == v for k, v in binding.items())
    start, complete, failed = read('started.json'), read('COMPLETE.json'), read('FAILED.json')
    result, failure, reported, review = read('result.json'), read('failure.json'), read('metrics.json'), read('retry-review.json')
    markers = [name for name in ('COMPLETE.json', 'FAILED.json') if (path / name).exists()]
    partials = [name for name in ('COMPLETE.json.partial', 'FAILED.json.partial') if (path / name).exists()]
    for name in partials:
        observe(name)
    if partials:
        findings.append({'code': 'partial_terminal_marker', 'files': partials})
    terminal = complete if 'COMPLETE.json' in markers else failed
    manifest = terminal.get('files') if isinstance(terminal, dict) else None
    finalized = len(markers) == 1 and identity(terminal) and isinstance(manifest, dict)
    if not identity(start):
        findings.append({'code': 'missing_or_wrong_start_binding'})
    if len(markers) > 1:
        findings.append({'code': 'both_terminal_markers'})
    elif markers and not finalized:
        findings.append({'code': 'terminal_binding_or_manifest_missing'})
    if finalized:
        for name, digest in manifest.items():
            try:
                target = (path / name).resolve(strict=True)
                r.require(not Path(name).is_absolute() and target.is_relative_to(path.resolve()), 'Manifest escape')
                bound.add(target, digest); hashes[name] = digest
            except (OSError, ValueError, TypeError):
                findings.append({'code': 'terminal_file_hash_failed', 'file': name}); finalized = False
        status = 'complete' if markers == ['COMPLETE.json'] else 'failed'
        if terminal.get('status') != status:
            findings.append({'code': 'terminal_status_differs'}); finalized = False
        if not {'started.json', 'journal.jsonl'} <= set(manifest):
            findings.append({'code': 'terminal_manifest_incomplete'}); finalized = False
    training_ok, checked = False, None
    training = observe('training_order.pt')
    if training:
        try:
            check_tensors(data.api.torch.load(training, weights_only=True, map_location='cpu'), expected['training'], data.api.torch)
            training_ok = True
        except (OSError, ValueError, TypeError, RuntimeError, KeyError, EOFError, pickle.UnpicklingError):
            findings.append({'code': 'training_order_verification_failed'})
    rank_path = observe('ranks.pt'); observe('map.pt')
    if rank_path:
        try:
            checked = rank_check(rank_path, reported, expected['evaluation'], data.strata, data.api)
            if checked['status'] != 'verified':
                findings.append({'code': 'reported_metrics_disagree'})
        except (OSError, ValueError, TypeError, RuntimeError, KeyError, EOFError, pickle.UnpicklingError):
            findings.append({'code': 'rank_identity_or_tie_verification_failed'})
    if markers == ['COMPLETE.json']:
        artifacts = ('training_order.pt', 'map.pt', 'ranks.pt', 'metrics.json')
        expected_counts = {'train_x': len(data.rows.train_x), 'train_y': len(mask.train_y_indices), 'validation': len(data.rows.validation)}
        fields = {'status': 'complete', 'fit_status': 'complete', 'evaluation_status': 'complete',
            'classification_rerun': False, 'counts': expected_counts, 'configured_batch_cap': arm.ALGORITHM['batch_size'],
            'refinement_batch_dimensions': {'x': min(arm.ALGORITHM['batch_size'], expected_counts['train_x']),
                                          'y': min(arm.ALGORITHM['batch_size'], expected_counts['train_y'])},
            'mask_sha256': mask.mask_sha256,
            'composition': json.loads(json.dumps(data.api.masks.composition(data.rows, mask), allow_nan=False))}
        if not identity(result) or any(result.get(k) != v for k, v in fields.items()):
            findings.append({'code': 'completion_result_contract_differs'})
        if not isinstance(manifest, dict) or not {'result.json', *artifacts} <= set(manifest):
            findings.append({'code': 'completion_manifest_incomplete'})
        elif not isinstance(result, dict) or result.get('artifact_hashes') != {name: hashes.get(name) for name in artifacts} or any(manifest.get(name) != hashes.get(name) for name in artifacts):
            findings.append({'code': 'completion_artifact_bindings_differ'})
    retry_verified = bool(identity(review) and review.get('reviewed') is True and review.get('source_or_numerics_changed') is False and
        review.get('reason_kind') in ('external_execution_failure', 'input_failure') and
        isinstance(review.get('reason'), str) and review['reason'].strip() and isinstance(review.get('previous_artifacts'), dict) and
        (not finalized or manifest.get('retry-review.json') == hashes.get('retry-review.json')))
    if artifact_names(path) != initial_files:
        findings.append({'code': 'artifact_names_changed_during_inspection'})
    eligible = bool(markers == ['COMPLETE.json'] and finalized and training_ok and checked and checked['status'] == 'verified' and not findings)
    return {'attempt': relative, 'terminal_markers': markers, 'terminal_partial_markers': partials, 'terminal_manifest_verified': finalized,
        'eligible': eligible, 'training_order_verified': training_ok, 'input_hashes': hashes, 'artifact_names': initial_files, 'findings': findings,
        'reported_metrics': reported, 'recomputed_rank_metrics': checked, 'result': result, 'failure': failure,
        'retry_review': review, 'retry_review_binding_verified': retry_verified}


def selection(attempts):
    attempts = sorted(attempts, key=lambda a: Path(a['attempt']).name)
    findings = []
    if attempts and Path(attempts[0]['attempt']).name != 'attempt-01':
        findings.append({'code': 'missing_first_attempt'})
    for index, current in enumerate(attempts[1:], 1):
        previous = attempts[index - 1]; review = current.get('retry_review')
        marker = 'FAILED.json' if 'FAILED.json' in previous['terminal_markers'] else 'started.json'
        expected = previous['input_hashes'].get(marker)
        valid = (Path(current['attempt']).name == f'attempt-{index+1:02d}' and 'COMPLETE.json' not in previous['terminal_markers'] and
            not previous.get('terminal_partial_markers') and
            current.get('retry_review_binding_verified') and isinstance(review, dict) and
            review.get('previous_attempt') == Path(previous['attempt']).name and expected is not None and
            isinstance(review.get('previous_artifacts'), dict) and review['previous_artifacts'].get(marker) == expected)
        if not valid:
            findings.append({'code': 'undocumented_or_invalid_retry', 'attempt': current['attempt']})
    complete_count = sum('COMPLETE.json' in a['terminal_markers'] for a in attempts)
    eligible = [a for a in attempts if a['eligible']]
    status = 'available' if len(eligible) == 1 else 'unavailable'
    if complete_count > 1:
        status = 'multiple_complete_attempts'
    elif findings:
        status = 'retry_history_unavailable'
    chosen = eligible[0] if status == 'available' else None
    scores = None
    if chosen:
        metrics = chosen['recomputed_rank_metrics']['recomputed']
        scores = {'original_overall': {m: metrics['overall'][m] for m in METRICS},
                  **{name: {m: values[m] for m in METRICS} for name, values in metrics['fixed_strata'].items()}}
    return {'status': status, 'attempt_count': len(attempts), 'complete_markers': complete_count,
            'verified_complete_attempts': len(eligible), 'selected_attempt': chosen['attempt'] if chosen else None,
            'retry_history_findings': findings, 'scores': scores}


def aggregate(values, required):
    available = [item for item in required if values[str(item)]['status'] == 'available']
    complete = len(available) == len(required) and bool(required)
    scores = [values[str(item)]['metrics'] for item in required] if complete else []
    return {'status': 'available' if complete else 'unavailable', 'required': required, 'available': available,
        'individual': values, 'mean': {m: statistics.mean(x[m] for x in scores) for m in METRICS} if complete else None,
        'sample_sd': {m: statistics.stdev(x[m] for x in scores) for m in METRICS} if complete and len(scores) > 1 else None,
        'minimum': {m: min(x[m] for x in scores) for m in METRICS} if complete else None,
        'maximum': {m: max(x[m] for x in scores) for m in METRICS} if complete else None, 'ddof': 1}


def descriptive(observations, protocol, strata):
    names = mask_names(protocol); views = ['original_overall', *strata]
    def value(mask, seed, view):
        observation = observations[key(mask, seed)]
        return {'status': observation['status'], 'metrics': observation['scores'][view] if observation['status'] == 'available' else None}
    individual = {f'{mask}/{view}': aggregate({str(seed): value(mask, seed, view) for seed in arm.SEEDS}, arm.SEEDS)
                  for mask in names for view in views}
    spread = {}
    for family in ('Rv', 'Rp'):
        family_names = [f'{family}-{seed}' for seed in protocol['selection_seeds'][family]]
        for view in views:
            spread[f'{family}/{view}'] = {str(seed): aggregate({name: value(name, seed, view) for name in family_names}, family_names)
                                        for seed in arm.SEEDS}
    contrasts = {}
    for view in strata:
        for label, left, right in (('C_minus_mean_Rv', 'C', 'Rv'), ('D_minus_mean_Rp', 'D', 'Rp'),
                                   ('C_minus_O', 'C', 'O'), ('D_minus_C', 'D', 'C')):
            values = {}
            operands = [left] + ([f'{right}-{s}' for s in protocol['selection_seeds'][right]] if right in ('Rv', 'Rp') else [right])
            for seed in arm.SEEDS:
                a = value(left, seed, view)
                b = spread[f'{right}/{view}'][str(seed)] if right in ('Rv', 'Rp') else value(right, seed, view)
                missing = [name for name in operands if observations[key(name, seed)]['status'] != 'available']
                other = b['mean'] if right in ('Rv', 'Rp') else b['metrics']
                values[str(seed)] = {'status': 'unavailable' if missing else 'available', 'missing_masks': missing,
                    'metrics': None if missing else {m: a['metrics'][m] - other[m] for m in METRICS}}
            contrasts[f'{view}/{label}'] = {'operands': operands, **aggregate(values, arm.SEEDS)}
    return {'metrics': list(METRICS), 'per_mask_five_seed': individual, 'within_seed_random_mask_spread': spread,
        'within_seed_contrasts': contrasts, 'direction': 'Positive rank/FOSCTTM differences and negative Recall differences mean worse retrieval.',
        'inference': 'Descriptive conditional variation only. Masks share sources; no independent-corpus, significance, equivalence or power claim.'}


def summarize(args):
    r.require(not args.out.exists() and not args.out.is_symlink(), 'Output exists; use a fresh path')
    bindings = r.read_json(args.bindings)
    r.require(set(bindings) == {'runner_bindings', 'attempts_root'} and all(isinstance(v, str) and Path(v).is_absolute() for v in bindings.values()), 'Private absolute path bindings required')
    bound = r.BoundFiles(); bound.add(__file__, r.digest(__file__)); bound.add(args.bindings, r.digest(args.bindings))
    data = arm.preflight(SimpleNamespace(protocol=args.protocol, protocol_sha256=args.protocol_sha256,
        masks=args.masks, masks_sha256=args.masks_sha256, bindings=Path(bindings['runner_bindings']), mask='O', seed=arm.SEEDS[0]), bound)
    bundle = r.read_json(args.masks)
    masks = arm.check_bundle(bundle, data.protocol, args.protocol_sha256, data.rows, data.api.masks)
    names = mask_names(data.protocol)
    tree, findings = inventory(bindings['attempts_root'], names)
    attempts, mappings, schedule = {}, {}, []
    for mask in names:
        for seed in arm.SEEDS:
            binding = {'schema_version': 1, 'protocol_id': arm.PROTOCOL_ID, 'protocol_sha256': args.protocol_sha256,
                       'masks_sha256': args.masks_sha256, 'mask': mask, 'seed': seed}
            orders = expected_orders(data.rows, masks[mask], seed, data.api)
            mappings[str(seed)] = {name: value.tolist() for name, value in orders['evaluation'].items()}
            schedule.append({'index': len(schedule), 'mask': mask, 'mask_sha256': masks[mask].mask_sha256, 'seed': seed,
                'training_order_sha256': {name: r.json_hash(value.tolist()) for name, value in orders['training'].items()}})
            attempts[key(mask, seed)] = [inspect_attempt(path, str(path.relative_to(bindings['attempts_root'])), binding,
                orders, data, masks[mask], bound) for path in tree[key(mask, seed)]]
    observations = {name: selection(values) for name, values in attempts.items()}
    if findings:
        for value in observations.values():
            value.update(status='invalid_inventory', scores=None, selected_attempt=None)
    result = {'schema_version': 1, 'protocol_id': arm.PROTOCOL_ID, 'recorded_at_utc': datetime.now(timezone.utc).isoformat(),
        'status': 'complete' if all(v['status'] == 'available' for v in observations.values()) else 'incomplete',
        'input_hashes': {'protocol': args.protocol_sha256, 'masks': args.masks_sha256, 'summarizer': r.digest(__file__),
            'runner': RUNNER_SHA256, 'private_bindings': r.digest(args.bindings), 'runner_bindings': r.digest(bindings['runner_bindings']),
            'prespecified_reporting': r.json_hash(data.protocol['research_design']['reporting'])},
        'schedule': schedule, 'planned_masks': len(names), 'planned_fits': len(schedule), 'inventory_findings': findings,
        'attempts': attempts, 'observations': observations, 'validation_order_and_coco_ids': mappings,
        'descriptive': descriptive(observations, data.protocol, list(data.strata)),
        'limitations': ['Saved ranks, higher/tie counts, metrics and original RNG orders are checked; fitted maps are not rescored.',
            'Retry reasons require human review. Inventory cannot prove that an operator never deleted or relocated an attempt.',
            'This measures fixed-pipeline source-removal behavior and does not establish encoder pretraining cleanliness.']}
    bound.recheck()
    new_tree, new_findings = inventory(bindings['attempts_root'], names)
    if new_tree != tree or new_findings != findings:
        findings.append({'code': 'inventory_changed_during_summary'})
    for name, paths in tree.items():
        for path, attempt in zip(paths, attempts[name]):
            if artifact_names(path) != attempt['artifact_names']:
                findings.append({'code': 'attempt_artifacts_changed_during_summary', 'attempt': attempt['attempt']})
    if findings:
        for value in observations.values():
            value.update(status='invalid_inventory', scores=None, selected_attempt=None)
        result['status'] = 'incomplete'
        result['descriptive'] = descriptive(observations, data.protocol, list(data.strata))
    r.no_local_paths(result); r.publish_marker(args.out, result)
    fd = os.open(args.out.parent, os.O_RDONLY)
    try: os.fsync(fd)
    finally: os.close(fd)
    return {'status': result['status'], 'summary_sha256': r.digest(args.out), 'planned_fits': len(schedule),
            'available_fits': sum(v['status'] == 'available' for v in observations.values()), 'fits_performed': 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('protocol', 'masks', 'bindings', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    for name in ('protocol-sha256', 'masks-sha256'):
        parser.add_argument('--' + name, required=True)
    print(json.dumps(summarize(parser.parse_args()), indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
