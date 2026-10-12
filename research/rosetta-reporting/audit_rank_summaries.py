#!/usr/bin/env python3
"""Independent arithmetic comparison of accepted frozen Rosetta rank summaries.

No scientific module imports, fitting, rescoring, selection or numeric pass rule.
PyTorch is used only to deserialize CPU tensors and check their storage schema.
"""
from __future__ import annotations

import argparse
from collections import Counter
from decimal import Decimal, localcontext
from fractions import Fraction
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import sys

import torch

ROWS = 40504
PRECISION = 80
SEEDS = (734796314, 576165995, 2197670066, 839703249, 2584932063)
VISION = ('dinov2', 'dino', 'moco', 'mae')
TEXT = ('mpnet', 'glove', 'elmo')
METHODS = ('rosetta', 'orthogonal_paired', 'orthogonal_shuffled', 'linear_paired', 'linear_shuffled')
METRICS = ('foscttm', 'mean_rank')
SOURCE_METRICS = (*METRICS, 'median_rank', 'recall_at_1', 'recall_at_5', 'recall_at_10')
STRATA = {'never_known_spc_overlap': 37169, 'original_spc_overlap': 3335, 'all': ROWS}
MODEL_ORDER = ('validation_image_ids', 'validation_indices', 'query_order', 'key_order', 'partner_in_shuffled_keys')
SOURCE_ORDER = ('validation_order', 'query_order', 'key_order', 'canonical_coco_image_ids')
PINS = {
    'model': {'protocol': 'c9f9d1444e457d919ab7861fddc142290f06be76d2d5959c7ed5d9dfa2724a5a',
              'feature_lock': 'bb0ee134e0ba8e9d72b6c2f13a7bb1f0660f74b2da7af9cd09454185b3ef94d2',
              'summarizer': '790592ca38bbedc09557c4e3438718767f7771326009751b889e926f85f09d26'},
    'source': {'protocol': '75076af19cdeeefd3d839e9eb2db9a33af2c47e17a7a0600339763b95b90eeae',
               'masks': '1dd8f1fa7fd33207ecfe359643009352251e59820a1c12e936c75650e5f16ab6',
               'summarizer': '1c59ca05f15df720bb66160934a725837cc9953481f71781c39a76ad567dcb85',
               'runner': '6dbf7c3132a75607d5d7912759a88feba3ef943deeb89f3c6d81afcc045648af'},
}


def need(value, message):
    if not value:
        raise ValueError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def object_hash(value):
    return sha(json.dumps(value, separators=(',', ':'), allow_nan=False).encode())


def valid_sha(value):
    return isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value) is not None


def pinned(path, expected):
    need(valid_sha(expected), 'An externally recorded SHA-256 is required')
    data = Path(path).read_bytes()
    need(sha(data) == expected, 'Input bytes differ from the supplied SHA-256')
    return data


def read_json(data):
    def unique(items):
        result = {}
        for key, value in items:
            need(key not in result, 'Duplicate JSON key')
            result[key] = value
        return result
    def reject(value):
        raise ValueError('Nonfinite JSON literal')
    return json.loads(data, object_pairs_hook=unique, parse_constant=reject)


def fractions(metrics, names):
    need(isinstance(metrics, dict) and set(metrics) == set(names), 'Metric fields differ')
    need(all(type(v) in (int, float) and math.isfinite(v) for v in metrics.values()), 'Nonfinite/missing metric')
    return {name: Fraction(metrics[name]) for name in names}


def stats(values, names):
    """Exact rational mean/variance; independent 80-digit square root for SD."""
    need(len(values) > 1, 'A sample SD requires at least two prescribed observations')
    result = {name: {} for name in ('mean', 'sample_sd', 'minimum', 'maximum')}
    for metric in names:
        xs = [v[metric] for v in values]
        mean = sum(xs, Fraction()) / len(xs)
        variance = sum(((x - mean) ** 2 for x in xs), Fraction()) / (len(xs) - 1)
        with localcontext() as context:
            context.prec = PRECISION
            sd = (Decimal(variance.numerator) / Decimal(variance.denominator)).sqrt()
        result['mean'][metric] = mean
        result['sample_sd'][metric] = sd
        result['minimum'][metric] = min(xs)
        result['maximum'][metric] = max(xs)
    return result


def rank_metrics(ranks, higher=None, ties=None):
    """Integer/rational arithmetic, no tensor reductions or runner helpers."""
    count = len(ranks)
    need(count > 0, 'Empty prescribed stratum')
    twice = [int(value * 2) for value in ranks]
    total = sum(twice)
    result = {'foscttm': Fraction(total - 2 * count, 2 * count * (ROWS - 1)),
              'mean_rank': Fraction(total, 2 * count)}
    if higher is not None:
        need(len(higher) == len(ties) == count, 'Tie vector length differs')
        ordered = sorted(twice)
        result['median_rank'] = Fraction(ordered[(count - 1) // 2] + ordered[count // 2], 4)
        blocks = Counter(zip(higher, ties))
        for k in (1, 5, 10):
            result[f'recall_at_{k}'] = sum((multiplicity * Fraction(min(max(k - h, 0), t), t)
                for (h, t), multiplicity in blocks.items()), Fraction()) / count
    return result


def comparison(frozen, calculated, path):
    need(type(frozen) in (int, float) and math.isfinite(frozen), f'Missing/nonfinite scalar: {path}')
    if isinstance(calculated, Fraction):
        error = Fraction(frozen) - calculated
        precision = {'independent_exact_fraction': str(calculated), 'difference_exact_fraction': str(error)}
    else:
        with localcontext() as context:
            context.prec = PRECISION
            saved = Decimal(frozen) if type(frozen) is int else Decimal.from_float(frozen)
            error = saved - calculated
        precision = {'independent_decimal': str(calculated), 'difference_decimal': str(error),
                     'decimal_precision': PRECISION, 'decimal_is_approximate': True}
    return {'field': path, 'frozen': frozen, 'independent': float(calculated),
            'exactly_equal_at_reported_precision': error == 0,
            'binary64_exactly_equal': frozen == float(calculated),
            'signed_difference_frozen_minus_independent': float(error),
            'absolute_difference': float(abs(error) if isinstance(error, Fraction) else error.copy_abs()), **precision}


def compare_metrics(frozen, calculated, path, output):
    need(isinstance(frozen, dict) and set(frozen) == set(calculated), f'Metric coverage differs: {path}')
    for name, value in calculated.items():
        output.append(comparison(frozen[name], value, f'{path}/{name}'))


def vector(value, dtype):
    need(isinstance(value, torch.Tensor) and value.device.type == 'cpu' and
         value.dtype == dtype and tuple(value.shape) == (ROWS,), 'Tensor dtype/device/shape differs')
    return value.tolist()


def permutation(value):
    need(isinstance(value, list) and len(value) == ROWS and all(type(x) is int for x in value)
         and sorted(value) == list(range(ROWS)), 'Incomplete/noninteger permutation')


def mapping_check(mapping, study):
    fields = MODEL_ORDER if study == 'model' else SOURCE_ORDER
    need(set(mapping) == set(fields) | ({'sort_positions_by_image_id', 'training_split_hashes'} if study == 'model' else set()),
         'Mapping fields differ')
    for name in fields:
        need(isinstance(mapping[name], list) and len(mapping[name]) == ROWS and
             all(type(x) is int for x in mapping[name]), 'Invalid mapping vector')
    id_key = 'validation_image_ids' if study == 'model' else 'canonical_coco_image_ids'
    ids = mapping[id_key]
    need(len(set(ids)) == ROWS and min(ids) >= 0, 'Duplicate/invalid COCO IDs')
    for name in ('query_order', 'key_order', 'validation_indices' if study == 'model' else 'validation_order'):
        permutation(mapping[name])
    if study == 'model':
        permutation(mapping['sort_positions_by_image_id'])
        need(mapping['sort_positions_by_image_id'] == sorted(range(ROWS), key=ids.__getitem__), 'COCO sort mapping differs')
        inverse = [0] * ROWS
        for index, row in enumerate(mapping['key_order']):
            inverse[row] = index
        need(mapping['partner_in_shuffled_keys'] == [inverse[q] for q in mapping['query_order']], 'Partner mapping differs')
        canonical = [None] * ROWS
        for row, image_id in zip(mapping['validation_indices'], ids):
            canonical[row] = image_id
        return canonical
    return ids


def selected_attempt(summary, identity, selected, prefix):
    need(isinstance(selected, str) and re.fullmatch(re.escape(prefix) + r'/attempt-(0[1-9]|[1-9][0-9])', selected),
         'Selected attempt path differs')
    attempts = summary['attempts'][identity]
    need(isinstance(attempts, list) and len({a['attempt'] for a in attempts}) == len(attempts), 'Duplicate attempt identities')
    matches = [a for a in attempts if a['attempt'] == selected]
    need(len(matches) == 1, 'Selected attempt absent/ambiguous')
    chosen = matches[0]
    need(chosen.get('terminal_manifest_verified') is True and chosen.get('findings') == [] and
         chosen.get('terminal_partial_markers') == [], 'Selected attempt has unverified artifacts')
    return chosen


def rank_record(root, relative, expected_sha, mapping, study):
    target = root
    for part in Path(relative).parts:
        need(part not in ('..', '.') and not Path(part).is_absolute(), 'Unsafe artifact path')
        target = target / part
        need(not target.is_symlink(), 'Symlinked artifact path')
    need(target.resolve().is_relative_to(root), 'Artifact escapes root')
    record = torch.load(io.BytesIO(pinned(target, expected_sha)), weights_only=True, map_location='cpu')
    fields = MODEL_ORDER if study == 'model' else SOURCE_ORDER
    values = ('ranks',) if study == 'model' else ('ranks', 'num_higher', 'tie_count')
    need(isinstance(record, dict) and set(record) == set(fields) | set(values), 'Rank artifact fields differ')
    for field in fields:
        need(vector(record[field], torch.int64) == mapping[field], 'Rank artifact mapping differs from accepted summary')
    ranks = vector(record['ranks'], torch.float64)
    need(all(math.isfinite(x) and 1 <= x <= ROWS and 2*x == int(2*x) for x in ranks), 'Invalid full-gallery midranks')
    result = {'ranks': ranks}
    if study == 'source':
        higher, ties = vector(record['num_higher'], torch.int64), vector(record['tie_count'], torch.int64)
        need(all(h >= 0 and t >= 1 and h+t <= ROWS and 2*r == 2*h+t+1
                 for r, h, t in zip(ranks, higher, ties)), 'Ranks and exact tie counts disagree')
        result.update(num_higher=higher, tie_count=ties)
    return result


def compare_group(group, values, names, path, output, *, source=False, contrast=False):
    required = list(values)
    need(group.get('status') == 'available' and group.get('ddof') == 1, f'Unavailable aggregate: {path}')
    if source:
        need(group.get('required') == required and group.get('available') == required, f'Missing aggregate operand: {path}')
        individuals = group['individual']
    else:
        need(group.get('required_seeds') == list(SEEDS) and group.get('available_seeds') == list(SEEDS), f'Missing seed: {path}')
        individuals = group['per_seed']
    need(set(individuals) == {str(x) for x in required}, f'Individual coverage differs: {path}')
    for identity, metrics in values.items():
        item = individuals[str(identity)]
        need(item.get('status') == 'available', f'Unavailable individual: {path}')
        if contrast:
            need(item.get('missing_masks' if source else 'missing_entries') == [], f'Missing contrast operand: {path}')
        compare_metrics(item['metrics'], metrics, f'{path}/individual/{identity}', output)
    result = stats(list(values.values()), names)
    for name in ('mean', 'sample_sd', 'minimum', 'maximum') if source else ('mean', 'sample_sd'):
        compare_metrics(group[name], result[name], f'{path}/{name}', output)


def model_series(scores):
    series = {}
    for v in VISION:
        for t in TEXT:
            for method in METHODS:
                series[f'cell_scores/{v}/{t}/{method}'] = ({s: scores[f'{v}/{t}/{s}/{method}'] for s in SEEDS}, None)
    terms = {}
    for method in METHODS:
        for t in TEXT[1:]:
            terms[f'primary/text/{t}/{method}'] = [(1, 'dinov2', t, method), (-1, 'dinov2', 'mpnet', method)]
        for v in VISION[1:]:
            terms[f'primary/vision/{v}/{method}'] = [(1, v, 'mpnet', method), (-1, 'dinov2', 'mpnet', method)]
            for t in TEXT[1:]:
                terms[f'interaction/{v}/{t}/{method}'] = [(1, v, t, method), (-1, v, 'mpnet', method),
                    (-1, 'dinov2', t, method), (1, 'dinov2', 'mpnet', method)]
    controls = (('orthogonal_pairing', METHODS[1], METHODS[2]), ('linear_pairing', METHODS[3], METHODS[4]),
                ('rosetta_minus_orthogonal_paired', METHODS[0], METHODS[1]), ('rosetta_minus_linear_paired', METHODS[0], METHODS[3]))
    for v in VISION:
        for t in TEXT:
            for label, left, right in controls:
                terms[f'within/{v}/{t}/{label}'] = [(1, v, t, left), (-1, v, t, right)]
    for label, operands in terms.items():
        values = {s: {metric: sum((coefficient * scores[f'{v}/{t}/{s}/{method}'][metric]
                    for coefficient, v, t, method in operands), Fraction()) for metric in METRICS} for s in SEEDS}
        series[f'contrasts/{label}'] = (values, [{'coefficient': c, 'vision': v, 'text': t, 'method': m} for c,v,t,m in operands])
    return series


def compare_model_descriptive(descriptive, scores, output):
    need(descriptive.get('metrics') == list(METRICS), 'Model descriptive metric list differs')
    series = model_series(scores)
    for category in ('cell_scores', 'contrasts'):
        need(set(descriptive[category]) == {name.split('/', 1)[1] for name in series if name.startswith(category + '/')}, 'Model series coverage differs')
    for name, (values, terms) in series.items():
        category, label = name.split('/', 1)
        group = descriptive[category][label]
        if terms is not None:
            need(group.get('terms') == terms, f'Model contrast operands differ: {name}')
        compare_group(group, values, METRICS, name, output, contrast=terms is not None)


def source_series(scores, masks, families):
    views = ('original_overall', *STRATA)
    series = {}
    for mask in masks:
        for view in views:
            series[f'per_mask_five_seed/{mask}/{view}'] = ({s: scores[f'{mask}/{s}'][view] for s in SEEDS}, None)
    for family, names in families.items():
        for view in views:
            for seed in SEEDS:
                series[f'within_seed_random_mask_spread/{family}/{view}/{seed}'] = ({m: scores[f'{m}/{seed}'][view] for m in names}, None)
    for view in STRATA:
        for label, left, right in (('C_minus_mean_Rv', 'C', 'Rv'), ('D_minus_mean_Rp', 'D', 'Rp'),
                                  ('C_minus_O', 'C', 'O'), ('D_minus_C', 'D', 'C')):
            rights = families[right] if right in families else [right]
            values = {s: {metric: scores[f'{left}/{s}'][view][metric] -
                sum((scores[f'{r}/{s}'][view][metric] for r in rights), Fraction()) / len(rights)
                for metric in SOURCE_METRICS} for s in SEEDS}
            series[f'within_seed_contrasts/{view}/{label}'] = (values, [left, *rights])
    return series


def compare_source_descriptive(descriptive, scores, masks, families, output):
    need(descriptive.get('metrics') == list(SOURCE_METRICS), 'Source descriptive metrics differ')
    series = source_series(scores, masks, families)
    for category in ('per_mask_five_seed', 'within_seed_contrasts'):
        need(set(descriptive[category]) == {name.split('/', 1)[1] for name in series if name.startswith(category + '/')}, 'Source series coverage differs')
    expected_spreads = {f'{family}/{view}' for family in families for view in ('original_overall', *STRATA)}
    need(set(descriptive['within_seed_random_mask_spread']) == expected_spreads, 'Random-mask spread coverage differs')
    for group in descriptive['within_seed_random_mask_spread'].values():
        need(set(group) == {str(s) for s in SEEDS}, 'Random-mask spread seed coverage differs')
    for name, (values, operands) in series.items():
        category, label = name.split('/', 1)
        if category == 'within_seed_random_mask_spread':
            label, seed = label.rsplit('/', 1)
            group = descriptive[category][label][seed]
        else:
            group = descriptive[category][label]
        if operands is not None:
            need(group.get('operands') == operands, f'Source contrast operands differ: {name}')
        compare_group(group, values, SOURCE_METRICS, name, output, source=True, contrast=operands is not None)


def model_inputs(summary, root, output):
    cells = {f'{v}/{t}/{s}' for v in VISION for t in TEXT for s in SEEDS}
    slots = {f'{cell}/{method}' for cell in cells for method in METHODS}
    need(summary.get('planned') == {'cells': 12, 'seeds': 5, 'method_slots': 300} and
         set(summary['attempts']) == cells and set(summary['observations']) == slots, 'Model identity coverage differs')
    need(summary['baseline_replay_gate'].get('status') == 'satisfied' and
         summary['baseline_replay_gate'].get('eligible_reference_cell_completions_by_seed') == {str(s): 1 for s in SEEDS}, 'Reference gate unavailable')
    mappings = summary['validation_order_and_coco_id_mapping']
    canonical = check_mappings(mappings, 'model')
    saved, rebuilt, evidence = {}, {}, []
    for identity in sorted(slots):
        v, t, seed, method = identity.split('/')
        observation = summary['observations'][identity]
        need(observation.get('status') == 'available' and observation.get('retry_history_findings') == [] and
             observation.get('repeat_agreement') in (None, True), 'Model observation unavailable')
        selected = observation['selected_attempt']
        attempt = selected_attempt(summary, f'{v}/{t}/{seed}', selected, f'{v}--{t}/seed-{seed}')
        need(attempt.get('terminal_marker') in ('COMPLETE.json', 'FAILED.json') and attempt.get('split_verified') is True, 'Model attempt unverified')
        value = attempt['methods'][method]; check = value['rank_check']
        need(value.get('status') == 'complete' and value.get('eligible') is True and value.get('findings') == [] and
             check.get('status') == 'verified' and check.get('order_ids_checked_against_original_rng') is True and
             check.get('gallery_count_matches') is True and check.get('validation_count') == ROWS, 'Model rank acceptance absent')
        saved[identity] = fractions(observation['metrics'], METRICS)
        need(all(observation['metrics'][m] == value['reported_metrics'][m] == check['recomputed_metrics'][m]
                 and check['comparisons'][m]['exactly_equal'] is True for m in METRICS), 'Frozen model scalar bindings differ')
        rank_sha = value['ranks_sha256']
        relative = f'{selected}/{method}/ranks.pt'
        need(rank_sha == attempt['input_hashes'][f'{method}/ranks.pt'], 'Model rank hashes disagree')
        record = rank_record(root, relative, rank_sha, mappings[seed], 'model')
        need(object_hash([record['ranks'][i] for i in mappings[seed]['sort_positions_by_image_id']]) ==
             check['canonical_ranks_by_coco_id_sha256'], 'Model canonical rank hash differs')
        rebuilt[identity] = rank_metrics(record['ranks'])
        compare_metrics(observation['metrics'], rebuilt[identity], identity, output)
        evidence.append({'identity': identity, 'selected_attempt': selected, 'terminal_marker': attempt['terminal_marker'], 'ranks_sha256': rank_sha})
    return saved, rebuilt, evidence, canonical


def check_mappings(mappings, study):
    need(set(mappings) == {str(s) for s in SEEDS}, 'Mapping seed coverage differs')
    canonical = [mapping_check(mappings[str(s)], study) for s in SEEDS]
    need(all(ids == canonical[0] for ids in canonical), 'Canonical COCO IDs differ between seeds')
    return object_hash(canonical[0])


def source_inputs(summary, root, bundle, protocol, output):
    families = {f: [f'{f}-{s}' for s in protocol['selection_seeds'][f]] for f in ('Rv', 'Rp')}
    masks = ['O', 'C', 'D', *families['Rv'], *families['Rp']]
    need(len(masks) == len(set(masks)) == 23 and set(bundle['masks']) == set(masks), 'Source mask coverage differs')
    need(set(bundle['strata']) == set(STRATA), 'Fixed stratum coverage differs')
    for name, count in STRATA.items():
        rows = bundle['strata'][name]
        need(isinstance(rows, list) and all(type(i) is int for i in rows) and len(rows) == count and
             rows == sorted(set(rows)) and all(0 <= i < ROWS for i in rows), 'Fixed stratum indices differ')
    a, b = (set(bundle['strata'][name]) for name in ('never_known_spc_overlap', 'original_spc_overlap'))
    need(not a & b and a | b == set(range(ROWS)) and bundle['strata']['all'] == list(range(ROWS)), 'Fixed strata do not partition the gallery')
    keys = [f'{mask}/{s}' for mask in masks for s in SEEDS]
    need(set(summary['observations']) == set(keys) == set(summary['attempts']) and
         summary.get('planned_masks') == 23 and summary.get('planned_fits') == 115, 'Source fit coverage differs')
    schedule = summary['schedule']
    need(len(schedule) == len(keys), 'Source schedule incomplete')
    for index, (item, identity) in enumerate(zip(schedule, keys)):
        mask, seed = identity.split('/')
        need(item.get('index') == index and item.get('mask') == mask and item.get('seed') == int(seed) and
             item.get('mask_sha256') == bundle['masks'][mask]['mask_sha256'], 'Source schedule identity/mask binding differs')
    mappings = summary['validation_order_and_coco_ids']; canonical = check_mappings(mappings, 'source')
    saved, rebuilt, evidence = {}, {}, []
    for identity in keys:
        mask, seed = identity.split('/'); observation = summary['observations'][identity]
        need(observation.get('status') == 'available' and observation.get('retry_history_findings') == [] and
             observation.get('complete_markers') == observation.get('verified_complete_attempts') == 1, 'Source observation unavailable')
        selected = observation['selected_attempt']; attempt = selected_attempt(summary, identity, selected, f'{mask}/seed-{seed}')
        check = attempt['recomputed_rank_metrics']
        need(attempt.get('terminal_markers') == ['COMPLETE.json'] and attempt.get('eligible') is True and
             attempt.get('training_order_verified') is True and check.get('status') == 'verified' and
             check.get('differences') == [] and check.get('original_evaluation_order_and_coco_ids_checked') is True, 'Source rank acceptance absent')
        rank_sha = attempt['input_hashes']['ranks.pt']
        need(rank_sha == attempt['result']['artifact_hashes']['ranks.pt'], 'Source rank hashes disagree')
        record = rank_record(root, f'{selected}/ranks.pt', rank_sha, mappings[seed], 'source')
        need(object_hash(record) == check['canonical_ranks_higher_ties_sha256'], 'Source canonical rank hash differs')
        reported = attempt['reported_metrics']
        need(reported == check['recomputed'] and reported.get('rank_order') == 'canonical_original_file_row' and
             reported.get('overall_reduction_order') == 'original_randomized_validation_order' and
             reported.get('tie_policy') == 'expected_credit_under_uniform_order_within_exact_score_tie_block', 'Source frozen metrics/order bindings differ')
        views = {'original_overall': mappings[seed]['validation_order'], **bundle['strata']}
        need(set(observation['scores']) == set(views) and set(reported['fixed_strata']) == set(STRATA), 'Source view coverage differs')
        saved[identity], rebuilt[identity] = {}, {}
        for view, indices in views.items():
            metrics = reported['overall'] if view == 'original_overall' else reported['fixed_strata'][view]
            need(set(metrics) == set(SOURCE_METRICS) | {'num_queries', 'gallery_size', 'foscttm_denominator'} and
                 metrics['num_queries'] == len(indices) and metrics['gallery_size'] == ROWS and
                 metrics['foscttm_denominator'] == ROWS-1, 'Source query/gallery counts differ')
            need(observation['scores'][view] == {m: metrics[m] for m in SOURCE_METRICS}, 'Source saved score binding differs')
            saved[identity][view] = fractions(observation['scores'][view], SOURCE_METRICS)
            rebuilt[identity][view] = rank_metrics(*([record[k][i] for i in indices] for k in ('ranks', 'num_higher', 'tie_count')))
            compare_metrics(observation['scores'][view], rebuilt[identity][view], f'{identity}/{view}', output)
        need(reported['original_overall'] == {m: reported['overall'][m] for m in METRICS}, 'Source original-overall binding differs')
        evidence.append({'identity': identity, 'selected_attempt': selected, 'terminal_marker': 'COMPLETE.json', 'ranks_sha256': rank_sha})
    return saved, rebuilt, evidence, canonical, masks, families


def audit(args):
    pins = PINS[args.study]
    need(args.protocol_sha256 == pins['protocol'], 'Unsupported frozen protocol')
    protocol = read_json(pinned(args.protocol, args.protocol_sha256))
    summary = read_json(pinned(args.summary, args.summary_sha256))
    accepted = read_json(pinned(args.acceptance, args.acceptance_sha256))
    study_receipt = read_json(pinned(args.study_receipt, args.study_receipt_sha256))
    need(accepted == {'summary_sha256': args.summary_sha256}, 'Reader acceptance does not bind this summary')
    receipt_name = Path(args.study_receipt).name
    need(receipt_name in (f'{args.study}-stage-complete.json', f'final-audit-{args.study}.json'),
         'A completed-study or final-audit receipt is required, not preflight')
    need(Path(args.summary).name == 'summary.json' and Path(args.acceptance).name == 'accepted.json' and
         Path(args.summary).resolve().parent == Path(args.acceptance).resolve().parent,
         'Keep the accepted summary and receipt together under their original filenames')
    need(set(study_receipt) == {'summary_path', 'summary_sha256', 'verification', 'ticket_sha256'} and
         study_receipt['summary_sha256'] == args.summary_sha256 and valid_sha(study_receipt['ticket_sha256']) and
         Path(study_receipt['summary_path']).name == 'summary.json' and
         Path(study_receipt['summary_path']).parent == Path(study_receipt['verification']),
         'Controller study receipt does not bind this summary')
    # The ticket is already bound by the controller receipt's canonical hash.
    # Compare recorded paths internally, so a byte-preserving local copy works.
    ticket_path = Path(args.acceptance).parent / 'ticket.json'
    ticket_data = ticket_path.read_bytes(); ticket = read_json(ticket_data)
    need(object_hash(ticket) == study_receipt['ticket_sha256'] and ticket.get('study') == args.study and
         ticket.get('preflight') is False and ticket.get('out') == study_receipt['summary_path'] and
         bool(ticket.get('final_audit')) == receipt_name.startswith('final-audit-'),
         'Controller reader ticket binding/type differs')
    expected_id = 'rosetta-model-controls-v2' if args.study == 'model' else 'rosetta-source-controls-v1'
    need(protocol.get('protocol_id') == summary.get('protocol_id') == expected_id and summary.get('schema_version') == 1 and
         protocol.get('seeds') == list(SEEDS) and protocol['counts']['validation'] == ROWS and
         summary.get('status') == 'complete' and summary.get('inventory_findings') == [], 'Complete frozen study required')
    hashes = summary['input_hashes']
    need(set(hashes) == set(pins) | {'prespecified_reporting', 'private_bindings', 'runner_bindings'} and
         all(valid_sha(v) for v in hashes.values()) and all(hashes.get(k) == v for k,v in pins.items()) and
         hashes['prespecified_reporting'] == object_hash(protocol['research_design']['reporting']), 'Frozen reader input bindings differ')
    root = Path(args.attempts_root).resolve(strict=True)
    need(root.is_dir(), 'Attempt root must be a directory')
    sections = {'rank_metrics': [], 'aggregates_from_saved_scalars': [], 'aggregates_from_reconstructed_metrics': []}
    if args.study == 'model':
        need(args.masks is None and args.masks_sha256 is None, 'Masks are only valid for source study')
        saved, rebuilt, evidence, canonical = model_inputs(summary, root, sections['rank_metrics'])
        for scores, name in ((saved, 'aggregates_from_saved_scalars'), (rebuilt, 'aggregates_from_reconstructed_metrics')):
            compare_model_descriptive(summary['descriptive'], scores, sections[name])
        coverage = {'scheduled_cells': 60, 'method_observations': 300, 'five_seed_series': 60, 'contrast_series': 103}
    else:
        need(args.masks is not None and args.masks_sha256 == pins['masks'], 'Pinned source masks required')
        bundle = read_json(pinned(args.masks, args.masks_sha256))
        need(bundle.get('protocol_sha256') == args.protocol_sha256 and bundle.get('protocol_id') == expected_id and
             bundle.get('row_identity_sha256') == protocol['row_identity_sha256'], 'Source mask bundle identity differs')
        saved, rebuilt, evidence, canonical, masks, families = source_inputs(summary, root, bundle, protocol, sections['rank_metrics'])
        for scores, name in ((saved, 'aggregates_from_saved_scalars'), (rebuilt, 'aggregates_from_reconstructed_metrics')):
            compare_source_descriptive(summary['descriptive'], scores, masks, families, sections[name])
        coverage = {'scheduled_fits': 115, 'masks': 23, 'five_seed_series': 92, 'random_mask_spread_series': 40, 'contrast_series': 12}
    result = {'schema_version': 1, 'status': 'comparison_generated', 'study': args.study,
        'scientific_acceptance': 'not_assessed', 'numeric_tolerance': None,
        'inputs': {'summary_sha256': args.summary_sha256, 'acceptance_sha256': args.acceptance_sha256,
                   'study_receipt_sha256': args.study_receipt_sha256, 'protocol_sha256': args.protocol_sha256,
                   'ticket_sha256': sha(ticket_data), 'ticket_canonical_sha256': study_receipt['ticket_sha256'],
                   'masks_sha256': args.masks_sha256, 'audit_code_sha256': sha(Path(__file__).read_bytes()),
                   'accepted_reader_input_hashes': hashes, 'canonical_coco_ids_sha256': canonical},
        'coverage': coverage, 'selected_artifacts': evidence,
        'arithmetic': {'means_and_contrasts': 'Exact rational arithmetic; saved binary64 inputs converted exactly, reconstructed metrics remain exact rationals.',
            'sample_sd': 'Exact rational centered sample variance with ddof=1; Decimal square root at 80 significant digits.',
            'rank_metrics': 'Integer doubled midranks; rational FOSCTTM/means/median and expected tie-block Recall credit.',
            'frozen_order': 'Model: saved validation-subset order. Source original_overall: validation_order. Source fixed strata: canonical file rows.',
            'independent_order': 'Exact rational sums are order independent; IDs are never sorted for scientific reductions.',
            'differences': 'Frozen minus independent. No threshold or scientific verdict. Decimal SD/difference strings are finite-precision approximations.'},
        'comparison_counts': {name: {'scalars': len(rows), 'binary64_differences': sum(not r['binary64_exactly_equal'] for r in rows),
                                   'nonzero_exact_or_decimal_differences': sum(not r['exactly_equal_at_reported_precision'] for r in rows)}
                              for name, rows in sections.items()}, 'comparisons': sections,
        'limitations': ['External evidence hashes and controller receipt provenance must be trusted and reviewed independently.',
            'Eligibility, retries, RNG, services, feature identities and masks remain the frozen readers/controller responsibility.',
            'No map rescoring, similarity recomputation, fit or author-value reproduction. Numeric differences are observations requiring review.',
            'This report does not certify the controller stage, scientific correctness, significance or interpretation.',
            'Implementation has synthetic-only validation until a separately documented real-output review occurs.']}
    out = Path(args.out).absolute()
    input_files = [args.summary, args.acceptance, args.study_receipt, args.protocol, ticket_path] + ([args.masks] if args.masks else [])
    need(not out.exists() and not out.is_symlink() and not out.resolve().is_relative_to(root) and
         not any(out.resolve().is_relative_to(Path(p).resolve().parent) for p in input_files), 'Use a fresh output outside all input directories')
    payload = (json.dumps(result, indent=2, allow_nan=False) + '\n').encode()
    out.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation: an interrupted file is not a valid JSON report.
    with out.open('xb') as handle:
        handle.write(payload); handle.flush(); os.fsync(handle.fileno())
    return {'status': result['status'], 'report_sha256': sha(payload), 'coverage': coverage,
            'comparison_counts': result['comparison_counts'], 'scientific_acceptance': 'not_assessed'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', choices=('model', 'source'), required=True)
    for name in ('summary', 'acceptance', 'study-receipt', 'protocol'):
        parser.add_argument('--' + name, type=Path, required=True)
        parser.add_argument('--' + name + '-sha256', required=True)
    parser.add_argument('--masks', type=Path)
    parser.add_argument('--masks-sha256')
    parser.add_argument('--attempts-root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    try:
        print(json.dumps(audit(parser.parse_args()), allow_nan=False))
    except (ValueError, KeyError, TypeError, OSError, RuntimeError) as error:
        # Errors may include local paths; stderr stays private, never in the report.
        print(f'Input rejected: {error}', file=sys.stderr)
        return 2
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
