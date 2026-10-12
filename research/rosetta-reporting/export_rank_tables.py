#!/usr/bin/env python3
"""Presentation-only export of an accepted, complete frozen model-panel summary."""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import tempfile

import torch

ROWS = 40504
SEEDS = (734796314, 576165995, 2197670066, 839703249, 2584932063)
VISION = ('dinov2', 'dino', 'moco', 'mae')
TEXT = ('mpnet', 'glove', 'elmo')
METHODS = ('rosetta', 'orthogonal_paired', 'orthogonal_shuffled',
           'linear_paired', 'linear_shuffled')
ORDER_FIELDS = ('validation_image_ids', 'validation_indices', 'query_order',
                'key_order', 'partner_in_shuffled_keys')
PINNED = {
    'protocol': 'c9f9d1444e457d919ab7861fddc142290f06be76d2d5959c7ed5d9dfa2724a5a',
    'feature_lock': 'bb0ee134e0ba8e9d72b6c2f13a7bb1f0660f74b2da7af9cd09454185b3ef94d2',
    'summarizer': '790592ca38bbedc09557c4e3438718767f7771326009751b889e926f85f09d26',
    'prespecified_reporting': '9bc343397702bdb90d3b271e6265702281f40a387095a82810ca3dd86ebb4e2f',
}
HEADER = ('vision', 'text', 'seed', 'method', 'attempt', 'cell_terminal',
          'validation_subset_position', 'coco_image_id',
          'validation_source_row_index', 'midrank')


def need(value, message):
    if not value:
        raise ValueError(message)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def object_hash(value):
    return digest(json.dumps(value, separators=(',', ':'), allow_nan=False).encode())


def read_json(data):
    def pairs(items):
        result = {}
        for key, value in items:
            need(key not in result, 'Duplicate JSON key')
            result[key] = value
        return result

    def nonfinite(value):
        raise ValueError('Nonfinite JSON number')

    return json.loads(data, object_pairs_hook=pairs, parse_constant=nonfinite)


def pinned_bytes(path, expected):
    need(isinstance(expected, str) and re.fullmatch('[0-9a-f]{64}', expected),
         'Expected SHA-256 required')
    data = Path(path).read_bytes()
    need(digest(data) == expected, 'Input SHA-256 mismatch')
    return data


def validate_summary(summary):
    need(summary.get('schema_version') == 1 and
         summary.get('protocol_id') == 'rosetta-model-controls-v2', 'Unsupported summary identity')
    need(summary.get('status') == 'complete' and summary.get('inventory_findings') == [],
         'Complete verified reader summary required')
    need(summary.get('planned') == {'cells': 12, 'seeds': 5, 'method_slots': 300},
         'Panel plan differs')
    hashes = summary['input_hashes']
    need(set(hashes) == set(PINNED) | {'private_bindings', 'runner_bindings'} and
         all(isinstance(v, str) and re.fullmatch('[0-9a-f]{64}', v) for v in hashes.values()),
         'Reader input hash schema differs')
    need(all(hashes.get(k) == v for k, v in PINNED.items()),
         'Frozen reader/protocol/feature/reporting identity differs')
    gate = summary['baseline_replay_gate']
    need(gate.get('status') == 'satisfied' and
         gate.get('eligible_reference_cell_completions_by_seed') == {str(s): 1 for s in SEEDS},
         'Reference replay gate unavailable')
    cells = {f'{v}/{t}/{s}' for v in VISION for t in TEXT for s in SEEDS}
    slots = {f'{c}/{m}' for c in cells for m in METHODS}
    need(set(summary['attempts']) == cells and set(summary['observations']) == slots,
         'Missing or extra cell/method identities')
    need(all(o.get('status') == 'available' for o in summary['observations'].values()),
         'Unavailable method observation')
    need(set(summary['validation_order_and_coco_id_mapping']) == {str(s) for s in SEEDS},
         'Seed mappings differ')


def validate_mapping(mapping):
    need(set(mapping) == set(ORDER_FIELDS) | {'sort_positions_by_image_id', 'training_split_hashes'},
         'Mapping schema differs')
    for key in (*ORDER_FIELDS, 'sort_positions_by_image_id'):
        values = mapping[key]
        need(isinstance(values, list) and len(values) == ROWS and
             all(type(v) is int for v in values), 'Invalid mapping vector')
    ids = mapping['validation_image_ids']
    need(len(set(ids)) == ROWS and min(ids) >= 0, 'COCO IDs are not unique/nonnegative')
    for key in ('validation_indices', 'query_order', 'key_order', 'sort_positions_by_image_id'):
        need(sorted(mapping[key]) == list(range(ROWS)), 'Invalid complete-gallery permutation')
    inverse = [0] * ROWS
    for position, value in enumerate(mapping['key_order']):
        inverse[value] = position
    need(mapping['partner_in_shuffled_keys'] == [inverse[q] for q in mapping['query_order']],
         'Partner mapping differs')
    positions = sorted(range(ROWS), key=ids.__getitem__)
    need(positions == mapping['sort_positions_by_image_id'], 'ID sort mapping differs')
    return positions


def selected_method(summary, v, t, seed, method):
    cell = f'{v}/{t}/{seed}'
    observation = summary['observations'][f'{cell}/{method}']
    need(observation.get('retry_history_findings') == [], 'Retry history is unavailable')
    selected = observation.get('selected_attempt')
    need(isinstance(selected, str) and re.fullmatch(
         re.escape(f'{v}--{t}/seed-{seed}/') + r'attempt-(0[1-9]|[1-9][0-9])', selected),
         'Selected attempt identity differs')
    attempts = summary['attempts'][cell]
    need(len({a['attempt'] for a in attempts}) == len(attempts), 'Duplicate attempt identity')
    matches = [a for a in attempts if a['attempt'] == selected]
    need(len(matches) == 1, 'Selected attempt absent or ambiguous')
    attempt = matches[0]
    need(attempt.get('terminal_marker') in ('COMPLETE.json', 'FAILED.json') and
         attempt.get('terminal_manifest_verified') is True and attempt.get('split_verified') is True and
         attempt.get('terminal_partial_markers') == [] and attempt.get('findings') == [],
         'Selected attempt is incomplete or unverified')
    value = attempt['methods'][method]
    check = value.get('rank_check', {})
    need(value.get('status') == 'complete' and value.get('eligible') is True and
         value.get('findings') == [] and check.get('status') == 'verified' and
         check.get('order_ids_checked_against_original_rng') is True and
         check.get('gallery_count_matches') is True and check.get('validation_count') == ROWS and
         check.get('ranks_dtype') == 'float64' and check.get('midrank_step') == 0.5,
         'Method rank verification unavailable')
    need(observation.get('repeat_agreement') is None or observation.get('repeat_agreement') is True,
         'Repeated results conflict')
    for metric in ('foscttm', 'mean_rank'):
        score = observation['metrics'][metric]
        need(type(score) in (int, float) and math.isfinite(score) and
             score == value['reported_metrics'][metric] == check['recomputed_metrics'][metric] and
             check['comparisons'][metric]['exactly_equal'] is True,
             'Stored metric records disagree')
    sha = value.get('ranks_sha256')
    need(sha == attempt['input_hashes'].get(f'{method}/ranks.pt'), 'Rank hash bindings disagree')
    return selected, attempt, value, sha


def load_ranks(attempts_root, selected, method, expected_hash, mapping, positions, value):
    relative = Path(selected) / method / 'ranks.pt'
    path = attempts_root
    for part in relative.parts:
        path = path / part
        need(not path.is_symlink(), 'Symlinked rank artifact path rejected')
    need(path.resolve().is_relative_to(attempts_root), 'Rank artifact escapes input root')
    # Load the exact bytes just hashed; never deserialize a second, mutable read.
    data = pinned_bytes(path, expected_hash)
    record = torch.load(io.BytesIO(data), weights_only=True, map_location='cpu')
    need(isinstance(record, dict) and set(record) == set(ORDER_FIELDS) | {'ranks'},
         'Rank artifact schema differs')
    for key in ORDER_FIELDS:
        vector = record[key]
        need(isinstance(vector, torch.Tensor) and vector.device.type == 'cpu' and
             vector.dtype == torch.int64 and tuple(vector.shape) == (ROWS,) and
             vector.tolist() == mapping[key], 'Rank artifact ID/order mapping differs')
    ranks = record['ranks']
    need(isinstance(ranks, torch.Tensor) and ranks.device.type == 'cpu' and
         ranks.dtype == torch.float64 and tuple(ranks.shape) == (ROWS,), 'Rank dtype/shape differs')
    values = ranks.tolist()
    need(all(math.isfinite(q) and 1 <= q <= ROWS and q * 2 == round(q * 2) for q in values),
         'Invalid full-gallery midrank')
    need(object_hash([values[i] for i in positions]) ==
         value['rank_check']['canonical_ranks_by_coco_id_sha256'], 'Canonical rank hash differs')
    return values, relative.as_posix()


def export_tables(summary_path, summary_sha256, acceptance_path, acceptance_sha256,
                  attempts_root, out):
    summary = read_json(pinned_bytes(summary_path, summary_sha256))
    acceptance = read_json(pinned_bytes(acceptance_path, acceptance_sha256))
    need(acceptance == {'summary_sha256': summary_sha256}, 'Accepted reader receipt differs')
    validate_summary(summary)
    mappings = summary['validation_order_and_coco_id_mapping']
    positions = {seed: validate_mapping(mappings[str(seed)]) for seed in SEEDS}
    attempts_root = Path(attempts_root).resolve(strict=True)
    need(attempts_root.is_dir(), 'Attempt root is not a directory')
    out = Path(out).absolute()
    resolved_out = out.resolve()
    need(not resolved_out.is_relative_to(attempts_root) and
         not resolved_out.is_relative_to(Path(summary_path).resolve().parent) and
         not resolved_out.is_relative_to(Path(acceptance_path).resolve().parent),
         'Output must be outside the input artifact and receipt directories')
    need(not out.exists() and not out.is_symlink(), 'Output already exists; use a fresh directory')
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.rank-export-', dir=out.parent) as temporary:
        stage = Path(temporary) / 'export'
        stage.mkdir()
        files = []
        for v in VISION:
            for t in TEXT:
                for seed in SEEDS:
                    mapping = mappings[str(seed)]
                    order = positions[seed]
                    for method in METHODS:
                        selected, attempt, value, sha = selected_method(summary, v, t, seed, method)
                        ranks, relative = load_ranks(attempts_root, selected, method, sha, mapping, order, value)
                        filename = f'{v}--{t}/seed-{seed}/{method}.csv.gz'
                        target = stage / filename
                        target.parent.mkdir(parents=True, exist_ok=True)
                        with target.open('xb') as raw:
                            with gzip.GzipFile(filename='', mode='wb', fileobj=raw, mtime=0, compresslevel=6) as zipped:
                                with io.TextIOWrapper(zipped, encoding='utf-8', newline='') as text:
                                    writer = csv.writer(text, lineterminator='\n')
                                    writer.writerow(HEADER)
                                    for i in order:
                                        writer.writerow((v, t, seed, method, selected, attempt['terminal_marker'],
                                                         i, mapping['validation_image_ids'][i],
                                                         mapping['validation_indices'][i], repr(ranks[i])))
                        files.append({'file': filename, 'vision': v, 'text': t, 'seed': seed, 'method': method,
                                      'selected_attempt': selected, 'cell_terminal': attempt['terminal_marker'],
                                      'rank_artifact': relative, 'rank_artifact_sha256': sha,
                                      'canonical_ranks_by_coco_id_sha256': value['rank_check']['canonical_ranks_by_coco_id_sha256'],
                                      'rows': ROWS, 'bytes': target.stat().st_size,
                                      'sha256': digest(target.read_bytes())})
        manifest = {'schema_version': 1, 'status': 'complete', 'protocol_id': summary['protocol_id'],
                    'summary_sha256': summary_sha256, 'accepted_reader_receipt_sha256': acceptance_sha256,
                    'exporter_sha256': digest(Path(__file__).read_bytes()), 'reader_input_hashes': summary['input_hashes'],
                    'partition_count': len(files), 'rows_per_partition': ROWS, 'columns': list(HEADER),
                    'scope': 'Presentation of the accepted reader-selected ranks only. Rows are sorted by COCO image ID. Original validation-subset positions and rank values are retained. No refitting, similarity rescoring, metric reduction, or new scientific verification is performed.',
                    'files': files}
        (stage / 'manifest.json').write_text(json.dumps(manifest, indent=2, allow_nan=False) + '\n')
        # Reserve the final destination exclusively; publish the manifest last.
        # An interrupted copy leaves no complete manifest and cannot be reused.
        out.mkdir(exist_ok=False)
        for directory in sorted(p for p in stage.iterdir() if p.is_dir()):
            os.rename(directory, out / directory.name)
        os.rename(stage / 'manifest.json', out / 'manifest.json')
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('summary', 'acceptance', 'attempts-root', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    for name in ('summary-sha256', 'acceptance-sha256'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args()
    manifest = export_tables(args.summary, args.summary_sha256, args.acceptance,
                             args.acceptance_sha256, args.attempts_root, args.out)
    print(json.dumps({'status': manifest['status'], 'partition_count': manifest['partition_count'],
                      'rows_per_partition': manifest['rows_per_partition'],
                      'manifest_sha256': digest((args.out / 'manifest.json').read_bytes())}))


if __name__ == '__main__':
    main()
