#!/usr/bin/env python3
"""Bind the five completed original results without authorizing new experiments.

Use the untouched reproduction release and its completed supervisor status,
raw results and frozen comparison report. Machine paths and raw host metadata
stay private. Numerical mismatches remain in the comparison; they are not a
reason to discard a seed or replace its result. This does not check process
quiescence or re-audit the current full input/runtime installation.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys


RELEASE_SHA256 = '3d52418723d5da4b3a208f5f2078656b35e508a322c555d07d7aacc822da9c36'
MODULE_SHA256 = {
    'run_locked': '64c02089a63bc7d8a77cc1014cc4ff4e0a35ae2fcd5b980fc954a7e22ef8f151',
    'compare': '565d371e862020fb334d91dad189691f8f1a9332a4a46d8c6d82aaad983662a0',
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, 'Duplicate JSON key')
            result[key] = value
        return result
    def invalid(_):
        raise ValueError('Nonfinite JSON value')
    return json.loads(Path(path).read_text(), object_pairs_hook=unique, parse_constant=invalid)


def frozen_api(release):
    release = Path(release).resolve(strict=True)
    require(digest(release / 'release-lock.json') == RELEASE_SHA256, 'Unexpected original release')
    modules = {}
    for name, expected in MODULE_SHA256.items():
        path = release / (name + '.py')
        require(digest(path) == expected, 'Original comparison utility changed')
        if name == 'run_locked' and name in sys.modules:
            require(Path(sys.modules[name].__file__).resolve() == path, 'Another original runner is already imported')
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        if name == 'run_locked':
            sys.modules[name] = module
        spec.loader.exec_module(module)
        modules[name] = module
    state = modules['run_locked'].release_state(release)
    require(state['release_lock_sha256'] == RELEASE_SHA256, 'Original release binding differs')
    return modules['run_locked'], modules['compare'], state


def assemble(release, results, status_path, comparison_path):
    release, results = Path(release).resolve(strict=True), Path(results).resolve(strict=True)
    runner, comparison, release_state = frozen_api(release)
    tracked = {}
    def tracked_json(path):
        path = Path(path).resolve(strict=True)
        tracked[path] = digest(path)
        return read_json(path)
    status = tracked_json(status_path)
    report = tracked_json(comparison_path)
    protocol = tracked_json(release / 'protocol.json')
    reference = tracked_json(release / 'reference.json')
    source_lock = tracked_json(release / 'source-lock.json')
    inputs_lock = tracked_json(release / 'inputs-lock.json')
    require(status.get('schema_version') == 1 and status.get('status') == 'complete' and
            type(status.get('child_returncode')) is int and status['child_returncode'] == 0 and
            status.get('signals_received') == [], 'Original supervisor must have completed without interruption')
    require(status.get('release_before') == status.get('release_after') == release_state,
            'Supervisor release differs from the untouched release')
    for name in ('source', 'inputs'):
        require(isinstance(status.get(name + '_before'), dict) and bool(status[name + '_before']) and
                status[name + '_before'] == status.get(name + '_after'), 'Recorded baseline inputs or source changed')
    source = status['source_before']
    require(source.get('source_files_sha256') == source_lock['files'] and
            source.get('pixi_binary_sha256') == source_lock['pixi_binary_sha256'] and
            source.get('upstream_commit') == protocol['upstream_commit'] and
            source.get('bootstrap_receipt_sha256') == protocol['bootstrap_receipt_sha256'],
            'Recorded baseline source does not match the original locks')
    expected_inputs = {entry['path']: {key: entry[key] for key in ('sha256', 'size_bytes')}
                       for entry in inputs_lock['files']}
    require(status['inputs_before'] == expected_inputs, 'Recorded baseline inputs do not match the original lock')
    initial_names = sorted(p.name for p in results.iterdir())
    records = comparison.load_results(results, protocol)
    names = [runner.expected_result_name(c) for c in runner.expected_configurations(protocol)]
    require(status.get('expected_result_files') == names, 'Supervisor scheduled different original results')
    measured = {}
    expected_status = []
    for record in records:
        path = results / (record['result_id'] + '.json')
        raw = tracked_json(path)
        require(tracked[path] == record['raw_result_sha256'], 'Result changed while reading')
        measured[str(record['seed'])] = {'sha256': tracked[path], 'size_bytes': path.stat().st_size,
                                        'file': path.name}
        expected_status.append({'file': path.name, 'sha256': tracked[path], 'seed': record['seed'],
                                'metrics': raw['metrics']})
    require(status.get('results') == sorted(expected_status, key=lambda item: item['file']),
            'Supervisor result inventory differs from the raw files')
    expected_report = {
        'schema_version': 1, 'protocol_id': protocol['protocol_id'],
        'release_lock_sha256': release_state['release_lock_sha256'],
        'protocol_sha256': release_state['files']['protocol.json'],
        'reference_sha256': release_state['files']['reference.json'],
        'comparison_script_sha256': release_state['files']['compare.py'],
        **comparison.compare(protocol, reference, records),
    }
    require(report == expected_report, 'Comparison report differs from frozen recomputation')
    lock = {
        'schema_version': 1, 'status': 'complete',
        'baseline_protocol_sha256': release_state['files']['protocol.json'],
        'baseline_release_lock_sha256': RELEASE_SHA256,
        'supervisor_status_sha256': tracked[Path(status_path).resolve()],
        'comparison_sha256': tracked[Path(comparison_path).resolve()],
        'builder_sha256': digest(__file__),
        'results': measured,
        'scope': 'Five original completed results and their recomputed frozen comparison. This lock records evidence; it does not authorize extraction or fitting or establish whole-paper replication.',
        'mismatch_policy': 'Keep every discrepancy. Any unexplained mismatch must be investigated before the separate model-control release.',
        'checks_not_performed': ['Live-process quiescence', 'Current full input/runtime re-audit',
                                 'Independent scientific interpretation of differences'],
    }
    require(runner.release_state(release) == release_state, 'Original release changed during binding')
    require(comparison.load_results(results, protocol) == records, 'Original results changed during binding')
    require(sorted(p.name for p in results.iterdir()) == initial_names, 'Result inventory changed during binding')
    for path, expected in tracked.items():
        require(digest(path) == expected, 'A baseline record changed during binding')
    return lock


def publish(path, value):
    path = Path(path)
    partial = path.with_name(path.name + '.partial')
    require(not path.exists() and not path.is_symlink() and not partial.exists() and not partial.is_symlink(),
            'Preserve existing output; choose a new path')
    with partial.open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    os.link(partial, path)
    partial.unlink()
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('release', 'results', 'status', 'comparison', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    lock = assemble(args.release, args.results, args.status, args.comparison)
    publish(args.out, lock)
    print(json.dumps({'status': 'baseline_evidence_bound', 'results': len(lock['results']),
                      'sha256': digest(args.out), 'new_experiments_authorized': False}))


if __name__ == '__main__':
    main()
