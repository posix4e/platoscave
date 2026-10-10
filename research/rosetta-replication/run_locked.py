#!/usr/bin/env python3
"""Supervise the unchanged authors' CLI; never import or patch its fitting code.

The release and successful preflight must be frozen before this command starts.
An existing run directory is a hard stop: failed/interrupted runs are preserved,
not silently resumed or replaced. The authors' CLI can swallow exceptions, so
its exit code alone is never accepted as proof of completion.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone


def utc():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def value_digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def read_json(path):
    with Path(path).open() as stream:
        return json.load(stream)


def write_json(path, value):
    path = Path(path)
    temp = path.with_name(path.name + '.tmp')
    with temp.open('w') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    temp.replace(path)


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def relative(base, name):
    p = Path(name)
    require(not p.is_absolute() and '..' not in p.parts and p.parts, f'Invalid relative path: {name}')
    return Path(base) / p


def release_state(release):
    """Verify the closed set of named release files against its recorded lock."""
    release = Path(release)
    lock = read_json(release / 'release-lock.json')
    files = lock['files']
    require(isinstance(files, dict) and files, 'Release lock files must be a nonempty dictionary')
    for name in ('protocol.json', 'inputs-lock.json', 'source-lock.json', 'environment.json', 'reference.json', 'upstream-tests.json', 'preflight.py', 'run_locked.py'):
        require(name in files, f'Release lock omits {name}')
    measured = {}
    for name, expected in files.items():
        require(isinstance(expected, str) and len(expected) == 64, f'Invalid release hash for {name}')
        measured[name] = digest(relative(release, name))
        require(measured[name] == expected, f'Release file changed: {name}')
    return {'release_lock_sha256': digest(release / 'release-lock.json'), 'files': measured}


def input_state(root, release, protocol):
    files = read_json(Path(release) / 'inputs-lock.json')['files']
    require(len(files) == protocol['expected_input_count'] == 13, 'Expected all 13 original input files')
    names = [f['path'] for f in files]
    require(len(set(names)) == len(names), 'Duplicate input paths')
    require(set(names) == set(protocol['expected_shapes']) == set(protocol['expected_dtypes']),
            'Input set differs from protocol shape/dtype declarations')
    measured = {}
    for record in files:
        path = relative(Path(root) / 'storage', record['path'])
        size = path.stat().st_size
        require(size == record['size_bytes'], f'Input byte size changed: {record["path"]}')
        sha = digest(path)
        require(sha == record['sha256'], f'Input hash changed: {record["path"]}')
        measured[record['path']] = {'sha256': sha, 'size_bytes': size}
    return measured


def source_state(root, protocol):
    root = Path(root)
    receipt_path = root / 'bootstrap' / 'bootstrap-receipt.json'
    receipt_sha = digest(receipt_path)
    require(receipt_sha == protocol['bootstrap_receipt_sha256'], 'Bootstrap receipt differs from frozen protocol')
    receipt = read_json(receipt_path)
    require(receipt['upstream_commit'] == protocol['upstream_commit'], 'Unexpected original source commit')
    originals = receipt['source_files_sha256']
    require(len(originals) == protocol['expected_source_file_count'] == 228, 'Expected all 228 authors source files')
    measured = {}
    for name, expected in originals.items():
        measured[name] = digest(relative(root / 'source', name))
        require(measured[name] == expected, f'Authors source file changed: {name}')
    pixi_sha = digest(root / 'bootstrap' / 'pixi')
    require(pixi_sha == receipt['pixi_binary_sha256'], 'Pixi executable changed')
    require(measured['pixi.lock'] == receipt['lock_sha256_before_install'], 'Original dependency lock changed')
    git_commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root / 'source', text=True).strip()
    require(git_commit == protocol['upstream_commit'], 'Source checkout HEAD is not the authors pinned commit')
    return {'bootstrap_receipt_sha256': receipt_sha, 'upstream_commit': git_commit,
            'pixi_binary_sha256': pixi_sha, 'source_files_sha256': measured}


def expected_configurations(protocol):
    configs = [{**protocol['configuration_except_seed'], 'seed': seed} for seed in protocol['seeds']]
    require(len(configs) == 5 and len(set(protocol['seeds'])) == 5, 'Exactly five distinct official seeds are required')
    return configs


def expected_result_name(config):
    # Exact serialization in unpaired_rosetta/results.py at the frozen commit.
    payload = json.dumps({'type': 'AlignmentRun', **config}, sort_keys=True, default=str)
    return hashlib.sha1(payload.encode()).hexdigest()[:16] + '.json'


def result_records(root, protocol, require_complete):
    folder = Path(root) / 'storage' / 'results' / 'unpaired'
    expected = {expected_result_name(c): c for c in expected_configurations(protocol)}
    paths = list(folder.glob('*.json')) if folder.exists() else []
    require(not (set(p.name for p in paths) - set(expected)), 'Unexpected result JSONs are present')
    if require_complete:
        require(set(p.name for p in paths) == set(expected), 'The authors command did not produce all five results')
        require(not list(folder.glob('*.tmp')), 'Unfinished temporary result files remain')
    records = []
    for path in sorted(paths):
        record = read_json(path)
        config = record['config']
        require(canonical(config) == canonical(expected[path.name]), f'Result configuration mismatch: {path.name}')
        require(record['git_commit'] == protocol['upstream_commit'], f'Result source commit mismatch: {path.name}')
        metrics = record['metrics']
        for metric in protocol['required_metrics']:
            value = metrics.get(metric)
            require(isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value),
                    f'Missing/nonfinite result metric {metric}: {path.name}')
        n = protocol['expected_validation_count']
        require(metrics['num_validation_samples'] == n, 'Result does not use the complete validation set')
        require(0 <= metrics['foscttm'] <= 1 and 1 <= metrics['mean_rank'] <= n, 'Invalid retrieval metric range')
        require(abs(metrics['foscttm'] - (metrics['mean_rank'] - 1) / (n - 1)) <= 1e-12,
                'FOSCTTM and mean rank disagree')
        for metric, value in metrics.items():
            if metric.endswith(('_top1', '_top5')):
                require(isinstance(value, (int, float)) and math.isfinite(value) and 0 <= value <= 1,
                        f'Invalid classification value: {metric}')
        require(metrics['fit_and_evaluation_seconds'] >= 0, 'Invalid duration')
        for name in ('torch', 'numpy', 'scipy', 'scikit-learn', 'pylibmgm'):
            require(record['versions'].get(name) == protocol['expected_versions'][name],
                    f'Result dependency version differs or is missing: {name}')
        records.append({'file': path.name, 'sha256': digest(path), 'seed': config['seed'], 'metrics': metrics})
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--release', type=Path, required=True)
    args = parser.parse_args()
    root, release = args.root.resolve(), args.release.resolve()
    require(Path(__file__).resolve() == release / 'run_locked.py', 'Run the script inside the frozen release directory')
    before_release = release_state(release)
    protocol = read_json(release / 'protocol.json')
    preflight_path = root / 'preflight.json'
    preflight = read_json(preflight_path)
    require(preflight.get('status') == 'passed', 'A successful preflight is required')
    require(preflight['release'] == before_release, 'Preflight belongs to a different release')
    require(preflight['root'] == str(root), 'Preflight belongs to a different execution root')
    require(preflight['expected_configurations'] == expected_configurations(protocol), 'Preflight configurations changed')
    require(preflight['versions'] == protocol['expected_versions'], 'Preflight versions differ from protocol')
    require(preflight['activation'] == protocol['expected_activation'], 'Preflight activation differs from protocol')
    require(digest(root / 'source' / '.pixi' / 'envs' / 'default' / 'bin' / 'python') ==
            preflight['python_sha256'], 'Original Python executable changed after preflight')
    before_sources = source_state(root, protocol)
    before_inputs = input_state(root, release, protocol)
    require(preflight['source'] == before_sources, 'Source state changed after preflight')
    require(preflight['inputs'] == before_inputs, 'Inputs changed after preflight')
    results = root / 'storage' / 'results' / 'unpaired'
    require(not results.exists() or not any(results.iterdir()), 'Existing results are not accepted for a new run')
    run_dir = root / 'run'
    run_dir.mkdir(exist_ok=False)
    execution = protocol['execution']
    require({k: execution[k] for k in ('workers', 'threads', 'environment')} ==
            {'workers': 4, 'threads': 4, 'environment': 'default'}, 'Unexpected execution settings')
    command = [str(root / 'bootstrap' / 'pixi'), 'run', '--frozen', '-e', 'default', 'python',
               '-m', 'experiments.unpaired', 'run', '--workers', '4', '--threads', '4',
               '--only', protocol['selection_filter']]
    env = os.environ.copy()
    # Avoid an inherited Python search path changing what the original CLI imports.
    for key in ('PYTHONPATH', 'PYTHONHOME'):
        env.pop(key, None)
    env['PYTHONNOUSERSITE'] = '1'
    env['UNPAIRED_ROSETTA_ROOT'] = str(root / 'storage')
    env['UNPAIRED_ROSETTA_RESULTS'] = 'results'
    state = {'schema_version': 1, 'status': 'starting', 'started_at_utc': utc(), 'supervisor_pid': os.getpid(),
             'supervisor_python': platform.python_version(), 'command': command, 'cwd': str(root / 'source'),
             'preflight_sha256': digest(preflight_path), 'release_before': before_release,
             'source_before': before_sources, 'inputs_before': before_inputs, 'expected_result_files':
             [expected_result_name(c) for c in expected_configurations(protocol)], 'attempt': 1}
    state_path = run_dir / 'status.json'
    write_json(state_path, state)
    child = None
    received_signals = []

    def forward(signum, _frame):
        received_signals.append(signum)
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signum)

    signal.signal(signal.SIGTERM, forward)
    signal.signal(signal.SIGINT, forward)
    try:
        with (run_dir / 'authors.log').open('xb') as log:
            child = subprocess.Popen(command, cwd=root / 'source', env=env, stdout=log,
                                     stderr=subprocess.STDOUT, start_new_session=True)
            state.update(status='running', child_pid=child.pid, child_started_at_utc=utc())
            write_json(state_path, state)
            while child.poll() is None:
                time.sleep(5)
                state.update(last_observed_at_utc=utc(), child_returncode=child.poll(),
                             result_files_present=sorted(p.name for p in results.glob('*.json')),
                             signals_received=received_signals[:])
                write_json(state_path, state)
            state.update(status='verifying', child_returncode=child.returncode,
                         child_finished_at_utc=utc(), signals_received=received_signals[:])
            write_json(state_path, state)
        # Always inspect immutable inputs after a terminated command, even if it failed.
        state['release_after'] = release_state(release)
        state['source_after'] = source_state(root, protocol)
        state['inputs_after'] = input_state(root, release, protocol)
        require(state['release_after'] == before_release, 'Release changed during the run')
        require(state['source_after'] == before_sources, 'Authors source changed during the run')
        require(state['inputs_after'] == before_inputs, 'Input files changed during the run')
        require(digest(preflight_path) == state['preflight_sha256'], 'Preflight receipt changed during run')
        require(child.returncode == 0, f'Authors process failed with return code {child.returncode}')
        require(not received_signals, 'Run was interrupted by a signal')
        state['results'] = result_records(root, protocol, require_complete=True)
        state['status'] = 'complete'
        state['completed_at_utc'] = utc()
        write_json(state_path, state)
        print(json.dumps({'status': 'complete', 'results': len(state['results']), 'status_file': str(state_path)}))
    except BaseException as error:
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            child.wait()
        state.update(status='failed', failed_at_utc=utc(), error=f'{type(error).__name__}: {error}',
                     child_returncode=None if child is None else child.returncode, signals_received=received_signals[:])
        write_json(state_path, state)
        raise


if __name__ == '__main__':
    main()
