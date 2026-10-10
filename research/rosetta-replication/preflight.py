#!/usr/bin/env python3
"""Verify the frozen reproduction inputs and original configuration without fitting.

Run this file through the authors' activated, frozen Pixi default environment.
Only configuration enumeration, data loading/reduction and index splitting are
performed. No encoder inference, aligner fitting or result evaluation occurs.
"""
from __future__ import annotations

import argparse
import dataclasses
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import sys

from run_locked import (canonical, digest, expected_configurations, expected_result_name,
                        input_state, read_json, release_state, require, source_state,
                        utc, value_digest, write_json)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--release', type=Path, required=True)
    args = parser.parse_args()
    root, release = args.root.resolve(), args.release.resolve()
    require(Path(__file__).resolve() == release / 'preflight.py', 'Use the frozen release preflight script')
    output = root / 'preflight.json'
    require(not output.exists(), 'An existing preflight receipt will not be overwritten')
    state = {'schema_version': 1, 'status': 'checking', 'started_at_utc': utc(), 'root': str(root),
             'scope': 'Input, environment, original configuration and split checks only; no fit or evaluation.'}
    try:
        state['release'] = release_state(release)
        protocol = read_json(release / 'protocol.json')
        state['source'] = source_state(root, protocol)
        published_source = read_json(release / 'source-lock.json')
        require(published_source['files'] == state['source']['source_files_sha256'] and
                published_source['pixi_binary_sha256'] == state['source']['pixi_binary_sha256'],
                'Public source lock and installed original source differ')
        tests = read_json(release / 'upstream-tests.json')
        require(tests.get('status') == 'passed' and tests.get('tests', 0) > 0 and
                all(tests.get(k) == 0 for k in ('failures', 'errors', 'skipped')),
                'Required upstream method/evaluation tests did not all pass')
        state['upstream_tests_sha256'] = digest(release / 'upstream-tests.json')
        state['inputs'] = input_state(root, release, protocol)
        env_prefix = root / 'source' / '.pixi' / 'envs' / 'default'
        require(Path(sys.prefix).resolve() == env_prefix.resolve(), 'Preflight is outside the original Pixi environment')
        require(Path(sys.executable).resolve() == (env_prefix / 'bin' / 'python').resolve(), 'Unexpected interpreter')
        require(platform.python_version() == protocol['expected_python'], 'Python version differs from protocol')
        state['python'] = platform.python_version()
        state['python_executable'] = str(Path(sys.executable).resolve())
        state['python_sha256'] = digest(sys.executable)
        state['activation'] = {k: os.environ.get(k) for k in protocol['expected_activation']}
        require(state['activation'] == protocol['expected_activation'], 'Pixi activation variables differ')
        state['versions'] = {name: version(name) for name in protocol['expected_versions']}
        require(state['versions'] == protocol['expected_versions'], 'A dependency version differs from protocol')
        require(os.environ.get('UNPAIRED_ROSETTA_ROOT') == str(root / 'storage'), 'UNPAIRED_ROSETTA_ROOT is not the locked storage')
        require(os.environ.get('UNPAIRED_ROSETTA_RESULTS', 'results') == 'results', 'Unexpected results folder override')
        sys.path.insert(0, str(root / 'source'))
        import torch
        import experiments.unpaired as original_experiment
        from unpaired_rosetta.determinism import FIT_THREADS, KMEANS_THREADS
        from unpaired_rosetta.embeddings import reduce_texts
        from unpaired_rosetta.evaluation import disjoint_split
        from unpaired_rosetta.randomness import Randomness, SEEDS
        from unpaired_rosetta.results import result_path
        from unpaired_rosetta.wasserstein_procrustes import WassersteinProcrustes
        require(Path(original_experiment.__file__).resolve() == root / 'source' / 'experiments' / 'unpaired.py',
                'Unexpected authors experiment import location')
        require(list(SEEDS) == protocol['seeds'], 'Official seeds differ from protocol')
        defaults = WassersteinProcrustes()
        state['algorithm_defaults'] = {name: getattr(defaults, name) for name in protocol['algorithm_defaults']}
        require(state['algorithm_defaults'] == protocol['algorithm_defaults'], 'Algorithm defaults differ')
        require(FIT_THREADS == 4 and KMEANS_THREADS == 1, 'Authors fixed thread settings differ')
        configurations = original_experiment.EXPERIMENT.selected(protocol['selection_filter'])
        observed = json.loads(json.dumps([dataclasses.asdict(c) for c in configurations]))
        expected = expected_configurations(protocol)
        require(observed == expected, 'Official CLI filter does not yield the five exact configurations in order')
        state['expected_configurations'] = observed
        state['expected_result_files'] = [expected_result_name(c) for c in expected]
        for config in configurations:
            require(result_path(root / 'storage' / 'results' / 'unpaired', config).name ==
                    expected_result_name(dataclasses.asdict(config)), 'Supervisor result names differ from original code')
        results = root / 'storage' / 'results' / 'unpaired'
        require(not results.exists() or not any(results.iterdir()), 'Preflight refuses existing experiment results')
        state['tensor_checks'] = {}
        for relative_name, expected_shape in protocol['expected_shapes'].items():
            tensor = torch.load(root / 'storage' / relative_name, mmap=True, map_location='cpu', weights_only=True)
            require(isinstance(tensor, torch.Tensor), f'Expected a tensor: {relative_name}')
            require(list(tensor.shape) == expected_shape, f'Unexpected tensor shape: {relative_name}')
            dtype = str(tensor.dtype)
            require(dtype == protocol['expected_dtypes'][relative_name], f'Unexpected tensor dtype: {relative_name}')
            details = {'shape': list(tensor.shape), 'dtype': dtype, 'valid_texts': 0, 'padded_texts': 0}
            for start in range(0, len(tensor), 8192):
                block = tensor[start:start + 8192]
                if tensor.ndim == 3:
                    valid = torch.isfinite(block).all(dim=-1)
                    padded = torch.isnan(block).all(dim=-1)
                    require(bool((valid | padded).all()), f'Partial NaN or infinite caption vectors: {relative_name}')
                    require(bool(valid.any(dim=1).all()), f'An item has no valid caption: {relative_name}')
                    details['valid_texts'] += int(valid.sum())
                    details['padded_texts'] += int(padded.sum())
                else:
                    require(bool(torch.isfinite(block).all()), f'Nonfinite tensor values: {relative_name}')
                if tensor.ndim >= 2:
                    reduced = reduce_texts(block)
                    require(bool(torch.isfinite(reduced).all()), f'Nonfinite reduced rows: {relative_name}')
                    require(bool((reduced.float().norm(dim=-1) > 0).all()), f'Zero reduced feature row: {relative_name}')
                    require(reduced.shape == (len(block), 768), f'Unexpected reduced shape: {relative_name}')
                else:
                    classes = 10 if '/CIFAR-10/' in relative_name else 100
                    require(bool(((block >= 0) & (block < classes)).all()), f'Out-of-range class labels: {relative_name}')
            state['tensor_checks'][relative_name] = details
            print(json.dumps({'checked': relative_name, 'shape': details['shape']}), flush=True)
            del tensor
        state['splits'] = []
        count = protocol['expected_training_counts']['x'] + protocol['expected_training_counts']['y']
        for seed in protocol['seeds']:
            x, y = disjoint_split(count, Randomness(seed))
            require(len(x) == protocol['expected_training_counts']['x'] and len(y) == protocol['expected_training_counts']['y'],
                    'Unexpected complete training half lengths')
            require(len(set(x.tolist()) & set(y.tolist())) == 0, 'Training halves overlap')
            require(torch.equal(torch.cat((x, y)).sort().values, torch.arange(count)), 'Training split omits/repeats items')
            state['splits'].append({'seed': seed, 'num_x': len(x), 'num_y': len(y), 'overlap': 0,
                                    'x_indices_sha256': value_digest(x.tolist()), 'y_indices_sha256': value_digest(y.tolist())})
        require(release_state(release) == state['release'], 'Release changed during preflight')
        require(source_state(root, protocol) == state['source'], 'Source changed during preflight')
        require(input_state(root, release, protocol) == state['inputs'], 'Inputs changed during preflight')
        state.update(status='passed', completed_at_utc=utc(), torch_cpu_capability=torch.backends.cpu.get_cpu_capability())
        write_json(output, state)
        print(json.dumps({'status': 'passed', 'receipt': str(output), 'sha256': digest(output)}), flush=True)
    except BaseException as error:
        state.update(status='failed', failed_at_utc=utc(), error=f'{type(error).__name__}: {error}')
        write_json(output, state)
        raise


if __name__ == '__main__':
    main()
