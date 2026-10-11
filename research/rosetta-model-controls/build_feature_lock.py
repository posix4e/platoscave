#!/usr/bin/env python3
"""Record the fixed 14 feature artifacts, then run the cell runner's full preflight.

Run in the frozen baseline Pixi runtime after all ten extraction attempts finish.
This program never encodes, fits, evaluates, chooses models, or revises a protocol.
It requires the exact final --protocol-sha256 and every completed baseline result.

Private --inputs is path-only JSON: source_root, python_executable, locks,
datasets, baseline_results, optional baseline_replays, and feature_inputs.
feature_inputs contains every model/split key from run_cell.MODEL_SPECS/SPLITS:
released entries point to .pt files; extracted entries point to attempt directories.
Other fields have exactly the run_cell --bindings schema. Paths must be absolute.

The public feature lock points to the immutable scientific protocol, never the
reverse. An outer release manifest can bind the builder before outputs exist.
--bindings-out must be outside this repository. Both outputs must be absent.
A partial publication is preserved; retry to new paths, never overwrite files.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace

import run_cell as runner
import extract_features as extraction

ROOT = Path(__file__).resolve().parent
runner.require(Path(runner.__file__).resolve() == ROOT / 'run_cell.py' and
               Path(extraction.__file__).resolve() == ROOT / 'extract_features.py',
               'Import runner and extractor from this same release')


def path_leaves(value):
    """Private bindings contain dictionaries and absolute path strings only."""
    if isinstance(value, dict):
        runner.require(all(isinstance(key, str) for key in value), 'Private binding keys must be strings')
        for child in value.values():
            path_leaves(child)
    else:
        runner.require(isinstance(value, str) and Path(value).is_absolute(), 'Private bindings must contain absolute paths only')


def expand_bindings(inputs):
    required = {'source_root', 'python_executable', 'locks', 'datasets', 'baseline_results', 'feature_inputs'}
    runner.require(required <= set(inputs) <= required | {'baseline_replays'}, 'Unexpected or missing private binding fields')
    path_leaves(inputs)
    expected = {f'{model}/{split}' for model in runner.MODEL_SPECS for split in runner.SPLITS}
    runner.require(set(inputs['feature_inputs']) == expected, 'Exactly14 model/split feature inputs are required')
    runner.require(set(inputs['locks']) == runner.LOCK_NAMES and set(inputs['datasets']) == set(runner.SPLITS), 'Incomplete lock/dataset paths')
    runner.require(set(inputs['baseline_results']) == {str(seed) for seed in runner.SEEDS}, 'All five baseline result paths are required')
    result = {key: value for key, value in inputs.items() if key != 'feature_inputs'}
    result.setdefault('baseline_replays', {})
    result['features'] = {}
    for name in sorted(expected):
        path = Path(inputs['feature_inputs'][name]).resolve(strict=True)
        if runner.MODEL_SPECS[name.split('/')[0]]['kind'] == 'released':
            runner.require(path.is_file(), 'Released input must be a tensor file')
            result['features'][name] = {'tensor': str(path)}
        else:
            runner.require(path.is_dir() and not (path / 'FAILED.json').exists(), 'Extraction must be an intact nonfailed directory')
            result['features'][name] = {key: str(path / filename) for key, filename in
                                       (('tensor', 'features.pt'), ('metadata', 'features.json'), ('complete', 'COMPLETE.json'))}
    return result


def datasets(extraction_lock, bindings, bound):
    candidate = {}
    for split in runner.SPLITS:
        expected = extraction_lock['datasets'][split]
        annotations = bound.json(bindings[split]['annotation'], expected['annotation_sha256'])
        rows = extraction.ordered_rows(annotations, split)
        candidate[split] = {**expected,
                            'row_ids_sha256': runner.json_hash([row['image_id'] for row in rows]),
                            'caption_annotation_ids_sha256': runner.json_hash([[c['annotation_id'] for c in row['captions']] for row in rows])}
    # Reuse the exact downstream full-count/order/manifest checks.
    rows = runner.dataset_rows(candidate, extraction_lock, bindings, bound, SimpleNamespace(extraction=extraction))
    return candidate, rows


def feature_entries(dataset_lock, bindings, baseline_inputs, bound):
    publisher = {item['path']: item for item in baseline_inputs['files']}
    result = {}
    for name, local in sorted(bindings.items()):
        model, split = name.split('/')
        spec = runner.MODEL_SPECS[model]
        tensor = Path(local['tensor']).resolve(strict=True)
        shape = [runner.SPLITS[split], spec['dimension']]
        if spec['modality'] == 'language':
            shape.insert(1, 7)
        entry = {'model': model, 'split': split, 'kind': spec['kind'], 'shape': shape, 'dtype': spec['dtype'],
                 'size_bytes': tensor.stat().st_size,
                 **{key: dataset_lock[split][key] for key in ('row_ids_sha256', 'caption_annotation_ids_sha256')}}
        if spec['kind'] == 'released':
            publisher_path = f"embeddings/coco_{split}/{spec['modality']}/{spec['source_model_id']}.pt"
            original = publisher[publisher_path]
            entry.update(sha256=original['sha256'], publisher_path=publisher_path,
                         row_id_provenance='authors_sorted_coco_image_ids')
            runner.require(entry['size_bytes'] == original['size_bytes'], 'Released tensor byte count differs')
        else:
            complete_sha = runner.digest(local['complete'])
            complete = bound.json(local['complete'], complete_sha)
            runner.require(complete.get('status') == 'complete', 'Incomplete extraction attempt')
            metadata_sha = runner.sha(complete.get('metadata_sha256'))
            bound.add(local['metadata'], metadata_sha)
            entry.update(sha256=runner.sha(complete.get('feature_sha256')),
                         metadata_sha256=metadata_sha, complete_sha256=complete_sha)
        result[name] = entry
    return result


def assemble(protocol_path, protocol_sha, inputs):
    bound = runner.BoundFiles()
    protocol = bound.json(protocol_path, runner.sha(protocol_sha))
    runner.check_protocol(protocol)
    bound.add(__file__, runner.digest(__file__))
    for name, expected in protocol['code_sha256'].items():
        bound.add(ROOT / name, expected)
    bindings = expand_bindings(inputs)
    extraction_lock = bound.json(bindings['locks']['extraction'], protocol['locks']['extraction'])
    runner.require(extraction_lock.get('schema_version') == 1 and extraction_lock.get('protocol_id') == runner.PROTOCOL_ID and
                   extraction_lock.get('status') == 'frozen_before_model_control_extraction', 'A final extraction lock is required')
    baseline_inputs = bound.json(bindings['locks']['baseline_inputs'], protocol['locks']['baseline_inputs'])
    dataset_lock, rows = datasets(extraction_lock, bindings['datasets'], bound)
    lock = {'schema_version': 1, 'protocol_id': runner.PROTOCOL_ID, 'status': 'frozen_before_model_control_fits',
            'protocol_sha256': protocol_sha, 'extraction_lock_sha256': protocol['locks']['extraction'],
            'datasets': dataset_lock, 'features': feature_entries(dataset_lock, bindings['features'], baseline_inputs, bound),
            'builder_sha256': runner.digest(__file__), 'recorded_at_utc': datetime.now(timezone.utc).isoformat(),
            'scope': 'Byte and identity inventory of the prespecified four publisher caches and ten completed extractions; no new model/data choice, feature inference, fitting or evaluation.'}
    runner.no_local_paths(lock)
    return lock, bindings, bound


def check_outputs(feature_out, bindings_out):
    feature_out, bindings_out = Path(feature_out), Path(bindings_out)
    runner.require(feature_out.resolve() != bindings_out.resolve(), 'Public and private outputs must differ')
    for path in (feature_out, bindings_out):
        runner.require(not path.exists() and not path.is_symlink(), 'Output already exists; preserve it and use new paths')
        runner.require(path.parent.is_dir(), 'Create output parent directories explicitly')
    runner.require(not bindings_out.resolve().is_relative_to(ROOT.parent.parent), 'Private bindings must stay outside this repository')


def build(args):
    check_outputs(args.out, args.bindings_out)
    lock, bindings, assembled = assemble(args.protocol, args.protocol_sha256, runner.read_json(args.inputs))
    # Temporary private files let the production runner validate the exact schema.
    # locked_inputs checks every tensor/metadata/COMPLETE identity, source/runtime,
    # and the untouched baseline results. It does not prepare data or fit any map.
    with tempfile.TemporaryDirectory(prefix='feature-lock-validation-', dir=args.bindings_out.parent) as temporary:
        root = Path(temporary)
        candidate, private = root / 'feature-lock.json', root / 'bindings.json'
        runner.write_json_new(candidate, lock); runner.write_json_new(private, bindings)
        bound = runner.BoundFiles()
        runner.locked_inputs(SimpleNamespace(protocol=args.protocol, protocol_sha256=args.protocol_sha256,
                                             feature_lock=candidate, feature_lock_sha256=runner.digest(candidate),
                                             bindings=private, vision='dinov2', text='mpnet'), bound)
        bound.recheck(); assembled.recheck()
        check_outputs(args.out, args.bindings_out)
        # Paths publish first; the public feature lock is the final commit marker.
        runner.publish_marker(args.bindings_out, bindings)
        runner.publish_marker(args.out, lock)
    for parent in {args.out.parent, args.bindings_out.parent}:
        fd = os.open(parent, os.O_RDONLY)
        try: os.fsync(fd)
        finally: os.close(fd)
    return {'status': 'complete', 'feature_entries': len(lock['features']),
            'protocol_sha256': args.protocol_sha256, 'feature_lock_sha256': runner.digest(args.out),
            'bindings_sha256': runner.digest(args.bindings_out), 'full_runner_preflight_passed': True,
            'feature_inference_or_fitting_performed': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('protocol', 'inputs', 'out', 'bindings-out'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--protocol-sha256', required=True)
    args = parser.parse_args()
    import json
    print(json.dumps(build(args), indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
