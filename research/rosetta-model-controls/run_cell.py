#!/usr/bin/env python3
"""DRAFT runner for one complete model-pair/seed cell; no fitting lock ships here.

Future public schema (all SHA256 values are hashes of exact file bytes):
  protocol: schema_version=1, protocol_id=rosetta-model-controls-v2,
    status=frozen_before_model_control_fits, upstream_commit, seeds=SEEDS,
    vision_models=VISION, text_models=TEXT, methods=METHODS, models=MODEL_SPECS,
    counts=COUNTS, algorithm_defaults=ALGORITHM, new_rosetta_fits=60,
    code_sha256={run_cell.py,control_primitives.py,extract_features.py},
    locks={source,environment,extraction,baseline_inputs,baseline_protocol,
           baseline_results} mapping logical names to SHA256,
    execution={num_workers:4,num_threads:4,device:cpu,evaluation_block_size:4096,
               torch_threads:<original runtime>,torch_interop_threads:<runtime>},
    baseline_replay={vision:dinov2,text:mpnet,absolute_tolerance:1e-10,
                     require_all_five_before_expansion:true}.
  feature-lock: schema_version=1, same protocol_id/status, protocol_sha256,
    extraction_lock_sha256, datasets={train2014,val2014}, features={model/split}.
    Each dataset has annotation_sha256,image_manifest_sha256,row_ids_sha256,
    caption_annotation_ids_sha256. ID hashes use compact JSON integer lists.
    All 14 feature entries have model,split,kind,sha256,size_bytes,shape,dtype,
    row_ids_sha256,caption_annotation_ids_sha256. Extracted entries additionally
    have metadata_sha256 and complete_sha256. Released entries additionally
    have publisher_path and row_id_provenance=authors_sorted_coco_image_ids.
  baseline_results lock: schema_version=1,status=complete,
    baseline_protocol_sha256, results={<decimal seed>:{sha256,size_bytes}}.

Private --bindings JSON supplies paths only: source_root, python_executable,
  locks={<logical name>:path}, datasets={split:{annotation,image_manifest}},
  features={model/split:{tensor[,metadata,complete]}},
  baseline_results={<seed>:path}, baseline_replays={<seed>:cell directory}.
No machine path belongs in a public lock. Both --protocol-sha256 and
--feature-lock-sha256 must be supplied explicitly. Every input/source is checked
before fitting and again before COMPLETE.json. No implicit retry or overwrite.

The 60 Rosetta fits include five new baseline replays because the original CLI
saved metrics, not maps or ranks. A baseline replay is compared with its untouched
raw result and never replaces it. All five baseline cell receipts must agree
before an expanded cell can run. Classification is not rerun in this panel.

Each method has a durable start/result journal entry and independent failure
record. COMPLETE.json requires all five methods, finite full-gallery ranks,
unchanged locks/inputs/source, and baseline agreement when applicable. A failed
method does not erase another method. Detailed tracebacks go to --private-log-dir.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import time
import traceback
from types import SimpleNamespace

PROTOCOL_ID = 'rosetta-model-controls-v2'
UPSTREAM = 'fdfad84ea488f29702a61a455be71622c4ff5d49'
SEEDS = [734796314, 576165995, 2197670066, 839703249, 2584932063]
VISION = ['dinov2', 'dino', 'moco', 'mae']
TEXT = ['mpnet', 'glove', 'elmo']
METHODS = ['rosetta', 'orthogonal_paired', 'orthogonal_shuffled', 'linear_paired', 'linear_shuffled']
COUNTS = {'train': 82783, 'fit_x': 41391, 'fit_y': 41392, 'validation': 40504}
ALGORITHM = {'num_clusters': 30, 'num_restarts': 30, 'batch_size': 10000, 'num_iterations': 100}
MODEL_SPECS = {
    'dinov2': {'modality': 'vision', 'dimension': 768, 'dtype': 'bfloat16', 'kind': 'released', 'source_model_id': 'dinov2_vit-b14@224_mean'},
    'dino': {'modality': 'vision', 'dimension': 768, 'dtype': 'float32', 'kind': 'extracted', 'source_model_id': 'dino'},
    'moco': {'modality': 'vision', 'dimension': 2048, 'dtype': 'float32', 'kind': 'extracted', 'source_model_id': 'moco'},
    'mae': {'modality': 'vision', 'dimension': 768, 'dtype': 'float32', 'kind': 'extracted', 'source_model_id': 'mae'},
    'mpnet': {'modality': 'language', 'dimension': 768, 'dtype': 'bfloat16', 'kind': 'released', 'source_model_id': 'mpnet'},
    'glove': {'modality': 'language', 'dimension': 300, 'dtype': 'float32', 'kind': 'extracted', 'source_model_id': 'glove'},
    'elmo': {'modality': 'language', 'dimension': 1024, 'dtype': 'float32', 'kind': 'extracted', 'source_model_id': 'elmo'},
}
LOCK_NAMES = {'source', 'environment', 'extraction', 'baseline_inputs', 'baseline_protocol', 'baseline_results'}
CODE_NAMES = {'run_cell.py', 'control_primitives.py', 'extract_features.py'}
SPLITS = {'train2014': COUNTS['train'], 'val2014': COUNTS['validation']}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            result.update(block)
    return result.hexdigest()


def json_hash(value):
    return hashlib.sha256(json.dumps(value, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def read_json(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, 'Duplicate JSON key')
            result[key] = value
        return result
    def invalid(value):
        raise ValueError('Nonfinite JSON number')
    return json.loads(Path(path).read_text(), object_pairs_hook=unique, parse_constant=invalid)


def sha(value):
    require(isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value), 'Missing complete SHA256')
    return value


def no_local_paths(value):
    if isinstance(value, dict):
        for key, child in value.items():
            no_local_paths(key)
            no_local_paths(child)
    elif isinstance(value, list):
        for child in value:
            no_local_paths(child)
    elif isinstance(value, str):
        require(not value.startswith(('/', '~/', 'file://')) and not re.match(r'^[A-Za-z]:[\\/]', value),
                'Machine-local paths belong in private bindings only')


def check_protocol(p):
    no_local_paths(p)
    fixed = {'schema_version': 1, 'protocol_id': PROTOCOL_ID, 'status': 'frozen_before_model_control_fits',
             'upstream_commit': UPSTREAM, 'seeds': SEEDS, 'vision_models': VISION, 'text_models': TEXT,
             'methods': METHODS, 'models': MODEL_SPECS, 'counts': COUNTS, 'algorithm_defaults': ALGORITHM,
             'new_rosetta_fits': 60,
             'baseline_replay': {'vision': 'dinov2', 'text': 'mpnet', 'absolute_tolerance': 1e-10,
                                 'require_all_five_before_expansion': True}}
    for name, expected in fixed.items():
        require(p.get(name) == expected, f'Unfrozen or changed protocol field: {name}')
    require(set(p.get('locks', {})) == LOCK_NAMES and set(p.get('code_sha256', {})) == CODE_NAMES,
            'Incomplete source/runtime/input/code lock')
    for value in [*p['locks'].values(), *p['code_sha256'].values()]:
        sha(value)
    execution = p['execution']
    for key, value in {'num_workers': 4, 'num_threads': 4, 'device': 'cpu', 'evaluation_block_size': 4096,
                       'torch_threads': 16, 'torch_interop_threads': 16}.items():
        require(execution.get(key) == value, f'Changed execution setting: {key}')


class BoundFiles:
    """Private paths, exact byte hashes, and a final unchanged-input check."""
    def __init__(self):
        self.files = {}

    def add(self, path, expected, size=None):
        path = Path(path).resolve(strict=True)
        require(path.is_file() and digest(path) == sha(expected), 'A bound file hash differs')
        if size is not None:
            require(type(size) is int and size > 0 and path.stat().st_size == size, 'A bound file size differs')
        require(path not in self.files or self.files[path] == expected, 'Conflicting file bindings')
        self.files[path] = expected
        return path

    def json(self, path, expected):
        return read_json(self.add(path, expected))

    def recheck(self):
        for path, expected in self.files.items():
            require(digest(path) == expected, 'A bound file changed during execution')


def normalized_versions(values):
    result = {}
    for key, value in values.items():
        name = re.sub('[-_.]+', '-', key).lower()
        require(name not in result, 'Duplicate installed distribution identity')
        result[name] = value
    return result


def check_runtime(environment, execution, source, python_executable):
    require(platform.python_version() == environment['python'].split()[0], 'Python version differs')
    require(Path(sys.executable).resolve() == Path(python_executable).resolve(strict=True), 'Wrong Python executable')
    require(Path(sys.prefix).resolve() == source / '.pixi' / 'envs' / 'default', 'Use the original frozen Pixi environment')
    installed = {d.metadata['Name']: d.version for d in importlib.metadata.distributions()}
    expected = environment.get('full_installed_distributions', {})
    require(bool(expected) and normalized_versions(installed) == normalized_versions(expected), 'Full runtime versions differ')
    for key, value in environment['activation'].items():
        require(os.environ.get(key) == value, 'Runtime activation differs')
    import torch
    require(torch.backends.cpu.get_cpu_capability() == environment['torch_cpu_capability'] == 'AVX2', 'Wrong CPU dispatch')
    require(torch.get_num_threads() == execution['torch_threads'] and
            torch.get_num_interop_threads() == execution['torch_interop_threads'], 'Original process thread counts differ')


def check_sources(source, manifest, bound):
    require(manifest.get('upstream_commit') == UPSTREAM and len(manifest.get('files', {})) == 228, 'Incomplete original source manifest')
    require(subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=source, text=True).strip() == UPSTREAM, 'Wrong source HEAD')
    subprocess.run(['git', 'diff', '--quiet', 'HEAD', '--'], cwd=source, check=True)
    for name, expected in manifest['files'].items():
        path = (source / name).resolve(strict=True)
        require(path.is_relative_to(source), 'Source manifest path escapes checkout')
        bound.add(path, expected)
    for folder in ('unpaired_rosetta', 'experiments'):
        require(all(str(path.relative_to(source)) in manifest['files'] for path in (source / folder).rglob('*.py')),
                'Extra Python source could change original imports')


def load_api(source):
    # Source bytes/runtime are checked before this function is called by main.
    sys.path.insert(0, str(source))
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import torch
    import control_primitives as controls
    import extract_features as extraction
    require(Path(controls.__file__).resolve() == Path(__file__).resolve().parent / 'control_primitives.py' and
            Path(extraction.__file__).resolve() == Path(__file__).resolve().parent / 'extract_features.py',
            'Controls/extractor imported from another release')
    from unpaired_rosetta.baselines.linear import LinearMap
    from unpaired_rosetta.baselines.orthogonal import OrthogonalMap
    from unpaired_rosetta.embeddings import load_rows
    from unpaired_rosetta.evaluation import disjoint_split
    from unpaired_rosetta.randomness import Randomness, SEEDS as official_seeds
    from unpaired_rosetta.wasserstein_procrustes import WassersteinProcrustes
    for name, module in list(sys.modules.items()):
        if name == 'unpaired_rosetta' or name.startswith('unpaired_rosetta.'):
            require(Path(module.__file__).resolve().is_relative_to(source), 'Original API imported from another checkout')
    require(official_seeds == SEEDS, 'Original seed list differs')
    original = WassersteinProcrustes()
    require({name: getattr(original, name) for name in ALGORITHM} == ALGORITHM, 'Original constructor defaults differ')
    return SimpleNamespace(torch=torch, controls=controls, extraction=extraction, LinearMap=LinearMap,
                           OrthogonalMap=OrthogonalMap, load_rows=load_rows, disjoint_split=disjoint_split,
                           Randomness=Randomness, WassersteinProcrustes=WassersteinProcrustes)


def dataset_rows(lock, extraction_lock, bindings, bound, api):
    require(set(lock) == set(SPLITS), 'Both full COCO splits must be locked')
    result = {}
    for split, count in SPLITS.items():
        entry = lock[split]
        expected = extraction_lock['datasets'][split]
        require({key: entry[key] for key in expected} == expected, 'Dataset differs from extraction lock')
        annotations = bound.json(bindings[split]['annotation'], entry['annotation_sha256'])
        manifest = bound.json(bindings[split]['image_manifest'], entry['image_manifest_sha256'])
        rows = api.extraction.ordered_rows(annotations, split)
        require(len(rows) == count, 'A reduced corpus is forbidden')
        require(manifest.get('split') == split and manifest.get('annotation_sha256') == entry['annotation_sha256'], 'Wrong image manifest')
        require(len(manifest['images']) == count, 'Incomplete image manifest')
        for i, (row, image) in enumerate(zip(rows, manifest['images'])):
            require(image['row'] == i and image['image_id'] == row['image_id'] and image['file'] == row['file'], 'Image manifest order differs')
            sha(image['sha256'])
        ids = [row['image_id'] for row in rows]
        annotation_ids = [[c['annotation_id'] for c in row['captions']] for row in rows]
        require(json_hash(ids) == entry['row_ids_sha256'] and json_hash(annotation_ids) == entry['caption_annotation_ids_sha256'], 'Dataset ID order hash differs')
        result[split] = {'image_ids': ids, 'caption_annotation_ids': annotation_ids,
                         'caption_counts': [len(row['captions']) for row in rows]}
    require(not set(result['train2014']['image_ids']) & set(result['val2014']['image_ids']), 'Training and validation image IDs overlap')
    train_captions = {i for row in result['train2014']['caption_annotation_ids'] for i in row}
    val_captions = {i for row in result['val2014']['caption_annotation_ids'] for i in row}
    require(not train_captions & val_captions, 'Training and validation caption IDs overlap')
    return result


def validate_tensor(path, spec, shape, caption_counts, api):
    torch = api.torch
    tensor = torch.load(path, mmap=True, weights_only=True, map_location='cpu')
    require(isinstance(tensor, torch.Tensor) and list(tensor.shape) == shape and str(tensor.dtype) == 'torch.' + spec['dtype'], 'Feature tensor shape/dtype differs')
    for start in range(0, len(tensor), 512):
        block = tensor[start:start + 512]
        if spec['modality'] == 'vision':
            require(bool(torch.isfinite(block).all()) and bool((block.float().norm(dim=-1) > 0).all()), 'Invalid image features')
        else:
            valid = torch.arange(7)[None, :] < torch.tensor(caption_counts[start:start + len(block)])[:, None]
            require(bool(torch.isfinite(block[valid]).all()) and bool(torch.isnan(block[~valid]).all()), 'Captions or NaN padding differ from annotation counts')
            require(bool((block[valid].float().norm(dim=-1) > 0).all()), 'Zero caption feature')


def feature_paths(flock, bindings, rows, extraction_lock, baseline_inputs, bound, api):
    expected_names = {f'{model}/{split}' for model in MODEL_SPECS for split in SPLITS}
    require(set(flock['features']) == expected_names and set(bindings) == expected_names, 'All14 model/split features are required')
    require(baseline_inputs.get('repository') == 'schnaus/unpaired-rosetta-embeddings' and
            baseline_inputs.get('revision') == 'b39767e18cdb5031020a0e1c4b8d9fba6ae13453', 'Wrong released feature source')
    publisher = {item['path']: item for item in baseline_inputs['files']}
    paths = {}
    for name in sorted(expected_names):
        model, split = name.split('/')
        spec, entry, local = MODEL_SPECS[model], flock['features'][name], bindings[name]
        count, dim = SPLITS[split], spec['dimension']
        shape = [count, dim] if spec['modality'] == 'vision' else [count, 7, dim]
        for key, value in {'model': model, 'split': split, 'kind': spec['kind'], 'shape': shape, 'dtype': spec['dtype']}.items():
            require(entry.get(key) == value, 'Feature model/split identity differs')
        for key in ('row_ids_sha256', 'caption_annotation_ids_sha256'):
            require(entry[key] == flock['datasets'][split][key], 'Feature row identity differs')
        path = bound.add(local['tensor'], entry['sha256'], entry['size_bytes'])
        if spec['kind'] == 'released':
            original_path = f"embeddings/coco_{split}/{spec['modality']}/{spec['source_model_id']}.pt"
            require(entry.get('publisher_path') == original_path and entry.get('row_id_provenance') == 'authors_sorted_coco_image_ids', 'Released feature provenance differs')
            require(publisher[original_path]['sha256'] == entry['sha256'] and publisher[original_path]['size_bytes'] == entry['size_bytes'], 'Released BF16 bytes differ from original baseline')
        else:
            require(path.name == 'features.pt' and Path(local['metadata']).resolve() == path.parent / 'features.json' and
                    Path(local['complete']).resolve() == path.parent / 'COMPLETE.json', 'Use one intact extraction attempt directory')
            require(not (path.parent / 'FAILED.json').exists(), 'Failed extraction cannot be consumed')
            metadata = bound.json(local['metadata'], entry['metadata_sha256'])
            complete = bound.json(local['complete'], entry['complete_sha256'])
            binding = {'protocol_id': PROTOCOL_ID, 'model': model, 'split': split,
                       'extractor_sha256': extraction_lock['extractor_sha256'],
                       'lock_sha256': flock['extraction_lock_sha256'], 'row_count': count,
                       'dataset': extraction_lock['datasets'][split]}
            require(all(metadata.get(k) == v and complete.get(k) == v for k, v in binding.items()), 'Extraction binding differs')
            require(complete.get('status') == 'complete' and complete.get('last_completed_row') == count - 1,
                    'Extraction is not complete')
            require(complete.get('feature_sha256') == entry['sha256'] and complete.get('metadata_sha256') == entry['metadata_sha256'], 'Extraction completion hashes differ')
            require(metadata.get('feature_sha256') == entry['sha256'] and metadata.get('shape') == complete.get('shape') == shape and metadata.get('dtype') == 'float32', 'Extraction tensor metadata differs')
            require(metadata.get('model_contract') == api.extraction.MODEL_CONTRACT[model], 'Encoder contract differs')
            require(metadata.get('preprocessing_id') == api.extraction.PREPROCESSING_ID and metadata.get('no_training') is True,
                    'Feature extraction recipe or no-training declaration differs')
            require(metadata.get('runtime_versions') == extraction_lock['models'][model]['runtime_versions'], 'Encoder runtime differs')
            required = api.extraction.REQUIRED[model]
            require(metadata.get('input_hashes') == {k: api.extraction.PINNED[k] for k in required}, 'Encoder source/checkpoint identities differ')
            require(all(metadata.get(k) == v for k, v in rows[split].items()), 'Extraction IDs/caption order differs')
        validate_tensor(path, spec, shape, rows[split]['caption_counts'], api)
        paths[name] = path
    return paths


def baseline_records(lock, protocol, bindings, bound):
    require(lock.get('schema_version') == 1 and lock.get('status') == 'complete' and
            set(lock.get('results', {})) == {str(seed) for seed in SEEDS}, 'Five completed original baseline results required')
    require(protocol['upstream_commit'] == UPSTREAM and protocol['seeds'] == SEEDS, 'Original baseline protocol differs')
    result = {}
    for seed in SEEDS:
        key, ref = str(seed), lock['results'][str(seed)]
        path = bound.add(bindings[key], ref['sha256'], ref['size_bytes'])
        record = read_json(path)
        config = {**protocol['configuration_except_seed'], 'seed': seed}
        require(record['config'] == config and record['git_commit'] == UPSTREAM, 'Original baseline configuration differs')
        require(config['model_x'] == 'dinov2_vit-b14@224_mean' and config['model_y'] == 'mpnet' and
                config['num_pairs'] == 0 and config['dataset'] == 'coco_train2014' and config['validation'] == 'coco_val2014' and
                config['train_size'] is None and config['validation_size'] is None and config['holdout'] == 0, 'Wrong baseline task')
        for key in protocol['required_metrics']:
            require(type(record['metrics'].get(key)) in (int, float) and math.isfinite(record['metrics'][key]), 'Incomplete original baseline metrics')
        require(record['metrics']['num_validation_samples'] == COUNTS['validation'], 'Original baseline validation is incomplete')
        for package in ('torch', 'numpy', 'scipy', 'scikit-learn', 'pylibmgm'):
            require(record['versions'][package] == protocol['expected_versions'][package], 'Original baseline runtime differs')
        result[seed] = {'sha256': ref['sha256'], 'metrics': record['metrics']}
    return result


def replay_comparison(metrics, reference):
    comparisons = {}
    for key in ('foscttm', 'mean_rank'):
        error = metrics[key] - reference['metrics'][key]
        comparisons[key] = {'observed': metrics[key], 'original': reference['metrics'][key],
                            'signed_error': error, 'absolute_error': abs(error), 'absolute_tolerance': 1e-10,
                            'agrees': abs(error) <= 1e-10}
    return {'original_result_sha256': reference['sha256'], 'metrics': comparisons,
            'agrees': all(value['agrees'] for value in comparisons.values()),
            'original_result_replaced': False}


def require_baseline_replays(paths, protocol_sha, feature_sha, references, bound):
    require(set(paths) == {str(seed) for seed in SEEDS}, 'All five completed baseline replays must precede expansion')
    for seed in SEEDS:
        root = Path(paths[str(seed)]).resolve(strict=True)
        marker_path = root / 'COMPLETE.json'
        marker = read_json(marker_path)
        bound.add(marker_path, digest(marker_path))
        expected = {'status': 'complete', 'protocol_sha256': protocol_sha, 'feature_lock_sha256': feature_sha,
                    'vision': 'dinov2', 'text': 'mpnet', 'seed': seed}
        require(all(marker.get(k) == v for k, v in expected.items()) and not (root / 'FAILED.json').exists(), 'Wrong or failed baseline replay')
        require('cell.json' in marker['files'] and all(f'{m}/result.json' in marker['files'] for m in METHODS), 'Incomplete baseline replay receipt')
        for name, expected_hash in marker['files'].items():
            path = (root / name).resolve(strict=True)
            require(path.is_relative_to(root), 'Replay artifact escapes attempt')
            bound.add(path, expected_hash)
        for method in METHODS:
            result = read_json(root / method / 'result.json')
            require(all(result.get(k) == v for k, v in expected.items()) and result.get('method') == method,
                    'Wrong or failed baseline cell method')
            require(marker['files'].get(f'{method}/map.pt') == result.get('map_sha256') and
                    marker['files'].get(f'{method}/ranks.pt') == result.get('ranks_sha256'), 'Replay map/rank hashes disagree')
            require(result['metrics'].get('num_validation_samples') == COUNTS['validation'], 'Incomplete baseline replay gallery')
        result = read_json(root / 'rosetta' / 'result.json')
        require(result.get('baseline_replay') == replay_comparison(result['metrics'], references[seed]) and
                result['baseline_replay']['agrees'], 'Baseline replay does not agree with untouched original metrics')


def prepare_data(paths, vision, text, seed, rows, api):
    """Use the original split stream and a separate original evaluation stream."""
    train_rng = api.Randomness(seed)
    ix, iy = api.disjoint_split(len(rows['train2014']['image_ids']), train_rng, None)
    px, py = paths[f'{vision}/train2014'], paths[f'{text}/train2014']
    cx, cy = api.load_rows(px, ix), api.load_rows(py, iy)
    full_x, full_y = api.load_rows(px), api.load_rows(py)
    require(api.torch.equal(cx, full_x[ix]) and api.torch.equal(cy, full_y[iy]), 'Full-bank and original subset reduction differ')
    eval_rng = api.Randomness(seed)
    validation_indices = eval_rng.subset(len(rows['val2014']['image_ids']), None)
    vx = api.load_rows(paths[f'{vision}/val2014'], validation_indices)
    vy = api.load_rows(paths[f'{text}/val2014'], validation_indices)
    for name, tensor in [('centering_x', cx), ('centering_y', cy), ('paired_x', full_x), ('paired_y', full_y), ('validation_x', vx), ('validation_y', vy)]:
        api.controls.features(name, tensor)
    train_ids = api.torch.tensor(rows['train2014']['image_ids'], dtype=api.torch.int64)
    validation_ids = api.torch.tensor(rows['val2014']['image_ids'], dtype=api.torch.int64)[validation_indices]
    return SimpleNamespace(cx=cx, cy=cy, full_x=full_x, full_y=full_y, vx=vx, vy=vy,
                           ix=ix, iy=iy, train_rng=train_rng, train_ids=train_ids,
                           validation_indices=validation_indices, validation_ids=validation_ids)


def fit_rosetta(data, api):
    aligner = api.WassersteinProcrustes(num_workers=4, num_threads=4)
    return aligner.fit(data.cx, data.cy, None, None, data.train_rng)


def evaluate_ranks(model, data, seed, api):
    rng = api.Randomness(seed)
    indices = rng.subset(len(data.vx), None)
    require(api.torch.equal(indices, data.validation_indices), 'Original validation subset stream differs')
    result = api.controls.retrieval_with_ranks(model, data.vx, data.vy, rng, block_size=4096)
    ranks = result['ranks']
    require(bool(api.torch.isfinite(ranks).all()) and bool(((ranks >= 1) & (ranks <= len(ranks))).all()), 'Invalid retrieval ranks')
    return result


def write_json_new(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())


def publish_marker(path, value):
    path = Path(path)
    partial = path.with_name(path.name + '.partial')
    write_json_new(partial, value)
    os.link(partial, path)
    partial.unlink()


def save_tensor_new(path, value, torch):
    with Path(path).open('xb') as stream:
        torch.save(value, stream)
        stream.flush(); os.fsync(stream.fileno())


class Attempt:
    def __init__(self, out, private, binding):
        self.out, self.private, self.binding = Path(out), Path(private), binding
        self.statuses = {method: 'not_attempted' for method in METHODS}
        self.out.mkdir(parents=True, exist_ok=False)
        self.private.mkdir(parents=True, exist_ok=False)
        write_json_new(self.out / 'started.json', {**binding, 'status': 'started', 'methods': METHODS,
                                                  'retry_policy': 'new_attempt_directory_only'})
        self.journal = (self.out / 'journal.jsonl').open('x')
        self.event('attempt_started')

    def event(self, event, **extra):
        self.journal.write(json.dumps({'event': event, 'at_utc': datetime.now(timezone.utc).isoformat(), **extra}, allow_nan=False) + '\n')
        self.journal.flush(); os.fsync(self.journal.fileno())

    def method(self, name, operation):
        folder = self.out / name
        folder.mkdir()
        self.statuses[name] = 'started'
        self.event('method_started', method=name)
        try:
            result = operation(folder)
            require(result['status'] in ('complete', 'failed'), 'Method did not return a final status')
        except Exception as error:
            with (self.private / (name + '.traceback.txt')).open('x') as stream:
                traceback.print_exc(file=stream)
            result = {'status': 'failed', 'error_type': type(error).__name__,
                      'failure_stage': 'fit_evaluation_or_serialization',
                      'detail': 'See the separate machine-local traceback; no retry was attempted.'}
        result = {**self.binding, 'method': name, **result}
        write_json_new(folder / 'result.json', result)
        self.statuses[name] = result['status']
        self.event('method_finished', method=name, status=result['status'])
        return result

    def finish(self, detail, bound):
        bound.recheck()
        complete = all(status == 'complete' for status in self.statuses.values()) and detail.get('baseline_replay_agrees', True)
        record = {**self.binding, **detail, 'status': 'complete' if complete else 'failed', 'methods': self.statuses}
        write_json_new(self.out / 'cell.json', record)
        self.event('attempt_finished', status=record['status'])
        self.journal.close()
        files = {str(path.relative_to(self.out)): digest(path) for path in sorted(self.out.rglob('*')) if path.is_file()}
        marker = {**self.binding, 'status': record['status'], 'files': files}
        publish_marker(self.out / ('COMPLETE.json' if complete else 'FAILED.json'), marker)
        descriptor = os.open(self.out, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        return complete

    def abort(self, error):
        with (self.private / 'attempt.traceback.txt').open('x') as stream:
            traceback.print_exc(file=stream)
        if not self.journal.closed:
            self.event('attempt_aborted', error_type=type(error).__name__, methods=self.statuses)
            self.journal.close()
        if not (self.out / 'FAILED.json').exists():
            publish_marker(self.out / 'FAILED.json', {**self.binding, 'status': 'failed', 'methods': self.statuses,
                                                    'error_type': type(error).__name__})


def run_methods(attempt, data, seed, reference, api):
    pairing = api.controls.training_partner_permutation(PROTOCOL_ID, seed, len(data.full_x))
    diagnostics = {}
    results = {}
    for method in METHODS:
        def operation(folder, method=method):
            started = time.monotonic()
            if method == 'rosetta':
                model = fit_rosetta(data, api)
            else:
                cls = api.OrthogonalMap if method.startswith('orthogonal') else api.LinearMap
                target = data.full_y[pairing.indices] if method.endswith('_shuffled') else data.full_y
                model = cls().fit(data.cx, data.cy, data.full_x, target)
            require(all(bool(api.torch.isfinite(getattr(model, key)).all()) for key in ('mean_x', 'mean_y', 'weight')), 'Nonfinite fitted map')
            save_tensor_new(folder / 'map.pt', {key: getattr(model, key) for key in ('mean_x', 'mean_y', 'weight')}, api.torch)
            fit_seconds = time.monotonic() - started
            evaluation = evaluate_ranks(model, data, seed, api)
            save_tensor_new(folder / 'ranks.pt', {'validation_image_ids': data.validation_ids,
                            'validation_indices': data.validation_indices, 'ranks': evaluation['ranks'],
                            'query_order': evaluation['query_order'], 'key_order': evaluation['key_order'],
                            'partner_in_shuffled_keys': evaluation['partner_in_shuffled_keys']}, api.torch)
            metrics = {**evaluation['metrics'], 'num_validation_samples': len(data.vx)}
            result = {'status': 'complete', 'metrics': metrics, 'fit_seconds': fit_seconds,
                      'fit_status': 'complete', 'evaluation_status': 'complete',
                      'fit_and_evaluation_seconds': time.monotonic() - started,
                      'metric_provenance': 'Original FP32 similarities, double midranks and formulas via synthetic-validated supplemental rank recorder; one full-gallery scoring pass.',
                      'map_sha256': digest(folder / 'map.pt'), 'ranks_sha256': digest(folder / 'ranks.pt')}
            if method != 'rosetta':
                result['paired_training_items'] = len(data.full_x)
                result['additional_x_counterparts'] = len(data.full_x) - len(data.cx)
                result['additional_y_counterparts'] = len(data.full_y) - len(data.cy)
                result['partner_permutation'] = pairing.metadata if method.endswith('_shuffled') else None
                result['information_budget'] = 'Receives full training counterparts beyond the disjoint unpaired banks.'
            if method.startswith('linear'):
                if not diagnostics:
                    # Once per cell, shared by both maps. A diagnostic failure is
                    # retained and never retried by the next method.
                    diagnostics['attempted'] = True
                    try:
                        diagnostics['value'] = api.controls.linear_pinv_diagnostics(data.full_x, model.mean_x)
                    except Exception as error:
                        diagnostics['error_type'] = type(error).__name__
                        with (attempt.private / (method + '.diagnostic.traceback.txt')).open('x') as stream:
                            traceback.print_exc(file=stream)
                if 'value' in diagnostics:
                    result['linear_pinv_diagnostic'] = diagnostics['value']
                    result['diagnostic_status'] = 'complete'
                else:
                    result.update(status='failed', diagnostic_status='failed',
                                  failure_stage='shared_post_fit_svd_diagnostic', error_type=diagnostics['error_type'],
                                  detail='Fit, evaluation, maps and ranks succeeded and are retained. The required shared rank diagnostic failed once and was not retried.')
            if method == 'rosetta' and reference is not None:
                result['baseline_replay'] = replay_comparison(metrics, reference)
            return result
        results[method] = attempt.method(method, operation)
    return results


def locked_inputs(args, bound):
    protocol = bound.json(args.protocol, args.protocol_sha256)
    check_protocol(protocol)
    flock = bound.json(args.feature_lock, args.feature_lock_sha256)
    no_local_paths(flock)
    require(flock.get('schema_version') == 1 and flock.get('protocol_id') == PROTOCOL_ID and
            flock.get('status') == 'frozen_before_model_control_fits' and flock.get('protocol_sha256') == args.protocol_sha256,
            'A complete feature lock bound to this protocol is required')
    bindings = read_json(args.bindings)
    source = Path(bindings['source_root']).resolve(strict=True)
    root = Path(__file__).resolve().parent
    for name, expected in protocol['code_sha256'].items():
        bound.add(root / name, expected)
    locks = {name: bound.json(bindings['locks'][name], expected) for name, expected in protocol['locks'].items()}
    require(flock.get('extraction_lock_sha256') == protocol['locks']['extraction'], 'Feature and extraction locks differ')
    check_sources(source, locks['source'], bound)
    check_runtime(locks['environment'], protocol['execution'], source, bindings['python_executable'])
    api = load_api(source)
    extraction = locks['extraction']
    require(extraction.get('schema_version') == 1 and extraction.get('protocol_id') == PROTOCOL_ID and
            extraction.get('status') == 'frozen_before_model_control_extraction', 'Wrong extraction lock')
    require(extraction.get('extractor_sha256') == protocol['code_sha256']['extract_features.py'] and
            extraction.get('caption_slots') == 7 and extraction.get('preprocessing_id') == api.extraction.PREPROCESSING_ID,
            'Extraction recipe differs')
    require(set(extraction['models']) == set(api.extraction.MODEL_CONTRACT), 'Incomplete encoder panel lock')
    for model, contract in api.extraction.MODEL_CONTRACT.items():
        require(all(extraction['models'][model].get(k) == v for k, v in contract.items()) and
                bool(extraction['models'][model].get('runtime_versions')), 'Encoder model/runtime contract differs')
    require(all(extraction['files'].get(k) == v for k, v in api.extraction.PINNED.items()), 'Encoder sources/checkpoints differ')
    require(locks['baseline_results']['baseline_protocol_sha256'] == protocol['locks']['baseline_protocol'], 'Original result lock refers to another baseline protocol')
    references = baseline_records(locks['baseline_results'], locks['baseline_protocol'], bindings['baseline_results'], bound)
    rows = dataset_rows(flock['datasets'], extraction, bindings['datasets'], bound, api)
    paths = feature_paths(flock, bindings['features'], rows, extraction, locks['baseline_inputs'], bound, api)
    if (args.vision, args.text) != ('dinov2', 'mpnet'):
        require_baseline_replays(bindings.get('baseline_replays', {}), args.protocol_sha256,
                                args.feature_lock_sha256, references, bound)
    return protocol, api, rows, paths, references


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('protocol', 'feature-lock', 'bindings', 'out', 'private-log-dir'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--protocol-sha256', required=True)
    parser.add_argument('--feature-lock-sha256', required=True)
    parser.add_argument('--vision', choices=VISION, required=True)
    parser.add_argument('--text', choices=TEXT, required=True)
    parser.add_argument('--seed', type=int, choices=SEEDS, required=True)
    args = parser.parse_args()
    sha(args.protocol_sha256); sha(args.feature_lock_sha256)
    require(not args.out.exists() and not args.out.is_symlink() and not args.private_log_dir.exists(), 'Fresh attempt and private log directories are required')
    require(not args.private_log_dir.resolve().is_relative_to(args.out.resolve()), 'Detailed logs must be separate from public artifacts')
    binding = {'schema_version': 1, 'protocol_id': PROTOCOL_ID, 'protocol_sha256': args.protocol_sha256,
               'feature_lock_sha256': args.feature_lock_sha256, 'vision': args.vision, 'text': args.text, 'seed': args.seed}
    attempt = Attempt(args.out, args.private_log_dir, binding)
    bound = BoundFiles()
    try:
        protocol, api, rows, paths, references = locked_inputs(args, bound)
        bound.recheck()
        attempt.event('preflight_passed')
        data = prepare_data(paths, args.vision, args.text, args.seed, rows, api)
        require((len(data.cx), len(data.cy), len(data.vx)) == (COUNTS['fit_x'], COUNTS['fit_y'], COUNTS['validation']), 'Full original sample counts required')
        save_tensor_new(args.out / 'split.pt', {'training_image_ids': data.train_ids, 'indices_x': data.ix,
                        'indices_y': data.iy, 'validation_indices': data.validation_indices,
                        'validation_image_ids': data.validation_ids}, api.torch)
        reference = references[args.seed] if (args.vision, args.text) == ('dinov2', 'mpnet') else None
        results = run_methods(attempt, data, args.seed, reference, api)
        replay = results['rosetta'].get('baseline_replay')
        detail = {'counts': COUNTS, 'algorithm_defaults': ALGORITHM, 'models': {m: MODEL_SPECS[m] for m in (args.vision, args.text)},
                  'split_sha256': digest(args.out / 'split.pt'), 'new_rosetta_fit': True,
                  'classification_rerun': False, 'baseline_replay': replay,
                  'baseline_replay_agrees': reference is None or (replay is not None and replay['agrees'])}
        complete = attempt.finish(detail, bound)
    except BaseException as error:
        attempt.abort(error)
        raise
    return 0 if complete else 1


if __name__ == '__main__':
    raise SystemExit(main())
