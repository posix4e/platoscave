#!/usr/bin/env python3
"""Compare all five original results to references frozen before fitting.

Standard library only. The output contains numbers, generic configuration fields
and hashes; it excludes hostnames, filesystem paths, timestamps and raw inputs.
Each agreement flag concerns this particular configuration and available metric,
not the whole paper. A non-matching number is reported rather than discarded.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics

from run_locked import (canonical, digest, expected_configurations, expected_result_name,
                        read_json, release_state, require)


def load_results(results, protocol):
    results = Path(results)
    paths = sorted(results.glob('*.json'))
    require(len(paths) == 5, 'Exactly five original result JSON files are required')
    require(not list(results.glob('*.tmp')), 'An unfinished result file is present')
    configs = expected_configurations(protocol)
    expected = {c['seed']: c for c in configs}
    records = {}
    for path in paths:
        record = read_json(path)
        config = record['config']
        seed = config.get('seed')
        require(isinstance(seed, int) and not isinstance(seed, bool) and seed in expected, 'Unexpected result seed')
        require(seed not in records, 'Duplicate result seed')
        require(canonical(config) == canonical(expected[seed]), 'Result configuration differs from the frozen selection')
        require(path.name == expected_result_name(config), 'Result filename differs from the authors configuration ID')
        require(record.get('git_commit') == protocol['upstream_commit'], 'Unexpected result source commit')
        for package in ('torch', 'numpy', 'scipy', 'scikit-learn', 'pylibmgm'):
            require(record.get('versions', {}).get(package) == protocol['expected_versions'][package],
                    f'Missing or incorrect recorded version: {package}')
        metrics = {name: record['metrics'].get(name) for name in protocol['required_metrics']}
        require(all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)
                    for v in metrics.values()), 'A required metric is missing or nonfinite')
        n = protocol['expected_validation_count']
        require(metrics['num_validation_samples'] == n, 'The complete validation set was not evaluated')
        require(0 <= metrics['foscttm'] <= 1 and 1 <= metrics['mean_rank'] <= n, 'Invalid retrieval metric range')
        require(abs(metrics['foscttm'] - (metrics['mean_rank'] - 1) / (n - 1)) <= 1e-12,
                'FOSCTTM and mean rank are inconsistent')
        require(metrics['fit_and_evaluation_seconds'] >= 0, 'Invalid duration')
        for name, value in metrics.items():
            if name.endswith(('_top1', '_top5')):
                require(0 <= value <= 1, 'Classification accuracy is outside [0,1]')
        records[seed] = {'seed': seed, 'result_id': path.stem, 'raw_result_sha256': digest(path), 'metrics': metrics}
    require(set(records) == set(expected), 'A predetermined seed is missing')
    return [records[seed] for seed in protocol['seeds']]


def numeric_comparison(observed, reference, tolerance):
    error = observed - reference
    return {'observed': observed, 'reference': reference, 'signed_error': error,
            'absolute_error': abs(error), 'absolute_tolerance': tolerance,
            'strict_numerical_agreement': abs(error) <= tolerance}


def display_comparison(observed, reference_display, decimal_places, units):
    # Strip the SD's printed ± sign. No assumption of an unrounded value is made.
    reference_display = str(reference_display).removeprefix('±')
    observed_display = f'{observed:.{decimal_places}f}'
    center = float(reference_display)
    error = observed - center
    return {'observed': observed, 'observed_display': observed_display,
            'reference_display': reference_display, 'decimal_places': decimal_places, 'units': units,
            'paper_display_agreement': observed_display == reference_display,
            'signed_error_from_displayed_value': error, 'absolute_error_from_displayed_value': abs(error),
            'error_reference_is_rounded': True}


def compare(protocol, reference, records):
    """Pure deterministic arithmetic, separate from input-file validation."""
    require([r['seed'] for r in records] == protocol['seeds'], 'Comparison requires every seed in protocol order')
    summaries = {}
    for metric in protocol['required_metrics']:
        values = [record['metrics'][metric] for record in records]
        require(len(values) == 5 and all(math.isfinite(v) for v in values), 'Invalid aggregate inputs')
        summaries[metric] = {'count': len(values), 'mean': statistics.mean(values),
                             'sample_standard_deviation': statistics.stdev(values), 'ddof': 1}
    policy = protocol['comparison']
    tolerance = policy['strict_numerical_agreement_absolute_tolerance']
    require(tolerance == 1e-10, 'This comparison requires the original fixed 1e-10 tolerance')
    precise = policy['full_precision_foscttm']
    ref = reference['retrieval_reference']
    require(precise == {'mean': ref['foscttm'], 'sample_sd': ref['foscttm_std'],
                        'specific_seed': ref['seed'], 'specific_seed_value': ref['foscttm_seed']},
            'Protocol and archived retrieval reference disagree')
    observed_seed = next(r['metrics']['foscttm'] for r in records if r['seed'] == precise['specific_seed'])
    exact = {
        'foscttm_mean': numeric_comparison(summaries['foscttm']['mean'], precise['mean'], tolerance),
        'foscttm_sample_sd': numeric_comparison(summaries['foscttm']['sample_standard_deviation'], precise['sample_sd'], tolerance),
        'foscttm_named_seed': {**numeric_comparison(observed_seed, precise['specific_seed_value'], tolerance),
                               'seed': precise['specific_seed']},
    }
    precision = policy['paper_display_agreement']
    places = precision['foscttm_mean_and_sample_sd_decimal_places']
    require(places == 2, 'Unexpected paper retrieval display precision')
    paper = {'foscttm': {
        'mean': display_comparison(summaries['foscttm']['mean'], reference['paper_foscttm_display']['mean'], places, 'fraction'),
        'sample_standard_deviation': display_comparison(summaries['foscttm']['sample_standard_deviation'],
                                                        reference['paper_foscttm_display']['std'], places, 'fraction'),
    }}
    places = precision['classification_mean_and_sample_sd_percent_decimal_places']
    require(places == 0, 'Unexpected paper classification display precision')
    classification = reference['classification_reference']
    require(set(classification) == {'CIFAR-10_top1', 'CIFAR-100_top5', 'ImageNet-100_top5'},
            'Unexpected classification reference set')
    for metric, ref_metric in classification.items():
        paper[metric] = {
            'mean': display_comparison(100 * summaries[metric]['mean'], ref_metric['mean_percent_display'], places, 'percent'),
            'sample_standard_deviation': display_comparison(100 * summaries[metric]['sample_standard_deviation'],
                                                            ref_metric['sample_std_percent_display'], places, 'percentage_points'),
        }
    return {'per_seed': records, 'aggregates': summaries, 'full_precision_comparisons': exact,
            'paper_display_comparisons': paper,
            'scope': 'One complete published zero-pair DINOv2 B/14 mean × MPNet MS COCO configuration, over the five predetermined seeds.',
            'interpretation': {
                'strict_tolerance_attribution': policy['strict_tolerance_attribution'],
                'paper_display_agreement_is_limited_to_printed_precision': True,
                'four_individual_author_seed_scores_unavailable': True,
                'unrounded_classification_references_unavailable': True,
                'aggregate_agreement_does_not_establish_each_seed_agrees': True,
                'whole_paper_verdict': 'Not assessed by this comparison.',
                'mismatches': 'Preserve each difference and investigate source, environment, input precision and random streams. No seed is discarded or replaced.',
            }}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release', type=Path, required=True)
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), 'Refusing to overwrite an existing comparison')
    release = args.release.resolve()
    require(Path(__file__).resolve() == release / 'compare.py', 'Use the comparison script from the frozen release')
    locked = release_state(release)
    require(locked['files'].get('compare.py') == digest(__file__), 'Comparison script is absent from or differs from release lock')
    protocol = read_json(release / 'protocol.json')
    reference_name = protocol['comparison']['reference_file']
    require(reference_name == 'reference.json', 'Unexpected reference file')
    reference = read_json(release / reference_name)
    records = load_results(args.results, protocol)
    report = {'schema_version': 1, 'protocol_id': protocol['protocol_id'],
              'release_lock_sha256': locked['release_lock_sha256'],
              'protocol_sha256': locked['files']['protocol.json'], 'reference_sha256': locked['files'][reference_name],
              'comparison_script_sha256': locked['files']['compare.py'], **compare(protocol, reference, records)}
    require(release_state(release) == locked, 'Release changed during comparison')
    require(load_results(args.results, protocol) == records, 'Raw results changed during comparison')
    # Exclusive creation preserves any earlier report. A failed write is retained, never replaced silently.
    with args.output.open('x') as stream:
        json.dump(report, stream, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'status': 'comparison_written', 'sha256': digest(args.output), 'seeds': len(records)}))


if __name__ == '__main__':
    main()
