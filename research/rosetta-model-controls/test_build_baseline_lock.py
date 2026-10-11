"""Artificial result JSON only; uses the unchanged original comparison code."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import build_baseline_lock as b


RELEASE = Path(__file__).resolve().parent.parent / 'rosetta-replication'


def write(path, value):
    Path(path).write_text(json.dumps(value, sort_keys=True, allow_nan=False) + '\n')


class BaselineLockTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.results = self.root / 'results'; self.results.mkdir()
        self.status_path, self.report_path = self.root / 'status.json', self.root / 'comparison.json'
        self.runner, self.comparison, self.release = b.frozen_api(RELEASE)
        self.protocol = b.read_json(RELEASE / 'protocol.json')
        reference = b.read_json(RELEASE / 'reference.json')
        source_lock = b.read_json(RELEASE / 'source-lock.json')
        inputs_lock = b.read_json(RELEASE / 'inputs-lock.json')
        configs = self.runner.expected_configurations(self.protocol)
        status_results = []
        for index, config in enumerate(configs):
            fos = .1 + index * .01
            metrics = {key: .5 for key in self.protocol['required_metrics']}
            metrics.update(foscttm=fos, mean_rank=1 + 40503 * fos,
                           num_validation_samples=40504, fit_and_evaluation_seconds=1.5)
            record = {'config': config, 'git_commit': self.protocol['upstream_commit'],
                      'versions': self.protocol['expected_versions'], 'metrics': metrics,
                      'hostname': 'PRIVATE_SYNTHETIC_HOST', 'path': '/private/synthetic-only'}
            path = self.results / self.runner.expected_result_name(config)
            write(path, record)
            status_results.append({'file': path.name, 'seed': config['seed'], 'sha256': b.digest(path), 'metrics': metrics})
        source = {'source_files_sha256': source_lock['files'], 'upstream_commit': self.protocol['upstream_commit'],
                  'bootstrap_receipt_sha256': self.protocol['bootstrap_receipt_sha256'],
                  'pixi_binary_sha256': source_lock['pixi_binary_sha256']}
        inputs = {entry['path']: {key: entry[key] for key in ('sha256', 'size_bytes')} for entry in inputs_lock['files']}
        self.status = {'schema_version': 1, 'status': 'complete', 'child_returncode': 0, 'signals_received': [],
                       'release_before': self.release, 'release_after': self.release,
                       'source_before': source, 'source_after': source, 'inputs_before': inputs, 'inputs_after': inputs,
                       'expected_result_files': [self.runner.expected_result_name(c) for c in configs],
                       'results': sorted(status_results, key=lambda row: row['file'])}
        records = self.comparison.load_results(self.results, self.protocol)
        self.report = {'schema_version': 1, 'protocol_id': self.protocol['protocol_id'],
                       'release_lock_sha256': self.release['release_lock_sha256'],
                       'protocol_sha256': self.release['files']['protocol.json'],
                       'reference_sha256': self.release['files']['reference.json'],
                       'comparison_script_sha256': self.release['files']['compare.py'],
                       **self.comparison.compare(self.protocol, reference, records)}
        write(self.status_path, self.status); write(self.report_path, self.report)

    def assemble(self):
        return b.assemble(RELEASE, self.results, self.status_path, self.report_path)

    def test_all_five_are_bound_without_paths_or_hostnames_and_mismatches_survive(self):
        lock = self.assemble()
        self.assertEqual(set(lock['results']), {str(seed) for seed in self.protocol['seeds']})
        for record in lock['results'].values():
            path = self.results / record['file']
            self.assertEqual(record['sha256'], b.digest(path))
            self.assertEqual(record['size_bytes'], path.stat().st_size)
        text = json.dumps(lock)
        self.assertNotIn('PRIVATE_SYNTHETIC_HOST', text)
        self.assertNotIn('/private/', text)
        self.assertNotIn(str(self.root), text)
        self.assertFalse(self.report['full_precision_comparisons']['foscttm_mean']['strict_numerical_agreement'])
        self.assertEqual(lock['comparison_sha256'], b.digest(self.report_path))
        self.assertIn('does not authorize', lock['scope'])

    def test_incomplete_or_interrupted_supervisor_is_rejected(self):
        changes = [('status', 'running'), ('child_returncode', 1), ('child_returncode', False),
                   ('signals_received', [15]), ('expected_result_files', [])]
        for key, value in changes:
            with self.subTest(field=key, value=value):
                status = copy.deepcopy(self.status); status[key] = value; write(self.status_path, status)
                with self.assertRaises(ValueError): self.assemble()

    def test_wrong_recorded_source_inputs_or_result_inventory_are_rejected(self):
        for kind in ('source', 'inputs', 'pixi', 'inventory'):
            with self.subTest(kind=kind):
                status = copy.deepcopy(self.status)
                if kind in ('source', 'inputs'):
                    status[kind + '_before'] = status[kind + '_after'] = {'unrelated': 'a' * 64}
                elif kind == 'pixi':
                    status['source_before']['pixi_binary_sha256'] = '0' * 64
                    status['source_after']['pixi_binary_sha256'] = '0' * 64
                else:
                    status['results'][0]['sha256'] = '0' * 64
                write(self.status_path, status)
                with self.assertRaises(ValueError): self.assemble()

    def test_missing_seed_and_unfinished_result_cannot_be_locked(self):
        path = next(self.results.glob('*.json')); raw = path.read_bytes(); path.unlink()
        with self.assertRaises(RuntimeError): self.assemble()
        path.write_bytes(raw)
        (self.results / 'unfinished.tmp').write_bytes(b'{')
        with self.assertRaises(RuntimeError): self.assemble()

    def test_changed_comparison_is_rejected_even_when_raw_results_are_complete(self):
        self.report['full_precision_comparisons']['foscttm_mean']['strict_numerical_agreement'] = True
        write(self.report_path, self.report)
        with self.assertRaisesRegex(ValueError, 'frozen recomputation'): self.assemble()

    def test_late_raw_result_change_is_rejected(self):
        original = b.frozen_api
        def api_with_late_change(release):
            runner, comparison, state = original(release)
            compare = comparison.compare
            def change_after_comparison(*args):
                result = compare(*args)
                path = next(self.results.glob('*.json')); path.write_bytes(path.read_bytes() + b' ')
                return result
            comparison.compare = change_after_comparison
            return runner, comparison, state
        with patch.object(b, 'frozen_api', side_effect=api_with_late_change):
            with self.assertRaises(ValueError): self.assemble()

    def test_existing_output_or_partial_file_is_preserved(self):
        lock = self.assemble(); out = self.root / 'lock.json'
        b.publish(out, lock); before = out.read_bytes()
        with self.assertRaises(ValueError): b.publish(out, lock)
        self.assertEqual(out.read_bytes(), before)
        other = self.root / 'another.json'; partial = self.root / 'another.json.partial'
        partial.write_bytes(b'{')
        with self.assertRaises(ValueError): b.publish(other, lock)
        self.assertEqual(partial.read_bytes(), b'{')
        self.assertFalse(other.exists())

    def test_duplicate_or_nonfinite_record_is_rejected(self):
        for invalid in ('{"status":"complete","status":"running"}', '{"value":NaN}'):
            with self.subTest(record=invalid):
                self.status_path.write_text(invalid)
                with self.assertRaises(ValueError): self.assemble()


if __name__ == '__main__':
    unittest.main()
