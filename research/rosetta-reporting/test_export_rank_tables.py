"""Synthetic presentation fixtures only; no real scientific output is verified."""
import copy
import csv
import gzip
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
import export_rank_tables as e


class ExportTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.inputs = self.root / 'inputs'
        self.inputs.mkdir()
        self.attempts = self.inputs / 'cells'
        self.attempts.mkdir()
        self.summary_path = self.inputs / 'summary.json'
        self.receipt_path = self.inputs / 'accepted.json'
        self.rows_patch = patch.object(e, 'ROWS', 4)
        self.rows_patch.start()
        self.addCleanup(self.rows_patch.stop)
        self.mapping = {
            'validation_image_ids': [30, 10, 40, 20],
            'validation_indices': [2, 0, 3, 1],
            'query_order': [2, 0, 3, 1], 'key_order': [1, 3, 0, 2],
            'partner_in_shuffled_keys': [3, 2, 1, 0],
            'sort_positions_by_image_id': [1, 3, 0, 2],
            'training_split_hashes': {'indices_x': 'a' * 64, 'indices_y': 'b' * 64},
        }
        self.record = {k: torch.tensor(self.mapping[k], dtype=torch.int64) for k in e.ORDER_FIELDS}
        self.record['ranks'] = torch.tensor([3.5, 1.0, 4.0, 2.0], dtype=torch.float64)
        stream = io.BytesIO()
        torch.save(self.record, stream)
        data = stream.getvalue()
        sha = e.digest(data)
        ranks = self.record['ranks']
        metrics = {'foscttm': ((ranks - 1) / 3).mean().item(), 'mean_rank': ranks.mean().item()}
        self.summary = {
            'schema_version': 1, 'protocol_id': 'rosetta-model-controls-v2', 'status': 'complete',
            'input_hashes': {**e.PINNED, 'private_bindings': 'a' * 64, 'runner_bindings': 'b' * 64},
            'planned': {'cells': 12, 'seeds': 5, 'method_slots': 300}, 'inventory_findings': [],
            'baseline_replay_gate': {'status': 'satisfied',
                'eligible_reference_cell_completions_by_seed': {str(s): 1 for s in e.SEEDS}},
            'attempts': {}, 'observations': {},
            'validation_order_and_coco_id_mapping': {str(s): copy.deepcopy(self.mapping) for s in e.SEEDS},
        }
        for v in e.VISION:
            for t in e.TEXT:
                for seed in e.SEEDS:
                    cell = f'{v}/{t}/{seed}'
                    selected = f'{v}--{t}/seed-{seed}/attempt-01'
                    terminal = 'FAILED.json' if (v, t) == ('mae', 'elmo') else 'COMPLETE.json'
                    attempt = {'attempt': selected, 'terminal_marker': terminal,
                        'terminal_manifest_verified': True, 'split_verified': True,
                        'terminal_partial_markers': [], 'findings': [], 'methods': {}, 'input_hashes': {}}
                    for method in e.METHODS:
                        p = self.attempts / selected / method / 'ranks.pt'
                        p.parent.mkdir(parents=True)
                        p.write_bytes(data)
                        attempt['input_hashes'][f'{method}/ranks.pt'] = sha
                        attempt['methods'][method] = {'status': 'complete', 'eligible': True,
                            'findings': [], 'ranks_sha256': sha, 'reported_metrics': metrics,
                            'rank_check': {'status': 'verified', 'order_ids_checked_against_original_rng': True,
                                'gallery_count_matches': True, 'validation_count': 4, 'ranks_dtype': 'float64',
                                'midrank_step': 0.5, 'recomputed_metrics': metrics,
                                'comparisons': {k: {'exactly_equal': True} for k in metrics},
                                'canonical_ranks_by_coco_id_sha256': e.object_hash([1.0, 2.0, 3.5, 4.0])}}
                        self.summary['observations'][f'{cell}/{method}'] = {'status': 'available',
                            'selected_attempt': selected, 'retry_history_findings': [],
                            'repeat_agreement': None, 'metrics': metrics}
                    self.summary['attempts'][cell] = [attempt]
        self.cell = f'dinov2/mpnet/{e.SEEDS[0]}'
        self.slot = f'{self.cell}/rosetta'
        self.rank_path = self.attempts / f'dinov2--mpnet/seed-{e.SEEDS[0]}/attempt-01/rosetta/ranks.pt'

    def export(self, name='export'):
        raw = (json.dumps(self.summary, indent=2) + '\n').encode()
        self.summary_path.write_bytes(raw)
        sha = e.digest(raw)
        receipt = json.dumps({'summary_sha256': sha}).encode()
        self.receipt_path.write_bytes(receipt)
        return e.export_tables(self.summary_path, sha, self.receipt_path, e.digest(receipt),
                               self.attempts, self.root / name)

    def test_sorted_alignment_and_all_identifiers_preserved(self):
        manifest = self.export()
        self.assertEqual(manifest['partition_count'], 300)
        target = self.root / 'export' / manifest['files'][0]['file']
        with gzip.open(target, 'rt', newline='') as f:
            rows = list(csv.DictReader(f))
        self.assertEqual([int(r['coco_image_id']) for r in rows], [10, 20, 30, 40])
        self.assertEqual([int(r['validation_source_row_index']) for r in rows], [0, 1, 2, 3])
        self.assertEqual([int(r['validation_subset_position']) for r in rows], [1, 3, 0, 2])
        self.assertEqual([float(r['midrank']) for r in rows], [1.0, 2.0, 3.5, 4.0])
        self.assertEqual({(r['vision'], r['text'], r['seed'], r['method']) for r in rows},
                         {('dinov2', 'mpnet', str(e.SEEDS[0]), 'rosetta')})
        for entry in manifest['files']:
            self.assertEqual(e.digest((self.root / 'export' / entry['file']).read_bytes()), entry['sha256'])
        self.assertTrue(any(f['cell_terminal'] == 'FAILED.json' for f in manifest['files']))
        before = (self.root / 'export' / 'manifest.json').read_bytes()
        with self.assertRaises(ValueError): self.export()
        self.assertEqual((self.root / 'export' / 'manifest.json').read_bytes(), before)

    def test_rejects_incomplete_unverified_retry_or_mapping_inputs(self):
        original = copy.deepcopy(self.summary)
        mutations = [
            lambda s: s.update(status='incomplete'),
            lambda s: s['attempts'][self.cell][0]['methods']['rosetta'].update(eligible=False),
            lambda s: s['observations'][self.slot].update(retry_history_findings=['invalid_retry']),
            lambda s: s['observations'][self.slot].update(selected_attempt='../escape'),
            lambda s: s['validation_order_and_coco_id_mapping'][str(e.SEEDS[0])].update(sort_positions_by_image_id=[0, 1, 2, 3]),
        ]
        for i, mutation in enumerate(mutations):
            with self.subTest(case=i):
                self.summary = copy.deepcopy(original)
                mutation(self.summary)
                with self.assertRaises(ValueError): self.export(f'invalid-{i}')
                self.assertFalse((self.root / f'invalid-{i}').exists())

    def test_rejects_changed_artifact_bytes(self):
        with self.rank_path.open('ab') as f: f.write(b'changed')
        with self.assertRaisesRegex(ValueError, 'SHA-256'): self.export()
        self.assertFalse((self.root / 'export').exists())

    def test_rejects_mapping_mismatch_even_with_updated_artifact_hash(self):
        self.record['validation_image_ids'] = torch.tensor([10, 30, 40, 20], dtype=torch.int64)
        torch.save(self.record, self.rank_path)
        sha = e.digest(self.rank_path.read_bytes())
        attempt = self.summary['attempts'][self.cell][0]
        attempt['input_hashes']['rosetta/ranks.pt'] = sha
        attempt['methods']['rosetta']['ranks_sha256'] = sha
        with self.assertRaisesRegex(ValueError, 'ID/order'): self.export()

    def test_rejects_output_inside_scientific_input_tree(self):
        with self.assertRaisesRegex(ValueError, 'outside'): self.export('inputs/cells/export')
        self.assertFalse((self.attempts / 'export').exists())

    def test_rejects_mismatched_acceptance_receipt(self):
        raw = json.dumps(self.summary).encode()
        self.summary_path.write_bytes(raw)
        receipt = json.dumps({'summary_sha256': '0' * 64}).encode()
        self.receipt_path.write_bytes(receipt)
        with self.assertRaisesRegex(ValueError, 'receipt differs'):
            e.export_tables(self.summary_path, e.digest(raw), self.receipt_path, e.digest(receipt),
                            self.attempts, self.root / 'export')


if __name__ == '__main__':
    unittest.main()
