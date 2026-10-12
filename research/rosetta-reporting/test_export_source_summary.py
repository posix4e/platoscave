"""Synthetic four-row source study, all 115 fits; no real experiment outputs."""
import builtins
import copy
import json
from itertools import combinations
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import audit_rank_summaries as audit
import export_source_summary as projection
import test_audit_rank_summaries as fixture_module

encode = fixture_module.encode


class SourceSummaryTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixture_module.AuditTests('test_even_median_ties_and_independent_sample_sd')
        self.fixture.setUp(); self.addCleanup(self.fixture.doCleanups)
        self.fixture.prepare('source')
        self.root, self.inputs, self.summary = self.fixture.root, self.fixture.inputs, self.fixture.summary
        self.args = SimpleNamespace(summary=self.inputs/'summary.json', acceptance=self.inputs/'accepted.json',
            stage_receipt=self.inputs/'source-stage-complete.json', protocol=self.inputs/'protocol.json',
            masks=self.inputs/'masks.json', compositions=self.inputs/'compositions.json', intersections=self.inputs/'intersections.json',
            audit=self.inputs/'source-audit.json', out=self.root/'projection'/'source-summary.json')
        self.protocol = json.loads(self.args.protocol.read_bytes())
        self.protocol['counts'].update(train_x=4, original_y=4)
        self.protocol['arm_counts'] = {'O': 4, 'C': 3, 'D': 2, 'Rv': 3, 'Rp': 2}
        self.protocol['research_design'].update(stage='Invented exploratory study.', arms={'O': 'Invented original'},
            source_identity={'unknown': 'Unknown remains unknown.'}, random_removal={'uniform_row_or_admissible_subset_sampling': False},
            fitting={'known_pairs': False}, evaluation={'classification_extension': False}, interpretation_limits=['Invented limitation.'])
        self.bundle = json.loads(self.args.masks.read_bytes()); self.bundle.update(schema_version=1, status='frozen_before_source_control_fits')
        names = ['O', 'C', 'D', *[f'Rv-{i}' for i in range(1000, 1010)], *[f'Rp-{i}' for i in range(2000, 2010)]]
        common = {'schema_version': 1, 'operation': 'prepare_all_source_masks', 'protocol_id': 'rosetta-source-controls-v1',
            'row_identity_sha256': self.protocol['row_identity_sha256'],
            'release_lock_sha256': 'fc65f7a87b5956d97074e850cc7dc31b3f3df9316398d7fbdf2ec0a0f3205d31'}
        self.compositions = {**common, 'masks': {}}
        self.intersections = {**common, 'pairs': []}
        variants = {'O': 'original', 'C': 'validation_clean', 'D': 'fully_dedup',
            'Rv': 'random_drop_validation_size_matched', 'Rp': 'random_drop_size_matched'}
        for name in names:
            family = name.split('-')[0]; n = self.protocol['arm_counts'][family]
            seed = int(name.split('-')[1]) if '-' in name else None
            mask = self.bundle['masks'][name]
            mask.update(row_identity_sha256=self.protocol['row_identity_sha256'], selection_seed=seed,
                train_y_indices=list(range(n)), variant=variants[family])
            self.compositions['masks'][name] = {k: mask[k] for k in ('mask_sha256', 'row_identity_sha256', 'variant', 'selection_seed')}
            self.compositions['masks'][name].update(paragraph_rows=n, source_groups=n, duplicate_source_rows=0,
                training_image_rows=4, group_size_histogram={'1': n},
                source_membership_rows={'known_coco_train': 0, 'known_coco_validation': 0, 'mapped_coco_outside_reference': 0, 'no_coco_mapping': n},
                source_membership_groups={'known_coco_train': 0, 'known_coco_validation': 0, 'mapped_coco_outside_reference': 0, 'no_coco_mapping': n},
                scope='Known metadata identities only; no image-byte, near-duplicate or pretraining-cleanliness claim.')
        for left, right in combinations(names, 2):
            lrows = set(self.bundle['masks'][left]['train_y_indices']); rrows = set(self.bundle['masks'][right]['train_y_indices'])
            self.intersections['pairs'].append({'left': left, 'right': right,
                'left_mask_sha256': self.bundle['masks'][left]['mask_sha256'], 'right_mask_sha256': self.bundle['masks'][right]['mask_sha256'],
                'shared_rows': len(lrows & rrows), 'union_rows': len(lrows | rrows),
                'shared_source_groups': len(lrows & rrows), 'union_source_groups': len(lrows | rrows),
                'shared_known_training_sources': 0, 'shared_known_validation_sources': 0})
        self.summary['recorded_at_utc'] = '2026-10-12T00:00:00+00:00'
        self.summary['descriptive'].update(direction='Positive rank/FOSCTTM differences and negative Recall differences mean worse retrieval.',
            inference='Descriptive conditional variation only. Masks share sources; no independent-corpus, significance, equivalence or power claim.')
        for schedule in self.summary['schedule']:
            schedule['training_order_sha256'] = {'indices_x': 'a'*64, 'indices_y': 'b'*64, 'canonical_mask_y': 'c'*64}
        for identity, attempts in self.summary['attempts'].items():
            mask, seed = identity.split('/')
            item = attempts[0]; item.update(artifact_names=['COMPLETE.json', 'result.json', 'ranks.pt'],
                retry_review=None, retry_review_binding_verified=False, failure=None)
            item['input_hashes'].update({'map.pt': 'a'*64, 'training_order.pt': 'b'*64, 'metrics.json': 'c'*64,
                'result.json': 'd'*64, 'COMPLETE.json': 'e'*64})
            item['result'].update(schema_version=1, protocol_id='rosetta-source-controls-v1', mask=mask, seed=int(seed),
                status='complete', fit_status='complete', evaluation_status='complete', classification_rerun=False,
                mask_sha256=self.bundle['masks'][mask]['mask_sha256'], composition=copy.deepcopy(self.compositions['masks'][mask]),
                artifact_hashes={k: item['input_hashes'][k] for k in ('map.pt', 'training_order.pt', 'metrics.json', 'ranks.pt')})
            self.summary['observations'][identity]['attempt_count'] = 1
        self.pins = copy.deepcopy(projection.PINS); self.preparation_pins = copy.deepcopy(projection.PREPARATION_PINS)
        for key, value in [('PINS', self.pins), ('PREPARATION_PINS', self.preparation_pins), ('ROWS', 4),
                           ('STRATA', {'never_known_spc_overlap': 2, 'original_spc_overlap': 2, 'all': 4})]:
            p = patch.object(projection, key, value); p.start(); self.addCleanup(p.stop)
        self.write_preparation()
        self.audit_report = None
        self.write_inputs()
        # Produce a real independent arithmetic report on invented four-row tensors.
        self.fixture.args.study_receipt = self.args.stage_receipt
        self.fixture.args.study_receipt_sha256 = self.args.stage_receipt_sha256
        self.fixture.args.protocol_sha256 = self.args.protocol_sha256
        self.fixture.args.masks_sha256 = self.args.masks_sha256
        self.fixture.args.summary_sha256 = self.args.summary_sha256
        self.fixture.args.acceptance_sha256 = self.args.acceptance_sha256
        self.fixture.pins['source'].update(self.pins)
        audit.audit(self.fixture.args)
        self.audit_report = json.loads(self.fixture.args.out.read_bytes())
        self.write_report()

    def write_preparation(self):
        self.args.protocol.write_bytes(encode(self.protocol)); self.args.protocol_sha256 = projection.digest(self.args.protocol.read_bytes())
        self.pins['protocol'] = self.args.protocol_sha256
        self.bundle['protocol_sha256'] = self.args.protocol_sha256
        self.args.masks.write_bytes(encode(self.bundle)); self.args.masks_sha256 = projection.digest(self.args.masks.read_bytes())
        self.pins['masks'] = self.args.masks_sha256
        for name in ('compositions', 'intersections'):
            doc = getattr(self, name); doc.update(protocol_sha256=self.args.protocol_sha256, masks_sha256=self.args.masks_sha256)
            path = getattr(self.args, name); path.write_bytes(encode(doc)); sha = projection.digest(path.read_bytes())
            setattr(self.args, name + '_sha256', sha); self.preparation_pins[name] = sha
        self.summary['input_hashes'].update(self.pins)
        self.summary['input_hashes']['prespecified_reporting'] = projection.object_hash(self.protocol['research_design']['reporting'])
        for attempts in self.summary['attempts'].values():
            for attempt in attempts:
                if attempt.get('result'):
                    attempt['result'].update(protocol_sha256=self.args.protocol_sha256, masks_sha256=self.args.masks_sha256)

    def write_inputs(self):
        self.args.summary.write_bytes(encode(self.summary)); self.args.summary_sha256 = projection.digest(self.args.summary.read_bytes())
        self.args.acceptance.write_bytes(encode({'summary_sha256': self.args.summary_sha256}))
        self.args.acceptance_sha256 = projection.digest(self.args.acceptance.read_bytes())
        recorded = '/private/original/controller/verify-7777'
        self.ticket = {'kind': 'verification', 'study': 'source', 'preflight': False, 'final_audit': False,
            'out': recorded + '/summary.json', 'runner_bindings_sha256': self.summary['input_hashes']['runner_bindings'],
            'summary_bindings_sha256': self.summary['input_hashes']['private_bindings'],
            'argv': ['PRIVATE_ARGV_SENTINEL'], 'unit': 'PRIVATE_PROCESS_SENTINEL'}
        self.write_ticket()

    def write_ticket(self):
        data = encode(self.ticket); (self.inputs/'ticket.json').write_bytes(data)
        self.stage = {'summary_path': self.ticket['out'], 'summary_sha256': self.args.summary_sha256,
            'verification': str(Path(self.ticket['out']).parent),
            'ticket_sha256': fixture_module.fixture_controller_ticket_hash(self.ticket)}
        self.args.stage_receipt.write_bytes(encode(self.stage)); self.args.stage_receipt_sha256 = projection.digest(self.args.stage_receipt.read_bytes())
        if self.audit_report:
            self.audit_report['inputs'].update(summary_sha256=self.args.summary_sha256, acceptance_sha256=self.args.acceptance_sha256,
                study_receipt_sha256=self.args.stage_receipt_sha256, ticket_sha256=projection.digest(data), ticket_canonical_sha256=self.stage['ticket_sha256'])
            self.write_report()

    def write_report(self):
        self.args.audit.write_bytes(encode(self.audit_report)); self.args.audit_sha256 = projection.digest(self.args.audit.read_bytes())

    def run_export(self):
        original_import = builtins.__import__
        def guarded_import(name, *args, **kwargs):
            if name.split('.')[0] in {'run_cell', 'run_source_arm', 'summarize_source', 'source_masks', 'source_primitives', 'summarize_models'}:
                raise AssertionError('Scientific import forbidden')
            return original_import(name, *args, **kwargs)
        with patch.object(audit.torch, 'load', side_effect=AssertionError('Tensor load forbidden')), patch.object(builtins, '__import__', side_effect=guarded_import):
            result = projection.export_summary(self.args)
        return result, json.loads(self.args.out.read_bytes())

    def test_full_scalar_copy_audit_integration_and_privacy(self):
        result, output = self.run_export()
        self.assertEqual(len(output['observations']), 115)
        self.assertEqual(len(output['descriptive']['per_mask_five_seed']), 92)
        self.assertEqual(sum(len(v) for v in output['descriptive']['within_seed_random_mask_spread'].values()), 40)
        self.assertEqual(len(output['descriptive']['within_seed_contrasts']), 12)
        self.assertEqual(output['descriptive'], self.summary['descriptive'])
        self.assertEqual(output['mask_provenance']['masks'], self.compositions['masks'])
        self.assertEqual(output['mask_provenance']['pairwise_intersections'], self.intersections['pairs'])
        for identity, observation in output['observations'].items():
            self.assertEqual(observation['scores'], self.summary['observations'][identity]['scores'])
        counts = output['independent_audit']['comparison_counts']
        self.assertEqual({k: v['scalars'] for k, v in counts.items()},
            {'rank_metrics': 2760, 'aggregates_from_saved_scalars': 8976, 'aggregates_from_reconstructed_metrics': 8976})
        self.assertFalse(result['publication_authorized']); self.assertEqual(output['scientific_acceptance'], 'not_assessed')
        self.assertIsNone(output['publication_review_reference'])
        text = self.args.out.read_text()
        for private in ('/private/original', 'PRIVATE_PROCESS_SENTINEL', 'PRIVATE_ARGV_SENTINEL', 'canonical_coco_image_ids', 'train_y_indices'):
            self.assertNotIn(private, text)
        self.assertFalse(any('original_overall/' in key for key in output['descriptive']['within_seed_contrasts']))

    def test_failed_history_preserved_without_arbitrary_detail(self):
        identity = f'O/{projection.SEEDS[0]}'; attempts = self.summary['attempts'][identity]
        complete = attempts[0]; failed = copy.deepcopy(complete)
        failed.update(terminal_markers=['FAILED.json'], eligible=False, training_order_verified=False,
            failure={'error': '/Users/PRIVATE_FAILURE_SENTINEL'}, findings=[{'detail': 'PRIVATE_FINDING_SENTINEL'}])
        failed['input_hashes']['FAILED.json'] = 'f'*64
        failed['input_hashes']['/private/PRIVATE_HASH_KEY'] = 'a'*64
        failed['artifact_names'].append('/private/PRIVATE_ARTIFACT_NAME')
        complete['attempt'] = complete['attempt'].replace('attempt-01', 'attempt-02')
        complete.update(retry_review_binding_verified=True, retry_review={'reviewed': True, 'source_or_numerics_changed': False,
            'reason_kind': 'external_execution_failure', 'reason': 'PRIVATE_REASON_SENTINEL', 'previous_attempt': 'attempt-01',
            'previous_artifacts': {'FAILED.json': 'f'*64}})
        attempts.insert(0, failed)
        self.summary['observations'][identity].update(attempt_count=2, selected_attempt=complete['attempt'])
        self.write_inputs()
        next(x for x in self.audit_report['selected_artifacts'] if x['identity'] == identity)['selected_attempt'] = complete['attempt']
        self.write_report(); _, output = self.run_export()
        self.assertEqual(output['history_counts']['attempts'], 116)
        self.assertEqual(output['history_counts']['failed_terminal_attempts'], 1)
        history = output['attempt_history'][identity]
        self.assertEqual(history[0]['reported_metrics']['views']['original_overall']['foscttm'], failed['reported_metrics']['overall']['foscttm'])
        self.assertEqual(history[0]['failure']['canonical_json_sha256'], projection.object_hash(failed['failure']))
        self.assertEqual(history[1]['retry_review']['reason_kind'], 'external_execution_failure')
        self.assertNotIn('PRIVATE_', self.args.out.read_text())

    def test_complete_summary_cannot_hide_incomplete_or_duplicate_fit(self):
        identity = f'O/{projection.SEEDS[0]}'; original = copy.deepcopy(self.summary)
        changes = [lambda: self.summary['attempts'][identity][-1].update(terminal_markers=['FAILED.json']),
            lambda: self.summary['attempts'][identity][-1].update(eligible=False),
            lambda: self.summary['attempts'][identity][-1].update(terminal_manifest_verified=False),
            lambda: self.summary['attempts'][identity][-1].update(terminal_partial_markers=['COMPLETE.json.partial']),
            lambda: self.summary['attempts'][identity].append(copy.deepcopy(self.summary['attempts'][identity][0])),
            lambda: self.summary['observations'][identity].update(verified_complete_attempts=2),
            lambda: self.summary['attempts'][identity][-1]['result'].update(mask='C')]
        for change in changes:
            self.summary = copy.deepcopy(original); change(); self.write_inputs()
            with self.subTest(change=change), self.assertRaises(ValueError): projection.export_summary(self.args)

    def test_missing_masks_series_seeds_operands_and_wrong_sign_rejected(self):
        original = copy.deepcopy(self.summary)
        changes = [lambda: self.summary['observations'].pop(f'Rp-2009/{projection.SEEDS[-1]}'),
            lambda: self.summary['descriptive']['per_mask_five_seed'].pop('O/all'),
            lambda: self.summary['descriptive']['within_seed_random_mask_spread']['Rv/all'].pop(str(projection.SEEDS[0])),
            lambda: self.summary['descriptive']['within_seed_contrasts']['all/C_minus_O'].update(operands=['O', 'C']),
            lambda: self.summary['descriptive']['within_seed_contrasts']['all/C_minus_O']['individual'][str(projection.SEEDS[0])].update(missing_masks=['C']),
            lambda: self.summary['descriptive'].update(direction='Positive recall differences mean worse retrieval.'),
            lambda: self.summary['schedule'][0].update(mask='C')]
        for change in changes:
            self.summary = copy.deepcopy(original); change(); self.write_inputs()
            with self.subTest(change=change), self.assertRaises(ValueError): projection.export_summary(self.args)

    def test_audit_tampering_missing_coverage_and_selection_rejected(self):
        original = copy.deepcopy(self.audit_report)
        changes = [lambda: self.audit_report['inputs'].update(masks_sha256='0'*64),
            lambda: self.audit_report['selected_artifacts'][0].update(ranks_sha256='0'*64),
            lambda: self.audit_report['comparisons']['rank_metrics'].pop(),
            lambda: self.audit_report['comparisons']['aggregates_from_saved_scalars'][0].update(field='wrong'),
            lambda: self.audit_report['comparisons']['aggregates_from_reconstructed_metrics'][0].update(frozen=99.0),
            lambda: self.audit_report.update(scientific_acceptance='passed')]
        for change in changes:
            self.audit_report = copy.deepcopy(original); change(); self.write_report()
            with self.subTest(change=change), self.assertRaises(ValueError): projection.export_summary(self.args)

    def test_preparation_masks_counts_and_intersections_rejected(self):
        originals = [copy.deepcopy(v) for v in (self.bundle, self.compositions, self.intersections)]
        changes = [lambda: self.bundle['strata']['all'].pop(),
            lambda: self.compositions['masks']['O'].update(mask_sha256='0'*64),
            lambda: self.compositions['masks']['O'].update(paragraph_rows=99),
            lambda: self.intersections['pairs'].pop(),
            lambda: self.intersections['pairs'][0].update(left='D')]
        for change in changes:
            self.bundle, self.compositions, self.intersections = copy.deepcopy(originals)
            change(); self.write_preparation(); self.write_inputs()
            with self.subTest(change=change), self.assertRaises(ValueError): projection.export_summary(self.args)

    def test_receipt_and_ticket_canonical_hash_and_types(self):
        ticket = {'z': 1, 'a': 'ticket'}
        expected = b'{\n  "a": "ticket",\n  "z": 1\n}\n'
        self.assertEqual(projection.controller_ticket_hash(ticket), projection.digest(expected))
        self.assertNotEqual(projection.controller_ticket_hash(ticket), projection.object_hash(ticket))
        reordered = dict(reversed(list(self.ticket.items())))
        data = (json.dumps(reordered, separators=(',', ':')) + '\n').encode()
        (self.inputs/'ticket.json').write_bytes(data)
        self.audit_report['inputs']['ticket_sha256'] = projection.digest(data); self.write_report()
        self.run_export(); self.args.out.unlink()
        self.stage['ticket_sha256'] = projection.object_hash(self.ticket)
        self.args.stage_receipt.write_bytes(encode(self.stage)); self.args.stage_receipt_sha256 = projection.digest(self.args.stage_receipt.read_bytes())
        with self.assertRaisesRegex(ValueError, 'ticket'): projection.export_summary(self.args)
        for key in ('preflight', 'final_audit'):
            self.write_inputs(); self.ticket[key] = True; self.write_ticket()
            with self.assertRaisesRegex(ValueError, 'ticket'): projection.export_summary(self.args)
        self.write_inputs(); self.args.stage_receipt = self.inputs/'final-audit-source.json'
        self.args.stage_receipt.write_bytes(encode(self.stage)); self.args.stage_receipt_sha256 = projection.digest(self.args.stage_receipt.read_bytes())
        with self.assertRaisesRegex(ValueError, 'source-stage'): projection.export_summary(self.args)

    def test_nonzero_differences_not_an_approval_rule(self):
        row = self.audit_report['comparisons']['rank_metrics'][0]
        row.update(independent=row['frozen'] + .125, signed_difference_frozen_minus_independent=-.125,
            absolute_difference=.125, binary64_exactly_equal=False, exactly_equal_at_reported_precision=False)
        section = self.audit_report['comparisons']['rank_metrics']
        self.audit_report['comparison_counts']['rank_metrics'].update(binary64_differences=sum(not x['binary64_exactly_equal'] for x in section),
            nonzero_exact_or_decimal_differences=sum(not x['exactly_equal_at_reported_precision'] for x in section))
        self.write_report(); _, output = self.run_export()
        self.assertFalse(output['publication_authorized'])
        self.assertEqual(output['scientific_acceptance'], 'not_assessed')
        self.assertEqual(output['descriptive'], self.summary['descriptive'])

    def test_input_hash_duplicate_json_and_output_protection(self):
        original = self.args.acceptance.read_bytes(); self.args.acceptance.write_text('{}')
        with self.assertRaisesRegex(ValueError, 'SHA-256'): projection.export_summary(self.args)
        self.args.acceptance.write_bytes(original)
        self.args.audit.write_text('{"x":1,"x":2}'); self.args.audit_sha256 = projection.digest(self.args.audit.read_bytes())
        with self.assertRaisesRegex(ValueError, 'Duplicate'): projection.export_summary(self.args)
        self.write_report()
        for destination in (self.inputs/'new.json', self.root/'existing.json', self.root/'link.json'):
            self.args.out = destination
            if destination.name == 'existing.json': destination.write_text('preserve')
            if destination.name == 'link.json': destination.symlink_to(self.inputs/'missing.json')
            with self.subTest(destination=destination), self.assertRaisesRegex(ValueError, 'fresh output'): projection.export_summary(self.args)
        self.assertEqual((self.root/'existing.json').read_text(), 'preserve')


if __name__ == '__main__':
    unittest.main()
