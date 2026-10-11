"""Artificial mask checks plus optional read-only checks of the prior metadata.

Run with unittest. Set ROSETTA_AUDIT_CACHE and ROSETTA_ARCHIVED_MANIFESTS to the
existing public metadata cache and old manifests.tar.gz to enable the audit
checks. They validate existing masks only; no new real masks are issued or saved.
No embeddings, models, private photographs, inference or fits are used.
"""
from collections import Counter
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import tarfile
import unittest
import zipfile

import source_masks as sm


def artificial_records():
    coco = [{'cocoid': i, 'filepath': split} for split, ids in
            [('val2014', (50, 30, 40)), ('train2014', (20, 10))] for i in ids]
    mapping = {100: 10, 101: 10, 102: 20, 103: 30, 104: 30, 105: 40,
               200: None, 201: None, 202: None, 203: None, 204: 999}
    vg = [{'image_id': k, 'coco_id': v} for k, v in mapping.items()]
    paragraphs = [{'image_id': i} for i in (100, 200, 103, 102, 201, 104, 101, 102, 200, 105, 202, 203, 204)]
    return coco, paragraphs, vg


class SyntheticMasks(unittest.TestCase):
    def setUp(self):
        self.rows = sm.source_rows_from_records(*artificial_records())

    def masks(self):
        return sm.build_masks(self.rows, validation_seeds=(0, 2), pairing_seeds=(0, 2))

    def test_order_aliases_repeated_paragraphs_and_unknowns(self):
        self.assertEqual(self.rows.train_x, ('coco:10', 'coco:20'))
        self.assertEqual(self.rows.validation, ('coco:30', 'coco:40', 'coco:50'))
        self.assertEqual(self.rows.train_y_vg_ids, (100, 200, 103, 102, 201, 104, 101, 102, 200, 105, 202, 203, 204))
        groups = sm.source_groups(self.rows)
        self.assertEqual(groups['coco:10'], (0, 6))  # Different VG IDs, one COCO source.
        self.assertEqual(groups['coco:30'], (2, 5))  # Second alias, across the validation boundary.
        self.assertEqual(groups['coco:20'], (3, 7))  # Repeated paragraph source.
        self.assertEqual(groups['vg:200'], (1, 8))
        self.assertEqual(groups['coco:999'], (12,))  # A mapping outside known COCO references is not unmapped.

    def test_exact_targeted_rows_and_constant_full_image_bank(self):
        masks = self.masks()
        self.assertEqual([m.variant for m in masks[:3]], [sm.ORIGINAL, sm.VALIDATION, sm.BOTH])
        self.assertEqual(masks[0].train_y_indices, tuple(range(13)))
        self.assertEqual(masks[1].train_y_indices, (0, 1, 3, 4, 6, 7, 8, 10, 11, 12))
        self.assertEqual(masks[2].train_y_indices, (1, 4, 8, 10, 11, 12))
        for mask in masks:
            self.assertEqual(sm.composition(self.rows, mask)['training_image_rows'], 2)
            selected = set(mask.train_y_indices)
            for group in sm.source_groups(self.rows).values():
                self.assertTrue(not selected.intersection(group) or set(group) <= selected)

    def test_frozen_synthetic_answers_match_prior_audited_sampler(self):
        # Expected indices obtained from the prior builder on this artificial fixture.
        expected = [(0, 1, 2, 3, 5, 6, 7, 8, 11, 12), (0, 2, 3, 4, 5, 6, 7, 9, 10, 11),
                    (0, 3, 6, 7, 11, 12), (0, 3, 4, 6, 7, 10)]
        self.assertEqual([m.train_y_indices for m in self.masks()[3:]], expected)
        self.assertEqual(self.masks(), self.masks())
        self.assertEqual([len(m.train_y_indices) for m in self.masks()[3:]], [10, 10, 6, 6])

    def test_sampler_failure_does_not_split_or_choose_another_seed(self):
        # This greedy order cannot fill ten rows, despite feasible other subsets.
        with self.assertRaisesRegex(ValueError, 'Cannot fill exact'):
            sm.build_masks(self.rows, validation_seeds=(1,), pairing_seeds=())
        with self.assertRaisesRegex(ValueError, 'source group was split'):
            sm.sample_source_groups(self.rows, (0,), 1, 0)
        with self.assertRaisesRegex(ValueError, 'Cannot fill exact'):
            sm.sample_source_groups(self.rows, (0, 6), 1, 0)
        self.assertEqual(sm.sample_source_groups(self.rows, (0, 6), 0, 0), ())
        self.assertEqual(sm.sample_source_groups(self.rows, (0, 6), 2, 0), (0, 6))

    def test_no_implicit_budget_or_invalid_seed_lists(self):
        with self.assertRaises(TypeError):
            sm.build_masks(self.rows)
        for seeds in ((0, 0), (-1,), (True,), [0]):
            with self.subTest(seeds=seeds), self.assertRaises(ValueError):
                sm.build_masks(self.rows, validation_seeds=seeds, pairing_seeds=())

    def test_fixed_strata_survive_removal_and_partition_full_query_population(self):
        strata = sm.fixed_strata(self.rows)
        self.assertEqual(strata, {'never_known_spc_overlap': (2,), 'original_spc_overlap': (0, 1), 'all': (0, 1, 2)})
        clean_sources = {self.rows.train_y[i] for i in self.masks()[2].train_y_indices}
        self.assertFalse(clean_sources & set(self.rows.validation))
        self.assertEqual(sm.fixed_strata(self.rows), strata)  # Exposure definition did not disappear.
        self.assertFalse(set(strata['never_known_spc_overlap']) & set(strata['original_spc_overlap']))
        self.assertEqual(sorted(strata['never_known_spc_overlap'] + strata['original_spc_overlap']), list(strata['all']))

    def test_composition_and_intersections_count_sources_not_only_vg_ids(self):
        original, clean, both = self.masks()[:3]
        summary = sm.composition(self.rows, original)
        self.assertEqual((summary['paragraph_rows'], summary['source_groups'], summary['duplicate_source_rows']), (13, 9, 4))
        self.assertEqual(summary['group_size_histogram'], {1: 5, 2: 4})
        self.assertEqual(summary['source_membership_rows'], {'known_coco_train': 4, 'known_coco_validation': 3,
                                                            'no_coco_mapping': 5, 'mapped_coco_outside_reference': 1})
        self.assertEqual(summary['source_membership_groups'], {'known_coco_train': 2, 'known_coco_validation': 2,
                                                              'no_coco_mapping': 4, 'mapped_coco_outside_reference': 1})
        overlap = sm.intersection(self.rows, clean, both)
        self.assertEqual((overlap['shared_rows'], overlap['shared_source_groups']), (6, 5))
        self.assertEqual((overlap['union_rows'], overlap['union_source_groups']), (10, 7))
        self.assertEqual(overlap['shared_known_validation_sources'], 0)

    def test_masks_bind_source_order_seed_and_membership(self):
        original, clean, both, random = self.masks()[:4]
        bad = [replace(original, row_identity_sha256='0' * 64), replace(clean, train_y_indices=tuple(range(10))),
               replace(both, train_y_indices=(1, 4, 8, 10, 11)), replace(random, selection_seed=7),
               replace(original, train_y_indices=(0, 0)), replace(original, train_y_indices=(6, 0)),
               replace(original, train_y_indices=(True,)), replace(original, train_y_indices=(13,)),
               replace(original, variant='certified_clean')]
        for mask in bad:
            with self.subTest(mask=mask), self.assertRaises(ValueError):
                sm.validate_mask(self.rows, mask)
        changed = replace(self.rows, train_y=self.rows.train_y[::-1], train_y_vg_ids=self.rows.train_y_vg_ids[::-1])
        self.assertNotEqual(changed.identity_sha256, self.rows.identity_sha256)
        self.assertNotEqual(replace(random, selection_seed=7).mask_sha256, random.mask_sha256)

    def test_standalone_json_rows_need_explicit_archive_and_identity_bindings(self):
        expected = {'synthetic.zip': 'a' * 64}
        document = {'schema_version': 1, 'source_sha256': expected, 'rows': self.rows.payload(),
                    'row_identity_sha256': self.rows.identity_sha256}
        self.assertEqual(sm.source_rows_from_document(document, expected_source_sha256=expected), self.rows)
        with self.assertRaisesRegex(ValueError, 'bindings'):
            sm.source_rows_from_document(document, expected_source_sha256={'synthetic.zip': 'b' * 64})
        with self.assertRaisesRegex(ValueError, 'content hash'):
            sm.source_rows_from_document({**document, 'row_identity_sha256': '0' * 64}, expected_source_sha256=expected)

    def test_invalid_source_records_fail_before_selection(self):
        for case in ('duplicate_coco', 'wrong_split', 'wrong_filename', 'duplicate_vg', 'missing_vg', 'boolean_id', 'bad_mapping'):
            coco, paragraphs, vg = artificial_records()
            if case == 'duplicate_coco': coco.append(dict(coco[0]))
            if case == 'wrong_split': coco[0]['filepath'] = 'restval'
            if case == 'wrong_filename': coco[0]['filename'] = 'wrong.jpg'
            if case == 'duplicate_vg': vg.append(dict(vg[0]))
            if case == 'missing_vg': paragraphs[0]['image_id'] = 10000
            if case == 'boolean_id': paragraphs[0]['image_id'] = True
            if case == 'bad_mapping': vg[0]['coco_id'] = '10'
            with self.subTest(case=case), self.assertRaises(ValueError):
                sm.source_rows_from_records(coco, paragraphs, vg)
        with self.assertRaisesRegex(ValueError, 'conflicting'):
            replace(self.rows, train_y_vg_ids=(100, 200, 100) + self.rows.train_y_vg_ids[3:])
        with self.assertRaises(ValueError):
            replace(self.rows, train_x=('coco:010', 'coco:20'))
        with self.assertRaises(ValueError):
            replace(self.rows, validation=('coco:10', 'coco:30'))


@unittest.skipUnless(os.environ.get('ROSETTA_AUDIT_CACHE') and os.environ.get('ROSETTA_ARCHIVED_MANIFESTS'),
                     'Explicit existing public metadata paths required; no download or search')
class ExistingMetadata(unittest.TestCase):
    def test_rebuild_identities_strata_and_validate_preexisting_masks_only(self):
        cache = Path(os.environ['ROSETTA_AUDIT_CACHE'])
        expected_hashes = {
            'caption_datasets.zip': '4cfd70132527b80933105e5829dc9034eaab9573482e2e680abbab6130244817',
            'vg_image_data.json.zip': 'b87a94918cb2ff4d952cf1dfeca0b9cf6cd6fd204c2f8704645653be1163681a',
            'paragraphs_v1.json.zip': '8407f1068e0711e5f6550d641c523ee26984ee8e02c1d51564369cccd56e8b26'}
        records = {}
        for filename, expected in expected_hashes.items():
            path = cache / filename
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), expected)
            member = {'caption_datasets.zip': 'dataset_coco.json', 'vg_image_data.json.zip': 'image_data.json',
                      'paragraphs_v1.json.zip': 'paragraphs_v1.json'}[filename]
            with zipfile.ZipFile(path) as archive:
                matches = [n for n in archive.namelist() if Path(n).name == member]
                self.assertEqual(len(matches), 1)
                records[filename] = json.loads(archive.read(matches[0]))
        rows = sm.source_rows_from_records(records['caption_datasets.zip']['images'], records['paragraphs_v1.json.zip'],
                                           records['vg_image_data.json.zip'])
        self.assertEqual(rows.identity_sha256, 'e4e0e295e82ead25e7afca9a81b4d0c9513e6f56c64247848f6cb5bf9006b455')
        self.assertEqual(tuple(map(len, (rows.train_x, rows.train_y, rows.validation))), (82783, 19561, 40504))
        groups = sm.source_groups(rows)
        self.assertEqual(Counter(map(len, groups.values())), {1: 19537, 2: 12})
        for source, expected in [('coco:112040', {286037, 2340703}), ('coco:322194', {286065, 2322189})]:
            self.assertEqual({rows.train_y_vg_ids[i] for i in groups[source]}, expected)
        strata = sm.fixed_strata(rows)
        self.assertEqual({k: len(v) for k, v in strata.items()},
                         {'never_known_spc_overlap': 37169, 'original_spc_overlap': 3335, 'all': 40504})
        checked = 0
        with tarfile.open(os.environ['ROSETTA_ARCHIVED_MANIFESTS']) as archive:
            saved_rows = json.load(archive.extractfile('manifests/source_rows.json'))
            self.assertEqual(saved_rows['rows'], rows.payload())
            self.assertEqual(saved_rows['source_sha256'], expected_hashes)
            self.assertEqual(sm.source_rows_from_document(saved_rows, expected_source_sha256=expected_hashes), rows)
            for member in archive.getmembers():
                if not member.name.endswith('.json') or Path(member.name).name in ('source_rows.json', 'summary.json'):
                    continue
                record = json.load(archive.extractfile(member))
                self.assertEqual(record['manifest_sha256'], sm.canonical_hash({k: v for k, v in record.items() if k != 'manifest_sha256'}))
                mask = sm.SourceMask(record['variant'], record['selection_seed'], record['row_identity_sha256'],
                                     tuple(record['indices']['train_y']))
                sm.validate_mask(rows, mask)
                summary = sm.composition(rows, mask)
                self.assertEqual(summary['paragraph_rows'], record['audit']['n_train_y'])
                self.assertEqual(summary['duplicate_source_rows'], record['audit']['train_y_duplicate_source_rows'])
                if mask.variant == sm.BOTH:
                    self.assertEqual(summary['source_membership_rows']['no_coco_mapping'], 9963)
                    self.assertEqual(summary['source_membership_groups']['no_coco_mapping'], 9956)
                checked += 1
        self.assertEqual(checked, 13)


if __name__ == '__main__':
    unittest.main()
