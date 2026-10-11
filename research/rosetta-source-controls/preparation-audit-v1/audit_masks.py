#!/usr/bin/env python3
"""Private independent audit; run only after the preparation release is published.

Reads prepared metadata and official archives, no images/features/fits. Imports
none of the production scientific helpers. Expected masks are reconstructed in
memory for comparison, never written as new experimental masks. The output is
an exclusive new audit receipt, never a modification of preparation artifacts.
"""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import zipfile

ARCHIVES = {'caption_datasets.zip': 'dataset_coco.json', 'vg_image_data.json.zip': 'image_data.json', 'paragraphs_v1.json.zip': 'paragraphs_v1.json'}
VARIANTS = {'O': 'original', 'C': 'validation_clean', 'D': 'fully_dedup', 'Rv': 'random_drop_validation_size_matched', 'Rp': 'random_drop_size_matched'}
SEEDS = [734796314, 576165995, 2197670066, 839703249, 2584932063]
PROTOCOL_SHA = '75076af19cdeeefd3d839e9eb2db9a33af2c47e17a7a0600339763b95b90eeae'
MEMBERS = {'started.json', 'journal.jsonl', 'source_rows.json', 'masks.json', 'compositions.json', 'intersections.json', 'schedule.json', 'result.json', 'COMPLETE.json'}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def unique(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, 'Duplicate JSON field')
        result[key] = value
    return result


def read(path):
    return json.loads(Path(path).read_text(), object_pairs_hook=unique)


def integer(value):
    return type(value) is int and value >= 0


def same_fields(record, expected):
    require(all(record.get(k) == v for k, v in expected.items()), 'Artifact binding differs')


def audit(args):
    require(re.fullmatch(r'[0-9a-f]{40}', args.published_commit), 'Explicit published commit required')
    require(args.protocol_sha256 == PROTOCOL_SHA, 'Unexpected independently selected protocol')
    out = args.prepared.resolve(strict=True)
    bound = {}
    def bind(path, expected=None):
        path = Path(path).resolve(strict=True)
        observed = digest(path)
        require(expected is None or observed == expected, 'Input/output digest differs')
        if path in bound:
            require(bound[path] == observed, 'Bound bytes changed')
        bound[path] = observed
        return path
    protocol = read(bind(args.protocol, args.protocol_sha256))
    release = read(bind(args.preparation_release, args.preparation_release_sha256))
    require(protocol['status'] == release['status'] == 'frozen_before_source_control_fits', 'Preparation protocol/release is not frozen')
    require(protocol['research_design']['budget']['status'] == 'frozen_before_source_control_masks_and_fits' and
            protocol['research_design']['budget']['fits'] == 115, 'Wrong fixed source budget')
    require(protocol['seeds'] == SEEDS and protocol['selection_seeds'] == {'Rv': list(range(1000, 1010)), 'Rp': list(range(2000, 2010))}, 'Wrong complete seed schedule')
    same_fields(release, {'schema_version': 1, 'protocol_id': protocol['protocol_id'], 'protocol_sha256': args.protocol_sha256})
    require(set(release['code_sha256']) == {'prepare_masks.py', 'source_masks.py', 'source_primitives.py', 'run_source_arm.py', 'run_cell.py'}, 'Code closure differs')
    for name, expected in protocol['code_sha256'].items():
        require(release['code_sha256'][name] == expected, 'Protocol and preparation helper hashes differ')
    for name, expected in release['code_sha256'].items():
        path = args.code_root / ('rosetta-model-controls' if name == 'run_cell.py' else 'rosetta-source-controls') / name
        bind(path, expected)
    require(release.get('operation') == 'prepare_all_declared_source_masks_only' and release.get('fits_authorized') is False, 'Release scope differs')
    for name, expected in release['supporting_sha256'].items():
        require(Path(name).name == name, 'Supporting metadata must be a direct release file')
        bind(args.preparation_release.parent / name, expected)
    bindings = read(bind(args.bindings))
    require(set(bindings) == {'archives'} and set(bindings['archives']) == set(ARCHIVES), 'Private archive bindings differ')
    contents, member_receipts = {}, {}
    for archive_name, basename in ARCHIVES.items():
        archive_path = bind(bindings['archives'][archive_name], protocol['source_archive_sha256'][archive_name])
        with zipfile.ZipFile(archive_path) as archive:
            matches = [x for x in archive.infolist() if not x.is_dir() and Path(x.filename).name == basename]
            require(len(matches) == 1, 'Expected one exact metadata ZIP member')
            payload = archive.read(matches[0])  # also checks member CRC
        contents[archive_name] = json.loads(payload, object_pairs_hook=unique)
        member_receipts[archive_name] = {'sha256': bound[archive_path], 'size_bytes': archive_path.stat().st_size,
            'member': matches[0].filename, 'member_sha256': hashlib.sha256(payload).hexdigest(), 'member_size_bytes': len(payload)}
    # Independent join: preserve paragraph rows, sort original COCO IDs, and
    # collapse all mapped VG aliases onto one canonical COCO source.
    coco = {'train2014': set(), 'val2014': set()}; seen = set()
    for row in contents['caption_datasets.zip']['images']:
        number, split = row['cocoid'], row['filepath']
        require(integer(number) and number not in seen and split in coco, 'Invalid COCO identity/split')
        require(row.get('filename', f'COCO_{split}_{number:012d}.jpg') == f'COCO_{split}_{number:012d}.jpg', 'COCO filename differs')
        coco[split].add(number); seen.add(number)
    mapped = {}
    for row in contents['vg_image_data.json.zip']:
        number, target = row['image_id'], row.get('coco_id')
        require(integer(number) and number not in mapped and (target is None or integer(target)), 'Invalid VG mapping')
        mapped[number] = target
    paragraph_vg, paragraph_source = [], []
    for row in contents['paragraphs_v1.json.zip']:
        vg = row['image_id']; require(integer(vg) and vg in mapped, 'Paragraph lacks VG metadata')
        paragraph_vg.append(vg)
        paragraph_source.append(f'vg:{vg}' if mapped[vg] is None else f'coco:{mapped[vg]}')
    rows = {'train_x': [f'coco:{i}' for i in sorted(coco['train2014'])], 'train_y': paragraph_source,
            'validation': [f'coco:{i}' for i in sorted(coco['val2014'])], 'train_y_vg_ids': paragraph_vg}
    identity = canonical_hash(rows)
    require(identity == protocol['row_identity_sha256'], 'Independent canonical source join differs')
    require([len(rows[k]) for k in ('train_x', 'train_y', 'validation')] == [82783, 19561, 40504], 'Full source populations differ')
    groups, aliases = defaultdict(set), defaultdict(set)
    for index, (source, vg) in enumerate(zip(paragraph_source, paragraph_vg)):
        groups[source].add(index); aliases[source].add(vg)
    source_document = {'schema_version': 1, 'source_sha256': protocol['source_archive_sha256'], 'row_identity_sha256': identity, 'rows': rows}
    expected_rows_bytes = (json.dumps(source_document, separators=(',', ':')) + '\n').encode()
    source_rows_path = bind(out / 'source_rows.json', protocol['locks']['source_rows'])
    require(source_rows_path.read_bytes() == expected_rows_bytes, 'Historical source-row bytes/order differ')
    names_before = sorted(str(p.relative_to(out)) for p in out.rglob('*') if p.is_file())
    require(set(names_before) == MEMBERS, 'Missing/extra/partial/failed preparation artifact')
    for name in MEMBERS:
        bind(out / name)
    binding = {'schema_version': 1, 'protocol_id': protocol['protocol_id'], 'protocol_sha256': args.protocol_sha256,
               'release_lock_sha256': args.preparation_release_sha256, 'operation': 'prepare_all_source_masks'}
    complete = read(out / 'COMPLETE.json'); same_fields(complete, {**binding, 'status': 'complete'})
    require(set(complete['files']) == MEMBERS - {'COMPLETE.json'}, 'Incomplete terminal manifest')
    for name, expected in complete['files'].items():
        require(bound[(out / name).resolve()] == expected, 'Terminal artifact hash differs')
    same_fields(read(out / 'started.json'), {**binding, 'status': 'started'})
    events = [json.loads(line) for line in (out / 'journal.jsonl').read_text().splitlines()]
    require([x['event'] for x in events] == ['attempt_started', 'source_identity_and_bytes_verified', 'complete_mask_schedule_prepared', 'attempt_complete'], 'Preparation journal did not finish in expected order')
    bundle = read(out / 'masks.json'); bundle_sha = bound[(out / 'masks.json').resolve()]
    same_fields(bundle, {'schema_version': 1, 'protocol_id': protocol['protocol_id'], 'status': protocol['status'],
                        'protocol_sha256': args.protocol_sha256, 'row_identity_sha256': identity})
    names = ['O', 'C', 'D'] + [f'{family}-{seed}' for family in ('Rv', 'Rp') for seed in protocol['selection_seeds'][family]]
    require(set(bundle['masks']) == set(names) and len(names) == 23, 'Mask identities differ')
    x_sources, val_sources = set(rows['train_x']), set(rows['validation'])
    targets = {'O': set(range(len(paragraph_source))), 'C': {i for i, source in enumerate(paragraph_source) if source not in val_sources}}
    targets['D'] = {i for i in targets['C'] if paragraph_source[i] not in x_sources}
    require({name: len(indices) for name, indices in targets.items()} == {'O': 19561, 'C': 16224, 'D': 9963}, 'Targeted exclusions differ')
    expected_masks, summaries = {}, {}
    for name in names:
        family = name.split('-')[0]; seed = int(name.split('-')[1]) if '-' in name else None
        if family in targets:
            wanted = targets[family]
        else:
            parent, goal = ('O', 'C') if family == 'Rv' else ('C', 'D')
            candidates = {source: positions for source, positions in groups.items() if positions <= targets[parent]}
            ordering = sorted(candidates, key=lambda source: (hashlib.sha256(f'{seed}:{source}'.encode()).digest(), source))
            wanted = set(); remaining = len(targets[goal])
            for source in ordering:
                if len(candidates[source]) <= remaining:
                    wanted.update(candidates[source]); remaining -= len(candidates[source])
            require(remaining == 0, 'Exact whole-source greedy budget cannot be filled')
        actual = bundle['masks'][name]
        require(all(integer(i) for i in actual['train_y_indices']) and actual['train_y_indices'] == sorted(wanted), 'Mask differs from independent exact selection')
        require(all(not positions & wanted or positions <= wanted for positions in groups.values()), 'A repeated or aliased source is split')
        expected = {'variant': VARIANTS[family], 'selection_seed': seed, 'row_identity_sha256': identity, 'train_y_indices': sorted(wanted)}
        expected['mask_sha256'] = canonical_hash(expected)
        require(actual == expected, 'Mask schema/canonical hash differs')
        expected_masks[name] = expected
        selected_groups = {paragraph_source[i] for i in wanted}
        summaries[name] = {'mask_sha256': expected['mask_sha256'], 'rows': len(wanted), 'canonical_source_groups': len(selected_groups),
                          'groups_with_multiple_vg_aliases': sum(len(aliases[source]) > 1 for source in selected_groups)}
    expected_strata = {'never_known_spc_overlap': [i for i, source in enumerate(rows['validation']) if source not in groups],
                      'original_spc_overlap': [i for i, source in enumerate(rows['validation']) if source in groups], 'all': list(range(len(rows['validation'])))}
    require(bundle['strata'] == expected_strata and {k: len(v) for k, v in expected_strata.items()} == protocol['strata_counts'], 'Fixed original-SPC strata differ')
    common = {**binding, 'masks_sha256': bundle_sha, 'row_identity_sha256': identity}
    compositions = read(out / 'compositions.json'); same_fields(compositions, common)
    require(set(compositions['masks']) == set(names), 'Composition table incomplete')
    for name, mask in expected_masks.items():
        counts = Counter(paragraph_source[i] for i in mask['train_y_indices'])
        row_counts = dict.fromkeys(('known_coco_train', 'known_coco_validation', 'no_coco_mapping', 'mapped_coco_outside_reference'), 0)
        group_counts = dict(row_counts)
        for source, n in counts.items():
            category = ('known_coco_train' if source in x_sources else 'known_coco_validation' if source in val_sources
                        else 'no_coco_mapping' if source.startswith('vg:') else 'mapped_coco_outside_reference')
            row_counts[category] += n; group_counts[category] += 1
        expected = {k: mask[k] for k in ('variant', 'selection_seed', 'mask_sha256', 'row_identity_sha256')}
        expected.update(training_image_rows=len(rows['train_x']), paragraph_rows=sum(counts.values()), source_groups=len(counts),
                        duplicate_source_rows=sum(counts.values()) - len(counts), group_size_histogram={str(k): v for k, v in sorted(Counter(counts.values()).items())},
                        source_membership_rows=row_counts, source_membership_groups=group_counts,
                        scope='Known metadata identities only; no image-byte, near-duplicate or pretraining-cleanliness claim.')
        require(compositions['masks'][name] == expected, 'Composition differs')
    intersections = read(out / 'intersections.json'); same_fields(intersections, common)
    expected_pairs = []
    for i, left in enumerate(names):
        for right in names[i + 1:]:
            a, b = set(expected_masks[left]['train_y_indices']), set(expected_masks[right]['train_y_indices'])
            ga, gb = {paragraph_source[j] for j in a}, {paragraph_source[j] for j in b}
            expected_pairs.append({'left': left, 'right': right, 'left_mask_sha256': expected_masks[left]['mask_sha256'],
                'right_mask_sha256': expected_masks[right]['mask_sha256'], 'shared_rows': len(a & b), 'union_rows': len(a | b),
                'shared_source_groups': len(ga & gb), 'union_source_groups': len(ga | gb),
                'shared_known_training_sources': len(ga & gb & x_sources), 'shared_known_validation_sources': len(ga & gb & val_sources)})
    require(len(expected_pairs) == 253 and intersections['pairs'] == expected_pairs, 'Complete intersection table differs')
    schedule = read(out / 'schedule.json'); same_fields(schedule, common)
    expected_schedule = [{'index': i, 'mask': name, 'mask_sha256': expected_masks[name]['mask_sha256'], 'seed': seed}
                         for i, (name, seed) in enumerate((name, seed) for name in names for seed in SEEDS)]
    require(schedule['planned_fits'] == 115 and schedule['order'] == 'mask_major_original_algorithm_seed_order' and schedule['runs'] == expected_schedule, 'Full115-job schedule differs')
    result = read(out / 'result.json'); same_fields(result, {**binding, 'status': 'complete', 'prepared_masks': 23, 'planned_fits': 115,
        'strata_counts': protocol['strata_counts'], 'archive_members': member_receipts, 'source_rows_sha256': protocol['locks']['source_rows'],
        'masks_sha256': bundle_sha, 'feature_loads': 0, 'fits': 0})
    for path, expected in bound.items():
        require(digest(path) == expected, 'Bound input/output changed during audit')
    require(sorted(str(p.relative_to(out)) for p in out.rglob('*') if p.is_file()) == names_before, 'Prepared artifact inventory changed during audit')
    return {'schema_version': 1, 'status': 'independently_checked', 'observed_at_utc': datetime.now(timezone.utc).isoformat(),
        'published_commit': args.published_commit, 'protocol_sha256': args.protocol_sha256,
        'preparation_release_sha256': args.preparation_release_sha256, 'auditor_sha256': digest(__file__),
        'preparation_code_sha256': release['code_sha256'], 'archive_sha256': protocol['source_archive_sha256'],
        'prepared_files_sha256': {name: bound[(out / name).resolve()] for name in sorted(MEMBERS)},
        'canonical_source_identity_sha256': identity, 'masks_checked': 23, 'intersections_checked': 253, 'scheduled_fits_checked': 115,
        'strata_counts': {k: len(v) for k, v in expected_strata.items()}, 'masks': summaries,
        'original_source_groups': len(groups), 'original_duplicate_source_rows': len(paragraph_source) - len(groups),
        'original_groups_with_multiple_vg_aliases': sum(len(v) > 1 for v in aliases.values()),
        'whole_group_and_alias_splitting_violations': 0, 'independent_archive_join_and_historical_order': 'passed',
        'exact_sha_ranked_greedy_rv_rp_rule': 'passed', 'compositions_and_all_pair_intersections': 'passed',
        'complete_manifest_and_final_byte_recheck': 'passed', 'features_loaded': 0, 'fits_performed': 0,
        'scope': 'Independent metadata and mask-integrity audit only. Known source identities do not establish image-byte independence, absence of near duplicates, or clean encoder pretraining.'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('protocol', 'preparation-release', 'code-root', 'bindings', 'prepared', 'out'):
        p.add_argument('--' + name, type=Path, required=True)
    for name in ('protocol-sha256', 'preparation-release-sha256', 'published-commit'):
        p.add_argument('--' + name, required=True)
    args = p.parse_args(); require(not args.out.exists(), 'Audit receipt already exists')
    result = audit(args)
    with args.out.open('x') as f:
        json.dump(result, f, indent=2, allow_nan=False); f.write('\n')
    print(json.dumps({'status': result['status'], 'receipt_sha256': digest(args.out), 'masks_checked': 23, 'fits_performed': 0}))


if __name__ == '__main__':
    main()
