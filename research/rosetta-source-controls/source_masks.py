"""Draft pure source-identity masks; no downloads, file writes, features or fits.

The caller must freeze source-archive hashes, this code and the seed lists before
using real masks. No removal count or seed list is chosen here. Historical arm
names are compatibility labels: ``fully_dedup`` means known-ID exclusions only,
never a certificate that unmapped images or pretrained models are uncontaminated.

Random controls preserve complete canonical source groups, rank groups by
SHA256(f"{seed}:{source}"), and greedily skip groups exceeding the remaining row
budget. This exactly matches the audited old sampler, not uniform row deletion
or a uniform draw from all admissible group subsets. Failure to fill the exact
budget is an error; there is no partial-group or approximate-count fallback.
"""
from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import json

ORIGINAL = 'original'
VALIDATION = 'validation_clean'
BOTH = 'fully_dedup'
RANDOM_VALIDATION = 'random_drop_validation_size_matched'
RANDOM_PAIRING = 'random_drop_size_matched'
ARMS = (ORIGINAL, VALIDATION, BOTH, RANDOM_VALIDATION, RANDOM_PAIRING)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def integer(value):
    return type(value) is int and value >= 0


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def source_id(value, prefix=None):
    require(isinstance(value, str) and value.count(':') == 1, 'Invalid source identity')
    kind, number = value.split(':')
    require(kind in ('coco', 'vg') and number.isascii() and number.isdecimal() and str(int(number)) == number,
            'Source identity must have a canonical nonnegative integer')
    require(prefix is None or kind == prefix, 'Wrong source identity kind')
    return int(number)


@dataclass(frozen=True)
class SourceRows:
    train_x: tuple[str, ...]
    train_y: tuple[str, ...]
    validation: tuple[str, ...]
    train_y_vg_ids: tuple[int, ...]

    def __post_init__(self):
        require(all(type(v) is tuple and v for v in (self.train_x, self.train_y, self.validation, self.train_y_vg_ids)),
                'Source rows must be nonempty tuples')
        require(len(self.train_y) == len(self.train_y_vg_ids), 'Paragraph source and VG row counts differ')
        for values in (self.train_x, self.validation):
            numbers = [source_id(v, 'coco') for v in values]
            require(numbers == sorted(set(numbers)), 'COCO rows must retain ascending unique numeric IDs')
        require(not set(self.train_x) & set(self.validation), 'COCO training and validation sources intersect')
        by_vg = {}
        for source, vg in zip(self.train_y, self.train_y_vg_ids):
            source_id(source)
            require(integer(vg), 'Invalid VG image ID')
            require(not source.startswith('vg:') or source == f'vg:{vg}', 'Unmapped source differs from its VG ID')
            require(vg not in by_vg or by_vg[vg] == source, 'One VG ID has conflicting canonical sources')
            by_vg[vg] = source

    def payload(self):
        return {key: list(getattr(self, key)) for key in ('train_x', 'train_y', 'validation', 'train_y_vg_ids')}

    @property
    def identity_sha256(self):
        return canonical_hash(self.payload())


def source_rows_from_records(coco_images, paragraphs, vg_images):
    """Same join/order as the audited builder, including distinct VG aliases."""
    coco = {'train2014': [], 'val2014': []}
    seen = set()
    for row in coco_images:
        image, split = row['cocoid'], row['filepath']
        require(integer(image) and split in coco and image not in seen, 'Invalid/duplicate COCO ID or original split')
        expected = f'COCO_{split}_{image:012d}.jpg'
        require(row.get('filename', expected) == expected, 'COCO filename disagrees with source/split')
        seen.add(image)
        coco[split].append(image)
    vg = {}
    for row in vg_images:
        image, mapped = row['image_id'], row.get('coco_id')
        require(integer(image) and image not in vg, 'Invalid/duplicate VG metadata ID')
        require(mapped is None or integer(mapped), 'Invalid mapped COCO ID')
        vg[image] = mapped
    paragraph_ids, sources = [], []
    for row in paragraphs:
        image = row['image_id']
        require(integer(image) and image in vg, 'Paragraph source lacks VG metadata')
        paragraph_ids.append(image)
        sources.append(f'vg:{image}' if vg[image] is None else f'coco:{vg[image]}')
    return SourceRows(tuple(f'coco:{i}' for i in sorted(coco['train2014'])), tuple(sources),
                      tuple(f'coco:{i}' for i in sorted(coco['val2014'])), tuple(paragraph_ids))


def source_rows_from_document(document, *, expected_source_sha256):
    """Read the old standalone JSON payload after the caller verifies its bytes.

    Archive digests must be supplied explicitly from the reviewed input record;
    the document's own declarations are not independent provenance evidence.
    """
    require(document.get('schema_version') == 1 and document.get('source_sha256') == expected_source_sha256,
            'Wrong source archive bindings or source-row schema')
    require(bool(expected_source_sha256) and all(isinstance(v, str) and len(v) == 64 and
            all(c in '0123456789abcdef' for c in v) for v in expected_source_sha256.values()), 'Invalid archive digests')
    payload = document['rows']
    require(set(payload) == {'train_x', 'train_y', 'validation', 'train_y_vg_ids'} and
            all(type(v) is list for v in payload.values()), 'Incomplete source-row payload')
    rows = SourceRows(**{key: tuple(value) for key, value in payload.items()})
    require(document.get('row_identity_sha256') == rows.identity_sha256, 'Source-row content hash differs')
    return rows


def source_groups(rows):
    groups = defaultdict(list)
    for index, source in enumerate(rows.train_y):
        groups[source].append(index)
    return {key: tuple(values) for key, values in groups.items()}


def validate_indices(values, count):
    require(type(values) is tuple and all(integer(i) and i < count for i in values), 'Invalid original row index')
    require(values == tuple(sorted(set(values))), 'Rows must be unique and retain original file order')


def validate_whole_groups(rows, indices):
    validate_indices(indices, len(rows.train_y))
    selected = set(indices)
    require(all(not selected.intersection(group) or set(group) <= selected for group in source_groups(rows).values()),
            'A repeated/aliased source group was split')


def targeted_indices(rows):
    x, val = set(rows.train_x), set(rows.validation)
    original = tuple(range(len(rows.train_y)))
    clean = tuple(i for i, source in enumerate(rows.train_y) if source not in val)
    both = tuple(i for i in clean if rows.train_y[i] not in x)
    return {ORIGINAL: original, VALIDATION: clean, BOTH: both}


def sample_source_groups(rows, candidates, size, seed):
    """Audited nonuniform SHA-ranked greedy sampler, with an exact row budget."""
    validate_whole_groups(rows, candidates)
    require(integer(seed) and integer(size) and size <= len(candidates), 'Invalid selection seed or row budget')
    if size == 0:
        return ()
    groups = defaultdict(list)
    for index in candidates:
        groups[rows.train_y[index]].append(index)
    ranked = sorted(groups, key=lambda key: (hashlib.sha256(f'{seed}:{key}'.encode()).hexdigest(), key))
    chosen, remaining = [], size
    for key in ranked:
        if len(groups[key]) <= remaining:
            chosen.extend(groups[key])
            remaining -= len(groups[key])
        if remaining == 0:
            break
    require(remaining == 0, 'Cannot fill exact row budget without splitting a source group')
    return tuple(sorted(chosen))


@dataclass(frozen=True)
class SourceMask:
    variant: str
    selection_seed: int | None
    row_identity_sha256: str
    train_y_indices: tuple[int, ...]

    @property
    def mask_sha256(self):
        return canonical_hash({'variant': self.variant, 'selection_seed': self.selection_seed,
                               'row_identity_sha256': self.row_identity_sha256,
                               'train_y_indices': list(self.train_y_indices)})


def validate_mask(rows, mask):
    require(mask.variant in ARMS and mask.row_identity_sha256 == rows.identity_sha256, 'Wrong arm or source-row identity')
    validate_whole_groups(rows, mask.train_y_indices)
    targets = targeted_indices(rows)
    if mask.variant in targets:
        require(mask.selection_seed is None and mask.train_y_indices == targets[mask.variant], 'Wrong targeted membership')
    else:
        require(integer(mask.selection_seed), 'Random control needs an explicit selection seed')
        parent, target = ((ORIGINAL, VALIDATION) if mask.variant == RANDOM_VALIDATION else (VALIDATION, BOTH))
        require(set(mask.train_y_indices) <= set(targets[parent]) and len(mask.train_y_indices) == len(targets[target]),
                'Random control has wrong parent population or row count')
        require(mask.train_y_indices == sample_source_groups(rows, targets[parent], len(targets[target]), mask.selection_seed),
                'Random mask differs from its declared seed and sampler')


def build_masks(rows, *, validation_seeds, pairing_seeds):
    """No default seed lists: the caller must make both budget choices explicit."""
    for seeds in (validation_seeds, pairing_seeds):
        require(type(seeds) is tuple and all(integer(s) for s in seeds) and len(set(seeds)) == len(seeds),
                'Prespecified selection seeds must be unique nonnegative integer tuples')
    targets = targeted_indices(rows)
    masks = [SourceMask(name, None, rows.identity_sha256, indices) for name, indices in targets.items()]
    for name, seeds, parent, target in ((RANDOM_VALIDATION, validation_seeds, ORIGINAL, VALIDATION),
                                       (RANDOM_PAIRING, pairing_seeds, VALIDATION, BOTH)):
        for seed in seeds:
            masks.append(SourceMask(name, seed, rows.identity_sha256,
                                    sample_source_groups(rows, targets[parent], len(targets[target]), seed)))
    for mask in masks:
        validate_mask(rows, mask)
    return tuple(masks)


def fixed_strata(rows):
    """Original-SPC membership, never recomputed from a cleaned/random arm.

    All strata use the same full validation gallery. These are query indices,
    not instructions to restrict candidate captions.
    """
    sources = set(rows.train_y)
    return {'never_known_spc_overlap': tuple(i for i, source in enumerate(rows.validation) if source not in sources),
            'original_spc_overlap': tuple(i for i, source in enumerate(rows.validation) if source in sources),
            'all': tuple(range(len(rows.validation)))}


def composition(rows, mask):
    validate_mask(rows, mask)
    x, val = set(rows.train_x), set(rows.validation)
    counts = Counter(rows.train_y[i] for i in mask.train_y_indices)
    def membership(source):
        if source in x:
            return 'known_coco_train'
        if source in val:
            return 'known_coco_validation'
        return 'no_coco_mapping' if source.startswith('vg:') else 'mapped_coco_outside_reference'
    keys = ('known_coco_train', 'known_coco_validation', 'no_coco_mapping', 'mapped_coco_outside_reference')
    row_counts, group_counts = dict.fromkeys(keys, 0), dict.fromkeys(keys, 0)
    for source, count in counts.items():
        key = membership(source)
        row_counts[key] += count
        group_counts[key] += 1
    return {'variant': mask.variant, 'selection_seed': mask.selection_seed, 'mask_sha256': mask.mask_sha256,
            'row_identity_sha256': rows.identity_sha256, 'training_image_rows': len(rows.train_x),
            'paragraph_rows': sum(counts.values()), 'source_groups': len(counts),
            'duplicate_source_rows': sum(counts.values()) - len(counts),
            'group_size_histogram': dict(sorted(Counter(counts.values()).items())),
            'source_membership_rows': row_counts, 'source_membership_groups': group_counts,
            'scope': 'Known metadata identities only; no image-byte, near-duplicate or pretraining-cleanliness claim.'}


def intersection(rows, left, right):
    """Rows/groups may recur across arms and masks; they are not new samples."""
    validate_mask(rows, left)
    validate_mask(rows, right)
    a, b = set(left.train_y_indices), set(right.train_y_indices)
    ga, gb = {rows.train_y[i] for i in a}, {rows.train_y[i] for i in b}
    return {'left_mask_sha256': left.mask_sha256, 'right_mask_sha256': right.mask_sha256,
            'shared_rows': len(a & b), 'union_rows': len(a | b),
            'shared_source_groups': len(ga & gb), 'union_source_groups': len(ga | gb),
            'shared_known_training_sources': len(ga & gb & set(rows.train_x)),
            'shared_known_validation_sources': len(ga & gb & set(rows.validation))}
