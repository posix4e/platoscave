"""Synthetic-tested building blocks; the calling runner must enforce frozen locks.

These functions neither choose models nor acquire/reduce embeddings. Callers must
freeze the complete extraction/model/runtime/data protocol before real use. The
paired controls deliberately receive full training counterparts beyond the two
disjoint centering banks; their information budget is therefore larger than the
unpaired arm's. Nothing here reports baseline or paper replication completion.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import torch
from torch import Tensor

import unpaired_rosetta
from unpaired_rosetta.baselines.linear import LinearMap
from unpaired_rosetta.baselines.orthogonal import OrthogonalMap
from unpaired_rosetta.determinism import FIT_THREADS, torch_threads
from unpaired_rosetta.evaluation import midranks
from unpaired_rosetta.linalg import center_and_normalize
from unpaired_rosetta.randomness import Randomness

UPSTREAM_COMMIT = 'fdfad84ea488f29702a61a455be71622c4ff5d49'
API_SHA256 = {
    'baselines/orthogonal.py': '44071eb419fba9428a6be8aef12307c4ae77d0afef97b97754c340fd113931cb',
    'baselines/linear.py': '08ac28d6cae40e4a75aeda7e54d5cc0a62e6b3c6a7cf7cd097c95b0f02a6e620',
    'evaluation.py': '8f5289092b13229c74ed0f312ec1ff79149b1cdeb4e13295d52df62e3939eed7',
    'linalg.py': 'a83047eb44fae9b22006ba87351baf9c69cc09de9c5fc182a9cec075901da0b6',
    'determinism.py': '8f6f0b878aaee225cf5e73617af16270dc1d3c07d8caec47311428534c1d4e60',
    'randomness.py': '37acc3e14e63862f8b776c82d2273b2b1516ce5578496b15eb87f180173b2cb2',
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def upstream_api_hashes():
    folder = Path(unpaired_rosetta.__file__).resolve().parent
    actual = {name: hashlib.sha256((folder / name).read_bytes()).hexdigest() for name in API_SHA256}
    require(actual == API_SHA256, 'The imported authors APIs differ from the pinned source')
    return actual


def indices_hash(indices):
    """SHA256 of UTF-8 JSON integer list, with compact separators and no newline."""
    payload = json.dumps(indices.tolist(), separators=(',', ':'), allow_nan=False)
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()


@dataclass(frozen=True)
class TrainingPermutation:
    indices: Tensor
    metadata: dict


def training_partner_permutation(protocol_id: str, split_seed: int, num_items: int) -> TrainingPermutation:
    """Uniform torch.randperm with a separate domain-separated RNG; fixed points stay.

    Derivation: first eight SHA256 bytes, interpreted as an unsigned big-endian
    integer, from compact sorted-key UTF-8 JSON of domain/protocol_id/split_seed.
    The item count controls randperm's length, not the seed derivation. The exact
    torch version must be pinned by the eventual protocol.
    """
    require(isinstance(protocol_id, str) and bool(protocol_id), 'A nonempty protocol ID is required')
    require(isinstance(split_seed, int) and not isinstance(split_seed, bool) and split_seed >= 0, 'Invalid split seed')
    require(isinstance(num_items, int) and not isinstance(num_items, bool) and num_items >= 2, 'At least two items are required')
    domain = {'domain': 'rosetta-training-partner-permutation-v1', 'protocol_id': protocol_id, 'split_seed': split_seed}
    payload = json.dumps(domain, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode('utf-8')
    seed_digest = hashlib.sha256(payload).digest()
    seed = int.from_bytes(seed_digest[:8], 'big', signed=False)
    generator = torch.Generator(device='cpu').manual_seed(seed)
    permutation = torch.randperm(num_items, generator=generator)
    return TrainingPermutation(permutation, {
        **domain, 'num_items': num_items,
        'seed_derivation_sha256': seed_digest.hex(), 'torch_seed_decimal': str(seed),
        'torch_version': torch.__version__, 'permutation_sha256': indices_hash(permutation),
        'fixed_point_count': int((permutation == torch.arange(num_items)).sum()),
        'forced_derangement': False,
    })


def features(name, value):
    require(isinstance(value, Tensor) and value.ndim == 2 and value.shape[0] > 0 and value.shape[1] > 0,
            f'{name} must be a nonempty matrix')
    require(value.device.type == 'cpu' and value.dtype == torch.float32, f'{name} must be CPU float32')
    require(bool(torch.isfinite(value).all()), f'{name} contains nonfinite values')


def item_ids(name, value, count):
    require(isinstance(value, Tensor) and value.device.type == 'cpu' and value.dtype == torch.int64 and
            value.ndim == 1 and len(value) == count, f'{name} must contain one CPU int64 row ID per item')
    require(len(set(value.tolist())) == count, f'{name} contains duplicate row IDs')


@torch.inference_mode()
@torch_threads()
def linear_pinv_diagnostics(paired_x: Tensor, fitted_mean_x: Tensor):
    """Describe the authors' default pseudoinverse cutoff without changing a fit.

    This repeats one SVD of the original FP32 centered/normalized X, under the
    original thread count. Both linear controls share this design. The spectrum
    is diagnostic: it is not captured from the SVD inside either original fit.
    A value near the cutoff can change sides with numerical variation; report
    its margin rather than calling this an unrestricted full-rank linear map.

    PyTorch 2.14's non-Hermitian pinv calls Tensor.svd() with vectors, promotes
    default CPU tolerances to float64, and retains strictly S > tolerance:
    https://github.com/pytorch/pytorch/blob/v2.14.0/aten/src/ATen/native/LinearAlgebra.cpp#L410-L499
    https://docs.pytorch.org/docs/2.14/generated/torch.linalg.pinv.html
    """
    upstream_api_hashes()
    require(torch.__version__.split('+')[0] == '2.14.0',
            'Review the default-pinv diagnostic before changing the audited torch 2.14.0 runtime')
    features('paired_x', paired_x)
    features('fitted_mean_x', fitted_mean_x)
    require(fitted_mean_x.shape == (1, paired_x.shape[1]), 'Invalid fitted X mean shape')
    design = center_and_normalize(paired_x, fitted_mean_x)
    features('centered_normalized_paired_x', design)
    # Match pinv's vector-producing SVD path, once for both controls. Do not use
    # a Gram matrix (squared condition number) or replace the original solver.
    left, singular, right = torch.svd(design, some=True, compute_uv=True)
    del left, right
    epsilon = torch.finfo(design.dtype).eps
    rtol = max(design.shape) * epsilon
    # The length-one FP64 tensor also preserves pinv's promotion at its cutoff.
    cutoff = singular[:1] * torch.tensor([rtol], dtype=torch.float64)
    retained = singular > cutoff
    spectrum = singular.tolist()
    spectrum_bytes = json.dumps(spectrum, separators=(',', ':'), allow_nan=False).encode('utf-8')
    sigma_max = singular[0].item()
    margin = (singular.double() - cutoff).abs().min().item()
    return {
        'kind': 'independent_post_fit_svd_diagnostic',
        'applies_to': ['linear_paired', 'linear_shuffled'],
        'solver': 'original LinearMap.fit: torch.linalg.pinv(X_hat) @ Y_hat',
        'fit_modified': False, 'fit_internal_spectrum_captured': False,
        'diagnostic_svd_count': 1, 'diagnostic_svd': 'torch.Tensor.svd(some=True, compute_uv=True)',
        'torch_version': torch.__version__, 'threads': FIT_THREADS,
        'design_shape': list(design.shape), 'design_dtype': str(design.dtype),
        'design_preprocessing': 'original center_and_normalize(paired_x, fitted mean_x)',
        'pinv_arguments': {'atol': None, 'rtol': None, 'hermitian': False},
        'effective_atol': 0.0, 'dtype_epsilon': epsilon,
        'default_rtol_rule': 'max(rows, columns) * dtype_epsilon', 'effective_rtol': rtol,
        'cutoff_rule': 'max(effective_atol, sigma_max * effective_rtol)',
        'cutoff_dtype': str(cutoff.dtype), 'absolute_cutoff': cutoff.item(),
        'retention_rule': 'singular_value > absolute_cutoff',
        'effective_rank_by_default_cutoff': int(retained.sum()),
        'maximum_possible_rank': len(singular), 'discarded_singular_values': int((~retained).sum()),
        'sigma_max': sigma_max,
        'smallest_retained_singular_value': singular[retained][-1].item() if bool(retained.any()) else None,
        'largest_discarded_singular_value': singular[~retained][0].item() if bool((~retained).any()) else None,
        'nearest_cutoff_absolute_margin': margin,
        'nearest_cutoff_margin_over_sigma_max': margin / sigma_max if sigma_max else None,
        'singular_values': spectrum, 'singular_values_json_sha256': hashlib.sha256(spectrum_bytes).hexdigest(),
        'interpretation': 'Default-cutoff pseudoinverse fit; rank is a separate diagnostic, not an assertion of unrestricted full rank. Values near the cutoff may change retention under numerical variation.',
    }


def fit_paired_controls(centering_x: Tensor, centering_y: Tensor, paired_x: Tensor, paired_y: Tensor, *,
                        centering_ids_x: Tensor, centering_ids_y: Tensor, paired_ids: Tensor,
                        protocol_id: str, split_seed: int):
    """Four original supervised controls using the same original centering banks.

    paired_x and paired_y must be in the same paired_ids order and cover the full
    union of the two disjoint training banks. This adds Y counterparts for X-bank
    rows and X counterparts for Y-bank rows. The caller must establish that these
    IDs are training-only and that every model uses the same corpus/order.
    """
    sources = upstream_api_hashes()
    for name, value in [('centering_x', centering_x), ('centering_y', centering_y),
                        ('paired_x', paired_x), ('paired_y', paired_y)]:
        features(name, value)
    n = len(paired_x)
    require(len(paired_y) == n and n >= 2, 'Paired banks must have the same nontrivial row count')
    require(centering_x.shape[1] == paired_x.shape[1] and centering_y.shape[1] == paired_y.shape[1],
            'Centering and paired feature widths differ')
    item_ids('centering_ids_x', centering_ids_x, len(centering_x))
    item_ids('centering_ids_y', centering_ids_y, len(centering_y))
    item_ids('paired_ids', paired_ids, n)
    ids_x, ids_y, ids_all = map(lambda v: set(v.tolist()), (centering_ids_x, centering_ids_y, paired_ids))
    require(not ids_x & ids_y and ids_x | ids_y == ids_all, 'Centering banks must be disjoint and exhaust the paired training corpus')
    rows = {row_id: i for i, row_id in enumerate(paired_ids.tolist())}
    for bank, ids, full in [(centering_x, centering_ids_x, paired_x), (centering_y, centering_ids_y, paired_y)]:
        selected = torch.tensor([rows[row_id] for row_id in ids.tolist()], dtype=torch.int64)
        require(torch.equal(bank, full[selected]), 'A centering bank is not the corresponding original full-bank rows')
    pairing = training_partner_permutation(protocol_id, split_seed, n)
    shuffled_y = paired_y[pairing.indices]
    models = {}
    for name, cls in [('orthogonal', OrthogonalMap), ('linear', LinearMap)]:
        models[name + '_paired'] = cls().fit(centering_x, centering_y, paired_x, paired_y)
        models[name + '_shuffled'] = cls().fit(centering_x, centering_y, paired_x, shuffled_y)
    require(torch.equal(models['linear_paired'].mean_x, models['linear_shuffled'].mean_x),
            'The two linear controls must share their X centering mean')
    linear_diagnostic = linear_pinv_diagnostics(paired_x, models['linear_paired'].mean_x)
    return models, {
        'upstream_commit': UPSTREAM_COMMIT, 'upstream_api_sha256': sources,
        'paired_training_items': n, 'centering_items_x': len(centering_x), 'centering_items_y': len(centering_y),
        'additional_x_counterparts': n - len(centering_x), 'additional_y_counterparts': n - len(centering_y),
        'centering_ids_x_sha256': indices_hash(centering_ids_x), 'centering_ids_y_sha256': indices_hash(centering_ids_y),
        'paired_ids_sha256': indices_hash(paired_ids), 'partner_permutation': pairing.metadata,
        'information_budget': 'Supervised controls receive additional training counterparts; they do not match the unpaired information budget.',
        'same_target_permutation_for_both_maps': True,
        'linear_pinv_diagnostic': linear_diagnostic,
    }


@torch.inference_mode()
def retrieval_with_ranks(aligner, val_x: Tensor, val_y: Tensor, randomness: Randomness, block_size: int = 4096):
    """Supplement the untouched authors metric with ranks in incoming row order.

    Pass a separate Randomness with the SAME state the original evaluator would
    receive. This function consumes exactly its two validation permutations;
    it does not reset a seed or repeat the earlier validation-subset draw.
    Similarity arithmetic remains CPU float32; only then are scores converted
    to float64, as in the authors evaluator. No paired IDs reach the aligner.
    """
    upstream_api_hashes()
    features('val_x', val_x)
    features('val_y', val_y)
    n = len(val_x)
    require(n == len(val_y) and n >= 2, 'Validation needs at least two paired rows')
    require(isinstance(randomness, Randomness), 'Use the original Randomness instance with its current state')
    require(isinstance(block_size, int) and not isinstance(block_size, bool) and block_size > 0, 'Invalid block size')
    order_x = randomness.permutation(n)
    order_y = randomness.permutation(n)
    shuffled_x, shuffled_y = val_x[order_x], val_y[order_y]
    partner = torch.argsort(order_y)[order_x]
    ranks = torch.empty(n, dtype=torch.float64)
    for start in range(0, n, block_size):
        stop = min(start + block_size, n)
        raw_scores = aligner.similarity(shuffled_x[start:stop], shuffled_y)
        require(raw_scores.device.type == 'cpu' and raw_scores.dtype == torch.float32 and
                raw_scores.shape == (stop - start, n) and bool(torch.isfinite(raw_scores).all()),
                'Aligner must return finite CPU float32 similarities of the expected shape')
        scores = raw_scores.double().cpu()
        true_scores = scores[torch.arange(stop - start), partner[start:stop]]
        ranks[order_x[start:stop]] = midranks(scores, true_scores)
    return {'ranks': ranks, 'query_order': order_x, 'key_order': order_y,
            'partner_in_shuffled_keys': partner,
            'metrics': {'foscttm': ((ranks - 1) / (n - 1)).mean().item(), 'mean_rank': ranks.mean().item()}}
