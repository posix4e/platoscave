"""Frozen MoCo-v2 and MAE vision encoders; no downloads or image decoding.

Input tensors must come from the unchanged fresh-pilot preprocess_bytes helper.
Both encoders use that common bicubic resize-256 / center-224 input, including
for MoCo whose publisher's linear-evaluation script used bilinear resizing.
The common transform is a declared panel choice, not checkpoint-specific tuning.
"""
from __future__ import annotations

import collections.abc
import hashlib
import importlib.metadata
import importlib.util
import os
from pathlib import Path
import sys
import types


MOCO_SHA256 = "5e4a3fd7178a95837c0ad71c265ed5f477d271e753ba505b6b6a5e86829eeb7e"
MAE_SHA256 = "aec5f0b68e5f3193a00b07bc65a37440db549c15b36b8bea242606cc40c4bc5d"
HELPER_SHA256 = "3e8445f22a8f255dcf47af89ac0b8c45461ba98046549702abc7b09b6b59a624"
MAE_COMMIT = "efb2a8062c206524e35e47d04501ed4f544c0ae8"
MOCO_COMMIT = "78b69cafae80bc74cd1a89ac3fb365dc20d157d3"
RESNET_SOURCE_SHA256 = "7499a5059ad7b1a1f8f7543c04b4a13795140fad837c38530b48ff5d9610bf6e"
VENDOR_SHA256 = {
    "mae/models_vit.py": "6770ed2226d46ab8ba2b1ebc6662cc353dabf759679569acd004f6b3ac868154",
    "mae/LICENSE": "ffe3b1e0b58c1f1250ce48f84d8c37328f9a1ff59bb77b63bc97480cf2778b64",
    "timm-0.3.2-py3-none-any.whl": "c1598ef61c246e38835a6b834811dcb8d62ae56b644cd54dc3ef9565c8387875",
}
RUNTIME = {"torch": "2.7.1+cu128", "torchvision": "0.22.1+cu128", "numpy": "2.5.3"}
MOCO_REMOVED_KEYS = {
    "module.encoder_q.fc.0.weight", "module.encoder_q.fc.0.bias",
    "module.encoder_q.fc.2.weight", "module.encoder_q.fc.2.bias",
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_hash(path, expected):
    actual = sha256(path)
    if actual != expected:
        raise ValueError(f"SHA256 mismatch for {Path(path).name}: {actual}")
    return actual


def configure_runtime():
    """Use the already installed, fixed FP32 runtime without changing packages."""
    configured = os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if configured != ":4096:8":
        raise ValueError("CUBLAS_WORKSPACE_CONFIG must be :4096:8 before CUDA initialization")
    actual = {name: importlib.metadata.version(name) for name in RUNTIME}
    if actual != RUNTIME:
        raise ValueError(f"Frozen panel runtime mismatch: {actual}")
    import torch
    torch.manual_seed(20261010)
    torch.cuda.manual_seed_all(20261010)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    return actual


def _freeze(model, device, metadata):
    model = model.float().eval().requires_grad_(False).to(device)
    model.feature_metadata = metadata
    return model


def moco_backbone_state(state):
    """The locked publisher artifact has query weights and exactly four FC keys."""
    if not isinstance(state, dict) or len(state) != 322:
        raise ValueError("MoCo checkpoint must contain its 322 original state entries")
    prefix = "module.encoder_q."
    removed = {name for name in state if name.startswith(prefix + "fc.")}
    if removed != MOCO_REMOVED_KEYS or any(not name.startswith(prefix) for name in state):
        raise ValueError("Unexpected MoCo encoder or projection-head keys")
    return {name[len(prefix):]: value for name, value in state.items() if name not in removed}


def load_moco(checkpoint_path, device="cuda"):
    """Return the frozen query encoder's 2048d average-pool representation."""
    verify_hash(checkpoint_path, MOCO_SHA256)
    runtime = configure_runtime()
    import torch
    import torchvision.models.resnet as resnet
    verify_hash(resnet.__file__, RESNET_SOURCE_SHA256)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if set(checkpoint) != {"state_dict", "arch", "epoch"} or checkpoint["arch"] != "resnet50":
        raise ValueError("Unexpected MoCo checkpoint envelope")
    model = resnet.resnet50(weights=None)
    model.fc = torch.nn.Identity()
    state = moco_backbone_state(checkpoint["state_dict"])
    # Exact key equality also catches missing BN counters that PyTorch can migrate.
    if set(state) != set(model.state_dict()):
        raise ValueError("MoCo backbone keys do not exactly match torchvision ResNet-50")
    model.load_state_dict(state, strict=True)
    return _freeze(model, device, {
        "model": "moco_v2_r50_800ep", "checkpoint_sha256": MOCO_SHA256,
        "training_source_commit": MOCO_COMMIT, "runtime": runtime,
        "resnet_source_sha256": RESNET_SOURCE_SHA256,
        "dimension": 2048, "pooling": "final adaptive average pool before projection head",
        "loaded_state_keys": len(state), "discarded_keys": sorted(MOCO_REMOVED_KEYS),
        "preprocessing": "unchanged frozen fresh-pilot helper; common bicubic resize256 center224",
    })


def _mae_module(vendor_dir):
    vendor = Path(vendor_dir).resolve(strict=True)
    for name, expected in VENDOR_SHA256.items():
        verify_hash(vendor / name, expected)
    wheel = vendor / "timm-0.3.2-py3-none-any.whl"
    loaded = sys.modules.get("timm")
    if loaded is not None and not str(getattr(loaded, "__file__", "")).startswith(str(wheel) + "/"):
        raise ValueError("A different timm is already imported; use an isolated process")
    # timm 0.3.2 imports only container_abcs from removed torch._six. This alias
    # preserves the old helper's isinstance semantics; no model math is changed.
    previous_six = sys.modules.get("torch._six")
    if previous_six is not None and getattr(previous_six, "container_abcs", None) is not collections.abc:
        raise ValueError("Unexpected pre-existing torch._six compatibility module")
    if previous_six is None:
        shim = types.ModuleType("torch._six")
        shim.container_abcs = collections.abc
        sys.modules["torch._six"] = shim
    sys.path.insert(0, str(wheel))
    try:
        import timm
        if timm.__version__ != "0.3.2" or not str(timm.__file__).startswith(str(wheel) + "/"):
            raise ValueError("Failed to import the exact vendored timm wheel")
        spec = importlib.util.spec_from_file_location("_model_choice_mae_models_vit", vendor / "mae/models_vit.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(wheel))
        if previous_six is None:
            sys.modules.pop("torch._six", None)


def load_mae(checkpoint_path, vendor_dir, device="cuda"):
    """Return official MAE ViT-B/16's final-LayerNorm CLS, with stable patches."""
    verify_hash(checkpoint_path, MAE_SHA256)
    runtime = configure_runtime()
    import torch
    module = _mae_module(vendor_dir)
    # Official models_vit.forward_features uses every patch in its fixed raster
    # order. Unlike models_mae.forward_encoder it never calls random_masking.
    model = module.vit_base_patch16(num_classes=0, global_pool=False)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if set(checkpoint) != {"model"}:
        raise ValueError("Unexpected MAE checkpoint envelope")
    state = checkpoint["model"]
    if len(state) != 150 or set(state) != set(model.state_dict()):
        raise ValueError("MAE checkpoint keys must exactly match the official headless encoder")
    model.load_state_dict(state, strict=True)
    return _freeze(model, device, {
        "model": "mae_vit_b16", "checkpoint_sha256": MAE_SHA256,
        "source_commit": MAE_COMMIT, "vendor_sha256": dict(VENDOR_SHA256), "runtime": runtime,
        "dimension": 768, "pooling": "final encoder LayerNorm CLS; global_pool=False",
        "patch_order": "fixed raster order; all 196 patches; no masking or shuffling",
        "loaded_state_keys": len(state), "discarded_keys": [],
        "compatibility_change": "temporary torch._six.container_abcs alias to collections.abc during timm import only",
        "preprocessing": "unchanged frozen fresh-pilot helper; common bicubic resize256 center224",
    })


def extract(model, tensors, batch_size=5):
    """Return finite, L2-normalized float32 NumPy rows in input order."""
    import numpy as np
    import torch
    if not isinstance(tensors, torch.Tensor) or tensors.dtype != torch.float32:
        raise ValueError("Input must be a float32 torch.Tensor from the frozen image helper")
    if tensors.ndim != 4 or tuple(tensors.shape[1:]) != (3, 224, 224) or tensors.shape[0] == 0:
        raise ValueError("Expected nonempty [N,3,224,224] input")
    if not torch.isfinite(tensors).all() or not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size < 1:
        raise ValueError("Nonfinite input or invalid batch size")
    if model.training or any(module.training for module in model.modules()):
        raise ValueError("Every encoder module must remain in evaluation mode")
    if any(parameter.requires_grad or parameter.dtype != torch.float32 for parameter in model.parameters()):
        raise ValueError("Encoder must remain frozen FP32")
    if torch.is_autocast_enabled() or torch.is_autocast_enabled("cpu"):
        raise ValueError("Autocast must be disabled")
    if not torch.are_deterministic_algorithms_enabled() or torch.backends.cuda.matmul.allow_tf32 or torch.backends.cudnn.allow_tf32:
        raise ValueError("Deterministic FP32 runtime settings changed after loading")
    device = next(model.parameters()).device
    dimension = model.feature_metadata["dimension"]
    chunks = []
    with torch.inference_mode():
        for start in range(0, len(tensors), batch_size):
            value = model(tensors[start:start + batch_size].to(device=device, dtype=torch.float32))
            if tuple(value.shape) != (min(batch_size, len(tensors) - start), dimension) or value.dtype != torch.float32:
                raise ValueError("Unexpected encoder output shape or precision")
            chunks.append(value.cpu().numpy())
    array = np.concatenate(chunks, axis=0)
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    if not np.isfinite(array).all() or not np.isfinite(norms).all() or np.any(norms <= 0):
        raise ValueError("Zero or nonfinite image representation")
    result = (array / norms).astype(np.float32)
    if not np.isfinite(result).all() or not np.allclose(np.linalg.norm(result, axis=1), 1, atol=1e-6, rtol=0):
        raise ValueError("Image normalization failed")
    return result
