#!/usr/bin/env python3
"""Test encoder readiness using generated pixels only; never accepts photographs."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import importlib.util
import io
import json
from pathlib import Path
import platform
import time

import vision_features as vision


ATOL = 2e-6


def synthetic_inputs(helper_path):
    """Five deterministic in-memory RGB PNGs, without semantic labels."""
    import numpy as np
    from PIL import Image, ImageOps
    import torch
    vision.verify_hash(helper_path, vision.HELPER_SHA256)
    if importlib.metadata.version("Pillow") != "12.3.0":
        raise ValueError("Frozen preprocessing requires Pillow 12.3.0")
    spec = importlib.util.spec_from_file_location("_vision_readiness_frozen_helper", helper_path)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    records, arrays = [], []
    for index, (width, height) in enumerate([(224, 224), (351, 229), (231, 349), (397, 257), (257, 397)]):
        yy, xx = np.indices((height, width), dtype=np.int32)
        pixels = np.stack(((xx * 7 + yy * 3 + index * 17) % 256,
                           ((xx // (index + 1)) ^ (yy * 3) ^ (index * 29)) % 256,
                           (xx * yy + index * 47) % 256), axis=-1).astype(np.uint8)
        stream = io.BytesIO()
        Image.fromarray(pixels).save(stream, format="PNG")
        raw = stream.getvalue()
        arrays.append(helper.preprocess_bytes(raw, np, Image, ImageOps))
        records.append({"fixture_id": f"synthetic-{index:02d}", "width": width, "height": height,
                        "png_sha256": hashlib.sha256(raw).hexdigest()})
    batch = np.stack(arrays).astype(np.float32)
    return torch.from_numpy(batch), records, hashlib.sha256(batch.tobytes()).hexdigest()


def check_encoder(model, tensors):
    import numpy as np
    import torch
    start = time.monotonic()
    buffers_before = {name: value.detach().cpu().clone() for name, value in model.named_buffers()}
    cpu_rng = torch.get_rng_state().clone()
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
    base = vision.extract(model, tensors, batch_size=5)
    repeat = vision.extract(model, tensors, batch_size=5)
    reversed_rows = vision.extract(model, tensors.flip(0), batch_size=5)[::-1].copy()
    single = vision.extract(model, tensors, batch_size=1)
    paired = vision.extract(model, tensors, batch_size=2)
    errors = {"repeat": float(np.max(np.abs(base - repeat))),
              "reordered": float(np.max(np.abs(base - reversed_rows))),
              "batch_size_1": float(np.max(np.abs(base - single))),
              "batch_size_2": float(np.max(np.abs(base - paired)))}
    checks = {
        "repeat_bitwise_equal": bool(np.array_equal(base, repeat)),
        "reordered_within_tolerance": bool(np.allclose(base, reversed_rows, atol=ATOL, rtol=0)),
        "batch_size_1_within_tolerance": bool(np.allclose(base, single, atol=ATOL, rtol=0)),
        "batch_size_2_within_tolerance": bool(np.allclose(base, paired, atol=ATOL, rtol=0)),
        "finite_unit_vectors": bool(np.isfinite(base).all() and np.allclose(np.linalg.norm(base, axis=1), 1, atol=1e-6, rtol=0)),
        "all_buffers_unchanged": all(torch.equal(buffers_before[name], value.detach().cpu()) for name, value in model.named_buffers()),
        "cpu_rng_unchanged": bool(torch.equal(cpu_rng, torch.get_rng_state())),
        "cuda_rng_unchanged": all(torch.equal(before, after) for before, after in zip(cuda_rng, torch.cuda.get_rng_state_all() if cuda_rng else [])),
    }
    return {"provenance": model.feature_metadata, "shape": list(base.shape), "dtype": str(base.dtype),
            "normalized_feature_sha256": hashlib.sha256(base.tobytes()).hexdigest(),
            "absolute_tolerance": ATOL, "relative_tolerance": 0, "max_absolute_errors": errors,
            "checks": checks, "passed": all(checks.values()), "elapsed_seconds": round(time.monotonic() - start, 3)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--moco-checkpoint", type=Path, required=True)
    parser.add_argument("--mae-checkpoint", type=Path, required=True)
    parser.add_argument("--vendor-dir", type=Path, required=True)
    parser.add_argument("--helper", type=Path, required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError("Readiness report must be a new file")
    runtime = vision.configure_runtime()
    import torch
    source_hashes = {"vision_features.py": vision.sha256(Path(vision.__file__)),
                     "check_vision_readiness.py": vision.sha256(Path(__file__)),
                     "preprocess_helper": vision.verify_hash(args.helper, vision.HELPER_SHA256)}
    tensors, fixtures, tensor_hash = synthetic_inputs(args.helper)
    reports = {}
    for name, loader, positional in [
        ("moco_v2_r50_800ep", vision.load_moco, [args.moco_checkpoint]),
        ("mae_vit_b16", vision.load_mae, [args.mae_checkpoint, args.vendor_dir]),
    ]:
        model = loader(*positional, device=args.device)
        reports[name] = check_encoder(model, tensors)
        del model
        if args.device == "cuda":
            torch.cuda.empty_cache()
    # Recheck all executable/vendor/weight inputs after the synthetic inference.
    vision.verify_hash(args.moco_checkpoint, vision.MOCO_SHA256)
    vision.verify_hash(args.mae_checkpoint, vision.MAE_SHA256)
    for name, expected in vision.VENDOR_SHA256.items():
        vision.verify_hash(args.vendor_dir / name, expected)
    for name, path in [("vision_features.py", Path(vision.__file__)), ("check_vision_readiness.py", Path(__file__)), ("preprocess_helper", args.helper)]:
        vision.verify_hash(path, source_hashes[name])
    report = {"schema_version": 1, "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
              "scope": "Software readiness on five generated RGB PNG fixtures only. No human photographs, text, labels, alignment, or evaluation metrics.",
              "python": platform.python_version(), "runtime": runtime,
              "pillow": importlib.metadata.version("Pillow"), "device": args.device,
              "gpu_name": torch.cuda.get_device_name(0) if args.device == "cuda" else None,
              "source_sha256": source_hashes, "fixtures": fixtures, "input_tensor_sha256": tensor_hash,
              "encoders": reports, "passed": all(item["passed"] for item in reports.values())}
    with args.out.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"passed": report["passed"], "checks": {name: item["max_absolute_errors"] for name, item in reports.items()}}))
    if not report["passed"]:
        raise SystemExit("Synthetic readiness checks failed; report retained")


if __name__ == "__main__":
    main()
