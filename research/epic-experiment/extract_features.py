#!/usr/bin/env python3
"""Frozen DINOv1/GloVe features for the prespecified EPIC-55 selection.

Input: JSON {"metadata": ..., "rows": [...]}, a JSON list, or JSONL. Each row
contains uid, participant_id, video_id, narration, split, noun_class, verb_class,
start_timestamp, stop_timestamp and exactly four frame_paths. Optional
frame_times_seconds must equal the four uniform action-window bin centers.

Output NPZ has image (N,768), text (N,300), eligible (N,), and row metadata.
All arrays load with allow_pickle=False. Zero vectors remain in the output;
the same eligible mask must be applied to every experimental arm. A JSON
sidecar records provenance, coverage, collisions, and partial-run status.
No parameters are fitted to these data. The cache stores normalized frame
features by video and verifies input bytes before reusing them.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import sys
import time
import unicodedata
import zipfile


DINO_COMMIT = "7c446df5b9f45747937fb0d72314eb9f7b66930a"
DINO_SHA256 = "bf34ad0f424b9029b593e8dc3ed553bf26e88bcba0d32bf3e62a6209cb64c85e"
GLOVE_ZIP_SHA256 = "6471382cdd837544bf3ac72497a38715e845897d265b2b424b4761832009c837"
GLOVE_MEMBER_SHA256 = "a12599d41e3589c7160be27fffe5b0080eccd0f0c75f46666c59f90188093c40"
SOURCE_SHA256 = {
    "vision_transformer.py": "b1f998d5f49ab43666b9fc6d007c5f6540c3ead2e8645e144f74045f63ff44d7",
    "utils.py": "962a97e1acda1c986dfd921275325e4083f9016dddadcf018f1ccf03fa600eab",
}
TOKEN_PATTERN = r"[a-z]+(?:'[a-z]+)?|[0-9]+"
FRAME_RECIPE = {
    "version": 1,
    "model": "original DINOv1 ViT-B/16, final-block normalized CLS",
    "source_commit": DINO_COMMIT,
    "source_sha256": SOURCE_SHA256,
    "checkpoint_sha256": DINO_SHA256,
    "rgb": True,
    "resize_short_side": 256,
    "interpolation": "Pillow BICUBIC",
    "center_crop": 224,
    "center_crop_rounding": "round((resized_dimension - 224) / 2)",
    "normalization_mean": [0.485, 0.456, 0.406],
    "normalization_std": [0.229, 0.224, 0.225],
    "arithmetic": "float32; TF32 disabled; deterministic algorithms",
    "frame_normalization": "L2",
    "action_pooling": "mean of four L2-normalized frame features, then L2",
}
METADATA_KEYS = (
    "uid", "participant_id", "video_id", "narration", "split", "noun_class",
    "verb_class", "start_timestamp", "stop_timestamp",
)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_hash(path, expected):
    actual = sha256(path)
    if actual != expected:
        raise ValueError(f"SHA256 mismatch for {path}: {actual} != {expected}")
    return actual


def timestamp_seconds(value):
    if isinstance(value, (float, int)):
        return float(value)
    fields = str(value).split(":")
    if len(fields) == 1:
        return float(fields[0])
    if len(fields) != 3:
        raise ValueError(f"Invalid timestamp: {value!r}")
    return int(fields[0]) * 3600 + int(fields[1]) * 60 + float(fields[2])


def read_manifest(path, frame_root, allow_partial=False):
    raw = path.read_text()
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        payload = [json.loads(line) for line in raw.splitlines() if line.strip()]
    if isinstance(payload, dict):
        rows = payload["rows"]
        metadata = payload.get("metadata", {})
    else:
        rows, metadata = payload, {}
    if not isinstance(rows, list) or not rows:
        raise ValueError("Manifest must contain a nonempty list of rows")
    ready, missing, seen = [], [], set()
    checked_frame_times = 0
    for source_row in rows:
        row = dict(source_row)
        for key in METADATA_KEYS:
            if key not in row:
                raise ValueError(f"Missing {key!r} in a manifest row")
        uid = str(row["uid"])
        if uid in seen:
            raise ValueError(f"Repeated uid {uid}")
        seen.add(uid)
        if not isinstance(row["narration"], str):
            raise ValueError(f"Narration must be a string: {uid}")
        paths = row.get("frame_paths", [])
        if len(paths) != 4 or any(not isinstance(p, str) for p in paths):
            raise ValueError(f"Expected exactly four frame_paths for uid {uid}")
        start = timestamp_seconds(row["start_timestamp"])
        stop = timestamp_seconds(row["stop_timestamp"])
        if not math.isfinite(start + stop) or stop <= start:
            raise ValueError(f"Invalid action window for uid {uid}")
        expected = [start + (index + 0.5) * (stop - start) / 4 for index in range(4)]
        if "frame_times_seconds" in row:
            times = row["frame_times_seconds"]
            if len(times) != 4 or any(
                not math.isclose(float(t), expected[i], rel_tol=0, abs_tol=1e-5)
                for i, t in enumerate(times)
            ):
                raise ValueError(f"Frame times are not the four bin centers for uid {uid}")
            checked_frame_times += 1
        row["_frame_paths"] = [
            str((frame_root / p).resolve()) if not Path(p).is_absolute()
            else str(Path(p).resolve()) for p in paths
        ]
        absent = [p for p in row["_frame_paths"] if not Path(p).is_file()]
        if absent:
            missing.append({"uid": uid, "missing_paths": absent})
        else:
            ready.append(row)
    if missing and not allow_partial:
        raise FileNotFoundError(
            f"{len(missing)} selected rows have missing frames. First: {missing[0]}"
        )
    if not ready:
        raise ValueError("No complete four-frame actions are available")
    return ready, {
        "selection_metadata": metadata,
        "selected_rows": len(rows),
        "output_rows": len(ready),
        "missing_rows": missing,
        "rows_with_checked_bin_center_times": checked_frame_times,
        "is_final": not allow_partial and not missing,
        "allow_partial": allow_partial,
    }


def tokenize(text):
    return re.findall(TOKEN_PATTERN, unicodedata.normalize("NFKC", text).lower())


def normalize(array, np):
    norms = np.linalg.norm(array, axis=-1, keepdims=True)
    return np.divide(array, norms, out=np.zeros_like(array), where=norms > 0)


def text_features(rows, glove_path, np):
    tokens = [tokenize(row["narration"]) for row in rows]
    requested = {token for row_tokens in tokens for token in row_tokens}
    vectors = {}
    digest = hashlib.sha256()
    archive_digest = None
    archive = None
    if zipfile.is_zipfile(glove_path):
        archive_digest = verify_hash(glove_path, GLOVE_ZIP_SHA256)
        archive = zipfile.ZipFile(glove_path)
        stream = archive.open("glove.6B.300d.txt")
    else:
        stream = glove_path.open("rb")
    try:
        for raw in stream:
            digest.update(raw)
            word_bytes, _, vector_bytes = raw.partition(b" ")
            word = word_bytes.decode("utf-8")
            if word in requested:
                vector = np.fromstring(vector_bytes.decode("ascii"), sep=" ", dtype=np.float32)
                if vector.shape != (300,) or not np.isfinite(vector).all():
                    raise ValueError(f"Invalid 300d GloVe vector for {word!r}")
                vectors[word] = vector
    finally:
        stream.close()
        if archive is not None:
            archive.close()
    if digest.hexdigest() != GLOVE_MEMBER_SHA256:
        raise ValueError("Original GloVe 6B 300d member SHA256 mismatch")
    features = np.zeros((len(rows), 300), dtype=np.float32)
    row_coverage, missing_counts = [], Counter()
    for index, row_tokens in enumerate(tokens):
        known = [token for token in row_tokens if token in vectors]
        unknown = [token for token in row_tokens if token not in vectors]
        missing_counts.update(unknown)
        if known:
            # Sort only to make bag-of-words collisions byte-identical even if
            # phrases reorder their words; this does not retain word order.
            features[index] = np.stack([vectors[token] for token in sorted(known)]).mean(axis=0)
        row_coverage.append({
            "uid": str(rows[index]["uid"]), "tokens": row_tokens,
            "known_tokens": len(known), "total_tokens": len(row_tokens),
            "oov_tokens": unknown,
        })
    features = normalize(features, np)
    collision_groups = defaultdict(list)
    for row, vector in zip(rows, features):
        collision_groups[hashlib.sha256(vector.tobytes()).hexdigest()].append(str(row["uid"]))
    total_tokens = sum(map(len, tokens))
    known_tokens = sum(item["known_tokens"] for item in row_coverage)
    metadata = {
        "artifact_sha256": archive_digest or GLOVE_MEMBER_SHA256,
        "member_sha256": digest.hexdigest(),
        "tokenization": {"unicode_normalization": "NFKC", "case": "lower", "regex": TOKEN_PATTERN},
        "pooling": "unweighted mean of known token vectors, including repeated tokens; L2 normalize",
        "oov_handling": "omit unknown tokens; all-unknown or empty narration produces zero vector",
        "fitting": "none; vocabulary lookup only, including evaluation narrations",
        "unique_requested_tokens": len(requested), "unique_known_tokens": len(vectors),
        "total_tokens": total_tokens, "known_tokens": known_tokens,
        "token_coverage": known_tokens / total_tokens if total_tokens else 0,
        "oov_counts": dict(sorted(missing_counts.items())),
        "row_coverage": row_coverage,
        "identical_vector_groups": [ids for ids in collision_groups.values() if len(ids) > 1],
    }
    return features, metadata


def load_dino(source, checkpoint, torch):
    verify_hash(checkpoint, DINO_SHA256)
    for name, expected in SOURCE_SHA256.items():
        verify_hash(source / name, expected)
    # Pin both imports by verified local filenames, without hub downloads or
    # importing an unrelated `utils` from the process's Python search path.
    for name, filename in [("utils", "utils.py"), ("epic_dino_vit", "vision_transformer.py")]:
        spec = importlib.util.spec_from_file_location(name, source / filename)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    model = sys.modules["epic_dino_vit"].vit_base(patch_size=16, num_classes=0)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval().requires_grad_(False)
    return model


def preprocess(path, np, torch, Image):
    with Image.open(path) as source:
        image = source.convert("RGB")
        width, height = image.size
        if width <= height:
            resized = (256, int(256 * height / width))
        else:
            resized = (int(256 * width / height), 256)
        image = image.resize(resized, resample=Image.Resampling.BICUBIC)
        left = int(round((resized[0] - 224) / 2))
        top = int(round((resized[1] - 224) / 2))
        image = image.crop((left, top, left + 224, top + 224))
        pixels = np.asarray(image, dtype=np.float32).copy() / np.float32(255.0)
    pixels = (pixels - np.asarray(FRAME_RECIPE["normalization_mean"], dtype=np.float32)) / np.asarray(
        FRAME_RECIPE["normalization_std"], dtype=np.float32
    )
    return torch.from_numpy(pixels.transpose(2, 0, 1).copy())


def atomic_npz(path, np, **arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, path)


def image_features(rows, model, args, np, torch, Image, pillow_version):
    runtime_recipe = {
        **FRAME_RECIPE, "torch_version": torch.__version__, "numpy_version": np.__version__,
        "pillow_version": pillow_version, "device": str(args.device),
        "cuda_version": torch.version.cuda, "batch_size": args.batch_size,
        "device_name": torch.cuda.get_device_name(torch.device(args.device)) if args.device.startswith("cuda") else "CPU",
        "extractor_sha256": sha256(Path(__file__)),
    }
    fingerprint = hashlib.sha256(json.dumps(runtime_recipe, sort_keys=True).encode()).hexdigest()
    by_video = defaultdict(set)
    for row in rows:
        by_video[str(row["video_id"])].update(row["_frame_paths"])
    features_by_path, input_hashes = {}, {}
    reused, computed = 0, 0
    model = model.to(args.device)
    for video, paths in sorted(by_video.items()):
        paths = sorted(paths)
        digests = {path: sha256(path) for path in paths}
        input_hashes.update(digests)
        safe_video = re.sub(r"[^a-zA-Z0-9_-]", "_", video)
        cache_path = args.cache_dir / (safe_video + "-" + hashlib.sha256(video.encode()).hexdigest()[:8] + ".npz")
        cached = {}
        if cache_path.exists():
            with np.load(cache_path, allow_pickle=False) as cache:
                if str(cache["fingerprint"].item()) == fingerprint:
                    cache_vectors = cache["features"]
                    if cache_vectors.shape != (len(cache["frame_paths"]), 768) or not np.isfinite(cache_vectors).all():
                        raise ValueError(f"Invalid feature cache {cache_path}")
                    for path, digest, vector in zip(cache["frame_paths"], cache["digests"], cache_vectors):
                        if path in digests and digest == digests[path]:
                            cached[str(path)] = vector.copy()
        pending = [path for path in paths if path not in cached]
        reused += len(cached)
        print(json.dumps({"video": video, "cached_frames": len(cached), "pending_frames": len(pending)}), flush=True)
        for start in range(0, len(pending), args.batch_size):
            batch_paths = pending[start:start + args.batch_size]
            inputs = torch.stack([preprocess(path, np, torch, Image) for path in batch_paths]).to(args.device)
            with torch.inference_mode():
                output = model(inputs).float()
                output = torch.nn.functional.normalize(output, p=2, dim=1).cpu().numpy()
            if output.shape != (len(batch_paths), 768) or not np.isfinite(output).all():
                raise ValueError("DINO produced invalid features")
            cached.update(zip(batch_paths, output))
            computed += len(batch_paths)
        if pending or not cache_path.exists():
            atomic_npz(cache_path, np, frame_paths=np.asarray(paths),
                       digests=np.asarray([digests[p] for p in paths]),
                       features=np.stack([cached[p] for p in paths]),
                       fingerprint=np.asarray(fingerprint))
        features_by_path.update(cached)
    pooled = np.stack([
        np.stack([features_by_path[path] for path in row["_frame_paths"]]).mean(axis=0)
        for row in rows
    ]).astype(np.float32)
    pooled = normalize(pooled, np)
    metadata = {
        "recipe": runtime_recipe, "cache_fingerprint": fingerprint,
        "unique_frames": len(input_hashes), "computed_frames": computed,
        "reused_frames": reused,
        "frame_sha256": [{"path": path, "sha256": digest} for path, digest in sorted(input_hashes.items())],
    }
    return pooled, metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--frame-root", type=Path, default=Path("."))
    parser.add_argument("--dino-source", type=Path, required=True)
    parser.add_argument("--dino-checkpoint", type=Path, required=True)
    parser.add_argument("--glove", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    args.cache_dir = args.cache_dir or args.output.parent / "frame-cache"
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import numpy as np
    import torch
    import PIL
    from PIL import Image

    torch.manual_seed(0)
    np.random.seed(0)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; refusing implicit CPU fallback")
    began = time.monotonic()
    rows, selection = read_manifest(args.manifest, args.frame_root, args.allow_partial)
    print(json.dumps({"stage": "manifest", "selected": selection["selected_rows"], "ready": len(rows)}), flush=True)
    text, text_metadata = text_features(rows, args.glove, np)
    print(json.dumps({"stage": "glove", "coverage": text_metadata["token_coverage"]}), flush=True)
    model = load_dino(args.dino_source, args.dino_checkpoint, torch)
    image, image_metadata = image_features(rows, model, args, np, torch, Image, PIL.__version__)
    image_nonzero = np.linalg.norm(image, axis=1) > 0
    text_nonzero = np.linalg.norm(text, axis=1) > 0
    eligible = image_nonzero & text_nonzero
    arrays = {key: np.asarray([str(row[key]) for row in rows]) for key in METADATA_KEYS}
    arrays.update(image=image, text=text, eligible=eligible)
    arrays["frame_paths"] = np.asarray([row["frame_paths"] for row in rows])
    arrays["text_token_count"] = np.asarray([item["total_tokens"] for item in text_metadata["row_coverage"]], dtype=np.int64)
    arrays["text_known_token_count"] = np.asarray([item["known_tokens"] for item in text_metadata["row_coverage"]], dtype=np.int64)
    atomic_npz(args.output, np, **arrays)
    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "script_sha256": sha256(Path(__file__)), "manifest_sha256": sha256(args.manifest),
        "feature_file_sha256": sha256(args.output), "feature_file": args.output.name,
        "selection": selection, "vision": image_metadata, "language": text_metadata,
        "eligibility_rule": "Include a row in every arm iff four frames are available and both pooled vectors are finite and nonzero; no score-dependent filtering.",
        "eligibility": {
            "eligible": int(eligible.sum()), "ineligible": int((~eligible).sum()),
            "zero_image_uids": [str(row["uid"]) for row, ok in zip(rows, image_nonzero) if not ok],
            "zero_text_uids": [str(row["uid"]) for row, ok in zip(rows, text_nonzero) if not ok],
        },
        "shapes": {key: list(value.shape) for key, value in arrays.items()},
        "device_name": torch.cuda.get_device_name(torch.device(args.device)) if args.device.startswith("cuda") else "CPU",
        "seconds": time.monotonic() - began,
    }
    sidecar = args.output.with_suffix(".json")
    temporary = sidecar.with_name(sidecar.name + ".tmp")
    temporary.write_text(json.dumps(metadata, indent=2) + "\n")
    os.replace(temporary, sidecar)
    print(json.dumps({"stage": "complete", "output": str(args.output), "rows": len(rows), "eligible": int(eligible.sum()), "is_final": selection["is_final"], "seconds": metadata["seconds"]}), flush=True)


if __name__ == "__main__":
    main()
