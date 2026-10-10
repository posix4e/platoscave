#!/usr/bin/env python3
"""Extract locked DINO/GloVe features from one complete sealed fresh-photo pilot.

This performs inference only. It neither authenticates human capture chronology
nor establishes absence of familiar concepts in pretraining. It refuses partial
packets, changed published source/model bytes, invalid vectors and existing
output directories. No network access or model downloads are implemented.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import ctypes
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import importlib.util
import io
import json
import os
from pathlib import Path, PurePosixPath
import platform
import re
import shutil
import sys
import tempfile
import unicodedata
import zipfile


PROTOCOL_ID = "fresh-objects-v1"
CLASSES = ("mug", "bowl", "plate", "bottle", "spoon", "fork", "book", "shoe")
LOCKED_FILES = {"protocol.json", "intake.py", "extract.py", "analyze.py"}
INPUT_FILES = {"protocol.json", "release-lock.json", "session-plan.json", "rounds.csv", "samples.csv"}
DINO_COMMIT = "7c446df5b9f45747937fb0d72314eb9f7b66930a"
DINO_SHA256 = "bf34ad0f424b9029b593e8dc3ed553bf26e88bcba0d32bf3e62a6209cb64c85e"
GLOVE_ZIP_SHA256 = "6471382cdd837544bf3ac72497a38715e845897d265b2b424b4761832009c837"
GLOVE_MEMBER = "glove.6B.300d.txt"
GLOVE_MEMBER_SHA256 = "a12599d41e3589c7160be27fffe5b0080eccd0f0c75f46666c59f90188093c40"
SOURCE_SHA256 = {
    "vision_transformer.py": "b1f998d5f49ab43666b9fc6d007c5f6540c3ead2e8645e144f74045f63ff44d7",
    "utils.py": "962a97e1acda1c986dfd921275325e4083f9016dddadcf018f1ccf03fa600eab",
}
TOKEN_PATTERN = r"[a-z]+(?:'[a-z]+)?|[0-9]+"
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_hash(path, expected):
    if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected):
        raise ValueError(f"Invalid expected SHA256 for {path}")
    actual = sha256(path)
    if actual != expected:
        raise ValueError(f"SHA256 mismatch for {path}: {actual} != {expected}")
    return actual


def read_json(path):
    def unique_keys(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate JSON key {key!r} in {path}")
            result[key] = value
        return result
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle, object_pairs_hook=unique_keys)


def csv_records(path, key):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    if not reader.fieldnames or len(set(reader.fieldnames)) != len(reader.fieldnames):
        raise ValueError(f"Missing or repeated CSV columns: {path}")
    if any(None in row or any(value is None for value in row.values()) or not row.get(key) for row in rows):
        raise ValueError(f"Malformed CSV row: {path}")
    result = {row[key]: row for row in rows}
    if len(result) != len(rows):
        raise ValueError(f"Duplicate CSV {key}: {path}")
    return result


def packet_file(packet, relative):
    """Resolve a packet-relative regular file, rejecting traversal and symlinks."""
    if not isinstance(relative, str) or not relative or "\\" in relative:
        raise ValueError("Packet paths must be nonempty relative POSIX paths")
    path = PurePosixPath(relative)
    if path.is_absolute() or any(part in ("", ".", "..") for part in relative.split("/")):
        raise ValueError(f"Unsafe packet path: {relative}")
    current = Path(packet).resolve(strict=True)
    for part in path.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError(f"Symlink in packet path: {relative}")
    if not current.is_file():
        raise ValueError(f"Missing regular packet file: {relative}")
    return current


def validate_packet(sealed_path, packet, source_dir=None):
    """Verify the published code lock and sealed packet before any model load."""
    source_dir = Path(source_dir) if source_dir is not None else Path(__file__).resolve().parent
    sealed_path = Path(sealed_path).resolve(strict=True)
    sealed_digest = sha256(sealed_path)
    sealed = read_json(sealed_path)
    if sealed.get("schema_version") != 1 or sealed.get("protocol_id") != PROTOCOL_ID:
        raise ValueError("Not a sealed fresh-objects-v1 packet")
    lock_path = packet_file(packet, "release-lock.json")
    lock_digest = verify_hash(lock_path, sealed.get("release_lock_sha256"))
    verify_hash(source_dir / "release-lock.json", lock_digest)
    lock = read_json(lock_path)
    if lock.get("schema_version") != 1 or set(lock.get("files", {})) != LOCKED_FILES:
        raise ValueError("Release lock must name exactly the four published protocol/source files")
    snapshot = {sealed_path: sealed_digest, lock_path: lock_digest,
                source_dir / "release-lock.json": lock_digest}
    for name, digest in lock["files"].items():
        snapshot[source_dir / name] = verify_hash(source_dir / name, digest)
    protocol_path = packet_file(packet, "protocol.json")
    protocol_digest = verify_hash(protocol_path, sealed.get("protocol_sha256"))
    if lock["files"]["protocol.json"] != protocol_digest:
        raise ValueError("Packet protocol differs from the published release lock")
    protocol = read_json(protocol_path)
    if protocol.get("protocol_id") != PROTOCOL_ID or protocol.get("classes") != list(CLASSES):
        raise ValueError("Unexpected protocol identity or noun classes")
    if sealed.get("intake_sha256") != lock["files"]["intake.py"]:
        raise ValueError("Seal was not made with the locked intake source")
    records = sealed.get("input_records", {})
    if set(records) != INPUT_FILES:
        raise ValueError("Seal lacks the complete five-file packet input record")
    for name, record in records.items():
        path = packet_file(packet, name)
        snapshot[path] = verify_hash(path, record.get("sha256"))
        if path.stat().st_size != record.get("size_bytes"):
            raise ValueError(f"Sealed byte count changed: {name}")
    if records["protocol.json"]["sha256"] != protocol_digest or records["release-lock.json"]["sha256"] != lock_digest:
        raise ValueError("Seal contains conflicting protocol or release lock digests")
    if sealed.get("session_plan_sha256") != records["session-plan.json"]["sha256"]:
        raise ValueError("Seal contains conflicting session plan digests")
    plan = read_json(packet_file(packet, "session-plan.json"))
    for key in ("protocol_id", "protocol_sha256", "release_lock_sha256", "intake_sha256", "run_id", "collector_id"):
        if plan.get(key) != sealed.get(key):
            raise ValueError(f"Seal differs from session plan: {key}")
    sample_records = csv_records(packet_file(packet, "samples.csv"), "sample_id")
    round_records = csv_records(packet_file(packet, "rounds.csv"), "round_id")
    attestation = sealed.get("attestation", {})
    for field in ("human_capture", "human_captions", "marker_photos_inspected"):
        if attestation.get(field) is not True:
            raise ValueError(f"Missing human intake attestation: {field}")
    rows = sealed.get("rows")
    rounds = sealed.get("rounds")
    if not isinstance(rows, list) or len(rows) != 48 or not isinstance(rounds, list) or len(rounds) != 6:
        raise ValueError("Exactly 48 scored rows and six rounds are required")
    round_ids = [f"r{i:02d}" for i in range(1, 7)]
    if [row.get("round_id") for row in rounds] != round_ids:
        raise ValueError("Sealed rounds must be ordered r01 through r06")
    if len(plan.get("rounds", [])) != 6 or set(round_records) != set(round_ids):
        raise ValueError("Plan or CSV does not contain all six rounds")
    expected_ids, marker_digests, image_digests = [], set(), set()
    for index, round_row in enumerate(rounds):
        for record in (plan["rounds"][index], round_records[round_ids[index]]):
            if any(round_row.get(key) != value for key, value in record.items()):
                raise ValueError("Sealed round differs from the original plan or CSV")
        order = round_row.get("order", [])
        if len(order) != 8 or set(order) != set(CLASSES):
            raise ValueError("Each round must contain each noun class once")
        expected_ids.extend(f"{round_ids[index]}-{label}" for label in order)
        expected_split = "fit" if index < 4 else "development"
        if round_row.get("split") != expected_split:
            raise ValueError("Round partition disagrees with frozen protocol")
        marker = packet_file(packet, round_row["marker_file"])
        digest = verify_hash(marker, round_row.get("marker_sha256"))
        if marker.stat().st_size != round_row.get("marker_size_bytes"):
            raise ValueError("Marker byte count changed")
        if digest in marker_digests:
            raise ValueError("Each round requires a distinct marker photograph")
        marker_digests.add(digest)
        snapshot[marker] = digest
    if [row.get("sample_id") for row in rows] != expected_ids:
        raise ValueError("Sample identities or order differ from the sealed round schedule")
    if set(sample_records) != set(expected_ids):
        raise ValueError("Sample CSV identities differ from the sealed packet")
    for row in rows:
        if any(row.get(key) != value for key, value in sample_records[row["sample_id"]].items()):
            raise ValueError("Sealed sample differs from the original CSV")
        round_index = round_ids.index(row["round_id"])
        label = row.get("class_name")
        expected_split = "fit" if round_index < 4 else "development"
        instance = f"{label}-{'A' if round_index < 4 else 'B'}"
        if label not in CLASSES or row.get("split") != expected_split or row.get("instance_id") != instance:
            raise ValueError(f"Class, instance or partition mismatch: {row['sample_id']}")
        if row["sample_id"] != f"{row['round_id']}-{label}":
            raise ValueError("Sample ID does not match its class and round")
        if not isinstance(row.get("caption"), str) or not row["caption"].strip():
            raise ValueError(f"Missing human caption: {row['sample_id']}")
        path = packet_file(packet, row["image_file"])
        digest = verify_hash(path, row.get("image_sha256"))
        if path.stat().st_size != row.get("image_size_bytes"):
            raise ValueError(f"Image byte count changed: {row['sample_id']}")
        if digest in image_digests or digest in marker_digests:
            raise ValueError("Repeated image bytes or marker photograph used as a scored image")
        image_digests.add(digest)
        snapshot[path] = digest
    if Counter(row["split"] for row in rows) != {"fit": 32, "development": 16}:
        raise ValueError("Expected 32 fitting and 16 development photographs")
    if sha256(sealed_path) != sealed_digest:
        raise ValueError("Seal changed during validation")
    return sealed, snapshot


def tokenize(caption):
    return re.findall(TOKEN_PATTERN, unicodedata.normalize("NFKC", caption).lower())


def unit_rows(array, np, name):
    array = np.asarray(array, dtype=np.float32)
    if array.ndim != 2 or not np.isfinite(array).all():
        raise ValueError(f"Invalid {name} matrix")
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    if not np.isfinite(norms).all() or np.any(norms <= 0):
        raise ValueError(f"Zero or nonfinite {name} vector")
    output = (array / norms).astype(np.float32)
    if not np.isfinite(output).all() or not np.allclose(np.linalg.norm(output, axis=1), 1, atol=1e-4, rtol=0):
        raise ValueError(f"Failed to normalize {name} vectors")
    return output


def text_features(rows, glove_path, np):
    """Read the entire pinned member before using any requested word vectors."""
    verify_hash(glove_path, GLOVE_ZIP_SHA256)
    token_rows = [tokenize(row["caption"]) for row in rows]
    requested = set(token for tokens in token_rows for token in tokens)
    vectors, digest = {}, hashlib.sha256()
    with zipfile.ZipFile(glove_path) as archive:
        if archive.namelist().count(GLOVE_MEMBER) != 1:
            raise ValueError("GloVe archive must contain exactly one original 300d member")
        with archive.open(GLOVE_MEMBER) as stream:
            for raw in stream:
                digest.update(raw)
                word_bytes, _, vector_bytes = raw.partition(b" ")
                word = word_bytes.decode("utf-8")
                if word in requested:
                    vector = np.fromstring(vector_bytes.decode("ascii"), sep=" ", dtype=np.float32)
                    if vector.shape != (300,) or not np.isfinite(vector).all() or word in vectors:
                        raise ValueError(f"Invalid or repeated GloVe vector: {word}")
                    vectors[word] = vector
    if digest.hexdigest() != GLOVE_MEMBER_SHA256:
        raise ValueError("Original GloVe 6B 300d member SHA256 mismatch")
    means, coverage, unknown_counts = [], [], Counter()
    for row, tokens in zip(rows, token_rows):
        known = [token for token in tokens if token in vectors]
        unknown = [token for token in tokens if token not in vectors]
        if not known:
            raise ValueError(f"Caption has no known GloVe tokens: {row['sample_id']}")
        means.append(np.stack([vectors[token] for token in sorted(known)]).mean(axis=0, dtype=np.float32))
        unknown_counts.update(unknown)
        coverage.append({"sample_id": row["sample_id"], "tokens": tokens,
                         "known_tokens": len(known), "total_tokens": len(tokens), "oov_tokens": unknown})
    features = unit_rows(np.stack(means), np, "text")
    groups = defaultdict(list)
    for row, vector in zip(rows, features):
        groups[hashlib.sha256(vector.tobytes()).hexdigest()].append(row["sample_id"])
    return features, {"rows": coverage, "oov_counts": dict(sorted(unknown_counts.items())),
                      "tokenizer": {"normalization": "NFKC", "case": "lower", "regex": TOKEN_PATTERN},
                      "pooling": "float32 mean of sorted known token occurrences, retaining repeats and stopwords; L2",
                      "identical_vector_groups": [ids for ids in groups.values() if len(ids) > 1]}


def preprocess_bytes(raw, np, Image, ImageOps):
    with Image.open(io.BytesIO(raw)) as source:
        if source.format not in {"JPEG", "PNG"} or getattr(source, "n_frames", 1) != 1:
            raise ValueError("Only single-frame original JPEG or PNG photographs are accepted")
        image = ImageOps.exif_transpose(source).convert("RGB")
        width, height = image.size
        if min(width, height) < 224:
            raise ValueError("Original photograph short side is below 224 pixels")
        resized = (256, int(256 * height / width)) if width <= height else (int(256 * width / height), 256)
        image = image.resize(resized, resample=Image.Resampling.BICUBIC)
        left, top = int(round((resized[0] - 224) / 2)), int(round((resized[1] - 224) / 2))
        image = image.crop((left, top, left + 224, top + 224))
        pixels = np.asarray(image, dtype=np.float32).copy() / np.float32(255)
    pixels = (pixels - np.asarray(MEAN, dtype=np.float32)) / np.asarray(STD, dtype=np.float32)
    return pixels.transpose(2, 0, 1).copy()


def load_dino(source, checkpoint, torch):
    verify_hash(checkpoint, DINO_SHA256)
    for filename, expected in SOURCE_SHA256.items():
        verify_hash(source / filename, expected)
    module_names = ("utils", "_fresh_pilot_dino_vit")
    previous = {name: sys.modules.get(name) for name in module_names}
    try:
        loaded = {}
        for name, filename in zip(module_names, ("utils.py", "vision_transformer.py")):
            spec = importlib.util.spec_from_file_location(name, source / filename)
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
            loaded[name] = module
        model = loaded["_fresh_pilot_dino_vit"].vit_base(patch_size=16, num_classes=0)
        model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True)
        return model.eval().requires_grad_(False)
    finally:
        for name, module in previous.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


def rename_new_directory(source, destination):
    """Atomic, no-replacement rename on the supported Linux and macOS hosts."""
    libc = ctypes.CDLL(None, use_errno=True)
    encoded = (os.fsencode(source), os.fsencode(destination))
    if sys.platform == "darwin":
        function = libc.renamex_np
        function.argtypes = (ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint)
        function.restype = ctypes.c_int
        result = function(*encoded, 4)  # RENAME_EXCL
    elif sys.platform.startswith("linux") and hasattr(libc, "renameat2"):
        function = libc.renameat2
        function.argtypes = (ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint)
        function.restype = ctypes.c_int
        result = function(-100, encoded[0], -100, encoded[1], 1)  # AT_FDCWD, RENAME_NOREPLACE
    else:
        raise RuntimeError("This host lacks a supported atomic no-replacement directory rename")
    if result:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error), str(destination))


def publish_output(output, arrays, metadata, np):
    """Write a new directory only after both artifacts have been completed."""
    output = Path(output).absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Refusing existing output path: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent))
    try:
        feature_path = staging / "features.npz"
        with feature_path.open("wb") as handle:
            np.savez_compressed(handle, **arrays)
        metadata = {**metadata, "feature_sha256": sha256(feature_path)}
        with (staging / "features.json").open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2, ensure_ascii=False, allow_nan=False)
            handle.write("\n")
        if output.exists() or output.is_symlink():
            raise FileExistsError(f"Output path appeared during extraction: {output}")
        rename_new_directory(staging, output)
        return metadata
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sealed", required=True, type=Path)
    parser.add_argument("--packet", required=True, type=Path)
    parser.add_argument("--dino-checkpoint", required=True, type=Path)
    parser.add_argument("--dino-source", required=True, type=Path)
    parser.add_argument("--glove-zip", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    if args.out.exists() or args.out.is_symlink():
        raise FileExistsError(f"Refusing existing output path: {args.out}")
    sealed, snapshot = validate_packet(args.sealed, args.packet)
    snapshot[args.dino_checkpoint] = verify_hash(args.dino_checkpoint, DINO_SHA256)
    snapshot[args.glove_zip] = verify_hash(args.glove_zip, GLOVE_ZIP_SHA256)
    for filename, digest in SOURCE_SHA256.items():
        snapshot[args.dino_source / filename] = verify_hash(args.dino_source / filename, digest)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    import numpy as np
    import torch
    import PIL
    from PIL import Image, ImageOps
    torch.manual_seed(0)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; refusing implicit CPU fallback")
    rows = sealed["rows"]
    text, token_metadata = text_features(rows, args.glove_zip, np)
    # The complete GloVe member, model files, source and all image bytes have now
    # been checked before the first image-model inference.
    model = load_dino(args.dino_source, args.dino_checkpoint, torch).to(args.device, dtype=torch.float32)
    image_vectors = []
    for row in rows:
        raw = packet_file(args.packet, row["image_file"]).read_bytes()
        if hashlib.sha256(raw).hexdigest() != row["image_sha256"]:
            raise ValueError(f"Photograph changed before inference: {row['sample_id']}")
        pixels = preprocess_bytes(raw, np, Image, ImageOps)
        inputs = torch.from_numpy(pixels).unsqueeze(0).to(args.device, dtype=torch.float32)
        with torch.inference_mode():
            vector = model(inputs).float().cpu().numpy()
        if vector.shape != (1, 768):
            raise ValueError("DINO did not return one 768-dimensional CLS vector")
        image_vectors.append(vector[0])
    image = unit_rows(np.stack(image_vectors), np, "image")
    if image.shape != (48, 768) or text.shape != (48, 300):
        raise ValueError("Incomplete feature matrices")
    for path, digest in snapshot.items():
        verify_hash(path, digest)
    versions = {"python": platform.python_version(), "numpy": np.__version__,
                "torch": torch.__version__, "pillow": PIL.__version__}
    try:
        versions["torchvision"] = importlib.metadata.version("torchvision")
    except importlib.metadata.PackageNotFoundError:
        versions["torchvision"] = None
    metadata = {
        "schema_version": 1, "protocol_id": PROTOCOL_ID, "status": "complete",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_sha256": sealed["protocol_sha256"], "sealed_sha256": snapshot[args.sealed.resolve()],
        "release_lock_sha256": sealed["release_lock_sha256"], "extractor_sha256": sha256(Path(__file__)),
        "sample_count": 48, "split_counts": {"fit": 32, "development": 16},
        "sample_ids": [row["sample_id"] for row in rows],
        "models_verified": {"image": True, "text": True},
        "model_hashes": {"dino_checkpoint_sha256": DINO_SHA256, "glove_zip_sha256": GLOVE_ZIP_SHA256,
                         "glove_member_sha256": GLOVE_MEMBER_SHA256},
        "source_hashes": SOURCE_SHA256, "dino_source_commit": DINO_COMMIT,
        "runtime": {**versions, "device": args.device, "cuda": torch.version.cuda,
                    "device_name": torch.cuda.get_device_name(0) if args.device == "cuda" else platform.processor(),
                    "torch_threads": torch.get_num_threads(), "float32": True, "tf32": False,
                    "deterministic_algorithms": True, "cublas_workspace_config": ":4096:8",
                    "image_batch_size": 1},
        "tokens": token_metadata,
        "provenance": {"run_id": sealed.get("run_id"), "collector_id": sealed.get("collector_id"),
                       "images": [{"sample_id": row["sample_id"], "image_file": row["image_file"],
                                   "image_sha256": row["image_sha256"]} for row in rows],
                       "no_model_training": True,
                       "limits": "Hash checks identify files; human capture chronology remains an attested assumption. Familiar concepts and phrases, curation and model-choice effects remain possible. No unpaired-alignment result is produced here."},
    }
    saved = publish_output(args.out, {"image": image, "text": text,
                                    "sample_ids": np.asarray(metadata["sample_ids"], dtype=str)}, metadata, np)
    print(json.dumps({"status": "complete", "output": str(args.out), "sample_count": 48,
                      "feature_sha256": saved["feature_sha256"]}))


if __name__ == "__main__":
    main()
