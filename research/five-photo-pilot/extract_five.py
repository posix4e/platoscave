#!/usr/bin/env python3
"""Extract one locked five-photo feature set and separate blind distance graphs.

All outputs are private. The numerical solver receives blind-distances.json
only. Keep private-ground-truth.json closed until the solver output is saved.
There is no model fitting, label rewriting, download, or correspondence search
here. A failed attempt retains preflight/failure records in its new directory.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path, PurePosixPath
import platform
import random
import re
import sys
import time
import types


PROTOCOL_ID = "five-photo-geometry-v1"
HELPER_SHA256 = "3e8445f22a8f255dcf47af89ac0b8c45461ba98046549702abc7b09b6b59a624"
LABELS = ("Slide", "Trashcan", "Tree", "Basketball", "Water fountain")
LOCK_FILES = {"protocol.json", "extract_five.py", "solve.py"}
DEFAULT_PROTOCOL = Path(__file__).with_name("protocol.json")
BLIND_FIELDS = {"schema_version", "protocol_id", "protocol_sha256", "feature_sha256",
                "image_ids", "text_ids", "image_distances", "text_distances"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_hash(path, expected):
    require(isinstance(expected, str) and re.fullmatch(r"[0-9a-f]{64}", expected), f"Invalid expected SHA256: {path}")
    actual = sha256(path)
    require(actual == expected, f"SHA256 mismatch: {path}")
    return actual


def read_json(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, f"Duplicate JSON key: {key}")
            result[key] = value
        return result
    def reject(value):
        raise ValueError(f"Nonfinite JSON value: {value}")
    return json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=unique, parse_constant=reject)


def local_file(root, name):
    require(isinstance(name, str) and name and "\\" not in name and "\x00" not in name,
            "Image file must be a relative POSIX path")
    relative = PurePosixPath(name)
    require(not relative.is_absolute() and all(part not in {"", ".", ".."} for part in name.split("/")),
            f"Unsafe image path: {name}")
    path = Path(root)
    require(path.is_dir() and not path.is_symlink(), "Image/source root must be a nonsymlink directory")
    for part in relative.parts:
        path = path / part
        require(not path.is_symlink(), f"Symlink rejected: {name}")
    require(path.is_file() and path.resolve().is_relative_to(Path(root).resolve()), f"Missing or external file: {name}")
    return path


def validate_manifest(manifest, image_root):
    manifest, image_root = Path(manifest).absolute(), Path(image_root).absolute()
    require(manifest.is_file() and not manifest.is_symlink(), "Manifest must be a regular nonsymlink file")
    before = sha256(manifest)
    document = read_json(manifest)
    require(set(document) == {"schema_version", "protocol_id", "rows"}
            and document.get("schema_version") == 1 and document.get("protocol_id") == PROTOCOL_ID,
            "Unexpected five-photo manifest schema")
    rows = document["rows"]
    require(isinstance(rows, list) and len(rows) == 5 and all(isinstance(row, dict) for row in rows),
            "Exactly five manifest rows are required")
    ids, labels, paths, hashes = set(), set(), set(), set()
    snapshot = {manifest: before}
    for row in rows:
        require(set(row) == {"candidate_id", "image_file", "label_verbatim", "image_sha256"},
                "Unexpected manifest row fields")
        candidate = row["candidate_id"]
        require(isinstance(candidate, str) and candidate.strip() and candidate not in ids,
                "Candidate IDs must be nonempty unique strings")
        ids.add(candidate)
        label = row["label_verbatim"]
        require(isinstance(label, str) and label in LABELS and label not in labels,
                "Use each of the five exact human labels once, without rewriting")
        labels.add(label)
        path = local_file(image_root, row["image_file"])
        require(path.resolve() not in paths, "Each candidate requires a distinct image path")
        paths.add(path.resolve())
        digest = verify_hash(path, row["image_sha256"])
        require(digest not in hashes, "Duplicate image bytes in the five-photo manifest")
        hashes.add(digest)
        snapshot[path] = digest
    require(labels == set(LABELS), "The human label set differs from the protocol")
    require(sha256(manifest) == before, "Manifest changed during validation")
    return rows, snapshot


def verify_release(protocol):
    protocol = Path(protocol).absolute()
    require(protocol.name == "protocol.json" and protocol.is_file() and not protocol.is_symlink(),
            "Use the release's regular protocol.json file")
    source_dir = protocol.parent
    lock_path = local_file(source_dir, "release-lock.json")
    lock = read_json(lock_path)
    require(lock.get("schema_version") == 1 and isinstance(lock.get("files"), dict)
            and set(lock["files"]) == LOCK_FILES, "Release lock must name exactly protocol.json, extract_five.py and solve.py")
    snapshot = {lock_path: sha256(lock_path)}
    for name, expected in lock["files"].items():
        path = local_file(source_dir, name)
        snapshot[path] = verify_hash(path, expected)
    require(lock["files"]["extract_five.py"] == sha256(__file__), "Running extractor differs from the locked release")
    document = read_json(protocol)
    require(document.get("schema_version") == 1 and document.get("protocol_id") == PROTOCOL_ID,
            "Unexpected protocol identity")
    require(document.get("data", {}).get("items") == 5
            and document.get("data", {}).get("human_labels_verbatim") == list(LABELS), "Protocol labels/count differ")
    require(document.get("models", {}).get("helper_sha256") == HELPER_SHA256, "Protocol helper hash differs")
    return document, snapshot


def load_locked_helper(path):
    path = Path(path).absolute()
    require(path.is_file() and not path.is_symlink(), "Helper must be a regular nonsymlink file")
    # Hash the exact bytes subsequently compiled: no code executes beforehand.
    raw = path.read_bytes()
    require(hashlib.sha256(raw).hexdigest() == HELPER_SHA256, "Frozen helper SHA256 mismatch; refusing import")
    module = types.ModuleType("_five_photo_locked_helper")
    module.__file__ = str(path)
    exec(compile(raw, str(path), "exec"), module.__dict__)
    return module


def build_blind_distances(image, text, rows, np, protocol_sha256, feature_sha256):
    require(len(rows) == 5, "Blind graphs require exactly five rows")
    for value, name in ((protocol_sha256, "protocol"), (feature_sha256, "feature")):
        require(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value), f"Invalid {name} hash")
    image_order, text_order = list(range(5)), list(range(5))
    random.Random(20261010).shuffle(image_order)
    random.Random(20261011).shuffle(text_order)
    def distances(features, dimensions, order):
        values = np.asarray(features, dtype=np.float64)
        require(values.shape == (5, dimensions) and np.isfinite(values).all(), "Invalid five-row feature dimensions or values")
        norms = np.linalg.norm(values, axis=1, keepdims=True)
        require(np.isfinite(norms).all() and (norms > 0).all(), "Invalid feature norms")
        values = (values / norms)[order]
        result = 1.0 - values @ values.T
        result = (result + result.T) / 2.0
        np.fill_diagonal(result, 0.0)
        require(np.isfinite(result).all(), "Nonfinite cosine distances")
        return result.tolist()
    image_ids, text_ids = [f"I{i:02d}" for i in range(5)], [f"T{i:02d}" for i in range(5)]
    blind = {"schema_version": 1, "protocol_id": PROTOCOL_ID, "protocol_sha256": protocol_sha256,
             "feature_sha256": feature_sha256, "image_ids": image_ids, "text_ids": text_ids,
             "image_distances": distances(image, 768, image_order), "text_distances": distances(text, 300, text_order)}
    require(set(blind) == BLIND_FIELDS, "Unexpected field in blind solver input")
    truth = {"mapping": {image_ids[i]: text_ids[text_order.index(original)] for i, original in enumerate(image_order)},
             "labels": {text_ids[i]: rows[original]["label_verbatim"] for i, original in enumerate(text_order)},
             "image_labels": {image_ids[i]: rows[original]["label_verbatim"] for i, original in enumerate(image_order)}}
    private_orders = {"image_shuffle_seed": 20261010, "text_shuffle_seed": 20261011,
                      "image_manifest_indices": image_order, "text_manifest_indices": text_order,
                      "image_candidate_ids": [rows[i]["candidate_id"] for i in image_order],
                      "text_candidate_ids": [rows[i]["candidate_id"] for i in text_order]}
    return blind, truth, private_orders


def write_new_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def extract(args):
    output = Path(args.out).absolute()
    require(not output.exists() and not output.is_symlink(), "Output directory already exists; refusing overwrite")
    began = time.monotonic()
    protocol, snapshot = verify_release(args.protocol)
    rows, image_snapshot = validate_manifest(args.manifest, args.image_root)
    snapshot.update(image_snapshot)
    require(set(row["candidate_id"] for row in rows) == set(protocol["data"]["candidate_ids"]), "Manifest candidate IDs differ from protocol")
    helper = load_locked_helper(args.helper)
    snapshot[Path(args.helper).absolute()] = HELPER_SHA256
    image_model, text_model = protocol["models"]["image"], protocol["models"]["text"]
    require(image_model["checkpoint_sha256"] == helper.DINO_SHA256
            and image_model["source_sha256"] == helper.SOURCE_SHA256
            and image_model["source_commit"] == helper.DINO_COMMIT
            and text_model["archive_sha256"] == helper.GLOVE_ZIP_SHA256
            and text_model["member_sha256"] == helper.GLOVE_MEMBER_SHA256, "Protocol model locks differ from the frozen helper")
    for path, expected in ((Path(args.dino_checkpoint), helper.DINO_SHA256), (Path(args.glove_zip), helper.GLOVE_ZIP_SHA256)):
        snapshot[path.absolute()] = verify_hash(path, expected)
    for filename, expected in helper.SOURCE_SHA256.items():
        path = Path(args.dino_source) / filename
        snapshot[path.absolute()] = verify_hash(path, expected)
    protocol_hash = snapshot[Path(args.protocol).absolute()]
    preflight = {"schema_version": 1, "protocol_id": PROTOCOL_ID, "status": "inputs_verified_before_inference",
                 "recorded_at_utc": datetime.now(timezone.utc).isoformat(), "protocol_sha256": protocol_hash,
                 "extractor_sha256": sha256(__file__), "helper_sha256": HELPER_SHA256,
                 "manifest_sha256": snapshot[Path(args.manifest).absolute()],
                 "input_sha256": {str(path): digest for path, digest in snapshot.items()},
                 "private_manifest_rows": rows, "device_requested": args.device, "batch_size": 1}
    output.mkdir(parents=True, exist_ok=False, mode=0o700)
    write_new_json(output / "preflight.json", preflight)
    try:
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
        require(args.device != "cuda" or torch.cuda.is_available(), "CUDA requested but unavailable; no CPU fallback")
        # The helper sees each exact human string as its caption input. Its
        # fixed tokenizer/pooling is unchanged; no prompt or word is appended.
        helper_rows = [{"sample_id": row["candidate_id"], "caption": row["label_verbatim"]} for row in rows]
        text, text_metadata = helper.text_features(helper_rows, Path(args.glove_zip), np)
        total_tokens = sum(row["total_tokens"] for row in text_metadata["rows"])
        known_tokens = sum(row["known_tokens"] for row in text_metadata["rows"])
        text_metadata = {**text_metadata, "total_tokens": total_tokens, "known_tokens": known_tokens,
                         "token_coverage": known_tokens / total_tokens if total_tokens else 0.0}
        model = helper.load_dino(Path(args.dino_source), Path(args.dino_checkpoint), torch).to(args.device, dtype=torch.float32)
        vectors = []
        for row in rows:
            path = local_file(Path(args.image_root).absolute(), row["image_file"])
            raw = path.read_bytes()
            require(hashlib.sha256(raw).hexdigest() == row["image_sha256"], "Photograph changed before inference")
            pixels = helper.preprocess_bytes(raw, np, Image, ImageOps)
            inputs = torch.from_numpy(pixels).unsqueeze(0).to(args.device, dtype=torch.float32)
            with torch.inference_mode():
                vector = model(inputs).float().cpu().numpy()
            require(vector.shape == (1, 768), "DINO must return one 768-dimensional CLS per photograph")
            vectors.append(vector[0])
        image = helper.unit_rows(np.stack(vectors), np, "five images")
        for matrix, dimension in ((image, 768), (text, 300)):
            require(matrix.dtype == np.dtype(np.float32) and matrix.shape == (5, dimension)
                    and np.isfinite(matrix).all() and np.allclose(np.linalg.norm(matrix, axis=1), 1, atol=1e-4, rtol=0),
                    "Incomplete, nonfinite or nonunit feature matrix")
        for path, expected in snapshot.items():
            verify_hash(path, expected)
        with (output / "features.npz").open("xb") as stream:
            np.savez_compressed(stream, image=image, text=text,
                                sample_ids=np.asarray([row["candidate_id"] for row in rows]))
        feature_hash = sha256(output / "features.npz")
        blind, truth, private_orders = build_blind_distances(image, text, rows, np, protocol_hash, feature_hash)
        write_new_json(output / "blind-distances.json", blind)
        write_new_json(output / "private-ground-truth.json", truth)
        versions = {"python": platform.python_version(), "numpy": np.__version__, "torch": str(torch.__version__), "pillow": PIL.__version__}
        try:
            versions["torchvision"] = importlib.metadata.version("torchvision")
        except importlib.metadata.PackageNotFoundError:
            versions["torchvision"] = None
        metadata = {"schema_version": 1, "protocol_id": PROTOCOL_ID, "status": "complete",
                    "created_at_utc": datetime.now(timezone.utc).isoformat(), "protocol_sha256": protocol_hash,
                    "extractor_sha256": preflight["extractor_sha256"], "helper_sha256": HELPER_SHA256,
                    "release_lock_sha256": snapshot[Path(args.protocol).absolute().parent / "release-lock.json"],
                    "manifest_sha256": preflight["manifest_sha256"], "preflight_sha256": sha256(output / "preflight.json"),
                    "feature_sha256": feature_hash, "input_sha256": preflight["input_sha256"],
                    "sample_count": 5, "sample_ids": [row["candidate_id"] for row in rows],
                    "private_manifest_rows": rows, "private_blind_order": private_orders,
                    "models_verified": {"image": True, "text": True},
                    "model_hashes": {"dino_checkpoint_sha256": helper.DINO_SHA256, "glove_zip_sha256": helper.GLOVE_ZIP_SHA256,
                                     "glove_member_sha256": helper.GLOVE_MEMBER_SHA256},
                    "dino_source_sha256": helper.SOURCE_SHA256, "dino_source_commit": helper.DINO_COMMIT,
                    "text_coverage": text_metadata,
                    "runtime": {**versions, "device": args.device, "cuda": torch.version.cuda,
                                "device_name": torch.cuda.get_device_name(0) if args.device == "cuda" else platform.processor(),
                                "batch_size": 1, "torch_threads": torch.get_num_threads(), "float32": True,
                                "tf32": False, "deterministic_algorithms": True, "cublas_workspace_config": ":4096:8"},
                    "distance_arithmetic": "Renormalize each feature row in float64; 1-dot; symmetrize; diagonal zero; no extra distance normalization",
                    "artifact_sha256": {name: sha256(output / name) for name in ("features.npz", "blind-distances.json", "private-ground-truth.json")},
                    "elapsed_seconds": time.monotonic() - began,
                    "privacy": "All outputs private. Give only blind-distances.json to the solver; evaluate ground truth only after solver output is saved."}
        write_new_json(output / "features.json", metadata)
        return metadata
    except BaseException as error:
        write_new_json(output / "failure.json", {"schema_version": 1, "protocol_id": PROTOCOL_ID, "status": "failed",
                       "recorded_at_utc": datetime.now(timezone.utc).isoformat(), "error_type": type(error).__name__,
                       "error": str(error), "preflight_sha256": sha256(output / "preflight.json")})
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("manifest", "image-root", "helper", "dino-source", "dino-checkpoint", "glove-zip", "out"):
        parser.add_argument("--" + flag, type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args(argv)
    try:
        metadata = extract(args)
    except (ValueError, OSError, RuntimeError, ImportError) as error:
        print(f"Five-photo extraction refused: {error}", file=sys.stderr)
        return 2
    print(json.dumps({"status": metadata["status"], "sample_count": metadata["sample_count"],
                      "feature_sha256": metadata["feature_sha256"], "output": str(args.out)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
