#!/usr/bin/env python3
"""COCO encoder adapters. Corpus extraction requires a reviewed, byte-bound lock.

No extraction lock ships with this draft. ``benchmark`` uses generated pixels or
artificial words only; it cannot accept a corpus or arbitrary caption inputs.
The model implementations remain the previously frozen helpers. This wrapper
changes only the prospectively declared COCO input/serialization contract.

The future JSON lock must have schema_version=1, protocol_id=
rosetta-model-controls-v2, status=frozen_before_model_control_extraction,
extractor_sha256, preprocessing_id=authors_dinov2_square224_v1, caption_slots=7,
models containing all five MODEL_CONTRACT entries plus exact runtime_versions,
files mapping every required logical input to its SHA256, and datasets containing
both train2014 and val2014 with annotation_sha256 and image_manifest_sha256.
--inputs supplies only machine-local paths for those logical inputs; directories
are derived from their file paths. --lock-sha256 binds the reviewed lock bytes.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import importlib.util
import io
import json
import os
from pathlib import Path
import platform
import traceback
import time
import zipfile

PROTOCOL_ID = "rosetta-model-controls-v2"
PREPROCESSING_ID = "authors_dinov2_square224_v1"
VISION_ENVIRONMENT = {"ATEN_CPU_CAPABILITY": "avx2", "MKL_CBWR": "AVX2,STRICT"}
EXPECTED_ITEMS = {"train2014": 82783, "val2014": 40504}
MODEL_CONTRACT = {
    "dino": {"dimension": 768, "dtype": "float32", "batch_size": 1, "device": "cuda", "torch_threads": 1, "interop_threads": 1, "runtime_environment": VISION_ENVIRONMENT, "readout": "original_final_cls"},
    "moco": {"dimension": 2048, "dtype": "float32", "batch_size": 1, "device": "cuda", "torch_threads": 1, "interop_threads": 1, "runtime_environment": VISION_ENVIRONMENT, "readout": "query_backbone_avgpool"},
    "mae": {"dimension": 768, "dtype": "float32", "batch_size": 1, "device": "cuda", "torch_threads": 1, "interop_threads": 1, "runtime_environment": VISION_ENVIRONMENT, "readout": "unmasked_final_cls"},
    "glove": {"dimension": 300, "dtype": "float32", "batch_size": 1, "device": "cpu", "readout": "mean_sorted_known_tokens", "oov_policy": "omit_and_count_unknown_tokens_fail_if_none_known"},
    "elmo": {"dimension": 1024, "dtype": "float32", "batch_size": 1, "device": "cpu", "torch_threads": 4, "interop_threads": 1, "readout": "top_contextual_real_token_mean_fresh_state", "oov_policy": "character_input_fail_empty_or_token_over_48_utf8_bytes"},
}
PINNED = {
    "fresh_helper": "3e8445f22a8f255dcf47af89ac0b8c45461ba98046549702abc7b09b6b59a624",
    "vision_helper": "68278d4d0827acf748dc1a254b18ff39bb918c878311ef6caabcb1453eedc1a6",
    "elmo_helper": "0fff41c3740a7ece2f89a4b9ba4b50d2d370606357f2ff57cce310a12e3d09ce",
    "author_encoders": "54cc1ec028a35138edf2bb06a76b484db3c7bdbc6ed49b31e8eeb538db01c48a",
    "dino_checkpoint": "bf34ad0f424b9029b593e8dc3ed553bf26e88bcba0d32bf3e62a6209cb64c85e",
    "dino_vit": "b1f998d5f49ab43666b9fc6d007c5f6540c3ead2e8645e144f74045f63ff44d7",
    "dino_utils": "962a97e1acda1c986dfd921275325e4083f9016dddadcf018f1ccf03fa600eab",
    "moco_checkpoint": "5e4a3fd7178a95837c0ad71c265ed5f477d271e753ba505b6b6a5e86829eeb7e",
    "mae_checkpoint": "aec5f0b68e5f3193a00b07bc65a37440db549c15b36b8bea242606cc40c4bc5d",
    "mae_vit": "6770ed2226d46ab8ba2b1ebc6662cc353dabf759679569acd004f6b3ac868154",
    "mae_license": "ffe3b1e0b58c1f1250ce48f84d8c37328f9a1ff59bb77b63bc97480cf2778b64",
    "timm_wheel": "c1598ef61c246e38835a6b834811dcb8d62ae56b644cd54dc3ef9565c8387875",
    "glove_zip": "6471382cdd837544bf3ac72497a38715e845897d265b2b424b4761832009c837",
    "elmo_weights": "a6b92b9ccbb13fe489faacd2221c795d4bebe84b75f8aded9d85ea464fa5c994",
    "elmo_options": "86cb4c1a7e2f25e3867d06810e33ae27fbbd511ac77ca377be5fb7405a7085b1",
}
REQUIRED = {
    "dino": ("fresh_helper", "vision_helper", "author_encoders", "dino_checkpoint", "dino_vit", "dino_utils"),
    "moco": ("fresh_helper", "vision_helper", "author_encoders", "moco_checkpoint"),
    "mae": ("fresh_helper", "vision_helper", "author_encoders", "mae_checkpoint", "mae_vit", "mae_license", "timm_wheel"),
    "glove": ("fresh_helper", "glove_zip"),
    "elmo": ("fresh_helper", "elmo_helper", "elmo_weights", "elmo_options"),
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def verify(path, expected):
    require(isinstance(expected, str) and len(expected) == 64, "A complete SHA256 is required")
    require(sha256(path) == expected, f"Changed input: {Path(path).name}")


def module(path, digest, name):
    verify(path, digest)
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def paths_for(model, path_file):
    supplied = json.loads(Path(path_file).read_text())
    result = {}
    for role in REQUIRED[model]:
        require(role in supplied, f"Missing path for {role}")
        path = Path(supplied[role]).resolve(strict=True)
        verify(path, PINNED[role])
        result[role] = path
    return result


def runtime_versions():
    values = {"python": platform.python_version()}
    for package in ("torch", "torchvision", "numpy", "Pillow", "allennlp", "h5py"):
        try:
            values[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    return values


def extraction_lock(path, expected_sha, model):
    """Fail before corpus access unless the complete future release is bound."""
    require(path is not None and expected_sha is not None, "Corpus extraction needs the reviewed release lock and its published SHA256")
    verify(path, expected_sha)
    lock = json.loads(Path(path).read_text())
    require(lock.get("schema_version") == 1 and lock.get("protocol_id") == PROTOCOL_ID, "Wrong extraction protocol")
    require(lock.get("status") == "frozen_before_model_control_extraction", "Draft protocols cannot extract corpus features")
    require(lock.get("extractor_sha256") == sha256(__file__), "Extractor differs from reviewed lock")
    require(lock.get("preprocessing_id") == PREPROCESSING_ID and lock.get("caption_slots") == 7, "Changed input recipe")
    require(set(lock.get("models", {})) == set(MODEL_CONTRACT), "Freeze all five model contracts before extraction")
    for name, contract in MODEL_CONTRACT.items():
        value = lock["models"][name]
        require(all(value.get(k) == v for k, v in contract.items()), f"Changed model contract: {name}")
        require(bool(value.get("runtime_versions")), f"Missing runtime lock: {name}")
    require(all(lock.get("files", {}).get(k) == v for k, v in PINNED.items()), "Incomplete model/source lock")
    require(set(lock.get("datasets", {})) == set(EXPECTED_ITEMS), "Freeze both complete COCO datasets")
    for split in EXPECTED_ITEMS:
        for key in ("annotation_sha256", "image_manifest_sha256"):
            require(len(lock["datasets"][split].get(key, "")) == 64, f"Missing {split} {key}")
    require(runtime_versions() == lock["models"][model]["runtime_versions"], "Runtime differs from reviewed model environment")
    return lock


def ordered_rows(document, split):
    """Original COCO order: sorted image IDs; original annotation order per image."""
    require(split in EXPECTED_ITEMS, "Unknown split")
    images = sorted(document["images"], key=lambda r: r["id"])
    ids = [r["id"] for r in images]
    require(all(type(i) is int and i >= 0 for i in ids) and len(ids) == len(set(ids)), "Invalid/duplicate image IDs")
    by_id = defaultdict(list)
    annotation_ids = set()
    for item in document["annotations"]:
        aid = item["id"]
        require(type(aid) is int and aid not in annotation_ids, "Invalid/duplicate caption ID")
        annotation_ids.add(aid)
        require(isinstance(item["caption"], str) and item["caption"].strip(), "Missing/empty human caption")
        by_id[item["image_id"]].append({"annotation_id": aid, "caption": item["caption"]})
    require(set(by_id) == set(ids), "Missing captions or unknown caption source")
    rows = []
    for image in images:
        image_id = image["id"]
        filename = f"COCO_{split}_{image_id:012d}.jpg"
        require(image["file_name"] == filename, "Filename disagrees with original COCO identity")
        captions = by_id[image_id]
        require(1 <= len(captions) <= 7, "Every item needs 1–7 original captions")
        rows.append({"image_id": image_id, "file": f"{split}/{filename}", "captions": captions})
    return rows


def verified_dataset(root, split, lock):
    root = Path(root).resolve(strict=True)
    annotation = root / "annotations" / f"captions_{split}.json"
    manifest_path = root / f"{split}-images.json"
    binding = lock["datasets"][split]
    verify(annotation, binding["annotation_sha256"])
    verify(manifest_path, binding["image_manifest_sha256"])
    rows = ordered_rows(json.loads(annotation.read_text()), split)
    require(len(rows) == EXPECTED_ITEMS[split], "A capped or incomplete corpus is forbidden")
    manifest = json.loads(manifest_path.read_text())
    require(manifest.get("schema_version") == 1 and manifest.get("split") == split, "Wrong image manifest")
    require(manifest.get("annotation_sha256") == binding["annotation_sha256"], "Image manifest has different annotation bytes")
    require(len(manifest["images"]) == len(rows), "Incomplete image manifest")
    for index, (row, image) in enumerate(zip(rows, manifest["images"])):
        require(image["row"] == index and image["image_id"] == row["image_id"] and image["file"] == row["file"], "Image manifest order differs from caption rows")
        require(len(image["sha256"]) == 64 and type(image["bytes"]) is int and image["bytes"] > 0, "Missing image byte identity")
        row.update(image_sha256=image["sha256"], image_bytes=image["bytes"])
    return rows


def configure_vision_dispatch():
    """Set process-only flags before torch imports; reject an incompatible host."""
    for key, value in VISION_ENVIRONMENT.items():
        require(os.environ.setdefault(key, value) == value, f"Wrong process environment: {key}")
    import torch
    require(torch.backends.cpu.get_cpu_capability() == "AVX2", "Actual PyTorch CPU capability must be AVX2")


def square_transform(author_source):
    """Execute the exact author's transform function without importing encoders."""
    configure_vision_dispatch()
    verify(author_source, PINNED["author_encoders"])
    from torchvision.transforms import v2
    import torch
    tree = ast.parse(Path(author_source).read_text())
    function = [x for x in tree.body if isinstance(x, ast.FunctionDef) and x.name == "dinov2_transform"]
    require(len(function) == 1, "Missing original square transform")
    namespace = {"v2": v2, "torch": torch, "Callable": object,
                 "IMAGENET_MEAN": (0.485, 0.456, 0.406), "IMAGENET_STD": (0.229, 0.224, 0.225)}
    exec(compile(ast.Module(body=function, type_ignores=[]), str(author_source), "exec"), namespace)
    return namespace["dinov2_transform"]()


def vision_encoder(name, paths):
    configure_vision_dispatch()
    vision = module(paths["vision_helper"], PINNED["vision_helper"], "_rosetta_frozen_vision")
    vision.configure_runtime()
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    require(torch.cuda.is_available(), "Locked vision runtime requires CUDA; no fallback")
    if name == "dino":
        fresh = module(paths["fresh_helper"], PINNED["fresh_helper"], "_rosetta_frozen_fresh")
        require(paths["dino_vit"].parent == paths["dino_utils"].parent, "DINO sources must share their original directory")
        model = fresh.load_dino(paths["dino_vit"].parent, paths["dino_checkpoint"], torch).float().to("cuda")
        model.feature_metadata = {"dimension": 768, "pooling": "final normalized CLS", "checkpoint_sha256": PINNED["dino_checkpoint"]}
    elif name == "moco":
        model = vision.load_moco(paths["moco_checkpoint"], "cuda")
    else:
        vendor = paths["mae_vit"].parent.parent
        require(paths["timm_wheel"].parent == vendor and paths["mae_license"].parent == vendor / "mae", "MAE vendor paths disagree")
        model = vision.load_mae(paths["mae_checkpoint"], vendor, "cuda")
    model.feature_metadata = {**model.feature_metadata, "preprocessing": PREPROCESSING_ID,
                              "torch_threads": torch.get_num_threads(), "interop_threads": torch.get_num_interop_threads(),
                              "runtime_environment": dict(VISION_ENVIRONMENT), "cpu_capability": torch.backends.cpu.get_cpu_capability(),
                              "preprocessing_change": "This new wrapper replaces the old five-photo crop; frozen backbone code/weights are unchanged."}
    transform = square_transform(paths["author_encoders"])
    def encode(image):
        # The authors' PairedDataset decodes RGB without EXIF transposition.
        tensor = transform(image.convert("RGB")).unsqueeze(0)
        require(tensor.dtype == torch.float32 and tuple(tensor.shape) == (1, 3, 224, 224), "Unexpected square224 tensor")
        return vision.extract(model, tensor, batch_size=1)[0]
    return encode, model.feature_metadata


def glove_encoder(paths, captions):
    """Load requested words once; keep the original sorted-token FP32 mean."""
    import numpy as np
    fresh = module(paths["fresh_helper"], PINNED["fresh_helper"], "_rosetta_frozen_fresh")
    requested = {t for text in captions for t in fresh.tokenize(text)}
    vectors = {}; digest = hashlib.sha256()
    with zipfile.ZipFile(paths["glove_zip"]) as archive:
        require(archive.namelist().count(fresh.GLOVE_MEMBER) == 1, "Wrong GloVe member")
        with archive.open(fresh.GLOVE_MEMBER) as stream:
            for raw in stream:
                digest.update(raw)
                word_bytes, _, values = raw.partition(b" ")
                word = word_bytes.decode("utf-8")
                if word in requested:
                    vector = np.fromstring(values.decode("ascii"), sep=" ", dtype=np.float32)
                    require(word not in vectors and vector.shape == (300,) and np.isfinite(vector).all(), "Invalid GloVe vector")
                    vectors[word] = vector
    require(digest.hexdigest() == fresh.GLOVE_MEMBER_SHA256, "Changed complete GloVe member")
    report = Counter()
    def encode(text):
        tokens = fresh.tokenize(text)
        known = [t for t in tokens if t in vectors]
        require(bool(known), "Caption has no known GloVe tokens; no item may be dropped")
        report["captions"] += 1; report["tokens"] += len(tokens)
        report["unknown_tokens"] += len(tokens) - len(known)
        report["captions_with_unknown_tokens"] += int(len(tokens) != len(known))
        mean = np.stack([vectors[t] for t in sorted(known)]).mean(axis=0, dtype=np.float32)
        return fresh.unit_rows(mean[None], np, "GloVe")[0]
    return encode, report


def text_encoder(name, paths, captions):
    if name == "glove":
        return glove_encoder(paths, captions)
    helper = module(paths["elmo_helper"], PINNED["elmo_helper"], "_rosetta_frozen_elmo")
    encoder = helper.FrozenElmo(paths["elmo_options"], paths["elmo_weights"], threads=4)
    report = {"fresh_state_per_caption": True, "character_truncation_allowed": False}
    def encode(text):
        return encoder.batch([helper.tokenize(text)])[0]
    encode.finalize = encoder.metadata
    return encode, report


def padded_text(rows, encode, dimension, completed=None):
    import numpy as np
    output = np.full((len(rows), 7, dimension), np.nan, dtype=np.float32)
    for row, item in enumerate(rows):
        for column, caption in enumerate(item["captions"]):
            value = encode(caption["caption"])
            require(value.dtype == np.float32 and value.shape == (dimension,) and np.isfinite(value).all(), "Invalid caption representation")
            output[row, column] = value
        if completed is not None:
            completed(row)
    return output


def write_json_new(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False); stream.write("\n")
        stream.flush(); os.fsync(stream.fileno())


def sync_directory(path):
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


class Attempt:
    """Every attempted inference is durable; consumers must require COMPLETE.json.

    Journal rows mean fully encoded images or fully encoded image-caption groups.
    A process killed without cleanup leaves started.json and its last durable row.
    Retry from scratch in a NEW directory; no implicit resume or silent overwrite.
    """
    def __init__(self, destination, binding):
        self.destination = Path(destination)
        self.binding = binding
        self.last_completed_row = -1

    def __enter__(self):
        self.destination.mkdir(parents=True, exist_ok=False)
        write_json_new(self.destination / "started.json", {
            **self.binding, "status": "started", "started_at_utc": datetime.now(timezone.utc).isoformat(),
            "pid": os.getpid(), "completion_marker": "COMPLETE.json",
            "retry_policy": "from_scratch_in_new_attempt_directory_only"})
        self.journal = (self.destination / "rows.jsonl").open("x")
        self.journal.flush(); os.fsync(self.journal.fileno())
        sync_directory(self.destination)
        return self

    def completed(self, row):
        require(row == self.last_completed_row + 1, "Progress rows must be contiguous")
        self.journal.write(json.dumps({"last_completed_row": row, "at_utc": datetime.now(timezone.utc).isoformat()}) + "\n")
        self.journal.flush(); os.fsync(self.journal.fileno())
        self.last_completed_row = row

    def __exit__(self, kind, error, trace):
        self.journal.close()
        if error is not None:
            write_json_new(self.destination / "FAILED.json", {
                **self.binding, "status": "failed", "failed_at_utc": datetime.now(timezone.utc).isoformat(),
                "last_completed_row": self.last_completed_row,
                "error_type": kind.__name__, "error": str(error),
                "traceback": "".join(traceback.format_exception(kind, error, trace)),
                "retry_policy": "from_scratch_in_new_attempt_directory_only"})
            sync_directory(self.destination)
        return False


def save_tensor_new(attempt, features, rows, metadata, paths, final_checks):
    """Publish COMPLETE.json last, after both outputs and all source rechecks."""
    import torch
    destination = attempt.destination
    require(attempt.last_completed_row == len(rows) - 1, "Incomplete row journal")
    partial = destination / "features.pt.partial"
    with partial.open("xb") as f:
        torch.save(torch.from_numpy(features), f)
        f.flush(); os.fsync(f.fileno())
    for role, path in paths.items():
        verify(path, PINNED[role])
    record = {**metadata, "completion_marker": "COMPLETE.json", "feature_sha256": sha256(partial),
              "shape": list(features.shape), "dtype": str(features.dtype),
              "image_ids": [r["image_id"] for r in rows], "caption_counts": [len(r["captions"]) for r in rows],
              "caption_annotation_ids": [[c["annotation_id"] for c in r["captions"]] for r in rows],
              "input_hashes": {k: PINNED[k] for k in paths}, "runtime_versions": runtime_versions()}
    metadata_partial = destination / "features.json.partial"
    write_json_new(metadata_partial, record)
    final_checks()
    for source, name in ((partial, "features.pt"), (metadata_partial, "features.json")):
        os.link(source, destination / name); source.unlink()
    sync_directory(destination)
    # A final filename alone never denotes a completed feature artifact.
    result = {**attempt.binding, "status": "complete", "shape": list(features.shape),
              "last_completed_row": attempt.last_completed_row,
              "feature_sha256": record["feature_sha256"], "metadata_sha256": sha256(destination / "features.json"),
              "completed_at_utc": datetime.now(timezone.utc).isoformat()}
    marker_partial = destination / "COMPLETE.json.partial"
    write_json_new(marker_partial, result)
    os.link(marker_partial, destination / "COMPLETE.json"); marker_partial.unlink()
    sync_directory(destination)
    return result


def extract(args):
    # No corpus path is opened before all protocol/model/runtime gates pass.
    if args.model in ("dino", "moco", "mae"):
        configure_vision_dispatch()
    lock = extraction_lock(args.lock, args.lock_sha256, args.model)
    require(not args.out.exists() and not args.out.is_symlink(), "Output already exists")
    paths = paths_for(args.model, args.inputs)
    rows = verified_dataset(args.data_root, args.split, lock)
    binding = {"protocol_id": PROTOCOL_ID, "model": args.model, "split": args.split,
               "extractor_sha256": lock["extractor_sha256"], "lock_sha256": args.lock_sha256,
               "row_count": len(rows), "dataset": lock["datasets"][args.split]}
    with Attempt(args.out, binding) as attempt:
        import numpy as np
        dimension = MODEL_CONTRACT[args.model]["dimension"]
        if args.model in ("dino", "moco", "mae"):
            from PIL import Image, ImageFile
            Image.MAX_IMAGE_PIXELS = None
            ImageFile.LOAD_TRUNCATED_IMAGES = True  # authors' dataset decoder setting
            encode, encoder_record = vision_encoder(args.model, paths)
            output = np.empty((len(rows), dimension), dtype=np.float32)
            for index, row in enumerate(rows):
                path = (args.data_root / row["file"]).resolve(strict=True)
                require(path.is_relative_to(args.data_root.resolve()), "Image escapes corpus root")
                require(path.stat().st_size == row["image_bytes"], "Changed image byte count")
                raw = path.read_bytes()
                require(len(raw) == row["image_bytes"] and hashlib.sha256(raw).hexdigest() == row["image_sha256"], "Changed image bytes")
                with Image.open(io.BytesIO(raw)) as image:
                    value = encode(image)
                require(value.dtype == np.float32 and value.shape == (dimension,) and np.isfinite(value).all(), "Invalid image representation")
                output[index] = value
                attempt.completed(index)
        else:
            captions = (c["caption"] for row in rows for c in row["captions"])
            encode, encoder_record = text_encoder(args.model, paths, captions)
            output = padded_text(rows, encode, dimension, attempt.completed)
        if hasattr(encode, "finalize"):
            encoder_record = encode.finalize()
        def final_checks():
            verify(args.lock, args.lock_sha256)
            verify(__file__, lock["extractor_sha256"])
            verified_dataset(args.data_root, args.split, lock)
        metadata = {**binding, "created_at_utc": datetime.now(timezone.utc).isoformat(),
                    "preprocessing_id": PREPROCESSING_ID, "model_contract": MODEL_CONTRACT[args.model], "encoder_record": dict(encoder_record),
                    "caption_item_reduction": "Saved per-caption unit vectors with missing slots NaN; authors load_rows normalizes captions, averages non-NaN slots, then normalizes, in stored FP32 precision.",
                    "difference_from_reference": "Added encoders use locked FP32; the released DINOv2/MPNet reference retains stored bfloat16.", "no_training": True}
        return save_tensor_new(attempt, output, rows, metadata, paths, final_checks)


def benchmark(args):
    import numpy as np
    paths = paths_for(args.model, args.inputs)
    report = {"status": "artificial_inputs_only", "model": args.model, "extractor_sha256": sha256(__file__), "runtime_versions": runtime_versions(),
              "corpus_accessed": False, "private_photographs_accessed": False, "batch_size": 1, "iterations_per_fixture": 5}
    if args.model in ("dino", "moco", "mae"):
        from PIL import Image
        encode, info = vision_encoder(args.model, paths)
        fixtures = []
        for i, (w, h) in enumerate(((224, 224), (351, 229), (231, 349))):
            y, x = np.indices((h, w), dtype=np.int32)
            pixels = np.stack(((7*x+3*y+i*17)%256, (x^y)%256, (x*y+i*47)%256), -1).astype(np.uint8)
            fixtures.append(Image.fromarray(pixels))
        transform = square_transform(paths["author_encoders"])
        report["synthetic_tensor_sha256"] = [hashlib.sha256(transform(im.convert("RGB")).numpy().tobytes()).hexdigest() for im in fixtures]
        report["preprocessing_id"] = PREPROCESSING_ID
        report["encoder_record"] = info
        lengths = [f"{im.width}x{im.height}" for im in fixtures]
    else:
        words = "quiet letters move beneath several ordinary lines while readers compare them".split()
        fixtures = [" ".join(words[i % len(words)] for i in range(n)) for n in (5, 10, 20, 50)]
        encode, info = text_encoder(args.model, paths, fixtures)
        report["encoder_record"] = dict(info)
        lengths = [5, 10, 20, 50]
    baseline = [encode(value) for value in fixtures]  # warm each input before timing
    checks = []; throughput = []
    for size, fixture, reference in zip(lengths, fixtures, baseline):
        started = time.perf_counter(); outputs = [encode(fixture) for _ in range(5)]; elapsed = time.perf_counter()-started
        checks.append(all(np.array_equal(reference, value) for value in outputs))
        throughput.append({"fixture_size": size, "repetitions": 5, "seconds": elapsed, "seconds_per_item": elapsed/5})
    reverse = [encode(fixture) for fixture in reversed(fixtures)][::-1]
    report["synthetic_feature_sha256"] = hashlib.sha256(np.stack(baseline).tobytes()).hexdigest()
    report.update(repeat_bitwise_equal=all(checks), reorder_bitwise_equal=all(np.array_equal(a,b) for a,b in zip(baseline,reverse)),
                  finite_unit_vectors=all(np.isfinite(v).all() and np.isclose(np.linalg.norm(v.astype(np.float64)),1,atol=1e-5,rtol=0) for v in baseline), throughput=throughput)
    require(report["repeat_bitwise_equal"] and report["reorder_bitwise_equal"] and report["finite_unit_vectors"], "Synthetic repeat/order/shape check failed")
    for role, path in paths.items(): verify(path, PINNED[role])
    if hasattr(encode, "finalize"):
        report["encoder_record"] = encode.finalize()
    report["input_hashes"] = {k: PINNED[k] for k in paths}
    with args.out.open("x") as f:
        json.dump(report, f, indent=2, allow_nan=False); f.write("\n")
    return report


def self_test():
    import copy
    import tempfile
    import numpy as np
    doc = {"images": [{"id": n, "file_name": f"COCO_train2014_{n:012d}.jpg"} for n in (2,1)],
           "annotations": [{"id": 3, "image_id": 2, "caption": "second image"}, {"id": 4, "image_id": 1, "caption": "first caption"}, {"id": 5, "image_id": 1, "caption": "second caption"}]}
    rows = ordered_rows(doc, "train2014")
    require([r["image_id"] for r in rows] == [1,2] and [x["annotation_id"] for x in rows[0]["captions"]] == [4,5], "Ordering test")
    values = padded_text(rows, lambda text: np.ones(3,dtype=np.float32), 3)
    require(values.shape == (2,7,3) and np.isnan(values[0,2:]).all() and np.isnan(values[1,1:]).all() and np.isfinite(values[0,:2]).all(), "Caption padding test")
    rejected = 0
    for change in ("duplicate", "missing", "unknown", "empty", "too_many"):
        bad = copy.deepcopy(doc)
        if change == "duplicate": bad["annotations"].append(bad["annotations"][0])
        if change == "missing": bad["annotations"] = [x for x in bad["annotations"] if x["image_id"] != 2]
        if change == "unknown": bad["annotations"][0]["image_id"] = 99
        if change == "empty": bad["annotations"][0]["caption"] = " "
        if change == "too_many": bad["annotations"].extend({"id": i, "image_id": 2, "caption": "extra"} for i in range(100,108))
        try: ordered_rows(bad,"train2014")
        except ValueError: rejected += 1
    require(rejected == 5, "Invalid annotation guards failed")
    try: extraction_lock(None,None,"dino")
    except ValueError: pass
    else: raise AssertionError("Missing release lock must stop corpus access")
    # Artificial failure/success publication checks; no model or corpus inputs.
    with tempfile.TemporaryDirectory(prefix="rosetta-artificial-attempt-") as temporary:
        root = Path(temporary)
        for scenario in ("inference_failure", "recheck_failure", "complete"):
            destination = root / scenario
            try:
                with Attempt(destination, {"fixture": True}) as attempt:
                    attempt.completed(0)
                    if scenario == "inference_failure":
                        raise RuntimeError("artificial inference interruption")
                    attempt.completed(1)
                    def checks():
                        if scenario == "recheck_failure":
                            raise RuntimeError("artificial source recheck failure")
                    save_tensor_new(attempt, values, rows, {}, {}, checks)
            except RuntimeError:
                require(scenario != "complete", "Unexpected publication failure")
            require((destination / "started.json").is_file(), "Missing attempt receipt")
            if scenario == "complete":
                marker = json.loads((destination / "COMPLETE.json").read_text())
                verify(destination / "features.pt", marker["feature_sha256"])
                verify(destination / "features.json", marker["metadata_sha256"])
                require(not (destination / "FAILED.json").exists(), "Success marked failed")
            else:
                failure = json.loads((destination / "FAILED.json").read_text())
                require(failure["last_completed_row"] == (0 if scenario == "inference_failure" else 1), "Wrong durable failure position")
                require(not (destination / "COMPLETE.json").exists(), "Failure published success marker")
                require(not (destination / "features.pt").exists(), "Failure published final tensor")
        try:
            with Attempt(root / "complete", {}): pass
        except FileExistsError: pass
        else: raise AssertionError("Attempts must never overwrite")
    return {"attempt_failure_success_publication": True,"status":"passed","ordering":True,"padding":True,"invalid_annotation_cases_rejected":rejected,"missing_lock_rejected":True,"real_data_accessed":False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command",required=True)
    sub.add_parser("self-test")
    for command in ("benchmark", "extract"):
        p = sub.add_parser(command)
        p.add_argument("--model", choices=tuple(MODEL_CONTRACT), required=True)
        p.add_argument("--inputs", type=Path, required=True)
        p.add_argument("--out", type=Path, required=True)
        if command == "extract":
            p.add_argument("--lock", type=Path, required=True)
            p.add_argument("--lock-sha256", required=True)
            p.add_argument("--data-root", type=Path, required=True, help="Verified data/coco2014 directory")
            p.add_argument("--split", choices=tuple(EXPECTED_ITEMS), required=True)
    args = parser.parse_args()
    if hasattr(args,"out"):
        require(not args.out.exists() and not args.out.is_symlink(), "Output already exists; refuse replacement")
    result = self_test() if args.command == "self-test" else benchmark(args) if args.command == "benchmark" else extract(args)
    print(json.dumps(result,indent=2,allow_nan=False))

if __name__ == "__main__":
    main()
