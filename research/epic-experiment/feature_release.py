#!/usr/bin/env python3
"""Export/restore full frozen EPIC features without distributing narration text.

Export removes the narration array and nulls per-row token/OOV lists. Restore
obtains that text from the separately regenerated, byte-pinned EPIC manifest.
Both commands require all 9,392 selected rows. Export tests exact reconstruction
before publishing a new local directory; restore accepts only the original NPZ
and JSON hashes. Neither command runs inference, alignment, uploads or deploys.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import sys
import tempfile
import types
import zlib

import numpy as np


MANIFEST_SHA256 = "d1a34b2ad47a1456a011c6e4f2d6308bae34ebf6aed0c6fb335794ac3dc648be"
EXTRACTOR_SHA256 = "471ede5f611b17ab63252957c0f9785ff2afede22a656f8452fdadff4217f203"
EXPECTED_ROWS, IMAGE_DIM, TEXT_DIM = 9392, 768, 300
METADATA_KEYS = ("uid", "participant_id", "video_id", "narration", "split", "noun_class",
                 "verb_class", "start_timestamp", "stop_timestamp")
ARRAY_KEYS = METADATA_KEYS + ("image", "text", "eligible", "frame_paths", "text_token_count", "text_known_token_count")
FEATURE_PAYLOAD = "features-without-narration.npz"
METADATA_PAYLOAD = "feature-metadata-template.json"
RECEIPT = "feature-release.json"
STATUS = "full_frozen_feature_export_roundtrip_verified"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    def reject_constant(value):
        raise ValueError(f"Nonfinite JSON constant: {value}")
    def unique_fields(pairs):
        values = {}
        for key, value in pairs:
            require(key not in values, "Duplicate JSON key: " + key)
            values[key] = value
        return values
    return json.loads(Path(path).read_text(encoding="utf-8"), parse_constant=reject_constant,
                      object_pairs_hook=unique_fields)


def write_json(path, value):
    # This deliberately matches the frozen extractor's original JSON writer.
    Path(path).write_bytes((json.dumps(value, indent=2, ensure_ascii=True, allow_nan=False) + "\n").encode("utf-8"))


def original_inputs(manifest, extractor):
    require(sha256(manifest) == MANIFEST_SHA256, "Not the original frozen full selected-row manifest")
    require(sha256(extractor) == EXTRACTOR_SHA256, "Frozen extractor/tokenizer source differs")
    rows = read_json(manifest)["rows"]
    require(len(rows) == EXPECTED_ROWS and len({str(row["uid"]) for row in rows}) == EXPECTED_ROWS,
            "Full selected manifest has missing/duplicate rows")
    # Verified source only; executing it defines the frozen tokenizer without
    # invoking main(), running inference, or writing an import cache beside it.
    module = types.ModuleType("frozen_feature_release_tokenizer")
    module.__file__ = str(extractor)
    exec(compile(Path(extractor).read_bytes(), str(extractor), "exec"), module.__dict__)
    return rows, module.tokenize


def load_arrays(path):
    with np.load(path, allow_pickle=False) as archive:
        order = list(archive.files)
        require(len(order) == len(set(order)), "Duplicate NPZ members")
        return {key: np.array(archive[key], copy=True) for key in order}


def array_schema(arrays):
    return {key: {"dtype": value.dtype.str, "shape": list(value.shape)} for key, value in arrays.items()}


def validate_full(arrays, metadata, rows, tokenize):
    require(set(arrays) == set(ARRAY_KEYS), "Full feature array set differs from the frozen extractor")
    for key in METADATA_KEYS:
        value = arrays[key]
        require(value.dtype.kind == "U" and value.shape == (EXPECTED_ROWS,), "Invalid metadata array: " + key)
        require(np.array_equal(value, np.asarray([str(row[key]) for row in rows])), "Feature/manifest identity differs: " + key)
    require(arrays["frame_paths"].dtype.kind == "U" and arrays["frame_paths"].shape == (EXPECTED_ROWS, 4), "Frame-path array differs")
    require(np.array_equal(arrays["frame_paths"], np.asarray([row["frame_paths"] for row in rows])), "Frame paths differ from frozen manifest")
    require(arrays["eligible"].dtype == np.dtype(bool) and arrays["eligible"].shape == (EXPECTED_ROWS,) and arrays["eligible"].all(),
            "Eligibility must be bool[N] and all true; partial/ineligible exports are forbidden")
    for key, dimension in (("image", IMAGE_DIM), ("text", TEXT_DIM)):
        value = arrays[key]
        require(value.dtype == np.dtype(np.float32) and value.shape == (EXPECTED_ROWS, dimension), "Invalid feature dtype/shape: " + key)
        require(np.isfinite(value).all() and (np.linalg.norm(value, axis=1) > 1e-10).all(), "Nonfinite or zero features: " + key)
    require(metadata["script_sha256"] == EXTRACTOR_SHA256 and metadata["manifest_sha256"] == MANIFEST_SHA256,
            "Feature sidecar source/manifest hashes differ")
    selection = metadata["selection"]
    require(selection["is_final"] is True and selection["allow_partial"] is False, "Feature extraction is partial/nonfinal")
    require(selection["selected_rows"] == selection["output_rows"] == EXPECTED_ROWS and selection["missing_rows"] == [],
            "Feature extraction omitted selected rows")
    require(metadata["eligibility"]["eligible"] == EXPECTED_ROWS and metadata["eligibility"]["ineligible"] == 0,
            "Feature eligibility summary is incomplete")
    language = metadata["language"]
    coverage = language["row_coverage"]
    require(len(coverage) == EXPECTED_ROWS, "Token coverage must cover every selected row")
    require(all(type(count) is int and count > 0 for count in language["oov_counts"].values()), "Invalid aggregate OOV counts")
    oov_vocabulary = set(language["oov_counts"])
    observed_oov, totals, known_counts = Counter(), [], []
    for row, recorded in zip(rows, coverage):
        require(set(recorded) == {"uid", "tokens", "known_tokens", "total_tokens", "oov_tokens"},
                "Unexpected per-row language metadata; refusing an unreviewed text field")
        tokens = tokenize(row["narration"])
        unknown = [token for token in tokens if token in oov_vocabulary]
        require(str(recorded["uid"]) == str(row["uid"]), "Token coverage UID order differs")
        require(recorded["tokens"] == tokens and recorded["oov_tokens"] == unknown, "Per-row token/OOV reconstruction differs")
        require(recorded["total_tokens"] == len(tokens) and recorded["known_tokens"] == len(tokens) - len(unknown), "Per-row coverage counts differ")
        observed_oov.update(unknown)
        totals.append(len(tokens))
        known_counts.append(len(tokens) - len(unknown))
    require(dict(observed_oov) == language["oov_counts"], "Aggregate OOV counts differ from reconstructed rows")
    require(language["total_tokens"] == sum(totals) and language["known_tokens"] == sum(known_counts), "Aggregate token coverage differs")
    for key, values in (("text_token_count", totals), ("text_known_token_count", known_counts)):
        require(arrays[key].dtype == np.dtype(np.int64) and arrays[key].shape == (EXPECTED_ROWS,) and np.array_equal(arrays[key], values),
                "Token-count array differs: " + key)


def writer_versions():
    return {"python": platform.python_version(), "python_implementation": platform.python_implementation(),
            "numpy": np.__version__, "zlib_compile": zlib.ZLIB_VERSION, "zlib_runtime": zlib.ZLIB_RUNTIME_VERSION,
            "byteorder": sys.byteorder, "npz_writer": "numpy.savez_compressed, default ZIP_DEFLATED compression",
            "json_writer": "json.dumps(indent=2, ensure_ascii=True, allow_nan=False) plus newline, UTF-8"}


def snapshot(paths):
    return {str(path.resolve()): sha256(path) for path in paths}


def new_output(path):
    require(not path.exists() and not path.is_symlink(), "Output directory already exists; choose a new directory")
    path.parent.mkdir(parents=True, exist_ok=True)


def checked_name(value, suffix):
    require(isinstance(value, str) and Path(value).name == value and value.endswith(suffix), "Invalid original output filename")
    return value


def restore_into(manifest, bundle, extractor, output):
    """Restore into an existing private staging directory, never a public path."""
    rows, tokenize = original_inputs(manifest, extractor)
    receipt = read_json(bundle / RECEIPT)
    require(receipt["schema_version"] == 1 and receipt["status"] == STATUS, "Export receipt is incomplete/unverified")
    require(receipt["manifest_sha256"] == MANIFEST_SHA256 and receipt["extractor_sha256"] == EXTRACTOR_SHA256 and
            receipt["rows"] == EXPECTED_ROWS, "Export receipt is not for this frozen full sample")
    require(receipt["roundtrip"]["npz_bytes_identical"] is True and receipt["roundtrip"]["metadata_bytes_identical"] is True,
            "Export lacks successful lossless reconstruction checks")
    require(set(receipt["payloads"]) == {FEATURE_PAYLOAD, METADATA_PAYLOAD}, "Export payload set differs")
    for name, expected in receipt["payloads"].items():
        path = bundle / name
        require(path.is_file() and path.resolve().parent == bundle.resolve(), "Missing/external export payload")
        require(path.stat().st_size == expected["bytes"] and sha256(path) == expected["sha256"], "Export payload hash/size differs: " + name)
    arrays = load_arrays(bundle / FEATURE_PAYLOAD)
    require(set(arrays) == set(ARRAY_KEYS) - {"narration"}, "Public feature payload contains unexpected/missing arrays")
    original = receipt["original"]
    feature_name = checked_name(original["feature_filename"], ".npz")
    metadata_name = checked_name(original["metadata_filename"], ".json")
    require(Path(feature_name).with_suffix(".json").name == metadata_name, "Original feature/sidecar names disagree")
    order, schema = receipt["array_order"], receipt["array_schema"]
    require(len(order) == len(ARRAY_KEYS) and set(order) == set(ARRAY_KEYS) and set(schema) == set(ARRAY_KEYS), "Original array order/schema differs")
    require(list(arrays) == [key for key in order if key != "narration"], "Export array order differs from receipt")
    for key, value in arrays.items():
        require(schema[key] == {"dtype": value.dtype.str, "shape": list(value.shape)}, "Export array schema differs: " + key)
    dtype = np.dtype(schema["narration"]["dtype"])
    longest = max(map(len, (str(row["narration"]) for row in rows)))
    require(dtype.kind == "U" and dtype.itemsize == longest * 4 and schema["narration"]["shape"] == [EXPECTED_ROWS],
            "Original narration dtype/shape differs from the frozen extractor")
    arrays["narration"] = np.asarray([str(row["narration"]) for row in rows], dtype=dtype)
    arrays = {key: arrays[key] for key in order}
    metadata = read_json(bundle / METADATA_PAYLOAD)
    require(metadata["feature_file_sha256"] == original["feature_sha256"] and metadata["feature_file"] == feature_name,
            "Template and original feature hash/name disagree")
    coverage = metadata["language"]["row_coverage"]
    require(len(coverage) == EXPECTED_ROWS, "Template token coverage is incomplete")
    oov_vocabulary = set(metadata["language"]["oov_counts"])
    for row, recorded in zip(rows, coverage):
        require(str(row["uid"]) == str(recorded["uid"]), "Template UID order differs from manifest")
        require(recorded["tokens"] is None and recorded["oov_tokens"] is None, "Public metadata token fields must be null")
        recorded["tokens"] = tokenize(row["narration"])
        recorded["oov_tokens"] = [token for token in recorded["tokens"] if token in oov_vocabulary]
    validate_full(arrays, metadata, rows, tokenize)
    feature_path, metadata_path = output / feature_name, output / metadata_name
    np.savez_compressed(feature_path, **arrays)
    write_json(metadata_path, metadata)
    require(feature_path.stat().st_size == original["feature_bytes"] and sha256(feature_path) == original["feature_sha256"],
            "Restored NPZ bytes differ from original; use the recorded roundtrip writer versions (including NumPy/zlib)")
    require(metadata_path.stat().st_size == original["metadata_bytes"] and sha256(metadata_path) == original["metadata_sha256"],
            "Restored JSON bytes differ from original; serializer/order must match the recorded writer")
    return receipt


def export_bundle(manifest, features, extractor, output):
    manifest, features, extractor, output = map(Path, (manifest, features, extractor, output))
    new_output(output)
    sidecar = features.with_suffix(".json")
    originals = (manifest, features, sidecar, extractor)
    before = snapshot(originals)
    rows, tokenize = original_inputs(manifest, extractor)
    arrays, metadata = load_arrays(features), read_json(sidecar)
    require(metadata["feature_file_sha256"] == sha256(features) and metadata["feature_file"] == features.name,
            "Original feature sidecar hash/name differs")
    validate_full(arrays, metadata, rows, tokenize)
    template = read_json(sidecar)
    for row in template["language"]["row_coverage"]:
        row["tokens"] = None
        row["oov_tokens"] = None
    with tempfile.TemporaryDirectory(prefix=".epic-feature-export-", dir=output.parent) as temporary:
        stage = Path(temporary) / "bundle"
        stage.mkdir()
        np.savez_compressed(stage / FEATURE_PAYLOAD, **{key: value for key, value in arrays.items() if key != "narration"})
        write_json(stage / METADATA_PAYLOAD, template)
        receipt = {
            "schema_version": 1, "status": STATUS, "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "scope": "Lossless release preparation for full frozen features only; no benchmark alignment or result verification claim",
            "manifest_sha256": MANIFEST_SHA256, "extractor_sha256": EXTRACTOR_SHA256,
            "feature_release_script_sha256": sha256(__file__), "rows": EXPECTED_ROWS,
            "original": {"feature_filename": features.name, "feature_sha256": sha256(features), "feature_bytes": features.stat().st_size,
                         "metadata_filename": sidecar.name, "metadata_sha256": sha256(sidecar), "metadata_bytes": sidecar.stat().st_size},
            "array_order": list(arrays), "array_schema": array_schema(arrays),
            "payloads": {name: {"sha256": sha256(stage / name), "bytes": (stage / name).stat().st_size}
                         for name in (FEATURE_PAYLOAD, METADATA_PAYLOAD)},
            "roundtrip_writer": writer_versions(),
            "roundtrip": {"npz_bytes_identical": True, "metadata_bytes_identical": True},
            "withheld": ["narration NPZ array", "per-row tokens and oov_tokens in language.row_coverage"],
            "retained_text": "Aggregate OOV vocabulary/counts and technical metadata; no per-row token sequences",
            "restoration": "Regenerate the exact pinned full selected_rows.json from original EPIC annotations, then run feature_release.py restore. Original NPZ and sidecar hashes must match before using the unchanged frozen verifier.",
            "limits": ["Roundtrip identity is tested with the recorded writer stack; other NumPy/zlib/JSON writers may produce different bytes and will be rejected.",
                       "This package contains derived features, not original images, pretrained weights, or independent proof of model training histories."],
        }
        # The positive status lives only inside this private staging directory
        # until restore_into proves both claimed hashes against original bytes.
        write_json(stage / RECEIPT, receipt)
        restored = Path(temporary) / "roundtrip"
        restored.mkdir()
        restore_into(manifest, stage, extractor, restored)
        require(snapshot(originals) == before, "Original inputs changed during export; release refused")
        require(not output.exists() and not output.is_symlink(), "Output directory appeared during export")
        stage.rename(output)
    return receipt


def restore_bundle(manifest, bundle, extractor, output):
    manifest, bundle, extractor, output = map(Path, (manifest, bundle, extractor, output))
    new_output(output)
    inputs = (manifest, extractor, bundle / RECEIPT, bundle / FEATURE_PAYLOAD, bundle / METADATA_PAYLOAD)
    before = snapshot(inputs)
    with tempfile.TemporaryDirectory(prefix=".epic-feature-restore-", dir=output.parent) as temporary:
        stage = Path(temporary) / "restored"
        stage.mkdir()
        receipt = restore_into(manifest, bundle, extractor, stage)
        require(snapshot(inputs) == before, "Export/manifest/source inputs changed during restore")
        require(not output.exists() and not output.is_symlink(), "Output directory appeared during restore")
        stage.rename(output)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("export", "restore"):
        sub = commands.add_parser(command)
        sub.add_argument("--manifest", type=Path, required=True, help="Original full frozen selected_rows.json")
        sub.add_argument("--extractor", type=Path, default=Path(__file__).with_name("extract_features.py"))
        sub.add_argument("--output", type=Path, required=True, help="New output directory; must not already exist")
        sub.add_argument("--features" if command == "export" else "--bundle", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "export":
        receipt = export_bundle(args.manifest, args.features, args.extractor, args.output)
    else:
        receipt = restore_bundle(args.manifest, args.bundle, args.extractor, args.output)
    print(json.dumps({"command": args.command, "status": "exact_original_hashes_verified", "rows": receipt["rows"],
                      "directory": str(args.output), "benchmark_result_verified": False}), flush=True)


if __name__ == "__main__":
    main()
