#!/usr/bin/env python3
"""Fixed competence analysis for the 48-photograph fresh-object pilot.

This fits supervised ridge models only. It does not run Unpaired Rosetta or
test whether independently trained representation spaces align without pairs.
Retain stderr as the run log: failures emit a structured record there and do
not publish a partial output directory.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import sys
import tempfile
import unicodedata

import numpy as np


SCHEMA_VERSION = 1
PROTOCOL_ID = "fresh-objects-v1"
CLASSES = ("mug", "bowl", "plate", "bottle", "spoon", "fork", "book", "shoe")
FIT_ROUNDS = ("r01", "r02", "r03", "r04")
DEVELOPMENT_ROUNDS = ("r05", "r06")
ROUNDS = FIT_ROUNDS + DEVELOPMENT_ROUNDS
N_ROWS = 48
IMAGE_DIM = 768
TEXT_DIM = 300
RIDGE_ALPHA = 1.0
SHUFFLE_SEEDS = (0, 1, 2)
RELEASE_DIR = Path(__file__).resolve().parent
RELEASE_FILES = ("protocol.json", "intake.py", "extract.py", "analyze.py")
SCOPE = (
    "Supervised competence and feasibility on eight new development objects, "
    "photographed twice each. This is not an unpaired alignment result. "
    "No confidence interval, significance test, or population claim is made. "
    "Verified file hashes and model flags establish provenance consistency; "
    "they do not prove that pretraining corpora contain no related concepts."
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"Expected a JSON object: {path}")
    return value


def check_hash(value: object, label: str) -> None:
    require(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None,
            f"Invalid SHA-256 for {label}")


def validate_rows(sealed: dict, protocol: dict, protocol_hash: str) -> list[dict]:
    require(sealed.get("schema_version") == SCHEMA_VERSION, "Unsupported sealed schema")
    require(sealed.get("protocol_id") == PROTOCOL_ID, "Unexpected sealed protocol_id")
    require(protocol.get("protocol_id") == PROTOCOL_ID, "Unexpected protocol_id")
    require(sealed.get("protocol_sha256") == protocol_hash, "Sealed protocol hash mismatch")
    require(protocol.get("classes") == list(CLASSES), "Protocol class order differs from analysis")
    require(protocol.get("analysis", {}).get("ridge_alpha") == RIDGE_ALPHA,
            "Protocol ridge alpha differs from analysis")
    collection = protocol.get("collection", {})
    require(collection.get("fit_rounds") == list(FIT_ROUNDS)
            and collection.get("development_rounds") == list(DEVELOPMENT_ROUNDS),
            "Protocol rounds differ from analysis")
    rows = sealed.get("rows")
    require(isinstance(rows, list) and len(rows) == N_ROWS, "Exactly 48 sealed rows are required")
    expected_ids = {f"{round_id}-{label}" for round_id in ROUNDS for label in CLASSES}
    ids = []
    image_hashes = []
    for row in rows:
        require(isinstance(row, dict), "Every sealed row must be an object")
        for key in ("sample_id", "round_id", "split", "class_name", "instance_id",
                    "image_file", "captured_at_utc", "caption", "captioned_at_utc"):
            require(isinstance(row.get(key), str) and bool(row[key].strip()),
                    f"Missing or empty sealed row field: {key}")
        round_id, label = row["round_id"], row["class_name"]
        require(round_id in ROUNDS and label in CLASSES, "Unexpected round or class")
        sample_id = f"{round_id}-{label}"
        require(row["sample_id"] == sample_id, "Malformed sample_id")
        is_fit = round_id in FIT_ROUNDS
        require(row["split"] == ("fit" if is_fit else "development"),
                f"Wrong split for {sample_id}")
        require(row["instance_id"] == f"{label}-{'A' if is_fit else 'B'}",
                f"Wrong physical instance for {sample_id}")
        check_hash(row.get("image_sha256"), f"image {sample_id}")
        size = row.get("image_size_bytes")
        require(type(size) is int and size > 0, f"Invalid image size for {sample_id}")
        ids.append(sample_id)
        image_hashes.append(row["image_sha256"])
    require(len(set(ids)) == N_ROWS and set(ids) == expected_ids,
            "Sealed IDs must contain every class exactly once in each round")
    require(len(set(image_hashes)) == N_ROWS, "Repeated image bytes in sealed photographs")
    return rows


def load_inputs(features: Path, sealed_path: Path, protocol_path: Path) -> dict:
    sidecar_path = features.with_suffix(".json")
    paths = {"features": features, "feature_metadata": sidecar_path,
             "sealed": sealed_path, "protocol": protocol_path,
             "release_lock": RELEASE_DIR / "release-lock.json"}
    paths.update({f"release_{name}": RELEASE_DIR / name for name in RELEASE_FILES})
    hashes = {key: sha256(path) for key, path in paths.items()}
    lock = read_json(paths["release_lock"])
    require(lock.get("schema_version") == SCHEMA_VERSION, "Unsupported release lock schema")
    require(isinstance(lock.get("files"), dict) and set(lock["files"]) == set(RELEASE_FILES),
            "Release lock must name all four frozen files")
    for name in RELEASE_FILES:
        require(lock["files"][name] == hashes[f"release_{name}"],
                f"Release file hash mismatch: {name}")
    require(hashes["protocol"] == lock["files"]["protocol.json"], "Protocol differs from release lock")
    sealed = read_json(sealed_path)
    protocol = read_json(protocol_path)
    rows = validate_rows(sealed, protocol, hashes["protocol"])
    require(sealed.get("release_lock_sha256") == hashes["release_lock"],
            "Sealed release lock hash mismatch")
    require(sealed.get("intake_sha256") == lock["files"]["intake.py"],
            "Sealed intake source hash mismatch")
    metadata = read_json(sidecar_path)
    require(metadata.get("schema_version") == SCHEMA_VERSION
            and metadata.get("protocol_id") == PROTOCOL_ID and metadata.get("status") == "complete",
            "Feature metadata must describe a complete pilot extraction")
    require(metadata.get("sample_count") == N_ROWS
            and metadata.get("split_counts") == {"fit": 32, "development": 16},
            "Feature metadata must describe all 48 rows")
    for key, actual in (("protocol_sha256", hashes["protocol"]),
                        ("sealed_sha256", hashes["sealed"]),
                        ("feature_sha256", hashes["features"]),
                        ("release_lock_sha256", hashes["release_lock"])):
        require(metadata.get(key) == actual, f"Feature metadata {key} mismatch")
    verified = metadata.get("models_verified")
    require(isinstance(verified, dict) and verified.get("image") is True
            and verified.get("text") is True, "Both model verification flags must be true")
    require(metadata.get("extractor_sha256") == lock["files"]["extract.py"],
            "Feature extractor source hash mismatch")
    models = protocol.get("models", {})
    image_model, text_model = models.get("image", {}), models.get("text", {})
    expected_model_hashes = {
        "dino_checkpoint_sha256": image_model.get("checkpoint_sha256"),
        "glove_zip_sha256": text_model.get("archive_sha256"),
        "glove_member_sha256": text_model.get("member_sha256"),
    }
    for key, value in expected_model_hashes.items():
        check_hash(value, key)
    require(metadata.get("model_hashes") == expected_model_hashes,
            "Feature model hashes differ from the protocol")
    require(isinstance(image_model.get("source_sha256"), dict)
            and metadata.get("source_hashes") == image_model["source_sha256"]
            and metadata.get("dino_source_commit") == image_model.get("source_commit"),
            "Feature DINO source differs from the protocol")
    with np.load(features, allow_pickle=False) as archive:
        require(set(archive.files) == {"image", "text", "sample_ids"},
                "Feature archive must contain exactly image, text, and sample_ids")
        arrays = {key: archive[key] for key in archive.files}
    sample_ids = arrays["sample_ids"]
    require(sample_ids.dtype.kind == "U" and sample_ids.shape == (N_ROWS,),
            "sample_ids must be Unicode[48]")
    ids = sample_ids.tolist()
    require(metadata.get("sample_ids") == ids, "Feature metadata sample_ids differ from archive")
    row_ids = [row["sample_id"] for row in rows]
    require(len(set(ids)) == N_ROWS and set(ids) == set(row_ids),
            "Feature IDs do not match the complete sealed ID set")
    positions = {sample_id: i for i, sample_id in enumerate(ids)}
    order = np.asarray([positions[sample_id] for sample_id in row_ids], dtype=np.int64)
    aligned = {}
    for name, dimension in (("image", IMAGE_DIM), ("text", TEXT_DIM)):
        value = arrays[name]
        require(value.dtype == np.float32 and value.shape == (N_ROWS, dimension),
                f"{name} must be float32[48,{dimension}]")
        require(bool(np.isfinite(value).all()), f"Nonfinite {name} features")
        norms = np.linalg.norm(value.astype(np.float64), axis=1)
        require(bool((norms > 0).all()), f"Zero-norm {name} features")
        require(bool(np.allclose(norms, 1.0, rtol=0.0, atol=1e-4)),
                f"{name} features must already be L2 normalized")
        aligned[name] = value[order].astype(np.float64)
    require(all(sha256(path) == hashes[key] for key, path in paths.items()),
            "Inputs changed while being read")
    return {"rows": rows, **aligned, "hashes": hashes, "paths": paths,
            "feature_metadata": metadata, "protocol": protocol,
            "feature_row_order_changed": ids != row_ids}


def l2_rows(values: np.ndarray, label: str) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    require(values.ndim == 2 and bool(np.isfinite(values).all()),
            f"Invalid finite matrix for {label}")
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    require(bool((norms > 0).all()) and bool(np.isfinite(norms).all()),
            f"Zero or nonfinite norm after {label}")
    return values / norms


def center_from_fit(values: np.ndarray, fit_indices: np.ndarray,
                    label: str) -> tuple[np.ndarray, np.ndarray]:
    mean = np.asarray(values, dtype=np.float64)[fit_indices].mean(axis=0)
    return l2_rows(np.asarray(values, dtype=np.float64) - mean, f"{label} fit centering"), mean


def dual_ridge(x: np.ndarray, y: np.ndarray, alpha: float = RIDGE_ALPHA) -> np.ndarray:
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    require(x.ndim == y.ndim == 2 and x.shape[0] == y.shape[0] and x.shape[0] > 0,
            "Ridge expects two matrices with the same nonzero row count")
    require(bool(np.isfinite(x).all() and np.isfinite(y).all()) and alpha > 0,
            "Ridge inputs must be finite and alpha positive")
    return x.T @ np.linalg.solve(x @ x.T + alpha * np.eye(x.shape[0]), y)


def tie_expected_top1(scores: np.ndarray, query_classes: np.ndarray,
                      candidate_classes: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Exact expected class hit under uniform ordering of exactly equal maxima."""
    scores = np.asarray(scores, dtype=np.float64)
    query_classes = np.asarray(query_classes)
    candidate_classes = np.asarray(candidate_classes)
    require(scores.ndim == 2 and scores.shape == (len(query_classes), len(candidate_classes))
            and scores.shape[1] > 0 and bool(np.isfinite(scores).all()),
            "Scores and class labels have incompatible shapes or nonfinite values")
    tied = scores == scores.max(axis=1, keepdims=True)
    counts = tied.sum(axis=1)
    correct = (tied & (query_classes[:, None] == candidate_classes[None, :])).sum(axis=1)
    return correct.astype(np.float64) / counts, counts, correct


def shuffle_within_round(rows: list[dict], fit_indices: np.ndarray, seed: int) -> np.ndarray:
    """Return global text-row indices aligned to each fit image row.

    Each round gets an independent ordinary permutation. Fixed points are kept.
    """
    fit_indices = np.asarray(fit_indices, dtype=np.int64)
    result = fit_indices.copy()
    rng = np.random.default_rng(seed)
    for round_id in FIT_ROUNDS:
        positions = np.asarray(sorted(
            [i for i, index in enumerate(fit_indices) if rows[index]["round_id"] == round_id],
            key=lambda i: CLASSES.index(rows[fit_indices[i]]["class_name"])), dtype=np.int64)
        require(len(positions) == len(CLASSES), "Each shuffled fit round must have eight rows")
        result[positions] = rng.permutation(fit_indices[positions])
    return result


def aggregate(hits: np.ndarray, rows: list[dict], tie_counts: np.ndarray) -> dict:
    hits = np.asarray(hits, dtype=np.float64)
    require(hits.shape == (len(rows),), "Metric rows do not match predictions")

    def group(indices: list[int]) -> dict:
        return {"rows": len(indices), "noun_top1_expected": float(hits[indices].mean())}

    return {
        "rows": len(rows),
        "noun_top1_expected": float(hits.mean()),
        "per_round": {round_id: group([i for i, row in enumerate(rows)
                                       if row["round_id"] == round_id])
                      for round_id in DEVELOPMENT_ROUNDS},
        "per_class": {label: group([i for i, row in enumerate(rows)
                                    if row["class_name"] == label]) for label in CLASSES},
        "queries_with_top_score_ties": int(np.count_nonzero(tie_counts > 1)),
        "maximum_top_tie_size": int(tie_counts.max()),
        "mean_top_tie_size": float(tie_counts.mean()),
    }


def duplicate_groups(keys: list[object], rows: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for index, key in enumerate(keys):
        groups[key].append(index)
    return [{"sample_ids": [rows[i]["sample_id"] for i in indices],
             "class_names": [rows[i]["class_name"] for i in indices],
             "count": len(indices)} for indices in groups.values() if len(indices) > 1]


def summarize_duplicates(rows: list[dict], raw_text: np.ndarray, centered_text: np.ndarray) -> dict:
    captions = [row["caption"] for row in rows]
    normalized = [" ".join(unicodedata.normalize("NFKC", text).casefold().split())
                  for text in captions]
    literal = duplicate_groups(captions, rows)
    normalized_groups = duplicate_groups(normalized, rows)
    for group in literal:
        group["caption"] = captions[next(i for i, row in enumerate(rows)
                                           if row["sample_id"] == group["sample_ids"][0])]
    for group in normalized_groups:
        group["normalized_caption"] = normalized[next(i for i, row in enumerate(rows)
                                                      if row["sample_id"] == group["sample_ids"][0])]
    return {
        "gallery_rows": len(rows),
        "literal_caption_groups": literal,
        "normalized_caption_groups": normalized_groups,
        "raw_text_vector_groups": duplicate_groups([tuple(row) for row in raw_text], rows),
        "centered_text_vector_groups": duplicate_groups([tuple(row) for row in centered_text], rows),
        "vector_equality": "Exact component equality, without a tolerance; no gallery rows removed.",
        "caption_normalization": "Unicode NFKC, casefold, and whitespace collapse; diagnostic only.",
    }


def analyze(data: dict) -> tuple[dict, dict]:
    rows = data["rows"]
    fit = np.asarray([i for i, row in enumerate(rows) if row["split"] == "fit"], dtype=np.int64)
    development = np.asarray([i for i, row in enumerate(rows)
                              if row["split"] == "development"], dtype=np.int64)
    require(len(fit) == 32 and len(development) == 16, "Expected 32 fit and 16 development rows")
    dev_rows = [rows[i] for i in development]
    labels = np.asarray([CLASSES.index(row["class_name"]) for row in rows], dtype=np.int64)
    image, image_mean = center_from_fit(data["image"], fit, "image")
    text, text_mean = center_from_fit(data["text"], fit, "text")
    targets = np.eye(len(CLASSES), dtype=np.float64)[labels[fit]]
    summary = {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "analysis_kind": "supervised_competence_pilot",
        "scope": SCOPE,
        "counts": {"all_rows": len(rows), "fit_rows": len(fit), "development_rows": len(development),
                   "classes": len(CLASSES), "fit_physical_instances": 8,
                   "development_physical_instances": 8, "development_captures_per_instance": 2},
        "class_order": list(CLASSES),
        "baselines": {"balanced_random_class_top1": 1.0 / len(CLASSES)},
        "settings": {"ridge_alpha": RIDGE_ALPHA, "ridge_solver": "float64 dual",
                     "ridge_formula": "X.T @ solve(X @ X.T + alpha * I, Y)",
                     "intercept": False, "probe_targets": "uncentered one-hot class labels",
                     "preprocessing": "Separate fit-only means, then row L2 normalization in each modality",
                     "mapping_prediction_normalization": "row L2 before cosine to all 16 development captions",
                     "top1_ties": "Expected class hit under uniform ordering of exactly equal maxima",
                     "shuffled_seeds": list(SHUFFLE_SEEDS),
                     "shuffled_correspondence": "Independent ordinary permutation within each fit round; fixed points retained",
                     "hyperparameter_search": False, "unpaired_alignment_run": False},
        "provenance": {**data["hashes"], "analyzer_sha256": sha256(Path(__file__)),
                       "runtime": {"numpy": np.__version__, "python": sys.version,
                                   "platform": platform.platform()},
                       "feature_row_order_changed": data["feature_row_order_changed"]},
        "feature_metadata": data["feature_metadata"],
        "development_caption_duplicates": summarize_duplicates(
            dev_rows, data["text"][development], text[development]),
        "methods": {},
    }
    predictions = {
        "schema_version": SCHEMA_VERSION, "protocol_id": PROTOCOL_ID,
        "analysis_kind": "supervised_competence_pilot", "scope": SCOPE,
        "class_order": list(CLASSES),
        "fit_sample_ids": [rows[i]["sample_id"] for i in fit],
        "development_rows": [{key: row[key] for key in
                              ("sample_id", "round_id", "class_name", "instance_id", "caption")}
                             for row in dev_rows],
        "fit_means": {"image": image_mean.tolist(), "text": text_mean.tolist()},
        "methods": {},
    }

    def record(name: str, scores: np.ndarray, candidate_classes: np.ndarray,
               candidates: list[str], extra: dict | None = None) -> None:
        hits, tie_counts, correct_ties = tie_expected_top1(scores, labels[development], candidate_classes)
        summary["methods"][name] = aggregate(hits, dev_rows, tie_counts)
        predictions["methods"][name] = {
            "candidate_ids": candidates, "scores": scores.tolist(),
            "rows": [{"sample_id": row["sample_id"], "noun_top1_expected": float(hits[i]),
                      "top_tie_count": int(tie_counts[i]), "correct_top_tie_count": int(correct_ties[i]),
                      "top_candidate_ids": [candidates[j] for j in np.flatnonzero(scores[i] == scores[i].max())]}
                     for i, row in enumerate(dev_rows)],
        }
        if extra:
            summary["methods"][name].update(extra)
            predictions["methods"][name].update(extra)

    for modality, matrix in (("image", image), ("text", text)):
        weights = dual_ridge(matrix[fit], targets)
        record(f"{modality}_noun_probe", matrix[development] @ weights,
               np.arange(len(CLASSES)), list(CLASSES))
    candidates = [row["sample_id"] for row in dev_rows]
    for seed in (None, *SHUFFLE_SEEDS):
        training_text_rows = fit if seed is None else shuffle_within_round(rows, fit, seed)
        weights = dual_ridge(image[fit], text[training_text_rows])
        name = "paired_ridge" if seed is None else f"shuffled_ridge_seed_{seed}"
        mapped = l2_rows(image[development] @ weights, f"{name} projected image")
        extra = {
            "training_text_sample_ids": [rows[i]["sample_id"] for i in training_text_rows],
            "training_correspondence_fixed_points": int(np.count_nonzero(training_text_rows == fit)),
            "training_correspondence_fixed_points_per_round": {
                round_id: sum(int(source == target) for source, target in zip(fit, training_text_rows)
                              if rows[source]["round_id"] == round_id) for round_id in FIT_ROUNDS},
        }
        if seed is not None:
            extra["seed"] = seed
        record(name, mapped @ text[development].T, labels[development], candidates, extra)
    shuffled = [summary["methods"][f"shuffled_ridge_seed_{seed}"] for seed in SHUFFLE_SEEDS]
    summary["shuffled_reference_mean"] = {
        "seeds": list(SHUFFLE_SEEDS), "noun_top1_expected": float(np.mean([m["noun_top1_expected"] for m in shuffled])),
        "per_round": {r: float(np.mean([m["per_round"][r]["noun_top1_expected"] for m in shuffled]))
                      for r in DEVELOPMENT_ROUNDS},
        "per_class": {c: float(np.mean([m["per_class"][c]["noun_top1_expected"] for m in shuffled]))
                      for c in CLASSES},
        "interpretation": "Three descriptive correspondence references, not a calibrated null distribution.",
    }
    return summary, predictions


def write_json(path: Path, value: dict) -> None:
    with path.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def run(features: Path, sealed_path: Path, protocol_path: Path, output: Path) -> dict:
    output = output.absolute()
    require(not output.exists() and not output.is_symlink(), f"Output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    lock_path = output.parent / f".{output.name}.analysis.lock"
    try:
        lock_fd = os.open(lock_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as error:
        raise ValueError(f"Another analysis holds the output lock: {lock_path}") from error
    staging = None
    stage = "load_and_validate"
    try:
        os.close(lock_fd)
        require(not output.exists() and not output.is_symlink(), f"Output already exists: {output}")
        data = load_inputs(features, sealed_path, protocol_path)
        stage = "fit_and_score"
        summary, predictions = analyze(data)
        stage = "publish"
        staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.stage-", dir=output.parent))
        write_json(staging / "predictions.json", predictions)
        summary["predictions_sha256"] = sha256(staging / "predictions.json")
        summary["created_at_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(staging / "summary.json", summary)
        require(all(sha256(path) == data["hashes"][key] for key, path in data["paths"].items()),
                "Inputs changed during analysis")
        require(not output.exists() and not output.is_symlink(), f"Output already exists: {output}")
        os.rename(staging, output)
        staging = None
        return summary
    except Exception as error:
        error.analysis_stage = stage
        raise
    finally:
        if staging is not None:
            shutil.rmtree(staging)
        lock_path.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--sealed", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=Path(__file__).with_name("protocol.json"))
    parser.add_argument("--out", type=Path, required=True, help="New output directory; never overwrite")
    args = parser.parse_args()
    try:
        summary = run(args.features, args.sealed, args.protocol, args.out)
    except Exception as error:
        input_hashes = {}
        for name, path in (("features", args.features), ("feature_metadata", args.features.with_suffix(".json")),
                           ("sealed", args.sealed), ("protocol", args.protocol),
                           ("release_lock", RELEASE_DIR / "release-lock.json")):
            try:
                input_hashes[name] = sha256(path)
            except OSError:
                input_hashes[name] = None
        print(json.dumps({"status": "failed", "analysis_kind": "supervised_competence_pilot",
                          "scope": SCOPE, "stage": getattr(error, "analysis_stage", "preflight"),
                          "error_type": type(error).__name__, "error": str(error),
                          "input_hashes_observed_after_failure": input_hashes,
                          "analyzer_sha256": sha256(Path(__file__)),
                          "created_at_utc": datetime.now(timezone.utc).isoformat()}, indent=2), file=sys.stderr)
        raise SystemExit(1) from error
    print(json.dumps({"output": str(args.out), "analysis_kind": summary["analysis_kind"],
                      "output_hashes": {name: sha256(args.out / name)
                                        for name in ("summary.json", "predictions.json")},
                      "methods": {name: result["noun_top1_expected"]
                                  for name, result in summary["methods"].items()}}, indent=2))


if __name__ == "__main__":
    main()
