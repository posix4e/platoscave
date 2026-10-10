#!/usr/bin/env python3
"""Post-result numerical audit, independently implemented from saved features.

Requires private inputs supplied explicitly by the custodian. It imports no
experiment modules and never loads an encoder, image, weight file or caption
model. The emitted JSON includes only opaque IDs, numerical summaries and
artifact hashes; no raw vectors, private manifest fields or filesystem paths.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import itertools
import json
import math
from pathlib import Path
import platform
import random
import re
import sys
import unicodedata

import numpy as np


PROTOCOL = "five-photo-model-choice-v1"
RELEASE_SHA = "ff8ba482770a1e62d1fceed3a8fa5f4f8e411cbfaba1f70330798f8450dd819a"
PROTOCOL_SHA = "852b8b021dbaed43cdbc54ca44b577c13906f37edb1f6dd4706661b507fbef9a"
CORE_SHA = "6819921cecd461dc841b1ea89dddb18fab53fc0c36c07b631d7f755346ce5dfc"
IDS = [f"candidate-{i:03d}" for i in range(1, 6)]
IMAGES = ("dino", "moco", "mae")
TEXTS = ("glove", "elmo")
DIMS = {"dino": 768, "glove": 300, "moco": 2048, "mae": 768, "elmo": 1024}
PAIRS = [a + "__" + b for a in IMAGES for b in TEXTS]
PERMUTATIONS = list(itertools.permutations(range(5)))
EDGES = list(itertools.combinations(range(5), 2))
TIE_TOL = 1e-12
DISTANCE_TOL = 2e-14
SCORE_TOL = 2e-14
CORRELATION_TOL = 2e-12


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("Duplicate JSON key")
            result[key] = value
        return result
    def invalid(_):
        raise ValueError("Nonfinite JSON constant")
    return json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=unique, parse_constant=invalid)


class Audit:
    def __init__(self):
        self.checks = []
        self.snapshot = {}
        self.hashes = {}

    def check(self, name, condition):
        if not condition:
            raise ValueError("Audit failed: " + name)
        self.checks.append(name)

    def track(self, path, role=None, expected=None):
        path = Path(path)
        value = digest(path)
        if expected is not None:
            self.check((role or "artifact") + " hash", value == expected)
        if path in self.snapshot:
            self.check((role or "artifact") + " stable while read", value == self.snapshot[path])
        self.snapshot[path] = value
        if role is not None:
            self.hashes[role] = value
        return value

    def document(self, path, role=None, expected=None):
        self.track(path, role, expected)
        return read_json(path)

    def archive(self, path, role=None, expected=None):
        self.track(path, role, expected)
        with np.load(path, allow_pickle=False) as archive:
            self.check((role or "NPZ") + " unique members", len(archive.files) == len(set(archive.files)))
            return {name: archive[name].copy() for name in archive.files}


def scalar_distances(array, order):
    """Scalar binary64/math.fsum replay instead of the extractor's BLAS product."""
    rows = [[float(v) for v in array[index]] for index in order]
    norms = [math.sqrt(math.fsum(v * v for v in row)) for row in rows]
    rows = [[v / norm for v in row] for row, norm in zip(rows, norms)]
    return [[0.0 if i == j else 1.0 - math.fsum(a * b for a, b in zip(rows[i], rows[j]))
             for j in range(5)] for i in range(5)]


def score_permutations(x, y):
    """Vectorized independent 120-permutation MSE and centered-cosine Pearson."""
    xedges = np.asarray([x[i][j] for i, j in EDGES], dtype=np.float64)
    yedges = np.asarray([[y[p[i]][p[j]] for i, j in EDGES] for p in PERMUTATIONS], dtype=np.float64)
    costs = np.mean(np.square(yedges - xedges), axis=1)
    xc = xedges - xedges.mean()
    yc = yedges - yedges.mean(axis=1, keepdims=True)
    denominators = np.linalg.norm(xc) * np.linalg.norm(yc, axis=1)
    correlations = [None if d == 0 else float(np.dot(xc, row) / d) for row, d in zip(yc, denominators)]
    return costs, correlations


def compare_record(audit, record, permutation, cost, correlation, optimal, image_ids, text_ids, name):
    expected_mapping = {image_ids[i]: text_ids[j] for i, j in enumerate(permutation)}
    audit.check(name + " identity", record["text_indices"] == list(permutation) and record["mapping"] == expected_mapping)
    audit.check(name + " cost", abs(record["cost"] - cost) <= SCORE_TOL)
    audit.check(name + " correlation", record["distance_correlation"] is None if correlation is None else
                record["distance_correlation"] is not None and abs(record["distance_correlation"] - correlation) <= CORRELATION_TOL)
    audit.check(name + " optimum", record["optimal"] is bool(optimal))


def run(args):
    audit = Audit()
    lock = audit.document(args.release / "release-lock.json", "release_lock", RELEASE_SHA)
    audit.check("release schema", lock["schema_version"] == 1 and lock["protocol_id"] == PROTOCOL)
    for name, expected in lock["files"].items():
        relative = Path(name)
        audit.check("safe locked relative path " + name, not relative.is_absolute() and ".." not in relative.parts)
        audit.track(args.release / relative, "release/" + name, expected)
    protocol = audit.document(args.release / "protocol.json", "protocol", PROTOCOL_SHA)
    audit.check("frozen protocol", protocol["protocol_id"] == PROTOCOL and protocol["executable"] is True
                and protocol["status"] == "frozen_before_additional_photo_features")
    binding = protocol["data"]["same_five_existing_photographs"]
    manifest = audit.document(args.manifest, "original_manifest", binding["private_manifest_sha256"])
    rows = manifest["rows"]
    audit.check("fixed five manifest rows", [r["candidate_id"] for r in rows] == IDS)
    audit.check("fixed human labels", [r["label_verbatim"] for r in rows] == binding["human_labels_verbatim"])
    primary_features = audit.archive(args.primary_root / "features/features.npz", "original_features", binding["original_feature_npz_sha256"])
    primary_metadata = audit.document(args.primary_root / "features/features.json", "original_metadata", binding["original_feature_metadata_sha256"])
    primary_evaluation = audit.document(args.primary_root / "evaluation.json", "original_evaluation", binding["original_evaluation_sha256"])
    audit.track(args.primary_root / "selection/selected.json", "original_selected", binding["original_selected_result_sha256"])
    audit.check("original feature IDs/fields", set(primary_features) == {"image", "text", "sample_ids"}
                and primary_features["sample_ids"].dtype.kind == "U" and primary_features["sample_ids"].tolist() == IDS)
    vision_specs = {key: value for key, value in zip(IMAGES, protocol["vision_models"])}
    text_specs = {key: value for key, value in zip(TEXTS, protocol["text_models"])}
    audit.check("original feature model/source receipts", primary_metadata["status"] == "complete"
                and primary_metadata["sample_ids"] == IDS and primary_metadata["sample_count"] == 5
                and primary_metadata["models_verified"] == {"image": True, "text": True}
                and primary_metadata["model_hashes"] == {
                    "dino_checkpoint_sha256": vision_specs["dino"]["checkpoint_sha256"],
                    "glove_zip_sha256": text_specs["glove"]["archive_sha256"],
                    "glove_member_sha256": text_specs["glove"]["member_sha256"]}
                and primary_metadata["dino_source_sha256"] == vision_specs["dino"]["source_sha256"]
                and primary_metadata["feature_sha256"] == binding["original_feature_npz_sha256"])

    assembled_receipt = audit.document(args.assembled / "checksums.json", "assembled_checksums")
    audit.check("complete assembled receipt", set(assembled_receipt) == {
        "features.npz", "blind-panel.json", "private-ground-truth.json", "private-feature-records.json"})
    for name, expected in assembled_receipt.items():
        audit.track(args.assembled / name, "assembled/" + name, expected)
    features = audit.archive(args.assembled / "features.npz")
    records = audit.document(args.assembled / "private-feature-records.json")
    blind = audit.document(args.assembled / "blind-panel.json")
    truth = audit.document(args.assembled / "private-ground-truth.json")
    audit.check("five encoder archives and IDs", set(features) == set(DIMS) | {"sample_ids"}
                and features["sample_ids"].dtype.kind == "U" and features["sample_ids"].tolist() == IDS)
    for name, dimensions in DIMS.items():
        value = features[name]
        audit.check(name + " valid feature matrix", value.dtype == np.float32 and value.shape == (5, dimensions)
                    and np.isfinite(value).all() and np.allclose(np.linalg.norm(value.astype(np.float64), axis=1), 1, atol=1e-4, rtol=0))
    audit.check("original DINO rows unchanged", np.array_equal(features["dino"], primary_features["image"]))
    audit.check("original GloVe rows unchanged", np.array_equal(features["glove"], primary_features["text"]))
    provenance = {"protocol_id": PROTOCOL, "protocol_sha256": PROTOCOL_SHA,
                  "release_lock_sha256": RELEASE_SHA, "source_hashes": lock["files"]}
    inputs = {"original_features": binding["original_feature_npz_sha256"], "manifest": binding["private_manifest_sha256"]}
    vision_ready = audit.document(args.release / "vision-readiness.json")
    elmo_ready = audit.document(args.release / "elmo-readiness.json")
    audit.check("synthetic readiness records", vision_ready["passed"] is True and elmo_ready["status"] == "synthetic_readiness_passed")
    expected_tokens = [{"sample_id": row["candidate_id"], "tokens": re.findall(r"[a-z]+(?:'[a-z]+)?|[0-9]+",
                       unicodedata.normalize("NFKC", row["label_verbatim"]).lower())} for row in rows]
    for row in expected_tokens:
        row["count"] = len(row["tokens"])
    caption_document = {"sample_ids": IDS, "captions": [row["label_verbatim"] for row in rows]}
    caption_sha = hashlib.sha256((json.dumps(caption_document, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode()).hexdigest()
    for name in ("moco", "mae", "elmo"):
        feature_path, metadata_path = getattr(args, name + "_features"), getattr(args, name + "_metadata")
        additional = audit.archive(feature_path, name + "_features")
        metadata = audit.document(metadata_path, name + "_metadata")
        inputs[name + "_features"] = audit.hashes[name + "_features"]
        inputs[name + "_metadata"] = audit.hashes[name + "_metadata"]
        audit.check(name + " original feature archive and assembled receipt", set(additional) == {"features", "sample_ids"}
                    and additional["sample_ids"].dtype.kind == "U" and additional["sample_ids"].tolist() == IDS
                    and np.array_equal(additional["features"], features[name]) and metadata == records[name]
                    and metadata["feature_sha256"] == inputs[name + "_features"]
                    and metadata["status"] == "complete" and metadata["sample_count"] == 5)
        if name in IMAGES:
            expected_name = "moco_v2_r50_800ep" if name == "moco" else "mae_vit_b16"
            audit.check(name + " fixed source/model/preprocessing receipt",
                        all(metadata[k] == v for k, v in provenance.items())
                        and metadata["weights_sha256"] == vision_specs[name]["checkpoint_sha256"]
                        and metadata["encoder"] == name and metadata["batch_size"] == 1
                        and metadata["helper_sha256"] == protocol["shared_image_preprocessing"]["helper_sha256"]
                        and metadata["shape"] == [5, DIMS[name]] and metadata["dtype"] == "float32"
                        and metadata["model_loading"] == vision_ready["encoders"][expected_name]["provenance"]
                        and metadata["runtime"]["pillow"] == vision_ready["pillow"]
                        and all(metadata["runtime"][key] == vision_ready["runtime"][key] for key in ("torch", "numpy")))
        else:
            spec = text_specs["elmo"]
            audit.check("ELMo source/runtime/representation receipt",
                        metadata["weights_sha256"] == spec["weights_sha256"] and metadata["options_sha256"] == spec["options_sha256"]
                        and metadata["source_sha256"] == lock["files"]["elmo_features.py"]
                        and metadata["backend_source_sha256"] == spec["executable_implementation_lock"]["backend_source_sha256"]
                        and metadata["runtime"] == spec["executable_implementation_lock"]["runtime"]
                        and all(metadata[key] == elmo_ready[key] for key in ("representation", "recurrent_state", "scalar_mix",
                            "dropout", "character_truncation", "tokenization", "output_dtype", "feature_dimensions"))
                        and all(metadata[key] is True for key in ("frozen_model", "all_finite", "l2_normalized")))
            audit.check("ELMo exact human caption hash/token order", metadata["sample_ids"] == IDS
                        and metadata["captions_sha256"] == caption_sha and metadata["tokens"] == expected_tokens)

    audit.check("blind exact schema", set(blind) == set(provenance) | {
        "schema_version", "image_ids", "text_ids", "input_sha256_by_role", "image_distances", "text_distances"})
    audit.check("blind provenance", all(blind[key] == value for key, value in provenance.items())
                and blind["schema_version"] == 1 and blind["input_sha256_by_role"] == inputs)
    image_ids, text_ids = [f"I{i:02d}" for i in range(5)], [f"T{i:02d}" for i in range(5)]
    audit.check("blind fixed opaque IDs/model axes", blind["image_ids"] == image_ids and blind["text_ids"] == text_ids
                and list(blind["image_distances"]) == list(IMAGES) and list(blind["text_distances"]) == list(TEXTS))
    io, to = list(range(5)), list(range(5))
    random.Random(20261010).shuffle(io)
    random.Random(20261011).shuffle(to)
    expected_truth = {"mapping": {image_ids[i]: text_ids[to.index(index)] for i, index in enumerate(io)},
                      "image_labels": {image_ids[i]: rows[index]["label_verbatim"] for i, index in enumerate(io)},
                      "labels": {text_ids[i]: rows[index]["label_verbatim"] for i, index in enumerate(to)}}
    audit.check("human correspondence from manifest and declared shuffles", truth == expected_truth)
    audit.check("human correspondence unchanged from original", truth == {
        "mapping": primary_evaluation["true_mapping"], "labels": primary_evaluation["labels"],
        "image_labels": primary_evaluation["image_labels"]})
    matrices, distance_errors = {}, {}
    for name in DIMS:
        order, modality = (io, "image") if name in IMAGES else (to, "text")
        recomputed = scalar_distances(features[name], order)
        saved = np.asarray(blind[modality + "_distances"][name], dtype=np.float64)
        audit.check(name + " saved distance structure", saved.shape == (5, 5) and np.isfinite(saved).all()
                    and np.array_equal(saved, saved.T) and np.array_equal(np.diag(saved), np.zeros(5)))
        error = float(np.max(np.abs(saved - np.asarray(recomputed))))
        audit.check(name + " feature-to-distance replay", error <= DISTANCE_TOL)
        matrices[name], distance_errors[name] = recomputed, error

    receipt = audit.document(args.selection / "checksums.json", "selection_checksums")
    audit.check("complete selection receipt", set(receipt) == {"blind-panel.json", "assignments.json", "selected.json"})
    for name, expected in receipt.items():
        audit.track(args.selection / name, "selection/" + name, expected)
    selected_blind = audit.document(args.selection / "blind-panel.json")
    assignments = audit.document(args.selection / "assignments.json")
    selected = audit.document(args.selection / "selected.json")
    evaluation = audit.document(args.evaluation, "evaluation")
    audit.check("identical blind input retained", selected_blind == blind and receipt["blind-panel.json"] == assembled_receipt["blind-panel.json"])
    for name, document in (("assignments", assignments), ("selected", selected), ("evaluation", evaluation)):
        audit.check(name + " source provenance", all(document[key] == value for key, value in provenance.items()))
    audit.check("selection hash/core links", selected["assignments_sha256"] == receipt["assignments.json"]
                and selected["blind_input_sha256"] == receipt["blind-panel.json"] and selected["numerical_core_sha256"] == CORE_SHA)
    audit.check("evaluation complete and bound", evaluation["status"] == "complete" and evaluation["primary_result_unchanged"] is True
                and evaluation["selected_sha256"] == receipt["selected.json"]
                and evaluation["assignments_sha256"] == receipt["assignments.json"]
                and evaluation["ground_truth_sha256"] == assembled_receipt["private-ground-truth.json"]
                and evaluation["primary_evaluation_sha256"] == binding["original_evaluation_sha256"]
                and all(evaluation[key] == value for key, value in truth.items())
                and evaluation["random_expected_correct_out_of_5"] == 1.0)
    audit.check("all six cells in every output", list(assignments["assignments"]) == PAIRS
                and list(selected["pairs"]) == PAIRS and list(evaluation["pairs"]) == PAIRS)
    true_indices = tuple(to.index(index) for index in io)
    true_index = PERMUTATIONS.index(true_indices)
    summaries, max_cost_error, max_correlation_error = {}, 0.0, 0.0
    for pair in PAIRS:
        image, text = pair.split("__")
        costs, correlations = score_permutations(matrices[image], matrices[text])
        minimum = float(costs.min())
        optimal = costs <= minimum + TIE_TOL
        saved_records = assignments["assignments"][pair]
        audit.check(pair + " complete ordered enumeration", len(saved_records) == 120
                    and [tuple(record["text_indices"]) for record in saved_records] == PERMUTATIONS)
        for number, (permutation, cost, correlation, is_best, record) in enumerate(zip(PERMUTATIONS, costs, correlations, optimal, saved_records)):
            compare_record(audit, record, permutation, cost, correlation, is_best, image_ids, text_ids, pair + " assignment " + str(number))
            max_cost_error = max(max_cost_error, abs(record["cost"] - float(cost)))
            if correlation is not None:
                max_correlation_error = max(max_correlation_error, abs(record["distance_correlation"] - correlation))
        chosen = selected["pairs"][pair]
        evaluated = evaluation["pairs"][pair]
        best_indices = [i for i, value in enumerate(optimal) if value]
        best = [saved_records[i] for i in best_indices]
        gap = float(costs[~optimal].min()) - minimum if (~optimal).any() else None
        audit.check(pair + " all selected optima preserved", chosen["best_assignments"] == best
                    and chosen["tied_count"] == len(best) and abs(chosen["minimum_cost"] - minimum) <= SCORE_TOL)
        audit.check(pair + " next distinct cost gap", chosen["gap_to_first_distinct_cost"] is None if gap is None else
                    chosen["gap_to_first_distinct_cost"] is not None and abs(chosen["gap_to_first_distinct_cost"] - gap) <= 2 * SCORE_TOL)
        cost = float(costs[true_index])
        rank = [1 + int((costs < cost - TIE_TOL).sum()), int((costs <= cost + TIE_TOL).sum())]
        correct = [sum(a == b for a, b in zip(PERMUTATIONS[i], true_indices)) for i in best_indices]
        evaluated_best = [{**record, "correct_out_of_5": count} for record, count in zip(best, correct)]
        audit.check(pair + " evaluation reuses chosen records", all(evaluated[key] == value for key, value in chosen.items())
                    and evaluated["best_assignments_evaluated"] == evaluated_best)
        audit.check(pair + " human cost/rank/correctness", abs(evaluated["true_cost"] - cost) <= SCORE_TOL
                    and evaluated["true_assignment_rank_interval"] == rank
                    and evaluated["true_mapping_is_optimal"] is bool(optimal[true_index])
                    and evaluated["mean_correct_out_of_5"] == sum(correct) / len(correct)
                    and evaluated["correct_out_of_5_range"] == [min(correct), max(correct)])
        summaries[pair] = {"permutations_verified": 120, "minimum_cost": minimum,
            "true_cost": cost, "true_assignment_rank_interval": rank, "tied_count": len(best),
            "gap_to_first_distinct_cost": gap, "mean_correct_out_of_5": sum(correct) / len(correct),
            "correct_out_of_5_range": [min(correct), max(correct)],
            "optimal_text_indices": [list(PERMUTATIONS[i]) for i in best_indices]}
    audit.check("original numerical result unchanged", all(evaluation["pairs"]["dino__glove"][key] == primary_evaluation[key]
                for key in ("best_assignments_evaluated", "true_cost", "true_assignment_rank_interval")))
    audit.check("all audited inputs stable at completion", all(digest(path) == expected for path, expected in audit.snapshot.items()))
    return {"schema_version": 1, "protocol_id": PROTOCOL, "status": "passed",
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(), "audit_source_sha256": digest(__file__),
        "pre_inference_release_commit": "7fcbc37", "release_lock_sha256": RELEASE_SHA,
        "scope": "Independent numerical implementation from the same saved FP32 features. No experiment scoring module imported; no second encoder inference, image-byte recheck, training-corpus audit or capture-chronology authentication.",
        "checks_performed": len(audit.checks), "checks": audit.checks,
        "pairs": summaries, "permutations_verified": 720, "feature_matrices_verified": 5,
        "original_feature_rows_unchanged": True, "original_result_unchanged": True,
        "max_absolute_errors": {"distance_by_encoder": distance_errors, "assignment_cost": max_cost_error,
                                "distance_correlation": max_correlation_error},
        "audit_comparison_tolerances": {"distance": DISTANCE_TOL, "cost": SCORE_TOL, "correlation": CORRELATION_TOL},
        "frozen_selection_tie_tolerance": TIE_TOL,
        "artifact_sha256_by_role": audit.hashes,
        "runtime": {"python": platform.python_version(), "numpy": np.__version__},
        "disclosures": ["Audit written after the experiment source was frozen; it cannot retroactively preregister the design.",
                        "Model/runtime/source metadata checked against locked receipts; encoder computation was not independently rerun.",
                        "The same five photographs supply every cell. No significance, power or held-out accuracy claim.",
                        "Raw photographs, features, manifest fields and private paths are excluded from this report."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("assembled", "selection", "evaluation", "manifest", "primary-root", "release", "out",
                 "moco-features", "moco-metadata", "mae-features", "mae-metadata", "elmo-features", "elmo-metadata"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists() or args.out.is_symlink():
        raise ValueError("Audit report destination already exists")
    report = run(args)
    with args.out.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": report["status"], "checks_performed": report["checks_performed"],
                      "permutations_verified": report["permutations_verified"], "report_sha256": digest(args.out)}))


if __name__ == "__main__":
    main()
