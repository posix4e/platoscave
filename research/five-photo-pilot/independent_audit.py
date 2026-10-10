#!/usr/bin/env python3
"""Independent numerical replay; outputs only generic labels, numbers and hashes.

No primary extractor/solver functions are imported. This does not rerun the
encoders, inspect photographs, authenticate capture time, or establish general
scientific validity. Required input paths are supplied only on the command line
and are never included in the generated report.
"""

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import itertools
import json
import math
from pathlib import Path
import platform
import random
import re
import unicodedata

import numpy as np


PROTOCOL_ID = "five-photo-geometry-v1"
SOURCE_COMMIT = "feead1da4d77bb0dc674fcf0f5f38979fac9a90c"
PINS = {
    "protocol.json": "fcc1c4252ae268e7dcf645419eaea07e1196c09e939a7066e55cf084afddf603",
    "extract_five.py": "37665c8dbb0d2479bfc162854ad0508d6e1b89a6dc8ee51f75bb4d095c14a823",
    "solve.py": "6819921cecd461dc841b1ea89dddb18fab53fc0c36c07b631d7f755346ce5dfc",
}
HELPER_SHA = "3e8445f22a8f255dcf47af89ac0b8c45461ba98046549702abc7b09b6b59a624"
LABELS = ("Slide", "Trashcan", "Tree", "Basketball", "Water fountain")
IDS = [f"candidate-{i:03d}" for i in range(1, 6)]
TIE_TOL = 1e-12
REPLAY_TOL = 2e-14


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    def reject_constant(_):
        raise ValueError("Nonfinite JSON value")
    def unique_keys(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("Duplicate JSON key")
            result[key] = value
        return result
    return json.loads(Path(path).read_text(encoding="utf-8"),
                      object_pairs_hook=unique_keys, parse_constant=reject_constant)


def independent_distances(array, order):
    """Scalar accumulation from original features, without the saved matrix."""
    values = array.tolist()
    norms = [math.sqrt(math.fsum(x * x for x in row)) for row in values]
    result = np.zeros((5, 5), dtype=np.float64)
    for i, original_i in enumerate(order):
        for j in range(i + 1, 5):
            original_j = order[j]
            dot = math.fsum(a * b for a, b in zip(values[original_i], values[original_j]))
            result[i, j] = result[j, i] = 1.0 - dot / (norms[original_i] * norms[original_j])
    return result


def independent_enumeration(dx, dy):
    """NumPy reduction rather than the primary solver's Python math.fsum."""
    edges = list(itertools.combinations(range(5), 2))
    x = np.asarray([dx[i, j] for i, j in edges], dtype=np.float64)
    permutations = list(itertools.permutations(range(5)))
    costs, correlations = [], []
    for permutation in permutations:
        y = np.asarray([dy[permutation[i], permutation[j]] for i, j in edges], dtype=np.float64)
        costs.append(float(np.mean(np.square(x - y), dtype=np.float64)))
        if np.ptp(x) == 0 or np.ptp(y) == 0:
            correlations.append(None)
        else:
            correlations.append(float(np.corrcoef(x, y)[0, 1]))
    return permutations, costs, correlations


def audit(args):
    checked = []
    def check(condition, name):
        if not condition:
            raise ValueError("Audit check failed: " + name)
        if name not in checked:
            checked.append(name)
    def close(actual, expected, name, tolerance=REPLAY_TOL):
        if expected is None:
            check(actual is None, name)
        else:
            check(type(actual) in (int, float) and math.isfinite(actual)
                  and abs(actual - expected) <= tolerance, name)

    paths = {
        "features": args.features / "features.npz",
        "feature_metadata": args.features / "features.json",
        "extraction_preflight": args.features / "preflight.json",
        "extracted_blind_matrices": args.features / "blind-distances.json",
        "ground_truth": args.features / "private-ground-truth.json",
        "solver_blind_matrices": args.selection / "blind-distances.json",
        "assignments": args.selection / "assignments.json",
        "selected": args.selection / "selected.json",
        "selection_checksums": args.selection / "checksums.json",
        "evaluation": args.evaluation,
        "private_manifest": args.manifest,
        "release_lock": args.release / "release-lock.json",
        "frozen_helper": args.release.parent / "fresh-pilot" / "extract.py",
    }
    paths.update({"source_" + name: args.release / name for name in PINS})
    hashes = {role: digest(path) for role, path in paths.items()}
    lock = read_json(paths["release_lock"])
    check(lock["schema_version"] == 1 and lock["files"] == PINS, "published three-file source lock")
    check(all(hashes["source_" + name] == expected for name, expected in PINS.items()), "frozen source bytes")
    check(hashes["frozen_helper"] == HELPER_SHA, "frozen feature helper bytes")
    protocol = read_json(paths["source_protocol.json"])
    check(protocol["protocol_id"] == PROTOCOL_ID and protocol["data"]["human_labels_verbatim"] == list(LABELS)
          and protocol["analysis"]["tie_tolerance_absolute"] == TIE_TOL, "protocol identity, labels and tie tolerance")

    metadata = read_json(paths["feature_metadata"])
    preflight = read_json(paths["extraction_preflight"])
    manifest = read_json(paths["private_manifest"])
    rows = manifest["rows"]
    check(manifest["protocol_id"] == PROTOCOL_ID and len(rows) == 5
          and set(row["candidate_id"] for row in rows) == set(IDS)
          and set(row["label_verbatim"] for row in rows) == set(LABELS), "complete original manifest identity and labels")
    check(metadata["status"] == "complete" and metadata["schema_version"] == 1
          and metadata["protocol_id"] == PROTOCOL_ID and metadata["sample_count"] == 5,
          "complete feature metadata")
    for key, role in (("feature_sha256", "features"), ("manifest_sha256", "private_manifest"),
                      ("preflight_sha256", "extraction_preflight"), ("protocol_sha256", "source_protocol.json"),
                      ("release_lock_sha256", "release_lock"), ("extractor_sha256", "source_extract_five.py"),
                      ("helper_sha256", "frozen_helper")):
        check(metadata[key] == hashes[role], "feature provenance hashes")
    check(metadata["private_manifest_rows"] == rows and preflight["private_manifest_rows"] == rows
          and preflight["manifest_sha256"] == hashes["private_manifest"], "feature and preflight rows match original manifest")
    check(metadata["input_sha256"] == preflight["input_sha256"], "preflight and final input hash records agree")
    check(metadata["models_verified"] == {"image": True, "text": True}, "recorded model-verification flags")
    expected_models = {"dino_checkpoint_sha256": protocol["models"]["image"]["checkpoint_sha256"],
                       "glove_zip_sha256": protocol["models"]["text"]["archive_sha256"],
                       "glove_member_sha256": protocol["models"]["text"]["member_sha256"]}
    check(metadata["model_hashes"] == expected_models
          and metadata["dino_source_sha256"] == protocol["models"]["image"]["source_sha256"]
          and metadata["dino_source_commit"] == protocol["models"]["image"]["source_commit"],
          "model and source provenance matches protocol")
    runtime = metadata["runtime"]
    check(runtime["batch_size"] == 1 and runtime["float32"] is True and runtime["tf32"] is False
          and runtime["deterministic_algorithms"] is True and runtime["cublas_workspace_config"] == ":4096:8",
          "recorded frozen inference settings")
    for name, role in (("features.npz", "features"), ("blind-distances.json", "extracted_blind_matrices"),
                       ("private-ground-truth.json", "ground_truth")):
        check(metadata["artifact_sha256"][name] == hashes[role], "extraction artifact hashes")

    with np.load(paths["features"], allow_pickle=False) as archive:
        check(set(archive.files) == {"image", "text", "sample_ids"}, "exact feature archive fields")
        x, y, sample_ids = archive["image"], archive["text"], archive["sample_ids"]
    check(sample_ids.dtype.kind == "U" and sample_ids.shape == (5,)
          and sample_ids.tolist() == metadata["sample_ids"] == [row["candidate_id"] for row in rows],
          "feature row order and unique identities")
    for matrix, dimension in ((x, 768), (y, 300)):
        check(matrix.dtype == np.float32 and matrix.shape == (5, dimension) and np.isfinite(matrix).all(),
              "feature dimensions, dtype and finite values")
        norms = [math.sqrt(math.fsum(float(v) ** 2 for v in row)) for row in matrix]
        check(all(abs(norm - 1) <= 1e-4 for norm in norms), "nonzero unit feature vectors")

    orders = []
    for seed in (20261010, 20261011):
        order = list(range(5))
        random.Random(seed).shuffle(order)
        orders.append(order)
    private_orders = metadata["private_blind_order"]
    for modality, seed, order in zip(("image", "text"), (20261010, 20261011), orders):
        check(private_orders[modality + "_shuffle_seed"] == seed
              and private_orders[modality + "_manifest_indices"] == order
              and private_orders[modality + "_candidate_ids"] == [rows[i]["candidate_id"] for i in order],
              "declared independent shuffle seeds and index mappings")
    dx, dy = independent_distances(x, orders[0]), independent_distances(y, orders[1])
    blind = read_json(paths["extracted_blind_matrices"])
    check(set(blind) == {"schema_version", "protocol_id", "protocol_sha256", "feature_sha256", "image_ids",
                         "text_ids", "image_distances", "text_distances"}, "answer-free solver input schema")
    check(hashes["extracted_blind_matrices"] == hashes["solver_blind_matrices"], "solver received identical blind matrices")
    check(blind["protocol_sha256"] == PINS["protocol.json"] and blind["feature_sha256"] == hashes["features"],
          "blind matrix provenance")
    image_ids, text_ids = [f"I{i:02d}" for i in range(5)], [f"T{i:02d}" for i in range(5)]
    check(blind["image_ids"] == image_ids and blind["text_ids"] == text_ids, "opaque shuffled IDs")
    matrix_error = max(float(np.max(np.abs(matrix - np.asarray(blind[key], dtype=np.float64))))
                       for matrix, key in ((dx, "image_distances"), (dy, "text_distances")))
    check(matrix_error <= REPLAY_TOL, "all 50 matrix entries replayed from raw features")

    expected_mapping = {image_ids[i]: text_ids[orders[1].index(original)] for i, original in enumerate(orders[0])}
    expected_labels = {text_ids[i]: rows[original]["label_verbatim"] for i, original in enumerate(orders[1])}
    expected_image_labels = {image_ids[i]: rows[original]["label_verbatim"] for i, original in enumerate(orders[0])}
    truth = read_json(paths["ground_truth"])
    check(truth == {"mapping": expected_mapping, "labels": expected_labels, "image_labels": expected_image_labels},
          "human mapping replayed from original manifest and both shuffles")

    permutations, costs, correlations = independent_enumeration(dx, dy)
    saved = read_json(paths["assignments"])
    records = saved["assignments"]
    check(saved["image_ids"] == image_ids and saved["text_ids"] == text_ids and len(records) == 120
          and [tuple(record["text_indices"]) for record in records] == permutations,
          "all 120 bijections present exactly once in declared order")
    minimum = min(costs)
    optimal_indices = [i for i, cost in enumerate(costs) if cost <= minimum + TIE_TOL]
    max_cost_error, max_correlation_error = 0.0, 0.0
    for index, (permutation, cost, correlation, record) in enumerate(zip(permutations, costs, correlations, records)):
        check(record["mapping"] == {image_ids[i]: text_ids[permutation[i]] for i in range(5)}, "every saved assignment mapping")
        close(record["cost"], cost, "all 120 MSE costs independently recomputed")
        close(record["distance_correlation"], correlation, "all 120 distance correlations independently recomputed")
        check(record["optimal"] is (index in optimal_indices), "every tolerance-based optimum flag")
        max_cost_error = max(max_cost_error, abs(record["cost"] - cost))
        if correlation is not None:
            max_correlation_error = max(max_correlation_error, abs(record["distance_correlation"] - correlation))
    distinct = [cost for i, cost in enumerate(costs) if i not in optimal_indices]
    gap = min(distinct) - minimum if distinct else None
    true_perm = tuple(text_ids.index(expected_mapping[image_id]) for image_id in image_ids)
    true_index = permutations.index(true_perm)
    true_cost = costs[true_index]
    rank = [1 + sum(cost < true_cost - TIE_TOL for cost in costs), sum(cost <= true_cost + TIE_TOL for cost in costs)]
    correct = [sum(a == b for a, b in zip(permutations[index], true_perm)) for index in optimal_indices]
    selected = read_json(paths["selected"])
    check(selected["permutations_evaluated"] == 120 and selected["tie_tolerance"] == TIE_TOL
          and selected["tied_count"] == len(optimal_indices)
          and selected["best_assignments"] == [records[i] for i in optimal_indices], "selected optimum set and ties")
    close(selected["minimum_cost"], minimum, "minimum objective")
    close(selected["gap_to_first_distinct_cost"], gap, "gap to next distinct cost")
    for field, role in (("input_sha256", "solver_blind_matrices"), ("assignments_sha256", "assignments"),
                        ("feature_sha256", "features"), ("protocol_sha256", "source_protocol.json"),
                        ("release_lock_sha256", "release_lock")):
        check(selected[field] == hashes[role], "selected-result provenance hashes")
    check(selected["source_hashes"] == PINS, "selected-result frozen source hashes")
    checksums = read_json(paths["selection_checksums"])
    check(checksums == {"blind-distances.json": hashes["solver_blind_matrices"],
                        "assignments.json": hashes["assignments"], "selected.json": hashes["selected"]},
          "completed solver output checksums")

    evaluation = read_json(paths["evaluation"])
    for field, role in (("selected_sha256", "selected"), ("assignments_sha256", "assignments"),
                        ("ground_truth_sha256", "ground_truth"), ("blind_input_sha256", "solver_blind_matrices"),
                        ("checksums_sha256", "selection_checksums")):
        check(evaluation[field] == hashes[role], "evaluation provenance hashes")
    check(evaluation["true_mapping"] == expected_mapping and evaluation["labels"] == expected_labels
          and evaluation["image_labels"] == expected_image_labels, "evaluated human labels and pairing")
    check(evaluation["true_assignment_rank_interval"] == rank and evaluation["true_assignment_rank"] == rank[0]
          and evaluation["true_cost_tied_count"] == rank[1] - rank[0] + 1
          and evaluation["true_mapping_is_optimal"] is (true_index in optimal_indices)
          and evaluation["true_mapping_present_in_all_120"] is True, "human-assignment rank interval and optimum status")
    check(evaluation["tie_tolerance"] == TIE_TOL and evaluation["tied_count"] == len(optimal_indices)
          and evaluation["correct_out_of_5_range"] == [min(correct), max(correct)], "evaluated ties and correctness range")
    close(evaluation["true_cost"], true_cost, "human-assignment objective")
    close(evaluation["gap_to_first_distinct_cost"], gap, "evaluated objective gap")
    close(evaluation["mean_correct_out_of_5"], float(np.mean(correct)), "mean correct count")
    close(evaluation["mean_accuracy"], float(np.mean(correct)) / 5, "mean correct fraction")
    close(evaluation["random_individual_match_reference"], 0.2, "uniform random row-match reference")
    close(evaluation["random_expected_correct_out_of_5"], 1.0, "uniform random expected correct count")
    check(evaluation["best_assignments_evaluated"] == [dict(records[index], correct_out_of_5=count)
          for index, count in zip(optimal_indices, correct)], "each evaluated optimum and its correctness")

    coverage = metadata["text_coverage"]
    token_total, known_total, oov_counts = 0, 0, Counter()
    check(len(coverage["rows"]) == 5, "token report covers all five labels")
    public_tokens = []
    for row, token_row in zip(rows, coverage["rows"]):
        tokens = re.findall(r"[a-z]+(?:'[a-z]+)?|[0-9]+", unicodedata.normalize("NFKC", row["label_verbatim"]).lower())
        unknown = token_row["oov_tokens"]
        check(token_row["sample_id"] == row["candidate_id"] and token_row["tokens"] == tokens
              and token_row["total_tokens"] == len(tokens) and not (Counter(unknown) - Counter(tokens))
              and token_row["known_tokens"] == len(tokens) - len(unknown), "fixed tokenization and reported coverage accounting")
        token_total += len(tokens)
        known_total += token_row["known_tokens"]
        oov_counts.update(unknown)
        public_tokens.append({"label": row["label_verbatim"], "tokens": tokens, "reported_oov_tokens": unknown})
    check(coverage["total_tokens"] == token_total and coverage["known_tokens"] == known_total
          and coverage["oov_counts"] == dict(oov_counts), "aggregate token and OOV accounting")
    close(coverage["token_coverage"], known_total / token_total, "reported token-coverage fraction")
    check(all(digest(path) == hashes[role] for role, path in paths.items()), "all audited input bytes unchanged during replay")

    assignment_summaries = []
    for index, count in zip(optimal_indices, correct):
        permutation = permutations[index]
        assignment_summaries.append({"text_indices": list(permutation), "correct_out_of_5": count,
                                    "cost": costs[index], "distance_correlation": correlations[index],
                                    "matches": [{"image_id": image_ids[i], "human_label": expected_image_labels[image_ids[i]],
                                                 "assigned_label": expected_labels[text_ids[permutation[i]]],
                                                 "correct": text_ids[permutation[i]] == expected_mapping[image_ids[i]]}
                                                for i in range(5)]})
    return {
        "schema_version": 1, "status": "passed", "protocol_id": PROTOCOL_ID,
        "audited_at_utc": datetime.now(timezone.utc).isoformat(), "reference_source_commit": SOURCE_COMMIT,
        "audit_script_sha256": digest(__file__), "runtime": {"python": platform.python_version(), "numpy": np.__version__},
        "independence": "No primary extractor or solver code imported. Scalar fsum cosine geometry from saved float32 features; separate NumPy MSE/correlation replay of all 120 bijections.",
        "checks_performed": checked, "input_sha256_by_role": hashes,
        "replay_tolerance_absolute": REPLAY_TOL, "tie_tolerance_absolute": TIE_TOL,
        "maximum_distance_matrix_absolute_error": matrix_error, "maximum_cost_absolute_error": max_cost_error,
        "maximum_correlation_absolute_error": max_correlation_error,
        "permutations_replayed": 120, "distance_entries_replayed": 50,
        "minimum_cost": minimum, "optimal_assignment_count": len(optimal_indices),
        "optimal_assignments": assignment_summaries, "mean_correct_out_of_5": float(np.mean(correct)),
        "correct_out_of_5_range": [min(correct), max(correct)], "mean_correct_fraction": float(np.mean(correct)) / 5,
        "human_assignment_cost": true_cost, "human_assignment_rank_interval": rank,
        "gap_to_next_distinct_cost": gap, "uniform_random_expected_correct_out_of_5": 1.0,
        "tokens": public_tokens,
        "not_performed": ["No encoder inference replay or independent GloVe lookup", "No inspection or rehash of photograph bytes",
                          "No independent authentication of human labels or capture chronology", "No audit of process-level access to the answer file"],
        "scope": "Numerical and recorded-provenance consistency for five selected items only. Exploratory transductive comparison, not Rosetta, a p-value, a confidence interval, or held-out generalization.",
        "privacy": "No photographs, raw feature vectors, private paths or capture metadata are included.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("run-root", "manifest", "release", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    args.features = args.run_root / "features"
    args.selection = args.run_root / "selection"
    args.evaluation = args.run_root / "evaluation.json"
    if args.out.exists():
        raise FileExistsError("Audit output already exists")
    result = audit(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": result["status"], "permutations_replayed": 120,
                      "mean_correct_out_of_5": result["mean_correct_out_of_5"],
                      "human_assignment_rank_interval": result["human_assignment_rank_interval"],
                      "audit_sha256": digest(args.out)}))


if __name__ == "__main__":
    main()
