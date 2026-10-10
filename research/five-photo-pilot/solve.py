#!/usr/bin/env python3
"""Exhaustive five-photo geometry matching, followed by separate answer checking."""

import argparse
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import shutil
import tempfile


PROTOCOL_ID = "five-photo-geometry-v1"
SOURCE_DIR = Path(__file__).resolve().parent
LOCKED_FILES = {"protocol.json", "extract_five.py", "solve.py"}
INPUT_KEYS = {"schema_version", "protocol_id", "image_ids", "text_ids", "image_distances",
              "text_distances", "protocol_sha256", "feature_sha256"}
N = 5
TOLERANCE = 1e-12
EDGES = tuple(itertools.combinations(range(N), 2))
SCOPE = ("Exploratory matching of five photographs and five descriptions using their within-modality "
         "distances. This is not Rosetta, uses no paired fitting, and has no held-out sample. "
         "Ranks, ties and shuffled alternatives are descriptive, not p-values or confidence estimates.")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    require(isinstance(value, dict), "Expected a JSON object")
    return value


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                          encoding="utf-8")


def valid_hash(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def verify_release():
    lock_path = SOURCE_DIR / "release-lock.json"
    digest = sha256(lock_path)
    lock = read_json(lock_path)
    require(lock.get("schema_version") == 1 and isinstance(lock.get("files"), dict)
            and set(lock["files"]) == LOCKED_FILES, "Invalid three-file release lock")
    for name, expected in lock["files"].items():
        require(valid_hash(expected) and sha256(SOURCE_DIR / name) == expected,
                f"Release source hash mismatch: {name}")
    return {"release_lock_sha256": digest, "source_hashes": lock["files"]}


def validate_blind(data):
    require(set(data) == INPUT_KEYS, "Blind input must contain exactly the eight permitted keys; no answers or labels")
    require(data["schema_version"] == 1 and data["protocol_id"] == PROTOCOL_ID, "Wrong blind-input schema")
    for name in ("protocol_sha256", "feature_sha256"):
        require(valid_hash(data[name]), f"Invalid {name}")
    for modality in ("image", "text"):
        ids = data[f"{modality}_ids"]
        require(isinstance(ids, list) and len(ids) == N and all(isinstance(i, str) and i.strip() for i in ids)
                and len(set(ids)) == N, f"Expected five unique {modality} IDs")
        matrix = data[f"{modality}_distances"]
        require(isinstance(matrix, list) and len(matrix) == N
                and all(isinstance(row, list) and len(row) == N for row in matrix),
                f"{modality} distances must be 5 by 5")
        for row in matrix:
            for value in row:
                require(type(value) in (int, float) and math.isfinite(value)
                        and -TOLERANCE <= value <= 2 + TOLERANCE,
                        f"Invalid {modality} cosine distance")
        for i, row in enumerate(matrix):
            for j, value in enumerate(row):
                require(abs(value - matrix[j][i]) <= TOLERANCE, f"Asymmetric {modality} distances")
            require(abs(row[i]) <= TOLERANCE, f"Nonzero {modality} distance diagonal")


def correlation(x, y):
    mx, my = math.fsum(x) / len(x), math.fsum(y) / len(y)
    dx, dy = [a - mx for a in x], [b - my for b in y]
    vx, vy = math.fsum(a * a for a in dx), math.fsum(b * b for b in dy)
    if vx == 0 or vy == 0:
        return None
    return math.fsum(a * b for a, b in zip(dx, dy)) / math.sqrt(vx * vy)


def enumerate_assignments(data):
    """Use only the two distance matrices; IDs merely name the saved assignments."""
    validate_blind(data)
    x = [float(data["image_distances"][i][j]) for i, j in EDGES]
    records = []
    for permutation in itertools.permutations(range(N)):
        y = [float(data["text_distances"][permutation[i]][permutation[j]]) for i, j in EDGES]
        records.append({
            "text_indices": list(permutation),
            "mapping": {image: data["text_ids"][permutation[i]] for i, image in enumerate(data["image_ids"])},
            "cost": math.fsum((a - b) ** 2 for a, b in zip(x, y)) / len(EDGES),
            "distance_correlation": correlation(x, y),
        })
    minimum = min(record["cost"] for record in records)
    for record in records:
        record["optimal"] = record["cost"] <= minimum + TOLERANCE
    return records


def solve(input_path, output):
    input_path, output = Path(input_path), Path(output)
    require(not output.exists() and not output.is_symlink(), "Output already exists")
    release = verify_release()
    input_digest = sha256(input_path)
    data = read_json(input_path)
    validate_blind(data)
    require(data["protocol_sha256"] == release["source_hashes"]["protocol.json"], "Blind protocol hash mismatch")
    records = enumerate_assignments(data)
    minimum = min(record["cost"] for record in records)
    best = [record for record in records if record["optimal"]]
    distinct = [record["cost"] for record in records if not record["optimal"]]
    assignments = {"schema_version": 1, "protocol_id": PROTOCOL_ID,
                   "image_ids": data["image_ids"], "text_ids": data["text_ids"], "assignments": records}
    selected = {
        "schema_version": 1, "protocol_id": PROTOCOL_ID, "scope": SCOPE,
        "objective": "Mean squared raw cosine-distance difference across the ten unordered image pairs",
        "arithmetic": "Python binary64 float with math.fsum", "tie_tolerance": TOLERANCE,
        "image_ids": data["image_ids"], "text_ids": data["text_ids"],
        "permutations_evaluated": len(records), "best_assignments": best, "tied_count": len(best),
        "minimum_cost": minimum, "gap_to_first_distinct_cost": min(distinct) - minimum if distinct else None,
        "input_sha256": input_digest, "protocol_sha256": data["protocol_sha256"],
        "feature_sha256": data["feature_sha256"], **release,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    lock = output.parent / f".{output.name}.lock"
    descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    os.close(descriptor)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.stage-", dir=output.parent))
    try:
        shutil.copyfile(input_path, staging / "blind-distances.json")
        require(sha256(staging / "blind-distances.json") == input_digest, "Blind input changed while copied")
        write_json(staging / "assignments.json", assignments)
        selected["assignments_sha256"] = sha256(staging / "assignments.json")
        write_json(staging / "selected.json", selected)
        write_json(staging / "checksums.json", {name: sha256(staging / name) for name in
                                               ("blind-distances.json", "assignments.json", "selected.json")})
        require(sha256(input_path) == input_digest and verify_release() == release, "Inputs changed during solve")
        require(not output.exists() and not output.is_symlink(), "Output already exists")
        os.rename(staging, output)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
        lock.unlink()
    return selected


def evaluate(selection, ground_truth, output):
    selection, ground_truth, output = Path(selection), Path(ground_truth), Path(output)
    require(not output.exists() and not output.is_symlink(), "Evaluation output already exists")
    release = verify_release()
    paths = [selection / "selected.json", selection / "assignments.json", selection / "blind-distances.json"]
    digests = [sha256(path) for path in paths]
    receipt_path = selection / "checksums.json"
    receipt_digest = sha256(receipt_path)
    require(read_json(receipt_path) == {path.name: digest for path, digest in zip(paths, digests)},
            "Finalized selection checksum mismatch")
    selected, all_records, blind = [read_json(path) for path in paths]
    require(selected.get("schema_version") == 1 and selected.get("protocol_id") == PROTOCOL_ID,
            "Wrong selection schema")
    require(selected.get("assignments_sha256") == digests[1]
            and selected.get("protocol_sha256") == release["source_hashes"]["protocol.json"]
            and all(selected.get(key) == value for key, value in release.items()), "Selection provenance mismatch")
    require(selected.get("input_sha256") == digests[2], "Blind input checksum mismatch")
    require(all_records.get("assignments") == enumerate_assignments(blind),
            "Saved assignments differ from exhaustive blind-distance calculation")
    require(blind["protocol_sha256"] == selected["protocol_sha256"]
            and blind["feature_sha256"] == selected["feature_sha256"], "Blind provenance mismatch")
    image_ids, text_ids = selected["image_ids"], selected["text_ids"]
    require(all_records.get("schema_version") == 1 and all_records.get("protocol_id") == PROTOCOL_ID
            and all_records.get("image_ids") == image_ids and all_records.get("text_ids") == text_ids,
            "Selection IDs disagree")
    require(image_ids == blind["image_ids"] and text_ids == blind["text_ids"], "Blind IDs disagree")
    # Only after validating the finalized blind result do we open the answer file.
    truth_digest = sha256(ground_truth)
    truth = read_json(ground_truth)
    if "mapping" in truth:
        require(set(truth) <= {"mapping", "labels", "image_labels"}, "Unexpected private answer fields")
        mapping = truth["mapping"]
        labels = {key: truth[key] for key in ("labels", "image_labels") if key in truth}
    else:
        mapping, labels = truth, {}
    require(isinstance(mapping, dict) and set(mapping) == set(image_ids)
            and all(isinstance(value, str) for value in mapping.values())
            and set(mapping.values()) == set(text_ids), "Ground truth must be a complete one-to-one mapping")
    for key, value in labels.items():
        require(isinstance(value, dict) and set(value) <= set(text_ids if key == "labels" else image_ids)
                and all(isinstance(label, str) for label in value.values()), "Invalid optional human labels")
    records = all_records.get("assignments", [])
    require(len(records) == math.factorial(N)
            and {tuple(record["text_indices"]) for record in records} == set(itertools.permutations(range(N))),
            "Expected all 120 distinct assignments")
    for record in records:
        require(record["mapping"] == {image: text_ids[record["text_indices"][i]] for i, image in enumerate(image_ids)}
                and type(record["cost"]) in (int, float) and math.isfinite(record["cost"])
                and record["cost"] >= 0, "Invalid assignment record")
    minimum = min(record["cost"] for record in records)
    best = [record for record in records if record["cost"] <= minimum + TOLERANCE]
    distinct = [record["cost"] for record in records if record not in best]
    gap = min(distinct) - minimum if distinct else None
    require(selected.get("best_assignments") == best and selected.get("tied_count") == len(best)
            and selected.get("minimum_cost") == minimum
            and selected.get("permutations_evaluated") == 120 and selected.get("tie_tolerance") == TOLERANCE
            and selected.get("gap_to_first_distinct_cost") == gap
            and all(record.get("optimal") is (record in best) for record in records), "Best-assignment records disagree")
    true_record = next(record for record in records if record["mapping"] == mapping)
    true_cost = true_record["cost"]
    rank_low = 1 + sum(record["cost"] < true_cost - TOLERANCE for record in records)
    rank_high = sum(record["cost"] <= true_cost + TOLERANCE for record in records)
    evaluated = [{**record, "correct_out_of_5": sum(record["mapping"][image] == mapping[image] for image in image_ids)}
                 for record in best]
    correct = [record["correct_out_of_5"] for record in evaluated]
    result = {
        "schema_version": 1, "protocol_id": PROTOCOL_ID, "scope": SCOPE,
        "selected_sha256": digests[0], "assignments_sha256": digests[1], "ground_truth_sha256": truth_digest,
        "blind_input_sha256": digests[2], "checksums_sha256": receipt_digest,
        "best_assignments_evaluated": evaluated, "tied_count": len(best),
        "mean_correct_out_of_5": math.fsum(correct) / len(correct),
        "mean_accuracy": math.fsum(correct) / (N * len(correct)),
        "correct_out_of_5_range": [min(correct), max(correct)],
        "random_individual_match_reference": 0.2, "random_expected_correct_out_of_5": 1.0,
        "true_mapping": mapping, "true_mapping_present_in_all_120": True, "true_cost": true_cost,
        "true_assignment_rank": rank_low, "true_assignment_rank_interval": [rank_low, rank_high],
        "true_cost_tied_count": rank_high - rank_low + 1,
        "true_mapping_is_optimal": true_record in best,
        "gap_to_first_distinct_cost": gap,
        "tie_tolerance": TOLERANCE, **labels,
    }
    require([sha256(path) for path in paths] == digests and verify_release() == release,
            "Inputs changed during evaluation")
    require(sha256(ground_truth) == truth_digest and sha256(receipt_path) == receipt_digest,
            "Answers or receipt changed during evaluation")
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, staging_name = tempfile.mkstemp(prefix=f".{output.name}.stage-", dir=output.parent)
    os.close(descriptor)
    staging = Path(staging_name)
    try:
        write_json(staging, result)
        os.link(staging, output)
    finally:
        staging.unlink()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    solver = commands.add_parser("solve", help="Read only blind distances and enumerate all 120 mappings")
    solver.add_argument("--input", type=Path, required=True)
    solver.add_argument("--out", type=Path, required=True)
    evaluator = commands.add_parser("evaluate", help="Read private answers after the solver has written its selection")
    evaluator.add_argument("--selection", type=Path, required=True)
    evaluator.add_argument("--ground-truth", type=Path, required=True)
    evaluator.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "solve":
        result = solve(args.input, args.out)
        print(json.dumps({"output": str(args.out), "minimum_cost": result["minimum_cost"],
                          "selected_sha256": sha256(args.out / "selected.json"),
                          "tied_count": result["tied_count"]}, indent=2))
    else:
        result = evaluate(args.selection, args.ground_truth, args.out)
        print(json.dumps({"output": str(args.out), "mean_correct_out_of_5": result["mean_correct_out_of_5"],
                          "true_assignment_rank_interval": result["true_assignment_rank_interval"]}, indent=2))


if __name__ == "__main__":
    main()
