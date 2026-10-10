#!/usr/bin/env python3
"""Independently verify every EPIC-55 score, metric, map reference and probe.

This verifier imports neither the experiment runner nor Rosetta. It recomputes
scores with NumPy and retrieval hits by the k-th score boundary and exact
hypergeometric probabilities. Missing evidence or any failed check is fatal.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import itertools
import json
import math
from pathlib import Path
import sys
import unicodedata

import numpy as np

MANIFEST_SHA256 = "d1a34b2ad47a1456a011c6e4f2d6308bae34ebf6aed0c6fb335794ac3dc648be"
PROTOCOL_SHA256 = "656a4fad35609bb753fb5045268e3448e4c71a36bce70a4b20050fd16a547370"
METHOD_SEEDS = {
    "rosetta_disjoint": [0, 1, 2], "rosetta_shared": [0, 1, 2],
    "paired_orthogonal": [0], "paired_ridge": [0],
    "shuffled_orthogonal": [0, 1, 2], "shuffled_ridge": [0, 1, 2],
    "random_map": [0, 1, 2],
}
K_VALUES = (1, 5, 10)


def require(condition, message):
    if not bool(condition):
        raise ValueError(message)


def file_hash(path):
    checksum = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            checksum.update(block)
    return checksum.hexdigest()


def content_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def compare(actual, expected, context, atol=3e-12, rtol=3e-12):
    a, e = np.asarray(actual), np.asarray(expected)
    require(a.shape == e.shape, f"{context}: shape {a.shape} != {e.shape}")
    if a.dtype.kind in "OUS" or e.dtype.kind in "OUS":
        require(np.array_equal(a, e), f"{context}: identifiers/strings differ")
        return 0.0
    require(np.isfinite(a).all() and np.isfinite(e).all(), f"{context}: nonfinite value")
    require(np.allclose(a, e, atol=atol, rtol=rtol),
            f"{context}: values differ (maximum absolute error {np.max(np.abs(a - e)) if a.size else 0})")
    return float(np.max(np.abs(a - e))) if a.size else 0.0


def compare_structure(actual, expected, context):
    """Check every expected field; non-result descriptive fields may be extra."""
    if isinstance(expected, dict):
        require(isinstance(actual, dict), f"{context}: not an object")
        for key, value in expected.items():
            require(key in actual, f"{context}: missing {key}")
            compare_structure(actual[key], value, context + "/" + key)
    elif isinstance(expected, (str, bool)) or expected is None:
        require(actual == expected, f"{context}: {actual!r} != {expected!r}")
    else:
        compare(actual, expected, context)


def open_npz(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: np.array(archive[key]) for key in archive.files}


def normalize_rows(array):
    value = np.asarray(array, dtype=np.float64)
    lengths = np.sqrt(np.einsum("ij,ij->i", value, value))
    # Match the specified numerical floor for centered vectors. Raw features
    # are separately checked for nonzero norms before normalization.
    return value / np.maximum(lengths[:, None], 1e-10)


def boundary_probability(total, relevant, draws):
    """Exact fraction of size-draws subsets containing at least one relevant item."""
    if relevant == 0 or draws <= 0:
        return 0.0
    draws = min(int(draws), int(total))
    if draws > total - relevant:
        return 1.0
    return 1.0 - math.comb(int(total - relevant), draws) / math.comb(int(total), draws)


def independently_score_retrieval(scores, query_labels, gallery_labels, boundaries=None):
    """Use the top-k boundary, rather than the runner's best-positive formula."""
    scores = np.asarray(scores)
    query_labels, gallery_labels = np.asarray(query_labels), np.asarray(gallery_labels)
    require(scores.shape == (len(query_labels), len(gallery_labels)), "Metric shape mismatch")
    relevant = query_labels[:, None] == gallery_labels[None, :]
    positive_counts = relevant.sum(axis=1)
    require((positive_counts > 0).all(), "A retrieval query has no relevant gallery item")
    best_positive = np.where(relevant, scores, -np.inf).max(axis=1)
    before = (scores > best_positive[:, None]).sum(axis=1)
    same_best = scores == best_positive[:, None]
    equal = same_best.sum(axis=1)
    positive_equal = (same_best & relevant).sum(axis=1)
    output = {
        "expected_first_rank": before + (equal + 1) / (positive_equal + 1),
        "relevant_count": positive_counts,
        "greater_than_best_positive": before,
        "tied_at_best_positive": equal,
        "positive_ties_at_best": positive_equal,
    }
    n = scores.shape[1]
    for k in K_VALUES:
        take = min(k, n)
        cutoff = boundaries[k] if boundaries is not None else np.partition(scores, n - take, axis=1)[:, n - take]
        higher = scores > cutoff[:, None]
        tied = scores == cutoff[:, None]
        found_above = (higher & relevant).any(axis=1)
        remaining = take - higher.sum(axis=1)
        total_tied, relevant_tied = tied.sum(axis=1), (tied & relevant).sum(axis=1)
        hits = np.asarray([1.0 if found else boundary_probability(t, p, d)
                           for found, t, p, d in zip(found_above, total_tied, relevant_tied, remaining)])
        output[f"recall_at_{k}"] = hits
        output[f"chance_at_{k}"] = np.asarray([boundary_probability(n, p, take) for p in positive_counts])
    return output


def make_labels(rows):
    normalized = lambda text: " ".join(unicodedata.normalize("NFKC", text).casefold().split())
    return {
        "noun": np.asarray([str(row["noun_class"]) for row in rows]),
        "verb": np.asarray([str(row["verb_class"]) for row in rows]),
        "action": np.asarray([str(row["verb_class"]) + ":" + str(row["noun_class"]) for row in rows]),
        "narration": np.asarray([normalized(row["narration"]) for row in rows]),
        "instance_diagnostic": np.asarray([str(row["uid"]) for row in rows]),
    }


def independent_summary(values, participants):
    keys = [f"{prefix}_at_{k}" for prefix in ("recall", "chance") for k in K_VALUES]
    by_person = {}
    for person in sorted(set(participants.tolist())):
        mask = participants == person
        by_person[person] = {key: float(np.average(values[key][mask])) for key in keys}
        by_person[person]["queries"] = int(mask.sum())
    return {
        "queries": len(participants),
        "micro": {key: float(np.average(values[key])) for key in keys},
        "macro_participant": {key: float(np.average([v[key] for v in by_person.values()])) for key in keys},
        "by_participant": by_person,
        "median_expected_first_rank": float(np.median(values["expected_first_rank"])),
        "mean_relevant_gallery_entries": float(np.average(values["relevant_count"])),
    }


def verify_metrics(saved, reported, rows, context):
    summaries = {}
    labels = make_labels(rows)
    participants = np.asarray([row["participant_id"] for row in rows])
    for direction, matrix in (("image_to_text", saved["scores"]), ("text_to_image", saved["scores"].T)):
        summaries[direction] = {}
        n = matrix.shape[1]
        boundaries = {k: np.partition(matrix, n - min(k, n), axis=1)[:, n - min(k, n)] for k in K_VALUES}
        for definition, targets in labels.items():
            computed = independently_score_retrieval(matrix, targets, targets, boundaries)
            for key, value in computed.items():
                artifact_key = f"{direction}__{definition}__{key}"
                require(artifact_key in saved, f"{context}: absent per-query {artifact_key}")
                compare(saved[artifact_key], value, context + "/" + artifact_key)
            summary = independent_summary(computed, participants)
            compare_structure(reported[direction][definition], summary, context + "/summary/" + direction + "/" + definition)
            summaries[direction][definition] = summary
    return summaries


def verify_null(saved, reported, rows, context):
    matrix = saved["scores"]
    labels = make_labels(rows)["noun"]
    people = np.asarray([row["participant_id"] for row in rows])
    groups = [np.flatnonzero(people == p) for p in sorted(set(people.tolist()))]
    maxima = matrix.max(axis=1)
    top_counts = [Counter(labels[matrix[i] == maxima[i]].tolist()) for i in range(len(rows))]
    denominators = [sum(c.values()) for c in top_counts]

    def value_for_order(order):
        hits = np.asarray([top_counts[i].get(str(labels[target]), 0) / denominators[i]
                           for i, target in enumerate(order)])
        return float(np.average([np.average(hits[group]) for group in groups]))

    orders, original = saved["permutation_orders"], np.arange(len(rows))
    require(orders.shape == (999, len(rows)), context + ": null order shape differs")
    rng = np.random.default_rng(104729)
    null_values = []
    for draw, order in enumerate(orders):
        expected = original.copy()
        for group in groups:
            expected[group] = rng.permutation(group)
        compare(order, expected, context + f"/null permutation {draw}", atol=0, rtol=0)
        null_values.append(value_for_order(order))
    null_values = np.asarray(null_values)
    compare(saved["null_statistics"], null_values, context + "/null statistics")
    observed = value_for_order(original)
    expected = {
        "draws": 999, "seed": 104729, "observed": observed,
        "null_mean": float(np.average(null_values)),
        "null_q025": float(np.quantile(null_values, .025)), "null_q975": float(np.quantile(null_values, .975)),
        "null_tail_fraction": float((1 + (null_values >= observed).sum()) / 1000),
    }
    compare_structure(reported, expected, context + "/null summary")
    require("not exchangeable" in reported["inference_limit"] and "not a confirmatory" in reported["inference_limit"],
            context + ": null lacks temporal-dependence/inference limitation")
    return expected


def check_artifact(result_directory, record):
    filename = record["file"]
    require(Path(filename).name == filename, "Artifact path must be a filename inside result directory")
    path = result_directory / filename
    require(path.is_file(), f"Missing artifact {filename}")
    require(file_hash(path) == record["sha256"], f"Hash mismatch for {filename}")
    return path


def verify_manifest_and_features(args):
    require(file_hash(args.manifest) == MANIFEST_SHA256, "Not the original frozen selected-row manifest")
    require(file_hash(args.protocol) == PROTOCOL_SHA256, "Not the original frozen protocol")
    manifest = read_json(args.manifest)
    require(manifest["metadata"]["protocol_sha256"] == PROTOCOL_SHA256, "Manifest/protocol hash differs")
    rows = manifest["rows"]
    require(len(rows) == 9392 and len({str(r["uid"]) for r in rows}) == 9392, "Manifest rows/UIDs differ")
    index = {name: np.asarray([i for i, row in enumerate(rows) if row["split"] == name]) for name in ("fit", "dev", "test")}
    index.update({role: np.asarray([i for i, row in enumerate(rows) if row["split"] == "fit" and row["fit_role"] == role]) for role in ("x", "y")})
    require({key: len(ids) for key, ids in index.items()} == {"fit": 6639, "dev": 1127, "test": 1626, "x": 3187, "y": 3452}, "Wrong split/role counts")
    participants = {split: {rows[i]["participant_id"] for i in index[split]} for split in ("fit", "dev", "test")}
    require([len(participants[s]) for s in ("fit", "dev", "test")] == [16, 5, 7], "Wrong participant counts")
    for a, b in itertools.combinations(participants, 2):
        require(not participants[a] & participants[b], f"Participant overlap: {a}/{b}")
    require(not {rows[i]["video_id"] for i in index["x"]} & {rows[i]["video_id"] for i in index["y"]}, "Fitting modalities share videos")
    data = open_npz(args.features)
    sidecar = read_json(args.features.with_suffix(".json"))
    selection = sidecar["selection"]
    require(selection["is_final"] is True and selection["allow_partial"] is False, "Feature extraction is partial/nonfinal")
    require(selection["selected_rows"] == selection["output_rows"] == 9392 and selection["missing_rows"] == [], "Feature extraction omitted selected rows")
    require(sidecar["feature_file_sha256"] == file_hash(args.features) and sidecar["manifest_sha256"] == MANIFEST_SHA256, "Feature provenance hashes differ")
    require(sidecar["eligibility"]["eligible"] == 9392 and sidecar["eligibility"]["ineligible"] == 0, "Feature eligibility summary is incomplete")
    require("eligible" in data and data["eligible"].dtype == np.dtype(bool) and data["eligible"].shape == (9392,) and data["eligible"].all(), "Feature eligibility mask missing/invalid/incomplete")
    require(data["image"].shape == (9392, 768) and data["text"].shape == (9392, 300), "Incomplete/wrong feature shapes")
    for key in ("uid", "participant_id", "video_id", "narration", "split", "noun_class", "verb_class"):
        require(key in data, "Feature cache missing identity field " + key)
        compare(data[key].astype(str), np.asarray([str(r[key]) for r in rows]), "Feature identities " + key)
    for key in ("image", "text"):
        require(np.isfinite(data[key]).all(), "Nonfinite " + key + " features")
        require((np.linalg.norm(data[key], axis=1) > 1e-10).all(), "Zero " + key + " features")
    # The runner normalizes in float64, then materializes float32 fitting input.
    # Use NumPy's same norm reduction here to avoid changing float32 rounding
    # from an alternative reduction, while metrics below use independent logic.
    features = {key: (np.asarray(data[key], dtype=np.float64) /
                     np.linalg.norm(np.asarray(data[key], dtype=np.float64), axis=1, keepdims=True)).astype(np.float32)
                for key in ("image", "text")}
    return rows, index, features, participants


def verify_duplicate_audit(features, rows, index, reported):
    for split in ("fit", "dev", "test"):
        groups = defaultdict(list)
        ids = index[split]
        for i in ids:
            groups[features["text"][i].tobytes()].append(rows[i])
        narrations = make_labels([rows[i] for i in ids])["narration"]
        expected = {
            "rows": len(ids), "unique_normalized_narrations": len(set(narrations.tolist())),
            "duplicate_narration_rows_beyond_first": len(ids) - len(set(narrations.tolist())),
            "unique_float32_pooled_text_vectors": len(groups),
            "duplicate_vector_rows_beyond_first": len(ids) - len(groups),
            "largest_identical_vector_group": max(len(group) for group in groups.values()),
            "identical_vector_groups_with_multiple_noun_classes": sum(len({row["noun_class"] for row in group}) > 1 for group in groups.values()),
            "identical_vector_groups_with_multiple_action_classes": sum(len({(row["verb_class"], row["noun_class"]) for row in group}) > 1 for group in groups.values()),
        }
        compare_structure(reported[split], expected, "Duplicate audit/" + split)


def verify_map(model, method, seed, rows, index, features, context):
    for key, shape in (("weight", (768, 300)), ("mean_x", (1, 768)), ("mean_y", (1, 300))):
        require(model[key].shape == shape and np.isfinite(model[key]).all(), context + ": invalid " + key)
    xids, yids = model["train_x_indices"], model["train_y_indices"]
    expected_x = index["x"] if method in ("rosetta_disjoint", "random_map") else index["fit"]
    expected_y = index["y"] if method in ("rosetta_disjoint", "random_map") else index["fit"]
    for role, ids, expected in (("x", xids, expected_x), ("y", yids, expected_y)):
        require(ids.dtype.kind in "iu", context + ": non-integer training index")
        compare(np.sort(ids), expected, context + "/train-" + role + " identity set", atol=0, rtol=0)
    require(bool(model["input_unit_normalized"]), context + ": normalization flag differs")
    require(bool(model["normalize_projected"]) == ("ridge" in method), context + ": projection flag differs")
    if method.startswith("paired"):
        compare(xids, yids, context + "/paired correspondence", atol=0, rtol=0)
    if method.startswith("shuffled"):
        require((xids != yids).all(), context + ": shuffled pair contains fixed points")
        require(all(rows[x]["participant_id"] == rows[y]["participant_id"] for x, y in zip(xids, yids)), context + ": shuffled pair crosses participant")
    for name, ids, modality in (("mean_x", xids, "image"), ("mean_y", yids, "text")):
        # Torch and NumPy float32 reductions have different summation orders.
        compare(model[name], features[modality][ids].mean(axis=0, keepdims=True, dtype=np.float64),
                context + "/fit-only " + name, atol=5e-7, rtol=5e-6)
    x = normalize_rows(features["image"][xids] - model["mean_x"])
    y = normalize_rows(features["text"][yids] - model["mean_y"])
    weight = model["weight"]
    diagnostics = {}
    if "ridge" in method:
        # Check the optimality equation without reusing torch.linalg.solve.
        lhs = (x.T @ x + np.eye(768)) @ weight
        rhs = x.T @ y
        diagnostics["normal_equation_max_error"] = compare(lhs, rhs, context + "/ridge normal equation", atol=2e-8, rtol=2e-8)
    else:
        error = float(np.max(np.abs(weight.T @ weight - np.eye(300))))
        require(error <= 1e-4, context + ": column orthogonality violated")
        diagnostics["column_orthogonality_max_error"] = error
        if method in ("paired_orthogonal", "shuffled_orthogonal"):
            cross = x.T @ y
            h = weight.T @ cross
            compare(h, h.T, context + "/polar symmetry", atol=2e-8, rtol=2e-8)
            compare(weight @ h, cross, context + "/polar reconstruction", atol=2e-8, rtol=2e-8)
            require(np.linalg.eigvalsh((h + h.T) / 2).min() >= -2e-8, context + ": Procrustes optimum not positive semidefinite")
    return diagnostics


def replay_score_matrix(model, image, text):
    left = normalize_rows(image - model["mean_x"]) @ model["weight"]
    if bool(model["normalize_projected"]):
        left = normalize_rows(left)
    right = normalize_rows(text - model["mean_y"])
    return np.matmul(left, right.T)


def verify_identical_feature_ties(scores, image, text, context):
    """A tiny score tolerance must not hide artificial breaking of exact copies."""
    for name, features, matrix in (("image rows", image, scores), ("text columns", text, scores.T)):
        first = {}
        for i, feature in enumerate(features):
            key = feature.tobytes()
            if key in first:
                require(np.array_equal(matrix[i], matrix[first[key]]), context + ": exact duplicate " + name + " have different scores")
            else:
                first[key] = i


def verify_probes(directory, features, rows, index):
    document = read_json(directory / "probes.json")
    compare(document["train_indices"], index["fit"], "Probe fit indices", atol=0, rtol=0)
    require(document["ridge_alpha"] == 1.0, "Probe alpha differs")
    arrays = open_npz(check_artifact(directory, document["arrays"]))
    computed_results = {}
    maximum_replay_error = 0.0
    for modality in ("image", "text"):
        computed_results[modality] = {}
        for label in ("noun_class", "verb_class"):
            prefix = modality + "__" + label
            means, weights, priors = (arrays[prefix + "__" + k] for k in ("mean", "weight", "intercept"))
            all_labels = np.asarray([str(row[label]) for row in rows])
            classes = np.unique(all_labels[index["fit"]])
            compare(arrays[prefix + "__classes"], classes, "Probe classes")
            compare(means, features[modality][index["fit"]].mean(axis=0, keepdims=True, dtype=np.float64),
                    prefix + "/fit mean", atol=5e-7, rtol=5e-6)
            targets = (all_labels[index["fit"], None] == classes[None, :]).astype(np.float64)
            compare(priors, targets.mean(axis=0), prefix + "/fit class priors")
            train = normalize_rows(features[modality][index["fit"]] - means)
            compare((train.T @ train + np.eye(train.shape[1])) @ weights,
                    train.T @ (targets - priors), prefix + "/normal equation", atol=2e-8, rtol=2e-8)
            computed_results[modality][label] = {}
            for split in ("dev", "test"):
                ids = index[split]
                split_prefix = prefix + "__" + split
                scores = normalize_rows(features[modality][ids] - means) @ weights + priors
                maximum_replay_error = max(maximum_replay_error, compare(arrays[split_prefix + "__scores"], scores, split_prefix + "/score replay"))
                truth = all_labels[ids]
                compare(arrays[split_prefix + "__targets"], truth, split_prefix + "/targets")
                known = np.isin(truth, classes)
                hits = {k: np.zeros(len(ids)) for k in (1, 5)}
                for i, value in enumerate(truth):
                    if not known[i]:
                        continue
                    for k in (1, 5):
                        boundary = np.partition(scores[i], len(classes) - min(k, len(classes)))[len(classes) - min(k, len(classes))]
                        correct = scores[i, classes == value][0]
                        hits[k][i] = 1.0 if correct > boundary else (0.0 if correct < boundary else
                            (min(k, len(classes)) - int((scores[i] > boundary).sum())) / int((scores[i] == boundary).sum()))
                for k in (1, 5):
                    compare(arrays[split_prefix + f"__top{k}"], hits[k], split_prefix + f"/top{k}")
                people = np.asarray([rows[i]["participant_id"] for i in ids])
                summary = {
                    "top1": float(hits[1].mean()), "top5": float(hits[5].mean()),
                    "balanced_top1_all_test_classes": float(np.average([hits[1][truth == c].mean() for c in np.unique(truth)])),
                    "training_class_coverage": float(known.mean()), "num_train_classes": len(classes),
                    "num_evaluated_classes": len(np.unique(truth)),
                    "macro_participant_top1": float(np.average([hits[1][people == p].mean() for p in np.unique(people)])),
                    "majority_class_top1": float((truth == classes[int(np.argmax(priors))]).mean()),
                }
                compare_structure(document["results"][modality][label][split], summary, split_prefix + "/summary")
                computed_results[modality][label][split] = summary
    return computed_results, maximum_replay_error


def verify_aggregate(recomputed, summary):
    expected_results = {}
    for method, seeds in METHOD_SEEDS.items():
        people = sorted(next(iter(recomputed[method].values()))["by_participant"])
        seed_means = [recomputed[method][seed]["macro_participant"]["recall_at_1"] for seed in seeds]
        person_means = np.asarray([np.average([recomputed[method][seed]["by_participant"][p]["recall_at_1"] for seed in seeds]) for p in people])
        sampling = np.random.default_rng(65537).integers(0, len(people), size=(10000, len(people)))
        bootstrap = np.average(person_means[sampling], axis=1)
        expected = {
            "seeds": seeds, "seed_macro_recall_at_1": seed_means,
            "macro_participant_recall_at_1": float(person_means.mean()),
            "participant_bootstrap_95_interval": np.quantile(bootstrap, [.025, .975]).tolist(),
            "chance_macro_recall_at_1": float(np.average([recomputed[method][seeds[0]]["by_participant"][p]["chance_at_1"] for p in people])),
            "participants": people, "participant_seed_averaged_recall_at_1": person_means.tolist(),
        }
        compare_structure(summary["results"][method], expected, "Aggregate/" + method)
        expected_results[method] = expected
    comparisons = {}
    primary = expected_results["rosetta_disjoint"]
    for baseline in ("rosetta_shared", "paired_orthogonal", "paired_ridge", "shuffled_orthogonal", "random_map"):
        differences = np.asarray(primary["participant_seed_averaged_recall_at_1"]) - expected_results[baseline]["participant_seed_averaged_recall_at_1"]
        sample = np.random.default_rng(65537).integers(0, len(differences), size=(10000, len(differences)))
        name = "rosetta_disjoint_minus_" + baseline
        comparisons[name] = {"difference": float(differences.mean()),
                             "participant_bootstrap_95_interval": np.quantile(differences[sample].mean(axis=1), [.025, .975]).tolist()}
        compare_structure(summary["comparisons"][name], comparisons[name], "Aggregate comparison/" + name)
    require(summary["bootstrap"]["draws"] == 10000 and summary["bootstrap"]["seed"] == 65537, "Bootstrap settings differ")
    return expected_results, comparisons


def verify(args):
    rows, index, features, participants = verify_manifest_and_features(args)
    experiment = read_json(args.results / "experiment.json")
    config = experiment["config"]
    require(content_hash(config) == experiment["config_sha256"], "Experiment config checksum differs")
    require(config["manifest_sha256"] == MANIFEST_SHA256 and config["feature_sha256"] == file_hash(args.features), "Experiment input hashes differ")
    require(config["feature_metadata_sha256"] == file_hash(args.features.with_suffix(".json")), "Feature provenance sidecar differs from executed version")
    require(config["runner_sha256"] == file_hash(args.runner), "Runner source differs from execution record")
    require(config["seeds"] == [0, 1, 2] and set(config["methods"]) == set(METHOD_SEEDS), "Methods or seeds incomplete/different")
    require((config["clusters"], config["restarts"], config["iterations"], config["batch_size"], config["ridge_alpha"], config["permutation_draws"]) == (30, 30, 100, 2048, 1.0, 999), "Settings differ from protocol")
    expected_indices = {"train_x": index["x"], "train_y": index["y"], "paired_train": index["fit"], "shared_train": index["fit"], "dev": index["dev"], "test": index["test"]}
    for key, ids in expected_indices.items():
        compare(config["indices"][key], ids, "Config indices " + key, atol=0, rtol=0)
        require(config["sample_counts"][key] == len(ids), "Wrong config count " + key)
    compare_structure(config["participants"], {s: sorted(p) for s, p in participants.items()}, "Config participants")
    verify_duplicate_audit(features, rows, index, config["duplicate_audit"])
    require(content_hash(config["upstream_source_files"]) == config["upstream_source_sha256"], "Upstream source hash table differs")
    for relative, expected_hash in config["upstream_source_files"].items():
        source = args.upstream_source / relative
        require(source.is_file() and file_hash(source) == expected_hash, "Upstream executed source differs: " + relative)
    expected_run_names = {f"{method}_seed-{seed}" for method, seeds in METHOD_SEEDS.items() for seed in seeds}
    existing = {path.stem for path in args.results.glob("*_seed-*.json")}
    require(existing == expected_run_names and len(existing) == 17, "Expected exactly 17 completed fit records")
    summary = read_json(args.results / "summary.json")
    require(summary["status"] == "complete" and summary["completed_fits"] == 17 and summary["config_sha256"] == experiment["config_sha256"], "Final summary incomplete or inconsistent")
    recomputed, run_reports = defaultdict(dict), []
    largest_replay_error = 0.0
    matrices_checked = 0
    total_score_entries = 0
    rosetta_checked = 0
    for method, seeds in METHOD_SEEDS.items():
        for seed in seeds:
            name = f"{method}_seed-{seed}"
            result = read_json(args.results / (name + ".json"))
            require((result["status"], result["method"], result["seed"], result["config_sha256"]) == ("complete", method, seed, experiment["config_sha256"]), name + ": run record mismatch")
            require(set(result["artifacts"]) == {"map", "dev", "test"}, name + ": artifact set differs")
            model = open_npz(check_artifact(args.results, result["artifacts"]["map"]))
            map_diagnostics = verify_map(model, method, seed, rows, index, features, name)
            compare_structure(result["train_rows"], {"image": len(model["train_x_indices"]), "text": len(model["train_y_indices"])}, name + "/training counts")
            if method.startswith("rosetta"):
                compare_structure(result["fit_parameters"], {"num_clusters": 30, "num_restarts": 30, "num_iterations": 100, "batch_size": 2048}, name + "/solver parameters")
                rosetta_checked += 1
            else:
                require(result["fit_parameters"]["ridge_alpha"] == 1.0, name + ": control alpha differs")
            per_run_error = 0.0
            for split in ("dev", "test"):
                ids = index[split]
                selected = [rows[i] for i in ids]
                data = open_npz(check_artifact(args.results, result["artifacts"][split]))
                compare(data["row_indices"], ids, name + "/" + split + "/row indices", atol=0, rtol=0)
                compare(data["uid"].astype(str), np.asarray([str(r["uid"]) for r in selected]), name + "/" + split + "/UIDs")
                replayed = replay_score_matrix(model, features["image"][ids], features["text"][ids])
                error = compare(data["scores"], replayed, name + "/" + split + "/all-score replay", atol=2e-11, rtol=2e-11)
                verify_identical_feature_ties(data["scores"], features["image"][ids], features["text"][ids], name + "/" + split)
                per_run_error = max(per_run_error, error)
                matrices_checked += 1
                total_score_entries += data["scores"].size
                recomputed_metrics = verify_metrics(data, result["metrics"][split], selected, name + "/" + split)
                if split == "test":
                    recomputed[method][seed] = recomputed_metrics["image_to_text"]["noun"]
                    verify_null(data, result["permutation_null"], selected, name)
            largest_replay_error = max(largest_replay_error, per_run_error)
            run_reports.append({"name": name, "map_checks": map_diagnostics, "maximum_score_replay_error": per_run_error,
                                "all_dev_test_metrics_recomputed": True, "all_999_null_draws_recomputed": True})
            print(json.dumps({"verified_run": name, "verified_runs": len(run_reports), "expected_runs": 17}), flush=True)
    require(rosetta_checked == 6 and matrices_checked == 34, "Run/matrix completeness check failed")
    probe_results, probe_error = verify_probes(args.results, features, rows, index)
    compare_structure(summary["probes"], probe_results, "Final copied probe summaries")
    aggregates, comparisons = verify_aggregate(recomputed, summary)
    return {
        "passed": True, "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "verifier_sha256": file_hash(__file__), "manifest_sha256": file_hash(args.manifest),
        "protocol_sha256": file_hash(args.protocol), "features_sha256": file_hash(args.features),
        "experiment_config_sha256": experiment["config_sha256"], "runner_sha256": file_hash(args.runner),
        "scope": "All final benchmark maps/score matrices, duplicate-aware metrics, nulls, aggregate intervals, and probes; independent of runner metric functions",
        "checks": {
            "original_frozen_manifest_and_protocol": True, "all_9392_feature_UIDs_and_metadata": True,
            "final_feature_sidecar_hash_and_all_true_eligibility": True, "exact_text_vector_and_caption_duplicate_audit": True,
            "participant_disjoint_evaluation": True, "video_disjoint_primary_fit_marginals": True,
            "all_map_training_indices_are_correct_fit_rows": True, "fit_only_means_verified": True,
            "executed_runner_and_upstream_source_hashes": True,
            "completed_fits": len(run_reports), "rosetta_fits_C30_S30_R100_B2048": rosetta_checked,
            "full_score_matrices_replayed": matrices_checked, "individual_score_entries_replayed": total_score_entries,
            "maximum_score_replay_absolute_error": largest_replay_error,
            "score_replay_tolerance": {"absolute": 2e-11, "relative": 2e-11},
            "identical_features_retain_exact_score_ties": True,
            "bidirectional_definitions_recomputed_per_split_and_fit": ["noun", "verb", "action", "narration", "instance_diagnostic"],
            "hit_cutoffs_verified": [1, 5, 10], "participant_macro_and_per_query_values": True,
            "null_draws_recomputed": 17 * 999,
            "bootstrap_draws_per_method_and_comparison": 10000,
            "paired_and_shuffled_optimality_equations": True,
            "probe_score_matrices_replayed": 8, "maximum_probe_replay_absolute_error": probe_error,
            "probe_fit_equations_and_priors": True, "probe_top1_top5_unknown_classes_and_coverage": True,
        },
        "runs": run_reports, "recomputed_primary_aggregates": aggregates, "recomputed_comparisons": comparisons,
        "limitations": [
            "This verifier replays saved frozen features, not the DINO forward passes, video decoding, or GloVe extraction.",
            "Source hashes, fit-index records, configurations and score replay do not independently rerun all six stochastic Rosetta optimizations.",
            "This checks arithmetic and recorded protocol adherence; it cannot authenticate original model training histories or establish a causal contamination effect.",
            "Within-participant episode permutations remain diagnostic; temporal dependence prevents treating their tail fractions as calibrated significance tests."
        ],
    }


def self_test():
    for n in range(2, 8):
        for positives in range(1, n + 1):
            for draws in range(1, n + 1):
                subsets = list(itertools.combinations(range(n), draws))
                brute = sum(any(x < positives for x in subset) for subset in subsets) / len(subsets)
                compare(boundary_probability(n, positives, draws), brute, "Exhaustive tie subsets")
    scores = np.asarray([[.9, .9, .9, .3, .2, .1], [.5, .5, .5, .5, .5, .5]])
    query = np.asarray(["a", "b"])
    gallery = np.asarray(["a", "b", "b", "a", "a", "b"])
    values = independently_score_retrieval(scores, query, gallery)
    compare(values["recall_at_1"], [1 / 3, 1 / 2], "Mixed-positive ties")
    compare(values["recall_at_5"], [1, 1], "Multiple valid matches")
    compare(values["expected_first_rank"], [2, 7 / 4], "Expected first relevant rank")
    for permutation in ([5, 2, 1, 3, 0, 4], [1, 0, 2, 4, 3, 5]):
        changed = independently_score_retrieval(scores[:, permutation], query, gallery[permutation])
        for key in values:
            compare(changed[key], values[key], "Gallery permutation invariance")
    for corrupted in ([0, 1], [1, float("nan")], [[1, 2]]):
        try:
            compare(corrupted, [1, 2], "Deliberate corruption")
        except ValueError:
            pass
        else:
            raise ValueError("Verifier accepted deliberately corrupted values")
    print(json.dumps({"self_test": "passed", "checks": ["exhaustive tie-subset enumeration", "duplicate relevance", "first-positive order statistic", "column permutation invariance", "deliberate corruption rejection"]}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--protocol", type=Path, default=Path(__file__).with_name("protocol.json"))
    parser.add_argument("--features", type=Path)
    parser.add_argument("--results", type=Path)
    parser.add_argument("--runner", type=Path, default=Path(__file__).with_name("run_experiment.py"))
    parser.add_argument("--upstream-source", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if not all((args.manifest, args.features, args.results, args.upstream_source, args.output)):
        parser.error("--manifest, --features, --results, --upstream-source and --output are required")
    try:
        report = verify(args)
    except Exception as error:
        report = {"passed": False, "verified_at_utc": datetime.now(timezone.utc).isoformat(),
                  "verifier_sha256": file_hash(__file__), "error_type": type(error).__name__, "error": str(error),
                  "scope": "Final verification incomplete; no completion claim is valid from this report"}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        raise
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"passed": True, "report": str(args.output), "checks": report["checks"]}), flush=True)


if __name__ == "__main__":
    main()
