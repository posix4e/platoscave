#!/usr/bin/env python3
"""Run a frozen-encoder EPIC-55 control with the original Unpaired Rosetta code.

Inputs are a non-pickle NPZ (image, text, uid) and a JSON selection manifest
{metadata, rows}. Each row has split=fit/dev/test, and fit_role=x/y for fit.
Whole videos are assigned to one fit role; the two training marginals contain
different episodes. Shared and paired controls use ALL fit rows, with their
larger sample/information budget explicitly recorded.
Training IDs/labels never enter the unpaired fit API. Evaluation is held out by
participant. Full score matrices, map weights and per-query metrics are retained.

The input encoders are frozen. This script trains only maps and diagnostic linear
probes. It does not infer causality from the result of one encoder pair.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import platform
import sys
import tempfile
import time
import unicodedata
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

KS = (1, 5, 10)
METHODS = ("rosetta_disjoint", "rosetta_shared", "paired_orthogonal",
           "paired_ridge", "shuffled_orthogonal", "shuffled_ridge", "random_map")


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def normalized_text(text):
    return " ".join(unicodedata.normalize("NFKC", str(text)).casefold().split())


def unit(array):
    array = np.asarray(array, dtype=np.float64)
    return array / np.maximum(np.linalg.norm(array, axis=1, keepdims=True), 1e-10)


def safe_recall(greater, tied, positive_ties, k):
    """Expected hit@k under uniform tie breaking, not arbitrary row-order ties."""
    draws = min(max(int(k) - int(greater), 0), int(tied))
    if draws == 0:
        return 0.0
    if draws > tied - positive_ties:
        return 1.0
    miss = 1.0
    for index in range(draws):
        miss *= (tied - positive_ties - index) / (tied - index)
    return float(1.0 - miss)


def retrieval_per_query(scores, query_labels, gallery_labels):
    """Every item with the target label is relevant; ties are exact float ties."""
    query_labels, gallery_labels = np.asarray(query_labels), np.asarray(gallery_labels)
    if scores.shape != (len(query_labels), len(gallery_labels)):
        raise ValueError("Score matrix and evaluation labels have different shapes")
    output = {"expected_first_rank": np.zeros(len(scores), dtype=np.float64),
              "relevant_count": np.zeros(len(scores), dtype=np.int64),
              "greater_than_best_positive": np.zeros(len(scores), dtype=np.int64),
              "tied_at_best_positive": np.zeros(len(scores), dtype=np.int64),
              "positive_ties_at_best": np.zeros(len(scores), dtype=np.int64)}
    for k in KS:
        output[f"recall_at_{k}"] = np.zeros(len(scores), dtype=np.float64)
        output[f"chance_at_{k}"] = np.zeros(len(scores), dtype=np.float64)
    for i, (row, label) in enumerate(zip(scores, query_labels)):
        relevant = gallery_labels == label
        count = int(relevant.sum())
        if not count:
            raise ValueError("Each held-out query must have at least one relevant gallery entry")
        best = row[relevant].max()
        greater = int((row > best).sum())
        tied = int((row == best).sum())
        positive_ties = int(((row == best) & relevant).sum())
        output["expected_first_rank"][i] = greater + (tied + 1) / (positive_ties + 1)
        output["relevant_count"][i] = count
        output["greater_than_best_positive"][i] = greater
        output["tied_at_best_positive"][i] = tied
        output["positive_ties_at_best"][i] = positive_ties
        for k in KS:
            output[f"recall_at_{k}"][i] = safe_recall(greater, tied, positive_ties, k)
            output[f"chance_at_{k}"][i] = safe_recall(0, len(row), count, k)
    return output


def summarize_queries(per_query, participants):
    participants = np.asarray(participants)
    by_participant = {}
    keys = [f"recall_at_{k}" for k in KS] + [f"chance_at_{k}" for k in KS]
    for participant in sorted(set(participants.tolist())):
        mask = participants == participant
        by_participant[str(participant)] = {key: float(per_query[key][mask].mean()) for key in keys}
        by_participant[str(participant)]["queries"] = int(mask.sum())
    return {
        "queries": len(participants),
        "micro": {key: float(per_query[key].mean()) for key in keys},
        "macro_participant": {key: float(np.mean([p[key] for p in by_participant.values()])) for key in keys},
        "median_expected_first_rank": float(np.median(per_query["expected_first_rank"])),
        "mean_relevant_gallery_entries": float(per_query["relevant_count"].mean()),
        "by_participant": by_participant,
    }


def labels_for(rows):
    return {
        "action": np.asarray([f"{r['verb_class']}:{r['noun_class']}" for r in rows]),
        "noun": np.asarray([str(r["noun_class"]) for r in rows]),
        "verb": np.asarray([str(r["verb_class"]) for r in rows]),
        "narration": np.asarray([normalized_text(r["narration"]) for r in rows]),
        "instance_diagnostic": np.asarray([str(r["uid"]) for r in rows]),
    }


def evaluate(scores, rows):
    labels = labels_for(rows)
    participants = np.asarray([r["participant_id"] for r in rows])
    results, arrays = {}, {}
    for direction, matrix in (("image_to_text", scores), ("text_to_image", scores.T)):
        results[direction] = {}
        for definition, values in labels.items():
            per_query = retrieval_per_query(matrix, values, values)
            results[direction][definition] = summarize_queries(per_query, participants)
            arrays.update({f"{direction}__{definition}__{key}": value for key, value in per_query.items()})
    return results, arrays


def permutation_null(scores, rows, count=999, seed=104729):
    """Within-participant query-label permutation for the primary noun image→text R@1.

    Score rows remain untouched. Complete query noun labels are permuted within
    each participant; gallery labels and score ties are fixed. This tests extra
    noun correspondence beyond participant membership, not absence of all
    possible shared semantics. It is not a permutation test of pretraining cause.
    """
    labels = labels_for(rows)["noun"]
    participants = np.asarray([r["participant_id"] for r in rows])
    groups = [np.flatnonzero(participants == p) for p in sorted(set(participants.tolist()))]
    ties = scores == scores.max(axis=1, keepdims=True)
    classes, class_ids = np.unique(labels, return_inverse=True)
    top_prob = np.zeros((len(scores), len(classes)), dtype=np.float64)
    for i in range(len(scores)):
        top_prob[i] = np.bincount(class_ids[ties[i]], minlength=len(classes)) / ties[i].sum()
    row_ids = np.arange(len(rows))

    def macro(ids):
        values = top_prob[row_ids, ids]
        return float(np.mean([values[group].mean() for group in groups]))

    observed = macro(class_ids)
    rng = np.random.default_rng(seed)
    null = np.empty(count, dtype=np.float64)
    orders = np.empty((count, len(rows)), dtype=np.int32)
    for draw in range(count):
        order = row_ids.copy()
        for group in groups:
            order[group] = rng.permutation(group)
        orders[draw] = order
        null[draw] = macro(class_ids[order])
    result = {
        "metric": "image_to_text/noun/macro_participant/recall_at_1",
        "definition": "query noun labels permuted within participant; full gallery fixed",
        "draws": count, "seed": seed, "observed": observed,
        "null_mean": float(null.mean()), "null_q025": float(np.quantile(null, .025)),
        "null_q975": float(np.quantile(null, .975)),
        "null_tail_fraction": float((1 + (null >= observed).sum()) / (count + 1)),
        "tail_fraction_definition": "(1 + number of shuffled statistics >= observed) / (1 + number of draws); diagnostic, not calibrated population significance",
        "inference_limit": "conditional diagnostic only: overlapping/neighboring episodes are not exchangeable independent units; not a confirmatory significance test or a test of pretraining contamination",
    }
    return result, {"null_statistics": null, "permutation_orders": orders}


def require_indices(manifest, rows):
    indices = {
        "train_x": np.asarray([i for i, row in enumerate(rows) if row["split"] == "fit" and row["fit_role"] == "x"]),
        "train_y": np.asarray([i for i, row in enumerate(rows) if row["split"] == "fit" and row["fit_role"] == "y"]),
        "paired_train": np.asarray([i for i, row in enumerate(rows) if row["split"] == "fit"]),
        "shared_train": np.asarray([i for i, row in enumerate(rows) if row["split"] == "fit"]),
        **{split: np.asarray([i for i, row in enumerate(rows) if row["split"] == split]) for split in ("dev", "test")},
    }
    for key in ("train_x", "train_y", "paired_train", "shared_train", "dev", "test"):
        if key not in indices or not len(indices[key]):
            raise ValueError(f"Missing/empty manifest indices: {key}")
        values = indices[key]
        if len(set(values.tolist())) != len(values) or values.min() < 0 or values.max() >= len(rows):
            raise ValueError(f"Invalid/duplicate indices for {key}")
    if set(indices["train_x"]) & set(indices["train_y"]):
        raise ValueError("Unpaired disjoint training sets share a UID")
    if {rows[i]["video_id"] for i in indices["train_x"]} & {rows[i]["video_id"] for i in indices["train_y"]}:
        raise ValueError("Disjoint unpaired marginals share a video")
    train = set().union(*(set(indices[k]) for k in ("train_x", "train_y", "paired_train", "shared_train")))
    for i in train:
        if rows[i]["split"] not in ("fit", "train"):
            raise ValueError("A held-out row occurs in training indices")
    for split in ("dev", "test"):
        if any(rows[i]["split"] != split for i in indices[split]):
            raise ValueError(f"Incorrect split membership in {split}")
    participant_sets = {
        "fit": {rows[i]["participant_id"] for i in train},
        **{s: {rows[i]["participant_id"] for i in indices[s]} for s in ("dev", "test")},
    }
    for a, b in (("fit", "dev"), ("fit", "test"), ("dev", "test")):
        if participant_sets[a] & participant_sets[b]:
            raise ValueError(f"Participant leakage between {a} and {b}")
    # Explicitly reject shared video frames across the claimed disjoint marginals.
    if all("start_frame" in r and "stop_frame" in r for r in rows):
        videos = defaultdict(lambda: ([], []))
        for side, key in enumerate(("train_x", "train_y")):
            for i in indices[key]:
                row = rows[i]
                videos[row["video_id"]][side].append((int(row["start_frame"]), int(row["stop_frame"])))
        for video, (x_intervals, y_intervals) in videos.items():
            ys = sorted(y_intervals)
            for start, stop in x_intervals:
                if any(start <= y_stop and y_start <= stop for y_start, y_stop in ys):
                    raise ValueError(f"Disjoint arms have overlapping inclusive action frames in {video}")
    return indices, {key: sorted(value) for key, value in participant_sets.items()}


def load_inputs(args):
    manifest = json.loads(args.manifest.read_text())
    rows = manifest["rows"]
    if len({str(row["uid"]) for row in rows}) != len(rows):
        raise ValueError("Manifest has duplicate UIDs")
    sidecar_path = args.features.with_suffix(".json")
    if not sidecar_path.is_file():
        raise ValueError("Features require the adjacent extractor JSON provenance sidecar")
    sidecar = json.loads(sidecar_path.read_text())
    selection = sidecar.get("selection", {})
    if selection.get("is_final") is not True or selection.get("allow_partial") is not False:
        raise ValueError("Only a final extraction without --allow-partial can enter the experiment")
    if selection.get("selected_rows") != len(rows) or selection.get("output_rows") != len(rows):
        raise ValueError("Extractor selected/output row counts must equal the full frozen manifest")
    if selection.get("missing_rows") != []:
        raise ValueError("Extractor sidecar must report no missing rows")
    if sidecar.get("feature_file_sha256") != sha256(args.features):
        raise ValueError("Feature NPZ hash differs from its extractor sidecar")
    if sidecar.get("manifest_sha256") != sha256(args.manifest):
        raise ValueError("Extractor used a different manifest")
    eligibility_summary = sidecar.get("eligibility", {})
    if eligibility_summary.get("eligible") != len(rows) or eligibility_summary.get("ineligible") != 0:
        raise ValueError("Every frozen row must be eligible; no automatic filtering is permitted")
    with np.load(args.features, allow_pickle=False) as source:
        image, text = np.array(source["image"]), np.array(source["text"])
        if image.shape != (len(rows), 768) or text.shape != (len(rows), 300):
            raise ValueError(f"Expected N×768 DINO and N×300 GloVe, got {image.shape}, {text.shape}")
        for key in ("uid", "participant_id", "video_id", "narration", "split", "noun_class", "verb_class"):
            if key in source:
                expected = np.asarray([str(r[key]) for r in rows])
                if not np.array_equal(source[key].astype(str), expected):
                    raise ValueError(f"Feature/manifest row identity differs in {key}")
        if "uid" not in source:
            raise ValueError("Feature NPZ must contain ordered UID identities")
        if "eligible" not in source:
            raise ValueError("Feature NPZ must include its explicit eligibility mask")
        eligible = source["eligible"]
        if eligible.shape != (len(rows),) or eligible.dtype != np.dtype(bool) or not eligible.all():
            raise ValueError("Eligibility must be bool[N] and all true; stop rather than filtering invalid rows")
    for name, array in (("image", image), ("text", text)):
        if not np.isfinite(array).all() or (np.linalg.norm(array, axis=1) <= 1e-10).any():
            raise ValueError(f"Invalid or all-zero {name} features; do not silently exclude after selection")
    indices, participant_sets = require_indices(manifest, rows)
    # Matches upstream embedding-loader preprocessing before the aligner's own
    # train-mean subtraction and normalization. No PCA is fitted.
    return unit(image).astype(np.float32), unit(text).astype(np.float32), rows, indices, participant_sets


def duplicate_audit(text, rows, indices):
    result = {}
    for split, selected in (("fit", indices["paired_train"]), ("dev", indices["dev"]), ("test", indices["test"])):
        selected_rows = [rows[i] for i in selected]
        narrations = [normalized_text(row["narration"]) for row in selected_rows]
        _, inverse, counts = np.unique(text[selected], axis=0, return_inverse=True, return_counts=True)
        grouped_nouns, grouped_actions = defaultdict(set), defaultdict(set)
        for group, row in zip(inverse, selected_rows):
            grouped_nouns[int(group)].add(str(row["noun_class"]))
            grouped_actions[int(group)].add((str(row["verb_class"]), str(row["noun_class"])))
        result[split] = {
            "rows": len(selected), "unique_normalized_narrations": len(set(narrations)),
            "duplicate_narration_rows_beyond_first": len(selected) - len(set(narrations)),
            "unique_float32_pooled_text_vectors": len(counts),
            "duplicate_vector_rows_beyond_first": len(selected) - len(counts),
            "largest_identical_vector_group": int(counts.max()),
            "identical_vector_groups_with_multiple_noun_classes": sum(len(v) > 1 for v in grouped_nouns.values()),
            "identical_vector_groups_with_multiple_action_classes": sum(len(v) > 1 for v in grouped_actions.values()),
        }
    return result


def derangement(size, rng):
    if size < 2:
        raise ValueError("A shuffled paired control needs at least two observations")
    order = rng.permutation(size)
    fixed = np.flatnonzero(order == np.arange(size))
    if len(fixed) == 1:
        other = (int(fixed[0]) + 1) % size
        order[fixed[0]], order[other] = order[other], order[fixed[0]]
    elif len(fixed) > 1:
        order[fixed] = np.roll(order[fixed], 1)
    assert not np.any(order == np.arange(size))
    return order


def fit_map(method, seed, image, text, rows, indices, args):
    import torch
    from unpaired_rosetta import WassersteinProcrustes
    from unpaired_rosetta.linalg import polar

    if method == "rosetta_disjoint":
        x_ids, y_ids = indices["train_x"].copy(), indices["train_y"].copy()
    elif method == "random_map":
        x_ids, y_ids = indices["train_x"].copy(), indices["train_y"].copy()
    else:
        x_ids = indices["shared_train" if method == "rosetta_shared" else "paired_train"].copy()
        y_ids = x_ids.copy()
    rng = np.random.default_rng(seed + 7919)
    if method.startswith("rosetta"):
        x_ids, y_ids = rng.permutation(x_ids), rng.permutation(y_ids)
    elif method.startswith("shuffled"):
        people = np.asarray([rows[i]["participant_id"] for i in x_ids])
        for person in sorted(set(people.tolist())):
            positions = np.flatnonzero(people == person)
            y_ids[positions] = y_ids[positions][derangement(len(positions), rng)]
    train_x = torch.from_numpy(image[x_ids])
    train_y = torch.from_numpy(text[y_ids])
    params = {"num_clusters": args.clusters, "num_restarts": args.restarts,
              "num_iterations": args.iterations,
              "batch_size": min(args.batch_size, len(x_ids), len(y_ids))}
    if method.startswith("rosetta"):
        if params["num_clusters"] > min(len(x_ids), len(y_ids)):
            raise ValueError("More clusters than training points")
        model = WassersteinProcrustes(**params, num_workers=args.workers,
                                     num_threads=args.threads, verbose=True)
        # Only these two feature matrices and a seed reach the upstream API.
        model.fit(train_x, train_y, randomness=seed)
        mean_x, mean_y, weight = (t.detach().numpy() for t in (model.mean_x, model.mean_y, model.weight))
    else:
        mean_x, mean_y = image[x_ids].mean(axis=0, keepdims=True), text[y_ids].mean(axis=0, keepdims=True)
        x, y = unit(image[x_ids] - mean_x), unit(text[y_ids] - mean_y)
        if method == "random_map":
            matrix = torch.from_numpy(rng.standard_normal((image.shape[1], text.shape[1])))
            weight = polar(matrix).numpy()
        elif "ridge" in method:
            xtx = torch.from_numpy(x.T @ x)
            xty = torch.from_numpy(x.T @ y)
            weight = torch.linalg.solve(xtx + args.ridge_alpha * torch.eye(x.shape[1], dtype=xtx.dtype), xty).numpy()
        else:
            weight = polar(torch.from_numpy(x.T @ y)).numpy()
    return {
        "weight": np.asarray(weight), "mean_x": np.asarray(mean_x), "mean_y": np.asarray(mean_y),
        "train_x_indices": x_ids, "train_y_indices": y_ids,
        "normalize_projected": np.asarray("ridge" in method),
        "input_unit_normalized": np.asarray(True),
    }, params


def score_map(model, image, text):
    # float64 evaluation keeps duplicate texts tied exactly and makes map replay
    # independent of torch's fit precision. Rosetta uses its original bilinear
    # score; ridge predictions are row normalized before cosine scoring.
    projected = unit(image - model["mean_x"]) @ model["weight"]
    if bool(model["normalize_projected"]):
        projected = unit(projected)
    return projected @ unit(text - model["mean_y"]).T


def probe_diagnostics(image, text, rows, indices, args, output_dir):
    import torch
    result, arrays = {}, {}
    fit_ids = indices["paired_train"]
    for modality, features in (("image", image), ("text", text)):
        result[modality] = {}
        mean = features[fit_ids].mean(axis=0, keepdims=True)
        fit_x = unit(features[fit_ids] - mean)
        xtx = torch.from_numpy(fit_x.T @ fit_x)
        for label_key in ("noun_class", "verb_class"):
            labels = np.asarray([str(r[label_key]) for r in rows])
            classes = np.unique(labels[fit_ids])
            y = (labels[fit_ids, None] == classes[None, :]).astype(np.float64)
            intercept = y.mean(axis=0)
            weight = torch.linalg.solve(xtx + args.ridge_alpha * torch.eye(xtx.shape[0], dtype=xtx.dtype),
                                        torch.from_numpy(fit_x.T @ (y - intercept))).numpy()
            prefix = f"{modality}__{label_key}"
            arrays.update({f"{prefix}__mean": mean, f"{prefix}__weight": weight,
                           f"{prefix}__intercept": intercept, f"{prefix}__classes": classes})
            result[modality][label_key] = {}
            for split in ("dev", "test"):
                selected = indices[split]
                scores = unit(features[selected] - mean) @ weight + intercept
                target = labels[selected]
                covered = np.isin(target, classes)
                per = np.zeros((len(selected), 2), dtype=np.float64)
                for i, label in enumerate(target):
                    if covered[i]:
                        value = scores[i, classes == label][0]
                        greater, tied = int((scores[i] > value).sum()), int((scores[i] == value).sum())
                        for j, k in enumerate((1, 5)):
                            per[i, j] = safe_recall(greater, tied, 1, k)
                classwise = [per[target == label, 0].mean() for label in sorted(set(target.tolist()))]
                participants = np.asarray([rows[i]["participant_id"] for i in selected])
                result[modality][label_key][split] = {
                    "top1": float(per[:, 0].mean()), "top5": float(per[:, 1].mean()),
                    "balanced_top1_all_test_classes": float(np.mean(classwise)),
                    "training_class_coverage": float(covered.mean()),
                    "num_train_classes": len(classes), "num_evaluated_classes": len(set(target.tolist())),
                    "macro_participant_top1": float(np.mean([per[participants == p, 0].mean()
                                                            for p in sorted(set(participants.tolist()))])),
                    "majority_class_top1": float((target == classes[int(np.argmax(intercept))]).mean()),
                }
                arrays.update({f"{prefix}__{split}__scores": scores,
                               f"{prefix}__{split}__targets": target,
                               f"{prefix}__{split}__top1": per[:, 0],
                               f"{prefix}__{split}__top5": per[:, 1]})
    path = output_dir / "probe_arrays.npz"
    np.savez_compressed(path, **arrays)
    write_json(output_dir / "probes.json", {
        "description": "Supervised fixed-alpha ridge classifiers on each frozen modality; competence diagnostics only",
        "train_indices": fit_ids.tolist(), "ridge_alpha": args.ridge_alpha,
        "results": result, "arrays": {"file": path.name, "sha256": sha256(path)},
    })
    return result


def aggregate(runs, rows, indices, output_dir, bootstrap_draws=10000):
    """Average optimization seeds first, then bootstrap independent participants."""
    results = {}
    for method in sorted(set(r["method"] for r in runs)):
        selected_runs = [r for r in runs if r["method"] == method]
        by_p = defaultdict(list)
        chance = {}
        for run in selected_runs:
            metric = run["metrics"]["test"]["image_to_text"]["noun"]
            for participant, measures in metric["by_participant"].items():
                by_p[participant].append(measures["recall_at_1"])
                chance[participant] = measures["chance_at_1"]
        participants = sorted(by_p)
        means = np.asarray([np.mean(by_p[p]) for p in participants])
        rng = np.random.default_rng(65537)
        samples = rng.integers(len(means), size=(bootstrap_draws, len(means)))
        boot = means[samples].mean(axis=1)
        results[method] = {
            "seeds": [r["seed"] for r in selected_runs],
            "seed_macro_recall_at_1": [r["metrics"]["test"]["image_to_text"]["noun"]["macro_participant"]["recall_at_1"]
                                       for r in selected_runs],
            "macro_participant_recall_at_1": float(means.mean()),
            "participant_bootstrap_95_interval": [float(np.quantile(boot, .025)), float(np.quantile(boot, .975))],
            "chance_macro_recall_at_1": float(np.mean(list(chance.values()))),
            "participants": participants, "participant_seed_averaged_recall_at_1": means.tolist(),
        }
    comparisons = {}
    primary = results.get("rosetta_disjoint")
    if primary:
        for baseline in ("rosetta_shared", "paired_orthogonal", "paired_ridge", "shuffled_orthogonal", "random_map"):
            if baseline not in results:
                continue
            other = results[baseline]
            if primary["participants"] != other["participants"]:
                raise ValueError("Inconsistent evaluation participants across methods")
            differences = np.asarray(primary["participant_seed_averaged_recall_at_1"]) - other["participant_seed_averaged_recall_at_1"]
            rng = np.random.default_rng(65537)
            samples = rng.integers(len(differences), size=(bootstrap_draws, len(differences)))
            boot = differences[samples].mean(axis=1)
            comparisons[f"rosetta_disjoint_minus_{baseline}"] = {
                "difference": float(differences.mean()),
                "participant_bootstrap_95_interval": [float(np.quantile(boot, .025)), float(np.quantile(boot, .975))],
            }
    return {
        "primary_metric": "held-out test image_to_text/noun/macro_participant/recall_at_1",
        "primary_method": "rosetta_disjoint", "results": results, "comparisons": comparisons,
        "bootstrap": {"draws": bootstrap_draws, "seed": 65537,
                      "unit": "participant; optimization seeds averaged before resampling",
                      "limitation": "only the selected held-out kitchens; no checkpoint or population-wide causal inference"},
    }


def self_test():
    # Exact duplicate relevance, ties cutting the retrieval boundary, and ties
    # above a positive are failure-prone distinctions in this dataset.
    assert safe_recall(0, 4, 2, 1) == .5
    assert math.isclose(safe_recall(0, 4, 2, 2), 5 / 6)
    assert safe_recall(2, 4, 2, 1) == 0
    matrix = np.asarray([[.9, .9, .1], [.9, .9, .1], [.1, .1, .9]])
    labels = np.asarray(["a", "a", "b"])
    duplicate = retrieval_per_query(matrix, labels, labels)
    np.testing.assert_array_equal(duplicate["recall_at_1"], np.ones(3))
    strict = retrieval_per_query(matrix, np.arange(3), np.arange(3))
    np.testing.assert_allclose(strict["recall_at_1"], [.5, .5, 1])
    np.testing.assert_allclose(strict["expected_first_rank"], [1.5, 1.5, 1])
    np.testing.assert_allclose(duplicate["chance_at_1"], [2/3, 2/3, 1/3])
    # A common column permutation with labels must preserve every metric.
    permutation = [2, 0, 1]
    changed = retrieval_per_query(matrix[:, permutation], labels, labels[permutation])
    for key in duplicate:
        np.testing.assert_allclose(duplicate[key], changed[key])
    for n in (2, 3, 17):
        for seed in range(20):
            order = derangement(n, np.random.default_rng(seed))
            assert set(order.tolist()) == set(range(n)) and not np.any(order == np.arange(n))
    input_gate_self_test()
    print(json.dumps({"self_test": "passed", "checks": ["duplicate relevance", "tie probability",
                     "strict-pair diagnostic", "gallery permutation invariance", "derangement",
                     "final provenance and eligibility gates"]}), flush=True)


def input_gate_self_test():
    """Reject partial/stale inputs even when their array shape looks complete."""
    with tempfile.TemporaryDirectory(prefix="epic-input-gate-") as temporary:
        directory = Path(temporary)
        rows = []
        for i in range(8):
            split = "fit" if i < 4 else "dev" if i < 6 else "test"
            role = "x" if i < 2 else "y" if i < 4 else ""
            rows.append({"uid": str(i), "split": split, "fit_role": role,
                         "participant_id": split, "video_id": split + role,
                         "narration": "a cup", "noun_class": 1, "verb_class": 2})
        manifest_path, feature_path = directory / "manifest.json", directory / "features.npz"
        write_json(manifest_path, {"metadata": {}, "rows": rows})
        args = argparse.Namespace(features=feature_path, manifest=manifest_path)
        arrays = {"image": np.ones((8, 768), dtype=np.float32), "text": np.ones((8, 300), dtype=np.float32),
                  "uid": np.asarray([r["uid"] for r in rows]), "eligible": np.ones(8, dtype=bool)}

        def save(mask=arrays["eligible"]):
            candidate = dict(arrays)
            if mask is None:
                candidate.pop("eligible")
            else:
                candidate["eligible"] = mask
            np.savez(feature_path, **candidate)
            metadata = {"feature_file_sha256": sha256(feature_path), "manifest_sha256": sha256(manifest_path),
                        "selection": {"is_final": True, "allow_partial": False, "selected_rows": 8,
                                      "output_rows": 8, "missing_rows": []},
                        "eligibility": {"eligible": 8, "ineligible": 0}}
            write_json(feature_path.with_suffix(".json"), metadata)
            return metadata

        def rejects():
            try:
                load_inputs(args)
            except ValueError:
                return
            raise AssertionError("Unsafe input unexpectedly passed the final-input gate")

        save()
        assert load_inputs(args)[0].shape == (8, 768)
        for key, value in (("is_final", False), ("allow_partial", True), ("output_rows", 7),
                           ("selected_rows", 7), ("missing_rows", [{"uid": "0"}])):
            metadata = save()
            metadata["selection"][key] = value
            write_json(feature_path.with_suffix(".json"), metadata)
            rejects()
        for key in ("feature_file_sha256", "manifest_sha256"):
            metadata = save()
            metadata[key] = "0" * 64
            write_json(feature_path.with_suffix(".json"), metadata)
            rejects()
        for mask in (None, np.asarray([False] + [True] * 7), np.ones(8, dtype=np.int32), np.ones((8, 1), dtype=bool)):
            save(mask)
            rejects()
        save()
        feature_path.with_suffix(".json").unlink()
        rejects()


def synthetic_check(args):
    """Known planted geometry, disjoint training observations, unseen test rows.

    Uses the real 768/300 dimensions. It checks that the implementation can
    recover a favorable case; its small 12/5/10/1000 settings are solely for
    implementation validation and a rough runtime observation.
    """
    import torch
    from unpaired_rosetta import WassersteinProcrustes
    from unpaired_rosetta.linalg import polar
    from unpaired_rosetta.randomness import Randomness
    from unpaired_rosetta.evaluation import disjoint_split
    from unpaired_rosetta.determinism import enable_deterministic_algorithms

    self_test()
    enable_deterministic_algorithms()
    torch.set_num_threads(args.threads)
    generator = torch.Generator().manual_seed(0)
    centers = 3 * torch.randn(12, 16, generator=generator)
    latent = centers[torch.randint(0, 12, (4000,), generator=generator)] + torch.randn(4000, 16, generator=generator)
    embed_x = polar(torch.randn(16, 768, generator=generator))
    embed_y = polar(torch.randn(16, 300, generator=generator))
    image = torch.nn.functional.normalize(latent @ embed_x, dim=1)
    text = torch.nn.functional.normalize(latent @ embed_y, dim=1)
    randomness = Randomness(3)
    x_ids, y_ids = disjoint_split(3000, randomness)
    settings = dict(num_clusters=12, num_restarts=5, batch_size=1000, num_iterations=10)
    model = WassersteinProcrustes(**settings, num_workers=args.workers, num_threads=args.threads, verbose=True)
    started = time.monotonic()
    model.fit(image[x_ids], text[y_ids], randomness=randomness)
    fit_seconds = time.monotonic() - started
    scores = model.similarity(image[3000:], text[3000:]).numpy()
    per = retrieval_per_query(scores, np.arange(1000), np.arange(1000))
    foscttm = float(np.mean((per["expected_first_rank"] - 1) / 999))
    rng = np.random.default_rng(19)
    shuffled = rng.permutation(1000)
    destroyed = retrieval_per_query(scores, shuffled, np.arange(1000))
    result = {
        "status": "passed" if foscttm < .02 else "failed", "seed": 0,
        "settings": settings, "fit_seconds": fit_seconds, "train_x": 1500, "train_y": 1500,
        "test_rows": 1000, "dimensions": [768, 300], "paired_training_rows": 0,
        "foscttm": foscttm, "recall_at_1": float(per["recall_at_1"].mean()),
        "shuffled_test_correspondence_recall_at_1": float(destroyed["recall_at_1"].mean()),
        "gate": "held-out FOSCTTM<0.02 for known planted geometry; benchmark outcomes not inspected",
        "runtime_scope": "single smaller synthetic check; full benchmark has 30 restarts,100iterations,2048assignment batch",
    }
    args.output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output / "synthetic_check.npz", scores=scores,
                        train_x_indices=x_ids.numpy(), train_y_indices=y_ids.numpy(),
                        weight=model.weight.numpy(), mean_x=model.mean_x.numpy(), mean_y=model.mean_y.numpy(),
                        test_image=image[3000:].numpy(), test_text=text[3000:].numpy(),
                        test_label_shuffle=shuffled)
    result["artifact_sha256"] = sha256(args.output / "synthetic_check.npz")
    write_json(args.output / "synthetic_check.json", result)
    print(json.dumps(result), flush=True)
    if result["status"] != "passed":
        raise RuntimeError("Synthetic planted-map recovery gate failed")


def run(args):
    import torch
    from unpaired_rosetta.determinism import enable_deterministic_algorithms
    import unpaired_rosetta

    self_test()
    enable_deterministic_algorithms()
    torch.set_num_threads(args.threads)
    image, text, rows, indices, participants = load_inputs(args)
    args.output.mkdir(parents=True, exist_ok=True)
    source_root = Path(unpaired_rosetta.__file__).parent
    upstream_files = {str(p.relative_to(source_root)): sha256(p) for p in sorted(source_root.rglob("*.py"))}
    config = {
        "schema_version": 1, "feature_sha256": sha256(args.features), "manifest_sha256": sha256(args.manifest),
        "feature_metadata_sha256": sha256(args.features.with_suffix(".json")),
        "runner_sha256": sha256(__file__), "upstream_source_sha256": canonical_hash(upstream_files),
        "upstream_source_files": upstream_files, "seeds": args.seeds, "methods": args.methods,
        "clusters": args.clusters, "restarts": args.restarts, "iterations": args.iterations,
        "batch_size": min(args.batch_size, len(indices["train_x"]), len(indices["train_y"])),
        "workers": args.workers, "threads": args.threads, "ridge_alpha": args.ridge_alpha,
        "permutation_draws": args.permutations,
        "software": {name: importlib.metadata.version(name) for name in
                     ("numpy", "torch", "scipy", "scikit-learn", "pylibmgm")},
        "python": sys.version, "platform": platform.platform(), "participants": participants,
        "sample_counts": {key: len(values) for key, values in indices.items()},
        "indices": {key: values.tolist() for key, values in indices.items()},
        "preprocessing": "unit input rows; train-only centering then unit rows; no PCA",
        "relevance": "noun class primary; verb:noun action/verb/narration supplementary; exact UID diagnostic",
        "budget_note": "paired/shared controls use all fit rows; their larger data and pairing budget makes them competence controls, not matched-budget effect estimates",
        "shuffled_pair_definition": "deranged paired text targets within participant; retains participant conditioning",
        "tie_policy": "uniform random tie-breaking expectation, using exact equality of saved float64 scores",
        "selection_policy": "Frozen manifest; dev scores diagnostic only; no hyperparameter selection",
        "duplicate_audit": duplicate_audit(text, rows, indices),
        "arithmetic": "upstream Rosetta fits float32; paired/random maps float64; all evaluation scores float64",
    }
    digest = canonical_hash(config)
    config_path = args.output / "experiment.json"
    if config_path.exists() and json.loads(config_path.read_text())["config_sha256"] != digest:
        raise ValueError("Output directory contains a different experiment configuration; use a new directory")
    write_json(config_path, {"config_sha256": digest, "config": config})
    runs = []
    for method in args.methods:
        # Paired fits are deterministic; avoid reporting copies as independent
        # replications. Optimization/null methods have three seeds.
        seeds = [args.seeds[0]] if method.startswith("paired") else args.seeds
        for seed in seeds:
            name = f"{method}_seed-{seed}"
            result_path = args.output / f"{name}.json"
            if result_path.exists():
                previous = json.loads(result_path.read_text())
                if previous.get("status") == "complete" and previous.get("config_sha256") == digest:
                    for artifact in previous["artifacts"].values():
                        if sha256(args.output / artifact["file"]) != artifact["sha256"]:
                            raise ValueError("Resume artifact hash mismatch")
                    runs.append(previous)
                    print(f"Verified completed {name}", flush=True)
                    continue
            result = {"status": "running", "config_sha256": digest, "method": method, "seed": seed,
                      "started_utc": datetime.now(timezone.utc).isoformat(), "artifacts": {}, "metrics": {}}
            write_json(result_path, result)
            print(f"Fitting {name}", flush=True)
            started = time.monotonic()
            model, params = fit_map(method, seed, image, text, rows, indices, args)
            result["fit_seconds"] = time.monotonic() - started
            result["fit_parameters"] = params if method.startswith("rosetta") else {"ridge_alpha": args.ridge_alpha}
            result["train_rows"] = {"image": len(model["train_x_indices"]), "text": len(model["train_y_indices"])}
            if "ridge" not in method:
                orthogonality_error = float(np.abs(model["weight"].T @ model["weight"] - np.eye(text.shape[1])).max())
                if orthogonality_error > 1e-4:
                    raise ValueError(f"Semiorthogonal map invariant failed: {orthogonality_error}")
                result["max_abs_column_orthogonality_error"] = orthogonality_error
            map_path = args.output / f"{name}.map.npz"
            np.savez_compressed(map_path, **model)
            result["artifacts"]["map"] = {"file": map_path.name, "sha256": sha256(map_path)}
            for split in ("dev", "test"):
                selected = indices[split]
                selected_rows = [rows[i] for i in selected]
                scores = score_map(model, image[selected], text[selected])
                measures, arrays = evaluate(scores, selected_rows)
                result["metrics"][split] = measures
                if split == "test":
                    null, null_arrays = permutation_null(scores, selected_rows, args.permutations)
                    result["permutation_null"] = null
                    arrays.update(null_arrays)
                matrix_path = args.output / f"{name}.{split}.npz"
                np.savez_compressed(matrix_path, scores=scores, row_indices=selected,
                                    uid=np.asarray([str(r["uid"]) for r in selected_rows]), **arrays)
                result["artifacts"][split] = {"file": matrix_path.name, "sha256": sha256(matrix_path)}
            result.update(status="complete", total_seconds=time.monotonic() - started,
                          completed_utc=datetime.now(timezone.utc).isoformat())
            write_json(result_path, result)
            runs.append(result)
            print(json.dumps({"completed": name, "fit_seconds": result["fit_seconds"],
                              "test_noun": result["metrics"]["test"]["image_to_text"]["noun"]["macro_participant"]}), flush=True)
    probes = probe_diagnostics(image, text, rows, indices, args, args.output)
    summary = aggregate(runs, rows, indices, args.output)
    summary.update(status="complete", config_sha256=digest, completed_fits=len(runs),
                   probes=probes, completed_utc=datetime.now(timezone.utc).isoformat())
    write_json(args.output / "summary.json", summary)
    print(json.dumps(summary), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--clusters", type=int, default=30)
    parser.add_argument("--restarts", type=int, default=30)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    parser.add_argument("--permutations", type=int, default=999)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--synthetic-check", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if args.synthetic_check:
        if args.output is None:
            parser.error("--synthetic-check requires --output")
        synthetic_check(args)
        return
    if not all((args.features, args.manifest, args.output)):
        parser.error("--features, --manifest and --output are required")
    if min(args.clusters, args.restarts, args.batch_size, args.workers, args.threads, args.permutations) <= 0:
        parser.error("Counts must be positive")
    if args.iterations < 0 or args.ridge_alpha <= 0 or not args.seeds or len(set(args.seeds)) != len(args.seeds):
        parser.error("Invalid iterations, ridge alpha or seeds")
    try:
        run(args)
    except Exception as error:
        # Preserve a terminal failure record instead of leaving a stale
        # "running" fit after the process exits. Completed arms are retained.
        if args.output.exists():
            for path in args.output.glob("*_seed-*.json"):
                value = json.loads(path.read_text())
                if value.get("status") == "running":
                    value.update(status="failed", error_type=type(error).__name__, error=str(error),
                                 failed_utc=datetime.now(timezone.utc).isoformat())
                    write_json(path, value)
        raise


if __name__ == "__main__":
    main()
