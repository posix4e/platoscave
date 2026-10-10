#!/usr/bin/env python3
"""Prepare publication files only from complete, freshly verified EPIC results.

The supplied successful verification must match the actual frozen inputs and
verification source. Because older verification receipts do not hash every
result artifact, this helper reruns the independent verifier, then checks that
all inputs/results stayed byte-identical throughout verification and rendering.
It never fits a map, changes a protocol, deploys a site, or uploads anything.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile


MANIFEST_SHA256 = "d1a34b2ad47a1456a011c6e4f2d6308bae34ebf6aed0c6fb335794ac3dc648be"
PROTOCOL_SHA256 = "656a4fad35609bb753fb5045268e3448e4c71a36bce70a4b20050fd16a547370"
RUNNER_SHA256 = "5806ce2c397d2c1a473e2422a41f850d332f453237dc785fe09d1d077798b26d"
VERIFIER_SHA256 = "5c7ac4df0fe8991b7a3fa7d664c0ff89891f41925d3b444d7de0395e463402a8"
METHOD_SEEDS = {
    "rosetta_disjoint": [0, 1, 2], "rosetta_shared": [0, 1, 2],
    "paired_orthogonal": [0], "paired_ridge": [0],
    "shuffled_orthogonal": [0, 1, 2], "shuffled_ridge": [0, 1, 2],
    "random_map": [0, 1, 2],
}
METHOD_LABELS = {
    "rosetta_disjoint": "Unpaired; separate fitting videos",
    "rosetta_shared": "Unpaired; shared fitting episodes",
    "paired_orthogonal": "Known pairs; orthogonal map",
    "paired_ridge": "Known pairs; ridge map",
    "shuffled_orthogonal": "Shuffled pairs; orthogonal map",
    "shuffled_ridge": "Shuffled pairs; ridge map",
    "random_map": "Random map",
}
PRIMARY = "held-out test image_to_text/noun/macro_participant/recall_at_1"
TEST_PARTICIPANTS = ["P02", "P07", "P08", "P13", "P17", "P22", "P25"]
TEST_QUERY_COUNTS = {person: 90 if person == "P17" else 256 for person in TEST_PARTICIPANTS}
EXPECTED_RUNS = {f"{method}_seed-{seed}" for method, seeds in METHOD_SEEDS.items() for seed in seeds}
COMPARISON_BASELINES = ("rosetta_shared", "paired_orthogonal", "paired_ridge", "shuffled_orthogonal", "random_map")
TRUE_CHECKS = (
    "original_frozen_manifest_and_protocol", "all_9392_feature_UIDs_and_metadata",
    "final_feature_sidecar_hash_and_all_true_eligibility", "exact_text_vector_and_caption_duplicate_audit",
    "participant_disjoint_evaluation", "video_disjoint_primary_fit_marginals",
    "all_map_training_indices_are_correct_fit_rows", "fit_only_means_verified",
    "executed_runner_and_upstream_source_hashes", "identical_features_retain_exact_score_ties",
    "participant_macro_and_per_query_values", "paired_and_shuffled_optimality_equations",
    "probe_fit_equations_and_priors", "probe_top1_top5_unknown_classes_and_coverage",
)
COUNT_CHECKS = {
    "completed_fits": 17, "rosetta_fits_C30_S30_R100_B2048": 6,
    "full_score_matrices_replayed": 34,
    "individual_score_entries_replayed": 17 * (1127**2 + 1626**2),
    "null_draws_recomputed": 17 * 999,
    "bootstrap_draws_per_method_and_comparison": 10000,
    "probe_score_matrices_replayed": 8,
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def read_json(path):
    # Python's permissive JSON parser would otherwise accept NaN and Infinity.
    def reject_constant(value):
        raise ValueError(f"Nonfinite JSON constant {value} in {path}")
    def unique_fields(pairs):
        fields = {}
        for key, value in pairs:
            require(key not in fields, f"Duplicate JSON field {key} in {path}")
            fields[key] = value
        return fields
    return json.loads(Path(path).read_text(), parse_constant=reject_constant, object_pairs_hook=unique_fields)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def number(value, label, low=None, high=None):
    require(isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value), label + " must be finite numeric")
    require(low is None or value >= low, label + " below allowed bound")
    require(high is None or value <= high, label + " above allowed bound")
    return float(value)


def close(actual, expected, label):
    a, e = number(actual, label), number(expected, label)
    require(math.isclose(a, e, rel_tol=3e-12, abs_tol=3e-12), label + " differs from independent verification")


def check_interval(values, label, low, high):
    require(isinstance(values, list) and len(values) == 2, label + " must contain two endpoints")
    number(values[0], label + " lower", low, high)
    number(values[1], label + " upper", low, high)
    require(values[0] <= values[1], label + " is reversed")


def validate_verification(report):
    require(report.get("passed") is True, "Independent verification did not pass")
    require(report.get("manifest_sha256") == MANIFEST_SHA256, "Verification used a different frozen manifest")
    require(report.get("protocol_sha256") == PROTOCOL_SHA256, "Verification used a different frozen protocol")
    checks = report.get("checks", {})
    for name in TRUE_CHECKS:
        require(checks.get(name) is True, "Verification did not establish " + name)
    for name, expected in COUNT_CHECKS.items():
        require(checks.get(name) == expected, "Verification incomplete/different: " + name)
    require(checks.get("hit_cutoffs_verified") == [1, 5, 10], "Retrieval cutoffs not all verified")
    require(set(checks.get("bidirectional_definitions_recomputed_per_split_and_fit", [])) ==
            {"noun", "verb", "action", "narration", "instance_diagnostic"}, "Retrieval definitions not all verified")
    runs = report.get("runs", [])
    require(len(runs) == 17 and {run["name"] for run in runs} == EXPECTED_RUNS, "Verification lacks the exact 17 fits")
    for run in runs:
        require(run.get("all_dev_test_metrics_recomputed") is True and run.get("all_999_null_draws_recomputed") is True,
                "A fit lacks full independent metric/null verification")
        number(run.get("maximum_score_replay_error"), "Score replay error", 0)
    aggregates = report.get("recomputed_primary_aggregates", {})
    require(set(aggregates) == set(METHOD_SEEDS), "Independent method aggregates incomplete")
    for method, seeds in METHOD_SEEDS.items():
        aggregate = aggregates[method]
        require(aggregate["seeds"] == seeds, "Wrong independently verified seeds for " + method)
        require(aggregate["participants"] == TEST_PARTICIPANTS, "Wrong independently verified test participants")
        require(len(aggregate["seed_macro_recall_at_1"]) == len(seeds) and
                len(aggregate["participant_seed_averaged_recall_at_1"]) == 7, "Incomplete independently verified means")
        for key in ("macro_participant_recall_at_1", "chance_macro_recall_at_1"):
            number(aggregate[key], method + "/" + key, 0, 1)
        for value in aggregate["seed_macro_recall_at_1"] + aggregate["participant_seed_averaged_recall_at_1"]:
            number(value, "Independently verified constituent mean", 0, 1)
        close(sum(aggregate["seed_macro_recall_at_1"]) / len(seeds), aggregate["macro_participant_recall_at_1"], "Seed mean")
        close(sum(aggregate["participant_seed_averaged_recall_at_1"]) / 7, aggregate["macro_participant_recall_at_1"], "Participant mean")
        check_interval(aggregate["participant_bootstrap_95_interval"], "Bootstrap interval", 0, 1)
    expected_comparisons = {"rosetta_disjoint_minus_" + method for method in COMPARISON_BASELINES}
    require(set(report.get("recomputed_comparisons", {})) == expected_comparisons, "Independent control comparisons incomplete")
    for comparison in report["recomputed_comparisons"].values():
        number(comparison["difference"], "Control difference", -1, 1)
        check_interval(comparison["participant_bootstrap_95_interval"], "Control difference interval", -1, 1)


def result_artifact(directory, record):
    filename = record["file"]
    require(isinstance(filename, str) and Path(filename).name == filename, "Invalid result artifact filename")
    path = directory / filename
    require(path.is_file() and path.resolve().parent == directory.resolve(), "Missing or external result artifact " + filename)
    require(sha256(path) == record["sha256"], "Changed result artifact " + filename)
    return path


def inspect_current_state(args, verification):
    """Bind a successful report to the exact current input and result files."""
    validate_verification(verification)
    inputs = {
        "manifest": args.manifest, "protocol": args.protocol, "features": args.features,
        "feature_metadata": args.features.with_suffix(".json"), "runner": args.runner,
        "verifier": args.verifier, "supplied_verification": args.verification,
    }
    digests = {name: sha256(path) for name, path in inputs.items()}
    require(digests["manifest"] == MANIFEST_SHA256 and digests["protocol"] == PROTOCOL_SHA256, "Frozen experiment inputs changed")
    require(digests["runner"] == RUNNER_SHA256 and digests["verifier"] == VERIFIER_SHA256, "Frozen runner or independent verifier changed")
    for report_field, digest_name in (("features_sha256", "features"), ("runner_sha256", "runner"), ("verifier_sha256", "verifier")):
        require(verification[report_field] == digests[digest_name], "Stale independent verification: " + report_field)
    experiment_path = args.results / "experiment.json"
    experiment = read_json(experiment_path)
    config = experiment["config"]
    config_hash = json_hash(config)
    require(experiment["config_sha256"] == config_hash == verification["experiment_config_sha256"], "Experiment config differs from independent verification")
    require(config["feature_sha256"] == digests["features"] and config["feature_metadata_sha256"] == digests["feature_metadata"] and
            config["manifest_sha256"] == MANIFEST_SHA256 and config["runner_sha256"] == digests["runner"], "Experiment input/source hash mismatch")
    require(config["seeds"] == [0, 1, 2] and set(config["methods"]) == set(METHOD_SEEDS), "Experiment methods/seeds incomplete")
    require((config["clusters"], config["restarts"], config["iterations"], config["batch_size"], config["ridge_alpha"], config["permutation_draws"]) == (30, 30, 100, 2048, 1.0, 999), "Experiment settings differ")
    require(config["sample_counts"] == {"train_x": 3187, "train_y": 3452, "paired_train": 6639, "shared_train": 6639, "dev": 1127, "test": 1626}, "Experiment sample counts differ")
    require(sorted(config["participants"]["test"]) == TEST_PARTICIPANTS, "Wrong held-out participants")
    require(json_hash(config["upstream_source_files"]) == config["upstream_source_sha256"], "Upstream source table checksum differs")
    source_hashes = {}
    for relative, expected in config["upstream_source_files"].items():
        path = args.upstream_source / relative
        require(path.is_file() and path.resolve().is_relative_to(args.upstream_source.resolve()), "Missing/external upstream source")
        actual = sha256(path)
        require(actual == expected, "Executed upstream source changed: " + relative)
        source_hashes[relative] = actual
    summary = read_json(args.results / "summary.json")
    require(summary["status"] == "complete" and summary["completed_fits"] == 17 and summary["config_sha256"] == config_hash, "Final summary is partial or stale")
    require(summary["primary_method"] == "rosetta_disjoint" and summary["primary_metric"] == PRIMARY, "Unexpected primary endpoint")
    require({path.stem for path in args.results.glob("*_seed-*.json")} == EXPECTED_RUNS, "Exactly 17 fit records required")
    result_paths = {"experiment.json": experiment_path, "summary.json": args.results / "summary.json"}
    runs = {}
    for name in sorted(EXPECTED_RUNS):
        path = args.results / (name + ".json")
        result = read_json(path)
        require(result["status"] == "complete" and result["config_sha256"] == config_hash, "Partial or stale fit " + name)
        require(name == f"{result['method']}_seed-{result['seed']}", "Fit filename/identity differs")
        require(set(result["artifacts"]) == {"map", "dev", "test"}, "Fit artifacts incomplete")
        result_paths[path.name] = path
        for record in result["artifacts"].values():
            artifact = result_artifact(args.results, record)
            result_paths[artifact.name] = artifact
        runs[name] = result
    probe_path = args.results / "probes.json"
    probe = read_json(probe_path)
    artifact = result_artifact(args.results, probe["arrays"])
    result_paths.update({probe_path.name: probe_path, artifact.name: artifact})
    return {"input_sha256": digests, "result_file_sha256": {name: sha256(path) for name, path in sorted(result_paths.items())},
            "upstream_source_sha256": source_hashes}, config, summary, runs


def prepare_tables(verification, summary, runs):
    """Map verified fields by explicit identities; never by dictionary position."""
    validate_verification(verification)
    require(set(runs) == EXPECTED_RUNS, "Publication requires the exact 17 fits")
    require(set(summary["results"]) == set(METHOD_SEEDS), "Publication method set differs")
    participant_rows, method_rows, comparison_rows, null_rows = [], [], [], []
    for method, seeds in METHOD_SEEDS.items():
        aggregate = verification["recomputed_primary_aggregates"][method]
        reported = summary["results"][method]
        for key in ("macro_participant_recall_at_1", "chance_macro_recall_at_1"):
            close(reported[key], aggregate[key], method + "/" + key)
        check_interval(reported["participant_bootstrap_95_interval"], "Reported bootstrap interval", 0, 1)
        for a, b in zip(reported["participant_bootstrap_95_interval"], aggregate["participant_bootstrap_95_interval"]):
            close(a, b, method + "/bootstrap")
        budget = "3,187 images; 3,452 texts" if method in ("rosetta_disjoint", "random_map") else "6,639 images; 6,639 texts"
        method_rows.append({
            "method": method, "label": METHOD_LABELS[method], "seeds": " ".join(map(str, seeds)),
            "fit_budget": budget, "noun_hit_at_1": aggregate["macro_participant_recall_at_1"],
            "ci95_lower": aggregate["participant_bootstrap_95_interval"][0],
            "ci95_upper": aggregate["participant_bootstrap_95_interval"][1],
            "random_ranking_chance_hit_at_1": aggregate["chance_macro_recall_at_1"],
        })
        for position, seed in enumerate(seeds):
            run = runs[f"{method}_seed-{seed}"]
            require(run["method"] == method and run["seed"] == seed, "Detailed fit identity differs")
            metric = run["metrics"]["test"]["image_to_text"]["noun"]
            require(set(metric["by_participant"]) == set(TEST_PARTICIPANTS), "Per-participant scores incomplete")
            close(metric["macro_participant"]["recall_at_1"], aggregate["seed_macro_recall_at_1"][position], "Seed aggregate")
            for person in TEST_PARTICIPANTS:
                value = metric["by_participant"][person]
                require(type(value["queries"]) is int and value["queries"] == TEST_QUERY_COUNTS[person], "Wrong held-out query count")
                participant_rows.append({
                    "method": method, "seed": seed, "participant_id": person, "queries": value["queries"],
                    "noun_hit_at_1": number(value["recall_at_1"], "Participant hit@1", 0, 1),
                    "random_ranking_chance_hit_at_1": number(value["chance_at_1"], "Participant chance", 0, 1),
                })
            null = run["permutation_null"]
            require(null["draws"] == 999 and null["seed"] == 104729, "Null settings differ")
            close(null["observed"], metric["macro_participant"]["recall_at_1"], "Null observed endpoint")
            check_interval([null["null_q025"], null["null_q975"]], "Null quantiles", 0, 1)
            null_rows.append({
                "method": method, "fit_seed": seed, "draws": 999, "permutation_seed": 104729,
                "observed_noun_hit_at_1": number(null["observed"], "Null observed", 0, 1),
                "null_mean": number(null["null_mean"], "Null mean", 0, 1),
                "null_q025": number(null["null_q025"], "Null quantile", 0, 1),
                "null_q975": number(null["null_q975"], "Null quantile", 0, 1),
                "diagnostic_tail_fraction": number(null["null_tail_fraction"], "Null diagnostic tail", 0, 1),
                "interpretation": "diagnostic only; not a calibrated p-value",
            })
        # Ensure the detailed CSV has the same participant/seed mapping as the
        # independently recomputed aggregates, including unequal query counts.
        for person, expected in zip(aggregate["participants"], aggregate["participant_seed_averaged_recall_at_1"]):
            values = [r["noun_hit_at_1"] for r in participant_rows if r["method"] == method and r["participant_id"] == person]
            close(sum(values) / len(values), expected, "Participant seed mean")
    for baseline in COMPARISON_BASELINES:
        key = "rosetta_disjoint_minus_" + baseline
        value = verification["recomputed_comparisons"][key]
        close(summary["comparisons"][key]["difference"], value["difference"], "Control comparison")
        check_interval(summary["comparisons"][key]["participant_bootstrap_95_interval"], "Reported control interval", -1, 1)
        for reported, verified in zip(summary["comparisons"][key]["participant_bootstrap_95_interval"], value["participant_bootstrap_95_interval"]):
            close(reported, verified, "Control comparison interval")
        comparison_rows.append({
            "comparison": key, "baseline": baseline, "baseline_label": METHOD_LABELS[baseline],
            "noun_hit_at_1_difference": value["difference"],
            "ci95_lower": value["participant_bootstrap_95_interval"][0],
            "ci95_upper": value["participant_bootstrap_95_interval"][1],
            "direction": "primary unpaired minus baseline; negative means primary lower",
        })
    require(len(participant_rows) == 119 and len(method_rows) == 7 and len(comparison_rows) == 5 and len(null_rows) == 17, "Publication table completeness failed")
    return {"participant_scores": participant_rows, "method_summary": method_rows,
            "control_comparisons": comparison_rows, "diagnostic_nulls": null_rows}


def percentage(value):
    return f"{100 * value:.2f}%"


def interval(values):
    return percentage(values[0]) + "–" + percentage(values[1])


def points(value):
    return f"{100 * value:+.2f}"


def markdown_report(document):
    tables = document["tables"]
    primary = next(row for row in tables["method_summary"] if row["method"] == "rosetta_disjoint")
    lines = [
        "# What the EPIC-55 test found", "",
        f"The primary score was **{percentage(primary['noun_hit_at_1'])}** "
        f"(95% participant-bootstrap interval {interval([primary['ci95_lower'], primary['ci95_upper']])}). "
        "This is image-to-text noun hit@1, averaged equally across seven held-out participants after averaging solver seeds.", "",
        "The test used frozen original DINOv1 ViT-B/16 and original GloVe 6B. "
        "All 1,626 test clips formed the fixed gallery. Any caption with the same noun class counted as relevant; "
        "exact ties used their expected hit rate. This measures noun correspondence, not fine action recognition or temporal understanding.", "",
        "| Map | Seeds | Primary score | 95% participant interval |",
        "| --- | --- | ---: | ---: |",
    ]
    for row in tables["method_summary"]:
        lines.append(f"| {row['label']} | {row['seeds']} | {percentage(row['noun_hit_at_1'])} | {interval([row['ci95_lower'], row['ci95_upper']])} |")
    lines += ["", f"Random-ranking chance for this gallery and relevance rule was {percentage(primary['random_ranking_chance_hit_at_1'])}. "
              "The random-map control is a different baseline: it retains the learned features and their uneven class distribution.", "",
              "The primary map saw 3,187 image examples and 3,452 text examples from separate fitting videos. "
              "Shared-episode and paired controls each used all 6,639 fitting clips; paired controls also saw their correspondence. "
              "Their larger information budget makes them competence controls, not matched-budget estimates of a causal effect.", "",
              "| Primary minus control | Difference (percentage points) | 95% participant interval (points) |",
              "| --- | ---: | ---: |"]
    for row in tables["control_comparisons"]:
        lines.append(f"| {row['baseline_label']} | {points(row['noun_hit_at_1_difference'])} | {points(row['ci95_lower'])} to {points(row['ci95_upper'])} |")
    lines += ["", "Negative differences mean the primary unpaired map scored below that control.", "",
              "The following diagnostic keeps scores fixed and shuffles noun assignments within participant. "
              "Its tail fraction is **not a calibrated p-value**: neighboring and overlapping actions are dependent.", "",
              "| Primary fit seed | Observed | Shuffle mean | Shuffle 2.5–97.5% | Diagnostic tail fraction |",
              "| --- | ---: | ---: | ---: | ---: |"]
    for row in tables["diagnostic_nulls"]:
        if row["method"] == "rosetta_disjoint":
            lines.append(f"| {row['fit_seed']} | {percentage(row['observed_noun_hit_at_1'])} | {percentage(row['null_mean'])} | {interval([row['null_q025'], row['null_q975']])} | {row['diagnostic_tail_fraction']:.3f} |")
    lines += ["", "All 17 fits and 34 held-out score matrices passed independent replay before this report was written. "
              "The frozen protocol fixes models and endpoints before scoring and uses development scores only as diagnostics. "
              "Intervals resample the seven held-out participants; solver seeds do not count as new participants or new pretrained models.", "",
              "The older documented training corpora exclude these later recorded episodes under the published model histories. "
              "This experiment does not establish absence of shared concepts or ImageNet's human category selection, universal representations, "
              "or contamination's causal effect in the original Rosetta study. A weak unpaired result must be read alongside the paired controls "
              "and the single-modality noun/verb probes.", "",
              "[Machine-readable summary](results-summary.json) · [Participant and seed scores](participant-scores.csv) · "
              "[Method scores and intervals](method-summary.csv) · [Control differences and intervals](control-comparisons.csv) · "
              "[All diagnostic nulls](diagnostic-nulls.csv) · [Independent verification](verification.json) · [Artifact hashes](publication-receipt.json)", ""]
    return "\n".join(lines)


def write_csv(path, rows):
    require(bool(rows), "Refusing an empty publication CSV")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def publish(args):
    require(not args.output.exists(), "Output directory already exists; choose a new directory to avoid stale/partial publication")
    supplied = read_json(args.verification)
    before, config, summary, runs = inspect_current_state(args, supplied)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".epic-publication-", dir=args.output.parent) as scratch:
        staging = Path(scratch)
        fresh_path = staging / "verification.json"
        command = [args.python, str(args.verifier.resolve()), "--manifest", str(args.manifest.resolve()),
                   "--protocol", str(args.protocol.resolve()), "--features", str(args.features.resolve()),
                   "--results", str(args.results.resolve()), "--runner", str(args.runner.resolve()),
                   "--upstream-source", str(args.upstream_source.resolve()), "--output", str(fresh_path)]
        print("Rechecking all result artifacts with the independent verifier before publication.", flush=True)
        completed = subprocess.run(command, stdin=subprocess.DEVNULL, check=False)
        require(completed.returncode == 0 and fresh_path.is_file(), "Fresh independent verification failed; no publication files produced")
        fresh = read_json(fresh_path)
        after, config, summary, runs = inspect_current_state(args, fresh)
        require(before == after, "Inputs/results changed during verification; no publication files produced")
        tables = prepare_tables(fresh, summary, runs)
        document = {
            "schema_version": 1, "status": "complete_and_independently_verified",
            "created_at_utc": datetime.now(timezone.utc).isoformat(), "protocol_id": "epic55-clean-v1",
            "primary_endpoint": PRIMARY, "primary_method": "rosetta_disjoint",
            "dataset": {"fit_clips": 6639, "image_fit_clips_primary": 3187, "text_fit_clips_primary": 3452,
                        "dev_clips": 1127, "test_clips": 1626, "test_participants": TEST_PARTICIPANTS},
            "settings": {key: config[key] for key in ("clusters", "restarts", "iterations", "batch_size", "seeds", "ridge_alpha")},
            "tables": tables, "single_modality_probes": summary["probes"], "duplicate_audit": config["duplicate_audit"],
            "test_retrieval_metrics_by_fit": {
                name: {direction: {relevance: measures["macro_participant"] for relevance, measures in definitions.items()}
                       for direction, definitions in run["metrics"]["test"].items()}
                for name, run in sorted(runs.items())
            },
            "bootstrap": summary["bootstrap"],
            "interpretation": {
                "noun_primary": "Noun-class correspondence; not a fine-action or temporal-understanding claim.",
                "supplementary_metrics": "Both retrieval directions and all frozen relevance definitions are retained as participant-macro summaries per fit. Instance-ID retrieval is only a diagnostic; noun image-to-text hit@1 remains the primary endpoint.",
                "control_budgets": "Shared and paired controls use all 6,639 fit episodes; primary uses 3,187 image and 3,452 text episodes from disjoint videos. Paired controls also use correspondence. These are competence controls, not matched-budget effects.",
                "null_tail_fraction": "Empirical within-participant shuffle diagnostic only, not a calibrated p-value; temporal dependence remains.",
                "uncertainty": "Seven participants resampled after averaging optimization seeds; no independent pretraining-seed replication.",
                "provenance": "Temporal exclusion under documented model histories; shared concepts, similar scenes and human ImageNet category selection remain. No universal or causal contamination claim.",
            },
            "evidence": {"verification_sha256": sha256(fresh_path), "manifest_sha256": MANIFEST_SHA256,
                         "protocol_sha256": PROTOCOL_SHA256, "feature_sha256": before["input_sha256"]["features"],
                         "experiment_config_sha256": fresh["experiment_config_sha256"],
                         "publisher_sha256": sha256(__file__)},
        }
        write_json(staging / "results-summary.json", document)
        (staging / "RESULTS.md").write_text(markdown_report(document))
        for key, filename in (("participant_scores", "participant-scores.csv"), ("method_summary", "method-summary.csv"),
                              ("control_comparisons", "control-comparisons.csv"), ("diagnostic_nulls", "diagnostic-nulls.csv")):
            write_csv(staging / filename, tables[key])
        final_state, _, _, _ = inspect_current_state(args, fresh)
        require(final_state == before, "Inputs/results changed while rendering; publication refused")
        receipt = {"status": "complete", "publisher_sha256": sha256(__file__),
                   "fresh_verification_sha256": sha256(fresh_path), "verified_state": final_state,
                   "publication_files": {path.name: sha256(path) for path in sorted(staging.iterdir()) if path.is_file()},
                   "note": "Hashes bind the independently reverified inputs/results and the publication artifacts. No uploads or deployments performed."}
        write_json(staging / "publication-receipt.json", receipt)
        staging.rename(args.output)
    print(json.dumps({"status": "complete_and_independently_verified", "directory": str(args.output),
                      "completed_fits": 17, "participant_csv_rows": 119}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=Path(__file__).with_name("protocol.json"))
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--verification", type=Path, required=True)
    parser.add_argument("--verifier", type=Path, default=Path(__file__).with_name("verify_results.py"))
    parser.add_argument("--runner", type=Path, default=Path(__file__).with_name("run_experiment.py"))
    parser.add_argument("--upstream-source", type=Path, required=True)
    parser.add_argument("--python", default=sys.executable, help="Python interpreter with NumPy for independent verification")
    parser.add_argument("--output", type=Path, required=True, help="New local output directory; must not already exist")
    args = parser.parse_args()
    publish(args)


if __name__ == "__main__":
    main()
