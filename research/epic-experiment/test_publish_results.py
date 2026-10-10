#!/usr/bin/env python3
"""Publication-gate tests; all scores/artifacts below are disposable unit fixtures.

No fixture is a benchmark result. Numerical verification is never bypassed to
produce a publication: subprocess mocks exercise refusal paths only.
"""
from argparse import Namespace
from copy import deepcopy
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import publish_results as pub


HERE = Path(__file__).resolve().parent


def table_fixture():
    aggregates, runs = {}, {}
    for m, (method, seeds) in enumerate(pub.METHOD_SEEDS.items()):
        seed_scores, people_scores = [], {p: [] for p in pub.TEST_PARTICIPANTS}
        for seed in seeds:
            people = {}
            for p, person in enumerate(pub.TEST_PARTICIPANTS):
                score = .04 + .025 * m + .006 * p + .004 * seed * (p + 1)
                people[person] = {"queries": pub.TEST_QUERY_COUNTS[person], "recall_at_1": score,
                                  "chance_at_1": .01 + .003 * p}
                people_scores[person].append(score)
            mean = sum(person["recall_at_1"] for person in people.values()) / 7
            seed_scores.append(mean)
            # Reverse insertion order so tests detect positional mapping errors.
            runs[f"{method}_seed-{seed}"] = {
                "method": method, "seed": seed,
                "metrics": {"test": {"image_to_text": {"noun": {
                    "macro_participant": {"recall_at_1": mean},
                    "by_participant": dict(reversed(list(people.items())))}}}},
                "permutation_null": {"draws": 999, "seed": 104729, "observed": mean,
                                     "null_mean": .025, "null_q025": .01, "null_q975": .04,
                                     "null_tail_fraction": .123},
            }
        mean = sum(seed_scores) / len(seed_scores)
        aggregates[method] = {
            "seeds": seeds, "participants": pub.TEST_PARTICIPANTS,
            "seed_macro_recall_at_1": seed_scores,
            "participant_seed_averaged_recall_at_1": [sum(people_scores[p]) / len(seeds) for p in pub.TEST_PARTICIPANTS],
            "macro_participant_recall_at_1": mean, "chance_macro_recall_at_1": .019,
            # Toy intervals exercise field mapping, not statistical computation.
            "participant_bootstrap_95_interval": [mean - .01, mean + .01],
        }
    comparisons = {}
    for baseline in pub.COMPARISON_BASELINES:
        difference = aggregates["rosetta_disjoint"]["macro_participant_recall_at_1"] - aggregates[baseline]["macro_participant_recall_at_1"]
        comparisons["rosetta_disjoint_minus_" + baseline] = {
            "difference": difference, "participant_bootstrap_95_interval": [difference - .015, difference + .015]}
    checks = {name: True for name in pub.TRUE_CHECKS}
    checks.update(pub.COUNT_CHECKS)
    checks.update(hit_cutoffs_verified=[1, 5, 10],
                  bidirectional_definitions_recomputed_per_split_and_fit=["noun", "verb", "action", "narration", "instance_diagnostic"])
    report = {
        "passed": True, "manifest_sha256": pub.MANIFEST_SHA256, "protocol_sha256": pub.PROTOCOL_SHA256,
        "checks": checks,
        "runs": [{"name": name, "maximum_score_replay_error": 0., "all_dev_test_metrics_recomputed": True,
                  "all_999_null_draws_recomputed": True} for name in sorted(pub.EXPECTED_RUNS)],
        "recomputed_primary_aggregates": aggregates, "recomputed_comparisons": comparisons,
    }
    summary = {"results": deepcopy(aggregates), "comparisons": deepcopy(comparisons)}
    return report, summary, dict(reversed(list(runs.items())))


def file_fixture(root):
    """Invalid NPZ bytes suffice to exercise gates before numerical replay."""
    report, summary, runs = table_fixture()
    args = Namespace(manifest=root / "unit-fixture-manifest.json", protocol=HERE / "protocol.json",
                     runner=HERE / "run_experiment.py", verifier=HERE / "verify_results.py",
                     features=root / "unit-fixture-features.npz", results=root / "unit-fixture-results",
                     verification=root / "unit-fixture-verification.json", upstream_source=root / "source",
                     python=sys.executable, output=root / "must-not-be-published")
    args.results.mkdir()
    args.upstream_source.mkdir()
    args.features.write_bytes(b"UNIT TEST ONLY: not an NPZ or benchmark")
    pub.write_json(args.features.with_suffix(".json"), {"unit_test_fixture": True})
    source = args.upstream_source / "__init__.py"
    source.write_text("# Unit test source; never executed.\n")
    source_files = {source.name: pub.sha256(source)}
    config = {
        "manifest_sha256": pub.MANIFEST_SHA256, "feature_sha256": pub.sha256(args.features),
        "feature_metadata_sha256": pub.sha256(args.features.with_suffix(".json")),
        "runner_sha256": pub.sha256(args.runner), "seeds": [0, 1, 2], "methods": list(pub.METHOD_SEEDS),
        "clusters": 30, "restarts": 30, "iterations": 100, "batch_size": 2048,
        "ridge_alpha": 1., "permutation_draws": 999,
        "sample_counts": {"train_x": 3187, "train_y": 3452, "paired_train": 6639, "shared_train": 6639, "dev": 1127, "test": 1626},
        "participants": {"test": pub.TEST_PARTICIPANTS}, "upstream_source_files": source_files,
        "upstream_source_sha256": pub.json_hash(source_files),
    }
    digest = pub.json_hash(config)
    pub.write_json(args.results / "experiment.json", {"config": config, "config_sha256": digest})
    summary.update(status="complete", completed_fits=17, config_sha256=digest,
                   primary_method="rosetta_disjoint", primary_metric=pub.PRIMARY)
    pub.write_json(args.results / "summary.json", summary)
    for name, run in runs.items():
        run.update(status="complete", config_sha256=digest, artifacts={})
        for key in ("map", "dev", "test"):
            artifact = args.results / f"{name}.{key}.npz"
            artifact.write_bytes(("UNIT TEST ONLY " + name + key).encode())
            run["artifacts"][key] = {"file": artifact.name, "sha256": pub.sha256(artifact)}
        pub.write_json(args.results / (name + ".json"), run)
    probe = args.results / "probe_arrays.npz"
    probe.write_bytes(b"UNIT TEST ONLY: not probe arrays")
    pub.write_json(args.results / "probes.json", {"arrays": {"file": probe.name, "sha256": pub.sha256(probe)}})
    report.update(features_sha256=pub.sha256(args.features), runner_sha256=pub.sha256(args.runner),
                  verifier_sha256=pub.sha256(args.verifier), experiment_config_sha256=digest)
    pub.write_json(args.verification, report)
    return args, report


class TableTests(unittest.TestCase):
    def test_explicit_method_seed_participant_mapping(self):
        report, summary, runs = table_fixture()
        tables = pub.prepare_tables(report, summary, runs)
        self.assertEqual([len(tables[key]) for key in ("participant_scores", "method_summary", "control_comparisons", "diagnostic_nulls")], [119, 7, 5, 17])
        row = next(r for r in tables["participant_scores"] if (r["method"], r["seed"], r["participant_id"]) == ("shuffled_ridge", 2, "P17"))
        self.assertAlmostEqual(row["noun_hit_at_1"], .04 + .025 * 5 + .006 * 4 + .004 * 2 * 5)
        self.assertEqual(row["queries"], 90)
        comparison = next(r for r in tables["control_comparisons"] if r["baseline"] == "paired_ridge")
        expected = report["recomputed_comparisons"]["rosetta_disjoint_minus_paired_ridge"]
        self.assertEqual(comparison["noun_hit_at_1_difference"], expected["difference"])
        self.assertEqual(comparison["ci95_lower"], expected["participant_bootstrap_95_interval"][0])
        text = pub.markdown_report({"tables": tables})
        for phrase in ("not a calibrated p-value", "not fine action recognition", "6,639", "3,187", "3,452", "percentage points", "competence controls"):
            self.assertIn(phrase, text)

    def test_rejects_incomplete_or_unverified_receipts(self):
        original, _, _ = table_fixture()
        changes = [lambda r: r.update(passed=False), lambda r: r["runs"].pop(),
                   lambda r: r["checks"].update(full_score_matrices_replayed=33),
                   lambda r: r["checks"].update(probe_fit_equations_and_priors=False),
                   lambda r: r["recomputed_primary_aggregates"]["paired_ridge"].update(seeds=[0, 1]),
                   lambda r: r["recomputed_primary_aggregates"]["random_map"].update(participant_bootstrap_95_interval=[.4, .1])]
        for change in changes:
            with self.subTest(change=change):
                report = deepcopy(original)
                change(report)
                with self.assertRaises(ValueError):
                    pub.validate_verification(report)

    def test_rejects_mapping_and_summary_corruption(self):
        for mutation in ("participant", "queries", "seed", "interval", "control_interval"):
            with self.subTest(mutation=mutation):
                report, summary, runs = table_fixture()
                run = runs["rosetta_disjoint_seed-0"]
                if mutation == "participant":
                    run["metrics"]["test"]["image_to_text"]["noun"]["by_participant"]["P02"]["recall_at_1"] += .03
                elif mutation == "queries":
                    run["metrics"]["test"]["image_to_text"]["noun"]["by_participant"]["P17"]["queries"] = 256
                elif mutation == "seed":
                    run["seed"] = 1
                elif mutation == "interval":
                    summary["results"]["rosetta_disjoint"]["participant_bootstrap_95_interval"] = []
                else:
                    summary["comparisons"]["rosetta_disjoint_minus_random_map"]["participant_bootstrap_95_interval"][0] += .001
                with self.assertRaises(ValueError):
                    pub.prepare_tables(report, summary, runs)

    def test_rejects_nonfinite_and_duplicate_json_fields(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.json"
            for text in ('{"score": NaN}', '{"score": Infinity}', '{"passed": false, "passed": true}'):
                path.write_text(text)
                with self.assertRaises(ValueError):
                    pub.read_json(path)


class FileGateTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        # Narrations and the real selected manifest are intentionally absent
        # from the public repository. Only this gate fixture uses a temporary
        # manifest and test-scoped hash; production pins remain untouched.
        manifest = self.root / "unit-fixture-manifest.json"
        pub.write_json(manifest, {"unit_test_fixture": True, "rows": []})
        manifest_pin = patch.object(pub, "MANIFEST_SHA256", pub.sha256(manifest))
        manifest_pin.start()
        self.addCleanup(manifest_pin.stop)
        self.args, self.report = file_fixture(self.root)

    def tearDown(self):
        self.temporary.cleanup()

    def test_snapshot_covers_all_result_bytes(self):
        state, _, _, runs = pub.inspect_current_state(self.args, self.report)
        self.assertEqual(len(runs), 17)
        self.assertEqual(len(state["result_file_sha256"]), 72)  # 17*(JSON+3 NPZ), config, summary, probe JSON+NPZ.
        self.assertIn("features", state["input_sha256"])

    def test_rejects_missing_fit_and_altered_artifact(self):
        path = self.args.results / "random_map_seed-2.json"
        original = path.read_bytes()
        path.unlink()
        with self.assertRaisesRegex(ValueError, "17 fit records"):
            pub.inspect_current_state(self.args, self.report)
        path.write_bytes(original)
        artifact = self.args.results / "random_map_seed-2.map.npz"
        artifact.write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "Changed result artifact"):
            pub.inspect_current_state(self.args, self.report)

    def test_rejects_partial_fit_and_stale_feature_hash(self):
        path = self.args.results / "paired_ridge_seed-0.json"
        original = path.read_bytes()
        record = pub.read_json(path)
        record["status"] = "running"
        pub.write_json(path, record)
        with self.assertRaisesRegex(ValueError, "Partial or stale fit"):
            pub.inspect_current_state(self.args, self.report)
        path.write_bytes(original)
        self.args.features.write_bytes(b"changed features")
        with self.assertRaisesRegex(ValueError, "Stale independent verification"):
            pub.inspect_current_state(self.args, self.report)

    def test_rejects_changed_verifier_even_with_matching_receipt(self):
        changed = self.root / "changed_verifier.py"
        changed.write_text(self.args.verifier.read_text() + "\n# changed\n")
        self.args.verifier = changed
        self.report["verifier_sha256"] = pub.sha256(changed)
        with self.assertRaisesRegex(ValueError, "Frozen runner or independent verifier changed"):
            pub.inspect_current_state(self.args, self.report)

    def test_refuses_failed_replay_without_creating_publication(self):
        with patch.object(pub.subprocess, "run", return_value=subprocess.CompletedProcess([], 1)) as invoked:
            with self.assertRaisesRegex(ValueError, "Fresh independent verification failed"):
                pub.publish(self.args)
        self.assertEqual(invoked.call_count, 1)
        self.assertIn(str(self.args.verifier.resolve()), invoked.call_args.args[0])
        self.assertFalse(self.args.output.exists())
        self.assertEqual(list(self.root.glob(".epic-publication-*")), [])

    def test_refuses_concurrent_change_without_creating_publication(self):
        def replay_stub(command, **unused):
            # This mocked successful receipt is ONLY used to test refusal after a
            # byte change. It must never produce a final publication directory.
            target = Path(command[command.index("--output") + 1])
            pub.write_json(target, self.report)
            summary_path = self.args.results / "summary.json"
            summary = pub.read_json(summary_path)
            summary["concurrent_change"] = True
            pub.write_json(summary_path, summary)
            return subprocess.CompletedProcess(command, 0)
        with patch.object(pub.subprocess, "run", side_effect=replay_stub):
            with self.assertRaisesRegex(ValueError, "changed during verification"):
                pub.publish(self.args)
        self.assertFalse(self.args.output.exists())
        self.assertEqual(list(self.root.glob(".epic-publication-*")), [])

    def test_does_not_overwrite_existing_publication(self):
        self.args.output.mkdir()
        sentinel = self.args.output / "preserve.txt"
        sentinel.write_text("existing output")
        with patch.object(pub.subprocess, "run") as invoked:
            with self.assertRaisesRegex(ValueError, "already exists"):
                pub.publish(self.args)
        self.assertEqual(sentinel.read_text(), "existing output")
        invoked.assert_not_called()


if __name__ == "__main__":
    unittest.main()
