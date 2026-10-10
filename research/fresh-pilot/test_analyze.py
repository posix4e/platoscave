#!/usr/bin/env python3
"""Synthetic checks of the fresh pilot analysis; no human data or model jobs."""

import copy
from contextlib import redirect_stderr, redirect_stdout
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np


SPEC = importlib.util.spec_from_file_location("fresh_pilot_analyze", Path(__file__).with_name("analyze.py"))
analysis = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analysis)


class SyntheticPacket:
    def __init__(self, root):
        self.root = Path(root)
        self.release = self.root / "release"
        self.release.mkdir()
        self.protocol_path = self.release / "protocol.json"
        self.sealed_path = self.root / "sealed.json"
        self.features = self.root / "features.npz"
        self.output = self.root / "analysis"
        protocol = {
            "schema_version": 1, "protocol_id": analysis.PROTOCOL_ID,
            "classes": list(analysis.CLASSES), "analysis": {"ridge_alpha": 1.0},
            "collection": {"fit_rounds": list(analysis.FIT_ROUNDS),
                           "development_rounds": list(analysis.DEVELOPMENT_ROUNDS)},
            "models": {"image": {"checkpoint_sha256": "1" * 64, "source_commit": "abc",
                                   "source_sha256": {"vision_transformer.py": "4" * 64, "utils.py": "5" * 64}},
                       "text": {"archive_sha256": "2" * 64, "member_sha256": "3" * 64}},
        }
        analysis.write_json(self.protocol_path, protocol)
        for name in ("intake.py", "extract.py", "analyze.py"):
            (self.release / name).write_text(f"# Private synthetic release fixture: {name}\n")
        analysis.write_json(self.release / "release-lock.json", {
            "schema_version": 1,
            "files": {name: analysis.sha256(self.release / name) for name in analysis.RELEASE_FILES},
        })
        rng = np.random.default_rng(951)
        rows = []
        for round_id in analysis.ROUNDS:
            for class_index in rng.permutation(len(analysis.CLASSES)):
                label = analysis.CLASSES[class_index]
                sample_id = f"{round_id}-{label}"
                fit = round_id in analysis.FIT_ROUNDS
                rows.append({
                    "sample_id": sample_id, "round_id": round_id,
                    "split": "fit" if fit else "development", "class_name": label,
                    "instance_id": f"{label}-{'A' if fit else 'B'}",
                    "image_file": f"images/{sample_id}.jpg",
                    "captured_at_utc": "2026-10-11T12:00:00+00:00",
                    "caption": f"An ordinary {label}",
                    "captioned_at_utc": "2026-10-11T14:00:00+00:00",
                    "image_sha256": hashlib.sha256(sample_id.encode()).hexdigest(),
                    "image_size_bytes": 1234,
                })
        self.sealed = {
            "schema_version": 1, "protocol_id": analysis.PROTOCOL_ID,
            "protocol_sha256": analysis.sha256(self.protocol_path),
            "release_lock_sha256": analysis.sha256(self.release / "release-lock.json"),
            "intake_sha256": analysis.sha256(self.release / "intake.py"),
            "rows": rows,
        }
        self.write_sealed()
        labels = np.asarray([analysis.CLASSES.index(row["class_name"]) for row in rows])
        image = rng.normal(size=(48, analysis.IMAGE_DIM)) * 0.001
        image[np.arange(48), labels] += 1.0
        text = np.zeros((48, analysis.TEXT_DIM))
        text[np.arange(48), labels] = 1.0
        self.arrays = {
            "image": analysis.l2_rows(image, "fixture").astype(np.float32),
            "text": text.astype(np.float32),
            "sample_ids": np.asarray([row["sample_id"] for row in rows]),
        }
        self.write_features()

    def write_sealed(self):
        analysis.write_json(self.sealed_path, self.sealed)

    def metadata(self):
        with np.load(self.features, allow_pickle=False) as archive:
            ids = archive["sample_ids"].astype(str).tolist()
        return {
            "schema_version": 1, "protocol_id": analysis.PROTOCOL_ID, "status": "complete",
            "sample_count": 48, "split_counts": {"fit": 32, "development": 16},
            "sample_ids": ids,
            "protocol_sha256": analysis.sha256(self.protocol_path),
            "sealed_sha256": analysis.sha256(self.sealed_path),
            "feature_sha256": analysis.sha256(self.features),
            "release_lock_sha256": analysis.sha256(self.release / "release-lock.json"),
            "models_verified": {"image": True, "text": True},
            "extractor_sha256": analysis.sha256(self.release / "extract.py"),
            "model_hashes": {"dino_checkpoint_sha256": "1" * 64,
                             "glove_zip_sha256": "2" * 64, "glove_member_sha256": "3" * 64},
            "source_hashes": {"vision_transformer.py": "4" * 64, "utils.py": "5" * 64},
            "dino_source_commit": "abc",
            "tokens": {"fixture": "No real photographs or model outputs are present"},
        }

    def write_features(self, arrays=None):
        np.savez_compressed(self.features, **(self.arrays if arrays is None else arrays))
        analysis.write_json(self.features.with_suffix(".json"), self.metadata())

    def load(self):
        return analysis.load_inputs(self.features, self.sealed_path, self.protocol_path)

    def run(self):
        return analysis.run(self.features, self.sealed_path, self.protocol_path, self.output)


class AnalysisTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="fresh-pilot-analysis-test-")
        self.addCleanup(self.temp.cleanup)
        self.packet = SyntheticPacket(self.temp.name)
        self.release_patch = mock.patch.object(analysis, "RELEASE_DIR", self.packet.release)
        self.release_patch.start()
        self.addCleanup(self.release_patch.stop)

    def assert_no_output(self):
        self.assertFalse(self.packet.output.exists())
        self.assertEqual(list(self.packet.root.glob(".analysis.stage-*")), [])
        self.assertFalse((self.packet.root / ".analysis.analysis.lock").exists())

    def test_exact_tie_expectation_accepts_both_same_class_captions(self):
        gallery = np.repeat(np.arange(8), 2)
        scores = np.zeros((4, 16))
        scores[0, :2] = 1.0
        scores[1, [0, 2]] = 1.0
        scores[2, :] = 1.0
        scores[3, 0] = 1.0
        scores[3, 2] = np.nextafter(1.0, 2.0)
        hits, counts, correct = analysis.tie_expected_top1(scores, np.zeros(4), gallery)
        np.testing.assert_array_equal(hits, [1.0, 0.5, 0.125, 0.0])
        np.testing.assert_array_equal(counts, [2, 2, 16, 1])
        np.testing.assert_array_equal(correct, [2, 1, 2, 0])
        probe_hits, _, _ = analysis.tie_expected_top1(np.ones((8, 8)), np.arange(8), np.arange(8))
        np.testing.assert_array_equal(probe_hits, np.full(8, 0.125))

    def test_fit_centering_never_uses_development(self):
        fit = np.asarray([0, 1, 2])
        values = np.asarray([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.], [2., -1., 3.]])
        centered, mean = analysis.center_from_fit(values, fit, "test")
        changed = values.copy()
        changed[3] = [-1000., 100., 1e8]
        changed_centered, changed_mean = analysis.center_from_fit(changed, fit, "test")
        np.testing.assert_array_equal(mean, values[fit].mean(axis=0))
        np.testing.assert_array_equal(mean, changed_mean)
        np.testing.assert_array_equal(centered[fit], changed_centered[fit])
        self.assertFalse(np.array_equal(centered[3], changed_centered[3]))

    def test_dual_ridge_matches_primal_solution_and_rectangular_dimensions(self):
        rng = np.random.default_rng(49)
        x, y = rng.normal(size=(4, 7)), rng.normal(size=(4, 3))
        weights = analysis.dual_ridge(x.astype(np.float32), y.astype(np.float32))
        x, y = x.astype(np.float32).astype(np.float64), y.astype(np.float32).astype(np.float64)
        expected = np.linalg.solve(x.T @ x + np.eye(7), x.T @ y)
        self.assertEqual(weights.shape, (7, 3))
        self.assertEqual(weights.dtype, np.float64)
        np.testing.assert_allclose(weights, expected, rtol=1e-12, atol=1e-12)
        with self.assertRaisesRegex(ValueError, "same nonzero row count"):
            analysis.dual_ridge(x, y[:-1])

    def test_round_shuffles_follow_protocol_class_order_and_keep_fixed_points(self):
        rows = self.packet.sealed["rows"]
        fit = np.asarray([i for i, row in enumerate(rows) if row["split"] == "fit"])
        permutations = []
        for seed in analysis.SHUFFLE_SEEDS:
            rng = np.random.default_rng(seed)
            expected = {}
            for round_id in analysis.FIT_ROUNDS:
                source_ids = [f"{round_id}-{label}" for label in analysis.CLASSES]
                expected.update(zip(source_ids, rng.permutation(source_ids)))
            result = analysis.shuffle_within_round(rows, fit, seed)
            actual = {rows[a]["sample_id"]: rows[b]["sample_id"] for a, b in zip(fit, result)}
            self.assertEqual(actual, expected)
            self.assertEqual(set(result), set(fit))
            for a, b in zip(fit, result):
                self.assertEqual(rows[a]["round_id"], rows[b]["round_id"])
            reversed_fit = fit[::-1]
            reverse_result = analysis.shuffle_within_round(rows, reversed_fit, seed)
            self.assertEqual(actual, {rows[a]["sample_id"]: rows[b]["sample_id"]
                                      for a, b in zip(reversed_fit, reverse_result)})
            permutations.append(result)
        self.assertGreater(np.count_nonzero(permutations[0] == fit), 0)
        self.assertFalse(np.array_equal(permutations[0], permutations[1]))

    def test_complete_run_reports_all_six_fits_duplicates_and_every_prediction(self):
        summary = self.packet.run()
        predictions = analysis.read_json(self.packet.output / "predictions.json")
        self.assertEqual(set(summary["methods"]), {
            "image_noun_probe", "text_noun_probe", "paired_ridge",
            "shuffled_ridge_seed_0", "shuffled_ridge_seed_1", "shuffled_ridge_seed_2"})
        self.assertEqual(summary["counts"]["development_physical_instances"], 8)
        self.assertFalse(summary["settings"]["unpaired_alignment_run"])
        self.assertEqual(summary["baselines"]["balanced_random_class_top1"], 0.125)
        for name in ("image_noun_probe", "text_noun_probe", "paired_ridge"):
            self.assertEqual(summary["methods"][name]["noun_top1_expected"], 1.0)
        paired = summary["methods"]["paired_ridge"]
        self.assertEqual(paired["queries_with_top_score_ties"], 16)
        self.assertEqual(paired["maximum_top_tie_size"], 2)
        self.assertEqual(set(paired["per_round"]), {"r05", "r06"})
        self.assertEqual(set(paired["per_class"]), set(analysis.CLASSES))
        for result in predictions["methods"].values():
            self.assertEqual(len(result["rows"]), 16)
            self.assertEqual(len(result["scores"]), 16)
        duplicates = summary["development_caption_duplicates"]
        for key in ("literal_caption_groups", "normalized_caption_groups",
                    "raw_text_vector_groups", "centered_text_vector_groups"):
            self.assertEqual(len(duplicates[key]), 8)
            self.assertTrue(all(group["count"] == 2 for group in duplicates[key]))
        self.assertEqual(summary["predictions_sha256"],
                         analysis.sha256(self.packet.output / "predictions.json"))
        self.assertEqual(summary["feature_metadata"]["tokens"], self.packet.metadata()["tokens"])

    def test_feature_reordering_is_restored_by_id(self):
        expected = self.packet.load()
        order = np.random.default_rng(93).permutation(48)
        self.packet.write_features({key: value[order] for key, value in self.packet.arrays.items()})
        actual = self.packet.load()
        self.assertTrue(actual["feature_row_order_changed"])
        np.testing.assert_array_equal(actual["image"], expected["image"])
        np.testing.assert_array_equal(actual["text"], expected["text"])

    def test_identical_vectors_fail_when_fit_centering_produces_zero(self):
        for modality in ("image", "text"):
            with self.subTest(modality=modality):
                arrays = {key: value.copy() for key, value in self.packet.arrays.items()}
                arrays[modality][:] = 0
                arrays[modality][:, 0] = 1
                self.packet.write_features(arrays)
                with self.assertRaisesRegex(ValueError, "Zero or nonfinite norm"):
                    self.packet.run()
                self.assert_no_output()

    def test_bad_arrays_and_incomplete_or_duplicate_ids_are_rejected(self):
        cases = []
        for name, change in (
            ("duplicate IDs", lambda a: a["sample_ids"].__setitem__(0, a["sample_ids"][1])),
            ("wrong ID", lambda a: a["sample_ids"].__setitem__(0, "r99-mug")),
            ("zero image", lambda a: a["image"].__setitem__(0, 0)),
            ("nan text", lambda a: a["text"].__setitem__((0, 0), np.nan)),
            ("infinite image", lambda a: a["image"].__setitem__((0, 0), np.inf)),
            ("nonunit image", lambda a: a["image"].__setitem__(0, 2 * a["image"][0])),
        ):
            arrays = {key: value.copy() for key, value in self.packet.arrays.items()}
            change(arrays)
            cases.append((name, arrays))
        cases.extend([
            ("short", {key: value[:-1] for key, value in self.packet.arrays.items()}),
            ("bytes IDs", {**self.packet.arrays, "sample_ids": self.packet.arrays["sample_ids"].astype("S")}),
            ("wrong image width", {**self.packet.arrays, "image": self.packet.arrays["image"][:, :-1]}),
            ("float64 text", {**self.packet.arrays, "text": self.packet.arrays["text"].astype(np.float64)}),
            ("unknown array", {**self.packet.arrays, "labels": np.zeros(48)}),
        ])
        for name, arrays in cases:
            with self.subTest(name=name):
                self.packet.write_features(arrays)
                with self.assertRaises(ValueError):
                    self.packet.run()
                self.assert_no_output()

    def test_malformed_sealed_rows_and_instance_leakage_are_rejected(self):
        original = copy.deepcopy(self.packet.sealed)
        changes = {
            "short rows": lambda p: p["rows"].pop(),
            "wrong ID": lambda p: p["rows"][0].update(sample_id="r01-wrong"),
            "instance leakage": lambda p: p["rows"][-1].update(
                instance_id=f"{p['rows'][-1]['class_name']}-A"),
            "wrong split": lambda p: p["rows"][0].update(split="development"),
            "duplicate image": lambda p: p["rows"][0].update(image_sha256=p["rows"][1]["image_sha256"]),
        }
        for name, change in changes.items():
            with self.subTest(name=name):
                self.packet.sealed = copy.deepcopy(original)
                change(self.packet.sealed)
                self.packet.write_sealed()
                self.packet.write_features()
                with self.assertRaises(ValueError):
                    self.packet.run()
                self.assert_no_output()

    def test_provenance_and_release_hash_mismatches_are_rejected(self):
        for field in ("protocol_sha256", "sealed_sha256", "feature_sha256", "release_lock_sha256"):
            with self.subTest(field=field):
                metadata = self.packet.metadata()
                metadata[field] = "0" * 64
                analysis.write_json(self.packet.features.with_suffix(".json"), metadata)
                with self.assertRaisesRegex(ValueError, "mismatch"):
                    self.packet.run()
                self.assert_no_output()
        for modality in ("image", "text"):
            metadata = self.packet.metadata()
            metadata["models_verified"][modality] = False
            analysis.write_json(self.packet.features.with_suffix(".json"), metadata)
            with self.assertRaisesRegex(ValueError, "verification flags"):
                self.packet.run()
        for field, invalid in (("status", "partial"), ("sample_count", 47),
                               ("sample_ids", []), ("extractor_sha256", "0" * 64),
                               ("model_hashes", {}), ("source_hashes", {})):
            with self.subTest(field=field):
                metadata = self.packet.metadata()
                metadata[field] = invalid
                analysis.write_json(self.packet.features.with_suffix(".json"), metadata)
                with self.assertRaises(ValueError):
                    self.packet.run()
                self.assert_no_output()
        self.packet.write_features()
        for field in ("protocol_sha256", "release_lock_sha256"):
            original = self.packet.sealed[field]
            self.packet.sealed[field] = "0" * 64
            self.packet.write_sealed()
            self.packet.write_features()
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                self.packet.run()
            self.packet.sealed[field] = original
        self.packet.write_sealed()
        self.packet.write_features()
        (self.packet.release / "extract.py").write_text("# changed after release\n")
        with self.assertRaisesRegex(ValueError, "Release file hash mismatch: extract.py"):
            self.packet.run()
        self.assert_no_output()

    def test_existing_output_is_preserved_and_concurrent_writer_is_rejected(self):
        self.packet.output.mkdir()
        sentinel = self.packet.output / "keep.txt"
        sentinel.write_text("retain existing output")
        with self.assertRaisesRegex(ValueError, "Output already exists"):
            self.packet.run()
        self.assertEqual(sentinel.read_text(), "retain existing output")
        sentinel.unlink()
        self.packet.output.rmdir()
        lock = self.packet.root / ".analysis.analysis.lock"
        lock.write_text("existing writer")
        with self.assertRaisesRegex(ValueError, "holds the output lock"):
            self.packet.run()
        self.assertEqual(lock.read_text(), "existing writer")

    def test_input_change_during_analysis_does_not_publish_or_leave_staging(self):
        real_analyze = analysis.analyze

        def change_input_after_fit(data):
            result = real_analyze(data)
            with self.packet.sealed_path.open("a") as stream:
                stream.write("\n")
            return result

        with mock.patch.object(analysis, "analyze", side_effect=change_input_after_fit):
            with self.assertRaisesRegex(ValueError, "Inputs changed during analysis"):
                self.packet.run()
        self.assert_no_output()

    def test_cli_failure_preserves_structured_stderr_evidence(self):
        metadata = self.packet.metadata()
        metadata["models_verified"]["image"] = False
        analysis.write_json(self.packet.features.with_suffix(".json"), metadata)
        args = ["analyze.py", "--features", str(self.packet.features), "--sealed", str(self.packet.sealed_path),
                "--protocol", str(self.packet.protocol_path), "--out", str(self.packet.output)]
        stderr, stdout = io.StringIO(), io.StringIO()
        with mock.patch.object(analysis.sys, "argv", args), redirect_stderr(stderr), redirect_stdout(stdout):
            with self.assertRaises(SystemExit) as stopped:
                analysis.main()
        self.assertEqual(stopped.exception.code, 1)
        self.assertEqual(stdout.getvalue(), "")
        record = json.loads(stderr.getvalue())
        self.assertEqual(record["status"], "failed")
        self.assertEqual(record["stage"], "load_and_validate")
        self.assertIn("verification flags", record["error"])
        self.assertEqual(record["input_hashes_observed_after_failure"]["features"],
                         analysis.sha256(self.packet.features))
        self.assert_no_output()


if __name__ == "__main__":
    unittest.main()
