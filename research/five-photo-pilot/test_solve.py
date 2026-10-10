#!/usr/bin/env python3
"""Small synthetic tests; no photographs, models, or human answers are used."""

import copy
import importlib.util
import itertools
import math
from pathlib import Path
import random
import tempfile
import unittest
from unittest import mock


SPEC = importlib.util.spec_from_file_location("five_photo_solve", Path(__file__).with_name("solve.py"))
solver = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(solver)


def generic_distances():
    rng = random.Random(1486)
    vectors = [[rng.uniform(-1, 1) for _ in range(4)] for _ in range(5)]
    vectors = [[x / math.sqrt(sum(a * a for a in vector)) for x in vector] for vector in vectors]
    matrix = [[0.0] * 5 for _ in range(5)]
    for i, j in itertools.combinations(range(5), 2):
        matrix[i][j] = matrix[j][i] = 1 - math.fsum(a * b for a, b in zip(vectors[i], vectors[j]))
    return matrix


class SolverTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="five-photo-solver-test-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.release = self.root / "release"
        self.release.mkdir()
        solver.write_json(self.release / "protocol.json", {"protocol_id": solver.PROTOCOL_ID})
        for name in ("solve.py", "extract_five.py"):
            (self.release / name).write_text(f"# Private synthetic fixture for {name}\n")
        solver.write_json(self.release / "release-lock.json", {
            "schema_version": 1, "files": {name: solver.sha256(self.release / name) for name in solver.LOCKED_FILES}})
        patch = mock.patch.object(solver, "SOURCE_DIR", self.release)
        patch.start()
        self.addCleanup(patch.stop)
        self.permutation = (3, 0, 4, 1, 2)
        image = generic_distances()
        text = [[0.0] * 5 for _ in range(5)]
        for i in range(5):
            for j in range(5):
                text[self.permutation[i]][self.permutation[j]] = image[i][j]
        self.data = {
            "schema_version": 1, "protocol_id": solver.PROTOCOL_ID,
            "image_ids": [f"I{i:02d}" for i in range(5)], "text_ids": [f"T{i:02d}" for i in range(5)],
            "image_distances": image, "text_distances": text,
            "protocol_sha256": solver.sha256(self.release / "protocol.json"), "feature_sha256": "a" * 64,
        }
        self.input = self.root / "blind.json"
        self.output = self.root / "solution"
        self.answers = self.root / "answers.json"
        self.evaluation = self.root / "evaluation.json"
        self.mapping = {image: self.data["text_ids"][self.permutation[i]]
                        for i, image in enumerate(self.data["image_ids"])}
        solver.write_json(self.input, self.data)
        solver.write_json(self.answers, {"mapping": self.mapping,
                                        "labels": {i: f"private label {i}" for i in self.data["text_ids"]},
                                        "image_labels": {i: f"private label {i}" for i in self.data["image_ids"]}})

    def test_recovers_known_generic_permutation_and_enumerates_all_120(self):
        result = solver.solve(self.input, self.output)
        records = solver.read_json(self.output / "assignments.json")["assignments"]
        self.assertEqual(len(records), 120)
        self.assertEqual({tuple(r["text_indices"]) for r in records}, set(itertools.permutations(range(5))))
        self.assertEqual(result["tied_count"], 1)
        best = result["best_assignments"][0]
        self.assertEqual(best["text_indices"], list(self.permutation))
        self.assertEqual(best["mapping"], self.mapping)
        self.assertEqual(best["cost"], 0.0)
        self.assertAlmostEqual(best["distance_correlation"], 1.0)
        self.assertGreater(result["gap_to_first_distinct_cost"], 0)
        self.assertEqual((self.output / "blind-distances.json").read_bytes(), self.input.read_bytes())
        receipt = solver.read_json(self.output / "checksums.json")
        self.assertEqual(receipt["selected.json"], solver.sha256(self.output / "selected.json"))
        self.assertNotIn("private label", (self.output / "selected.json").read_text())

    def test_equal_distances_keep_all_ties_and_mean_one_correct_match(self):
        matrix = [[0.0 if i == j else 0.75 for j in range(5)] for i in range(5)]
        self.data["image_distances"] = self.data["text_distances"] = matrix
        solver.write_json(self.input, self.data)
        selection = solver.solve(self.input, self.output)
        self.assertEqual(selection["tied_count"], 120)
        self.assertIsNone(selection["gap_to_first_distinct_cost"])
        self.assertTrue(all(r["distance_correlation"] is None for r in selection["best_assignments"]))
        evaluation = solver.evaluate(self.output, self.answers, self.evaluation)
        self.assertEqual(evaluation["mean_correct_out_of_5"], 1.0)
        self.assertEqual(evaluation["mean_accuracy"], 0.2)
        self.assertEqual(evaluation["correct_out_of_5_range"], [0, 5])
        self.assertEqual(evaluation["true_assignment_rank_interval"], [1, 120])
        self.assertEqual(evaluation["true_cost_tied_count"], 120)

    def test_evaluation_correctness_and_original_selection_unchanged(self):
        solver.solve(self.input, self.output)
        before = {p.name: solver.sha256(p) for p in self.output.iterdir()}
        result = solver.evaluate(self.output, self.answers, self.evaluation)
        self.assertEqual(result["mean_correct_out_of_5"], 5.0)
        self.assertEqual(result["correct_out_of_5_range"], [5, 5])
        self.assertEqual(result["true_cost"], 0.0)
        self.assertEqual(result["true_assignment_rank_interval"], [1, 1])
        self.assertTrue(result["true_mapping_present_in_all_120"])
        self.assertTrue(result["true_mapping_is_optimal"])
        self.assertEqual(result["random_individual_match_reference"], 0.2)
        self.assertEqual(before, {p.name: solver.sha256(p) for p in self.output.iterdir()})

    def test_truth_rank_is_not_assumed_optimal_and_uses_cost_tie_interval(self):
        solver.solve(self.input, self.output)
        wrong = dict(zip(self.data["image_ids"], self.data["text_ids"]))
        solver.write_json(self.answers, wrong)
        result = solver.evaluate(self.output, self.answers, self.evaluation)
        actual_cost = sum((self.data["image_distances"][i][j] - self.data["text_distances"][i][j]) ** 2
                          for i, j in itertools.combinations(range(5), 2)) / 10
        self.assertAlmostEqual(result["true_cost"], actual_cost, places=15)
        all_costs = sorted(r["cost"] for r in solver.read_json(self.output / "assignments.json")["assignments"])
        positions = [i + 1 for i, cost in enumerate(all_costs) if abs(cost - result["true_cost"]) <= solver.TOLERANCE]
        self.assertEqual(result["true_assignment_rank_interval"], [min(positions), max(positions)])
        self.assertFalse(result["true_mapping_is_optimal"])
        self.assertEqual(result["mean_correct_out_of_5"], 0.0)

    def test_identifiers_cannot_change_cost_or_selection(self):
        original = solver.enumerate_assignments(self.data)
        renamed = copy.deepcopy(self.data)
        renamed["image_ids"] = ["aardvark", "zz", "answer", "0", "x"]
        renamed["text_ids"] = ["0", "aardvark", "x", "answer", "zz"]
        changed = solver.enumerate_assignments(renamed)
        self.assertEqual([(r["text_indices"], r["cost"], r["optimal"]) for r in original],
                         [(r["text_indices"], r["cost"], r["optimal"]) for r in changed])

    def test_rejects_extra_answer_fields_bad_ids_and_invalid_distances(self):
        changes = {
            "labels": lambda x: x.update(labels=["hidden answer"]),
            "ground truth": lambda x: x.update(ground_truth=self.mapping),
            "missing key": lambda x: x.pop("feature_sha256"),
            "duplicate IDs": lambda x: x["image_ids"].__setitem__(0, x["image_ids"][1]),
            "short IDs": lambda x: x["text_ids"].pop(),
            "nonsquare": lambda x: x["image_distances"][0].pop(),
            "nan": lambda x: x["image_distances"][0].__setitem__(1, float("nan")),
            "infinity": lambda x: x["image_distances"][0].__setitem__(1, float("inf")),
            "string distance": lambda x: x["image_distances"][0].__setitem__(1, "0.2"),
            "out of range": lambda x: x["image_distances"][0].__setitem__(1, 2.01),
            "negative": lambda x: x["image_distances"][0].__setitem__(1, -0.01),
            "asymmetry": lambda x: x["image_distances"][0].__setitem__(1, 0.23456),
            "diagonal": lambda x: x["text_distances"][2].__setitem__(2, 0.01),
            "wrong schema": lambda x: x.update(protocol_id="another"),
            "invalid hash": lambda x: x.update(feature_sha256="invalid"),
        }
        for name, change in changes.items():
            with self.subTest(name=name):
                value = copy.deepcopy(self.data)
                change(value)
                with self.assertRaises(ValueError):
                    solver.enumerate_assignments(value)

    def test_existing_outputs_are_not_overwritten(self):
        self.output.mkdir()
        sentinel = self.output / "retain.txt"
        sentinel.write_text("existing")
        with self.assertRaisesRegex(ValueError, "already exists"):
            solver.solve(self.input, self.output)
        self.assertEqual(sentinel.read_text(), "existing")
        sentinel.unlink()
        self.output.rmdir()
        solver.solve(self.input, self.output)
        self.evaluation.write_text("existing evaluation")
        with self.assertRaisesRegex(ValueError, "already exists"):
            solver.evaluate(self.output, self.answers, self.evaluation)
        self.assertEqual(self.evaluation.read_text(), "existing evaluation")

    def test_release_or_blind_protocol_mismatch_is_rejected(self):
        self.data["protocol_sha256"] = "0" * 64
        solver.write_json(self.input, self.data)
        with self.assertRaisesRegex(ValueError, "protocol hash mismatch"):
            solver.solve(self.input, self.output)
        (self.release / "extract_five.py").write_text("changed code")
        with self.assertRaisesRegex(ValueError, "Release source hash mismatch"):
            solver.solve(self.input, self.output)
        self.assertFalse(self.output.exists())

    def test_invalid_finalized_selection_rejected_before_private_answers_open(self):
        solver.solve(self.input, self.output)
        selected = self.output / "selected.json"
        selected.write_text(selected.read_text() + "\n")
        self.answers.unlink()
        with self.assertRaisesRegex(ValueError, "Finalized selection checksum mismatch"):
            solver.evaluate(self.output, self.answers, self.evaluation)
        self.assertFalse(self.evaluation.exists())

    def test_incomplete_or_nonbijective_ground_truth_is_rejected(self):
        solver.solve(self.input, self.output)
        bad_maps = [dict(list(self.mapping.items())[:-1]),
                    {image: "T00" for image in self.data["image_ids"]}]
        for mapping in bad_maps:
            solver.write_json(self.answers, {"mapping": mapping})
            with self.assertRaisesRegex(ValueError, "one-to-one"):
                solver.evaluate(self.output, self.answers, self.evaluation)
            self.assertFalse(self.evaluation.exists())


if __name__ == "__main__":
    unittest.main()
