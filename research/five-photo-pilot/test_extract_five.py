"""Synthetic software fixtures only; no photographs, model inference, or pilot scores.

These tests exercise the manifest and blinding boundary with invented file bytes
and small, analytically specified vectors padded to the encoder dimensions.
"""

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest import mock

import numpy as np


SOURCE = Path(__file__).with_name("extract_five.py")
SPEC = importlib.util.spec_from_file_location("five_photo_extract_under_test", SOURCE)
extract = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(extract)

LABELS = ["Slide", "Trashcan", "Tree", "Basketball", "Water fountain"]
BLIND_KEYS = {
    "schema_version", "protocol_id", "image_ids", "text_ids",
    "image_distances", "text_distances", "protocol_sha256", "feature_sha256",
}
PROTOCOL_SHA = "a" * 64
FEATURE_SHA = "b" * 64


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


class ManifestTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.images = self.root / "images"
        self.images.mkdir()
        self.path = self.root / "manifest.json"
        self.rows = []
        for index, label in enumerate(LABELS):
            image = self.images / f"synthetic-{index}.jpg"
            image.write_bytes(f"Synthetic software test bytes {index}; not an image".encode())
            self.rows.append({
                "candidate_id": f"fixture-{index}", "image_file": image.name,
                "label_verbatim": label, "image_sha256": digest(image),
            })
        self.manifest = {
            "schema_version": 1, "protocol_id": "five-photo-geometry-v1", "rows": self.rows,
        }
        write_json(self.path, self.manifest)

    def tearDown(self):
        self.temporary.cleanup()

    def validate(self, doc=None):
        if doc is not None:
            write_json(self.path, doc)
        return extract.validate_manifest(self.path, self.images)

    def test_complete_manifest_preserves_order_and_hash_snapshot(self):
        rows, snapshot = self.validate()
        snapshot = {path.resolve(): value for path, value in snapshot.items()}
        self.assertEqual(rows, self.rows)
        self.assertEqual(snapshot[self.path.resolve()], digest(self.path))
        for row in self.rows:
            image = (self.images / row["image_file"]).resolve()
            self.assertEqual(snapshot[image], row["image_sha256"])
        reordered = copy.deepcopy(self.manifest)
        reordered["rows"] = reordered["rows"][::-1]
        rows, _ = self.validate(reordered)
        self.assertEqual(rows, reordered["rows"])

    def test_requires_exactly_five_rows(self):
        for count in (0, 4, 6):
            doc = copy.deepcopy(self.manifest)
            doc["rows"] = (doc["rows"] + doc["rows"][:1])[:count]
            with self.subTest(count=count), self.assertRaises(ValueError):
                self.validate(doc)

    def test_candidate_ids_are_nonempty_strings_and_unique(self):
        for bad in ("", " ", None, 3, True, self.rows[1]["candidate_id"]):
            doc = copy.deepcopy(self.manifest)
            doc["rows"][0]["candidate_id"] = bad
            with self.subTest(value=bad), self.assertRaises(ValueError):
                self.validate(doc)

    def test_labels_are_exact_verbatim_unique_set(self):
        for bad in ("slide", " Slide", "Slide ", "Water Fountain", "Tree", "", None, 4):
            doc = copy.deepcopy(self.manifest)
            doc["rows"][0]["label_verbatim"] = bad
            with self.subTest(value=bad), self.assertRaises(ValueError):
                self.validate(doc)

    def test_rejects_wrong_schema_identity_and_extra_fields(self):
        mutations = [
            {**self.manifest, "schema_version": 2},
            {**self.manifest, "protocol_id": "other-experiment"},
            {**self.manifest, "unrecorded_override": True},
            {key: value for key, value in self.manifest.items() if key != "schema_version"},
        ]
        extra = copy.deepcopy(self.manifest)
        extra["rows"][0]["caption"] = "Unrecorded replacement"
        mutations.append(extra)
        missing = copy.deepcopy(self.manifest)
        del missing["rows"][0]["label_verbatim"]
        mutations.append(missing)
        for index, doc in enumerate(mutations):
            with self.subTest(mutation=index), self.assertRaises(ValueError):
                self.validate(doc)

    def test_bad_hashes_and_changed_image_bytes_fail(self):
        for bad in (None, "", "0" * 63, "z" * 64, "A" * 64, "0" * 64, 1):
            doc = copy.deepcopy(self.manifest)
            doc["rows"][0]["image_sha256"] = bad
            with self.subTest(hash=bad), self.assertRaises(ValueError):
                self.validate(doc)
        (self.images / self.rows[0]["image_file"]).write_bytes(b"changed synthetic bytes")
        with self.assertRaises(ValueError):
            self.validate(self.manifest)

    def test_repeated_paths_or_bytes_fail(self):
        duplicate_path = copy.deepcopy(self.manifest)
        duplicate_path["rows"][1]["image_file"] = self.rows[0]["image_file"]
        duplicate_path["rows"][1]["image_sha256"] = self.rows[0]["image_sha256"]
        with self.assertRaises(ValueError):
            self.validate(duplicate_path)
        first, second = (self.images / row["image_file"] for row in self.rows[:2])
        second.write_bytes(first.read_bytes())
        duplicate_bytes = copy.deepcopy(self.manifest)
        duplicate_bytes["rows"][1]["image_sha256"] = digest(second)
        with self.assertRaises(ValueError):
            self.validate(duplicate_bytes)

    def test_missing_file_and_unsafe_paths_fail(self):
        outside = self.root / "outside.jpg"
        outside.write_bytes((self.images / self.rows[0]["image_file"]).read_bytes())
        (self.images / "link.jpg").symlink_to(outside)
        (self.images / "linked-directory").symlink_to(self.root, target_is_directory=True)
        for bad in (
            "missing.jpg", "../outside.jpg", str(outside), "link.jpg",
            "linked-directory/outside.jpg", "./synthetic-0.jpg", "a//b.jpg",
            "a\\b.jpg", "", None,
        ):
            doc = copy.deepcopy(self.manifest)
            doc["rows"][0]["image_file"] = bad
            with self.subTest(path=bad), self.assertRaises(ValueError):
                self.validate(doc)

    def test_duplicate_json_keys_fail(self):
        encoded = json.dumps(self.manifest)
        self.path.write_text(encoded[:-1] + ', "schema_version": 1}', encoding="utf-8")
        with self.assertRaises(ValueError):
            self.validate()


class HelperImportTests(unittest.TestCase):
    def test_hash_checked_before_helper_execution(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            marker = root / "must-not-exist"
            helper = root / "malicious_fixture.py"
            helper.write_text(
                "from pathlib import Path\n" + f"Path({str(marker)!r}).write_text('executed')\n",
                encoding="utf-8",
            )
            self.assertNotEqual(digest(helper), extract.HELPER_SHA256)
            with self.assertRaises(ValueError):
                extract.load_locked_helper(helper)
            self.assertFalse(marker.exists(), "Untrusted helper executed before hash validation")

    def test_valid_locked_helper_is_imported(self):
        with tempfile.TemporaryDirectory() as temporary:
            helper = Path(temporary) / "synthetic_helper.py"
            helper.write_text("SYNTHETIC_TEST_VALUE = 17\n", encoding="utf-8")
            with mock.patch.object(extract, "HELPER_SHA256", digest(helper)):
                loaded = extract.load_locked_helper(helper)
            self.assertEqual(loaded.SYNTHETIC_TEST_VALUE, 17)


class ReleaseLockTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.protocol = self.root / "protocol.json"
        self.protocol.write_bytes(SOURCE.with_name("protocol.json").read_bytes())
        (self.root / "extract_five.py").write_bytes(SOURCE.read_bytes())
        (self.root / "solve.py").write_text("# Synthetic release-lock fixture only.\n", encoding="utf-8")
        self.lock = {
            "schema_version": 1,
            "files": {name: digest(self.root / name)
                      for name in ("protocol.json", "extract_five.py", "solve.py")},
        }
        write_json(self.root / "release-lock.json", self.lock)

    def tearDown(self):
        self.temporary.cleanup()

    def test_complete_release_and_snapshot(self):
        protocol, snapshot = extract.verify_release(self.protocol)
        snapshot = {path.resolve(): value for path, value in snapshot.items()}
        self.assertEqual(protocol, json.loads(self.protocol.read_text(encoding="utf-8")))
        for name, expected in self.lock["files"].items():
            self.assertEqual(snapshot[(self.root / name).resolve()], expected)
        self.assertEqual(snapshot[(self.root / "release-lock.json").resolve()],
                         digest(self.root / "release-lock.json"))

    def test_mutated_source_or_protocol_fails(self):
        for name in ("protocol.json", "extract_five.py", "solve.py"):
            path = self.root / name
            original = path.read_bytes()
            path.write_bytes(original + b"\n# altered synthetic fixture\n")
            with self.subTest(file=name), self.assertRaises(ValueError):
                extract.verify_release(self.protocol)
            path.write_bytes(original)

    def test_relocking_another_extractor_does_not_validate_running_code(self):
        path = self.root / "extract_five.py"
        path.write_bytes(path.read_bytes() + b"\n# Synthetic change after publication\n")
        self.lock["files"]["extract_five.py"] = digest(path)
        write_json(self.root / "release-lock.json", self.lock)
        with self.assertRaises(ValueError):
            extract.verify_release(self.protocol)

    def test_missing_or_extra_locked_files_fail(self):
        for file_set in (
            {name: value for name, value in self.lock["files"].items() if name != "solve.py"},
            {**self.lock["files"], "unexpected.py": "0" * 64},
        ):
            with self.subTest(files=list(file_set)), self.assertRaises(ValueError):
                write_json(self.root / "release-lock.json", {**self.lock, "files": file_set})
                extract.verify_release(self.protocol)

    def test_relocked_wrong_protocol_identity_labels_count_or_helper_fails(self):
        original = json.loads(self.protocol.read_text(encoding="utf-8"))
        variants = []
        for field, bad in (("protocol_id", "wrong-protocol"), ("schema_version", 2)):
            doc = copy.deepcopy(original)
            doc[field] = bad
            variants.append(doc)
        for field, bad in (("items", 4), ("human_labels_verbatim", LABELS[::-1])):
            doc = copy.deepcopy(original)
            doc["data"][field] = bad
            variants.append(doc)
        doc = copy.deepcopy(original)
        doc["models"]["helper_sha256"] = "0" * 64
        variants.append(doc)
        for index, doc in enumerate(variants):
            write_json(self.protocol, doc)
            self.lock["files"]["protocol.json"] = digest(self.protocol)
            write_json(self.root / "release-lock.json", self.lock)
            with self.subTest(mutation=index), self.assertRaises(ValueError):
                extract.verify_release(self.protocol)


class BlindDistanceTests(unittest.TestCase):
    def setUp(self):
        self.rows = [
            {"candidate_id": f"private-fixture-{index}", "label_verbatim": label,
             "image_file": f"private-photo-{index}.jpg", "image_sha256": str(index) * 64}
            for index, label in enumerate(LABELS)
        ]
        self.image = np.zeros((5, 768), dtype=np.float32)
        self.image[:, :2] = [(1, 0), (0, 2), (-3, 0), (4, 4), (5, -5)]
        self.text = np.zeros((5, 300), dtype=np.float32)
        self.text[:, :2] = [(0, 4), (2, 2), (3, 0), (4, -4), (-5, 0)]
        self.image_order, self.text_order = list(range(5)), list(range(5))
        random.Random(20261010).shuffle(self.image_order)
        random.Random(20261011).shuffle(self.text_order)

    def build(self, image=None, text=None, rows=None):
        return extract.build_blind_distances(
            self.image if image is None else image,
            self.text if text is None else text,
            self.rows if rows is None else rows,
            np, PROTOCOL_SHA, FEATURE_SHA,
        )

    def test_float64_renormalized_cosine_distances_with_independent_shuffles(self):
        blind, _, _ = self.build()
        c = 1 / np.sqrt(2)
        original = np.array([
            [0, 1, 2, 1-c, 1-c], [1, 0, 1, 1-c, 1+c],
            [2, 1, 0, 1+c, 1+c], [1-c, 1-c, 1+c, 0, 1],
            [1-c, 1+c, 1+c, 1, 0],
        ], dtype=np.float64)
        text_geometry = original[np.ix_([1, 3, 0, 4, 2], [1, 3, 0, 4, 2])]
        self.assertNotEqual(self.image_order, self.text_order)
        for key, expected in (
            ("image_distances", original[np.ix_(self.image_order, self.image_order)]),
            ("text_distances", text_geometry[np.ix_(self.text_order, self.text_order)]),
        ):
            actual = np.asarray(blind[key])
            self.assertEqual(actual.dtype, np.float64)
            np.testing.assert_allclose(actual, expected, atol=1e-14, rtol=0)
            np.testing.assert_array_equal(actual, actual.T)
            np.testing.assert_array_equal(np.diag(actual), np.zeros(5))

    def test_blind_output_has_exact_fields_and_no_private_values(self):
        blind, _, _ = self.build()
        self.assertEqual(set(blind), BLIND_KEYS)
        self.assertEqual(blind["schema_version"], 1)
        self.assertEqual(blind["protocol_id"], "five-photo-geometry-v1")
        self.assertEqual(blind["protocol_sha256"], PROTOCOL_SHA)
        self.assertEqual(blind["feature_sha256"], FEATURE_SHA)
        self.assertEqual(blind["image_ids"], [f"I{index:02}" for index in range(5)])
        self.assertEqual(blind["text_ids"], [f"T{index:02}" for index in range(5)])
        serialized = json.dumps(blind, allow_nan=False)
        for row in self.rows:
            for field in ("candidate_id", "label_verbatim", "image_file", "image_sha256"):
                self.assertNotIn(row[field], serialized)

    def test_private_truth_retains_actual_pairing_and_verbatim_labels(self):
        blind, truth, private = self.build()
        image_ids, text_ids = blind["image_ids"], blind["text_ids"]
        expected_mapping = {
            image_ids[image_position]: text_ids[self.text_order.index(row_index)]
            for image_position, row_index in enumerate(self.image_order)
        }
        self.assertEqual(truth["mapping"], expected_mapping)
        self.assertEqual(truth["labels"], {
            text_ids[position]: LABELS[row_index] for position, row_index in enumerate(self.text_order)
        })
        self.assertEqual(truth["image_labels"], {
            image_ids[position]: LABELS[row_index] for position, row_index in enumerate(self.image_order)
        })
        self.assertEqual(private["image_shuffle_seed"], 20261010)
        self.assertEqual(private["text_shuffle_seed"], 20261011)
        self.assertEqual(private["image_manifest_indices"], self.image_order)
        self.assertEqual(private["text_manifest_indices"], self.text_order)
        self.assertEqual(private["image_candidate_ids"],
                         [self.rows[index]["candidate_id"] for index in self.image_order])
        self.assertEqual(private["text_candidate_ids"],
                         [self.rows[index]["candidate_id"] for index in self.text_order])

    def test_rejects_incomplete_rows_and_bad_provenance_hashes(self):
        with self.assertRaises(ValueError):
            self.build(rows=self.rows[:4])
        for bad in (None, "", "a" * 63, "g" * 64, "A" * 64):
            for protocol_hash, feature_hash in ((bad, FEATURE_SHA), (PROTOCOL_SHA, bad)):
                with self.subTest(protocol=protocol_hash, feature=feature_hash), self.assertRaises(ValueError):
                    extract.build_blind_distances(
                        self.image, self.text, self.rows, np, protocol_hash, feature_hash,
                    )

    def test_repeatable_without_mutating_arrays_rows_or_global_random_state(self):
        image_before, text_before = self.image.copy(), self.text.copy()
        rows_before = copy.deepcopy(self.rows)
        random_before = random.getstate()
        first, second = self.build(), self.build()
        self.assertEqual(json.dumps(first, sort_keys=True), json.dumps(second, sort_keys=True))
        np.testing.assert_array_equal(self.image, image_before)
        np.testing.assert_array_equal(self.text, text_before)
        self.assertEqual(self.rows, rows_before)
        self.assertEqual(random.getstate(), random_before)

    def test_rejects_wrong_vector_shapes(self):
        for image, text in (
            (self.image[:4], self.text), (self.image, self.text[:4]),
            (self.image[:, :767], self.text), (self.image, self.text[:, :299]),
            (self.image[0], self.text),
        ):
            with self.subTest(image_shape=image.shape, text_shape=text.shape), self.assertRaises(ValueError):
                self.build(image=image, text=text)

    def test_rejects_zero_and_nonfinite_vectors(self):
        for side in ("image", "text"):
            for bad in (0, np.nan, np.inf, -np.inf):
                values = getattr(self, side).copy()
                values[0] = bad
                with self.subTest(side=side, value=bad), self.assertRaises(ValueError):
                    self.build(**{side: values})


if __name__ == "__main__":
    unittest.main()
