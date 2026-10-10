#!/usr/bin/env python3
"""Focused private-fixture tests for lossless, narration-free feature release.

No experiment artifacts are used or changed. Pinned constants are patched only
inside each temporary fixture; successful fixture exports are never retained.
Run with a Python environment containing NumPy:
    python -m unittest discover -s . -p test_feature_release.py -v
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

import feature_release as release


class FeatureReleaseTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="epic-feature-release-test-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.manifest = self.root / "selected_rows.json"
        self.features = self.root / "features.npz"
        self.sidecar = self.root / "features.json"
        self.extractor = self.root / "fixture_extractor.py"
        self.extractor.write_text("def tokenize(text):\n    return text.lower().split()\n")
        narrations = ["Take cup", "Put mysteryword bowl", "Open café drawer"]
        self.rows = [
            {"uid": str(i + 10), "participant_id": "P01", "video_id": f"P01_{i:02d}",
             "narration": narration, "split": "fit", "noun_class": i,
             "verb_class": i + 1, "start_timestamp": "00:00:01.000",
             "stop_timestamp": "00:00:03.000",
             "frame_paths": [f"frames/P01_{i:02d}/{i + 10}_f{slot}.jpg" for slot in range(4)]}
            for i, narration in enumerate(narrations)
        ]
        self._json(self.manifest, {"rows": self.rows})
        self.constants = mock.patch.multiple(
            release, MANIFEST_SHA256=release.sha256(self.manifest),
            EXTRACTOR_SHA256=release.sha256(self.extractor),
            EXPECTED_ROWS=3, IMAGE_DIM=4, TEXT_DIM=2,
        )
        self.constants.start()
        self.addCleanup(self.constants.stop)
        self.arrays = {key: np.asarray([str(row[key]) for row in self.rows]) for key in release.METADATA_KEYS}
        self.arrays.update(
            image=(np.arange(12, dtype=np.float32).reshape(3, 4) + 1),
            text=(np.arange(6, dtype=np.float32).reshape(3, 2) + 1),
            eligible=np.ones(3, dtype=bool),
            frame_paths=np.asarray([row["frame_paths"] for row in self.rows]),
            text_token_count=np.asarray([2, 3, 3], dtype=np.int64),
            text_known_token_count=np.asarray([2, 2, 3], dtype=np.int64),
        )
        coverage = []
        for row in self.rows:
            tokens = row["narration"].lower().split()
            oov = [word for word in tokens if word == "mysteryword"]
            coverage.append({"uid": row["uid"], "tokens": tokens,
                             "known_tokens": len(tokens) - len(oov),
                             "total_tokens": len(tokens), "oov_tokens": oov})
        self.metadata = {
            "script_sha256": release.EXTRACTOR_SHA256,
            "manifest_sha256": release.MANIFEST_SHA256,
            "feature_file_sha256": "written by _original",
            "feature_file": self.features.name,
            "selection": {"is_final": True, "allow_partial": False,
                          "selected_rows": 3, "output_rows": 3, "missing_rows": []},
            "eligibility": {"eligible": 3, "ineligible": 0},
            "language": {"row_coverage": coverage, "oov_counts": {"mysteryword": 1},
                         "total_tokens": 8, "known_tokens": 7},
            "vision": {"fixture": True, "unchanged_metadata": [1, 2, 3]},
        }
        self._original()

    @staticmethod
    def _json(path, value):
        path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")

    def _original(self, arrays=None, metadata=None):
        np.savez_compressed(self.features, **(self.arrays if arrays is None else arrays))
        sidecar = deepcopy(self.metadata if metadata is None else metadata)
        sidecar["feature_file_sha256"] = release.sha256(self.features)
        self._json(self.sidecar, sidecar)

    def _export(self, name="bundle"):
        output = self.root / name
        receipt = release.export_bundle(self.manifest, self.features, self.extractor, output)
        return output, receipt

    def _assert_no_output_or_staging(self, output):
        self.assertFalse(output.exists())
        self.assertEqual(list(self.root.glob(".epic-feature-export-*")), [])
        self.assertEqual(list(self.root.glob(".epic-feature-restore-*")), [])

    def _assert_export_rejected(self, name="rejected"):
        output = self.root / name
        with self.assertRaises(ValueError):
            release.export_bundle(self.manifest, self.features, self.extractor, output)
        self._assert_no_output_or_staging(output)

    def test_exact_roundtrip_and_narration_fields_withheld(self):
        originals = {path: path.read_bytes() for path in (self.manifest, self.features, self.sidecar, self.extractor)}
        bundle, receipt = self._export()
        self.assertEqual(receipt["status"], release.STATUS)
        self.assertTrue(receipt["roundtrip"]["npz_bytes_identical"])
        self.assertTrue(receipt["roundtrip"]["metadata_bytes_identical"])
        self.assertEqual(set(path.name for path in bundle.iterdir()),
                         {release.FEATURE_PAYLOAD, release.METADATA_PAYLOAD, release.RECEIPT})
        with np.load(bundle / release.FEATURE_PAYLOAD, allow_pickle=False) as payload:
            self.assertNotIn("narration", payload.files)
            self.assertEqual(payload.files, [key for key in self.arrays if key != "narration"])
            for key in payload.files:
                self.assertEqual(payload[key].dtype, self.arrays[key].dtype)
                np.testing.assert_array_equal(payload[key], self.arrays[key])
        template = json.loads((bundle / release.METADATA_PAYLOAD).read_text())
        for row in template["language"]["row_coverage"]:
            self.assertIsNone(row["tokens"])
            self.assertIsNone(row["oov_tokens"])
        restored = self.root / "restored"
        release.restore_bundle(self.manifest, bundle, self.extractor, restored)
        self.assertEqual((restored / self.features.name).read_bytes(), originals[self.features])
        self.assertEqual((restored / self.sidecar.name).read_bytes(), originals[self.sidecar])
        for path, value in originals.items():
            self.assertEqual(path.read_bytes(), value, f"Original changed: {path.name}")

    def test_partial_selection_is_rejected_even_with_complete_arrays(self):
        cases = {"not_final": ("is_final", False), "partial_flag": ("allow_partial", True),
                 "selected_count": ("selected_rows", 2), "output_count": ("output_rows", 2),
                 "missing_rows": ("missing_rows", [{"uid": "10"}])}
        for name, (key, value) in cases.items():
            with self.subTest(case=name):
                changed = deepcopy(self.metadata)
                changed["selection"][key] = value
                self._original(metadata=changed)
                self._assert_export_rejected(name)

    def test_missing_array_rows_are_rejected(self):
        changed = {key: value[:2] for key, value in self.arrays.items()}
        self._original(arrays=changed)
        self._assert_export_rejected()

    def test_short_manifest_is_rejected_even_if_its_hash_is_pinned(self):
        self._json(self.manifest, {"rows": self.rows[:2]})
        with mock.patch.object(release, "MANIFEST_SHA256", release.sha256(self.manifest)):
            self._assert_export_rejected()

    def test_invalid_feature_values_dtype_shape_or_eligibility_are_rejected(self):
        variants = {
            "ineligible": ("eligible", np.asarray([True, False, True])),
            "eligibility_int": ("eligible", np.ones(3, dtype=np.int64)),
            "eligibility_shape": ("eligible", np.ones((3, 1), dtype=bool)),
            "zero_image": ("image", np.zeros((3, 4), dtype=np.float32)),
            "zero_text": ("text", np.zeros((3, 2), dtype=np.float32)),
            "nan_image": ("image", np.full((3, 4), np.nan, dtype=np.float32)),
            "infinite_text": ("text", np.full((3, 2), np.inf, dtype=np.float32)),
            "wrong_dtype": ("image", np.ones((3, 4), dtype=np.float64)),
            "wrong_width": ("text", np.ones((3, 3), dtype=np.float32)),
        }
        for name, (key, value) in variants.items():
            with self.subTest(case=name):
                changed = dict(self.arrays)
                changed[key] = value
                self._original(arrays=changed)
                self._assert_export_rejected(name)

    def test_stale_original_feature_hash_is_rejected(self):
        metadata = json.loads(self.sidecar.read_text())
        metadata["feature_file_sha256"] = "0" * 64
        self._json(self.sidecar, metadata)
        self._assert_export_rejected()

    def test_changed_manifest_or_extractor_is_rejected(self):
        for name, path in (("manifest", self.manifest), ("extractor", self.extractor)):
            with self.subTest(case=name):
                original = path.read_bytes()
                path.write_bytes(original + b"\n")
                try:
                    self._assert_export_rejected(name)
                finally:
                    path.write_bytes(original)

    def test_tampered_feature_or_metadata_payload_is_rejected(self):
        for name in (release.FEATURE_PAYLOAD, release.METADATA_PAYLOAD):
            with self.subTest(payload=name):
                bundle, _ = self._export("bundle-" + name)
                payload = bundle / name
                payload.write_bytes(payload.read_bytes() + b"tampering")
                output = self.root / ("restore-" + name)
                with self.assertRaisesRegex(ValueError, "payload hash/size differs"):
                    release.restore_bundle(self.manifest, bundle, self.extractor, output)
                self._assert_no_output_or_staging(output)

    def test_incorrect_original_sidecar_hash_rejects_restore_after_reconstruction(self):
        bundle, _ = self._export()
        path = bundle / release.RECEIPT
        receipt = json.loads(path.read_text())
        receipt["original"]["metadata_sha256"] = "0" * 64
        self._json(path, receipt)
        output = self.root / "restore-rejected"
        with self.assertRaisesRegex(ValueError, "Restored JSON bytes differ"):
            release.restore_bundle(self.manifest, bundle, self.extractor, output)
        self._assert_no_output_or_staging(output)

    def test_changed_compression_rejects_export_self_roundtrip(self):
        output = self.root / "uncompressed-export"
        uncompressed_writer = np.savez
        with mock.patch.object(release.np, "savez_compressed", side_effect=uncompressed_writer):
            with self.assertRaisesRegex(ValueError, "Restored NPZ bytes differ"):
                release.export_bundle(self.manifest, self.features, self.extractor, output)
        self._assert_no_output_or_staging(output)

    def test_existing_export_or_restore_output_is_never_overwritten(self):
        existing = self.root / "existing-export"
        existing.mkdir()
        sentinel = existing / "keep.txt"
        sentinel.write_text("keep me")
        with self.assertRaisesRegex(ValueError, "Output directory already exists"):
            release.export_bundle(self.manifest, self.features, self.extractor, existing)
        self.assertEqual(sentinel.read_text(), "keep me")
        bundle, _ = self._export()
        restored = self.root / "existing-restore"
        restored.mkdir()
        sentinel = restored / "keep.txt"
        sentinel.write_text("keep me too")
        with self.assertRaisesRegex(ValueError, "Output directory already exists"):
            release.restore_bundle(self.manifest, bundle, self.extractor, restored)
        self.assertEqual(sentinel.read_text(), "keep me too")

    def test_unreviewed_per_row_text_field_is_rejected(self):
        changed = deepcopy(self.metadata)
        changed["language"]["row_coverage"][0]["extra_caption"] = "unreviewed text"
        self._original(metadata=changed)
        self._assert_export_rejected()


if __name__ == "__main__":
    unittest.main()
