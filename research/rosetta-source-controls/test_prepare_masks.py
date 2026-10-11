"""Standard-library artificial archives only; never issue real source masks."""

from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zipfile

import prepare_masks as prep


class PreparationTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.root = Path(self.stack.enter_context(tempfile.TemporaryDirectory()))
        coco = [{"cocoid": i, "filepath": split} for split, ids in (("train2014", (20, 10)), ("val2014", (50, 30, 40))) for i in ids]
        paragraphs = [{"image_id": i} for i in (100, 1, 103, 2, 102, 3, 4, 5)]
        vg = [{"image_id": i, "coco_id": {100: 10, 103: 30, 102: 20}.get(i)} for i in (100, 1, 103, 2, 102, 3, 4, 5)]
        self.records = {"caption_datasets.zip": {"images": coco}, "vg_image_data.json.zip": vg, "paragraphs_v1.json.zip": paragraphs}
        self.paths, hashes = {}, {}
        for name, record in self.records.items():
            path = self.root / name
            with zipfile.ZipFile(path, "w") as archive:
                archive.writestr("public/" + prep.MEMBERS[name], json.dumps(record))
            self.paths[name] = str(path)
            hashes[name] = prep.r.digest(path)
        self.rows = prep.masks.source_rows_from_records(coco, paragraphs, vg)
        for name, value in {"ARCHIVES": hashes, "ROW_IDENTITY": self.rows.identity_sha256,
                            "COUNTS": {"train_x": 2, "original_y": 8, "validation": 3},
                            "ARM_COUNTS": {"O": 8, "C": 7, "D": 5, "Rv": 7, "Rp": 5},
                            "STRATA_COUNTS": {"never_known_spc_overlap": 2, "original_spc_overlap": 1, "all": 3}}.items():
            self.stack.enter_context(patch.object(prep.arm, name, value))
        p = {"schema_version": 1, "protocol_id": prep.arm.PROTOCOL_ID, "status": prep.arm.STATUS,
             "upstream_commit": prep.arm.UPSTREAM, "seeds": prep.arm.SEEDS, "counts": prep.arm.COUNTS,
             "arm_counts": prep.arm.ARM_COUNTS, "algorithm_defaults": prep.arm.ALGORITHM,
             "execution": prep.arm.EXECUTION, "source_archive_sha256": hashes, "row_identity_sha256": self.rows.identity_sha256,
             "strata_counts": prep.arm.STRATA_COUNTS, "metric_rules": prep.arm.METRIC_RULES,
             "selection_seeds": {"Rv": [1000, 1001], "Rp": [2000, 2001]},
             "locks": {name: "b" * 64 for name in prep.arm.LOCKS},
             "code_sha256": {name: prep.r.digest(prep.arm.HELPER_PATH if name == "run_cell.py" else Path(prep.__file__).parent / name) for name in prep.arm.CODE}}
        document = {"schema_version": 1, "source_sha256": hashes, "row_identity_sha256": self.rows.identity_sha256, "rows": self.rows.payload()}
        self.expected_rows_bytes = (json.dumps(document, separators=(",", ":")) + "\n").encode()
        p["locks"]["source_rows"] = hashlib.sha256(self.expected_rows_bytes).hexdigest()
        self.protocol = p
        self.args = SimpleNamespace(protocol=self.root / "protocol.json", protocol_sha256="", release_lock=self.root / "release.json",
                                    release_lock_sha256="", bindings=self.root / "private-bindings.json", out=self.root / "prepared",
                                    private_log_dir=self.root / "private")
        self.args.bindings.write_text(json.dumps({"archives": self.paths}))
        self.update_locks()

    def update_locks(self):
        self.args.protocol.write_text(json.dumps(self.protocol))
        self.args.protocol_sha256 = prep.r.digest(self.args.protocol)
        self.release = {"schema_version": 1, "protocol_id": prep.arm.PROTOCOL_ID, "status": prep.arm.STATUS,
                        "protocol_sha256": self.args.protocol_sha256,
                        "code_sha256": {**self.protocol["code_sha256"], "prepare_masks.py": prep.r.digest(prep.__file__)}}
        self.args.release_lock.write_text(json.dumps(self.release))
        self.args.release_lock_sha256 = prep.r.digest(self.args.release_lock)

    def assert_failed(self):
        self.assertTrue((self.args.out / "FAILED.json").is_file())
        self.assertFalse((self.args.out / "COMPLETE.json").exists())
        self.assertTrue((self.args.private_log_dir / "attempt.traceback.txt").is_file())

    def test_independent_join_full_schedule_composition_and_completion(self):
        result = prep.prepare(self.args)
        self.assertEqual((result["prepared_masks"], result["planned_fits"], result["fits"]), (7, 35, 0))
        self.assertEqual((self.args.out / "source_rows.json").read_bytes(), self.expected_rows_bytes)
        bundle = prep.r.read_json(self.args.out / "masks.json")
        self.assertEqual(bundle["strata"], {"never_known_spc_overlap": [1, 2], "original_spc_overlap": [0], "all": [0, 1, 2]})
        self.assertEqual(set(bundle["masks"]), {"O", "C", "D", "Rv-1000", "Rv-1001", "Rp-2000", "Rp-2001"})
        self.assertEqual(bundle["masks"]["C"]["train_y_indices"], [0, 1, 3, 4, 5, 6, 7])
        self.assertEqual(bundle["masks"]["D"]["train_y_indices"], [1, 3, 5, 6, 7])
        schedule = prep.r.read_json(self.args.out / "schedule.json")
        self.assertEqual([entry["seed"] for entry in schedule["runs"][:5]], prep.arm.SEEDS)
        self.assertEqual([entry["mask"] for entry in schedule["runs"][:5]], ["O"] * 5)
        self.assertEqual(len({(entry["mask"], entry["seed"]) for entry in schedule["runs"]}), 35)
        composition = prep.r.read_json(self.args.out / "compositions.json")["masks"]
        self.assertEqual(composition["D"]["source_membership_rows"]["no_coco_mapping"], 5)
        self.assertEqual(composition["O"]["source_membership_rows"]["known_coco_validation"], 1)
        pairs = prep.r.read_json(self.args.out / "intersections.json")["pairs"]
        self.assertEqual(len(pairs), 21)
        cd = next(p for p in pairs if p["left"] == "C" and p["right"] == "D")
        self.assertEqual((cd["shared_rows"], cd["shared_source_groups"]), (5, 5))
        marker = prep.r.read_json(self.args.out / "COMPLETE.json")
        self.assertNotIn("COMPLETE.json", marker["files"])
        for name, expected in marker["files"].items(): self.assertEqual(prep.r.digest(self.args.out / name), expected)
        self.assertNotIn(str(self.root), (self.args.out / "result.json").read_text())
        self.assertFalse((self.args.out / "FAILED.json").exists())

    def test_final_budget_controls_complete_schedule_without_default_count(self):
        self.protocol["selection_seeds"] = {"Rv": list(range(1000, 1010)), "Rp": list(range(2000, 2010))}
        self.update_locks()
        result = prep.prepare(self.args)
        self.assertEqual((result["prepared_masks"], result["planned_fits"]), (23, 115))
        self.assertEqual(len(prep.r.read_json(self.args.out / "intersections.json")["pairs"]), 253)

    def test_draft_protocol_rejected_before_archive_reads_or_mask_construction(self):
        self.protocol["status"] = "draft"
        self.update_locks()
        with patch.object(prep, "rebuild_rows") as read, patch.object(prep, "build_bundle") as build:
            with self.assertRaises(ValueError): prep.prepare(self.args)
            read.assert_not_called(); build.assert_not_called()
        self.assert_failed()

    def test_changed_archive_blocks_all_new_masks(self):
        Path(self.paths["paragraphs_v1.json.zip"]).write_bytes(b"changed")
        with patch.object(prep, "build_bundle") as build:
            with self.assertRaisesRegex(ValueError, "hash differs"): prep.prepare(self.args)
            build.assert_not_called()
        self.assertFalse((self.args.out / "masks.json").exists())
        self.assert_failed()

    def test_wrong_rebuilt_document_bytes_prevent_any_mask_construction(self):
        self.protocol["locks"]["source_rows"] = "0" * 64
        self.update_locks()
        with patch.object(prep, "build_bundle") as build:
            with self.assertRaisesRegex(ValueError, "source_rows.json bytes"): prep.prepare(self.args)
            build.assert_not_called()
        self.assertFalse((self.args.out / "source_rows.json").exists())
        self.assert_failed()

    def test_wrong_identity_rejected_before_canonical_masking(self):
        with patch.object(prep.arm, "ROW_IDENTITY", "f" * 64):
            self.protocol["row_identity_sha256"] = "f" * 64
            self.update_locks()
            with patch.object(prep, "build_bundle") as build:
                with self.assertRaisesRegex(ValueError, "join differs"): prep.prepare(self.args)
                build.assert_not_called()
        self.assert_failed()

    def test_zip_member_name_must_be_unique_and_present(self):
        path = self.root / "ambiguous.zip"
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("a/image_data.json", "[]")
            archive.writestr("b/image_data.json", "[]")
        with self.assertRaisesRegex(ValueError, "exactly one"): prep.read_member(path, "image_data.json")
        with self.assertRaisesRegex(ValueError, "exactly one"): prep.read_member(path, "missing.json")

    def test_outer_release_must_bind_preparation_code_and_protocol(self):
        self.release["code_sha256"]["prepare_masks.py"] = "0" * 64
        self.args.release_lock.write_text(json.dumps(self.release))
        self.args.release_lock_sha256 = prep.r.digest(self.args.release_lock)
        with self.assertRaisesRegex(ValueError, "hash differs"): prep.prepare(self.args)
        self.assert_failed()

    def test_explicit_release_hash_mismatch_is_not_accepted(self):
        self.args.release_lock_sha256 = "0" * 64
        with self.assertRaisesRegex(ValueError, "hash differs"): prep.prepare(self.args)
        self.assert_failed()

    def test_greedy_failure_does_not_publish_partial_bundle_or_retry(self):
        with patch.object(prep.masks, "build_masks", side_effect=ValueError("synthetic unsatisfied whole-group budget")) as build:
            with self.assertRaisesRegex(ValueError, "unsatisfied"): prep.prepare(self.args)
            build.assert_called_once()
        self.assertTrue((self.args.out / "source_rows.json").is_file())
        self.assertFalse((self.args.out / "masks.json").exists())
        self.assert_failed()

    def test_changed_bound_archive_at_end_prevents_complete(self):
        original = prep.build_bundle
        def change_after(*args):
            result = original(*args)
            Path(self.paths["vg_image_data.json.zip"]).write_bytes(b"changed after join")
            return result
        with patch.object(prep, "build_bundle", side_effect=change_after):
            with self.assertRaisesRegex(ValueError, "changed during"): prep.prepare(self.args)
        self.assertTrue((self.args.out / "schedule.json").is_file())
        self.assert_failed()

    def test_existing_output_cannot_be_overwritten(self):
        prep.prepare(self.args)
        original = prep.r.digest(self.args.out / "COMPLETE.json")
        self.args.private_log_dir = self.root / "another-private"
        with self.assertRaises(FileExistsError): prep.prepare(self.args)
        self.assertEqual(prep.r.digest(self.args.out / "COMPLETE.json"), original)


if __name__ == "__main__":
    unittest.main()
