"""Software tests only: generated fixtures are never evidence of human capture."""
from __future__ import annotations

import csv
from datetime import datetime, timedelta, timezone
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
import unittest

from PIL import Image


SOURCE = Path(__file__).with_name("intake.py")
spec = importlib.util.spec_from_file_location("fresh_intake_under_test", SOURCE)
intake = importlib.util.module_from_spec(spec)
spec.loader.exec_module(intake)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def write_csv(path, fields, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


class IntakeTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="fresh-intake-SOFTWARE-TEST-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.release = self.root / "release"
        self.release.mkdir()
        self.protocol = self.release / "protocol.json"
        write_json(self.protocol, {"schema_version": 1, "protocol_id": intake.PROTOCOL_ID,
                                  "classes": list(intake.CLASSES), "collection": {
                                      "photographs": 48, "rounds": 6, "fit_count": 32, "development_count": 16,
                                      "fit_rounds": list(intake.ROUND_IDS[:4]), "development_rounds": list(intake.ROUND_IDS[4:])},
                                  "fixture_only": "SOFTWARE TEST. Not a collection protocol or camera evidence."})
        shutil.copyfile(SOURCE, self.release / "intake.py")
        for name in ("extract.py", "analyze.py"):
            (self.release / name).write_text("# SOFTWARE TEST FIXTURE ONLY: no experiment code.\n")
        write_json(self.release / "release-lock.json", {"schema_version": 1,
                   "files": {name: intake.digest(self.release / name) for name in sorted(intake.LOCK_FILES)}})
        self.packet = self.root / "packet"
        self.output = self.root / "sealed.json"
        self.plan = intake.prepare(self.packet, "software-fixture-only", self.protocol)

    def populate(self):
        """Fill all 54 image slots with visibly synthetic, unique PNG fixtures."""
        start = (datetime.now(timezone.utc) - timedelta(days=1)).replace(hour=12, minute=0, second=0, microsecond=0)
        self.plan["created_at_utc"] = start.isoformat()
        write_json(self.packet / "session-plan.json", self.plan)
        rounds = intake.read_csv(self.packet / "rounds.csv", intake.ROUND_FIELDS)
        samples = intake.read_csv(self.packet / "samples.csv", intake.SAMPLE_FIELDS)
        by_id = {row["sample_id"]: row for row in samples}
        image_index = 0
        for round_index, round_plan in enumerate(self.plan["rounds"]):
            row = rounds[round_index]
            round_start = start + timedelta(minutes=10 * (round_index + 1))
            row.update(started_at_utc=round_start.isoformat(), marker_file=f"markers/{row['round_id']}.png",
                       notes="SOFTWARE TEST FIXTURE ONLY. No human collection, handwritten nonce, or real object is claimed.")
            Image.new("RGB", (224, 225), (image_index, 127, 240 - image_index)).save(self.packet / row["marker_file"])
            image_index += 1
            for order, name in enumerate(round_plan["order"]):
                sample_id = f"{row['round_id']}-{name}"
                sample = by_id[sample_id]
                sample.update(image_file=f"images/{sample_id}.png", captured_at_utc=(round_start + timedelta(seconds=order + 1)).isoformat(),
                              caption="Software fixture caption only.")
                Image.new("RGB", (224, 225), (image_index, 127, 240 - image_index)).save(self.packet / sample["image_file"])
                image_index += 1
        for order, sample_id in enumerate(self.plan["caption_order"]):
            by_id[sample_id]["captioned_at_utc"] = (start + timedelta(minutes=70, seconds=order)).isoformat()
        write_csv(self.packet / "rounds.csv", intake.ROUND_FIELDS, rounds)
        write_csv(self.packet / "samples.csv", intake.SAMPLE_FIELDS, samples)
        return rounds, samples

    def seal(self):
        return intake.validate(self.packet, self.output, attest_human_capture=True, protocol=self.protocol)

    def refuse(self, pattern):
        with self.assertRaisesRegex(intake.IntakeError, pattern):
            self.seal()
        self.assertFalse(self.output.exists(), "Refused packets must not create a sealed output")

    def save_samples(self, rows):
        write_csv(self.packet / "samples.csv", intake.SAMPLE_FIELDS, rows)

    def test_prepare_is_empty_and_has_complete_randomized_schedule(self):
        self.assertEqual(len(self.plan["rounds"]), 6)
        self.assertEqual(len(set(row["nonce"] for row in self.plan["rounds"])), 6)
        self.assertEqual(len(set(self.plan["caption_order"])), 48)
        for row in self.plan["rounds"]:
            self.assertEqual(set(row["order"]), set(intake.CLASSES))
            self.assertRegex(row["nonce"], r"^[0-9a-f]{16}$")
        self.assertEqual(list((self.packet / "images").iterdir()), [])
        self.assertEqual(list((self.packet / "markers").iterdir()), [])
        samples = intake.read_csv(self.packet / "samples.csv", intake.SAMPLE_FIELDS)
        self.assertEqual(len(samples), 48)
        self.assertTrue(all(row["caption"] == row["captured_at_utc"] == row["captioned_at_utc"] == "" for row in samples))
        self.assertEqual((self.packet / "protocol.json").read_bytes(), self.protocol.read_bytes())
        sheet = (self.packet / "CAPTURE-SHEETS.txt").read_text()
        self.assertIn("ONLY AFTER ALL 48", sheet)
        self.assertTrue(all(row["nonce"] in sheet for row in self.plan["rounds"]))
        self.assertTrue(all(sample_id in sheet for sample_id in self.plan["caption_order"]))
        with self.assertRaisesRegex(intake.IntakeError, "already exists"):
            intake.prepare(self.packet, "fixture", self.protocol)

    def test_valid_packet_seals_in_plan_order_without_modifying_originals(self):
        self.populate()
        originals = {str(path): intake.digest(path) for path in self.packet.rglob("*") if path.is_file()}
        sealed = self.seal()
        self.assertEqual(len(sealed["rows"]), 48)
        self.assertEqual(sum(row["split"] == "fit" for row in sealed["rows"]), 32)
        self.assertEqual(sum(row["split"] == "development" for row in sealed["rows"]), 16)
        self.assertEqual([row["sample_id"] for row in sealed["rows"]],
                         [f"{r['round_id']}-{name}" for r in self.plan["rounds"] for name in r["order"]])
        self.assertEqual(len({row["image_sha256"] for row in sealed["rows"]}), 48)
        self.assertEqual(len(sealed["rounds"]), 6)
        self.assertTrue(sealed["attestation"]["marker_photos_inspected"])
        self.assertIn("does not", sealed["evidence_limits"])
        self.assertEqual(originals, {str(path): intake.digest(path) for path in self.packet.rglob("*") if path.is_file()})
        for name, record in sealed["input_records"].items():
            self.assertEqual(record["sha256"], intake.digest(self.packet / name))
            self.assertEqual(record["size_bytes"], (self.packet / name).stat().st_size)

    def test_attestation_and_no_overwrite(self):
        self.populate()
        with self.assertRaisesRegex(intake.IntakeError, "attestation"):
            intake.validate(self.packet, self.output, protocol=self.protocol)
        self.assertFalse(self.output.exists())
        self.seal()
        original = self.output.read_bytes()
        with self.assertRaisesRegex(intake.IntakeError, "overwritten"):
            self.seal()
        self.assertEqual(self.output.read_bytes(), original)

    def test_missing_or_duplicate_sample_ids_refused(self):
        _, rows = self.populate()
        self.save_samples(rows[:-1])
        self.refuse("48 unique")
        rows[-1]["sample_id"] = rows[0]["sample_id"]
        self.save_samples(rows)
        self.refuse("48 unique")

    def test_round_counts_and_splits_instances_refused(self):
        rounds, rows = self.populate()
        write_csv(self.packet / "rounds.csv", intake.ROUND_FIELDS, rounds[:-1])
        self.refuse("six round IDs")
        write_csv(self.packet / "rounds.csv", intake.ROUND_FIELDS, rounds)
        rows[-1]["instance_id"] = rows[-1]["class_name"] + "-A"
        self.save_samples(rows)
        self.refuse("physical-instance")
        rows[-1]["instance_id"] = rows[-1]["class_name"] + "-B"
        rows[-1]["split"] = "fit"
        self.save_samples(rows)
        self.refuse("physical-instance")

    def test_paths_cannot_traverse_or_be_absolute(self):
        _, rows = self.populate()
        for unsafe in ("../outside.png", "/tmp/outside.png", "images/../outside.png", "images\\outside.png"):
            rows[0]["image_file"] = unsafe
            self.save_samples(rows)
            self.refuse("path|POSIX")

    def test_symlink_file_and_directory_refused(self):
        _, rows = self.populate()
        original = self.packet / rows[0]["image_file"]
        external = self.root / "external.png"
        original.rename(external)
        original.symlink_to(external)
        self.refuse("Symlink")
        original.unlink()
        external.rename(original)
        (self.packet / "alias").symlink_to(self.packet / "images", target_is_directory=True)
        rows[0]["image_file"] = "alias/" + original.name
        self.save_samples(rows)
        self.refuse("Symlink")

    def test_duplicate_bytes_and_reused_marker_path_refused(self):
        rounds, rows = self.populate()
        first, second = (self.packet / row["image_file"] for row in rows[:2])
        original_second = second.read_bytes()
        second.write_bytes(first.read_bytes())
        self.refuse("Duplicate image bytes")
        second.write_bytes(original_second)
        rows[0]["image_file"] = rounds[0]["marker_file"]
        self.save_samples(rows)
        self.refuse("distinct file paths")

    def test_bad_or_unsupported_or_too_small_images_refused(self):
        _, rows = self.populate()
        path = self.packet / rows[0]["image_file"]
        path.write_bytes(b"not an image")
        self.refuse("Unreadable image")
        Image.new("RGB", (223, 500)).save(path)
        self.refuse("below 224")
        rows[0]["image_file"] = "images/unsupported.heic"
        (self.packet / rows[0]["image_file"]).write_bytes(b"not HEIC either")
        self.save_samples(rows)
        self.refuse("Most Compatible")

    def test_original_jpeg_supported_and_wrong_extension_refused(self):
        _, rows = self.populate()
        old = self.packet / rows[0]["image_file"]
        jpg = old.with_suffix(".JPEG")
        jpg.write_bytes(old.read_bytes())
        rows[0]["image_file"] = str(jpg.relative_to(self.packet))
        self.save_samples(rows)
        self.refuse("does not match")
        Image.new("RGB", (300, 224), (100, 5, 17)).save(jpg, format="JPEG", quality=95)
        seal = self.seal()
        self.assertEqual(seal["rows"][0]["image_format"], "JPEG")

    def test_bad_timezone_and_capture_order_refused(self):
        _, rows = self.populate()
        original = rows[0]["captured_at_utc"]
        rows[0]["captured_at_utc"] = original.replace("+00:00", "")
        self.save_samples(rows)
        self.refuse("ISO UTC")
        rows[0]["captured_at_utc"] = original.replace("+00:00", "+01:00")
        self.save_samples(rows)
        self.refuse("ISO UTC")
        rows[0]["captured_at_utc"] = rows[1]["captured_at_utc"]
        rows[1]["captured_at_utc"] = original
        self.save_samples(rows)
        self.refuse("planned capture order")

    def test_caption_must_follow_last_photo_and_shuffled_caption_order(self):
        _, rows = self.populate()
        by_id = {row["sample_id"]: row for row in rows}
        first = by_id[self.plan["caption_order"][0]]
        original = first["captioned_at_utc"]
        # A timestamp before the final photo; some orderings also put it before
        # this row's own photo. Either chronology error must be rejected.
        latest_photo = max(datetime.fromisoformat(row["captured_at_utc"]) for row in rows)
        first["captioned_at_utc"] = (latest_photo - timedelta(seconds=1)).isoformat()
        self.save_samples(rows)
        self.refuse("timestamp|Timestamp")
        first["captioned_at_utc"] = original
        second = by_id[self.plan["caption_order"][1]]
        first["captioned_at_utc"], second["captioned_at_utc"] = second["captioned_at_utc"], first["captioned_at_utc"]
        self.save_samples(rows)
        self.refuse("caption_order")

    def test_empty_caption_and_bad_csv_columns_refused(self):
        _, rows = self.populate()
        rows[0]["caption"] = " \n\t"
        self.save_samples(rows)
        self.refuse("Empty human caption")
        (self.packet / "samples.csv").write_text("sample_id,unknown\nanything,value\n")
        self.refuse("CSV columns")

    def test_round_notes_required_and_future_captions_refused(self):
        rounds, rows = self.populate()
        original_note = rounds[0]["notes"]
        rounds[0]["notes"] = "  "
        write_csv(self.packet / "rounds.csv", intake.ROUND_FIELDS, rounds)
        self.refuse("Round notes")
        rounds[0]["notes"] = original_note
        write_csv(self.packet / "rounds.csv", intake.ROUND_FIELDS, rounds)
        future = datetime.now(timezone.utc) + timedelta(days=1)
        by_id = {row["sample_id"]: row for row in rows}
        for index, sample_id in enumerate(self.plan["caption_order"]):
            by_id[sample_id]["captioned_at_utc"] = (future + timedelta(seconds=index)).isoformat()
        self.save_samples(rows)
        self.refuse("in the future")

    def test_tampered_protocol_or_source_lock_refused(self):
        self.populate()
        self.protocol.write_text(self.protocol.read_text() + " ")
        self.refuse("Release lock mismatch")
        # Restore the release; independently corrupt the packet copy.
        self.protocol.write_bytes((self.packet / "protocol.json").read_bytes())
        (self.packet / "protocol.json").write_text("{}")
        self.refuse("Packet protocol/release lock")

    def test_missing_lock_and_tampered_other_source_refused(self):
        lock = self.release / "release-lock.json"
        original = lock.read_bytes()
        lock.unlink()
        with self.assertRaisesRegex(intake.IntakeError, "Missing"):
            intake.prepare(self.root / "other", "fixture", self.protocol)
        lock.write_bytes(original)
        (self.release / "extract.py").write_text("# Changed software test fixture\n")
        with self.assertRaisesRegex(intake.IntakeError, "extract.py"):
            intake.prepare(self.root / "other", "fixture", self.protocol)

    def test_duplicate_nonce_and_caption_order_refused(self):
        self.populate()
        nonce = self.plan["rounds"][1]["nonce"]
        self.plan["rounds"][1]["nonce"] = self.plan["rounds"][0]["nonce"]
        write_json(self.packet / "session-plan.json", self.plan)
        self.refuse("nonces")
        self.plan["rounds"][1]["nonce"] = nonce
        self.plan["caption_order"][0] = self.plan["caption_order"][1]
        write_json(self.packet / "session-plan.json", self.plan)
        self.refuse("caption_order")

    def test_csv_order_does_not_change_planned_sealed_order(self):
        rounds, rows = self.populate()
        write_csv(self.packet / "rounds.csv", intake.ROUND_FIELDS, list(reversed(rounds)))
        self.save_samples(list(reversed(rows)))
        sealed = self.seal()
        self.assertEqual([row["sample_id"] for row in sealed["rows"]], [row["sample_id"] for row in rows])


if __name__ == "__main__":
    unittest.main()
