"""Synthetic helper fixtures only; these are not pilot observations or scores."""

import csv
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import zipfile

import numpy as np
from PIL import Image, ImageOps

SPEC = importlib.util.spec_from_file_location("fresh_extract_under_test", Path(__file__).with_name("extract.py"))
extract = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(extract)


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class ExtractionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def packet_fixture(self):
        source, packet = self.root / "release", self.root / "packet"
        source.mkdir()
        packet.mkdir()
        protocol = {"protocol_id": extract.PROTOCOL_ID, "classes": list(extract.CLASSES)}
        write_json(source / "protocol.json", protocol)
        for name in ("intake.py", "extract.py", "analyze.py"):
            (source / name).write_text(f"# Synthetic source fixture: {name}\n")
        lock = {"schema_version": 1, "files": {name: extract.sha256(source / name) for name in extract.LOCKED_FILES}}
        write_json(source / "release-lock.json", lock)
        for name in ("protocol.json", "release-lock.json"):
            (packet / name).write_bytes((source / name).read_bytes())
        plan = {"schema_version": 1, "protocol_id": extract.PROTOCOL_ID,
                "protocol_sha256": extract.sha256(source / "protocol.json"),
                "release_lock_sha256": extract.sha256(source / "release-lock.json"),
                "intake_sha256": extract.sha256(source / "intake.py"),
                "run_id": "synthetic-test-only", "collector_id": "fixture", "rounds": []}
        rows, rounds, recorded_rounds = [], [], []
        for index in range(1, 7):
            round_id = f"r{index:02d}"
            split = "fit" if index < 5 else "development"
            order = list(extract.CLASSES[index - 1:] + extract.CLASSES[:index - 1])
            planned = {"round_id": round_id, "split": split, "nonce": f"fixture-{index}", "order": order}
            plan["rounds"].append(planned)
            marker_name = f"marker-{round_id}.bin"
            (packet / marker_name).write_bytes(f"Synthetic marker bytes {index}".encode())
            recorded = {"round_id": round_id, "marker_file": marker_name,
                        "started_at_utc": "2030-01-01T00:00:00+00:00", "notes": "synthetic fixture"}
            recorded_rounds.append(recorded)
            rounds.append({**planned, **recorded, "marker_sha256": extract.sha256(packet / marker_name),
                           "marker_size_bytes": (packet / marker_name).stat().st_size})
            for label in order:
                sample_id = f"{round_id}-{label}"
                image_name = sample_id + ".bin"
                (packet / image_name).write_bytes(f"Synthetic image bytes {sample_id}".encode())
                rows.append({"sample_id": sample_id, "round_id": round_id, "split": split,
                             "class_name": label, "instance_id": f"{label}-{'A' if index < 5 else 'B'}",
                             "image_file": image_name, "captured_at_utc": "2030-01-01T00:01:00+00:00",
                             "caption": f"A {label}.", "captioned_at_utc": "2030-01-02T00:01:00+00:00"})
        write_json(packet / "session-plan.json", plan)
        write_csv(packet / "rounds.csv", recorded_rounds)
        write_csv(packet / "samples.csv", rows)
        records = {name: {"sha256": extract.sha256(packet / name), "size_bytes": (packet / name).stat().st_size}
                   for name in extract.INPUT_FILES}
        sealed_rows = [{**row, "image_sha256": extract.sha256(packet / row["image_file"]),
                        "image_size_bytes": (packet / row["image_file"]).stat().st_size} for row in rows]
        sealed = {**{key: value for key, value in plan.items() if key != "rounds"},
                  "session_plan_sha256": records["session-plan.json"]["sha256"],
                  "attestation": {"human_capture": True, "human_captions": True, "marker_photos_inspected": True},
                  "rows": sealed_rows, "rounds": rounds, "input_records": records}
        sealed_path = self.root / "sealed.json"
        write_json(sealed_path, sealed)
        return source, packet, sealed_path, sealed

    def test_complete_packet_and_order(self):
        source, packet, sealed_path, expected = self.packet_fixture()
        sealed, snapshot = extract.validate_packet(sealed_path, packet, source)
        self.assertEqual(sealed["rows"], expected["rows"])
        self.assertEqual(len(sealed["rows"]), 48)
        self.assertEqual(snapshot[sealed_path.resolve()], extract.sha256(sealed_path))

    def test_partial_and_wrong_order_packets_fail(self):
        source, packet, path, sealed = self.packet_fixture()
        original = list(sealed["rows"])
        for changed in (original[:-1], list(reversed(original))):
            with self.subTest(rows=len(changed), reversed=len(changed) == 48):
                sealed["rows"] = changed
                write_json(path, sealed)
                with self.assertRaises(ValueError):
                    extract.validate_packet(path, packet, source)

    def test_changed_published_code_fails(self):
        source, packet, path, _ = self.packet_fixture()
        (source / "extract.py").write_text("# Changed after the lock\n")
        with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
            extract.validate_packet(path, packet, source)

    def test_changed_image_bytes_fail(self):
        source, packet, path, sealed = self.packet_fixture()
        (packet / sealed["rows"][0]["image_file"]).write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
            extract.validate_packet(path, packet, source)

    def test_changed_sealed_caption_does_not_override_csv(self):
        source, packet, path, sealed = self.packet_fixture()
        sealed["rows"][0]["caption"] = "Unrecorded replacement caption"
        write_json(path, sealed)
        with self.assertRaisesRegex(ValueError, "original CSV"):
            extract.validate_packet(path, packet, source)

    def test_missing_attestation_fails(self):
        source, packet, path, sealed = self.packet_fixture()
        sealed["attestation"]["human_captions"] = False
        write_json(path, sealed)
        with self.assertRaisesRegex(ValueError, "attestation"):
            extract.validate_packet(path, packet, source)

    def test_paths_reject_traversal_and_symlinks(self):
        packet = self.root / "packet"
        packet.mkdir()
        outside = self.root / "outside.txt"
        outside.write_text("fixture")
        (packet / "link.txt").symlink_to(outside)
        for name in ("../outside.txt", str(outside), "link.txt", "./item", "a//b", "a\\b"):
            with self.subTest(path=name), self.assertRaises(ValueError):
                extract.packet_file(packet, name)

    def test_duplicate_json_keys_fail(self):
        path = self.root / "duplicate.json"
        path.write_text('{"a": 1, "a": 2}')
        with self.assertRaisesRegex(ValueError, "Duplicate JSON key"):
            extract.read_json(path)

    def test_csv_reader_accepts_bom_like_intake(self):
        path = self.root / "bom.csv"
        path.write_text("sample_id,caption\nfixture,A mug.\n", encoding="utf-8-sig")
        self.assertEqual(extract.csv_records(path, "sample_id"),
                         {"fixture": {"sample_id": "fixture", "caption": "A mug."}})

    def test_tokenizer_keeps_repeats_stopwords_and_normalizes(self):
        self.assertEqual(extract.tokenize("Ａ mug, a mug! Can't １２. café"),
                         ["a", "mug", "a", "mug", "can't", "12", "caf"])

    def glove_fixture(self, vectors):
        raw = "".join(word + " " + " ".join(str(float(x)) for x in vector) + "\n"
                      for word, vector in vectors.items()).encode()
        path = self.root / "glove-fixture.zip"
        with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr(extract.GLOVE_MEMBER, raw)
        return path, mock.patch.multiple(extract, GLOVE_ZIP_SHA256=extract.sha256(path),
                                         GLOVE_MEMBER_SHA256=hashlib.sha256(raw).hexdigest())

    def test_text_mean_repeats_stopwords_oov_and_bag_collisions(self):
        a, mug = np.zeros(300), np.zeros(300)
        a[0], mug[1] = 1, 1
        path, hashes = self.glove_fixture({"a": a, "mug": mug})
        rows = [{"sample_id": "one", "caption": "A mug mug unknown"},
                {"sample_id": "two", "caption": "mug a mug unknown"}]
        with hashes:
            features, meta = extract.text_features(rows, path, np)
        expected = np.zeros(300, dtype=np.float32)
        expected[:2] = [1 / np.sqrt(5), 2 / np.sqrt(5)]
        np.testing.assert_allclose(features[0], expected, atol=1e-7, rtol=0)
        np.testing.assert_array_equal(features[0], features[1])
        self.assertEqual(features.dtype, np.float32)
        self.assertEqual(meta["rows"][0]["oov_tokens"], ["unknown"])
        self.assertEqual(meta["oov_counts"], {"unknown": 2})
        self.assertEqual(meta["identical_vector_groups"], [["one", "two"]])

    def test_all_oov_and_wrong_member_hash_fail(self):
        path, hashes = self.glove_fixture({"mug": np.ones(300)})
        with hashes:
            with self.assertRaisesRegex(ValueError, "no known"):
                extract.text_features([{"sample_id": "one", "caption": "unknown"}], path, np)
            with mock.patch.object(extract, "GLOVE_MEMBER_SHA256", "0" * 64):
                with self.assertRaisesRegex(ValueError, "member SHA256 mismatch"):
                    extract.text_features([{"sample_id": "one", "caption": "mug"}], path, np)

    def test_invalid_vectors_fail(self):
        for values in (np.zeros((1, 300)), np.full((1, 300), np.nan), np.full((1, 300), np.inf)):
            with self.subTest(first=float(values[0, 0])), self.assertRaises(ValueError):
                extract.unit_rows(values, np, "fixture")

    @staticmethod
    def png_bytes(image, **kwargs):
        stream = io.BytesIO()
        image.save(stream, format="PNG", **kwargs)
        return stream.getvalue()

    def test_preprocess_applies_exif_before_crop(self):
        yy, xx = np.indices((240, 320))
        pixels = np.stack((xx % 256, yy % 256, (xx + yy) % 256), axis=-1).astype(np.uint8)
        image = Image.fromarray(pixels)
        exif = Image.Exif()
        exif[274] = 6
        raw = self.png_bytes(image, exif=exif)
        rotated = self.png_bytes(image.transpose(Image.Transpose.ROTATE_270))
        actual = extract.preprocess_bytes(raw, np, Image, ImageOps)
        expected = extract.preprocess_bytes(rotated, np, Image, ImageOps)
        np.testing.assert_array_equal(actual, expected)
        self.assertEqual(actual.shape, (3, 224, 224))
        self.assertEqual(actual.dtype, np.float32)

    def test_preprocess_color_normalization_and_too_small(self):
        raw = self.png_bytes(Image.new("RGB", (300, 224), (255, 0, 128)))
        actual = extract.preprocess_bytes(raw, np, Image, ImageOps)
        expected = (np.array([255, 0, 128], dtype=np.float32) / np.float32(255)
                    - np.asarray(extract.MEAN, dtype=np.float32)) / np.asarray(extract.STD, dtype=np.float32)
        np.testing.assert_array_equal(actual[:, 0, 0], expected)
        with self.assertRaisesRegex(ValueError, "below 224"):
            extract.preprocess_bytes(self.png_bytes(Image.new("RGB", (223, 300))), np, Image, ImageOps)

    def test_atomic_output_hash_and_no_overwrite(self):
        output = self.root / "features"
        arrays = {"image": np.eye(2, dtype=np.float32), "text": np.eye(2, dtype=np.float32),
                  "sample_ids": np.asarray(["fixture-one", "fixture-two"])}
        meta = extract.publish_output(output, arrays, {"fixture_only": True}, np)
        self.assertEqual(meta["feature_sha256"], extract.sha256(output / "features.npz"))
        with np.load(output / "features.npz", allow_pickle=False) as archive:
            self.assertEqual(set(archive.files), set(arrays))
        original = (output / "features.json").read_bytes()
        with self.assertRaises(FileExistsError):
            extract.publish_output(output, arrays, {}, np)
        self.assertEqual((output / "features.json").read_bytes(), original)

    def test_output_failure_cleans_staging(self):
        output = self.root / "failed-features"
        with mock.patch.object(np, "savez_compressed", side_effect=RuntimeError("fixture write failure")):
            with self.assertRaisesRegex(RuntimeError, "fixture write failure"):
                extract.publish_output(output, {}, {}, np)
        self.assertFalse(output.exists())
        self.assertEqual(list(self.root.glob(".failed-features.staging-*")), [])

    def test_atomic_rename_refuses_existing_empty_directory(self):
        staged, target = self.root / "staged", self.root / "target"
        staged.mkdir()
        target.mkdir()
        (staged / "payload").write_text("fixture")
        with self.assertRaises(FileExistsError):
            extract.rename_new_directory(staged, target)
        self.assertEqual(list(target.iterdir()), [])
        self.assertTrue((staged / "payload").is_file())


if __name__ == "__main__":
    unittest.main()
