#!/usr/bin/env python3
"""Prepare and seal the prospective 48-photograph human collection packet.

prepare creates instructions and empty records, never camera data or captions.
validate checks recorded consistency and files, not the truth of human capture.
Use original JPEG/PNG photographs; set an iPhone camera to Most Compatible to
avoid HEIC. Hashes, timestamps and attestations do not prove capture chronology.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import secrets
import sys
import uuid
import warnings


PROTOCOL_ID = "fresh-objects-v1"
CLASSES = ("mug", "bowl", "plate", "bottle", "spoon", "fork", "book", "shoe")
ROUND_IDS = tuple(f"r{i:02d}" for i in range(1, 7))
ROUND_FIELDS = ("round_id", "started_at_utc", "marker_file", "notes")
SAMPLE_FIELDS = ("sample_id", "round_id", "split", "class_name", "instance_id", "image_file",
                 "captured_at_utc", "caption", "captioned_at_utc")
LOCK_FILES = {"protocol.json", "intake.py", "extract.py", "analyze.py"}
DEFAULT_PROTOCOL = Path(__file__).with_name("protocol.json")
ATTESTATION = (
    "I inspected every original round marker and confirm its handwritten nonce and eight actual objects "
    "match the session plan and records. The scored images are new human camera photographs made after "
    "this packet was generated from the published release. The captions are human-written after all "
    "48 photographs existed, without generative assistance. The A and B objects are different physical "
    "instances, and I followed the locked collection protocol."
)
LIMITS = (
    "This seal checks recorded consistency, file bytes and a declared human attestation. "
    "It does not inspect marker text or objects automatically, authenticate capture time, "
    "prove the originals are unedited, prove publication time, or exclude fabrication."
)


class IntakeError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise IntakeError(message)


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            value.update(block)
    return value.hexdigest()


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def timestamp(value, label):
    require(isinstance(value, str) and re.fullmatch(
        r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?(?:Z|\+00:00)", value),
        f"{label} must be an ISO UTC timestamp with Z or +00:00")
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise IntakeError(f"Invalid timestamp for {label}: {value}") from error


def read_json(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, f"Duplicate JSON key: {key}")
            result[key] = value
        return result
    def nonfinite(value):
        raise IntakeError(f"Nonfinite JSON value: {value}")
    return json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=unique, parse_constant=nonfinite)


def contained_file(root, name):
    require(isinstance(name, str) and name and "\\" not in name and "\x00" not in name,
            "File paths must be nonempty relative POSIX paths")
    pure = PurePosixPath(name)
    require(not pure.is_absolute() and all(part not in {"", ".", ".."} for part in name.split("/")),
            f"Absolute or traversing file path rejected: {name}")
    path = root
    for part in pure.parts:
        path = path / part
        require(not path.is_symlink(), f"Symlink rejected: {name}")
    require(path.is_file() and path.resolve().is_relative_to(root.resolve()), f"Missing or external file: {name}")
    return path


def no_symlink_components(path):
    path = Path(path).absolute()
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current = current / part
        require(not current.is_symlink(), f"Symlink rejected: {current}")


def verify_release(protocol):
    protocol = Path(protocol).absolute()
    no_symlink_components(protocol)
    require(not protocol.is_symlink() and protocol.is_file(), "Protocol must be a regular, nonsymlink file")
    directory = protocol.parent
    require(protocol.name == "protocol.json", "The locked protocol filename must be protocol.json")
    lock_path = contained_file(directory, "release-lock.json")
    lock = read_json(lock_path)
    require(lock.get("schema_version") == 1 and isinstance(lock.get("files"), dict)
            and set(lock["files"]) == LOCK_FILES, "release-lock.json must lock protocol.json, intake.py, extract.py and analyze.py")
    for name, expected in lock["files"].items():
        require(isinstance(expected, str) and re.fullmatch(r"[0-9a-f]{64}", expected), f"Invalid release hash: {name}")
        require(digest(contained_file(directory, name)) == expected, f"Release lock mismatch: {name}")
    require(lock["files"]["intake.py"] == digest(__file__), "Running intake.py differs from the locked release")
    document = read_json(protocol)
    require(document.get("schema_version") == 1 and document.get("protocol_id") == PROTOCOL_ID,
            "Unsupported protocol schema or ID")
    require(document.get("classes") == list(CLASSES), "Protocol class list differs from this intake implementation")
    collection = document.get("collection", {})
    require(collection.get("fit_rounds") == list(ROUND_IDS[:4])
            and collection.get("development_rounds") == list(ROUND_IDS[4:])
            and collection.get("photographs") == 48 and collection.get("rounds") == 6
            and collection.get("fit_count") == 32 and collection.get("development_count") == 16,
            "Protocol collection schedule differs from this implementation")
    return protocol, lock_path, digest(protocol), digest(lock_path)


def split_for(round_id):
    return "fit" if round_id in ROUND_IDS[:4] else "development"


def write_json_new(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def prepare(out, collector, protocol=DEFAULT_PROTOCOL):
    require(isinstance(collector, str) and collector.strip(), "Collector must be a nonempty human alias")
    source_protocol, source_lock, protocol_hash, lock_hash = verify_release(protocol)
    out = Path(out).absolute()
    no_symlink_components(out)
    require(not out.exists() and not out.is_symlink(), "Packet output already exists; choose a new directory")
    plan = {"schema_version": 1, "protocol_id": PROTOCOL_ID, "protocol_sha256": protocol_hash,
            "release_lock_sha256": lock_hash, "intake_sha256": digest(__file__), "created_at_utc": utc_now(),
            "collector_id": collector.strip(), "run_id": str(uuid.uuid4()), "rounds": []}
    randomizer, nonces, samples = secrets.SystemRandom(), set(), []
    for round_id in ROUND_IDS:
        order = list(CLASSES)
        randomizer.shuffle(order)
        nonce = secrets.token_hex(8)
        while nonce in nonces:
            nonce = secrets.token_hex(8)
        nonces.add(nonce)
        split = split_for(round_id)
        plan["rounds"].append({"round_id": round_id, "split": split, "nonce": nonce, "order": order})
        for name in order:
            sample_id = f"{round_id}-{name}"
            samples.append({"sample_id": sample_id, "round_id": round_id, "split": split, "class_name": name,
                            "instance_id": name + ("-A" if split == "fit" else "-B"),
                            "image_file": f"images/{sample_id}.jpg", "captured_at_utc": "", "caption": "", "captioned_at_utc": ""})
    plan["caption_order"] = [row["sample_id"] for row in samples]
    randomizer.shuffle(plan["caption_order"])
    out.mkdir(parents=True, exist_ok=False)
    (out / "images").mkdir()
    (out / "markers").mkdir()
    (out / "protocol.json").write_bytes(source_protocol.read_bytes())
    (out / "release-lock.json").write_bytes(source_lock.read_bytes())
    write_json_new(out / "session-plan.json", plan)
    for filename, fields, rows in (
        ("rounds.csv", ROUND_FIELDS, [{"round_id": r, "started_at_utc": "", "marker_file": f"markers/{r}.jpg", "notes": ""} for r in ROUND_IDS]),
        ("samples.csv", SAMPLE_FIELDS, samples),
    ):
        with (out / filename).open("x", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    sheet = ["FRESH OBJECTS — CAPTURE SHEETS", "",
             "This packet is an empty plan. No photographs or captions have been made.",
             "Read protocol.json before collection. Use eight A objects for fit and eight different B objects for development.",
             "Keep original, unedited JPEG/PNG camera files, at least 224 pixels on the short side.",
             "Set the camera to Most Compatible to avoid HEIC. Do not convert or edit files for this packet.",
             "No generated images, stock pictures, screenshots, or AI-written captions. Do not consult model scores.",
             "Record UTC timestamps in rounds.csv and samples.csv. For round notes, write 'none' if no replacements occurred.",
             "Each marker is a separate photograph of the handwritten nonce with all eight actual objects.",
             "Keep the nonce paper out of the scored photographs. Inspect all markers before attesting.", ""]
    for row in plan["rounds"]:
        sheet.extend([f"{row['round_id']} — {row['split']} — handwritten nonce: {row['nonce']}",
                      "First photograph the marker. Then take the scored photographs in this order:"])
        sheet.extend(f"  {index}. {row['round_id']}-{name} — {name} "
                     + ("(object A)" if row["split"] == "fit" else "(object B)")
                     for index, name in enumerate(row["order"], 1))
        sheet.append("")
    sheet.extend(["CAPTIONS — ONLY AFTER ALL 48 PHOTOGRAPHS EXIST",
                  "View photographs in this order. Write one short factual human caption per photograph.",
                  "Record each caption and its UTC timestamp in samples.csv. Hide partition labels where practical."])
    sheet.extend(f"  {index}. {sample_id}" for index, sample_id in enumerate(plan["caption_order"], 1))
    sheet.extend(["", LIMITS, ""])
    (out / "CAPTURE-SHEETS.txt").write_text("\n".join(sheet), encoding="utf-8")
    require(digest(out / "protocol.json") == protocol_hash and digest(out / "release-lock.json") == lock_hash,
            "Release inputs changed during packet preparation")
    require(verify_release(protocol)[2:] == (protocol_hash, lock_hash), "Release changed during packet preparation")
    return plan


def read_csv(path, fields):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        require(reader.fieldnames == list(fields), f"Unexpected CSV columns in {Path(path).name}")
        rows = list(reader)
    require(all(set(row) == set(fields) and all(isinstance(value, str) for value in row.values()) for row in rows),
            f"Malformed CSV row in {Path(path).name}")
    return rows


def inspect_image(path):
    try:
        from PIL import Image, ImageFile
    except ImportError as error:
        raise IntakeError("Pillow is required to validate original JPEG/PNG files") from error
    suffix = path.suffix.lower()
    require(suffix in {".jpg", ".jpeg", ".png"}, "Use original JPG/JPEG/PNG only; set the camera to Most Compatible instead of HEIC")
    expected_format = "PNG" if suffix == ".png" else "JPEG"
    previous = ImageFile.LOAD_TRUNCATED_IMAGES
    try:
        ImageFile.LOAD_TRUNCATED_IMAGES = False
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(path) as image:
                require(image.format == expected_format, f"Image content does not match its extension: {path.name}")
                require(getattr(image, "n_frames", 1) == 1, f"A single original still image is required: {path.name}")
                width, height = image.size
                require(min(width, height) >= 224, f"Image short side is below 224 pixels: {path.name}")
                image.verify()
            with Image.open(path) as image:
                image.load()
    except IntakeError:
        raise
    except Exception as error:
        raise IntakeError(f"Unreadable image {path.name}: {error}") from error
    finally:
        ImageFile.LOAD_TRUNCATED_IMAGES = previous
    return {"sha256": digest(path), "size_bytes": path.stat().st_size, "width": width, "height": height, "format": expected_format}


def validate(packet, out, *, attest_human_capture=False, protocol=DEFAULT_PROTOCOL):
    require(attest_human_capture is True, "Human marker inspection and capture/caption attestation are required; use --attest-human-capture only after that human review")
    source_protocol, source_lock, protocol_hash, lock_hash = verify_release(protocol)
    packet, out = Path(packet).absolute(), Path(out).absolute()
    no_symlink_components(packet)
    no_symlink_components(out)
    require(packet.is_dir() and not packet.is_symlink(), "Packet must be a nonsymlink directory")
    require(not out.exists() and not out.is_symlink(), "Sealed output already exists; it will not be overwritten")
    require(out.parent.is_dir(), "Sealed output parent directory must exist")
    records = {}
    for name in ("protocol.json", "release-lock.json", "session-plan.json", "rounds.csv", "samples.csv"):
        path = contained_file(packet, name)
        records[name] = {"sha256": digest(path), "size_bytes": path.stat().st_size}
    require(records["protocol.json"]["sha256"] == protocol_hash and records["release-lock.json"]["sha256"] == lock_hash,
            "Packet protocol/release lock differs from the verified release")
    plan = read_json(packet / "session-plan.json")
    require(plan.get("schema_version") == 1 and plan.get("protocol_id") == PROTOCOL_ID
            and plan.get("protocol_sha256") == protocol_hash and plan.get("release_lock_sha256") == lock_hash
            and plan.get("intake_sha256") == digest(__file__), "Session plan protocol/source locks differ")
    require(isinstance(plan.get("collector_id"), str) and plan["collector_id"].strip(), "Missing human collector alias")
    try:
        require(str(uuid.UUID(plan["run_id"])) == plan["run_id"], "Run ID must be a canonical UUID")
    except (ValueError, TypeError, KeyError, AttributeError) as error:
        raise IntakeError("Run ID must be a canonical UUID") from error
    created = timestamp(plan.get("created_at_utc"), "plan.created_at_utc")
    rounds = plan.get("rounds")
    require(isinstance(rounds, list) and len(rounds) == 6 and all(isinstance(row, dict) for row in rounds)
            and [row.get("round_id") for row in rounds] == list(ROUND_IDS), "Plan must contain r01 through r06 in order")
    nonces, expected_ids = set(), []
    for row in rounds:
        require(row.get("split") == split_for(row["round_id"]), "Wrong round split in session plan")
        require(isinstance(row.get("nonce"), str) and re.fullmatch(r"[0-9a-f]{16}", row["nonce"])
                and row["nonce"] not in nonces, "Round nonces must be distinct 16-character lowercase hex strings")
        nonces.add(row["nonce"])
        require(isinstance(row.get("order"), list) and len(row["order"]) == 8
                and all(isinstance(name, str) for name in row["order"]) and set(row["order"]) == set(CLASSES),
                "Each round order must contain each of the eight classes exactly once")
        expected_ids.extend(f"{row['round_id']}-{name}" for name in row["order"])
    captions_order = plan.get("caption_order")
    require(isinstance(captions_order, list) and len(captions_order) == 48
            and all(isinstance(value, str) for value in captions_order) and set(captions_order) == set(expected_ids),
            "caption_order must contain all 48 sample IDs exactly once")
    round_rows = read_csv(packet / "rounds.csv", ROUND_FIELDS)
    sample_rows = read_csv(packet / "samples.csv", SAMPLE_FIELDS)
    require(len(round_rows) == 6 and {row["round_id"] for row in round_rows} == set(ROUND_IDS), "rounds.csv must contain exactly the six round IDs")
    require(len(sample_rows) == 48 and {row["sample_id"] for row in sample_rows} == set(expected_ids), "samples.csv must contain all 48 unique planned sample IDs")
    round_by_id = {row["round_id"]: row for row in round_rows}
    sample_by_id = {row["sample_id"]: row for row in sample_rows}
    starts = {round_id: timestamp(round_by_id[round_id]["started_at_utc"], round_id + ".started_at_utc") for round_id in ROUND_IDS}
    require(created <= starts[ROUND_IDS[0]] and all(starts[a] <= starts[b] for a, b in zip(ROUND_IDS, ROUND_IDS[1:])),
            "Round timestamps must follow plan generation and round order")
    seen_paths, seen_hashes, image_records = set(), set(), []
    def image_record(filename, prefix):
        path = contained_file(packet, filename)
        resolved = str(path.resolve())
        require(resolved not in seen_paths, "Images and markers must use distinct file paths")
        record = inspect_image(path)
        require(record["sha256"] not in seen_hashes, "Duplicate image bytes across scored images or round markers")
        seen_paths.add(resolved)
        seen_hashes.add(record["sha256"])
        image_records.append((path, record["sha256"]))
        return {prefix + "_" + key: value for key, value in record.items()}
    sealed_rounds, sealed_rows, captured, captioned = [], [], {}, {}
    for index, round_plan in enumerate(rounds):
        round_id = round_plan["round_id"]
        recorded_round = round_by_id[round_id]
        require(recorded_round["notes"].strip(), f"Round notes are required; write 'none' if no replacements: {round_id}")
        sealed_rounds.append({**round_plan, **recorded_round, **image_record(recorded_round["marker_file"], "marker")})
        previous_capture = starts[round_id]
        for name in round_plan["order"]:
            sample_id = f"{round_id}-{name}"
            row = sample_by_id[sample_id]
            split = split_for(round_id)
            require(row["round_id"] == round_id and row["class_name"] == name and row["split"] == split
                    and row["instance_id"] == name + ("-A" if split == "fit" else "-B"), f"Wrong sample identity, split or physical-instance assignment: {sample_id}")
            require(row["caption"].strip(), f"Empty human caption: {sample_id}")
            photo_time = timestamp(row["captured_at_utc"], sample_id + ".captured_at_utc")
            caption_time = timestamp(row["captioned_at_utc"], sample_id + ".captioned_at_utc")
            require(previous_capture <= photo_time <= caption_time, f"Capture/caption timestamps violate planned capture order: {sample_id}")
            if index < 5:
                require(photo_time <= starts[ROUND_IDS[index + 1]], f"Scored photo occurs after the next round started: {sample_id}")
            previous_capture = photo_time
            captured[sample_id], captioned[sample_id] = photo_time, caption_time
            sealed_rows.append({**row, **image_record(row["image_file"], "image")})
    last_photo = max(captured.values())
    previous_caption = last_photo
    for sample_id in captions_order:
        require(previous_caption <= captioned[sample_id], f"Caption timestamps must follow all photos and caption_order: {sample_id}")
        previous_caption = captioned[sample_id]
    require(previous_caption <= datetime.now(timezone.utc), "Caption timestamps are in the future; correct the capture log")
    # Recheck bytes before creating the seal. Raw inputs are never rewritten.
    for name, record in records.items():
        require(digest(contained_file(packet, name)) == record["sha256"], f"Input changed during validation: {name}")
    for path, expected in image_records:
        require(digest(path) == expected, f"Image changed during validation: {path.name}")
    require(verify_release(protocol)[2:] == (protocol_hash, lock_hash), "Release changed during validation")
    seal = {"schema_version": 1, "protocol_id": PROTOCOL_ID, "protocol_sha256": protocol_hash,
            "release_lock_sha256": lock_hash, "session_plan_sha256": records["session-plan.json"]["sha256"],
            "intake_sha256": digest(__file__), "run_id": plan["run_id"], "collector_id": plan["collector_id"],
            "sealed_at_utc": utc_now(), "attestation": {"human_capture": True, "human_captions": True,
            "marker_photos_inspected": True, "statement": ATTESTATION, "recorded_at_utc": utc_now()},
            "evidence_limits": LIMITS, "rounds": sealed_rounds, "rows": sealed_rows, "input_records": records}
    write_json_new(out, seal)
    return seal


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("prepare", help="Generate empty capture records after obtaining the published release")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--collector", required=True)
    p.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    v = commands.add_parser("validate", help="Inspect records and originals, then seal a complete packet")
    v.add_argument("--packet", type=Path, required=True)
    v.add_argument("--out", type=Path, required=True)
    v.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    v.add_argument("--attest-human-capture", action="store_true", help=ATTESTATION)
    args = parser.parse_args(argv)
    try:
        if args.command == "prepare":
            plan = prepare(args.out, args.collector, args.protocol)
            print(json.dumps({"status": "empty_packet_prepared", "packet": str(args.out), "run_id": plan["run_id"],
                              "rows_to_collect": 48, "note": "No photographs or captions were generated. Follow the protocol, capture order and caption_order. Use original JPEG/PNG; camera Most Compatible avoids HEIC.", "evidence_limits": LIMITS}))
        else:
            seal = validate(args.packet, args.out, attest_human_capture=args.attest_human_capture, protocol=args.protocol)
            print(json.dumps({"status": "packet_sealed", "output": str(args.out), "rows": len(seal["rows"]), "run_id": seal["run_id"], "evidence_limits": LIMITS}))
    except (IntakeError, OSError, json.JSONDecodeError, csv.Error) as error:
        print(f"Intake refused: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
