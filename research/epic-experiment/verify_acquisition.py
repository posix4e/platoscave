#!/usr/bin/env python3
"""Verify every prespecified frame and its acquisition record before extraction."""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from fractions import Fraction
from PIL import Image


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1048576), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--acquirer", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    manifest = json.loads(args.manifest.read_text())
    rows = manifest["rows"]
    expected = defaultdict(dict)
    for row in rows:
        for i, path in enumerate(row["frame_paths"]):
            expected[row["video_id"]][(str(row["uid"]), i, path)] = row
    if sum(len(v) for v in expected.values()) != 4 * len(rows):
        raise ValueError("Duplicate frame identities in selected manifest")
    manifest_hash, acquirer_hash = sha(args.manifest), sha(args.acquirer)
    frame_count = total_bytes = 0
    maximum_time_error = 0
    duplicates = []
    report_hashes = {}
    for video, requests in sorted(expected.items()):
        path = args.data / "acquisition" / (video + ".json")
        report = json.loads(path.read_text())
        if report["status"] != "complete" or report["video_id"] != video:
            raise ValueError(f"Incomplete or wrong video report: {video}")
        if report["manifest_sha256"] != manifest_hash or report["script_sha256"] != acquirer_hash:
            raise ValueError(f"Provenance mismatch: {video}")
        observed = {(f["uid"], f["slot"], f["path"]) for f in report["frames"]}
        if observed != requests.keys() or len(report["frames"]) != len(requests):
            raise ValueError(f"Missing, duplicate, or unexpected frame records: {video}")
        fps = float(Fraction(report["video_stream"]["average_rate"]))
        time_base = float(Fraction(report["video_stream"]["time_base"]))
        origin = report["video_stream"]["start_seconds"]
        width, height = report["video_stream"]["width"], report["video_stream"]["height"]
        pts_by_uid = defaultdict(list)
        for frame in report["frames"]:
            key = frame["uid"], frame["slot"], frame["path"]
            row = requests[key]
            url = f"https://data.bris.ac.uk/datasets/3h91syskeag572hl6tvuovwv4d/videos/train/{row['participant_id']}/{video}.MP4"
            if report["url"] != url:
                raise ValueError(f"Unexpected source URL: {video}")
            target_time = row["frame_times_seconds"][frame["slot"]]
            actual = frame["actual_seconds"]
            if not math.isfinite(actual) or abs(frame["requested_seconds"] - target_time) > 1e-7:
                raise ValueError(f"Requested-time mismatch: {key}")
            if not row["start_seconds"] <= actual <= row["stop_seconds"]:
                raise ValueError(f"Frame outside action window: {key}")
            error = abs(target_time - actual)
            if error > max(0.1, 2 / fps) or abs(frame["source_pts"] * time_base - origin - actual) > 1e-7:
                raise ValueError(f"Frame timing mismatch: {key}")
            image_path = args.data / frame["path"]
            if image_path.stat().st_size != frame["bytes"] or sha(image_path) != frame["sha256"]:
                raise ValueError(f"Frame-byte mismatch: {key}")
            with Image.open(image_path) as image:
                if image.format != "JPEG" or image.mode != "RGB" or image.size != (width, height):
                    raise ValueError(f"Frame format/dimensions mismatch: {key}")
                image.verify()
            frame_count += 1
            total_bytes += frame["bytes"]
            maximum_time_error = max(maximum_time_error, error)
            pts_by_uid[frame["uid"]].append(frame["source_pts"])
        duplicates.extend(uid for uid, pts in pts_by_uid.items() if len(set(pts)) < 4)
        report_hashes[video] = sha(path)
        print(json.dumps({"verified_video": video, "total_frames_verified": frame_count}), flush=True)
    output = {
        "status": "passed", "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "manifest_sha256": manifest_hash, "acquirer_sha256": acquirer_hash,
        "verifier_sha256": sha(__file__), "videos": len(expected), "rows": len(rows),
        "frames": frame_count, "frame_bytes": total_bytes,
        "maximum_requested_time_error_seconds": maximum_time_error,
        "action_uids_with_repeated_source_pts": duplicates,
        "acquisition_report_sha256": report_hashes,
        "scope": "All selected identities, byte hashes, image headers and recorded timestamps; not independent camera-date authentication or full source-video hashing.",
    }
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({key: value for key, value in output.items() if key != "acquisition_report_sha256"}), flush=True)


if __name__ == "__main__":
    main()
