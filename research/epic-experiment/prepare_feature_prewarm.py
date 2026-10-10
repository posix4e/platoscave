#!/usr/bin/env python3
"""Prepare a whole-video private cache snapshot; never launch extraction.

Run inside the VM. Only complete reports from the frozen acquisition version
qualify. The original full selection, extractor, and final cache are read-only.
"""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import re
import shlex


MANIFEST_SHA256 = "d1a34b2ad47a1456a011c6e4f2d6308bae34ebf6aed0c6fb335794ac3dc648be"
EXTRACTOR_SHA256 = "471ede5f611b17ab63252957c0f9785ff2afede22a656f8452fdadff4217f203"
ACQUIRER_SHA256 = "c3025e54c7f8db18fc7a883f1373f1ffaf8707d77a29371acb77735288d55f65"


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/opt/rosetta/epic-experiment"))
    parser.add_argument("--snapshot", default=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    args = parser.parse_args()
    require(bool(re.fullmatch(r"[A-Za-z0-9_-]+", args.snapshot)), "Invalid snapshot name")
    root = args.root.resolve()
    manifest, extractor = root / "selected_rows.json", root / "extract_features.py"
    for path, expected in ((manifest, MANIFEST_SHA256), (extractor, EXTRACTOR_SHA256),
                           (root / "acquire_frames.py", ACQUIRER_SHA256)):
        require(sha(path) == expected, f"Frozen input mismatch: {path}")
    from PIL import Image

    payload = json.loads(manifest.read_text())
    by_video = defaultdict(list)
    for row in payload["rows"]:
        by_video[row["video_id"]].append(row)
    # Reports are atomically replaced by acquisition. Snapshot candidates before
    # byte validation so later completions do not change this pass's membership.
    candidates, skipped_complete_reports = {}, []
    for video in sorted(by_video):
        path = root / "data" / "acquisition" / (video + ".json")
        if path.is_file():
            raw = path.read_bytes()
            report = json.loads(raw)
            if report.get("status") == "complete":
                report_hash = hashlib.sha256(raw).hexdigest()
                if (report.get("video_id") == video and report.get("manifest_sha256") == MANIFEST_SHA256
                        and report.get("script_sha256") == ACQUIRER_SHA256):
                    candidates[video] = (report, report_hash)
                else:
                    skipped_complete_reports.append({
                        "video_id": video, "report_sha256": report_hash,
                        "reported_video_id": report.get("video_id"),
                        "reported_manifest_sha256": report.get("manifest_sha256"),
                        "reported_script_sha256": report.get("script_sha256"),
                        "reason": "Complete report does not match frozen full-manifest/v2 provenance; left for main acquisition",
                    })
    require(bool(candidates), "No complete videos are available")
    verified, frame_bytes = {}, 0
    for video, (report, report_hash) in candidates.items():
        require(report.get("video_id") == video and report.get("manifest_sha256") == MANIFEST_SHA256
                and report.get("script_sha256") == ACQUIRER_SHA256, f"Report provenance mismatch: {video}")
        expected = {(str(row["uid"]), slot, path): row for row in by_video[video]
                    for slot, path in enumerate(row["frame_paths"])}
        frames = report["frames"]
        observed = {(frame["uid"], frame["slot"], frame["path"]) for frame in frames}
        require(observed == expected.keys() and len(frames) == len(expected), f"Frame identities differ: {video}")
        stream = report["video_stream"]
        fps, time_base = float(Fraction(stream["average_rate"])), float(Fraction(stream["time_base"]))
        require(fps > 0, f"Invalid frame rate: {video}")
        for frame in frames:
            key = frame["uid"], frame["slot"], frame["path"]
            row = expected[key]
            target = row["frame_times_seconds"][frame["slot"]]
            actual = frame["actual_seconds"]
            require(report["url"] == "https://data.bris.ac.uk/datasets/3h91syskeag572hl6tvuovwv4d/videos/train/"
                    + row["participant_id"] + "/" + video + ".MP4", f"Source URL mismatch: {key}")
            require(math.isfinite(actual) and abs(frame["requested_seconds"] - target) <= 1e-7
                    and row["start_seconds"] <= actual <= row["stop_seconds"]
                    and abs(actual - target) <= max(0.1, 2 / fps)
                    and abs(frame["source_pts"] * time_base - stream["start_seconds"] - actual) <= 1e-7,
                    f"Frame time mismatch: {key}")
            path = root / "data" / frame["path"]
            require(path.stat().st_size == frame["bytes"] and sha(path) == frame["sha256"],
                    f"Frame byte mismatch: {key}")
            with Image.open(path) as image:
                require(image.format == "JPEG" and image.mode == "RGB"
                        and image.size == (stream["width"], stream["height"]), f"Frame format mismatch: {key}")
                image.verify()
            frame_bytes += frame["bytes"]
        require(sha(root / "data" / "acquisition" / (video + ".json")) == report_hash,
                f"Report changed during validation: {video}")
        verified[video] = {"report_sha256": report_hash, "rows": len(by_video[video]), "frames": len(frames)}
    rows = [row for row in payload["rows"] if row["video_id"] in verified]
    destination = root / "frame-cache-prewarm" / args.snapshot
    destination.mkdir(parents=True, exist_ok=False)
    subset = destination / "selected-complete-videos.json"
    metadata = {**payload.get("metadata", {}), "prewarm": {
        "nonfinal": True, "source_manifest_sha256": MANIFEST_SHA256,
        "membership": "All rows of whole videos with complete frozen-v2 reports at snapshot time; original row order",
    }}
    subset.write_text(json.dumps({"metadata": metadata, "rows": rows}, indent=2) + "\n")
    command = ["timeout", "--signal=TERM", "--kill-after=10s", "240s", "env",
               "PYTHONUNBUFFERED=1", "OMP_NUM_THREADS=4", "MKL_NUM_THREADS=4",
               "PYTHONPATH=/opt/rosetta/repo", "CUBLAS_WORKSPACE_CONFIG=:4096:8",
               "/opt/rosetta/clean-example/.gpu-venv/bin/python",
               str(extractor), "--manifest", str(subset), "--frame-root", str(root / "data"),
               "--dino-source", str(root / "dino-source"), "--dino-checkpoint",
               "/opt/rosetta/clean-example/models/dino_vitbase16_pretrain.pth", "--glove",
               "/opt/rosetta/clean-example/models/glove.6B.zip", "--output", str(destination / "partial-features.npz"),
               "--cache-dir", str(destination / "cache"), "--batch-size", "64", "--device", "cuda", "--allow-partial"]
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(), "snapshot": args.snapshot,
        "preparer_sha256": sha(__file__), "source_manifest_sha256": MANIFEST_SHA256,
        "subset_manifest_sha256": sha(subset), "extractor_sha256": EXTRACTOR_SHA256,
        "acquirer_sha256": ACQUIRER_SHA256, "videos": verified, "rows": len(rows),
        "skipped_complete_reports": skipped_complete_reports,
        "frames": 4 * len(rows), "frame_bytes": frame_bytes,
        "command": command, "command_shell": shlex.join(command),
        "launch_performed": False,
        "launch_condition": "Only while acquisition remains active and supervisor waits; terminate only this warm process if acquisition ends. Original extractor argv retains the supervisor's competing-process protection.",
        "handoff": "After this process exits, validate complete private cache files before promotion; never copy .tmp files or overwrite final cache entries. Final extraction still validates fingerprints and frame hashes and writes a complete original-manifest feature artifact.",
    }
    (destination / "snapshot.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"snapshot_directory": str(destination), "videos": len(verified),
                      "rows": len(rows), "frames": 4 * len(rows), "command_shell": shlex.join(command)}, indent=2))


if __name__ == "__main__":
    main()
