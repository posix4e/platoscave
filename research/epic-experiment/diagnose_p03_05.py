#!/usr/bin/env python3
"""Bounded isolated reproduction of the P03_05 acquisition failure.

Reads one selected action with four original requested times in two HTTP
modes. Production acquisition frames and reports are never modified.
"""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import av
from benchmark_http_seek import one_clip


def main():
    root = Path(sys.argv[1])
    output = Path(sys.argv[2])
    manifest = root / "selected_rows.json"
    report = root / "data/acquisition/P03_05.json"
    original = report.read_bytes()
    rows = json.loads(manifest.read_text())["rows"]
    selected = [row for row in rows if row["video_id"] == "P03_05" and str(row["uid"]) == "6787"]
    if len(selected) != 1:
        raise ValueError("Expected precisely the original UID6787 action")
    failure = json.loads(original)
    result = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "failed_report_sha256": hashlib.sha256(original).hexdigest(),
        "failed_report_snapshot": failure,
        "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "benchmark_helper_sha256": hashlib.sha256(Path(__file__).with_name("benchmark_http_seek.py").read_bytes()).hexdigest(),
        "runs": [],
    }
    av.logging.set_level(av.logging.DEBUG)
    for mode in ("keepalive", "baseline"):
        with av.logging.Capture(local=False) as logs:
            run = one_clip(selected[0], mode)
        run["ffmpeg_logs"] = [{"level": level, "component": component, "message": message}
                              for level, component, message in logs]
        result["runs"].append(run)
        output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps({key: run[key] for key in ("mode", "status", "elapsed_seconds")}), flush=True)
    completed = [f for f in failure["frames"] if f["uid"] == "6787"]
    for run in result["runs"]:
        run["matches_already_acquired_action_frames"] = all(
            acquired["source_pts"] == observed["source_pts"]
            and acquired["sha256"] == observed["jpeg_sha256"]
            for acquired, observed in zip(completed, run["frames"])
        ) and len(run["frames"]) >= len(completed)
    if all(run["status"] == "complete" for run in result["runs"]):
        result["all_four_frames_match_between_modes"] = all(
            all(a[key] == b[key] for key in ("source_pts", "actual_seconds", "rgb_sha256", "jpeg_sha256"))
            for a, b in zip(*(run["frames"] for run in result["runs"]))
        )
    result["production_report_unchanged"] = report.read_bytes() == original
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"production_report_unchanged": result["production_report_unchanged"],
                      "all_four_frames_match_between_modes": result.get("all_four_frames_match_between_modes")}), flush=True)


if __name__ == "__main__":
    main()
