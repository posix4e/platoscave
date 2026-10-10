#!/usr/bin/env python3
"""Bounded transport benchmark on two fixed development action windows.

No production frames, reports, or services are modified. This changes only
HTTP connection/range options, keeping decoder, seek and nearest-frame rules.
Each mode returns four frames for each of two clips. Compare PTS and bytes,
not model scores. Network timings are exploratory and affected by other jobs.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import signal
import time

import av

BASE = "https://data.bris.ac.uk/datasets/3h91syskeag572hl6tvuovwv4d/videos/train"
MODES = {
    "baseline": {},
    "keepalive": {"multiple_requests": "1", "seekable": "1"},
    "keepalive_4mib": {"multiple_requests": "1", "seekable": "1",
                       "request_size": "4194304", "initial_request_size": "4194304",
                       "short_seek_size": "4194304"},
}


def one_clip(row, mode):
    began = time.monotonic()
    options = {"tls_verify": "1", "ca_file": "/etc/ssl/certs/ca-certificates.crt", **MODES[mode]}
    result = {"uid": str(row["uid"]), "video_id": row["video_id"], "mode": mode,
              "options": options, "frames": []}
    url = f"{BASE}/{row['participant_id']}/{row['video_id']}.MP4"
    try:
        with av.open(url, timeout=(20, 25), options=options) as container:
            result["open_seconds"] = time.monotonic() - began
            stream = container.streams.video[0]
            stream.thread_count = 2
            origin = float(stream.start_time * stream.time_base) if stream.start_time is not None else 0.0
            fps = float(stream.average_rate) if stream.average_rate else 0.0
            current = previous = None
            current_time = previous_time = None
            decoder = None
            for requested in row["frame_times_seconds"]:
                step_began = time.monotonic()
                if current is None or requested - current_time > 1.0:
                    container.seek(int((requested + origin) / stream.time_base), stream=stream,
                                   backward=True, any_frame=False)
                    decoder = iter(container.decode(stream))
                    current = previous = None
                    current_time = previous_time = None
                while current is None or current_time < requested:
                    if current is not None:
                        previous, previous_time = current, current_time
                    try:
                        current = next(decoder)
                        if current.time is None:
                            raise RuntimeError("Decoded frame lacks a timestamp")
                        current_time = float(current.time) - origin
                    except StopIteration:
                        current, current_time = None, None
                        break
                candidates = [(frame, stamp) for frame, stamp in
                              ((previous, previous_time), (current, current_time))
                              if frame is not None and row["start_seconds"] <= stamp <= row["stop_seconds"]]
                if not candidates:
                    raise RuntimeError("No frame inside action window")
                chosen, actual = min(candidates, key=lambda pair: (abs(pair[1] - requested), pair[1]))
                if abs(actual - requested) > (max(0.1, 2 / fps) if fps > 0 else 0.1):
                    raise RuntimeError("Frame time outside acquisition tolerance")
                image = chosen.to_image()
                output = io.BytesIO()
                image.save(output, format="JPEG", quality=95, subsampling=0)
                result["frames"].append({
                    "requested_seconds": requested, "actual_seconds": actual,
                    "source_pts": chosen.pts,
                    "rgb_sha256": hashlib.sha256(image.tobytes()).hexdigest(),
                    "jpeg_sha256": hashlib.sha256(output.getvalue()).hexdigest(),
                    "elapsed_seconds": time.monotonic() - step_began,
                })
        result["status"] = "complete"
    except Exception as error:
        result["status"] = "failed"
        result["error"] = f"{type(error).__name__}: {error}"
    result["elapsed_seconds"] = time.monotonic() - began
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = json.loads(args.manifest.read_text())["rows"]
    selected = []
    for row in rows:
        if (row["split"] == "dev" and row["stop_seconds"] - row["start_seconds"] >= 8
                and row["video_id"] not in {r["video_id"] for r in selected}):
            selected.append(row)
            if len(selected) == 2:
                break
    if len(selected) != 2:
        raise ValueError("Need two qualifying development clips")
    result = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "selection_rule": "First two selected dev actions lasting at least 8 seconds from distinct videos; no feature or score access",
        "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "av_version": av.__version__, "av_libraries": av.library_versions,
        "runs": [], "comparisons": [],
        "limit": "Two action windows; four requested frames per window and transport mode; not a throughput guarantee",
        "documentation": "https://ffmpeg.org/ffmpeg-protocols.html#http",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for index, row in enumerate(selected):
        modes = list(MODES) if index == 0 else list(reversed(MODES))
        for mode in modes:
            run = one_clip(row, mode)
            result["runs"].append(run)
            print(json.dumps({k: run[k] for k in ("uid", "video_id", "mode", "status", "elapsed_seconds")}), flush=True)
            args.output.write_text(json.dumps(result, indent=2) + "\n")
    for row in selected:
        runs = {r["mode"]: r for r in result["runs"] if r["uid"] == str(row["uid"])}
        baseline = runs["baseline"]
        for mode in ("keepalive", "keepalive_4mib"):
            run = runs[mode]
            comparison = {"uid": str(row["uid"]), "mode": mode}
            if baseline["status"] == run["status"] == "complete":
                keys = ("actual_seconds", "source_pts", "rgb_sha256", "jpeg_sha256")
                comparison["exact_frames_match"] = all(
                    all(a[key] == b[key] for key in keys)
                    for a, b in zip(baseline["frames"], run["frames"])
                ) and len(baseline["frames"]) == len(run["frames"]) == 4
                comparison["speedup_vs_baseline"] = baseline["elapsed_seconds"] / run["elapsed_seconds"]
            else:
                comparison["exact_frames_match"] = None
            result["comparisons"].append(comparison)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"comparisons": result["comparisons"]}), flush=True)


if __name__ == "__main__":
    main()
