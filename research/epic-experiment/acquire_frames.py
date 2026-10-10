#!/usr/bin/env python3
"""Fetch the four prespecified frames per action using HTTP video seeking.

No image content or class labels guide acquisition. Each completed video has a
byte-hashed frame report; reruns verify and reuse completed videos. Failed
videos remain failures, without replacement by other examples.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
import urllib.request

BASE = "https://data.bris.ac.uk/datasets/3h91syskeag572hl6tvuovwv4d/videos/train"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1048576), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".partial")
    tmp.write_text(json.dumps(data, indent=2) + "\n")
    tmp.replace(path)


def fetch_video(task):
    import av
    video, rows, root_string, manifest_hash, script_hash = task
    root = Path(root_string)
    report_path = root / "acquisition" / (video + ".json")
    if report_path.exists():
        existing = json.loads(report_path.read_text())
        expected = {(str(row["uid"]), i, path) for row in rows
                    for i, path in enumerate(row["frame_paths"])}
        observed = {(f["uid"], f["slot"], f["path"]) for f in existing.get("frames", [])}
        if (existing.get("manifest_sha256") == manifest_hash
                and existing.get("script_sha256") == script_hash
                and existing.get("status") == "complete"
                and len(existing["frames"]) == len(expected) and observed == expected
                and all((root / f["path"]).is_file() and sha(root / f["path"]) == f["sha256"]
                        for f in existing["frames"])):
            return {"video_id": video, "status": "complete", "cached": True,
                    "frame_count": len(existing["frames"])}
    started = time.monotonic()
    url = f"{BASE}/{rows[0]['participant_id']}/{video}.MP4"
    report = {"video_id": video, "url": url, "manifest_sha256": manifest_hash,
              "script_sha256": script_hash, "status": "running", "frames": [],
              "jpeg": {"quality": 95, "subsampling": 0, "resolution": "original decoded RGB"},
              "av_version": av.__version__, "started_at_utc": datetime.now(timezone.utc).isoformat()}
    write_json(report_path, report)
    try:
        with urllib.request.urlopen(urllib.request.Request(url, method="HEAD"), timeout=30) as response:
            report["source_headers"] = {key: response.headers.get(key) for key in
                                        ("Content-Length", "ETag", "Last-Modified", "Accept-Ranges")}
        requests = []
        for row in rows:
            for i, (target, path) in enumerate(zip(row["frame_times_seconds"], row["frame_paths"])):
                requests.append((float(target), str(row["uid"]), i, path,
                                 float(row["start_seconds"]), float(row["stop_seconds"])))
        requests.sort()
        options = {"tls_verify": "1", "ca_file": "/etc/ssl/certs/ca-certificates.crt",
                   "multiple_requests": "1", "seekable": "1"}
        report["http_options"] = options
        with av.open(url, timeout=(30, 60), options=options) as container:
            stream = container.streams.video[0]
            stream.thread_count = 2
            origin = float(stream.start_time * stream.time_base) if stream.start_time is not None else 0.0
            fps = float(stream.average_rate) if stream.average_rate else 0.0
            report["video_stream"] = {"time_base": str(stream.time_base), "start_seconds": origin,
                                      "average_rate": str(stream.average_rate),
                                      "width": stream.width, "height": stream.height}
            current = previous = None
            current_time = previous_time = None
            decoder = None
            for requested, uid, slot, path, action_start, action_stop in requests:
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
                              if frame is not None and action_start <= stamp <= action_stop]
                if not candidates:
                    raise RuntimeError(f"No frame inside action window for {uid}/{slot}")
                chosen, actual = min(candidates, key=lambda pair: (abs(pair[1] - requested), pair[1]))
                tolerance = max(0.1, 2 / fps) if fps > 0 else 0.1
                if abs(actual - requested) > tolerance:
                    raise RuntimeError(f"Frame too far from requested time for {uid}/{slot}: {actual} vs {requested}")
                target = root / path
                target.parent.mkdir(parents=True, exist_ok=True)
                temporary = target.with_suffix(".partial")
                chosen.to_image().save(temporary, format="JPEG", quality=95, subsampling=0)
                temporary.replace(target)
                report["frames"].append({"uid": uid, "slot": slot, "path": path,
                                         "requested_seconds": requested, "actual_seconds": actual,
                                         "source_pts": chosen.pts, "sha256": sha(target),
                                         "bytes": target.stat().st_size})
                if len(report["frames"]) % 64 == 0:
                    write_json(report_path, report)
        if len(report["frames"]) != len(rows) * 4:
            raise RuntimeError("Not all requested frames were acquired")
        report["status"] = "complete"
    except Exception as error:
        report["status"] = "failed"
        report["error"] = f"{type(error).__name__}: {error}"
    report["elapsed_seconds"] = time.monotonic() - started
    report["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    write_json(report_path, report)
    return {"video_id": video, "status": report["status"], "frame_count": len(report["frames"]),
            "elapsed_seconds": report["elapsed_seconds"], "error": report.get("error")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--videos", nargs="+")
    args = parser.parse_args()
    document = json.loads(args.manifest.read_text())
    groups = defaultdict(list)
    for row in document["rows"]:
        if args.videos is None or row["video_id"] in args.videos:
            groups[row["video_id"]].append(row)
    if not groups:
        raise ValueError("No requested videos in manifest")
    manifest_hash, script_hash = sha(args.manifest), sha(__file__)
    tasks = [(video, rows, str(args.output.resolve()), manifest_hash, script_hash)
             for video, rows in sorted(groups.items())]
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(fetch_video, task) for task in tasks]
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(json.dumps({"completed_videos": len(results), "total_videos": len(tasks), **result}), flush=True)
            write_json(args.output / "acquisition-progress.json", {
                "manifest_sha256": manifest_hash, "script_sha256": script_hash,
                "requested_videos": len(tasks), "finished_videos": len(results),
                "results": sorted(results, key=lambda row: row["video_id"]),
            })
    if any(result["status"] != "complete" for result in results):
        raise SystemExit("Some videos failed; see per-video reports. No substitutions made.")


if __name__ == "__main__":
    main()
