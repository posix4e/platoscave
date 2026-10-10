#!/usr/bin/env python3
"""Validate stopped warm-cache files and atomically add absent final entries.

Run only in the VM after the warm service is terminal, while the main pipeline
still waits for acquisition. No original input, frozen script, or existing
final cache entry is modified. No temporary cache files are adopted.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import stat
import subprocess


MANIFEST_SHA256 = "d1a34b2ad47a1456a011c6e4f2d6308bae34ebf6aed0c6fb335794ac3dc648be"
EXTRACTOR_SHA256 = "471ede5f611b17ab63252957c0f9785ff2afede22a656f8452fdadff4217f203"
ACQUIRER_SHA256 = "c3025e54c7f8db18fc7a883f1373f1ffaf8707d77a29371acb77735288d55f65"
RUNTIME = {"torch_version": "2.7.1+cu128", "numpy_version": "2.5.3",
           "pillow_version": "12.3.0", "device": "cuda", "cuda_version": "12.8",
           "batch_size": 64, "device_name": "NVIDIA H100 NVL"}


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def regular_file(path):
    require(stat.S_ISREG(path.lstat().st_mode), f"Not a regular file: {path}")


def check_quiescent(root, warm_service):
    command = ["systemctl", "show", warm_service, "--no-pager", "-p", "LoadState", "-p", "ActiveState",
               "-p", "SubState", "-p", "MainPID", "-p", "Result", "-p", "ExecMainStatus"]
    result = subprocess.run(command, check=True, capture_output=True, text=True, timeout=20)
    service = dict(line.split("=", 1) for line in result.stdout.splitlines() if "=" in line)
    terminal = service.get("ActiveState") in {"inactive", "failed"} or (
        service.get("ActiveState") == "active" and service.get("SubState") == "exited"
        and service.get("Result") == "success" and service.get("ExecMainStatus") == "0")
    require(service.get("LoadState") == "loaded" and terminal and service.get("MainPID") == "0",
            f"Warm unit is not confirmed terminal: {service}")
    extractor = str(root / "extract_features.py").encode()
    for path in Path("/proc").glob("[0-9]*/cmdline"):
        try:
            require(extractor not in path.read_bytes().split(b"\0"),
                    f"Extractor process still active: PID {path.parent.name}")
        except (FileNotFoundError, ProcessLookupError):
            pass
    state = json.loads((root / "pipeline-status.json").read_text())
    require(state.get("status") == "running" and state.get("current_stage") == "acquisition_wait"
            and state.get("stages", {}).get("acquisition_wait", {}).get("state") == "waiting",
            "Main pipeline is no longer waiting for acquisition; no further promotion permitted")
    return {"checked_utc": utc(), "warm_service": service, "pipeline_attempt": state.get("attempt"),
            "pipeline_updated_utc": state.get("updated_utc"), "pipeline_stage": state["current_stage"]}


def cache_name(video):
    safe = re.sub(r"[^a-zA-Z0-9_-]", "_", video)
    return safe + "-" + hashlib.sha256(video.encode()).hexdigest()[:8] + ".npz"


def frozen_recipe(root):
    import numpy as np
    import PIL
    import torch

    source = root / "extract_features.py"
    require(sha(source) == EXTRACTOR_SHA256, "Frozen extractor changed")
    spec = importlib.util.spec_from_file_location("frozen_epic_extractor", source)
    extractor = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(extractor)
    require(torch.cuda.is_available(), "Cannot confirm frozen CUDA runtime")
    runtime = {"torch_version": str(torch.__version__), "numpy_version": np.__version__,
               "pillow_version": PIL.__version__, "device": "cuda", "cuda_version": torch.version.cuda,
               "batch_size": 64, "device_name": torch.cuda.get_device_name(torch.device("cuda"))}
    require(runtime == RUNTIME, f"Runtime differs from frozen warm/final recipe: {runtime}")
    inputs = {"extract_features.py": EXTRACTOR_SHA256}
    for filename, digest in extractor.SOURCE_SHA256.items():
        actual = sha(root / "dino-source" / filename)
        require(actual == digest, f"DINO source changed: {filename}")
        inputs["dino-source/" + filename] = actual
    checkpoint = root.parent / "clean-example" / "models" / "dino_vitbase16_pretrain.pth"
    require(sha(checkpoint) == extractor.DINO_SHA256, "DINO checkpoint changed")
    inputs[str(checkpoint)] = extractor.DINO_SHA256
    recipe = {**extractor.FRAME_RECIPE, **runtime, "extractor_sha256": EXTRACTOR_SHA256}
    fingerprint = hashlib.sha256(json.dumps(recipe, sort_keys=True).encode()).hexdigest()
    return np, recipe, fingerprint, inputs


def validate_cache(video, source, rows, root, snapshot_video, fingerprint, np):
    regular_file(source)
    cache_hash = sha(source)
    expected_paths = sorted({str((root / "data" / frame).resolve())
                             for row in rows for frame in row["frame_paths"]})
    require(len(expected_paths) == 4 * len(rows), f"Repeated frame paths in {video}")
    report_path = root / "data" / "acquisition" / (video + ".json")
    report_hash = sha(report_path)
    require(report_hash == snapshot_video["report_sha256"], f"Acquisition report changed: {video}")
    report = json.loads(report_path.read_text())
    require(report.get("status") == "complete" and report.get("video_id") == video
            and report.get("manifest_sha256") == MANIFEST_SHA256
            and report.get("script_sha256") == ACQUIRER_SHA256, f"Wrong acquisition provenance: {video}")
    report_digests = {str((root / "data" / frame["path"]).resolve()): frame["sha256"]
                      for frame in report["frames"]}
    require(sorted(report_digests) == expected_paths and len(report["frames"]) == len(expected_paths),
            f"Acquisition report frame paths differ: {video}")
    with np.load(source, allow_pickle=False) as cache:
        require(set(cache.files) == {"frame_paths", "digests", "features", "fingerprint"},
                f"Unexpected cache fields: {source}")
        paths, digests, vectors, saved_fingerprint = (cache[name] for name in
                                                     ("frame_paths", "digests", "features", "fingerprint"))
        require(saved_fingerprint.shape == () and saved_fingerprint.dtype.kind == "U"
                and str(saved_fingerprint.item()) == fingerprint, f"Runtime fingerprint mismatch: {video}")
        require(paths.shape == (len(expected_paths),) and paths.dtype.kind == "U"
                and paths.tolist() == expected_paths, f"Cache is not the whole video's ordered frame set: {video}")
        require(digests.shape == paths.shape and digests.dtype.kind == "U", f"Bad digest array: {video}")
        require(vectors.shape == (len(expected_paths), 768) and vectors.dtype == np.dtype("float32")
                and np.isfinite(vectors).all(), f"Invalid saved vector dimensions/dtype/values: {video}")
        norms = np.linalg.norm(vectors, axis=1)
        require(np.allclose(norms, 1.0, rtol=0, atol=2e-6), f"Saved frame vectors are not unit norm: {video}")
        hashes = []
        for path, digest in zip(expected_paths, digests.tolist()):
            require(re.fullmatch(r"[0-9a-f]{64}", digest) is not None, f"Invalid frame hash: {path}")
            actual = sha(path)
            require(actual == digest == report_digests[path], f"Current frame bytes differ: {path}")
            hashes.append({"path": path, "sha256": actual})
    require(sha(source) == cache_hash and sha(report_path) == report_hash,
            f"Cache/report changed during validation: {video}")
    return {"video_id": video, "source_cache": str(source), "source_cache_sha256": cache_hash,
            "acquisition_report_sha256": report_hash, "rows": len(rows), "frames": len(expected_paths),
            "minimum_vector_norm": float(norms.min()), "maximum_vector_norm": float(norms.max()),
            "frame_sha256": hashes}


def flush_report(path, report):
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temporary.open("w") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/opt/rosetta/epic-experiment"))
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--warm-service", required=True)
    args = parser.parse_args()
    root, snapshot = args.root.resolve(), args.snapshot.resolve()
    require(snapshot.parent == root / "frame-cache-prewarm" and snapshot.is_dir(), "Unexpected snapshot directory")
    require(bool(re.fullmatch(r"rosetta-epic-prewarm-[A-Za-z0-9_.-]+\.service", args.warm_service)),
            "Unexpected warm service name")
    audit_path = snapshot / ("promotion-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ") + ".json")
    report = {"schema_version": 1, "started_utc": utc(), "status": "running", "helper_sha256": sha(__file__),
              "snapshot_directory": str(snapshot), "warm_service_name": args.warm_service,
              "validated": [], "promoted": [], "skipped_existing": [], "missing_private_cache_videos": [],
              "guard_checks": [], "policy": "Add verified complete .npz files with os.link; never overwrite existing final entries or adopt .tmp files"}
    try:
        report["guard_checks"].append(check_quiescent(root, args.warm_service))
        require(sha(root / "selected_rows.json") == MANIFEST_SHA256, "Frozen original selection changed")
        require(sha(root / "acquire_frames.py") == ACQUIRER_SHA256, "Frozen acquisition script changed")
        snapshot_path, subset_path = snapshot / "snapshot.json", snapshot / "selected-complete-videos.json"
        snapshot_hash = sha(snapshot_path)
        metadata = json.loads(snapshot_path.read_text())
        require(metadata["source_manifest_sha256"] == MANIFEST_SHA256
                and metadata["extractor_sha256"] == EXTRACTOR_SHA256
                and metadata["acquirer_sha256"] == ACQUIRER_SHA256
                and sha(subset_path) == metadata["subset_manifest_sha256"], "Snapshot provenance differs")
        original = json.loads((root / "selected_rows.json").read_text())["rows"]
        rows = json.loads(subset_path.read_text())["rows"]
        videos = set(metadata["videos"])
        require(rows == [row for row in original if row["video_id"] in videos],
                "Snapshot is not exactly whole original videos in original order")
        require(len(rows) == metadata["rows"] and 4 * len(rows) == metadata["frames"], "Snapshot counts differ")
        by_video = defaultdict(list)
        for row in rows:
            by_video[row["video_id"]].append(row)
        require(set(by_video) == videos, "Snapshot includes unexpected/missing videos")
        np, recipe, fingerprint, source_hashes = frozen_recipe(root)
        report.update(source_manifest_sha256=MANIFEST_SHA256, subset_manifest_sha256=sha(subset_path),
                      snapshot_sha256=snapshot_hash, source_sha256=source_hashes,
                      runtime_recipe=recipe, expected_cache_fingerprint=fingerprint)
        private_cache = snapshot / "cache"
        require(private_cache.is_dir(), "No private cache directory exists")
        expected_names = {cache_name(video) for video in videos}
        actual_names = {path.name for path in private_cache.glob("*.npz")}
        require(actual_names <= expected_names, f"Unexpected .npz cache files: {sorted(actual_names - expected_names)}")
        report["ignored_temporary_files"] = [str(path) for path in sorted(private_cache.iterdir())
                                             if path.name not in actual_names]
        for video in sorted(videos):
            source = private_cache / cache_name(video)
            if not source.exists():
                report["missing_private_cache_videos"].append(video)
                continue
            require(metadata["videos"][video]["rows"] == len(by_video[video])
                    and metadata["videos"][video]["frames"] == 4 * len(by_video[video]),
                    f"Snapshot per-video counts differ: {video}")
            report["validated"].append(validate_cache(video, source, by_video[video], root,
                                                        metadata["videos"][video], fingerprint, np))
        require(sha(snapshot_path) == snapshot_hash, "Snapshot changed during validation")
        final_cache = root / "frame-cache"
        report["guard_checks"].append(check_quiescent(root, args.warm_service))
        final_cache.mkdir(exist_ok=True)
        flush_report(audit_path, report)
        for validated in report["validated"]:
            report["guard_checks"].append(check_quiescent(root, args.warm_service))
            source = Path(validated["source_cache"])
            destination = final_cache / source.name
            require(sha(source) == validated["source_cache_sha256"], f"Private cache changed before handoff: {source}")
            try:
                # Same-filesystem atomic create: FileExistsError preserves any
                # final writer's entry, even if it appears after the guard.
                os.link(source, destination, follow_symlinks=False)
            except FileExistsError:
                regular_file(destination)
                report["skipped_existing"].append({"video_id": validated["video_id"],
                                                    "destination": str(destination), "sha256": sha(destination)})
            else:
                report["promoted"].append({"video_id": validated["video_id"], "destination": str(destination),
                                           "sha256": validated["source_cache_sha256"], "frames": validated["frames"]})
            flush_report(audit_path, report)
        report["status"] = "complete"
    except BaseException as error:
        report.update(status="failed", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        report.update(finished_utc=utc(), validated_cache_count=len(report["validated"]),
                      promoted_cache_count=len(report["promoted"]),
                      promoted_frame_count=sum(item["frames"] for item in report["promoted"]),
                      skipped_existing_count=len(report["skipped_existing"]))
        flush_report(audit_path, report)
        print(json.dumps({"report": str(audit_path), "status": report["status"],
                          "promoted_caches": report["promoted_cache_count"],
                          "promoted_frames": report["promoted_frame_count"],
                          "skipped_existing": report["skipped_existing_count"]}), flush=True)


if __name__ == "__main__":
    main()
