#!/usr/bin/env python3
"""One admission of DINO/train2014; no scheduler, retry, or resume operation.

Run the outer-manifest-bound copy in a fresh repository-shaped release.
Machine paths and the approved outer/private-input hashes are CLI arguments.
After any dispatch ambiguity, preserve OP and reconcile its existing ticket;
never rerun admission to compensate for a lost SSH response.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys

MODEL = "research/rosetta-model-controls/"
OUTER = MODEL + "release-lock.json"
BACKEND = "research/rosetta-execution/execution.py"
BACKEND_SHA = "332dec331fcfa58ee1fcbc2f30fbd7eee77406fd0992dbba6293338332a4d5fd"
EXTRACTOR_SHA = "2437693c5579b08941517b5b280403112e4d0350cf1e9e9283b1091e40aacb05"
SEEDS = [734796314, 576165995, 2197670066, 839703249, 2584932063]
VISION_ENV = {"PYTHONNOUSERSITE": "1", "ATEN_CPU_CAPABILITY": "avx2",
              "MKL_CBWR": "AVX2,STRICT", "CUBLAS_WORKSPACE_CONFIG": ":4096:8"}
NUMERIC_KEYS = set(VISION_ENV) | {"PYTHONPATH", "PYTHONHOME", "OMP_NUM_THREADS",
    "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "CUDA_VISIBLE_DEVICES", "PYTHONHASHSEED",
    "NUMEXPR_NUM_THREADS", "OMP_DYNAMIC", "MKL_DYNAMIC", "MALLOC_ARENA_MAX"}


def need(value, message):
    if not value:
        raise ValueError(message)


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(path.read_text())


def run(argv):
    return subprocess.run(argv, check=True, capture_output=True, text=True).stdout


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


def verify_release(root, expected):
    outer = root / OUTER
    need(re.fullmatch("[0-9a-f]{64}", expected) and digest(outer) == expected,
         "Outer manifest hash differs")
    lock = read(outer)
    need(lock.get("schema_version") == 1 and lock.get("path_base") == "repository_root"
         and lock.get("status") == "frozen_before_model_control_extraction", "Release is not final")
    need(lock["scope"].get("encoder_extraction_authorized") is True
         and lock["scope"].get("model_control_fits_authorized") is False, "Wrong admission scope")
    files = lock["files"]
    need(isinstance(files, dict) and files and OUTER not in files, "Invalid manifest closure")
    for name, sha in files.items():
        p = Path(name)
        need(not p.is_absolute() and ".." not in p.parts and re.fullmatch("[0-9a-f]{64}", sha),
             "Invalid manifest member")
        path = (root / p).resolve(strict=True)
        need(path.is_relative_to(root) and path.is_file() and digest(path) == sha,
             "Changed release member: " + name)

    def bound(name, expected_sha=None):
        need(name in files and (expected_sha is None or files[name] == expected_sha),
             "Required release binding missing: " + name)
        return root / name

    def by_hash(sha, basename):
        names = sorted(n for n, value in files.items() if value == sha and Path(n).name == basename)
        need(bool(names), "Missing bound record: " + basename)
        return root / names[0]

    need(bound(MODEL + "admit_extraction.py").resolve() == Path(__file__).resolve(),
         "Run the admission copy inside the verified release")
    need(lock["protocol"] == MODEL + "protocol.json" and lock["extraction"] == MODEL + "extraction-lock.json",
         "Outer scientific document paths differ")
    bound(BACKEND, BACKEND_SHA)
    extractor = bound(MODEL + "extract_features.py", EXTRACTOR_SHA)
    protocol_path = bound(MODEL + "protocol.json")
    extraction_path = bound(MODEL + "extraction-lock.json")
    protocol, extraction = read(protocol_path), read(extraction_path)
    need(protocol.get("protocol_id") == extraction.get("protocol_id") == "rosetta-model-controls-v2"
         and protocol.get("status") == "frozen_before_model_control_fits"
         and extraction.get("status") == "frozen_before_model_control_extraction", "Scientific locks are not final")
    need(protocol["seeds"] == SEEDS and protocol["locks"]["extraction"] == files[MODEL + "extraction-lock.json"]
         and protocol["code_sha256"]["extract_features.py"] == extraction["extractor_sha256"] == EXTRACTOR_SHA,
         "Protocol/extraction binding differs")
    for role, basename in [("source", "source-lock.json"), ("environment", "environment.json"),
                           ("baseline_inputs", "inputs-lock.json"), ("baseline_protocol", "protocol.json")]:
        by_hash(protocol["locks"][role], basename)
    evidence = lock["baseline_evidence"]
    baseline_path = bound(evidence["results_lock"], protocol["locks"]["baseline_results"])
    baseline = read(baseline_path)
    need(baseline["status"] == "complete" and set(baseline["results"]) == set(map(str, SEEDS))
         and baseline["baseline_protocol_sha256"] == protocol["locks"]["baseline_protocol"],
         "Incomplete or differently bound baseline")
    by_hash(baseline["baseline_release_lock_sha256"], "release-lock.json")
    comparison_path = bound(evidence["comparison"], baseline["comparison_sha256"])
    comparison = read(comparison_path)
    need(comparison["protocol_sha256"] == baseline["baseline_protocol_sha256"]
         and comparison["release_lock_sha256"] == baseline["baseline_release_lock_sha256"],
         "Baseline comparison binding differs")
    status = read(bound(evidence["status"]))
    need(status.get("status") == "complete" and status["supervisor_status_sha256"] == baseline["supervisor_status_sha256"]
         and status["baseline_results_lock_sha256"] == protocol["locks"]["baseline_results"]
         and status["comparison_sha256"] == baseline["comparison_sha256"], "Completed baseline status is not bound")
    review_path = bound(evidence["discrepancy_review"])
    review = read(review_path)
    need(review.get("status") == "bounded_investigation_complete_cause_unresolved"
         and review["comparison_sha256"] == baseline["comparison_sha256"]
         and review["baseline_results_lock_sha256"] == protocol["locks"]["baseline_results"],
         "Discrepancy review is incomplete or bound to different evidence")
    # The frozen extractor separately enforces all checkpoint/model/data/runtime contracts.
    return lock, protocol_path, extraction_path, extractor, baseline_path, review_path


def quiet_snapshot(unit):
    ancestors = {os.getpid()}
    pid = os.getpid()
    while pid > 1:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        pid = int(fields[1]); ancestors.add(pid)
    stopped, blockers = [], []
    markers = (b"extract_features.py", b"run_cell.py", b"run_source_arm.py", b"run_locked.py",
               b"rosetta-replication-v1", b"rosetta-model-controls-v2", b"model-choice-controls-v1", b"epic")
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit() or int(proc.name) in ancestors:
            continue
        try:
            comm = (proc / "comm").read_text().strip()
            args = (proc / "cmdline").read_bytes()
            fields = (proc / "stat").read_text().rsplit(")", 1)[1].split()
            if comm.lower().startswith(("python", "pixi")) or any(tag in args for tag in markers):
                item = {"pid": int(proc.name), "comm": comm, "state": fields[0], "starttime_ticks": int(fields[19])}
                (stopped if fields[0] in ("T", "t", "Z") else blockers).append(item)
        except (FileNotFoundError, ProcessLookupError):
            continue
    need(not blockers, "Competing or unclassified live scientific/Python process; review without signaling it")
    need(not run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"]).strip(),
         "GPU compute application already present")
    units = run(["systemctl", "--user", "list-units", "--all", "--type=service", "--no-pager", "--no-legend"])
    need(not any("rosetta" in line.lower() and not re.search(r"\b(exited|dead|failed)\b", line)
                 for line in units.splitlines()), "Prior Rosetta service has not ended")
    need(not run(["systemctl", "--user", "list-jobs", "--no-pager", "--no-legend"]).strip(),
         "User service job is queued")
    existing = subprocess.run(["systemctl", "--user", "show", unit, "--property=LoadState", "--value"],
                              text=True, capture_output=True)
    need(existing.returncode in (0, 4) and existing.stdout.strip() == "not-found", "Unit name was already used")
    manager = dict(line.split("=", 1) for line in run(["systemctl", "--user", "show-environment"]).splitlines() if "=" in line)
    for origin, env in [("caller", os.environ), ("manager", manager)]:
        need(all(key not in env or (key in VISION_ENV and env[key] == VISION_ENV[key]) for key in NUMERIC_KEYS),
             "Unreviewed numerical/Python environment in " + origin)
    mem = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
    return {"stopped_or_zombie_processes_retained": stopped, "mem_available": mem["MemAvailable"].strip(),
            "gpu_compute_applications": 0, "scope": "Point-in-time process/service observation; stopped EPIC is not terminated."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("release-root", "private-inputs", "data-root", "output", "op", "python"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("outer-sha256", "private-inputs-sha256", "unit"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    for name in ("release_root", "private_inputs", "data_root", "output", "op", "python"):
        p = getattr(args, name)
        need(p.is_absolute(), "Every machine path must be absolute")
        if name in ("output", "op"):
            need(not p.exists() and not p.is_symlink(), "Attempt path already exists")
        # A venv interpreter can be a symlink; resolving it loses that environment.
        setattr(args, name, p if name == "python" else p.resolve())
    need(re.fullmatch(r"rosetta-[a-z0-9-]+\.service", args.unit), "Invalid unique Rosetta unit name")
    need(not args.output.exists() and not args.op.exists(), "Output or operational attempt already exists; do not repeat admission")
    need(not any(a.is_relative_to(b) or b.is_relative_to(a) for a, b in
                 [(args.output, args.op), (args.output, args.release_root), (args.op, args.release_root),
                  (args.output, args.data_root), (args.op, args.data_root)]), "Attempt/release/data paths overlap")
    need(args.data_root.is_dir() and os.access(args.python, os.X_OK), "Missing dataset/runtime")
    need(re.fullmatch("[0-9a-f]{64}", args.private_inputs_sha256)
         and digest(args.private_inputs) == args.private_inputs_sha256, "Private input-path binding changed")
    lock, protocol, extraction, extractor, baseline, review = verify_release(args.release_root, args.outer_sha256)
    backend_module = load(args.release_root / BACKEND, "_rosetta_admission_backend")
    backend = backend_module.Systemd(); backend.gate()
    env_argv = ["/usr/bin/env", *[key + "=" + value for key, value in VISION_ENV.items()], str(args.python)]
    # Metadata-only original extractor gate: no Torch import, image read or inference.
    preflight = "import importlib.util,sys; s=importlib.util.spec_from_file_location('_preflight',sys.argv[1]); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); m.extraction_lock(sys.argv[2],sys.argv[3],'dino')"
    run([*env_argv, "-B", "-c", preflight, str(extractor), str(extraction), digest(extraction)])
    verify_release(args.release_root, args.outer_sha256)
    need(digest(args.private_inputs) == args.private_inputs_sha256 and not args.output.exists(), "Inputs/output changed")
    observation = quiet_snapshot(args.unit)
    disk = os.statvfs(args.release_root)
    observation["disk_available_bytes"] = disk.f_bavail * disk.f_frsize
    argv = [*env_argv, str(extractor), "extract", "--model", "dino", "--split", "train2014",
            "--inputs", str(args.private_inputs), "--data-root", str(args.data_root), "--lock", str(extraction),
            "--lock-sha256", digest(extraction), "--out", str(args.output)]
    backend_module.ensure_dir(args.op.parent)
    args.op.mkdir(exist_ok=False)
    backend_module.sync_dir(args.op.parent)
    ticket = {"unit": args.unit, "argv": argv, "cwd": str(args.release_root),
        "stdout": str(args.op / "stdout.log"), "stderr": str(args.op / "stderr.log"),
        "model": "dino", "split": "train2014", "output": str(args.output),
        "outer_sha256": args.outer_sha256, "private_inputs_sha256": args.private_inputs_sha256,
        "protocol_sha256": digest(protocol), "lock_sha256": digest(extraction), "baseline_results_sha256": digest(baseline),
        "discrepancy_review_sha256": digest(review), "extractor_sha256": EXTRACTOR_SHA,
        "backend_sha256": BACKEND_SHA, "admission_sha256": digest(Path(__file__)), "at_utc": backend_module.stamp()}
    backend_module.write_new(args.op / "admission.json", {"at_utc": backend_module.stamp(), **observation})
    backend_module.write_new(args.op / "ticket.json", ticket)
    backend_module.write_new(args.op / "dispatch.json", {"ticket_sha256": backend_module.object_hash(ticket), "at_utc": backend_module.stamp()})
    try:
        backend.launch(ticket)  # The only launch call. Never retry after uncertainty.
        state, observed = backend_module.reconcile_service(args.op, backend)
    except Exception as error:
        backend_module.write_new(args.op / "admission-error.json", {"at_utc": backend_module.stamp(),
            "error": str(error), "instruction": "Preserve this attempt and reconcile its existing ticket. Do not launch again."})
        raise
    print(json.dumps({"unit": args.unit, "state": state, "invocation_id": observed["invocation_id"],
                      "op": str(args.op), "scientific_completion_claimed": False}))


if __name__ == "__main__":
    main()
