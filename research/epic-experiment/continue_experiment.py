#!/usr/bin/env python3
"""Supervise the frozen EPIC experiment in its VM; default is a read-only plan.

Use --run to wait for the existing acquisition service and then validate frames,
extract every selected feature, run all 17 fits, and independently verify them.
After a terminal acquisition failure it may make one bounded repair attempt,
with the exact same frozen script and selection, preserving prior reports.
It never restarts a live job and never enables partial data.
--self-test exercises state transitions and concurrency guards using mocks.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import shutil
import subprocess
import sys
import tempfile
import time


FROZEN = {
    "selected_rows.json": "d1a34b2ad47a1456a011c6e4f2d6308bae34ebf6aed0c6fb335794ac3dc648be",
    "protocol.json": "656a4fad35609bb753fb5045268e3448e4c71a36bce70a4b20050fd16a547370",
    "acquire_frames.py": "c3025e54c7f8db18fc7a883f1373f1ffaf8707d77a29371acb77735288d55f65",
    "verify_acquisition.py": "fadbafc2d07b2b4e1bb146cbf7b6ba3e82c093b6d83015702140053ba6933e6a",
    "extract_features.py": "471ede5f611b17ab63252957c0f9785ff2afede22a656f8452fdadff4217f203",
    "run_experiment.py": "5806ce2c397d2c1a473e2422a41f850d332f453237dc785fe09d1d077798b26d",
    "verify_results.py": "5c7ac4df0fe8991b7a3fa7d664c0ff89891f41925d3b444d7de0395e463402a8",
}
UPSTREAM_SHA256 = "07491e636bd9fb37aa80ec8bc7752d418657b54e53f6fec1c75add4c8e4a15a5"
EXPECTED_ROWS, EXPECTED_VIDEOS, EXPECTED_FRAMES = 9392, 268, 37568
STAGES = ("acquisition_wait", "acquisition_verify", "features", "alignment", "results_verify")
ACTIVE_STATES = {"active", "activating", "deactivating", "reloading", "refreshing"}


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, value):
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


@contextmanager
def exclusive_lock(path):
    """Acquire before writing status, so a second supervisor cannot alter it."""
    with path.open("a+") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Another experiment supervisor holds the lock") from error
        handle.seek(0)
        handle.truncate()
        handle.write(json.dumps({"pid": os.getpid(), "started_utc": utc()}) + "\n")
        handle.flush()
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def process_identity(pid):
    """Linux PID identity guards against PID reuse when resuming an old state."""
    try:
        path = Path("/proc") / str(pid)
        fields = (path / "stat").read_text().rsplit(")", 1)[1].split()
        return {"pid": int(pid), "start_ticks": int(fields[19]),
                "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip()}
    except (OSError, ValueError, IndexError):
        return None


def reject_live_previous_child(previous):
    for name, stage in previous.get("stages", {}).items():
        identity = stage.get("child_identity")
        if stage.get("state") == "running" and identity and process_identity(identity["pid"]) == identity:
            raise RuntimeError(f"Previous {name} child {identity['pid']} is still alive; do not launch a duplicate")


class State:
    def __init__(self, root, plan):
        self.path = root / "pipeline-status.json"
        self.events = root / "pipeline-events.jsonl"
        previous = json.loads(self.path.read_text()) if self.path.exists() else {}
        reject_live_previous_child(previous)
        self.attempt = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        self.directory = root / "pipeline-history" / self.attempt
        self.directory.mkdir(parents=True)
        if previous:
            atomic_json(self.directory / "previous-status.json", previous)
        self.value = {"schema_version": 1, "attempt": self.attempt, "status": "running",
                      "started_utc": utc(), "supervisor_pid": os.getpid(), "supervisor_sha256": sha(__file__),
                      "plan": plan, "stages": {name: {"state": "pending"} for name in STAGES}}
        self.flush("supervisor_started")

    def flush(self, event):
        self.value["updated_utc"] = utc()
        atomic_json(self.path, self.value)
        atomic_json(self.directory / "status.json", self.value)
        with self.events.open("a") as stream:
            stream.write(json.dumps({"at": utc(), "attempt": self.attempt, "event": event,
                                     "status": self.value["status"], "stage": self.value.get("current_stage")}) + "\n")
            stream.flush()
            os.fsync(stream.fileno())

    def stage(self, name, state, **details):
        self.value["current_stage"] = name
        self.value["stages"].setdefault(name, {}).update(state=state, updated_utc=utc(), **details)
        self.flush(f"{name}:{state}")

    def finish(self, status, **details):
        self.value.update(status=status, finished_utc=utc(), **details)
        self.flush(f"supervisor:{status}")


def service_snapshot(name):
    try:
        result = subprocess.run(["systemctl", "show", name, "--no-pager", "-p", "LoadState", "-p", "ActiveState",
                                 "-p", "SubState", "-p", "MainPID", "-p", "Result", "-p", "ExecMainStatus"],
                                capture_output=True, text=True, timeout=20)
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"observation_error": f"{type(error).__name__}: {error}"}
    if result.returncode != 0:
        return {"observation_error": result.stderr.strip() or f"systemctl exited {result.returncode}"}
    data = dict(line.split("=", 1) for line in result.stdout.splitlines() if "=" in line)
    if "LoadState" not in data or "ActiveState" not in data:
        return {"observation_error": "systemctl response lacks required state fields"}
    return data


def acquisition_inventory(root):
    manifest = json.loads((root / "selected_rows.json").read_text())
    videos = {row["video_id"] for row in manifest["rows"]}
    counts = {"complete": 0, "running": 0, "failed": 0, "missing": 0, "invalid": 0}
    problems = []
    for video in sorted(videos):
        path = root / "data" / "acquisition" / (video + ".json")
        if not path.exists():
            counts["missing"] += 1
            continue
        try:
            report = json.loads(path.read_text())
            if (report.get("video_id") != video or report.get("manifest_sha256") != FROZEN["selected_rows.json"]
                    or report.get("script_sha256") != FROZEN["acquire_frames.py"]):
                counts["invalid"] += 1
                problems.append({"video": video, "reason": "report provenance mismatch"})
                continue
            status = report.get("status", "invalid")
            counts[status if status in counts else "invalid"] += 1
            if status == "failed":
                problems.append({"video": video, "reason": report.get("error")})
        except (OSError, json.JSONDecodeError) as error:
            counts["invalid"] += 1
            problems.append({"video": video, "reason": str(error)})
    return {"expected_videos": len(videos), "counts": counts, "problems": problems,
            "all_complete": len(videos) == EXPECTED_VIDEOS and counts["complete"] == len(videos)}


def wait_decision(snapshot, inventory):
    """Missing observations are not terminal; missing units need all reports."""
    if "observation_error" in snapshot:
        return "wait"
    if snapshot["ActiveState"] in ACTIVE_STATES:
        return "wait"
    if snapshot["ActiveState"] == "failed":
        return "fail"
    if snapshot["ActiveState"] != "inactive":
        return "wait"
    # Even after a disappeared transient unit, this merely permits independent
    # byte/frame verification; it does not certify acquisition success.
    return "verify" if inventory["all_complete"] else "fail"


def wait_acquisition(args, state, observer=service_snapshot, inventory=acquisition_inventory, sleeper=time.sleep):
    started = time.monotonic()
    retries_used = 0
    while True:
        snapshot = observer(args.service)
        records = inventory(args.root)
        decision = wait_decision(snapshot, records)
        state.stage("acquisition_wait", "waiting", service=snapshot, inventory=records,
                    observation_utc=utc(), elapsed_seconds=time.monotonic() - started)
        if decision == "verify":
            state.stage("acquisition_wait", "passed", qualification="All current-version reports claim completion; independent verification still required")
            return
        if decision == "fail":
            if retries_used >= args.acquisition_retries:
                raise RuntimeError("Acquisition is terminal without a complete successful current-version inventory; bounded retry budget exhausted; no downstream steps started")
            retries_used += 1
            if retry_acquisition(args, state, retries_used):
                state.stage("acquisition_wait", "passed", retries_used=retries_used,
                            qualification="Frozen-script repair completed every report; independent verification still required")
                return
            continue
        if args.max_wait_seconds and time.monotonic() - started >= args.max_wait_seconds:
            raise TimeoutError("Acquisition wait limit reached; acquisition was not restarted or declared failed")
        sleeper(args.poll_seconds)


def preflight(args):
    for name, expected in FROZEN.items():
        require(sha(args.root / name) == expected, f"Frozen file changed: {name}")
    files = {str(p.relative_to(args.upstream)): sha(p) for p in sorted(args.upstream.rglob("*.py"))}
    tree_hash = hashlib.sha256(json.dumps(files, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    require(tree_hash == UPSTREAM_SHA256, "Upstream Rosetta source differs from the inspected implementation")
    for executable in (args.gpu_python, args.solver_python):
        require(executable.is_file() and os.access(executable, os.X_OK), f"Missing Python runtime: {executable}")
    for path in (args.models / "dino_vitbase16_pretrain.pth", args.models / "glove.6B.zip",
                 args.root / "dino-source" / "vision_transformer.py", args.root / "dino-source" / "utils.py"):
        require(path.is_file(), f"Missing required input: {path}")
    rows = json.loads((args.root / "selected_rows.json").read_text())["rows"]
    require(len(rows) == EXPECTED_ROWS and len({r["video_id"] for r in rows}) == EXPECTED_VIDEOS,
            "Frozen selection counts differ")


def feature_reuse_allowed(root):
    feature, metadata = root / "features.npz", root / "features.json"
    if not feature.exists() and not metadata.exists():
        return False
    require(feature.is_file() and metadata.is_file(), "Partial feature artifact exists; preserve and investigate it before retrying")
    data = json.loads(metadata.read_text())
    selection, eligibility = data.get("selection", {}), data.get("eligibility", {})
    require(selection.get("is_final") is True and selection.get("allow_partial") is False,
            "Existing features are explicitly nonfinal; refusing overwrite or use")
    require(selection.get("selected_rows") == selection.get("output_rows") == EXPECTED_ROWS
            and selection.get("missing_rows") == [], "Existing features omit frozen rows")
    require(eligibility.get("eligible") == EXPECTED_ROWS and eligibility.get("ineligible") == 0,
            "Existing features contain invalid rows; no filtering permitted")
    require(data.get("script_sha256") == FROZEN["extract_features.py"]
            and data.get("manifest_sha256") == FROZEN["selected_rows.json"], "Existing feature provenance differs")
    require(data.get("feature_file_sha256") == sha(feature), "Existing feature bytes differ from sidecar")
    # The frozen runner additionally opens the NPZ, validates bool[N] eligibility,
    # finite/nonzero values, exact UID order and disjoint participant membership.
    return True


def build_plan(args):
    root, manifest = args.root, args.root / "selected_rows.json"
    feature = root / "features.npz"
    commands = {
        "acquisition_verify": [args.gpu_python, root / "verify_acquisition.py", "--manifest", manifest,
                               "--data", root / "data", "--acquirer", root / "acquire_frames.py",
                               "--output", root / "acquisition-validation.json"],
        "features": [args.gpu_python, root / "extract_features.py", "--manifest", manifest,
                     "--frame-root", root / "data", "--dino-source", root / "dino-source",
                     "--dino-checkpoint", args.models / "dino_vitbase16_pretrain.pth",
                     "--glove", args.models / "glove.6B.zip", "--output", feature,
                     "--cache-dir", root / "frame-cache", "--batch-size", "64", "--device", "cuda"],
        "alignment": [args.solver_python, root / "run_experiment.py", "--manifest", manifest,
                      "--features", feature, "--output", root / "results", "--seeds", "0", "1", "2",
                      "--clusters", "30", "--restarts", "30", "--iterations", "100", "--batch-size", "2048",
                      "--workers", "4", "--threads", "4", "--ridge-alpha", "1.0", "--permutations", "999"],
        "results_verify": [args.solver_python, root / "verify_results.py", "--manifest", manifest,
                           "--protocol", root / "protocol.json", "--features", feature, "--results", root / "results",
                           "--runner", root / "run_experiment.py", "--upstream-source", args.upstream,
                           "--output", root / "verification.json"],
    }
    commands["acquisition_retry"] = [args.gpu_python, root / "acquire_frames.py", "--manifest", manifest,
                                     "--output", root / "data", "--workers", "12"]
    return {"root": str(root), "acquisition_service": args.service, "expected_rows": EXPECTED_ROWS,
            "expected_videos": EXPECTED_VIDEOS, "expected_frames": EXPECTED_FRAMES,
            "expected_fits": 17, "frozen_sha256": FROZEN, "upstream_source_sha256": UPSTREAM_SHA256,
            "commands": {name: list(map(str, command)) for name, command in commands.items()},
            "acquisition_retries": args.acquisition_retries,
            "resume_policy": "Never restart a live acquisition; bounded same-code repair only after terminal failure, with reports archived; fully verified final features reused; map fits resume only through the frozen runner's hash checks",
            "observation_policy": "Poll existing systemd unit; transient observation errors keep waiting; missing unit still requires every report and independent verification"}


def reject_competing_process(command):
    """Do not collide with a manually launched extractor, runner or verifier."""
    script = command[1].encode()
    for path in Path("/proc").glob("[0-9]*/cmdline"):
        try:
            arguments = path.read_bytes().split(b"\0")
            if script in arguments:
                raise RuntimeError(f"An existing process is already running {command[1]}: PID {path.parent.name}")
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            pass


def execute_stage(name, command, args, state, launcher=subprocess.Popen, allow_nonzero=False):
    preflight(args)
    reject_competing_process(command)
    output_path = (args.root / "acquisition-validation.json" if name == "acquisition_verify" else
                   args.root / "verification.json" if name == "results_verify" else None)
    if output_path and output_path.exists():
        # Retain prior validation evidence before its verifier refreshes it.
        (state.directory / ("previous-" + output_path.name)).write_bytes(output_path.read_bytes())
    log = state.directory / (name + ".log")
    environment = os.environ.copy()
    environment.update(PYTHONUNBUFFERED="1", OMP_NUM_THREADS="4", MKL_NUM_THREADS="4",
                       PYTHONPATH=str(args.upstream.parent), CUBLAS_WORKSPACE_CONFIG=":4096:8")
    state.stage(name, "running", command=command, log=str(log), started_utc=utc())
    with log.open("a") as stream:
        child = launcher(command, stdout=stream, stderr=subprocess.STDOUT, cwd=args.root,
                         env=environment, start_new_session=True)
        state.stage(name, "running", child_pid=child.pid, child_identity=process_identity(child.pid))
        try:
            returncode = child.wait()
        except BaseException:
            # Only the newly launched child process group belongs to this
            # supervisor. Acquisition and unrelated jobs are never signaled.
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
            raise
    state.stage(name, "running", returncode=returncode, child_finished_utc=utc())
    if not allow_nonzero:
        require(returncode == 0, f"{name} exited {returncode}; see {log}")
    return returncode


def retry_acquisition(args, state, number):
    """A terminal transport repair, preserving all earlier evidence and rows."""
    preflight(args)
    archive = state.directory / f"acquisition-before-retry-{number}"
    archive.mkdir()
    reports = args.root / "data" / "acquisition"
    if reports.is_dir():
        shutil.copytree(reports, archive / "reports")
    progress = args.root / "data" / "acquisition-progress.json"
    if progress.exists():
        shutil.copyfile(progress, archive / "acquisition-progress.json")
    atomic_json(archive / "retry-reason.json", {
        "recorded_utc": utc(), "prior_wait_state": state.value["stages"]["acquisition_wait"],
        "acquirer_sha256": FROZEN["acquire_frames.py"], "manifest_sha256": FROZEN["selected_rows.json"],
        "policy": "Identical code and complete frozen selection. Acquirer itself hashes/reuses complete videos; only incomplete/mismatching videos are fetched again. No substitution or filtering.",
    })
    name = f"acquisition_retry_{number}"
    command = state.value["plan"]["commands"]["acquisition_retry"]
    code = execute_stage(name, command, args, state, allow_nonzero=True)
    records = acquisition_inventory(args.root)
    succeeded = code == 0 and records["all_complete"]
    state.stage(name, "passed" if succeeded else "failed", returncode=code, inventory=records,
                prior_evidence_archive=str(archive))
    return succeeded


def validate_stage(name, root):
    if name == "acquisition_verify":
        data = json.loads((root / "acquisition-validation.json").read_text())
        require(data.get("status") == "passed" and data.get("rows") == EXPECTED_ROWS
                and data.get("videos") == EXPECTED_VIDEOS and data.get("frames") == EXPECTED_FRAMES,
                "Acquisition verifier did not pass every selected frame")
        require(data.get("manifest_sha256") == FROZEN["selected_rows.json"]
                and data.get("acquirer_sha256") == FROZEN["acquire_frames.py"]
                and data.get("verifier_sha256") == FROZEN["verify_acquisition.py"], "Acquisition verification provenance differs")
        return {"verification_sha256": sha(root / "acquisition-validation.json")}
    if name == "features":
        require(feature_reuse_allowed(root), "Feature extraction produced no complete artifact")
        return {"features_sha256": sha(root / "features.npz"), "metadata_sha256": sha(root / "features.json")}
    if name == "alignment":
        data = json.loads((root / "results" / "summary.json").read_text())
        require(data.get("status") == "complete" and data.get("completed_fits") == 17,
                "Alignment runner did not finish all 17 required fits")
        return {"summary_sha256": sha(root / "results" / "summary.json"),
                "qualification": "Independent verification still pending"}
    if name == "results_verify":
        data = json.loads((root / "verification.json").read_text())
        require(data.get("passed") is True, "Independent result verification did not pass")
        require(data.get("verifier_sha256") == FROZEN["verify_results.py"], "Result verifier provenance differs")
        require(data.get("manifest_sha256") == FROZEN["selected_rows.json"]
                and data.get("checks", {}).get("completed_fits") == 17
                and data.get("checks", {}).get("full_score_matrices_replayed") == 34,
                "Independent verification did not cover the full frozen experiment")
        return {"verification_sha256": sha(root / "verification.json")}
    raise ValueError(name)


def supervise(args):
    require(args.root.is_dir(), "Experiment directory does not exist")
    with exclusive_lock(args.root / ".continue-experiment.lock"):
        preflight(args)
        state = State(args.root, build_plan(args))
        try:
            wait_acquisition(args, state)
            for name in STAGES[1:]:
                if name == "features" and feature_reuse_allowed(args.root):
                    state.stage(name, "passed", reused=True, **validate_stage(name, args.root))
                    continue
                execute_stage(name, state.value["plan"]["commands"][name], args, state)
                state.stage(name, "passed", **validate_stage(name, args.root))
            state.finish("complete", conclusion="All selected frames, full frozen features, 17 fits and independent result verification completed")
        except BaseException as error:
            stage = state.value.get("current_stage")
            status = "interrupted" if isinstance(error, (KeyboardInterrupt, InterruptedError)) else "failed"
            if stage:
                state.stage(stage, status, error_type=type(error).__name__, error=str(error))
            state.finish(status, error_type=type(error).__name__, error=str(error))
            raise


def self_test():
    import unittest.mock as mock
    incomplete = {"all_complete": False, "counts": {"complete": 2}}
    complete = {"all_complete": True, "counts": {"complete": EXPECTED_VIDEOS}}
    missing = {"LoadState": "not-found", "ActiveState": "inactive", "SubState": "dead"}
    active = {"LoadState": "loaded", "ActiveState": "active", "SubState": "running"}
    assert wait_decision(active, complete) == "wait"
    assert wait_decision({"observation_error": "temporary bus failure"}, complete) == "wait"
    assert wait_decision(missing, incomplete) == "fail"
    assert wait_decision(missing, complete) == "verify"
    assert wait_decision({"LoadState": "loaded", "ActiveState": "failed"}, complete) == "fail"
    with mock.patch(__name__ + ".subprocess.run", side_effect=subprocess.TimeoutExpired("systemctl", 20)):
        assert "observation_error" in service_snapshot("test.service")
    with tempfile.TemporaryDirectory(prefix="epic-supervisor-test-") as temporary:
        root = Path(temporary)
        args = argparse.Namespace(root=root, service="test.service", poll_seconds=0, max_wait_seconds=0, acquisition_retries=1)
        state = State(root, {"mock": True})
        observer = mock.Mock(side_effect=[active, {"observation_error": "timeout"}, missing])
        inventory = mock.Mock(side_effect=[incomplete, incomplete, complete])
        sleeper = mock.Mock()
        wait_acquisition(args, state, observer=observer, inventory=inventory, sleeper=sleeper)
        assert state.value["stages"]["acquisition_wait"]["state"] == "passed" and sleeper.call_count == 2
        with mock.patch(__name__ + ".retry_acquisition", return_value=True) as repair:
            wait_acquisition(args, state, observer=mock.Mock(return_value=missing),
                             inventory=mock.Mock(return_value=incomplete), sleeper=mock.Mock())
            assert repair.call_count == 1
        with mock.patch(__name__ + ".retry_acquisition", return_value=False) as repair:
            try:
                wait_acquisition(args, state, observer=mock.Mock(return_value=missing),
                                 inventory=mock.Mock(return_value=incomplete), sleeper=mock.Mock())
            except RuntimeError:
                pass
            else:
                raise AssertionError("Exhausted acquisition repair budget was ignored")
            assert repair.call_count == 1
        with exclusive_lock(root / "lock"):
            try:
                with exclusive_lock(root / "lock"):
                    raise AssertionError("Concurrent lock was accepted")
            except RuntimeError:
                pass
        child = {"pid": 123, "start_ticks": 456, "boot_id": "test"}
        prior = {"stages": {"alignment": {"state": "running", "child_identity": child}}}
        with mock.patch(__name__ + ".process_identity", return_value=child):
            try:
                reject_live_previous_child(prior)
            except RuntimeError:
                pass
            else:
                raise AssertionError("Live prior child was accepted")
        with mock.patch(__name__ + ".process_identity", return_value=None):
            reject_live_previous_child(prior)
        # A stage cannot pass on subprocess success alone; its output gate is
        # checked separately. Verify failure prevents later stages from running.
        plan = {"commands": {name: ["mock", name] for name in STAGES[1:]}}
        with mock.patch(__name__ + ".preflight"), mock.patch(__name__ + ".build_plan", return_value=plan), \
                mock.patch(__name__ + ".wait_acquisition"), mock.patch(__name__ + ".feature_reuse_allowed", return_value=False), \
                mock.patch(__name__ + ".execute_stage") as execute, \
                mock.patch(__name__ + ".validate_stage", side_effect=RuntimeError("deliberate validation failure")):
            try:
                supervise(args)
            except RuntimeError:
                pass
            else:
                raise AssertionError("Validation failure was ignored")
            assert execute.call_count == 1
            assert json.loads((root / "pipeline-status.json").read_text())["status"] == "failed"
        with mock.patch(__name__ + ".preflight"), mock.patch(__name__ + ".build_plan", return_value=plan), \
                mock.patch(__name__ + ".wait_acquisition"), mock.patch(__name__ + ".feature_reuse_allowed", return_value=False), \
                mock.patch(__name__ + ".execute_stage") as execute, mock.patch(__name__ + ".validate_stage", return_value={}):
            supervise(args)
            assert [call.args[0] for call in execute.call_args_list] == list(STAGES[1:])
            assert json.loads((root / "pipeline-status.json").read_text())["status"] == "complete"
    print(json.dumps({"self_test": "passed", "checks": ["live job remains waiting", "observation error is not terminal",
                      "missing unit cannot bypass complete-report gate", "failed unit is rejected", "exclusive flock",
                      "orphan child identity guard", "bounded same-selection terminal repair",
                      "verification failure stops downstream", "ordered complete pipeline"]}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--run", action="store_true", help="Execute in the VM; otherwise print the plan without writes")
    mode.add_argument("--dry-run", action="store_true", help="Explicit form of the default read-only plan")
    mode.add_argument("--self-test", action="store_true", help="Run local mock tests without touching the VM")
    parser.add_argument("--root", type=Path, default=Path("/opt/rosetta/epic-experiment"))
    parser.add_argument("--models", type=Path, default=Path("/opt/rosetta/clean-example/models"))
    parser.add_argument("--gpu-python", type=Path, default=Path("/opt/rosetta/clean-example/.gpu-venv/bin/python"))
    parser.add_argument("--solver-python", type=Path, default=Path("/opt/rosetta/.venv/bin/python"))
    parser.add_argument("--upstream", type=Path, default=Path("/opt/rosetta/repo/unpaired_rosetta"))
    parser.add_argument("--service", default="rosetta-epic-acquire-v2.service")
    parser.add_argument("--poll-seconds", type=float, default=30)
    parser.add_argument("--acquisition-retries", type=int, default=1, help="Bounded same-code repair attempts after terminal failure; 0 disables repair")
    parser.add_argument("--max-wait-seconds", type=float, default=0, help="0 waits without a deadline; positive limit exits without touching acquisition")
    args = parser.parse_args()
    if args.self_test:
        self_test()
    elif args.run:
        if args.poll_seconds <= 0 or args.max_wait_seconds < 0 or args.acquisition_retries < 0:
            parser.error("Polling interval must be positive; wait limit and retry count must be nonnegative")
        signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(InterruptedError("Supervisor received SIGTERM")))
        supervise(args)
    else:
        print(json.dumps({"mode": "dry-run", "validation_performed": False, "plan": build_plan(args)}, indent=2))


if __name__ == "__main__":
    main()
