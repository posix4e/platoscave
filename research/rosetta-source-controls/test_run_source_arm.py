"""Synthetic operational checks only; no optimization or published cache inputs."""

from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import run_source_arm as runner
import source_masks as masks
import source_primitives as primitives


def protocol():
    return {"schema_version": 1, "protocol_id": runner.PROTOCOL_ID, "status": runner.STATUS,
            "upstream_commit": runner.UPSTREAM, "seeds": runner.SEEDS, "counts": runner.COUNTS,
            "arm_counts": runner.ARM_COUNTS, "algorithm_defaults": runner.ALGORITHM, "execution": runner.EXECUTION,
            "source_archive_sha256": runner.ARCHIVES, "row_identity_sha256": runner.ROW_IDENTITY,
            "strata_counts": runner.STRATA_COUNTS, "metric_rules": runner.METRIC_RULES,
            "selection_seeds": {"Rv": [1000, 1001], "Rp": [2000, 2001]},
            "code_sha256": {k: runner.HELPER_SHA256 if k == "run_cell.py" else "a" * 64 for k in runner.CODE},
            "locks": {k: "b" * 64 for k in runner.LOCKS}}


@pytest.mark.parametrize("field,value", [("status", "draft"), ("counts", {"train_x": 100}),
    ("algorithm_defaults", {"num_clusters": 3}), ("execution", {"num_workers": 1}),
    ("seeds", [734796314]), ("metric_rules", {"foscttm_denominator": 3334}),
    ("selection_seeds", {"Rv": [], "Rp": [2000]}), ("selection_seeds", {"Rv": [1000], "Rp": [1000]})])
def test_changed_protocol_or_implicit_schedule_rejected(field, value):
    p = protocol()
    runner.check_protocol(p)
    p[field] = value
    with pytest.raises(ValueError):
        runner.check_protocol(p)


def small_rows():
    return masks.SourceRows(("coco:10", "coco:20"),
                            ("coco:10", "vg:1", "coco:30", "vg:2", "coco:20", "vg:3", "vg:4", "vg:5"),
                            ("coco:30", "coco:40", "coco:50"), (100, 1, 103, 2, 102, 3, 4, 5))


def small_bundle(p, rows):
    selected = masks.build_masks(rows, validation_seeds=tuple(p["selection_seeds"]["Rv"]), pairing_seeds=tuple(p["selection_seeds"]["Rp"]))
    values = {}
    for mask in selected:
        short = {masks.ORIGINAL: "O", masks.VALIDATION: "C", masks.BOTH: "D",
                 masks.RANDOM_VALIDATION: "Rv", masks.RANDOM_PAIRING: "Rp"}[mask.variant]
        key = short if mask.selection_seed is None else f"{short}-{mask.selection_seed}"
        values[key] = {"variant": mask.variant, "selection_seed": mask.selection_seed,
                       "row_identity_sha256": mask.row_identity_sha256, "train_y_indices": list(mask.train_y_indices),
                       "mask_sha256": mask.mask_sha256}
    return {"schema_version": 1, "protocol_id": runner.PROTOCOL_ID, "status": runner.STATUS,
            "protocol_sha256": "c" * 64, "row_identity_sha256": rows.identity_sha256, "masks": values,
            "strata": {k: list(v) for k, v in masks.fixed_strata(rows).items()}}


def test_complete_bundle_requires_every_mask_and_immutable_original_strata(monkeypatch):
    p, rows = protocol(), small_rows()
    bundle = small_bundle(p, rows)
    monkeypatch.setattr(runner, "ARM_COUNTS", {"O": 8, "C": 7, "D": 5, "Rv": 7, "Rp": 5})
    monkeypatch.setattr(runner, "STRATA_COUNTS", {"never_known_spc_overlap": 2, "original_spc_overlap": 1, "all": 3})
    assert len(runner.check_bundle(bundle, p, "c" * 64, rows, masks)) == 7
    for mutation in ("missing", "extra", "changed_seed", "changed_rows", "redefined_exposure", "draft", "wrong_protocol"):
        bad = deepcopy(bundle)
        if mutation == "missing": del bad["masks"]["Rp-2001"]
        if mutation == "extra": bad["masks"]["Rv-3000"] = deepcopy(bad["masks"]["Rv-1000"])
        if mutation == "changed_seed": bad["masks"]["Rv-1000"]["selection_seed"] = 17
        if mutation == "changed_rows": bad["masks"]["D"]["train_y_indices"] = [0, 1, 3, 5, 6]
        if mutation == "redefined_exposure": bad["strata"]["original_spc_overlap"] = []
        if mutation == "draft": bad["status"] = "draft"
        if mutation == "wrong_protocol": bad["protocol_sha256"] = "d" * 64
        with pytest.raises(ValueError):
            runner.check_bundle(bad, p, "c" * 64, rows, masks)


def args_fixture(tmp_path, p=None):
    p = protocol() if p is None else p
    path = tmp_path / "protocol.json"
    path.write_text(json.dumps(p))
    bundle = tmp_path / "bundle.json"
    bundle.write_text("{}")
    return SimpleNamespace(protocol=path, protocol_sha256=runner.r.digest(path), masks=bundle,
                           masks_sha256=runner.r.digest(bundle), bindings=tmp_path / "private-bindings.json",
                           mask="O", seed=runner.SEEDS[0], out=tmp_path / "attempt-01", private_log_dir=tmp_path / "private-01")


@pytest.mark.parametrize("failure", ["draft", "protocol_hash", "bundle_hash", "missing_bindings"])
def test_preflight_failures_have_fresh_attempt_receipts_and_never_fit(tmp_path, monkeypatch, failure):
    p = protocol()
    if failure == "draft": p["status"] = "draft"
    args = args_fixture(tmp_path, p)
    if failure == "protocol_hash": args.protocol_sha256 = "0" * 64
    if failure == "bundle_hash": args.masks_sha256 = "0" * 64
    called = []
    monkeypatch.setattr(runner, "execute", lambda *a: called.append(a))
    with pytest.raises((ValueError, FileNotFoundError)):
        runner.run(args)
    assert not called
    assert (args.out / "started.json").is_file()
    assert (args.out / "FAILED.json").is_file() and not (args.out / "COMPLETE.json").exists()
    assert (args.private_log_dir / "attempt.traceback.txt").is_file()
    failed = runner.r.read_json(args.out / "failure.json")
    assert failed["failure_stage"] == "preflight"
    assert str(tmp_path) not in (args.out / "failure.json").read_text()


def synthetic_data(tmp_path, monkeypatch):
    rng = torch.Generator().manual_seed(42)
    paths = {}
    for name, shape in (("train_x", (17, 4)), ("train_y", (11, 1, 4)), ("validation_x", (9, 4)), ("validation_y", (9, 3, 4))):
        value = torch.randn(shape, generator=rng).bfloat16()
        if name == "validation_y": value[::2, -1] = float("nan")
        paths[name] = tmp_path / (name + ".pt")
        torch.save(value, paths[name])
    rows = masks.SourceRows(tuple(f"coco:{i}" for i in range(100, 117)), tuple(f"vg:{i}" for i in range(11)),
                            tuple(f"coco:{i}" for i in range(200, 209)), tuple(range(11)))
    mask = masks.SourceMask(masks.ORIGINAL, None, rows.identity_sha256, tuple(range(11)))
    strata = {"never_known_spc_overlap": torch.ones(9, dtype=torch.bool), "original_spc_overlap": torch.zeros(9, dtype=torch.bool), "all": torch.ones(9, dtype=torch.bool)}
    monkeypatch.setattr(runner, "COUNTS", {"train_x": 17, "original_y": 11, "validation": 9})
    data = SimpleNamespace(api=SimpleNamespace(torch=torch, masks=masks, primitives=primitives), paths=paths,
                           rows=rows, mask=mask, strata=strata)
    observed = []
    def fake_fit(arm):
        observed.append(arm)
        return SimpleNamespace(mean_x=torch.zeros(1, 4), mean_y=torch.zeros(1, 4), weight=torch.eye(4), similarity=lambda x, y: x @ y.T)
    monkeypatch.setattr(primitives, "fit_original", fake_fit)
    return data, observed


def test_one_attempt_keeps_original_order_counts_canonical_ranks_and_complete_last(tmp_path, monkeypatch):
    args = args_fixture(tmp_path)
    data, fits = synthetic_data(tmp_path, monkeypatch)
    def ready(args, bound):
        for path in data.paths.values(): bound.add(path, runner.r.digest(path))
        return data
    monkeypatch.setattr(runner, "preflight", ready)
    result = runner.run(args)
    assert result["status"] == "complete" and len(fits) == 1
    assert len(fits[0].samples_x) == 17 and len(fits[0].samples_y) == 11
    raw = torch.load(args.out / "ranks.pt", weights_only=True)
    assert raw["canonical_coco_image_ids"].tolist() == list(range(200, 209))
    expected_ranks = raw["num_higher"].double() + (raw["tie_count"].double() - 1) / 2 + 1
    assert torch.equal(raw["ranks"], expected_ranks)
    metrics = runner.r.read_json(args.out / "metrics.json")
    assert metrics["overall"]["gallery_size"] == 9
    assert metrics["fixed_strata"]["original_spc_overlap"]["num_queries"] == 0
    assert metrics["fixed_strata"]["all"]["foscttm"] == ((raw["ranks"] - 1) / 8).mean().item()
    saved = runner.r.read_json(args.out / "result.json")
    assert saved["configured_batch_cap"] == 10000
    assert saved["refinement_batch_dimensions"] == {"x": 17, "y": 11}
    marker = runner.r.read_json(args.out / "COMPLETE.json")
    assert "COMPLETE.json" not in marker["files"] and not (args.out / "FAILED.json").exists()
    for name, digest in marker["files"].items(): assert runner.r.digest(args.out / name) == digest
    assert list(json.loads(line)["event"] for line in (args.out / "journal.jsonl").read_text().splitlines())[-1] == "attempt_complete"
    with pytest.raises(FileExistsError):
        runner.run(SimpleNamespace(**{**vars(args), "private_log_dir": tmp_path / "private-02"}))
    assert len(fits) == 1


def test_evaluation_failure_preserves_map_and_private_trace_without_retry(tmp_path, monkeypatch):
    args = args_fixture(tmp_path)
    data, fits = synthetic_data(tmp_path, monkeypatch)
    monkeypatch.setattr(runner, "preflight", lambda *a: data)
    def fail_evaluation(*a, **kw): raise RuntimeError("synthetic private machine detail")
    monkeypatch.setattr(primitives, "evaluate_full_gallery", fail_evaluation)
    with pytest.raises(RuntimeError): runner.run(args)
    assert len(fits) == 1 and (args.out / "map.pt").is_file()
    assert (args.out / "training_order.pt").is_file() and not (args.out / "ranks.pt").exists()
    marker = runner.r.read_json(args.out / "FAILED.json")
    assert marker["failure_stage"] == "evaluation" and "map.pt" in marker["files"]
    assert "synthetic private" not in (args.out / "failure.json").read_text()
    assert "synthetic private" in (args.private_log_dir / "attempt.traceback.txt").read_text()
    assert not (args.out / "COMPLETE.json").exists()


def test_final_changed_input_blocks_complete_but_retains_evaluation(tmp_path, monkeypatch):
    args = args_fixture(tmp_path)
    data, fits = synthetic_data(tmp_path, monkeypatch)
    mutable = tmp_path / "bound-input.txt"
    mutable.write_text("before")
    def ready(args, bound):
        bound.add(mutable, runner.r.digest(mutable))
        return data
    actual_execute = runner.execute
    def change_after_evaluation(*arguments):
        result = actual_execute(*arguments)
        mutable.write_text("after")
        return result
    monkeypatch.setattr(runner, "preflight", ready)
    monkeypatch.setattr(runner, "execute", change_after_evaluation)
    with pytest.raises(ValueError, match="changed during"):
        runner.run(args)
    assert len(fits) == 1 and (args.out / "metrics.json").is_file() and (args.out / "ranks.pt").is_file()
    assert not (args.out / "COMPLETE.json").exists()
    assert runner.r.read_json(args.out / "FAILED.json")["failure_stage"] == "final_input_recheck"


def test_tensor_preflight_preserves_bf16_and_rejects_partial_nan_or_zero_rows(tmp_path):
    path = tmp_path / "tensor.pt"
    good = torch.ones(5, 3, 4, dtype=torch.bfloat16)
    good[::2, -1] = float("nan")
    entry = {"shape": [5, 3, 4]}
    torch.save(good, path)
    runner.validate_feature(path, entry, torch)
    for value in (good.float(), torch.zeros_like(good), good.clone(), good.clone()):
        if value.dtype == torch.bfloat16 and torch.equal(value.isnan(), good.isnan()) and value.isfinite().any():
            value[0, 0, 0] = float("nan")
        torch.save(value, path)
        with pytest.raises(ValueError): runner.validate_feature(path, entry, torch)


def test_trace_directory_cannot_be_inside_public_output(tmp_path):
    with pytest.raises(ValueError, match="separate"):
        runner.Attempt(tmp_path / "public", tmp_path / "public" / "private", {})
    assert not (tmp_path / "public").exists()


def test_full_preflight_composes_all_locks_masks_features_and_runtime_gates(tmp_path, monkeypatch):
    # Shrink only this artificial contract. Production constants remain full size;
    # source/runtime checks are spies, while byte, tensor, identity and mask checks
    # all run. No fit function is called.
    rows = small_rows()
    monkeypatch.setattr(runner, "ROW_IDENTITY", rows.identity_sha256)
    monkeypatch.setattr(runner, "COUNTS", {"train_x": 2, "original_y": 8, "validation": 3})
    monkeypatch.setattr(runner, "ARM_COUNTS", {"O": 8, "C": 7, "D": 5, "Rv": 7, "Rp": 5})
    monkeypatch.setattr(runner, "STRATA_COUNTS", {"never_known_spc_overlap": 2, "original_spc_overlap": 1, "all": 3})
    archive_paths, archive_hashes = {}, {}
    for name in runner.ARCHIVES:
        path = tmp_path / name
        path.write_bytes(("synthetic archive " + name).encode())
        archive_paths[name], archive_hashes[name] = str(path), runner.r.digest(path)
    monkeypatch.setattr(runner, "ARCHIVES", archive_hashes)
    entries, feature_paths = {}, {}
    for name, shape in (("train_x", [2, 4]), ("train_y", [8, 1, 4]), ("validation_x", [3, 4]), ("validation_y", [3, 7, 4])):
        path = tmp_path / (name + ".pt")
        torch.save(torch.ones(shape, dtype=torch.bfloat16), path)
        feature_paths[name] = str(path)
        entries[name] = {"sha256": runner.r.digest(path), "shape": shape, "dtype": "bfloat16", "publisher_path": "synthetic/" + name}
    monkeypatch.setattr(runner, "FEATURES", entries)
    p = protocol()
    p["code_sha256"] = {name: runner.r.digest(runner.HELPER_PATH if name == "run_cell.py" else Path(runner.__file__).parent / name) for name in runner.CODE}
    documents = {"source": {"synthetic": True}, "environment": {"synthetic": True},
                 "source_rows": {"schema_version": 1, "source_sha256": archive_hashes, "rows": rows.payload(), "row_identity_sha256": rows.identity_sha256},
                 "inputs": {"schema_version": 1, "publisher_repo": "schnaus/unpaired-rosetta-embeddings",
                            "publisher_revision": "b39767e18cdb5031020a0e1c4b8d9fba6ae13453",
                            "features": {name: {**entry, "size_bytes": Path(feature_paths[name]).stat().st_size} for name, entry in entries.items()}}}
    lock_paths = {}
    for name, document in documents.items():
        path = tmp_path / (name + ".json")
        path.write_text(json.dumps(document))
        lock_paths[name] = str(path)
        p["locks"][name] = runner.r.digest(path)
    args = args_fixture(tmp_path, p)
    bundle = small_bundle(p, rows)
    bundle["protocol_sha256"] = args.protocol_sha256
    args.masks.write_text(json.dumps(bundle))
    args.masks_sha256 = runner.r.digest(args.masks)
    source = tmp_path / "source"
    source.mkdir()
    args.bindings.write_text(json.dumps({"source_root": str(source), "python_executable": str(Path(__import__("sys").executable).resolve()),
                                        "locks": lock_paths, "archives": archive_paths, "features": feature_paths}))
    calls = []
    monkeypatch.setattr(runner.r, "check_sources", lambda *a: calls.append("source"))
    monkeypatch.setattr(runner.r, "check_runtime", lambda *a: calls.append("runtime"))
    monkeypatch.setattr(runner, "load_api", lambda *a: SimpleNamespace(torch=torch, masks=masks, primitives=primitives))
    bound = runner.r.BoundFiles()
    data = runner.preflight(args, bound)
    assert calls == ["source", "runtime"] and len(data.mask.train_y_indices) == 8
    assert data.strata["original_spc_overlap"].tolist() == [True, False, False]
    assert set(data.paths) == set(entries)
    assert all(Path(path).resolve() in bound.files for path in [*archive_paths.values(), *feature_paths.values(), *lock_paths.values()])
    # A changed released input is rejected even if every other lock still agrees.
    Path(feature_paths["train_x"]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="hash differs"):
        runner.preflight(args, runner.r.BoundFiles())
