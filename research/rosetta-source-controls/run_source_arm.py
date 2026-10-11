#!/usr/bin/env python3
"""Draft supervisor for one full-bank source arm/mask/algorithm-seed attempt.

Future public schema; hashes always bind exact bytes unless explicitly canonical:
 protocol: schema_version=1, protocol_id=rosetta-source-controls-v1,
   status=frozen_before_source_control_fits, upstream_commit=UPSTREAM,
   seeds=SEEDS, counts=COUNTS, arm_counts=ARM_COUNTS,
   algorithm_defaults=ALGORITHM, execution=EXECUTION,
   selection_seeds={Rv:[explicit integers],Rp:[explicit integers]},
   code_sha256={run_source_arm.py,source_primitives.py,source_masks.py,run_cell.py},
   locks={source,environment,source_rows,inputs} mapping names to SHA256,
   source_archive_sha256=ARCHIVES, row_identity_sha256=ROW_IDENTITY,
   strata_counts=STRATA_COUNTS, metric_rules=METRIC_RULES.
 inputs: schema_version=1, publisher_repo=schnaus/unpaired-rosetta-embeddings,
   publisher_revision=b39767e18cdb5031020a0e1c4b8d9fba6ae13453,
   features={train_x,train_y,validation_x,validation_y}; each contains exactly
   sha256,size_bytes,shape,dtype,publisher_path, as specified by FEATURES.
 masks bundle: schema_version=1, same protocol_id/status, protocol_sha256,
   row_identity_sha256, masks={O,C,D,Rv-<seed>,Rp-<seed>}, strata={fixed name:
   increasing canonical validation row indices}. Every mask entry contains
   variant,selection_seed,row_identity_sha256,train_y_indices,mask_sha256;
   mask_sha256 is source_masks.SourceMask's canonical content hash.
   All declared masks must be present and match the algorithm, not just the one
   requested. Every mask is planned for every original algorithm seed.

Private --bindings JSON contains paths only: source_root, python_executable,
 locks={source,environment,source_rows,inputs}, archives={archive filename:path},
 features={train_x,train_y,validation_x,validation_y}. Local paths never belong
 in public locks. CLI needs both --protocol-sha256 and --masks-sha256 explicitly.
 The bundle binds the protocol; the protocol does not bind the bundle (no cycle).

Execution requires a fresh --out and separate fresh --private-log-dir. Attempt
start precedes preflight; failures preserve artifacts and a private traceback.
No automatic retry, selection, aggregation, or classification. COMPLETE.json is
last, after one unchanged upstream fit, full-gallery evaluation, and byte checks.
This file alone does not freeze the protocol or authorize a real experiment.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import traceback
from types import SimpleNamespace


HELPER_SHA256 = "611aad8867cf174e5c03fc7a534d6edc8215fe9f08bcd056f5cd19943fe2c10a"
HELPER_PATH = Path(__file__).resolve().parent.parent / "rosetta-model-controls" / "run_cell.py"
if hashlib.sha256(HELPER_PATH.read_bytes()).hexdigest() != HELPER_SHA256:
    raise ValueError("Pinned shared runner utilities differ")
_spec = importlib.util.spec_from_file_location("source_shared_runner", HELPER_PATH)
r = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(r)

PROTOCOL_ID = "rosetta-source-controls-v1"
STATUS = "frozen_before_source_control_fits"
UPSTREAM, SEEDS, ALGORITHM = r.UPSTREAM, r.SEEDS, r.ALGORITHM
COUNTS = {"train_x": 82783, "original_y": 19561, "validation": 40504}
ARM_COUNTS = {"O": 19561, "C": 16224, "D": 9963, "Rv": 16224, "Rp": 9963}
EXECUTION = {"num_workers": 4, "num_threads": 4, "device": "cpu", "evaluation_block_size": 4096,
             "torch_threads": 16, "torch_interop_threads": 16}
STRATA_COUNTS = {"never_known_spc_overlap": 37169, "original_spc_overlap": 3335, "all": 40504}
METRIC_RULES = {"primary": "never_known_spc_overlap/foscttm", "gallery": "full_40504",
                "foscttm_denominator": 40503, "recall_k": [1, 5, 10],
                "recall_ties": "expected_credit_under_uniform_order_within_exact_score_tie_block",
                "median": "arithmetic_mean_of_two_central_ranks_when_even", "inference": "descriptive_only"}
ROW_IDENTITY = "e4e0e295e82ead25e7afca9a81b4d0c9513e6f56c64247848f6cb5bf9006b455"
ARCHIVES = {
    "caption_datasets.zip": "4cfd70132527b80933105e5829dc9034eaab9573482e2e680abbab6130244817",
    "vg_image_data.json.zip": "b87a94918cb2ff4d952cf1dfeca0b9cf6cd6fd204c2f8704645653be1163681a",
    "paragraphs_v1.json.zip": "8407f1068e0711e5f6550d641c523ee26984ee8e02c1d51564369cccd56e8b26",
}
FEATURES = {
    "train_x": {"sha256": "431d0caf275e71837ced1d519d5a3464a8b6896744529188e1bae95b10cd013c", "shape": [82783, 768],
                "dtype": "bfloat16", "publisher_path": "embeddings/coco_train2014/vision/dinov2_vit-b14@224_mean.pt"},
    "train_y": {"sha256": "d9c85440e34084b70890e07ac16a3a089a1b54d461e50268b681c7485016ac45", "shape": [19561, 1, 768],
                "dtype": "bfloat16", "publisher_path": "embeddings/StanfordParagraphCaptioning/language/mpnet.pt"},
    "validation_x": {"sha256": "13bc24b7ed9164f605d0892fe2bbc17feb30d8fcc66bb2e94d2a0ad77e9f8e3a", "shape": [40504, 768],
                     "dtype": "bfloat16", "publisher_path": "embeddings/coco_val2014/vision/dinov2_vit-b14@224_mean.pt"},
    "validation_y": {"sha256": "dc960f4d1fda472f3ddfbd9e5e618e1ef43072261ec19e291c5cb8089006fb1a", "shape": [40504, 7, 768],
                     "dtype": "bfloat16", "publisher_path": "embeddings/coco_val2014/language/mpnet.pt"},
}
LOCKS = {"source", "environment", "source_rows", "inputs"}
CODE = {"run_source_arm.py", "source_primitives.py", "source_masks.py", "run_cell.py"}


def check_protocol(p):
    r.no_local_paths(p)
    for key, value in {"schema_version": 1, "protocol_id": PROTOCOL_ID, "status": STATUS,
                       "upstream_commit": UPSTREAM, "seeds": SEEDS, "counts": COUNTS,
                       "arm_counts": ARM_COUNTS, "algorithm_defaults": ALGORITHM, "execution": EXECUTION,
                       "source_archive_sha256": ARCHIVES, "row_identity_sha256": ROW_IDENTITY,
                       "strata_counts": STRATA_COUNTS, "metric_rules": METRIC_RULES}.items():
        r.require(p.get(key) == value, f"Unfrozen or changed protocol field: {key}")
    r.require(set(p.get("locks", {})) == LOCKS and set(p.get("code_sha256", {})) == CODE, "Incomplete code/input/runtime locks")
    for value in [*p["locks"].values(), *p["code_sha256"].values()]:
        r.sha(value)
    r.require(p["code_sha256"]["run_cell.py"] == HELPER_SHA256, "Shared helper differs")
    seeds = p.get("selection_seeds", {})
    r.require(set(seeds) == {"Rv", "Rp"}, "Both random-control mask budgets must be explicit")
    for values in seeds.values():
        r.require(type(values) is list and len(values) > 0 and all(type(s) is int and s >= 0 for s in values),
                  "Selection seeds must be an explicit nonempty integer list")
        r.require(len(set(values)) == len(values) and not set(values) & set(SEEDS), "Selection streams must be unique and separate from algorithm seeds")
    r.require(not set(seeds["Rv"]) & set(seeds["Rp"]), "Random-control seed families must be separate")


def check_bundle(bundle, p, protocol_sha, rows, masks_api):
    r.no_local_paths(bundle)
    for key, value in {"schema_version": 1, "protocol_id": PROTOCOL_ID, "status": STATUS,
                       "protocol_sha256": protocol_sha, "row_identity_sha256": rows.identity_sha256}.items():
        r.require(bundle.get(key) == value, f"Masks bundle differs: {key}")
    variants = {"O": masks_api.ORIGINAL, "C": masks_api.VALIDATION, "D": masks_api.BOTH}
    expected = {key: (variant, None) for key, variant in variants.items()}
    for arm, variant in (("Rv", masks_api.RANDOM_VALIDATION), ("Rp", masks_api.RANDOM_PAIRING)):
        for seed in p["selection_seeds"][arm]:
            expected[f"{arm}-{seed}"] = (variant, seed)
    r.require(set(bundle.get("masks", {})) == set(expected), "Incomplete or extra mask schedule")
    validated = {}
    for key, (variant, selection_seed) in expected.items():
        entry = bundle["masks"][key]
        fields = {"variant", "selection_seed", "row_identity_sha256", "train_y_indices", "mask_sha256"}
        r.require(set(entry) == fields and entry["variant"] == variant and entry["selection_seed"] == selection_seed,
                  "Mask key and declared arm/seed differ")
        mask = masks_api.SourceMask(entry["variant"], entry["selection_seed"], entry["row_identity_sha256"], tuple(entry["train_y_indices"]))
        masks_api.validate_mask(rows, mask)
        r.require(entry["mask_sha256"] == mask.mask_sha256, "Canonical mask hash differs")
        r.require(len(mask.train_y_indices) == ARM_COUNTS[key.split("-")[0]], "Wrong full-arm paragraph count")
        validated[key] = mask
    expected_strata = {k: list(v) for k, v in masks_api.fixed_strata(rows).items()}
    r.require(bundle.get("strata") == expected_strata, "Strata must remain fixed to original SPC source membership")
    r.require({k: len(v) for k, v in expected_strata.items()} == STRATA_COUNTS, "Wrong full-gallery stratum counts")
    return validated


def validate_feature(path, entry, torch):
    value = torch.load(path, mmap=True, weights_only=True, map_location="cpu")
    r.require(isinstance(value, torch.Tensor) and list(value.shape) == entry["shape"] and value.dtype == torch.bfloat16,
              "Publisher tensor shape/precision differs")
    for block in value.split(4096):
        if value.ndim == 3:
            valid = torch.isfinite(block).all(dim=-1)
            padding = torch.isnan(block).all(dim=-1)
            r.require(bool((valid | padding).all()) and bool(valid.any(dim=1).all()), "Invalid caption slots or all-padding item")
            vectors = block[valid]
        else:
            r.require(bool(torch.isfinite(block).all()), "Nonfinite image embeddings")
            vectors = block
        r.require(bool((vectors.float().norm(dim=-1) > 0).all()), "Zero-norm feature vector")


def load_api(source):
    sys.path.insert(0, str(source))
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import torch
    import source_masks
    import source_primitives
    from unpaired_rosetta.randomness import SEEDS as official_seeds
    for module in (source_masks, source_primitives):
        r.require(Path(module.__file__).resolve().parent == Path(__file__).resolve().parent, "Source helpers came from another release")
    for name, module in list(sys.modules.items()):
        if name == "unpaired_rosetta" or name.startswith("unpaired_rosetta."):
            r.require(Path(module.__file__).resolve().is_relative_to(source), "Original API came from another checkout")
    original = source_primitives.WassersteinProcrustes()
    r.require(official_seeds == SEEDS and {key: getattr(original, key) for key in ALGORITHM} == ALGORITHM, "Original seeds/defaults differ")
    return SimpleNamespace(torch=torch, masks=source_masks, primitives=source_primitives)


def preflight(args, bound):
    p = bound.json(args.protocol, args.protocol_sha256)
    check_protocol(p)
    bundle = bound.json(args.masks, args.masks_sha256)
    bindings = bound.json(args.bindings, r.digest(args.bindings))
    r.require(set(bindings) == {"source_root", "python_executable", "locks", "archives", "features"}, "Incomplete private paths")
    r.require(set(bindings["locks"]) == LOCKS and set(bindings["archives"]) == set(ARCHIVES) and
              set(bindings["features"]) == set(FEATURES), "Incomplete private path groups")
    for path in [bindings["source_root"], bindings["python_executable"], *bindings["locks"].values(), *bindings["archives"].values(), *bindings["features"].values()]:
        r.require(isinstance(path, str) and Path(path).is_absolute(), "Private bindings contain absolute paths only")
    for name, sha in p["code_sha256"].items():
        bound.add(HELPER_PATH if name == "run_cell.py" else Path(__file__).resolve().parent / name, sha)
    locks = {key: bound.json(bindings["locks"][key], p["locks"][key]) for key in LOCKS}
    for name, expected in ARCHIVES.items():
        bound.add(bindings["archives"][name], expected)
    source = Path(bindings["source_root"]).resolve(strict=True)
    r.check_sources(source, locks["source"], bound)
    r.check_runtime(locks["environment"], p["execution"], source, bindings["python_executable"])
    api = load_api(source)
    rows = api.masks.source_rows_from_document(locks["source_rows"], expected_source_sha256=ARCHIVES)
    r.require(rows.identity_sha256 == ROW_IDENTITY and
              tuple(map(len, (rows.train_x, rows.train_y, rows.validation))) == (COUNTS["train_x"], COUNTS["original_y"], COUNTS["validation"]),
              "Wrong original source identities/populations")
    masks = check_bundle(bundle, p, args.protocol_sha256, rows, api.masks)
    r.require(args.mask in masks and args.seed in SEEDS, "Arm or algorithm seed is outside the complete declared schedule")
    inputs = locks["inputs"]
    r.require(inputs.get("schema_version") == 1 and inputs.get("publisher_repo") == "schnaus/unpaired-rosetta-embeddings" and
              inputs.get("publisher_revision") == "b39767e18cdb5031020a0e1c4b8d9fba6ae13453" and set(inputs.get("features", {})) == set(FEATURES),
              "Wrong released feature inputs")
    paths = {}
    for name, expected in FEATURES.items():
        entry = inputs["features"][name]
        r.require(set(entry) == set(expected) | {"size_bytes"} and all(entry.get(k) == v for k, v in expected.items()), "Publisher feature contract differs")
        paths[name] = bound.add(bindings["features"][name], entry["sha256"], entry["size_bytes"])
        validate_feature(paths[name], entry, api.torch)
    strata = {}
    for name, indices in bundle["strata"].items():
        strata[name] = api.torch.zeros(COUNTS["validation"], dtype=api.torch.bool)
        strata[name][api.torch.tensor(indices, dtype=api.torch.int64)] = True
    bound.recheck()
    return SimpleNamespace(protocol=p, api=api, rows=rows, mask=masks[args.mask], paths=paths, strata=strata)


class Attempt:
    def __init__(self, out, private, binding):
        self.out, self.private, self.binding = Path(out), Path(private), binding
        r.require(not self.private.resolve().is_relative_to(self.out.resolve()) and not self.out.resolve().is_relative_to(self.private.resolve()),
                  "Public attempt and private trace directories must be separate")
        r.require(not self.private.exists(), "Private attempt directory already exists")
        self.out.mkdir(parents=True, exist_ok=False)
        self.private.mkdir(parents=True, exist_ok=False)
        self.stage = "preflight"
        r.write_json_new(self.out / "started.json", {**binding, "status": "started", "retry_policy": "fresh_explicit_attempt_only"})
        self.journal = (self.out / "journal.jsonl").open("x")
        self.event("attempt_started")

    def event(self, name):
        self.journal.write(json.dumps({"event": name, "stage": self.stage, "at_utc": datetime.now(timezone.utc).isoformat()}) + "\n")
        self.journal.flush()
        os.fsync(self.journal.fileno())

    def finish(self, result, bound):
        self.stage = "final_input_recheck"
        bound.recheck()
        r.write_json_new(self.out / "result.json", {**self.binding, "status": "complete", **result})
        self.event("attempt_complete")
        self.journal.close()
        self.marker("COMPLETE.json", "complete")

    def marker(self, name, status, **extra):
        files = {str(path.relative_to(self.out)): r.digest(path) for path in sorted(self.out.rglob("*")) if path.is_file()}
        r.publish_marker(self.out / name, {**self.binding, "status": status, "files": files, **extra})
        fd = os.open(self.out, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)

    def fail(self, error):
        with (self.private / "attempt.traceback.txt").open("x") as stream:
            traceback.print_exc(file=stream)
            stream.flush()
            os.fsync(stream.fileno())
        if not self.journal.closed:
            self.event("attempt_failed")
            self.journal.close()
        record = {**self.binding, "status": "failed", "failure_stage": self.stage, "error_type": type(error).__name__,
                  "detail": "Prior artifacts are retained. See the separate private traceback; no retry was attempted."}
        r.write_json_new(self.out / "failure.json", record)
        self.marker("FAILED.json", "failed", failure_stage=self.stage, error_type=type(error).__name__)


def execute(attempt, data, seed):
    api, torch = data.api, data.api.torch
    started = time.monotonic()
    attempt.stage = "training_load"
    arm = api.primitives.load_training_arm(data.paths["train_x"], data.paths["train_y"],
                                          torch.tensor(data.mask.train_y_indices, dtype=torch.int64), seed,
                                          source_keys=data.rows.train_y)
    r.require(len(arm.samples_x) == COUNTS["train_x"] and len(arm.samples_y) == len(data.mask.train_y_indices), "Fitting bank count differs")
    r.save_tensor_new(attempt.out / "training_order.pt", {"indices_x": arm.indices_x, "indices_y": arm.indices_y,
                                                        "canonical_mask_y": torch.tensor(data.mask.train_y_indices, dtype=torch.int64)}, torch)
    attempt.stage = "fit"
    attempt.event("fit_started")
    fit_started = time.monotonic()
    model = api.primitives.fit_original(arm)
    fit_seconds = time.monotonic() - fit_started
    r.require(all(bool(torch.isfinite(getattr(model, k)).all()) for k in ("mean_x", "mean_y", "weight")), "Nonfinite fitted map")
    r.save_tensor_new(attempt.out / "map.pt", {k: getattr(model, k) for k in ("mean_x", "mean_y", "weight")}, torch)
    attempt.event("fit_complete")
    attempt.stage = "evaluation"
    attempt.event("evaluation_started")
    record = api.primitives.evaluate_full_gallery(model, data.paths["validation_x"], data.paths["validation_y"], seed,
                                                 strata=data.strata, block_size=EXECUTION["evaluation_block_size"])
    r.require(len(record["ranks"]) == COUNTS["validation"] and record["overall"]["gallery_size"] == COUNTS["validation"], "Full gallery required")
    tensors = {k: record[k] for k in ("ranks", "num_higher", "tie_count", "query_order", "key_order", "validation_order")}
    tensors["canonical_coco_image_ids"] = torch.tensor([int(key.split(":")[1]) for key in data.rows.validation], dtype=torch.int64)
    r.save_tensor_new(attempt.out / "ranks.pt", tensors, torch)
    metrics = {"original_overall": record["metrics"], "overall": record["overall"], "fixed_strata": record["strata"],
               "rank_order": record["rank_order"], "tie_policy": record["tie_policy"],
               "overall_reduction_order": record["overall_reduction_order"]}
    r.write_json_new(attempt.out / "metrics.json", metrics)
    attempt.event("evaluation_complete")
    return {"fit_status": "complete", "evaluation_status": "complete", "fit_seconds": fit_seconds,
            "fit_and_evaluation_seconds": time.monotonic() - started, "classification_rerun": False,
            "counts": {"train_x": len(arm.samples_x), "train_y": len(arm.samples_y), "validation": COUNTS["validation"]},
            "configured_batch_cap": ALGORITHM["batch_size"],
            "refinement_batch_dimensions": {"x": min(ALGORITHM["batch_size"], len(arm.samples_x)), "y": min(ALGORITHM["batch_size"], len(arm.samples_y))},
            "batch_dimensions_provenance": "Derived from unchanged upstream subset(size=batch_size) and full bank counts; no extra sampling.",
            "mask_sha256": data.mask.mask_sha256, "composition": api.masks.composition(data.rows, data.mask),
            "artifact_hashes": {name: r.digest(attempt.out / name) for name in ("training_order.pt", "map.pt", "ranks.pt", "metrics.json")},
            "scope": "Full-bank source-overlap retrieval control; known metadata identities only. No pretraining-cleanliness claim."}


def run(args):
    binding = {"schema_version": 1, "protocol_id": PROTOCOL_ID, "protocol_sha256": args.protocol_sha256,
               "masks_sha256": args.masks_sha256, "mask": args.mask, "seed": args.seed}
    r.no_local_paths(binding)
    attempt = Attempt(args.out, args.private_log_dir, binding)
    bound = r.BoundFiles()
    try:
        r.install_retry_review(args, attempt, bound)
        data = preflight(args, bound)
        attempt.event("preflight_passed")
        result = execute(attempt, data, args.seed)
        attempt.finish(result, bound)
    except BaseException as error:
        attempt.fail(error)
        raise
    return {"status": "complete", "completion_sha256": r.digest(args.out / "COMPLETE.json")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("protocol", "masks", "bindings", "out", "private-log-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("protocol-sha256", "masks-sha256", "mask"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--retry-review", type=Path)
    parser.add_argument("--retry-review-sha256")
    print(json.dumps(run(parser.parse_args()), indent=2))


if __name__ == "__main__":
    main()
