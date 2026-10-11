#!/usr/bin/env python3
"""Prepare every locked source mask from independently joined public metadata.

CLI: --protocol PATH --protocol-sha256 SHA --release-lock PATH
 --release-lock-sha256 SHA --bindings PRIVATE_JSON --out NEW_DIRECTORY
 --private-log-dir NEW_PRIVATE_DIRECTORY. No tensor/runtime/model dependencies.

Protocol follows run_source_arm.py and must have final status
frozen_before_source_control_fits, with an explicit final selection-seed budget.
Outer release lock: schema_version=1, same protocol_id/status, protocol_sha256,
code_sha256={prepare_masks.py,run_source_arm.py,source_primitives.py,source_masks.py,
run_cell.py}. The four overlapping hashes must equal protocol.code_sha256. Both
locks' exact byte hashes are explicit CLI inputs; the release is frozen after the
protocol, so neither lock requires a hash of a future mask bundle.

Private bindings: {archives:{caption_datasets.zip:ABSOLUTE_PATH,
vg_image_data.json.zip:ABSOLUTE_PATH,paragraphs_v1.json.zip:ABSOLUTE_PATH}}.
No downloads or path discovery. Exactly one member with each expected basename
must exist in its ZIP. The source rows are independently rebuilt from that join,
then serialized in historical insertion order and checked against the protocol's
locked source_rows file hash before any mask is constructed.

Outputs: source_rows.json (historical byte format); masks.json (the runner's
complete bundle); compositions.json; intersections.json (every unordered pair);
schedule.json (mask-major, original seed order); start/journal/result and COMPLETE.
Every output binds the final protocol; the bundle does not bind its own hash.
A schedule has (3 + len(Rv) + len(Rp))*5 fits:115 only if10+10 is finally chosen.
Failure leaves prior artifacts and a private traceback, never automatic retries.
This is preparation only: it loads no features and runs no fits or scores.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import zipfile

import run_source_arm as arm
import source_masks as masks

r = arm.r
MEMBERS = {"caption_datasets.zip": "dataset_coco.json", "vg_image_data.json.zip": "image_data.json",
           "paragraphs_v1.json.zip": "paragraphs_v1.json"}
CODE = arm.CODE | {"prepare_masks.py"}


def validate_release(protocol, release, protocol_sha, bound):
    arm.check_protocol(protocol)
    r.no_local_paths(release)
    for key, expected in {"schema_version": 1, "protocol_id": arm.PROTOCOL_ID, "status": arm.STATUS,
                          "protocol_sha256": protocol_sha}.items():
        r.require(release.get(key) == expected, f"Unfrozen preparation release: {key}")
    r.require(set(release.get("code_sha256", {})) == CODE, "Preparation helper and dependencies must all be in the release")
    r.require(all(release["code_sha256"][key] == value for key, value in protocol["code_sha256"].items()),
              "Preparation and execution code locks disagree")
    for name, expected in release["code_sha256"].items():
        bound.add(arm.HELPER_PATH if name == "run_cell.py" else Path(__file__).resolve().parent / name, expected)
    r.require(Path(arm.__file__).resolve() == Path(__file__).resolve().parent / "run_source_arm.py" and
              Path(masks.__file__).resolve() == Path(__file__).resolve().parent / "source_masks.py", "Preparation imports came from another release")


def read_member(path, basename):
    with zipfile.ZipFile(path) as archive:
        matches = [member for member in archive.infolist() if not member.is_dir() and Path(member.filename).name == basename]
        r.require(len(matches) == 1, "Metadata ZIP must have exactly one expected named member")
        payload = archive.read(matches[0])
    value = json.loads(payload)
    return value, {"member": matches[0].filename, "member_sha256": hashlib.sha256(payload).hexdigest(), "member_size_bytes": len(payload)}


def rebuild_rows(archive_paths, protocol, bound):
    r.require(set(archive_paths) == set(arm.ARCHIVES), "All three actual source archives are required")
    values, receipts = {}, {}
    for name, expected in arm.ARCHIVES.items():
        r.require(isinstance(archive_paths[name], str) and Path(archive_paths[name]).is_absolute(), "Archive bindings must be absolute private paths")
        path = bound.add(archive_paths[name], expected)
        values[name], member = read_member(path, MEMBERS[name])
        receipts[name] = {"sha256": expected, "size_bytes": path.stat().st_size, **member}
    rows = masks.source_rows_from_records(values["caption_datasets.zip"]["images"], values["paragraphs_v1.json.zip"], values["vg_image_data.json.zip"])
    r.require(rows.identity_sha256 == protocol["row_identity_sha256"], "Independent source-row join differs from locked identity")
    r.require(tuple(map(len, (rows.train_x, rows.train_y, rows.validation))) ==
              (protocol["counts"]["train_x"], protocol["counts"]["original_y"], protocol["counts"]["validation"]), "Source populations differ")
    # Preserve the archived source_rows.json's insertion order and compact bytes.
    document = {"schema_version": 1, "source_sha256": arm.ARCHIVES,
                "row_identity_sha256": rows.identity_sha256, "rows": rows.payload()}
    encoded = (json.dumps(document, separators=(",", ":")) + "\n").encode()
    r.require(hashlib.sha256(encoded).hexdigest() == protocol["locks"]["source_rows"], "Rebuilt source_rows.json bytes differ from the frozen document")
    return rows, encoded, receipts


def build_bundle(rows, protocol, protocol_sha):
    selected = masks.build_masks(rows, validation_seeds=tuple(protocol["selection_seeds"]["Rv"]),
                                  pairing_seeds=tuple(protocol["selection_seeds"]["Rp"]))
    short_names = {masks.ORIGINAL: "O", masks.VALIDATION: "C", masks.BOTH: "D",
                   masks.RANDOM_VALIDATION: "Rv", masks.RANDOM_PAIRING: "Rp"}
    by_key = {}
    for mask in selected:
        short = short_names[mask.variant]
        key = short if mask.selection_seed is None else f"{short}-{mask.selection_seed}"
        by_key[key] = mask
    bundle = {"schema_version": 1, "protocol_id": arm.PROTOCOL_ID, "status": arm.STATUS,
              "protocol_sha256": protocol_sha, "row_identity_sha256": rows.identity_sha256,
              "masks": {key: {"variant": mask.variant, "selection_seed": mask.selection_seed,
                              "row_identity_sha256": mask.row_identity_sha256, "train_y_indices": list(mask.train_y_indices),
                              "mask_sha256": mask.mask_sha256} for key, mask in by_key.items()},
              "strata": {key: list(indices) for key, indices in masks.fixed_strata(rows).items()}}
    arm.check_bundle(bundle, protocol, protocol_sha, rows, masks)
    return bundle, by_key


def prepare(args):
    binding = {"schema_version": 1, "protocol_id": arm.PROTOCOL_ID, "protocol_sha256": args.protocol_sha256,
               "release_lock_sha256": args.release_lock_sha256, "operation": "prepare_all_source_masks"}
    attempt = arm.Attempt(args.out, args.private_log_dir, binding)
    bound = r.BoundFiles()
    try:
        protocol = bound.json(args.protocol, args.protocol_sha256)
        release = bound.json(args.release_lock, args.release_lock_sha256)
        validate_release(protocol, release, args.protocol_sha256, bound)
        bindings = bound.json(args.bindings, r.digest(args.bindings))
        r.require(set(bindings) == {"archives"}, "Only private archive paths belong in preparation bindings")
        attempt.stage = "independent_metadata_join"
        rows, encoded, receipts = rebuild_rows(bindings["archives"], protocol, bound)
        with (args.out / "source_rows.json").open("xb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        attempt.event("source_identity_and_bytes_verified")
        attempt.stage = "all_declared_masks"
        bundle, selected = build_bundle(rows, protocol, args.protocol_sha256)
        r.write_json_new(args.out / "masks.json", bundle)
        mask_bundle_sha = r.digest(args.out / "masks.json")
        common = {**binding, "masks_sha256": mask_bundle_sha, "row_identity_sha256": rows.identity_sha256}
        r.write_json_new(args.out / "compositions.json", {**common, "masks": {key: masks.composition(rows, mask) for key, mask in selected.items()}})
        keys = list(selected)
        intersections = [{"left": left, "right": right, **masks.intersection(rows, selected[left], selected[right])}
                         for i, left in enumerate(keys) for right in keys[i + 1:]]
        r.write_json_new(args.out / "intersections.json", {**common, "pairs": intersections})
        schedule = [{"index": index, "mask": key, "mask_sha256": selected[key].mask_sha256, "seed": seed}
                    for index, (key, seed) in enumerate((key, seed) for key in keys for seed in protocol["seeds"])]
        expected_fits = (3 + sum(len(v) for v in protocol["selection_seeds"].values())) * len(protocol["seeds"])
        r.require(len(schedule) == expected_fits and len({(row["mask"], row["seed"]) for row in schedule}) == expected_fits,
                  "Every declared mask must have every original seed exactly once")
        r.write_json_new(args.out / "schedule.json", {**common, "planned_fits": expected_fits, "order": "mask_major_original_algorithm_seed_order", "runs": schedule})
        attempt.event("complete_mask_schedule_prepared")
        result = {"prepared_masks": len(selected), "planned_fits": len(schedule), "strata_counts": {k: len(v) for k, v in bundle["strata"].items()},
                  "archive_members": receipts, "source_rows_sha256": r.digest(args.out / "source_rows.json"),
                  "masks_sha256": mask_bundle_sha, "feature_loads": 0, "fits": 0,
                  "scope": "Known source metadata only; masks remove no encoder pretraining exposure."}
        r.no_local_paths(result)
        attempt.finish(result, bound)
    except BaseException as error:
        attempt.fail(error)
        raise
    return {"status": "complete", "masks_sha256": mask_bundle_sha, "prepared_masks": len(selected),
            "planned_fits": len(schedule), "completion_sha256": r.digest(args.out / "COMPLETE.json"), "fits": 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("protocol", "release-lock", "bindings", "out", "private-log-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("protocol-sha256", "release-lock-sha256"):
        parser.add_argument("--" + name, required=True)
    print(json.dumps(prepare(parser.parse_args()), indent=2))


if __name__ == "__main__":
    main()
