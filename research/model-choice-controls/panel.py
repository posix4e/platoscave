#!/usr/bin/env python3
"""Frozen six-pair follow-up: extract, assemble blind matrices, select, then score."""

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import itertools
import json
import os
from pathlib import Path
import random
import re
import sys
import unicodedata


ROOT = Path(__file__).resolve().parent
PROTOCOL_ID = "five-photo-model-choice-v1"
IDS = [f"candidate-{i:03d}" for i in range(1, 6)]
IMAGES = ("dino", "moco", "mae")
TEXTS = ("glove", "elmo")
DIMENSIONS = {"dino": 768, "moco": 2048, "mae": 768, "glove": 300, "elmo": 1024}
MANIFEST_SHA = "252eab59d902fcb3a772a73782d46156d6f53c9bf67bc3f79010b52e95287ece"
BASE_SHA = "d9924e021e2b4233493930454ca7d32be2cde55617d1ed8aaca83bf11b528515"
PRIMARY_EVALUATION_SHA = "6561389d21da6a9450205cf9fc158b291ea17af8ca1253c2ec129b13606cfe01"
CORE_SHA = "6819921cecd461dc841b1ea89dddb18fab53fc0c36c07b631d7f755346ce5dfc"
HELPER_SHA = "3e8445f22a8f255dcf47af89ac0b8c45461ba98046549702abc7b09b6b59a624"
WEIGHTS = {"moco": "5e4a3fd7178a95837c0ad71c265ed5f477d271e753ba505b6b6a5e86829eeb7e",
           "mae": "aec5f0b68e5f3193a00b07bc65a37440db549c15b36b8bea242606cc40c4bc5d",
           "elmo": "a6b92b9ccbb13fe489faacd2221c795d4bebe84b75f8aded9d85ea464fa5c994"}
TOL = 1e-12


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, "Duplicate JSON key")
            result[key] = value
        return result
    def invalid(value):
        raise ValueError("Nonfinite JSON constant")
    return json.loads(Path(path).read_text(), object_pairs_hook=unique, parse_constant=invalid)


def write(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def release():
    lock = read(ROOT / "release-lock.json")
    required = {"protocol.json", "panel.py", "vision_features.py", "elmo_features.py", "vision-vendor-lock.json"}
    require(required <= set(lock["files"]), "Incomplete release lock")
    for name, expected in lock["files"].items():
        path = Path(name)
        require(not path.is_absolute() and ".." not in path.parts, "Unsafe release path")
        require(digest(ROOT / name) == expected, f"Changed frozen source: {name}")
    protocol = read(ROOT / "protocol.json")
    require(protocol["protocol_id"] == PROTOCOL_ID, "Wrong protocol")
    require(protocol.get("status") == "frozen_before_additional_photo_features", "Protocol is not frozen")
    return {"protocol_id": PROTOCOL_ID, "protocol_sha256": digest(ROOT / "protocol.json"),
            "release_lock_sha256": digest(ROOT / "release-lock.json"), "source_hashes": lock["files"]}


def load_module(path, name, expected):
    require(digest(path) == expected, "External source hash mismatch")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def manifest(path):
    require(digest(path) == MANIFEST_SHA, "Changed five-photo manifest")
    rows = read(path)["rows"]
    require([r["candidate_id"] for r in rows] == IDS, "Changed sample order")
    return rows


def orders():
    a, b = list(range(5)), list(range(5))
    random.Random(20261010).shuffle(a)
    random.Random(20261011).shuffle(b)
    return a, b


def check_features(x, name, np):
    require(x.dtype == np.float32 and x.shape == (5, DIMENSIONS[name]), f"Wrong {name} feature shape/dtype")
    require(np.isfinite(x).all() and np.allclose(np.linalg.norm(x, axis=1), 1, atol=1e-4, rtol=0),
            f"Invalid {name} features")


def cosine(x, order, np):
    a = x.astype(np.float64)[order]
    a /= np.linalg.norm(a, axis=1, keepdims=True)
    d = 1 - a @ a.T
    d = (d + d.T) / 2
    np.fill_diagonal(d, 0)
    return d.tolist()


def captions(args):
    release()
    rows = manifest(args.manifest)
    write(args.out, {"sample_ids": IDS, "captions": [r["label_verbatim"] for r in rows]})


def vision(args):
    provenance = release()
    rows = manifest(args.manifest)
    require(digest(args.checkpoint) == WEIGHTS[args.encoder], "Wrong checkpoint bytes")
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    helper = load_module(args.helper, "frozen_preprocess", HELPER_SHA)
    adapter = load_module(ROOT / "vision_features.py", "panel_vision", provenance["source_hashes"]["vision_features.py"])
    import numpy as np
    import torch
    import PIL
    from PIL import Image, ImageOps
    require(PIL.__version__ == "12.3.0", "Pillow differs from the frozen common image preprocessing runtime")
    require(torch.cuda.is_available(), "CUDA is required; no silent CPU fallback")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    args.out.mkdir(mode=0o700)
    write(args.out / "preflight.json", {**provenance, "encoder": args.encoder,
          "manifest_sha256": MANIFEST_SHA, "weights_sha256": WEIGHTS[args.encoder],
          "recorded_at_utc": datetime.now(timezone.utc).isoformat(), "status": "verified_before_inference"})
    try:
        model = (adapter.load_moco(args.checkpoint, device="cuda") if args.encoder == "moco"
                 else adapter.load_mae(args.checkpoint, args.vendor, device="cuda"))
        pixels = []
        for row in rows:
            path = args.images / row["image_file"]
            raw = path.read_bytes()
            require(hashlib.sha256(raw).hexdigest() == row["image_sha256"], "Changed photograph bytes")
            pixels.append(helper.preprocess_bytes(raw, np, Image, ImageOps))
        features = adapter.extract(model, torch.from_numpy(np.stack(pixels)), batch_size=1)
        check_features(features, args.encoder, np)
        for row in rows:
            require(digest(args.images / row["image_file"]) == row["image_sha256"], "Photograph changed during inference")
        require(release() == provenance and digest(args.checkpoint) == WEIGHTS[args.encoder], "Sources changed during inference")
        with (args.out / "features.npz").open("xb") as stream:
            np.savez_compressed(stream, features=features, sample_ids=np.asarray(IDS))
        write(args.out / "metadata.json", {**provenance, "encoder": args.encoder, "status": "complete",
              "feature_sha256": digest(args.out / "features.npz"), "sample_count": 5,
              "weights_sha256": WEIGHTS[args.encoder], "helper_sha256": HELPER_SHA,
              "shape": list(features.shape), "dtype": str(features.dtype), "batch_size": 1,
              "model_loading": model.feature_metadata,
              "completed_at_utc": datetime.now(timezone.utc).isoformat(),
              "runtime": {"torch": str(torch.__version__), "numpy": np.__version__,
                          "pillow": PIL.__version__, "cuda": torch.version.cuda, "device": torch.cuda.get_device_name(0)}})
    except Exception as error:
        write(args.out / "failure.json", {"error_type": type(error).__name__, "message": str(error)})
        raise


def assemble(args):
    provenance = release()
    import numpy as np
    rows = manifest(args.manifest)
    require(digest(args.base) == BASE_SHA, "Original feature archive changed")
    with np.load(args.base, allow_pickle=False) as archive:
        require(set(archive.files) == {"image", "text", "sample_ids"} and archive["sample_ids"].tolist() == IDS,
                "Original features have wrong IDs or fields")
        arrays = {"dino": archive["image"], "glove": archive["text"]}
    receipts = {}
    inputs = {"original_features": BASE_SHA, "manifest": MANIFEST_SHA}
    snapshots = {args.base: BASE_SHA, args.manifest: MANIFEST_SHA}
    for encoder in ("moco", "mae", "elmo"):
        path, meta = getattr(args, encoder), getattr(args, encoder + "_metadata")
        feature_digest, metadata_digest = digest(path), digest(meta)
        record = read(meta)
        require(record["status"] == "complete", "Incomplete feature extraction")
        require(record["feature_sha256"] == feature_digest, "Feature/metadata hash mismatch")
        require(record["weights_sha256"] == WEIGHTS[encoder], "Feature checkpoint mismatch")
        require(record["sample_count"] == 5, "Wrong feature sample count")
        if encoder == "elmo":
            require(record["source_sha256"] == provenance["source_hashes"]["elmo_features.py"], "ELMo source mismatch")
            require(record["options_sha256"] == "86cb4c1a7e2f25e3867d06810e33ae27fbbd511ac77ca377be5fb7405a7085b1", "ELMo options mismatch")
            document = {"sample_ids": IDS, "captions": [r["label_verbatim"] for r in rows]}
            caption_hash = hashlib.sha256((json.dumps(document, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode()).hexdigest()
            require(record["captions_sha256"] == caption_hash and record["sample_ids"] == IDS, "ELMo caption input changed")
            token_rows = []
            for row in rows:
                tokens = re.findall(r"[a-z]+(?:'[a-z]+)?|[0-9]+", unicodedata.normalize("NFKC", row["label_verbatim"]).lower())
                token_rows.append({"sample_id": row["candidate_id"], "tokens": tokens, "count": len(tokens)})
            require(record["tokens"] == token_rows, "ELMo token record changed")
        else:
            require(all(record[key] == value for key, value in provenance.items()), "Vision feature source mismatch")
            require(record["encoder"] == encoder and record["helper_sha256"] == HELPER_SHA
                    and record["batch_size"] == 1 and record["runtime"]["pillow"] == "12.3.0", "Vision extraction recipe mismatch")
        with np.load(path, allow_pickle=False) as archive:
            require(set(archive.files) == {"features", "sample_ids"} and archive["sample_ids"].tolist() == IDS,
                    "Additional features have wrong IDs or fields")
            arrays[encoder] = archive["features"]
        receipts[encoder] = record
        inputs[encoder + "_features"] = feature_digest
        inputs[encoder + "_metadata"] = metadata_digest
        snapshots[path] = inputs[encoder + "_features"]
        snapshots[meta] = inputs[encoder + "_metadata"]
    for name, array in arrays.items():
        check_features(array, name, np)
    io, to = orders()
    args.out.mkdir(mode=0o700)
    blind = {**provenance, "schema_version": 1, "image_ids": [f"I{i:02d}" for i in range(5)],
             "text_ids": [f"T{i:02d}" for i in range(5)], "input_sha256_by_role": inputs,
             "image_distances": {name: cosine(arrays[name], io, np) for name in IMAGES},
             "text_distances": {name: cosine(arrays[name], to, np) for name in TEXTS}}
    write(args.out / "blind-panel.json", blind)
    truth = {"mapping": {f"I{i:02d}": f"T{to.index(index):02d}" for i, index in enumerate(io)},
             "image_labels": {f"I{i:02d}": rows[index]["label_verbatim"] for i, index in enumerate(io)},
             "labels": {f"T{i:02d}": rows[index]["label_verbatim"] for i, index in enumerate(to)}}
    write(args.out / "private-ground-truth.json", truth)
    write(args.out / "private-feature-records.json", receipts)
    with (args.out / "features.npz").open("xb") as stream:
        np.savez_compressed(stream, **arrays, sample_ids=np.asarray(IDS))
    require(all(digest(path) == expected for path, expected in snapshots.items()) and release() == provenance,
            "Feature input or source changed during assembly")
    write(args.out / "checksums.json", {p.name: digest(p) for p in sorted(args.out.iterdir())})


def panel_records(blind, core, provenance):
    require(set(blind) == {"protocol_id", "protocol_sha256", "release_lock_sha256", "source_hashes",
                          "schema_version", "image_ids", "text_ids", "input_sha256_by_role",
                          "image_distances", "text_distances"}, "Unexpected blind fields")
    require(all(blind[key] == value for key, value in provenance.items()), "Blind source provenance changed")
    require(blind["schema_version"] == 1, "Wrong blind schema")
    require(list(blind["image_distances"]) == list(IMAGES) and list(blind["text_distances"]) == list(TEXTS), "Incomplete model panel")
    result = {}
    for image, text in itertools.product(IMAGES, TEXTS):
        # The unchanged numerical core validates its original eight-field schema.
        # Its protocol ID is a calculator interface here, not this study's identity.
        data = {"schema_version": 1, "protocol_id": core.PROTOCOL_ID,
                "protocol_sha256": blind["protocol_sha256"],
                "feature_sha256": hashlib.sha256(json.dumps([blind["image_distances"][image], blind["text_distances"][text]], allow_nan=False).encode()).hexdigest(),
                "image_ids": blind["image_ids"], "text_ids": blind["text_ids"],
                "image_distances": blind["image_distances"][image], "text_distances": blind["text_distances"][text]}
        result[image + "__" + text] = core.enumerate_assignments(data)
    return result


def solve(args):
    provenance = release()
    core = load_module(args.core, "frozen_graph_core", CORE_SHA)
    blind = read(args.input)
    records = panel_records(blind, core, provenance)
    args.out.mkdir(mode=0o700)
    write(args.out / "blind-panel.json", blind)
    write(args.out / "assignments.json", {**provenance, "assignments": records})
    chosen = {}
    for pair, values in records.items():
        best = [v for v in values if v["optimal"]]
        minimum = min(v["cost"] for v in values)
        other = [v["cost"] for v in values if not v["optimal"]]
        chosen[pair] = {"best_assignments": best, "minimum_cost": minimum, "tied_count": len(best),
                        "gap_to_first_distinct_cost": min(other) - minimum if other else None}
    write(args.out / "selected.json", {**provenance, "numerical_core_sha256": CORE_SHA,
          "blind_input_sha256": digest(args.input), "assignments_sha256": digest(args.out / "assignments.json"),
          "pairs": chosen, "recorded_at_utc": datetime.now(timezone.utc).isoformat()})
    require(release() == provenance, "Source changed during solve")
    write(args.out / "checksums.json", {p.name: digest(p) for p in sorted(args.out.iterdir())})


def evaluate(args):
    provenance = release()
    core = load_module(args.core, "frozen_graph_core", CORE_SHA)
    receipt = read(args.selection / "checksums.json")
    require(set(receipt) == {"blind-panel.json", "assignments.json", "selected.json"}, "Incomplete selection")
    for name, expected in receipt.items():
        require(digest(args.selection / name) == expected, "Selected output changed")
    blind = read(args.selection / "blind-panel.json")
    records = panel_records(blind, core, provenance)
    require(read(args.selection / "assignments.json")["assignments"] == records, "Assignment replay mismatch")
    selected = read(args.selection / "selected.json")
    require(all(selected[key] == value for key, value in provenance.items()), "Selection source changed")
    require(selected["assignments_sha256"] == receipt["assignments.json"], "Assignment link changed")
    require(selected["blind_input_sha256"] == receipt["blind-panel.json"], "Blind input link changed")
    require(set(selected["pairs"]) == set(records), "Incomplete selected panel")
    require(selected["numerical_core_sha256"] == CORE_SHA, "Numerical core changed")
    for pair, values in records.items():
        chosen = selected["pairs"][pair]
        best = [v for v in values if v["optimal"]]
        minimum = min(v["cost"] for v in values)
        other = [v["cost"] for v in values if not v["optimal"]]
        require(chosen == {"best_assignments": best, "minimum_cost": minimum, "tied_count": len(best),
                          "gap_to_first_distinct_cost": min(other) - minimum if other else None}, "Selected result mismatch")
    # Correct correspondences are opened only after the completed selection replays.
    truth = read(args.ground_truth)
    require(set(truth) == {"mapping", "labels", "image_labels"}, "Unexpected truth schema")
    require(set(truth["mapping"]) == set(blind["image_ids"]) and set(truth["mapping"].values()) == set(blind["text_ids"]), "Incomplete truth bijection")
    results = {}
    for pair, values in records.items():
        true = [v for v in values if v["mapping"] == truth["mapping"]]
        require(len(true) == 1, "Human mapping missing")
        cost = true[0]["cost"]
        rank = [1 + sum(v["cost"] < cost - TOL for v in values), sum(v["cost"] <= cost + TOL for v in values)]
        best = [{**v, "correct_out_of_5": sum(v["mapping"][i] == t for i, t in truth["mapping"].items())}
                for v in values if v["optimal"]]
        counts = [v["correct_out_of_5"] for v in best]
        results[pair] = {**selected["pairs"][pair], "best_assignments_evaluated": best,
                         "mean_correct_out_of_5": sum(counts) / len(counts), "correct_out_of_5_range": [min(counts), max(counts)],
                         "true_cost": cost, "true_assignment_rank_interval": rank, "true_mapping_is_optimal": true[0]["optimal"]}
    require(digest(args.primary_evaluation) == PRIMARY_EVALUATION_SHA, "Original evaluation bytes changed")
    base = read(args.primary_evaluation)
    require(truth == {"mapping": base["true_mapping"], "labels": base["labels"], "image_labels": base["image_labels"]},
            "Human label/mapping record changed from the original experiment")
    require(results["dino__glove"]["best_assignments_evaluated"] == base["best_assignments_evaluated"]
            and results["dino__glove"]["true_cost"] == base["true_cost"]
            and results["dino__glove"]["true_assignment_rank_interval"] == base["true_assignment_rank_interval"],
            "Original DINO/GloVe result did not reproduce exactly")
    output = {**provenance, "status": "complete", "pairs": results, **truth,
              "selected_sha256": receipt["selected.json"], "assignments_sha256": receipt["assignments.json"],
              "ground_truth_sha256": digest(args.ground_truth), "primary_evaluation_sha256": digest(args.primary_evaluation),
              "primary_result_unchanged": True, "random_expected_correct_out_of_5": 1.0,
              "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
              "scope": "All six prespecified model pairs on the same five items. Exploratory model sensitivity of distance-graph matching; no Rosetta run, held-out accuracy, significance claim, winner selection, or isolated causal effect of architecture/training objective."}
    write(args.out, output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    fields = {"captions": ["manifest", "out"],
              "vision": ["manifest", "images", "helper", "checkpoint", "vendor", "out"],
              "assemble": ["manifest", "base", "moco", "moco-metadata", "mae", "mae-metadata", "elmo", "elmo-metadata", "out"],
              "solve": ["input", "core", "out"],
              "evaluate": ["selection", "core", "ground-truth", "primary-evaluation", "out"]}
    for command, names in fields.items():
        p = sub.add_parser(command)
        for name in names:
            p.add_argument("--" + name, type=Path, required=True)
        if command == "vision":
            p.add_argument("--encoder", choices=("moco", "mae"), required=True)
    args = parser.parse_args()
    globals()[args.command](args)
    print(json.dumps({"status": "complete", "command": args.command}))


if __name__ == "__main__":
    main()
