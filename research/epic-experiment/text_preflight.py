#!/usr/bin/env python3
"""Audit every frozen EPIC selection's GloVe vector without loading images.

This calls the hash-locked extractor's text_features function. It does not fit a
model, change selected rows, exclude rows, or calculate alignment scores.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import time
import unicodedata


EXTRACTOR_SHA256 = "471ede5f611b17ab63252957c0f9785ff2afede22a656f8452fdadff4217f203"
MANIFEST_SHA256 = "d1a34b2ad47a1456a011c6e4f2d6308bae34ebf6aed0c6fb335794ac3dc648be"
EXPECTED_ROWS = 9392


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalized_caption(text):
    return " ".join(unicodedata.normalize("NFKC", str(text)).casefold().split())


def duplicate_summary(indices, keys):
    counts = Counter(keys[index] for index in indices)
    sizes = [count for count in counts.values() if count > 1]
    n = len(indices)
    return {
        "rows": n,
        "unique_values": len(counts),
        "duplicated_groups": len(sizes),
        "rows_beyond_first": n - len(counts),
        "rows_beyond_first_fraction": (n - len(counts)) / n if n else None,
        "rows_in_duplicated_groups": sum(sizes),
        "rows_in_duplicated_groups_fraction": sum(sizes) / n if n else None,
        "largest_group_rows": max(counts.values(), default=0),
    }


def group_summary(indices, rows, coverage, captions, vector_keys, norms, finite):
    total_tokens = sum(coverage[index]["total_tokens"] for index in indices)
    known_tokens = sum(coverage[index]["known_tokens"] for index in indices)
    oov = Counter(token for index in indices for token in coverage[index]["oov_tokens"])
    return {
        "rows": len(indices),
        "total_tokens": total_tokens,
        "known_tokens": known_tokens,
        "oov_tokens": total_tokens - known_tokens,
        "token_coverage": known_tokens / total_tokens if total_tokens else None,
        "rows_with_any_oov": sum(bool(coverage[index]["oov_tokens"]) for index in indices),
        "empty_normalized_caption_uids": [str(rows[i]["uid"]) for i in indices if not captions[i]],
        "empty_tokenization_uids": [str(rows[i]["uid"]) for i in indices if not coverage[i]["total_tokens"]],
        "all_oov_nonempty_uids": [str(rows[i]["uid"]) for i in indices
                                  if coverage[i]["total_tokens"] and not coverage[i]["known_tokens"]],
        "nonfinite_vector_uids": [str(rows[i]["uid"]) for i in indices if not finite[i]],
        "zero_or_tiny_vector_uids": [str(rows[i]["uid"]) for i in indices if finite[i] and norms[i] <= 1e-10],
        "oov_counts": dict(sorted(oov.items())),
        "normalized_caption_duplicates": duplicate_summary(indices, captions),
        "exact_float32_vector_duplicates": duplicate_summary(indices, vector_keys),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--glove", type=Path, required=True)
    parser.add_argument("--extractor", type=Path, default=Path(__file__).with_name("extract_features.py"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    started = datetime.now(timezone.utc).isoformat()
    start_time = time.monotonic()
    extractor_digest, manifest_digest = sha256(args.extractor), sha256(args.manifest)
    if extractor_digest != EXTRACTOR_SHA256:
        raise ValueError(f"Frozen extractor SHA256 mismatch: {extractor_digest}")
    if manifest_digest != MANIFEST_SHA256:
        raise ValueError(f"Frozen selection SHA256 mismatch: {manifest_digest}")
    payload = json.loads(args.manifest.read_text())
    rows = payload["rows"]
    if len(rows) != EXPECTED_ROWS:
        raise ValueError(f"Expected all {EXPECTED_ROWS} selected rows, got {len(rows)}")
    uids = [str(row["uid"]) for row in rows]
    if len(set(uids)) != len(uids):
        raise ValueError("Duplicate selected UIDs")
    spec = importlib.util.spec_from_file_location("frozen_epic_extractor", args.extractor)
    extractor = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(extractor)
    import numpy as np

    features, metadata = extractor.text_features(rows, args.glove, np)
    if features.shape != (EXPECTED_ROWS, 300) or features.dtype != np.float32:
        raise ValueError(f"Unexpected feature shape/dtype: {features.shape}/{features.dtype}")
    coverage = metadata["row_coverage"]
    if [row["uid"] for row in coverage] != uids:
        raise ValueError("Feature coverage UID order does not match selection")
    captions = [normalized_caption(row["narration"]) for row in rows]
    # Use exact bytes for equality; hashes below are only reported identifiers.
    vector_keys = [vector.tobytes() for vector in features]
    finite = np.isfinite(features).all(axis=1)
    norms = np.linalg.norm(features, axis=1)
    eligible = finite & np.isfinite(norms) & (norms > 1e-10)
    groups = {"all": list(range(len(rows)))}
    for key in ("split", "participant_id"):
        for value in sorted({str(row[key]) for row in rows}):
            groups[f"{key}:{value}"] = [i for i, row in enumerate(rows) if str(row[key]) == value]
    for value in ("x", "y"):
        groups[f"fit_role:{value}"] = [i for i, row in enumerate(rows)
                                        if row["split"] == "fit" and row.get("fit_role") == value]
    summaries = {name: group_summary(indices, rows, coverage, captions, vector_keys, norms, finite)
                 for name, indices in groups.items()}
    vector_groups, caption_groups = defaultdict(list), defaultdict(list)
    for index, (vector, caption) in enumerate(zip(vector_keys, captions)):
        vector_groups[vector].append(index)
        caption_groups[caption].append(index)
    distinct_caption_collisions = [indices for indices in vector_groups.values()
                                   if len({captions[i] for i in indices}) > 1]
    cross_split = {}
    for name, mapping in (("normalized_captions", caption_groups), ("exact_float32_vectors", vector_groups)):
        collisions = [indices for indices in mapping.values()
                      if len({rows[i]["split"] for i in indices}) > 1]
        cross_split[name] = {
            "groups_present_in_multiple_splits": len(collisions),
            "rows_in_those_groups": sum(map(len, collisions)),
            "interpretation": "Repeated captions or pooled text vectors; this is not evidence of pretrained image contamination.",
        }
    report = {
        "schema_version": 1,
        "purpose": "Full-selection text eligibility preflight; no images, alignment, fitted parameters, or selection changes",
        "started_at_utc": started,
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.monotonic() - start_time,
        "provenance": {
            "script_sha256": sha256(__file__),
            "extractor_sha256": extractor_digest,
            "manifest_sha256": manifest_digest,
            "glove_artifact_sha256": metadata["artifact_sha256"],
            "glove_member_sha256": metadata["member_sha256"],
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
            "execution": "VM CPU; frozen extractor.text_features; no image reads",
        },
        "rows": len(rows),
        "unique_uids": len(set(uids)),
        "row_order": "Unchanged selected manifest order",
        "uid_order_sha256": hashlib.sha256(json.dumps(uids, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest(),
        "text_matrix": {
            "shape": list(features.shape), "dtype": features.dtype.str,
            "sha256_c_order_bytes": hashlib.sha256(features.tobytes(order="C")).hexdigest(),
            "finite_rows": int(finite.sum()),
            "minimum_finite_norm": float(norms[finite].min()) if finite.any() else None,
            "maximum_finite_norm": float(norms[finite].max()) if finite.any() else None,
        },
        "eligibility_gate": {
            "rule": "Every selected text vector must be finite with L2 norm > 1e-10; any failing row blocks alignment, without filtering or substitution",
            "eligible_rows": int(eligible.sum()),
            "ineligible_uids": [uids[i] for i in np.flatnonzero(~eligible)],
            "passed": bool(eligible.all()),
            "scope": "Text only; final image features and final complete-cache provenance still require validation",
        },
        "caption_normalization": "Unicode NFKC, casefold, collapse whitespace",
        "summaries": summaries,
        "cross_split_repetition": cross_split,
        "vector_collisions_between_distinct_normalized_captions": {
            "groups": len(distinct_caption_collisions),
            "rows_in_groups": sum(map(len, distinct_caption_collisions)),
            "group_uids": [[uids[i] for i in indices] for indices in distinct_caption_collisions],
        },
        "frozen_extractor_text_metadata": metadata,
    }
    # Ensure the inputs did not change while the original ZIP was streamed.
    if sha256(args.extractor) != extractor_digest or sha256(args.manifest) != manifest_digest:
        raise ValueError("Frozen inputs changed during preflight")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(json.dumps({"output": str(args.output), "rows": len(rows),
                      "eligible_rows": int(eligible.sum()), "token_coverage": metadata["token_coverage"],
                      "elapsed_seconds": report["elapsed_seconds"], "passed": bool(eligible.all())}))
    return 0 if eligible.all() else 2


if __name__ == "__main__":
    raise SystemExit(main())
