#!/usr/bin/env python3
"""Verify recording dates and action-caption multiplicity in pinned EPIC metadata.

Downloads metadata only. This is a provenance check, not a representation or
alignment experiment. It does not verify that published model training claims
match the training execution or audit ImageNet bytes for duplicates.
"""

import argparse
import collections
import csv
import datetime
import hashlib
import io
import json
from pathlib import Path
import urllib.request


REPOSITORY = "epic-kitchens/epic-kitchens-55-annotations"
COMMIT = "63fa7d78eba41a8d2280374061a6b1bab3e7e41d"
BASE = f"https://raw.githubusercontent.com/{REPOSITORY}/{COMMIT}/"
FILES = {
    "EPIC_descriptions.csv": "b73ad13c8546e65413fd0923b9c37a02b90510733b547830f6ba283306dcb507",
    "EPIC_train_action_narrations.csv": "806159089bad092e6f9a963ed0b14dc3e895d6b76b6ad0f7e02cec2448ef86e4",
    "EPIC_train_action_labels.csv": "faef8ad29b163b28ec8ca32b4222f137aba8f8b07eca0ae004652937ffaf811e",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).parent)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "dataset": "EPIC-KITCHENS-55", "repository": REPOSITORY,
        "commit": COMMIT, "files": {},
        "audit_scope": "Recording-date metadata, annotation references, exact string repetition, and source-byte hashes only.",
        "limits": [
            "Recording dates are publisher-provided metadata, not independently authenticated capture times.",
            "This script does not inspect any model weights or pretraining corpus.",
            "It does not show that ordinary narration strings never occurred in earlier text.",
            "Temporal exclusion from pretraining requires a separate argument using documented model training sources.",
        ],
    }
    loaded = {}
    for name, expected in FILES.items():
        with urllib.request.urlopen(BASE + name, timeout=30) as response:
            raw = response.read()
        observed = hashlib.sha256(raw).hexdigest()
        if observed != expected:
            raise RuntimeError(f"Hash mismatch for {name}: {observed}")
        (args.output / name).write_bytes(raw)
        rows = list(csv.DictReader(io.StringIO(raw.decode("utf-8-sig"))))
        loaded[name] = rows
        manifest["files"][name] = {"url": BASE + name, "sha256": observed,
                                   "bytes": len(raw), "rows": len(rows),
                                   "columns": list(rows[0])}
    descriptions = loaded["EPIC_descriptions.csv"]
    dates = {}
    for row in descriptions:
        fmt = "%Y-%m-%d" if "-" in row["date"] else "%d/%m/%Y"
        date = datetime.datetime.strptime(row["date"], fmt).date()
        if row["video_id"] in dates:
            raise RuntimeError(f"Duplicate video date: {row['video_id']}")
        dates[row["video_id"]] = date.isoformat()
    references = {}
    for name in ("EPIC_train_action_narrations.csv", "EPIC_train_action_labels.csv"):
        video_ids = {row["video_id"] for row in loaded[name]}
        missing = sorted(video_ids - dates.keys())
        if missing:
            raise RuntimeError(f"Videos without recording date in {name}: {missing}")
        references[name] = {"distinct_video_ids": len(video_ids), "missing_recording_dates": missing}
    manifest["annotation_reference_checks"] = references
    actions = loaded["EPIC_train_action_labels.csv"]
    narrations = collections.Counter(row["narration"] for row in actions)
    manifest["observations"] = {
        "recording_date_min": min(dates.values()),
        "recording_date_max": max(dates.values()),
        "all_432_dates_parsed": len(dates) == 432,
        "date_years": sorted({date[:4] for date in dates.values()}),
        "train_action_rows": len(actions),
        "train_video_count": len({row["video_id"] for row in actions}),
        "train_participant_count": len({row["participant_id"] for row in actions}),
        "train_video_ids_missing_recording_dates": references["EPIC_train_action_labels.csv"]["missing_recording_dates"],
        "distinct_exact_narrations": len(narrations),
        "rows_with_repeated_exact_narrations": sum(n for n in narrations.values() if n > 1),
        "max_exact_narration_multiplicity": max(narrations.values()),
        "most_common_narrations": narrations.most_common(5),
    }
    target = args.output / "epic55-provenance-audit.json"
    target.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest["observations"], indent=2))


if __name__ == "__main__":
    main()
